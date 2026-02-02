from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from openai import OpenAI

from tts_engine import synthesize_speech

History = List[Tuple[str, str]]


def _load_dotenv_if_present(dotenv_path: str = ".env") -> None:
    """
    Minimal .env loader:
    - reads KEY=VALUE lines
    - ignores blank lines and comments (#)
    - does not override already-set environment variables
    """
    if not os.path.exists(dotenv_path):
        return

    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


_load_dotenv_if_present()
client = OpenAI()

LATIN_RE = re.compile(r"[A-Za-z]")
IRINA_NAME = "Д-р Ирина (учёный)"
IRINA_FIRST_PHRASE = "Давай начнём с самого простого."

# Пролог перед первой фразой Ирины (только в самом первом ходе)
IRINA_PREFACE = "Добрый день. Да, спасибо."


# Убираем только если это стоит СТРОГО в начале реплики (не трогаем середину текста).
SPEAKER_PREFIX_RE = re.compile(
    r"^\s*(?:Д-?р\s*)?(?:Ирина|Алексей)\s*\((?:учёный|скептик)\)\s*:\s*",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Agent:
    name: str
    system_prompt: str


def _history_to_messages(history: Sequence[Tuple[str, str]]) -> List[dict]:
    messages: List[dict] = []
    for speaker, text in history:
        messages.append({"role": "user", "content": f"{speaker}: {text}".strip()})
    return messages


def _rewrite_without_latin(text: str, agent: Agent) -> str:
    """
    Если в реплике проскочила латиница — просим модель переписать СТРОГО без латинских букв.
    Делаем 1 попытку, чтобы не зациклиться.
    """
    if not LATIN_RE.search(text):
        return text

    messages = [
        {
            "role": "system",
            "content": (
                f"{agent.system_prompt}\n\n"
                "ЖЁСТКОЕ ПРАВИЛО: в ответе не должно быть НИ ОДНОЙ латинской буквы (A–Z, a–z). "
                "Никаких английских слов, вставок, транслитерации. Только кириллица, цифры и знаки."
            ),
        },
        {
            "role": "user",
            "content": (
                "Перепиши текст ниже так, чтобы смысл сохранился, но в нём не было ни одной латинской буквы.\n\n"
                f"Текст:\n{text}"
            ),
        },
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=260,
        temperature=0.3,
    )
    rewritten = (resp.choices[0].message.content or "").strip()
    # На всякий случай: если всё равно есть латиница — просто возвращаем оригинал без второй попытки
    return rewritten if not LATIN_RE.search(rewritten) else text


def _strip_leading_speaker_prefix(reply: str) -> str:
    """
    Убираем "Д-р Ирина (учёный):" / "Д-р Алексей (скептик):" если это случайно попало в НАЧАЛО реплики.
    """
    cleaned = reply.lstrip()
    cleaned = SPEAKER_PREFIX_RE.sub("", cleaned).strip()
    return cleaned


def _enforce_irina_first_phrase(reply: str) -> str:
    """
    Гарантируем, что первая реплика Ирины начинается с нужной фразы.
    """
    cleaned = reply.lstrip()
    if cleaned.startswith(IRINA_FIRST_PHRASE):
        return cleaned
    return f"{IRINA_FIRST_PHRASE} {cleaned}".strip()


def _add_irina_preface_if_first_turn(reply: str) -> str:
    """
    Добавляем пролог перед обязательной первой фразой Ирины:
    "Добрый день. Да, спасибо. Давай начнём с самого простого. ..."
    Делается только если реплика уже начинается с IRINA_FIRST_PHRASE.
    """
    cleaned = reply.lstrip()
    if not cleaned.startswith(IRINA_FIRST_PHRASE):
        return cleaned
    # уже с прологом? (на всякий случай, чтобы не задвоить)
    if cleaned.startswith(f"{IRINA_PREFACE} {IRINA_FIRST_PHRASE}"):
        return cleaned
    return f"{IRINA_PREFACE} {cleaned}".strip()


def generate_reply(
    agent: Agent,
    history: History,
    topic: Optional[str] = None,
    max_tokens: int = 350,
) -> Tuple[str, History]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Не найден OPENAI_API_KEY. Создайте файл .env в корне проекта и добавьте строку "
            "OPENAI_API_KEY=ваш_ключ (или задайте переменную окружения OPENAI_API_KEY)."
        )

    is_first_turn = not history
    effective_topic = topic if is_first_turn else None
    topic_line = f"Тема: {effective_topic}." if effective_topic else ""

    if is_first_turn and agent.name == IRINA_NAME:
        user_content = (
            "Сделай ПЕРВУЮ реплику диалога по указанной теме.\n"
            "Это самое начало: до этого никто ничего не говорил.\n"
            f"Реплика ОБЯЗАТЕЛЬНО начинается ровно с фразы: «{IRINA_FIRST_PHRASE}».\n"
            "Дальше — спокойное, уверенное введение в тему (2–4 предложения), без ощущения, что спор уже идёт.\n"
            "Ирина — эксперт: она объясняет и ведёт разговор.\n"
            "Не используй английские слова, вставки и латиницу.\n"
            "Не перечисляй правила и не используй разметку."
        )
    else:
        user_content = (
            "Продолжи диалог следующей репликой от лица персонажа.\n"
            "Ирина — эксперт, объясняет и ведёт разговор.\n"
            "Алексей — собеседник, который задаёт неудобные, уточняющие вопросы.\n"
            "Не используй английские слова, вставки и латиницу.\n"
            "Не перечисляй правила и не используй разметку."
        )

    messages: List[dict] = [
        {
            "role": "system",
            "content": (
                f"{agent.system_prompt}\n\n"
                "Общие правила:\n"
                "- Отвечай строго на русском.\n"
                "- 2–4 предложения.\n"
                "- Научно, но понятно, без сухого академизма.\n"
                "- ЖЁСТКИЙ ЗАПРЕТ: никаких английских слов, вставок, транслитерации, латиницы.\n"
                "- Не пиши своё имя и должность в начале реплики, говори просто от первого лица.\n"
                "- Вопросы не должны занимать весь объём реплики: сначала мысль/позиция, затем (если уместно) 1 прицельный вопрос.\n"
                "- Ирина чаще утверждает и объясняет; Алексей чаще задаёт вопросы и сомневается.\n"
                f"{topic_line}"
            ).strip(),
        },
        *_history_to_messages(history),
        {"role": "user", "content": user_content},
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
    )

    reply_text = (resp.choices[0].message.content or "").strip()

    # Косметика: убираем "Д-р Ирина...:", если попало в начало реплики
    reply_text = _strip_leading_speaker_prefix(reply_text)

    # "Гарантии качества"
    if is_first_turn and agent.name == IRINA_NAME:
        reply_text = _enforce_irina_first_phrase(reply_text)
        reply_text = _add_irina_preface_if_first_turn(reply_text)

    reply_text = _rewrite_without_latin(reply_text, agent)

    updated_history: History = list(history)
    updated_history.append((agent.name, reply_text))
    return reply_text, updated_history


def run_dialog(topic: str, turns: int = 10) -> History:
    agent1 = Agent(
        name=IRINA_NAME,
        system_prompt=(
            "Ты — учёная-исследовательница и научный популяризатор. Пиши спокойно, уверенно и чётко, "
            "как умная собеседница в подкасте. Объясняй сложные вещи человеческим языком и простыми примерами, "
            "не скатываясь в сюсюканье и не приукрашивая факты. "
            "Говори от первого лица в женском роде (я согласна, я считаю, я хотела бы, мне кажется) "
            "и избегай мужских форм вроде «я согласен». "
            "Твоя роль — эксперт: ты ведёшь разговор и даёшь объяснения и выводы. "
            "Вопросы используй редко и только по делу; не перекладывай экспертность на собеседника. "
            "Юмор лёгкий и интеллигентный, дозировано. "
            "ЖЁСТКО: никаких английских слов, латиницы, транслитерации."
        ),
    )
    agent2 = Agent(
        name="Д-р Алексей (скептик)",
        system_prompt=(
            "Ты — скептичный учёный и собеседник, который задаёт неудобные, но уместные вопросы. "
            "Говори от первого лица в мужском роде и избегай женских форм. "
            "Твоя роль — уточнять и проверять: спрашивай про данные, метод, ограничения и альтернативные объяснения. "
            "Структура реплики: 1–2 предложения сомнения/уточнения + 1–2 прицельных вопроса. "
            "Не уходи в длинные монологи. "
            "Юмор лёгкий, чуть ехидный, но без грубости. "
            "ЖЁСТКО: никаких английских слов, латиницы, транслитерации."
        ),
    )

    history: History = []
    agents = [agent1, agent2]

    voice_map = {
        IRINA_NAME: "nova",
        "Д-р Алексей (скептик)": "alloy",
    }

    for i in range(turns):
        agent = agents[i % 2]
        reply, history = generate_reply(agent=agent, history=history, topic=topic)
        print(f"{agent.name}: {reply}\n")

        try:
            voice = voice_map.get(agent.name, "alloy")
            audio_path = synthesize_speech(
                text=reply,
                speaker=agent.name,
                turn_index=i + 1,
                voice=voice,
            )
            print(f"[AUDIO] Сохранён файл: {audio_path}")
        except Exception as e:
            print(f"[AUDIO][ERROR] Не удалось озвучить реплику: {e}")

    return history


if __name__ == "__main__":
    run_dialog(
        topic="Почему разные органы стареют с разной скоростью и можно ли это изменить?",
        turns=4,
    )
