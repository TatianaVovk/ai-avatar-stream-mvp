from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI


# --- Загрузка .env ЛОКАЛЬНО для этого модуля ---


def _load_dotenv_if_present(dotenv_name: str = ".env") -> None:
    """
    Минималистичный загрузчик .env:
    - ищет файл .env в корне проекта (на уровень выше src)
    - читает строки KEY=VALUE
    - не перезаписывает уже выставленные переменные окружения
    """
    # Корень проекта: на уровень выше папки src.
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / dotenv_name

    if not env_path.exists():
        return

    try:
        with env_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                # Не трогаем уже выставленные переменные
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        # Если не получилось прочитать файл — просто пропускаем
        return


# Загружаем .env ПЕРЕД созданием клиента OpenAI
_load_dotenv_if_present()


# Создаём глобальный клиент OpenAI (использует OPENAI_API_KEY из переменных окружения).
client = OpenAI()

# Корень проекта: на уровень выше папки src.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Папка для аудиофайлов.
AUDIO_DIR = PROJECT_ROOT / "audio"


def _simplify_speaker_name(speaker: str) -> str:
    """
    Упрощает имя диктора для использования в имени файла:
    - приводит к нижнему регистру
    - убирает пробелы и скобки
    """
    simplified = speaker.lower()
    for ch in (" ", "(", ")", "[", "]", "{", "}"):
        simplified = simplified.replace(ch, "")
    # На всякий случай, если имя полностью "стерлось"
    return simplified or "speaker"


def synthesize_speech(text: str, speaker: str, turn_index: int, voice: str = "alloy") -> str:
    """
    Синтезирует речь для заданного текста и диктора.

    - Проверяет наличие OPENAI_API_KEY
    - Создаёт папку audio в корне проекта
    - Генерирует имя файла вида 001_speaker.mp3
    - Вызывает OpenAI TTS (gpt-4o-mini-tts) и сохраняет результат
    - Возвращает путь к файлу как строку
    """
    # Проверяем, что ключ API доступен в переменных окружения.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Не найден OPENAI_API_KEY. Создайте файл .env в корне проекта и добавьте строку "
            "OPENAI_API_KEY=ваш_ключ (или задайте переменную окружения OPENAI_API_KEY)."
        )

    # Убеждаемся, что папка audio существует.
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Формируем упрощённое имя диктора и имя файла с ведущими нулями для индекса хода.
    speaker_simplified = _simplify_speaker_name(speaker)
    filename = f"{turn_index:03d}_{speaker_simplified}.mp3"
    output_path = AUDIO_DIR / filename

    # Запрашиваем синтез речи в OpenAI TTS.
    # Используем потоковый ответ и сохраняем напрямую в файл.
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        response_format="mp3",
    ) as response:
        # Сохраняем аудио в mp3-файл.
        response.stream_to_file(output_path)

    # Возвращаем путь к файлу в виде строки.
    return str(output_path)

