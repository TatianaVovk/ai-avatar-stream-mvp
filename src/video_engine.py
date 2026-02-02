from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from openai import OpenAI
from moviepy.editor import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_videoclips,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_ROOT / "audio"
AVATARS_DIR = PROJECT_ROOT / "avatars"
VIDEO_DIR = PROJECT_ROOT / "video"
TMP_DIR = VIDEO_DIR / "_tmp"
ASSETS_DIR = PROJECT_ROOT / "assets"

OUTPUT_PATH = VIDEO_DIR / "dialog_test.mp4"

RESOLUTION: Tuple[int, int] = (1280, 720)
BG_COLOR = (17, 17, 17)  # #111111

# Голоса: интро и представления - голос, отличный от спикеров
NEUTRAL_VOICE = "onyx"
TTS_MODEL = "gpt-4o-mini-tts"


def _load_dotenv_if_present(dotenv_name: str = ".env") -> None:
    """
    Минималистичный загрузчик .env:
    - ищет .env в корне проекта
    - не перезаписывает уже выставленные переменные окружения
    """
    env_path = PROJECT_ROOT / dotenv_name
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
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


_load_dotenv_if_present()
client = OpenAI()

# Тема берётся из env 
TOPIC = os.getenv(
    "TOPIC",
    "Почему разные органы стареют с разной скоростью и можно ли это изменить?",
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


MAX_AUDIO_FILES = _env_int("MAX_AUDIO_FILES", 0)  # 0 = без ограничения


def detect_speaker_from_filename(filename: str) -> str:
    lower = filename.lower()
    if "ирина" in lower:
        return "irina"
    if "алексей" in lower:
        return "alexey"
    return "unknown"


def _tts_to_file(text: str, out_path: Path, voice: str) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Не найден OPENAI_API_KEY. Создайте файл .env в корне проекта и добавьте строку "
            "OPENAI_API_KEY=ваш_ключ (или задайте переменную окружения OPENAI_API_KEY)."
        )

    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=voice,
        input=text,
        response_format="mp3",
    ) as response:
        response.stream_to_file(out_path)

    return out_path


def _fit_image_on_bg(image_path: Path, duration: float) -> CompositeVideoClip:
    """
    Фон #111111, картинка по центру, без растяжения, пропорции сохранены.
    """
    bg = ColorClip(size=RESOLUTION, color=BG_COLOR).set_duration(duration)

    img = ImageClip(str(image_path)).set_duration(duration)
    w, h = img.size
    rw, rh = RESOLUTION

    scale = min(rw / w, rh / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    img = img.resize(newsize=(new_w, new_h)).set_position(("center", "center"))
    return CompositeVideoClip([bg, img], size=RESOLUTION).set_duration(duration)


def _make_cover_bg(duration: float) -> CompositeVideoClip:
    """
    Фон для ведущего assets/cover.png на всём экране.
    Если cover.png нет — используем просто тёмный фон, но не падаем.
    """
    cover_path = ASSETS_DIR / "cover.png"
    if not cover_path.exists():
        print(f"[COVER][WARN] Не найден файл: {cover_path} — будет использован тёмный фон.")
        return ColorClip(size=RESOLUTION, color=BG_COLOR).set_duration(duration)

    # Используем ту же посадку (без растяжения), чтобы не ломать пропорции.
    return _fit_image_on_bg(image_path=cover_path, duration=duration)


def make_topic_intro_clip(topic: str) -> CompositeVideoClip:
    """
    Общее голосовое интро темы, длительность = длине аудио.
    """
    print("[INTRO] Добавляем интро темы...")

    # Чтобы интро соответствовало теме, можно озвучить сам topic.
    # Но чтобы не рисковать латиницей/странными символами, делаем нейтральную вводную + тема одной фразой.
    text = (
        "Сегодня мы обсуждаем научную тему. "
        f"Тема выпуска: {topic}"
    )

    audio_path = _tts_to_file(
        text=text,
        out_path=TMP_DIR / "000_topic_intro.mp3",
        voice=NEUTRAL_VOICE,
    )

    audio = AudioFileClip(str(audio_path))
    duration = audio.duration

    bg = _make_cover_bg(duration).set_audio(audio)
    return bg


def make_speaker_intro_clip(speaker_key: str, avatar_path: Path) -> CompositeVideoClip:
    """
    Голосовое представление спикера на фоне cover.png
    """
    if speaker_key == "irina":
        text = "Ирина — учёный и исследователь. Она поможет разобраться в теме."
        fname = "010_intro_irina.mp3"
    elif speaker_key == "alexey":
        text = "Алексей — собеседник, который задаёт неудобные вопросы и проверяет аргументы."
        fname = "011_intro_alexey.mp3"
    else:
        text = "Участник диалога."
        fname = "012_intro_unknown.mp3"

    print(f"[INTRO] Добавляем представление спикера: {speaker_key}...")

    audio_path = _tts_to_file(
        text=text,
        out_path=TMP_DIR / fname,
        voice=NEUTRAL_VOICE,
    )

    audio = AudioFileClip(str(audio_path))
    duration = audio.duration

    bg = _make_cover_bg(duration).set_audio(audio)
    return bg


def make_avatar_clip(image_path: Path, audio_path: Path) -> CompositeVideoClip:
    """
    Реплика диалога: аватар (без растяжения) + аудио реплики.
    """
    audio = AudioFileClip(str(audio_path))
    duration = audio.duration
    clip = _fit_image_on_bg(image_path=image_path, duration=duration).set_audio(audio)
    return clip


def main() -> None:
    print("=== Сборка видео диалога ===")

    if not AUDIO_DIR.exists():
        print(f"[ERROR] Папка с аудио не найдена: {AUDIO_DIR}")
        return

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Берём все реплики диалога (mp3 с ведущими цифрами в имени)
    audio_files: List[Path] = sorted(AUDIO_DIR.glob("*.mp3"))
    audio_files = [p for p in audio_files if p.name[:3].isdigit()]
    audio_files = sorted(audio_files, key=lambda p: p.name)

    if MAX_AUDIO_FILES and MAX_AUDIO_FILES > 0:
        audio_files = audio_files[:MAX_AUDIO_FILES]

    if not audio_files:
        print(f"[ERROR] Не найдено ни одного аудиофайла в {AUDIO_DIR}")
        return

    print("Найдены аудиофайлы:")
    for p in audio_files:
        print(f" - {p.name}")

    avatar_map = {
        "irina": AVATARS_DIR / "irina.jpg",
        "alexey": AVATARS_DIR / "alexey.jpg",
    }

    # Проверка аватаров
    for k, path in avatar_map.items():
        if not path.exists():
            print(f"[ERROR] Не найден аватар для {k}: {path}")
            return

    clips = []

    # Общее интро темы (1 раз), на фоне cover.png
    try:
        clips.append(make_topic_intro_clip(TOPIC))
        print("[OK] Интро темы добавлено.")
    except Exception as e:
        print(f"[ERROR] Не удалось создать интро темы: {e}")
        return

    # Представление каждого спикера (строго 1 раз), фоне cover.png
    introduced = {"irina": False, "alexey": False}

    for idx, audio_path in enumerate(audio_files, start=1):
        speaker_key = detect_speaker_from_filename(audio_path.name)
        avatar_path = avatar_map.get(speaker_key)

        if avatar_path is None:
            print(
                f"[WARN] Не удалось определить спикера для {audio_path.name}. "
                f"Ожидалось 'ирина' или 'алексей' в имени. Файл будет пропущен."
            )
            continue

        # логи длительности реплики
        try:
            a = AudioFileClip(str(audio_path))
            duration = a.duration
            a.close()
        except Exception as e:
            print(f"[ERROR] Не удалось прочитать аудио {audio_path.name}: {e}")
            continue

        print(
            f"[STEP {idx}] Реплика: {audio_path.name} | Спикер: {speaker_key} | Длительность: {duration:.2f} сек"
        )

        # если спикер ещё не представлен — добавляем интро-clip
        if speaker_key in introduced and not introduced[speaker_key]:
            try:
                intro_clip = make_speaker_intro_clip(speaker_key, avatar_path)
                clips.append(intro_clip)
                introduced[speaker_key] = True
                print(f"[OK] Представление {speaker_key} добавлено.")
            except Exception as e:
                print(f"[ERROR] Не удалось создать представление спикера {speaker_key}: {e}")
                return

        # Реплика диалога (голос спикера уже в mp3)
        try:
            clip = make_avatar_clip(image_path=avatar_path, audio_path=audio_path)
            clips.append(clip)
        except Exception as e:
            print(f"[ERROR] Не удалось создать клип реплики для {audio_path.name}: {e}")

    if not clips:
        print("[ERROR] Не удалось создать ни одного клипа — выход.")
        return

    final_clip = concatenate_videoclips(clips, method="compose")

    print(f"Сохраняем итоговое видео в {OUTPUT_PATH} ...")

    try:
        final_clip.write_videofile(
            str(OUTPUT_PATH),
            codec="libx264",
            audio_codec="aac",
            fps=24,
        )
        print(f"[OK] Итоговый файл сохранён: {OUTPUT_PATH}")
    except Exception as e:
        msg = str(e).lower()
        print(f"[ERROR] Не удалось сохранить видео: {e}")
        if "ffmpeg" in msg:
            print(
                "\nПохоже, ffmpeg недоступен.\n"
                "Установите ffmpeg и убедитесь, что он есть в PATH.\n"
                "Пример (Windows, через choco): choco install ffmpeg\n"
                "И перезапустите: python src/video_engine.py"
            )
    finally:
        final_clip.close()
        for c in clips:
            try:
                c.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
