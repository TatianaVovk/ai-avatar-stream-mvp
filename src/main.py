from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dialog_engine import run_dialog
import video_engine


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
AUDIO_DIR = PROJECT_ROOT / "audio"


def _load_dotenv_if_present(dotenv_name: str = ".env") -> None:
    """
    Минималистичный загрузчик .env:
    - ищет файл .env в корне проекта
    - читает строки KEY=VALUE
    - игнорирует пустые строки и комментарии (#)
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


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _write_transcript(history, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for speaker, text in history:
            f.write(f"{speaker}:\n{text}\n\n")


def _clean_audio_dir() -> None:
    if not AUDIO_DIR.exists():
        return
    for p in AUDIO_DIR.glob("*.mp3"):
        try:
            p.unlink()
        except Exception:
            pass


def main() -> None:
    _load_dotenv_if_present()

    # Настройки (через .env)
    topic: str = os.getenv(
        "TOPIC",
        "Почему разные органы стареют с разной скоростью и можно ли это изменить?",
    )
    turns: int = _env_int("TURNS", 22)  # 22 обычно ~15–20 минут
    clean_audio: bool = _env_bool("CLEAN_AUDIO_BEFORE_RUN", True)

    if clean_audio:
        print("[MAIN] Очищаем папку audio/ перед запуском...")
        _clean_audio_dir()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_path = LOGS_DIR / f"transcript_{stamp}.txt"

    print("=== Генерация диалога ===")
    print(f"[MAIN] TOPIC: {topic}")
    print(f"[MAIN] TURNS: {turns}")
    print(f"[MAIN] Transcript: {transcript_path}")

    history = run_dialog(topic=topic, turns=turns)
    _write_transcript(history=history, out_path=transcript_path)
    print(f"[OK] Транскрипт сохранён: {transcript_path}")

    # Передаём тему в video_engine через переменную окружения
    os.environ["TOPIC"] = topic

    print("\n=== Сборка видео ===")
    video_engine.main()
    print("\n[OK] Готово.")
    print(f"[INFO] Видео: {video_engine.OUTPUT_PATH}")
    print(f"[INFO] Транскрипт: {transcript_path}")



if __name__ == "__main__":
    main()
