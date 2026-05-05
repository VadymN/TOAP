#!/usr/bin/env python3
"""
Транскрипція аудіо українською мовою за допомогою OpenAI Whisper.
Використання: python transcribe.py <шлях_до_файлу>
"""

import sys
import os

def transcribe(audio_path: str, model_size: str = "medium") -> str:
    try:
        import whisper
    except ImportError:
        print("Встановлюю OpenAI Whisper...")
        os.system(f"{sys.executable} -m pip install openai-whisper")
        import whisper

    print(f"Завантажую модель '{model_size}'...")
    model = whisper.load_model(model_size)

    print(f"Розпізнаю: {audio_path}")
    result = model.transcribe(audio_path, language="uk", verbose=False)

    return result["text"]


def save_transcript(text: str, audio_path: str) -> str:
    base = os.path.splitext(audio_path)[0]
    output_path = base + "_transcript.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Використання: python transcribe.py <шлях_до_аудіофайлу> [розмір_моделі]")
        print("Розміри моделі: tiny, base, small, medium (за замовчуванням), large")
        sys.exit(1)

    audio_path = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "medium"

    if not os.path.exists(audio_path):
        print(f"Помилка: файл не знайдено: {audio_path}")
        sys.exit(1)

    text = transcribe(audio_path, model_size)

    print("\n--- ТРАНСКРИПЦІЯ ---")
    print(text)
    print("--------------------\n")

    output_path = save_transcript(text, audio_path)
    print(f"Збережено: {output_path}")


if __name__ == "__main__":
    main()
