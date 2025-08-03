import os
import json
from utils.yt_download import download_audio
from utils.whisper_transcribe import transcribe_audio
from utils.ocr_extract import extract_ocr_from_video

def process_talk(youtube_url, output_jsonl, whisper_model="base"):
    print(f"[INFO] Processing: {youtube_url}")
    audio_path, video_path = download_audio(youtube_url)
    transcript_segments = transcribe_audio(audio_path, whisper_model)
    ocr_segments = extract_ocr_from_video(video_path)
    with open(output_jsonl, "a", encoding="utf-8") as fout:
        for segment in transcript_segments:
            record = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            }
            for ocr in ocr_segments:
                if ocr["start"] <= segment["start"] <= ocr["end"]:
                    record["ocr"] = ocr["text"]
                    break
            fout.write(json.dumps(record) + "\n")
    print(f"[DONE] Transcription saved to {output_jsonl}")

if __name__ == "__main__":
    urls = [
        "https://www.youtube.com/watch?v=GPDks5OXAzw"
    ]
    for url in urls:
        process_talk(url, "talks_transcripts.jsonl")
