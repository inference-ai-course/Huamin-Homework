import yt_dlp
import os

def download_audio(url, out_dir="downloads"):
    os.makedirs(out_dir, exist_ok=True)
    video_path = os.path.join(out_dir, "video.mp4")
    audio_path = os.path.join(out_dir, "audio.wav")

    ydl_opts = {
        'format': 'bestaudio+bestaudio',
        'outtmpl': video_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return audio_path, video_path
