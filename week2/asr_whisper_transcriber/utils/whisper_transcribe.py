import whisper

def transcribe_audio(audio_path, model_size="base"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)

    segments = result['segments']  # start, end, text
    return segments
