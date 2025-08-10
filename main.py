from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from asr_module import transcribe_audio
from llm_module import generate_response
from tts_module import synthesize_speech

app = FastAPI()

@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    user_text = transcribe_audio(audio_bytes)
    print("User said:", user_text)
    
    bot_text = generate_response(user_text)
    print("Bot responded:", bot_text)

    audio_path = synthesize_speech(bot_text)
    return FileResponse(audio_path, media_type="audio/wav")
