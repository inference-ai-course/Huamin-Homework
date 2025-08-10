import pyttsx3

# 初始化 TTS 引擎（全局）
tts_engine = pyttsx3.init()

def synthesize_speech(text, filename="response.wav"):
    # 配置参数（可选）
    tts_engine.setProperty('rate', 180)   # 语速
    tts_engine.setProperty('volume', 1.0) # 音量 0~1
    # 可设置 voice（男性/女性）根据系统语音引擎而定

    # 保存到文件
    tts_engine.save_to_file(text, filename)
    tts_engine.runAndWait()
    return filename
