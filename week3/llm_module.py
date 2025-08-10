from transformers import pipeline

# 更换为轻量级开源模型 gpt2
llm = pipeline("text-generation", model="gpt2", device=-1)

def generate_response(prompt: str) -> str:
    result = llm(prompt, max_new_tokens=50, do_sample=False)
    return result[0]["generated_text"]
