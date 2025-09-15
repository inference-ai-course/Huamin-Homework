# project_config.py
import os
from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "arxiv_pdfs"
TEXT_DIR = BASE_DIR / "arxiv_texts"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# 创建目录
for directory in [PDF_DIR, TEXT_DIR, DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")

# 模型配置
MODEL_NAME = "unsloth/llama-3.1-7b-unsloth-bnb-4bit"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# 生成配置
NUM_PAPERS_TO_PROCESS = 3
NUM_QA_PAIRS_PER_PAPER = 2

print(f"配置文件加载完成")
print(f"PDF目录: {PDF_DIR}")
print(f"文本目录: {TEXT_DIR}")