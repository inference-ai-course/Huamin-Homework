# 
from typing import List
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import faiss
import os
import re
import requests
import feedparser
import fitz  # PyMuPDF

# 存储目录
pdf_dir = Path("./arxiv_pdfs")
os.makedirs(pdf_dir, exist_ok=True)

# 目标分类：cs.CL (Computation and Language)
base_url = "http://export.arxiv.org/api/query?"

# 获取最新的 50 篇论文
query = "search_query=cat:cs.CL&start=0&max_results=50&sortBy=submittedDate&sortOrder=descending"

feed = feedparser.parse(base_url + query)
print("找到论文数量:", len(feed.entries))

for entry in feed.entries:
    # 获取 PDF 链接
    pdf_url = entry.id.replace("abs", "pdf")
    title = entry.title.replace(" ", "_").replace("/", "_")
    filename = os.path.join(pdf_dir, f"{title}.pdf")

    if not os.path.exists(filename):
        try:
            r = requests.get(pdf_url + ".pdf")
            with open(filename, "wb") as f:
                f.write(r.content)
            print("已下载:", filename)
        except Exception as e:
            print("下载失败:", pdf_url, e)

# -----------------------------
# 1️⃣ PDF 文本提取
# -----------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        page_text = page.get_text("text")
        text += page_text + "\n"
    return text.strip()

# 遍历目录，提取所有 PDF 的文本
documents = []
for fname in os.listdir(pdf_dir):
    if fname.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, fname)
        raw_text = extract_text_from_pdf(pdf_path)
        
        if raw_text:  # 避免空文档
            documents.append(raw_text)
            print(f"抽取完成: {fname}, 文本长度: {len(raw_text)}")

def clean_text(text):
    # 去掉多余空格和换行
    text = re.sub(r"\s+", " ", text)
    return text.strip()

documents = [clean_text(doc) for doc in documents]
print("最终文档数:", len(documents))

out_dir = Path("./arxiv_texts")
os.makedirs(out_dir, exist_ok=True)

for i, doc in enumerate(documents):
    with open(os.path.join(out_dir, f"doc_{i+1}.txt"), "w", encoding="utf-8") as f:
        f.write(doc)

# -----------------------------
# 2️⃣ 文本切块
# -----------------------------
def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
    tokens = text.split()
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + max_tokens]
        chunks.append(" ".join(chunk))
    return chunks

# -----------------------------
# 3️⃣ 批量处理 PDF → chunks
# -----------------------------
pdf_dir = Path("arxiv_pdfs")  # PDF 所在目录
all_chunks = []

for pdf_file in pdf_dir.glob("*.pdf"):
    text = extract_text_from_pdf(pdf_file)
    chunks = chunk_text(text)
    all_chunks.extend(chunks)

print(f"总共生成 {len(all_chunks)} 个文本块")

# -----------------------------
# 4️⃣ 生成 embeddings
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_chunks, normalize_embeddings=True)
embeddings = np.array(embeddings, dtype='float32')
print("Embeddings shape:", embeddings.shape)

# -----------------------------
# 5️⃣ 建立 FAISS 索引
# -----------------------------
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
print("FAISS index created with", index.ntotal, "vectors")

# -----------------------------
# 6️⃣ 查询示例
# -----------------------------
query_text = "What is retrieval-augmented generation?"
query_embedding = model.encode([query_text], normalize_embeddings=True)
query_embedding = np.array(query_embedding, dtype='float32')

k = 3
distances, indices = index.search(query_embedding, k)

print("Top-k chunk indices:", indices[0])
print("Distances:", distances[0])
print("Top-k chunks:")
for idx in indices[0]:
    print("\n--- Chunk ---\n", all_chunks[idx])
