# Cell 1: 导入和配置
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from fastapi import FastAPI
import uvicorn
from pathlib import Path
import re
from datetime import datetime
from contextlib import asynccontextmanager
import nest_asyncio
import requests
from typing import List, Dict
import json
import glob

# 修复事件循环问题
nest_asyncio.apply()

# -----------------------------
# 全局配置
# -----------------------------
PDF_DIR = Path("arxiv_pdfs")
TXT_DIR = Path("arxiv_texts")  # 使用week4已经生成的文本文件
DB_PATH = "hybrid_search.db"
MODEL_NAME = "all-MiniLM-L6-v2"

# 确保目录存在
PDF_DIR.mkdir(exist_ok=True)
TXT_DIR.mkdir(exist_ok=True)

# Cell 2: 文本处理函数
def extract_metadata_from_filename(filename):
    """从文件名中提取元数据"""
    # 移除文件扩展名
    name_without_ext = os.path.splitext(filename)[0]
    
    # 根据实际文件名格式进行调整
    
    # 示例：简单的分割方式
    parts = name_without_ext.split('_')
    
    metadata = {
        "title": parts[0] if len(parts) > 0 else "未知标题",
        "authors": parts[1] if len(parts) > 1 else "未知作者",
        "year": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 2023,
        "abstract": "",  # 默认空摘要
        "keywords": ""   # 默认空关键词
    }
    
    return metadata

def load_text_from_file(txt_file: Path) -> str:
    """从文本文件加载内容"""
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return ""

def chunk_text(doc_id: str, text: str, chunk_size=500, overlap=100) -> list:
    """文本分块"""
    words = text.split()
    chunks = []
    i = 0
    chunk_id = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "text": chunk_text
        })
        i += chunk_size - overlap
        chunk_id += 1
    return chunks

# Cell 3: 数据库初始化
def init_database():
    """初始化数据库，创建必要的表"""
    conn = sqlite3.connect('document_search.db')
    
    # 创建文档表 - 修正为与插入语句匹配的结构
    conn.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        authors TEXT,
        year INTEGER,
        abstract TEXT,
        keywords TEXT,
        file_path TEXT NOT NULL,
        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 创建FTS5虚拟表用于全文搜索
    conn.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks USING fts5(
        chunk_id UNINDEXED,
        content,
        doc_id UNINDEXED,
        title
    );
    """)
    
    conn.commit()
    print("数据库初始化完成！")
    return conn

# Cell 4: 数据库填充函数
def split_text_into_chunks(text, chunk_size=512, overlap=50):
    """
    将文本分割成重叠的块
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        if end > text_length:
            end = text_length
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        # 移动起始位置，考虑重叠
        start += chunk_size - overlap
        
        # 确保不会无限循环
        if start >= text_length:
            break
    
    return chunks

def populate_database(conn):
    """从文本文件填充数据库"""
    cursor = conn.cursor()
    
    # 获取所有文本文件
    txt_files = glob.glob(os.path.join(TXT_DIR, "*.txt"))
    print(f"找到 {len(txt_files)} 个文本文件")
    
    for txt_file in txt_files:
        # 从文件名提取文档ID（不含扩展名）
        doc_id = os.path.splitext(os.path.basename(txt_file))[0]
        
        # 从文件名提取元数据
        metadata = extract_metadata_from_filename(doc_id)
        
        # 插入文档元数据 - 添加 file_path 字段
        cursor.execute("""
            INSERT OR REPLACE INTO documents (doc_id, title, authors, year, abstract, keywords, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (doc_id, metadata["title"], metadata["authors"], 
              metadata["year"], metadata["abstract"], "", txt_file))  # 添加 txt_file 作为 file_path
        
        # 加载文本内容
        text = load_text_from_file(txt_file)
        
        # 分割文本为块
        chunks = split_text_into_chunks(text)
        
        # 插入文本块到FTS表
        for i, chunk in enumerate(chunks):
            cursor.execute("""
                INSERT INTO doc_chunks (chunk_id, content, doc_id, title)
                VALUES (?, ?, ?, ?)
            """, (i, chunk, doc_id, metadata["title"]))
    
    conn.commit()
    print(f"成功处理 {len(txt_files)} 个文件")

# -----------------------------
# 4️⃣ 混合索引类
# -----------------------------
class HybridIndex:
    def __init__(self, conn, embed_model="all-MiniLM-L6-v2"):
        self.conn = conn
        self.embedder = SentenceTransformer(embed_model)
        self.index = None
        self.bm25 = None
        self.chunk_data = []
        self.chunk_texts = []

    def build_indices(self):
        """构建FAISS和BM25索引"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT doc_id, chunk_id, content FROM doc_chunks")
        rows = cursor.fetchall()
        
        embeddings, self.chunk_data, self.chunk_texts = [], [], []
        
        for doc_id, chunk_id, content in rows:
            vec = self.embedder.encode(content, convert_to_numpy=True)
            embeddings.append(vec)
            self.chunk_data.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "content": content
            })
            self.chunk_texts.append(content)
                   
        if embeddings:
            embeddings = np.array(embeddings).astype("float32")
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings)
            
            tokenized_docs = [doc.split() for doc in self.chunk_texts]
            self.bm25 = BM25Okapi(tokenized_docs)
            
            print(f"Built indices with {len(embeddings)} chunks")
        else:
            print("Warning: No embeddings created")

    def faiss_search(self, query, k=3):
        if self.index is None:
            return []
        
        q_vec = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(q_vec, k)
        
        results = []
        for i, d in zip(I[0], D[0]):
            if i < len(self.chunk_data):
                similarity = float(1 / (1 + d))
                chunk_info = self.chunk_data[i]
                results.append({
                    "doc_id": chunk_info["doc_id"],
                    "chunk_id": chunk_info["chunk_id"],
                    "score": similarity,
                    "content": chunk_info["content"][:200] + "..."
                })
        return results

    def fts5_search(self, query, k=3):
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT doc_id, chunk_id, content 
            FROM doc_chunks 
            WHERE doc_chunks MATCH ? 
            ORDER BY rank
            LIMIT ?
        """, (query, k))
        
        results = []
        for doc_id, chunk_id, content in cursor.fetchall():
            results.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "score": 0.5,
                "content": content[:200] + "..."
            })
        
        return results

    def bm25_search(self, query, k=3):
        if not self.bm25:
            return []
        
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        
        for idx in top_indices:
            if idx < len(self.chunk_data):
                normalized_score = float(min(scores[idx] / 20, 1.0))
                chunk_info = self.chunk_data[idx]
                results.append({
                    "doc_id": chunk_info["doc_id"],
                    "chunk_id": chunk_info["chunk_id"],
                    "score": normalized_score,
                    "content": chunk_info["content"][:200] + "..."
                })
        
        return results

    def hybrid_search(self, query, k=3, alpha=0.6, use_fts5=False):
        vector_results = self.faiss_search(query, k*2)
        if use_fts5:
            keyword_results = self.fts5_search(query, k*2)
        else:
            keyword_results = self.bm25_search(query, k*2)
        
        def get_key(result):
            return f"{result['doc_id']}_{result['chunk_id']}"
        
        combined = {}
        
        for result in vector_results:
            key = get_key(result)
            combined[key] = {**result, "vector_score": result["score"], "keyword_score": 0.0}
       
        for result in keyword_results:
            key = get_key(result)
            if key in combined:
                combined[key]["keyword_score"] = result["score"]
            else:
                combined[key] = {**result, "vector_score": 0.0, "keyword_score": result["score"]}
        
        final_results = []
        for key, result in combined.items():
            hybrid_score = float(alpha * result["vector_score"] + (1 - alpha) * result["keyword_score"])
            final_results.append({
                "doc_id": result["doc_id"],
                "chunk_id": result["chunk_id"],
                "score": hybrid_score,
                "vector_score": result["vector_score"],
                "keyword_score": result["keyword_score"],
                "content": result["content"]
            })
        
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:k]

# -----------------------------
# 5️⃣ Jupyter专用函数
# -----------------------------
def initialize_system():
    """在Jupyter中初始化整个系统"""
    print("Initializing database...")
    conn = init_database()
    print("Populating database from PDF files...")
    populate_database(conn)
    print("Building search indices...")
    hindex = HybridIndex(conn)
    hindex.build_indices()
    print("System initialized successfully!")
    return conn, hindex

def test_queries(hindex, queries=None):
    """测试查询"""
    if queries is None:
        queries = [
            "machine translation", "transformer", "neural network",
            "deep learning", "artificial intelligence", "natural language processing",
            "computer vision", "reinforcement learning", "graph neural networks",
            "self-supervised learning", "unsupervised learning", "supervised learning",
        ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        results = hindex.hybrid_search(query, 3)
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result['score']:.3f} | Doc: {result['doc_id']} | Chunk: {result['chunk_id']}")
            print(f"     Content: {result['content']}")

def run_server_in_background():
    """在后台运行服务器（Jupyter专用）"""
    import threading
    
    def start_server():
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        server.run()
         
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    print("Server started in background on http://localhost:8000")
    return server_thread
