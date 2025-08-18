# main.py
from RAG_pipeline_ollama import RAGClass
from IPython.display import display, HTML

# -----------------------------
# 1. 初始化 RAGClass
# -----------------------------
data_file = "my_text_file.txt"  # 替换成你的文本文件路径
rag = RAGClass(data_path=data_file)

# -----------------------------
# 2. 加载和预览文档
# -----------------------------
rag.load_documents()

# -----------------------------
# 3. 拆分文档为 chunks
# -----------------------------
rag.split_documents(chunk_size=500, chunk_overlap=50)

# -----------------------------
# 4. 创建向量库 (vectorstore)
# -----------------------------
rag.create_vectorstore()

# -----------------------------
# 5. 设置检索器 (retriever)
# -----------------------------
rag.setup_retriever()

# -----------------------------
# 6. 初始化 QA 链
# -----------------------------
# model_name 可根据 Ollama 本地模型替换，比如 "llama3", "mistral", "qwen2"
rag.setup_qa_chain(model_name="llama3")

# -----------------------------
# 7. 测试回答一个查询
# -----------------------------
test_query = "What is Retrieval-Augmented Generation?"
answer = rag.answer_query(test_query)
display(HTML(f"<h3>Test Query Answer:</h3>{answer}"))

# -----------------------------
# 8. 系统评估示例
# -----------------------------
sample_queries = [
    "Define RAG.",
    "Explain vector databases."
]
sample_ground_truths = [
    "Retrieval-Augmented Generation",
    "Vector databases store embeddings"
]

accuracy = rag.evaluate(sample_queries, sample_ground_truths)
display(HTML(f"<h3>Evaluation Accuracy: {accuracy*100:.2f}%</h3>"))
