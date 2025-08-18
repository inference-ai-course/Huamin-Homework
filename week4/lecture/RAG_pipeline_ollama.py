# RAG_pipeline_ollama.py
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from IPython.display import display, HTML


class RAGClass:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.documents = []
        self.text_chunks = []
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None

    def load_documents(self):
        loader = TextLoader(self.data_path)
        self.documents = loader.load()
        for i, doc in enumerate(self.documents):
            preview = doc.page_content[:200].replace('\n', '')
            display(HTML(f"Document {i+1} content preview:{preview}{'...' if len(doc.page_content) > 200 else ''}"))
        return self.documents

    def split_documents(self, chunk_size=500, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.text_chunks = text_splitter.split_documents(self.documents)
        print(f"Split documents into {len(self.text_chunks)} chunks.")
        for i, chunk in enumerate(self.text_chunks):
            formatted_text = chunk.page_content.replace('. ', '.')
            display(HTML(f"Chunk {i+1}:{formatted_text}"))
        return self.text_chunks

    def create_vectorstore(self):
        if not self.text_chunks:
            raise ValueError("No text chunks found. Please split documents before creating the vector store.")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = Chroma.from_documents(
            self.text_chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )
        print("✅ Vectorstore created with local Ollama embeddings.")
        display(HTML("Vectorstore Contents:"))
        for i, doc in enumerate(self.text_chunks):
            formatted_text = doc.page_content.replace('. ', '.')
            display(HTML(f"Document {i+1}: {formatted_text}"))
        return self.vectorstore

    def setup_retriever(self):
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized.")
        self.retriever = self.vectorstore.as_retriever()
        print("Retriever set up from vectorstore.")
        display(HTML(f"Retriever details: {self.retriever}"))
        return self.retriever

    def setup_qa_chain(self, model_name="llama3"):
        if self.retriever is None:
            raise ValueError("Retriever not initialized.")
        llm = OllamaLLM(model=model_name, temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.retriever
        )
        print(f"✅ QA chain initialized with local Ollama model: {model_name}")
        display(HTML(f"QA chain details: {self.qa_chain}"))
        return self.qa_chain

    # ------------------------
    # 辅助方法：统一提取文本
    # ------------------------
    def _extract_text(self, result):
        """
        提取 QA chain 返回文本
        """
        if isinstance(result, dict):
            # 优先取 'result'，其次 'output_text'
            if "result" in result:
                return result["result"]
            elif "output_text" in result:
                return result["output_text"]
            else:
                return str(result)
        return str(result)

    def answer_query(self, query: str):
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized.")
        result = self.qa_chain.invoke(query)
        answer = self._extract_text(result)
        display(HTML(f"Query: {query}<br>Answer: {answer}"))
        return answer

    def evaluate(self, queries: list, ground_truths: list):
        if len(queries) != len(ground_truths):
            raise ValueError("Queries and ground truths must be of the same length.")
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized.")
        correct = 0
        for idx, (query, truth) in enumerate(zip(queries, ground_truths)):
            result = self.qa_chain.invoke(query)
            answer = self._extract_text(result)
            display(HTML(f"Query {idx+1}: {query}<br>Expected: {truth}<br>Model Answer: {answer}"))
            if truth.lower() in answer.lower():
                correct += 1
        accuracy = correct / len(queries)
        display(HTML(f"Evaluation Accuracy: {accuracy * 100:.2f}%"))
        return accuracy
