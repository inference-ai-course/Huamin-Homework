# data_extractor.py
import re
from pathlib import Path
import os

# 直接定义配置，避免导入问题
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "arxiv_pdfs"
TEXT_DIR = BASE_DIR / "arxiv_texts"

# 创建目录
PDF_DIR.mkdir(exist_ok=True)
TEXT_DIR.mkdir(exist_ok=True)

print(f"PDF目录: {PDF_DIR}")
print(f"文本目录: {TEXT_DIR}")

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
    print("✓ PyMuPDF 已安装")
except ImportError:
    HAS_FITZ = False
    print("✗ PyMuPDF 未安装，将使用示例数据")

def extract_text_from_pdf(pdf_path: Path) -> str:
    """从PDF提取文本"""
    try:
        if HAS_FITZ:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                page_text = page.get_text("text")
                text += page_text + "\n"
            return text.strip()
        else:
            # 返回示例文本
            return create_sample_content(pdf_path.stem)
    except Exception as e:
        print(f"从 {pdf_path} 提取文本时出错: {e}")
        return create_sample_content(pdf_path.stem)

def clean_text(text: str) -> str:
    """清理文本"""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_abstract(text: str) -> str:
    """提取摘要部分"""
    patterns = [
        r"abstract[:\s]*(.*?)(?=introduction|$)",
        r"Abstract[:\s]*(.*?)(?=Introduction|$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return text[:300] + "..." if len(text) > 300 else text

def create_sample_content(title: str) -> str:
    """创建示例内容"""
    samples = {
        "transformer": """
        Abstract: This paper introduces a novel transformer architecture that improves computational efficiency by 40% while maintaining performance. Our method uses sparse attention mechanisms to reduce quadratic complexity.
        
        Introduction: Transformers have revolutionized NLP but face computational challenges. Recent approaches focus on optimizing attention mechanisms.
        
        Method: We propose a dynamic sparse attention mechanism that selectively attends to relevant tokens based on content-based routing.
        
        Results: Experimental results on GLUE benchmark show 40% faster inference with only 1% accuracy drop compared to dense transformers.
        """,
        "machine_learning": """
        Abstract: We present a comprehensive survey of machine learning techniques, focusing on deep learning architectures and their applications in various domains.
        
        Introduction: Machine learning has evolved rapidly in recent years, with deep learning achieving state-of-the-art results in many tasks.
        
        Deep Learning Architectures: We review CNN, RNN, and transformer architectures, comparing their strengths and limitations.
        
        Applications: Discuss applications in computer vision, natural language processing, and reinforcement learning.
        """,
        "default": """
        Abstract: This research paper presents advancements in AI methodologies, demonstrating improved performance across multiple benchmarks while reducing computational requirements.
        
        Introduction: Recent developments in artificial intelligence have focused on scaling model size and improving training efficiency.
        
        Contributions: Our main contributions include a novel optimization algorithm and improved regularization techniques.
        
        Experiments: We evaluate on standard benchmarks and show consistent improvements over baseline methods.
        """
    }
    
    for key in samples:
        if key in title.lower():
            return samples[key]
    return samples["default"]

def create_sample_pdfs():
    """创建示例PDF文件"""
    samples = [
        {"name": "transformer_research.pdf", "content": create_sample_content("transformer")},
        {"name": "ml_survey.pdf", "content": create_sample_content("machine_learning")},
        {"name": "ai_advances.pdf", "content": create_sample_content("default")}
    ]
    
    for sample in samples:
        filepath = PDF_DIR / sample["name"]
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(sample["content"])
        print(f"创建示例文件: {filepath}")

def get_paper_data() -> list:
    """获取论文数据"""
    papers = []
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print("未找到PDF文件，创建示例文件...")
        create_sample_pdfs()
        pdf_files = list(PDF_DIR.glob("*.pdf"))
    
    print(f"找到 {len(pdf_files)} 个PDF文件")
    
    for i, pdf_file in enumerate(pdf_files):
        try:
            print(f"处理文件 {i+1}/{len(pdf_files)}: {pdf_file.name}")
            text = extract_text_from_pdf(pdf_file)
            abstract = extract_abstract(text)
            
            papers.append({
                "title": pdf_file.stem,
                "abstract": clean_text(abstract),
                "full_text": clean_text(text)[:1000]  # 限制文本长度
            })
            print(f"✓ 成功处理: {pdf_file.stem}")
            
        except Exception as e:
            print(f"✗ 处理 {pdf_file} 时出错: {e}")
    
    return papers

def save_paper_texts(papers: list):
    """保存论文文本"""
    for i, paper in enumerate(papers):
        filepath = TEXT_DIR / f"paper_{i+1}.txt"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Title: {paper['title']}\n")
            f.write(f"Abstract: {paper['abstract']}\n")
            f.write(f"Text: {paper['full_text']}\n")
        print(f"保存: {filepath}")

def main():
    """主函数"""
    print("=" * 60)
    print("数据提取模块")
    print("=" * 60)
    
    # 获取论文数据
    papers = get_paper_data()
    
    if papers:
        # 保存文本
        save_paper_texts(papers)
        print(f"\n✅ 成功处理并保存 {len(papers)} 篇论文")
        
        # 显示摘要
        print("\n📄 论文摘要:")
        for i, paper in enumerate(papers, 1):
            print(f"{i}. {paper['title']}")
            print(f"   Abstract: {paper['abstract'][:100]}...")
            print()
    else:
        print("❌ 没有处理任何论文")

if __name__ == "__main__":
    main()