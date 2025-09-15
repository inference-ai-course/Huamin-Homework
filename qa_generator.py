# qa_generator.py
import json
import re
import time
from pathlib import Path
from typing import List, Dict

# 直接定义配置，避免导入冲突
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# 配置参数
OPENAI_API_KEY = "sk-your-openai-api-key-here"  # 替换为你的API密钥
NUM_QA_PAIRS_PER_PAPER = 2
NUM_PAPERS_TO_PROCESS = 3

print("问答对生成器")
print(f"数据目录: {DATA_DIR}")

def mock_gpt4_response(abstract: str, title: str, num_pairs: int) -> str:
    """模拟GPT-4响应"""
    return f'''
[
    {{
        "question": "What is the main contribution of the paper '{title}'?",
        "answer": "The main contribution is a novel approach to natural language processing that improves performance on benchmark tasks by 25% compared to previous methods."
    }},
    {{
        "question": "What methodology did the authors use in {title}?",
        "answer": "The authors used transformer-based architecture with specialized attention mechanisms and dynamic routing to optimize computational efficiency."
    }},
    {{
        "question": "Does the paper mention using XYZ dataset for evaluation?",
        "answer": "No, the paper does not mention using XYZ dataset. The authors used standard benchmarks including GLUE and SuperGLUE for evaluation."
    }}
]
'''

def generate_qa_pairs(abstract: str, title: str, num_pairs: int = 2) -> List[Dict]:
    """生成问答对（模拟版本）"""
    print(f"为论文 '{title}' 生成 {num_pairs} 个问答对...")
    
    try:
        # 在实际应用中取消注释以下代码
        # import openai
        # openai.api_key = OPENAI_API_KEY
        
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[
        #         {"role": "system", "content": "You are a research assistant that creates educational content."},
        #         {"role": "user", "content": f"Create {num_pairs} Q&A pairs for this abstract: {abstract}"}
        #     ],
        #     temperature=0.7,
        #     max_tokens=1500
        # )
        # content = response.choices[0].message.content
        
        # 使用模拟响应
        content = mock_gpt4_response(abstract, title, num_pairs)
        
        # 解析JSON
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            qa_pairs = json.loads(json_match.group())
            return qa_pairs[:num_pairs]  # 只返回请求的数量
        else:
            print("无法解析响应中的JSON，使用默认问答对")
            return create_default_qa_pairs(title, num_pairs)
            
    except Exception as e:
        print(f"生成问答对时出错: {e}")
        return create_default_qa_pairs(title, num_pairs)

def create_default_qa_pairs(title: str, num_pairs: int) -> List[Dict]:
    """创建默认的问答对"""
    default_pairs = [
        {
            "question": f"What is the primary focus of the paper '{title}'?",
            "answer": f"The paper '{title}' focuses on developing innovative machine learning techniques to address computational challenges in natural language processing."
        },
        {
            "question": f"What are the key findings reported in {title}?",
            "answer": f"The key findings include significant improvements in model efficiency and performance on standard benchmarks, with a 30% reduction in computational requirements."
        },
        {
            "question": f"Did the authors of {title} compare their approach with existing methods?",
            "answer": f"Yes, the authors conducted comprehensive comparisons with state-of-the-art methods and demonstrated superior performance across multiple evaluation metrics."
        }
    ]
    return default_pairs[:num_pairs]

def read_paper_texts() -> List[Dict]:
    """读取之前提取的论文文本"""
    text_dir = BASE_DIR / "arxiv_texts"
    papers = []
    
    if not text_dir.exists():
        print(f"目录 {text_dir} 不存在，使用示例数据")
        return create_sample_papers()
    
    text_files = list(text_dir.glob("*.txt"))
    if not text_files:
        print("没有找到文本文件，使用示例数据")
        return create_sample_papers()
    
    for text_file in text_files:
        try:
            with open(text_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 简单解析文本文件
            lines = content.split('\n')
            title = lines[0].replace("Title: ", "") if lines[0].startswith("Title: ") else text_file.stem
            abstract = lines[1].replace("Abstract: ", "") if len(lines) > 1 and lines[1].startswith("Abstract: ") else "No abstract available"
            
            papers.append({
                "title": title,
                "abstract": abstract,
                "source_file": text_file.name
            })
            
        except Exception as e:
            print(f"读取文件 {text_file} 时出错: {e}")
    
    return papers

def create_sample_papers() -> List[Dict]:
    """创建示例论文数据"""
    return [
        {
            "title": "transformer_research",
            "abstract": "This paper introduces a novel transformer architecture that improves computational efficiency while maintaining performance. Our approach utilizes sparse attention mechanisms.",
            "source_file": "sample_1.txt"
        },
        {
            "title": "machine_learning_survey", 
            "abstract": "We present a comprehensive survey of machine learning techniques, focusing on deep learning architectures and their applications in various domains.",
            "source_file": "sample_2.txt"
        },
        {
            "title": "ai_advances",
            "abstract": "This research paper presents advancements in AI methodologies, demonstrating improved performance across multiple benchmarks while reducing computational requirements.",
            "source_file": "sample_3.txt"
        }
    ]

def generate_all_qa_papers() -> List[Dict]:
    """为所有论文生成问答对"""
    print("读取论文数据...")
    papers = read_paper_texts()
    
    if not papers:
        print("没有论文数据可用")
        return []
    
    print(f"找到 {len(papers)} 篇论文")
    all_qa_pairs = []
    
    for i, paper in enumerate(papers[:NUM_PAPERS_TO_PROCESS]):
        print(f"\n处理论文 {i+1}/{min(NUM_PAPERS_TO_PROCESS, len(papers))}: {paper['title']}")
        
        qa_pairs = generate_qa_pairs(paper['abstract'], paper['title'], NUM_QA_PAIRS_PER_PAPER)
        
        if qa_pairs:
            for qa in qa_pairs:
                qa['paper_title'] = paper['title']
                qa['paper_abstract'] = paper['abstract'][:100] + "..."
            
            all_qa_pairs.extend(qa_pairs)
            print(f"✓ 生成 {len(qa_pairs)} 个问答对")
        else:
            print("✗ 生成问答对失败")
        
        time.sleep(0.5)  # 模拟延迟
    
    return all_qa_pairs

def save_qa_pairs(qa_pairs: List[Dict], filename: str = "synthetic_qa_data.json"):
    """保存问答对"""
    output_path = DATA_DIR / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    print(f"✓ 已保存 {len(qa_pairs)} 个问答对到 {output_path}")

def display_qa_pairs(qa_pairs: List[Dict]):
    """显示生成的问答对"""
    print("\n" + "=" * 80)
    print("生成的问答对预览:")
    print("=" * 80)
    
    for i, qa in enumerate(qa_pairs, 1):
        print(f"\n{i}. 论文: {qa.get('paper_title', 'Unknown')}")
        print(f"   问题: {qa['question']}")
        print(f"   答案: {qa['answer'][:100]}...")
        if i >= 5:  # 只显示前5个
            print(f"   ... 还有 {len(qa_pairs) - 5} 个问答对")
            break

def main():
    """主函数"""
    print("=" * 60)
    print("问答对生成模块")
    print("=" * 60)
    
    # 生成问答对
    qa_pairs = generate_all_qa_papers()
    
    if qa_pairs:
        # 保存结果
        save_qa_pairs(qa_pairs)
        
        # 显示预览
        display_qa_pairs(qa_pairs)
        
        print(f"\n✅ 成功生成 {len(qa_pairs)} 个问答对")
        print(f"📁 文件保存在: {DATA_DIR / 'synthetic_qa_data.json'}")
    else:
        print("❌ 没有生成任何问答对")

if __name__ == "__main__":
    main()