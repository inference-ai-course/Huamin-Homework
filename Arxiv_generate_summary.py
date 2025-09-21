import arxiv
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings
warnings.filterwarnings('ignore')

class ArxivSummaryGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # 初始化LLaMA 3模型和tokenizer
        self.model_name = "meta-llama/Meta-Llama-3-8B"  # 使用meta-llama/Meta-Llama-3-8B模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        # 创建文本生成pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

    def search_arxiv_papers(self, query="machine learning", max_results=10):
        """从Arxiv搜索论文"""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        for result in search.results():
            paper_info = {
                'id': result.entry_id.split('/')[-1],
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'published': result.published.strftime('%Y-%m-%d'),
                'summary': result.summary,
                'pdf_url': result.pdf_url,
                'categories': result.categories
            }
            papers.append(paper_info)
        
        return papers

    def clean_text(self, text):
        """清理文本，移除多余空格和特殊字符"""
        text = re.sub(r'\s+', ' ', text)  # 移除多余空格
        text = re.sub(r'\\[ntr]', ' ', text)  # 移除转义字符
        text = text.strip()
        return text

    def generate_summary(self, paper_text, prompt_template, temperature=0.7):
        """使用LLaMA 3生成摘要"""
        prompt = prompt_template.format(paper_text=paper_text[:3000])  # 限制输入长度
        
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=512,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = outputs[0]['generated_text']
            # 提取生成的摘要部分（移除提示词）
            summary = generated_text.replace(prompt, '').strip()
            return self.clean_text(summary)
            
        except Exception as e:
            print(f"生成摘要时出错: {e}")
            return "Error generating summary"

    def generate_summary_pairs(self, papers):
        """为每篇论文生成两个不同的摘要"""
        summary_pairs = []
        
        # 两种不同的提示词模板
        prompt_templates = [
            "Please provide a comprehensive summary of the following research paper excerpt. Focus on the main contributions, methodology, and key findings:\n\n{paper_text}\n\nSummary:",
            "Summarize the key points of this academic paper in a concise manner. Highlight the problem statement, approach, and results:\n\n{paper_text}\n\nConcise Summary:"
        ]
        
        for i, paper in enumerate(papers):
            print(f"处理论文 {i+1}/{len(papers)}: {paper['title'][:50]}...")
            
            # 生成第一个摘要（使用第一个提示词，较低temperature）
            summary1 = self.generate_summary(
                paper['summary'], 
                prompt_templates[0], 
                temperature=0.7
            )
            
            # 生成第二个摘要（使用第二个提示词，较高temperature）
            summary2 = self.generate_summary(
                paper['summary'],
                prompt_templates[1],
                temperature=0.9
            )
            
            pair = {
                'paper_id': paper['id'],
                'paper_title': paper['title'],
                'summary_1': summary1,
                'summary_2': summary2,
                'prompt_1': prompt_templates[0],
                'prompt_2': prompt_templates[1],
                'temperature_1': 0.7,
                'temperature_2': 0.9
            }
            
            summary_pairs.append(pair)
            print(f"  摘要1生成完成: {len(summary1)} 字符")
            print(f"  摘要2生成完成: {len(summary2)} 字符")
            print("-" * 50)
        
        return summary_pairs

    def save_to_jsonl(self, data, filename):
        """保存数据到JSONL文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"数据已保存到 {filename}")

def main():
    # 初始化生成器
    generator = ArxivSummaryGenerator()
    
    # 搜索论文
    print("正在从Arxiv搜索机器学习相关论文...")
    papers = generator.search_arxiv_papers(query="machine learning", max_results=10)
    print(f"找到 {len(papers)} 篇论文")
    
    # 生成摘要对
    print("开始生成摘要对...")
    summary_pairs = generator.generate_summary_pairs(papers)
    
    # 保存原始摘要对（供手动标注）
    generator.save_to_jsonl(summary_pairs, "raw_summary_pairs.jsonl")
    
    print("=" * 60)
    print("第一步完成！")
    print("已生成 raw_summary_pairs.jsonl 文件")
    print("下一步：请手动编辑该文件，将较好的摘要键名改为 'chosen'，较差的改为 'rejected'")
    print("编辑完成后，将文件重命名为 'reward_data.jsonl' 以供后续训练使用")

if __name__ == "__main__":
    main()