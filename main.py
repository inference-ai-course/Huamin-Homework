# main.py
import json
import sys
from pathlib import Path
import subprocess

# 配置参数
NUM_PAPERS_TO_PROCESS = 10    # 处理多少篇论文（你有50篇，这里处理10篇）
NUM_QA_PAIRS_PER_PAPER = 3    # 每篇论文生成几个问答对

def run_w4_homework_if_needed():
    """如果需要，运行w4-homework.py来获取数据"""
    pdf_dir = Path("arxiv_pdfs")
    text_dir = Path("arxiv_texts")
    
    # 检查是否已有数据
    has_pdfs = pdf_dir.exists() and any(pdf_dir.glob("*.pdf"))
    has_texts = text_dir.exists() and any(text_dir.glob("*.txt"))
    
    if has_pdfs and has_texts:
        print("✅ 发现已有的PDF和文本数据，直接使用")
        pdf_files = list(pdf_dir.glob("*.pdf"))
        text_files = list(text_dir.glob("*.txt"))
        print(f"   PDF文件: {len(pdf_files)} 个")
        print(f"   文本文件: {len(text_files)} 个")
        return True
    
    print("📥 没有找到足够的数据，需要运行w4-homework.py...")
    try:
        # 运行你的w4-homework.py
        result = subprocess.run(
            [sys.executable, "w4-Homework.py"],
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        if result.returncode == 0:
            print("✅ w4-homework.py 运行成功")
            print(result.stdout[-500:])  # 显示最后500字符的输出
            return True
        else:
            print("❌ w4-homework.py 运行失败")
            print("错误输出:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ w4-homework.py 运行超时")
        return False
    except Exception as e:
        print(f"❌ 运行w4-homework.py时出错: {e}")
        return False

def get_existing_papers():
    """从已有文本文件中获取论文数据"""
    text_dir = Path("arxiv_texts")
    papers = []
    
    if not text_dir.exists():
        print("❌ arxiv_texts目录不存在")
        return []
    
    text_files = list(text_dir.glob("*.txt"))
    if not text_files:
        print("❌ 没有找到文本文件")
        return []
    
    print(f"📖 从 {len(text_files)} 个文本文件中读取数据...")
    
    for i, text_file in enumerate(text_files[:NUM_PAPERS_TO_PROCESS]):
        try:
            with open(text_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 简单解析文本文件内容
            lines = content.split('\n')
            title = "未知标题"
            abstract = "无摘要"
            full_text = content
            
            # 尝试提取标题和摘要
            for line in lines:
                if line.startswith("Title:"):
                    title = line.replace("Title:", "").strip()
                elif line.startswith("Abstract:"):
                    abstract = line.replace("Abstract:", "").strip()
                elif line.startswith("Text:"):
                    full_text = line.replace("Text:", "").strip()
            
            papers.append({
                "title": title,
                "abstract": abstract,
                "full_text": full_text[:1000],  # 限制长度
                "source_file": text_file.name
            })
            
            print(f"   ✅ 已加载: {title}")
            
        except Exception as e:
            print(f"   ❌ 读取 {text_file} 失败: {e}")
    
    return papers

def run_qa_generation(papers):
    """运行问答对生成"""
    print("\n🤖 生成问答对...")
    try:
        # 动态导入以避免循环依赖
        from qa_generator import generate_all_qa_papers, save_qa_pairs
        
        qa_pairs = generate_all_qa_papers()
        if qa_pairs:
            save_qa_pairs(qa_pairs)
            print(f"✅ 成功生成 {len(qa_pairs)} 个问答对")
            
            # 显示一些示例
            print("\n📋 问答对示例:")
            for i, qa in enumerate(qa_pairs[:3]):
                print(f"   {i+1}. Q: {qa['question'][:50]}...")
                print(f"      A: {qa['answer'][:50]}...")
                print()
            
            return qa_pairs
        else:
            print("❌ 没有生成问答对")
            return None
            
    except Exception as e:
        print(f"❌ 问答生成错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_data_formatting(qa_pairs):
    """运行数据格式化"""
    print("\n📝 格式化数据...")
    try:
        from data_formatter import convert_to_jsonl_format
        
        formatted_data = convert_to_jsonl_format(qa_pairs)
        print(f"✅ 成功格式化 {len(formatted_data)} 条数据")
        return formatted_data
        
    except Exception as e:
        print(f"❌ 数据格式化错误: {e}")
        return None
    
def run_model_training():
    """运行模型训练（真实GPU训练）"""
    print("\n⚙️ 训练模型...")
    try:
        from model_trainer import real_fine_tuning  # ← 导入真实训练函数
        
        model_info = real_fine_tuning()  # ← 调用真实训练函数
        print("✅ 模型训练完成")
        return model_info
    except Exception as e:
        print(f"❌ 模型训练错误: {e}")
        return {"status": "error", "message": str(e)}

def run_evaluation():
    """运行模型评估"""
    print("\n📊 评估模型...")
    try:
        from evaluator import create_test_questions, mock_evaluation, save_evaluation_results
        
        test_questions = create_test_questions()
        evaluation_results = mock_evaluation(test_questions)
        save_evaluation_results(evaluation_results)
        print(f"✅ 完成 {len(evaluation_results)} 个测试问题的评估")
        return evaluation_results
        
    except Exception as e:
        print(f"❌ 模型评估错误: {e}")
        return []

def generate_summary_report(papers, qa_pairs, model_info, evaluation_results):
    """生成总结报告"""
    report = {
        "total_papers_available": len(papers),
        "total_qa_pairs_generated": len(qa_pairs) if qa_pairs else 0,
        "model_training_info": model_info,
        "evaluation_summary": {
            "questions_tested": len(evaluation_results),
            "improvements_noted": len([r for r in evaluation_results if "improvement" in r])
        },
        "config": {
            "papers_processed": NUM_PAPERS_TO_PROCESS,
            "qa_per_paper": NUM_QA_PAIRS_PER_PAPER
        }
    }
    
    with open("data/summary_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("📊 总结报告已生成: data/summary_report.json")
    
    # 打印简要信息
    print(f"\n🎯 项目总结:")
    print(f"   • 可用论文: {len(papers)} 篇")
    print(f"   • 生成问答: {len(qa_pairs) if qa_pairs else 0} 对")
    print(f"   • 测试问题: {len(evaluation_results)} 个")
    print(f"   • 模型状态: {model_info.get('status', 'N/A') if model_info else 'N/A'}")

def main():
    """主函数"""
    print("=" * 60)
    print("🤖 Week 7 Homework: 学术问答系统")
    print("=" * 60)
    
    # 确保必要的目录存在
    required_dirs = ["arxiv_pdfs", "arxiv_texts", "data"]
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"📁 确保目录存在: {dir_path}")
    
    # 1. 数据准备 - 运行或使用现有的w4-homework数据
    print("\n1. 📥 数据准备阶段")
    if not run_w4_homework_if_needed():
        print("❌ 数据准备失败，程序退出")
        return
    
    # 2. 获取论文数据
    print("\n2. 📄 读取论文数据")
    papers = get_existing_papers()
    if not papers:
        print("❌ 没有可用的论文数据，程序退出")
        return
    
    # 3. 生成问答对
    qa_pairs = run_qa_generation(papers)
    if not qa_pairs:
        print("❌ 问答对生成失败，程序退出")
        return
    
    # 4. 数据格式化
    formatted_data = run_data_formatting(qa_pairs)
    if not formatted_data:
        print("⚠️  数据格式化失败，继续后续步骤")
    
    # 5. 模型训练（模拟）
    model_info = run_model_training()
    
    # 6. 模型评估
    evaluation_results = run_evaluation()
    
    # 7. 生成总结报告
    generate_summary_report(papers, qa_pairs, model_info, evaluation_results)
    
    print("\n" + "=" * 60)
    print("🎉 所有步骤完成！")
    print("=" * 60)
    
    # 显示下一步建议
    print("\n📋 下一步建议:")
    print("1. 查看 data/synthetic_qa_data.json - 生成的问答对")
    print("2. 查看 data/summary_report.json - 项目总结报告")
    print("3. 查看 arxiv_texts/ - 提取的论文文本")
    print("4. 如果需要真实训练，设置OPENAI_API_KEY并修改代码")

if __name__ == "__main__":
    main()