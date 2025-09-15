# evaluator.py
import json
from typing import List, Dict

def create_test_questions() -> List[str]:
    """创建测试问题"""
    return [
        "What is the main contribution of transformer architectures?",
        "How do authors typically evaluate NLP models?",
        "What datasets are commonly used in machine learning research?",
        "What limitations are often mentioned in AI papers?",
        "How does this approach compare to previous work in the field?"
    ]

def mock_evaluation(test_questions: List[str]) -> List[Dict]:
    """模拟模型评估"""
    print("📊 开始模拟模型评估...")
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"  评估问题 {i}/{len(test_questions)}: {question[:50]}...")
        
        # 模拟基础模型回答
        base_answer = f"Base model response to: {question}. This is a generic response without specific domain knowledge, providing broad overview but lacking technical depth."
        
        # 模拟微调模型回答
        ft_answer = f"Fine-tuned model response to: {question}. This response demonstrates specialized academic knowledge, using domain-specific terminology and citing relevant research findings from the training data."
        
        results.append({
            "question": question,
            "base_model_answer": base_answer,
            "fine_tuned_answer": ft_answer,
            "improvement": "Fine-tuned model provides more specific, accurate, and technically detailed information"
        })
    
    return results

def save_evaluation_results(results: List[Dict], filename: str = "evaluation_results.json"):
    """保存评估结果"""
    output_path = f"data/{filename}"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ 已保存评估结果到 {output_path}")

if __name__ == "__main__":
    questions = create_test_questions()
    results = mock_evaluation(questions)
    save_evaluation_results(results)