# evaluator.py
from evaluate import load

def evaluate_summaries():
   
    # 示例数据 - 这里应该是您生成的新摘要和参考摘要
    generated_summaries = [
        "这是生成的摘要1",
        "这是生成的摘要2"
    ]
    
    reference_summaries = [
        "这是参考摘要1", 
        "这是参考摘要2"
    ]
    
    # 计算ROUGE和BERTScore
    rouge = load("rouge")
    bertscore = load("bertscore")
    
    results_rouge = rouge.compute(predictions=generated_summaries, references=reference_summaries)
    results_bertscore = bertscore.compute(predictions=generated_summaries, references=reference_summaries, lang="en")
    
    print("ROUGE结果:", results_rouge)
    print("BERTScore结果:", results_bertscore)
    
    return results_rouge, results_bertscore

if __name__ == "__main__":
    evaluate_summaries()
