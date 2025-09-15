# data_formatter.py
import json
from pathlib import Path
from typing import List, Dict

def convert_to_jsonl_format(qa_list: List[Dict], output_file: str = "synthetic_qa.jsonl") -> List[Dict]:
    """将问答对转换为JSONL格式"""
    
    system_prompt = "You are a helpful academic Q&A assistant specialized in scholarly content."
    formatted_data = []
    
    for qa in qa_list:
        user_q = qa["question"]
        assistant_a = qa["answer"]
        
        # 构建对话格式
        full_prompt = f"<|system|>{system_prompt}<|user|>{user_q}<|assistant|>{assistant_a}"
        
        formatted_data.append({"text": full_prompt})
    
    # 写入JSONL文件
    output_path = Path("data") / output_file
    with open(output_path, "w", encoding="utf-8") as outfile:
        for entry in formatted_data:
            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"✅ 已转换并保存 {len(formatted_data)} 条数据到 {output_path}")
    return formatted_data

if __name__ == "__main__":
    # 测试代码
    test_qa = [
        {"question": "What is AI?", "answer": "Artificial Intelligence is..."},
        {"question": "What is ML?", "answer": "Machine Learning is..."}
    ]
    convert_to_jsonl_format(test_qa, "test_output.jsonl")
