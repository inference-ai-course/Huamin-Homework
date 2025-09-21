# data_formatter-test.py, Deepseek
import json

# 假设您已经将标注数据加载到 summary_pairs 变量中
# 这部分应该在代码外部完成

def create_reward_dataset(summary_pairs):
    data = []
    for pair in summary_pairs:
        data.append({
            "chosen": pair["preferred"],
            "rejected": pair["other"]
        })

    with open("reward_data.jsonl", "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    return data

# 使用示例
if __name__ == "__main__":
    # 首先从文件加载数据（这部分可以单独进行）
    summary_pairs = []
    try:
        with open("annotated_summaries.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                summary_pairs.append(json.loads(line.strip()))
    except FileNotFoundError:
        print("请先创建 annotated_summaries.jsonl 文件")
        exit()

    # 然后使用作业中的代码格式化
    formatted_data = create_reward_dataset(summary_pairs)
    print(f"创建了 {len(formatted_data)} 个训练样本")
