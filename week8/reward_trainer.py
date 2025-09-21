# reward_trainer.py
from trl import RewardTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from datasets import load_dataset

def train_reward_model():
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=1)
    
    # 加载数据集
    dataset = load_dataset("json", data_files="reward_data.jsonl", split="train")
    
    # 预处理函数
    def preprocess(example):
        return tokenizer(example["chosen"], example["rejected"], truncation=True, padding="max_length")
    
    dataset = dataset.map(preprocess, batched=True)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="reward_model",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="no",
        save_strategy="epoch",
        logging_steps=10,
        fp16=True
    )
    
    # 创建训练器
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    
    # 开始训练
    print("开始训练奖励模型...")
    trainer.train()
    trainer.save_model()
    print("奖励模型训练完成！")

if __name__ == "__main__":
    train_reward_model()
