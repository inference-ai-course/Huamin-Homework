# model_trainer.py
import json
import torch
from pathlib import Path
from datasets import load_dataset

def check_gpu():
    """检查GPU可用性"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU可用: {gpu_name}")
        print(f"   GPU数量: {gpu_count}")
        print(f"   显存: {gpu_memory:.1f} GB")
        return True
    else:
        print("❌ 没有检测到GPU，将使用CPU（不推荐）")
        return False

def real_fine_tuning(data_file: str = "data/synthetic_qa.jsonl"):
    """真实的QLoRA微调"""
    print("🤖 开始真实模型微调...")
    
    # 检查GPU
    if not check_gpu():
        return {"status": "error", "message": "No GPU available"}
    
    try:
        from unsloth import FastLanguageModel
        from transformers import TrainingArguments
        from trl import SFTTrainer
        import transformers
        transformers.logging.set_verbosity_error()
        
        print("🔧 配置模型参数...")
        
        # 模型配置 - 使用4bit量化节省显存
        model_name = "unsloth/llama-3-8b-bnb-4bit"
        # 或者使用更小的模型："unsloth/llama-3-7b-bnb-4bit"
        
        # 加载模型（这步需要GPU）
        print("📦 加载4-bit量化模型...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = 2048,  # 可以减小以节省内存
            load_in_4bit = True,
            # token = "hf_...",  # 如果需要HuggingFace token
        )
        
        # 准备模型用于训练
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,  # LoRA rank
            lora_alpha = 32,  # LoRA alpha
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_dropout = 0.05,
            bias = "none",
            use_gradient_checkpointing = True,
            random_state = 3407,
            use_rslora = False,
            loftq_config = None,
        )
        
        # 加载数据集
        print("📊 加载训练数据...")
        dataset = load_dataset("json", data_files=data_file, split="train")
        
        if len(dataset) == 0:
            return {"status": "error", "message": "No training data found"}
        
        print(f"  训练样本数: {len(dataset)}")
        
        # 配置训练参数 - 针对16G GPU优化
        training_args = TrainingArguments(
            output_dir = "./llama3-8b-academic-qa",
            per_device_train_batch_size = 2,  # 根据显存调整
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 2,  # 可以增加到3-4个epoch
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            save_strategy = "epoch",
            report_to = None,
        )
        
        # 创建训练器
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = 1024,
            args = training_args,
        )
        
        print("🔄 开始训练循环（这需要一些时间）...")
        print("   注意：训练时间取决于数据量和GPU性能")
        print("   预计时间: 10-30分钟")
        
        # 开始训练！
        trainer.train()
        
        print("✅ 训练完成！")
        
        # 保存模型
        print("💾 保存模型...")
        trainer.save_model()
        
        # 保存训练信息
        model_info = {
            "model_name": model_name,
            "training_samples": len(dataset),
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate,
            "status": "success",
            "gpu_used": torch.cuda.get_device_name(0),
            "training_time": "查看日志获取具体时间",
        }
        
        with open("data/training_info.json", "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        return model_info
        
    except Exception as e:
        error_msg = f"训练错误: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": error_msg}

def mock_fine_tuning(data_file: str = "data/synthetic_qa.jsonl"):
    """模拟微调（备用）"""
    print("⚠️  使用模拟训练（GPU不可用）")
    # ... 保持原有的模拟代码 ...

# 自动选择训练模式
def fine_tune_model(data_file: str = "data/synthetic_qa.jsonl"):
    """自动选择训练模式"""
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 10 * 1024**3:
        return real_fine_tuning(data_file)
    else:
        return mock_fine_tuning(data_file)

if __name__ == "__main__":
    fine_tune_model()