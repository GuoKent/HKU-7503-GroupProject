import os
import torch

from transformers import AutoModelForCausalLM, TrainingArguments
from transformers import Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from dataset.dataloader import get_dataset


if __name__ == "__main__":
    model_path = "../DeepSeek/DeepSeek-R1-Distill-Qwen-1.5B"
    data_path = "./dataset/qwen_data.jsonl"
    print("当前工作目录:", os.getcwd())

    train_dataset = get_dataset(data_path)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="float32",
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
        # add_pooling_layer=False,
        # load_in_4bit=True,  # |qlora| 4bit量化加载
        # bnb_4bit_compute_dtype=torch.float16,  # |qlora|
    )

    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=1,
        remove_unused_columns=True,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        optim="adamw_torch",
        num_train_epochs=3,
        logging_steps=50,
        fp16=True  # 启用混合精度训练
    )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 查看可训练参数比例

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
    )

    trainer.train()