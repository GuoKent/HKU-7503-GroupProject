from datasets import load_dataset

from transformers import AutoTokenizer

model_path = "../DeepSeek/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

def format_data(example):
    sys_prompt = "<|im_start|>system\nYou are DeepSeek-R1, a helpful assistant.<|im_end|>\n"
    text = [
        sys_prompt + "<|im_start|>user\n" + example["conversations"][0]["content"] + 
        "<|im_end|>\n<|im_start|>assistant\n" + 
        example["conversations"][1]["content"] + "<|im_end|>"
    ]
    text_process = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    print(text_process.keys())
    return text_process

def preprocess(example):
    MAX_LENGTH = 2048
    input_ids, attention_mask, labels = [], [], []

    # for message in example["messages"]:
    # 使用ChatML格式
    tokenized = tokenizer.apply_chat_template(
        # message,
        example["messages"],
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_dict=True,
        truncation=True,  # 启用截断
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    for key in tokenized:
        tokenized[key] = tokenized[key].squeeze()
    # 截断处理
    if len(tokenized["input_ids"]) > MAX_LENGTH:
        tokenized["input_ids"] = tokenized["input_ids"][-MAX_LENGTH:]
        tokenized["attention_mask"] = tokenized["attention_mask"][-MAX_LENGTH:]
    
    # 因果语言建模时 labels 与 input_ids 相同
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

def get_dataset(path):
    dataset = load_dataset("json", data_files={"train": path})
    formatted_dataset = dataset.map(
        preprocess,
        batched=True,
        # remove_columns=dataset.column_names,
    )
    return formatted_dataset

if __name__ == "__main__":
    path = "./dataset/train.json"
    dataset = get_dataset(path)
    print(dataset)