import json
from tqdm import tqdm
from datasets import load_dataset


def transform_data_to_json(path):
    ds = load_dataset(path)
    txt = ds['train']['text']
    res = []
    for text in tqdm(txt, total=len(txt)):
        parts = text.split('<HUMAN>: ')[1].split('<ASSISTANT>: ')
        # 第二步：提取并清理内容
        human_content = parts[0].strip("\n")
        assistant_content = parts[1]
        res.append(
            {
                "conversations": [
                    {"role": "user", "content": human_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            }
        )
    
    json_file_path = './dataset/train.json'
    json_file = open(json_file_path, mode='w')
    json.dump(res, json_file, indent=4)

def convert_format(original_data):
    converted_data = []
    for dialog in original_data:
        # 添加系统提示（可选）
        messages = [{"role": "system", "content": "You are a helpful psychologist, so please give advice based on the questions your patient asks."}]
        
        for conv in dialog["conversations"]:
            messages.append({
                "role": conv["role"],
                "content": conv["content"]
            })
        converted_data.append({"messages": messages})
    return converted_data

if __name__ == "__main__":
    # transform_data_to_json("heliosbrahma/mental_health_chatbot_dataset")
    # ds = load_dataset("heliosbrahma/mental_health_chatbot_dataset")
    # txt = ds['train']['text']
    # print(len(txt))

    # 加载原始数据
    with open("dataset/train.json", "r") as f:
        original_data = json.load(f)

    # 转换格式
    converted_data = convert_format(original_data)

    # 保存转换后的数据
    with open("dataset/qwen_data.jsonl", "w") as f:
        for item in converted_data:
            f.write(json.dumps(item) + "\n")