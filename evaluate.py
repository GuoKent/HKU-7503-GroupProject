from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import time
import json
from tqdm import tqdm

from utils.eval import get_metrics

def generate_response(model, tokenizer, text, max_new_tokens=512):
    # 构建 ChatML 格式
    messages = [
        {"role": "system", "content": "You are a helpful psychologist, so please give advice based on the questions your patient asks."},
        {"role": "user", "content": text}
    ]
    start_time = time.time()
    
    # 应用模板并生成
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_dict=True,
        truncation=True,  # 启用截断
        max_length=max_new_tokens,
        return_tensors="pt",
    ).to(model.device)
    
    # 生成参数配置
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1,         # 禁用束搜索
            temperature=0.7,           # 控制随机性（0~1，越大越随机）
            top_p=0.9,                 # 核采样阈值（0~1，通常与temperature配合）
            repetition_penalty=1.1,    # 重复惩罚（>1时抑制重复）
            do_sample=False,             # 启用采样模式
            use_cache=True,       # 启用KV缓存
            # streamer = streamer,  # 流式输出
        )
    end_time = time.time()
    
    token_num = len(outputs[0])
    token_speed = token_num/(end_time-start_time)
    # 解码时跳过特殊token
    response = tokenizer.decode(
        outputs[0][len(inputs[0]):], 
        skip_special_tokens=True
    )
    return response, token_speed

def get_trained_model(base_model_path, lora_weights):
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        # add_pooling_layer=False,
        # load_in_4bit=True,    # 4bit 量化
        # bnb_4bit_compute_dtype=torch.float16
    )

    if lora_weights:
    # 加载 LoRA 适配器
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16
        )
        model = model.merge_and_unload()  # 合并权重到基础模型（可选，提升推理速度）

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    # 设置为评估模式
    model.eval()

    return model, tokenizer

def evaluate(model, tokenizer, data_path, eval_len=50):
    dataset = load_dataset("json", data_files={"train": data_path})
    dataset = dataset["train"]["messages"][:eval_len]
    total_num = len(dataset)
    total_P, total_R, total_F1 = 0, 0, 0
    for chat in tqdm(dataset):
        question, answer = chat[1]["content"], chat[2]["content"]
        model_answer, token_speed = generate_response(model, tokenizer, text=question)
        # print(question, answer)
        scores = get_metrics([answer], [model_answer])
        print(scores, token_speed)
        total_P += scores["Precision"]
        total_R += scores["Recall"]
        total_F1 += scores["F1_Score"]
    return {
        "avg_p": total_P/total_num,
        "avg_r": total_R/total_num,
        "avg_f1": total_F1/total_num,
        "total_num": total_num,
    }

if __name__ == "__main__":
    # 基础模型路径
    base_model_path = "../DeepSeek/DeepSeek-R1-Distill-Qwen-1.5B"
    # 训练保存的适配器路径
    lora_weights = "./output/checkpoint-63/"
    data_path = "./dataset/qwen_data.jsonl"

    model, tokenizer = get_trained_model(base_model_path, lora_weights=None)
    # metrics = evaluate(model, tokenizer, data_path)
    # print(metrics)

    # res, metrics = generate_response(model, tokenizer, text="Are patients with schizophrenia violent?", max_new_tokens=1024)
    # print(res)


    ref = "Due to incorrect representation of the disease in media & books, there is a myth that schizophrenics are violent. The truth is most schizophrenics are docile and keep to themselves. The schizophrenics who have demonstrated bursts of violence are either in an acute stage of psychosis or are abusing an addictive substance."
    baseline = '''The likelihood of patients with schizophrenia exhibiting violent tendencies is a topic of interest, but the evidence is mixed. Here's a structured summary:

1. **Psychological Factors**:
   - Schizophrenia involves altered thought patterns and behaviors, potentially leading to delusions of the future or past. These delusions can cause confusion and stress, prompting individuals to act violently.      
   - Individuals may engage in self-harm behaviors, such as substance abuse, which can escalate violent actions.

2. **Behavioral Signs**:
   - Social difficulties due to mental health issues can lead to violent means to maintain relationships, especially under threat.
   - Stress or danger can trigger violent reactions, though this varies among individuals.

3. **Evidence and Considerations**:
   - Some studies suggest violent tendencies in schizophrenia patients, particularly in response to stress or danger.
   - However, no direct evidence links schizophrenia to violent behavior exists, making conclusions uncertain.

4. **Conclusion**:
   - While it is plausible that schizophrenia patients may be more likely to be violent, the evidence is inconclusive. Without specific case studies, this cannot be definitively stated.

Thus, while schizophrenia can contribute to violent tendencies, the extent and nature of these tendencies remain subjects of ongoing research.
'''
    lora = "The relationship between schizophrenia and violent behavior is complex and not definitively established. schizophrenia, a severe form of depression, can lead to various mental disturbances and behaviors, including violent ones. However, there is no conclusive evidence linking schizophrenia directly to higher rates of violence. Studies often have small sample sizes and observational designs, making the results less definitive. While some research suggests a correlation, it may not be a direct cause. Further studies, including observational studies, would be needed to determine the extent of this link. Additionally, other factors such as socioeconomic status, access to mental health services, and personal circumstances can influence the likelihood of violent behavior in individuals with schizophrenia. Genetic factors are also a consideration, though the link to violence is not well-established. It is recommended to consult specific research studies for more detailed insights."
    baseline_scores = get_metrics([baseline], [ref])
    lora_scores = get_metrics([lora], [ref])
    print(f"baseline_scores: {baseline_scores}")
    print(f"lora_scores: {lora_scores}")