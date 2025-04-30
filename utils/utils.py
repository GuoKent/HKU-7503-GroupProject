import torch
import time

def print_model_dtype(model):
    for name, param in model.named_parameters():
        if "q_proj" in name:
            print(f"Layer: {name}, dtype: {param.dtype}")

def chat_mode(model, tokenizer, streamer, sys_prompt=None, kv_cache=True, do_sample=False):
    sys_prompt = "You are DeepSeek-R1, a helpful assistant." if not sys_prompt else sys_prompt
    messages = [
        {"role": "system", "content": sys_prompt},
    ]

    while True:
        prompt = input(">>>")
        if prompt == '/exit':
            break
        elif prompt == '/clean':
            messages = []
            continue
        end = time.time()
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        # model_inputs = tokenizer([text], return_tensors="pt")

        with torch.amp.autocast("cuda", enabled=True):
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=2048,
                num_beams=1,         # 禁用束搜索
                do_sample=do_sample,     # 禁用采样
                use_cache=kv_cache,      # 启用KV缓存
                pad_token_id=tokenizer.pad_token_id,  # 配置padding_id
                eos_token_id=tokenizer.eos_token_id,  # 配置eos_id
                streamer = streamer,  # 流式输出
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]  # 模型生成输出包括 问题+输出，需要把问题从最终结果中去除
        token_num = len(generated_ids[0])
        cost_time = time.time() - end
        token_speed = token_num / cost_time

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        messages.append({"role": "assistant", "content": response})
        print(response)
        print(f"Cost time: {cost_time:.2f}s | Token nums: {token_num} | Token speed: {token_speed:.2f} token/s")