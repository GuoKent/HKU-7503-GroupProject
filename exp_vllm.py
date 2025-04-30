from vllm import LLM, SamplingParams
import time

if __name__ == "__main__":
    model_path = "../DeepSeek/DeepSeek-R1-Distill-Qwen-1.5B"
    # 初始化模型和分词器
    model = LLM(
        model=model_path,
        trust_remote_code=False,
        tensor_parallel_size=1,
        # dtype="bfloat16",
        dtype="half"
    )

    # 配置生成参数
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.9,
        max_tokens=256
    )

    # 准备输入
    prompts = [
        "你好，请介绍一下你自己。",
        "What's the capital of France?"
    ]

    # 生成文本
    start_time = time.time()
    outputs = model.generate(prompts, sampling_params)
    end_time = time.time()

    # 输出结果
    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated text: {output.outputs[0].text}\n")
        generated_tokens = len(output.outputs[0].token_ids)  # 生成的 Token 数
        prompt_tokens = len(output.prompt_token_ids)         # 输入提示词的 Token 数
        total_tokens = prompt_tokens + generated_tokens      # 总 Token 数
        token_speed = generated_tokens / (end_time-start_time)
        print(f"token_speed: {token_speed}")