# -*- coding: utf-8 -*-
import os
import torch
import time
# from vllm import LLM, SamplingParams
from transformers import TextStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.quant import bf16_model, quantize_4bit_model, quantize_8bit_model
from utils.utils import chat_mode, print_model_dtype

if __name__ == "__main__":
    '''
    实验: 对比kv cache效果
    '''
    model_path = "../DeepSeek/DeepSeek-R1-Distill-Qwen-1.5B"

    model = bf16_model(model_path)  # baseline

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    streamer = TextStreamer(tokenizer)

    print(model)
    print_model_dtype(model)
    print(model.device)

    # chat_mode(model, tokenizer, streamer, kv_cache=False)  # 禁用kv cache  |17.67 token/s|
    chat_mode(model, tokenizer, streamer, kv_cache=True)   # 启用kv cache  |21.93 token/s|