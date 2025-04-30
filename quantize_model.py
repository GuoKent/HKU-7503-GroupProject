# -*- coding: utf-8 -*-
import os
import torch
import time
# from vllm import LLM, SamplingParams
from transformers import TextStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.quant import bf16_model, quantize_4bit_model, quantize_8bit_model
from utils.utils import chat_mode, print_model_dtype

MODELS = (
    "DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-14B"
)


if __name__ == "__main__":
    '''
    实验: 量化参数的推理速度
    '''
    model_path = "../DeepSeek/DeepSeek-R1-Distill-Qwen-1.5B"

    model = bf16_model(model_path)  # baseline  |21 tokens/s|
    # model = quantize_4bit_model(model_path)  # 4bit量化  |31 tokens/s|
    # model = quantize_8bit_model(model_path)  # 8bit量化  |13 token/s|

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    streamer = TextStreamer(tokenizer)

    print(model)
    print_model_dtype(model)
    print(model.device)

    chat_mode(model, tokenizer, streamer)