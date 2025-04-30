import torch
import torch.nn as nn

import os
import time
# from vllm import LLM, SamplingParams
from transformers import TextStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

def quantize_4bit_model(model_path="../DeepSeek/DeepSeek-R1-Distill-Qwen-14B"):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    
    return model

def quantize_8bit_model(model_path):
    # quantization_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     llm_int8_threshold=6.0,  # 控制离群值的阈值
    #     llm_int8_skip_modules=None,   # 确保不跳过任何模块
    #     llm_int8_enable_fp32_cpu_offload=True,  # 禁用混合精度卸载,
    #     # llm_int8_skip_modules=["lm_head"],  # 跳过输出层（可选）
    #     llm_int8_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 强制量化注意力层
    # )
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,  # 禁用混合精度卸载,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    return model

def bf16_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="float32",
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    return model

def quantize_dynamic(model: nn.Module, dtype = None, layers: nn.Module = None):
    '''
    训练后动态量化
    '''
    layers = set(layers) if layers else {torch.nn.Linear}
    dtype = dtype if dtype else torch.qint8
    quantized_model = torch.quantization.quantize_dynamic(
        model,       # 量化的模型
        layers,      # 指定要量化的层类型
        dtype=dtype  # 量化数据类型
    )
    return quantized_model


def quantize_static(model: nn.Module, ):
    '''
    训练后静态量化
    '''
    # x86: fbgemm    arm: qnnpack
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # 准备校准
    model = torch.quantization.prepare(model)
    # 转换为量化模型
    quantized_model = torch.quantization.convert(model)
    return quantized_model