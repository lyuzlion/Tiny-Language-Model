import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import random
import math
import numpy as np
from transformers import AutoTokenizer
from model.model_tinylm import TinyLMForCausalLM
from datetime import datetime

def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6 # 计算模型总参数量，单位为百万
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0)) # 获取路由专家数量
    n_active = getattr(config, 'num_experts_per_tok', 0) # 获取每个token激活的专家数量
    n_shared = getattr(config, 'n_shared_experts', 0) # 获取共享专家数量
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6 # 计算单个专家的参数量
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6 # 计算单个共享专家的参数量
    base = total - (expert * n_routed) - (shared_expert * n_shared) # 计算基础模型参数量
    active = base + (expert * n_active) + (shared_expert * n_shared) # 计算激活参数量
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M') # 输出模型参数量信息
    else: Logger(f'Model Params: {total:.2f}M') # 输出模型参数量信息

def is_main_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

def Logger(content):
    if is_main_process():
        print(content)

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_distributed_mode(): # 初始化分布式训练环境,返回本地GPU编号
    if int(os.environ.get("RANK", -1)) == -1:
        return 0

    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def init_model(config, tokenizer_path='../tokenizer'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = TinyLMForCausalLM(config)
    get_model_params(model, config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model, tokenizer

def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def save_checkpoint(model, epoch, save_dir, method, config):
    os.makedirs(save_dir, exist_ok=True)
    ckp_path = f'{save_dir}/{method}_{config.hidden_size}_epoch{epoch}.pth'
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    torch.save(raw_model.state_dict(), ckp_path)
    Logger(f'Checkpoint saved to {ckp_path}')
    return ckp_path


def save_model(model, tokenizer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    raw_model.config.auto_map = {
        "AutoConfig": "model_tinylm.TinyLMConfig",
        "AutoModelForCausalLM": "model_tinylm.TinyLMForCausalLM"
    }
    raw_model.tie_weights()
    raw_model.save_pretrained(
        save_dir, 
        safe_serialization=True, 
        is_main_process=True
    )
    tokenizer.save_pretrained(save_dir)
    raw_model.config.save_pretrained(save_dir) 
    print(f"Model successfully saved to {save_dir}")
