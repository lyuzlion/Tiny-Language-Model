import torch.distributed as dist
import torch
import random
import numpy as np
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
    return not dist.is_initialized() or dist.get_rank() == 0

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