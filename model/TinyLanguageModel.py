import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

class TinyLanguageModelConfig(PretrainedConfig):
    def __init__(self,
                 vocab_size: int = 30522,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 hidden_act: str = "gelu",
                 max_position_embeddings: int = 512,
                 layer_norm_eps: float = 1e-5,
                 dropout_prob: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.dropout_prob = dropout_prob


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-8):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size)) # 可学习的缩放参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        例子：x=tensor([[1., 2., 3.],
                       [4., 5., 6.]])
        norm_x = x.norm(2, dim=-1, keepdim=True)
        tensor([[3.7417],
                [8.7749]])
        rms_x = norm_x * (self.hidden_size ** -0.5)
        tensor([[2.1602],
                [5.0710]])
        x_normed = x / (rms_x + self.eps)
        tensor([[0.4629, 0.9258, 1.3887],
                [0.7882, 0.9853, 1.1824]])
        '''
        # RMSNorm公式： x_normed = x / (rms(x) + eps) * weight
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * (self.hidden_size ** -0.5)
        x_normed = x / (rms_x + self.eps)

        return x_normed * self.weight

