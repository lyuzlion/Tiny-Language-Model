import os
import sys
__package__ = "model"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

class TinyLMConfig(PretrainedConfig):
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


class GroupQueryAttention(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_query_heads: int,
                 num_kv_heads: int,
                 dropout_prob: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_query_heads
        assert self.head_dim * num_query_heads == hidden_size, "hidden_size must be divisible by num_query_heads"
        assert hidden_size % num_kv_heads == 0, "hidden_size must be divisible by num_kv_heads"
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self,
                hidden_states: torch.Tensor,
                key_value_states: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        '''
        hidden_states: (batch_size, seq_len, hidden_size)
        key_value_states: (batch_size, kv_seq_len, hidden_size)
        past_key_value: ((batch_size, num_kv_heads, past_kv_seq_len, head_dim), (batch_size, num_kv_heads, past_kv_seq_len, head_dim))
        '''
        bsz, tgt_len, _ = hidden_states.size()
        if key_value_states is None:
            key_value_states = hidden_states
        kv_bsz, kv_len, _ = key_value_states.size() 
        query_states = self.q_proj(hidden_states)  # (bsz, tgt_len, hidden_size)
        key_states = self.k_proj(key_value_states)  # (bsz, kv_len, hidden_size)
        value_states = self.v_proj(key_value_states)  # (bsz, kv_len, hidden_size)
        query_states = query_states.view(bsz, tgt_len, self.num_query_heads, self.head_dim).transpose(1, 2)  # (bsz, num_query_heads, tgt_len, head_dim)
        key_states = key_states.view(kv_bsz, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (bsz, num_kv_heads, kv_len, head_dim)
        value_states = value_states.view(kv_bsz, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (bsz, num_kv_heads, kv_len, head_dim)
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)  # (bsz, num_kv_heads, past_kv_len + kv_len, head_dim)
            value_states = torch.cat([past_value, value_states], dim=2)  # (bsz, num_kv_heads, past_kv_len + kv_len, head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))  # (bsz, num_query_heads, tgt_len, kv_len)
        attn_weights = attn_weights * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_probs = F.softmax(attn_weights, dim=-1)  # (bsz, num_query_heads, tgt_len, kv_len)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, value_states)  # (bsz, num_query_heads, tgt_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.hidden_size)  # (bsz, tgt_len, hidden_size)
        attn_output = self.out_proj(attn_output)  # (bsz, tgt_len, hidden_size)
        outputs = (attn_output, )
        if output_attentions:
            outputs += (attn_probs, )
        if past_key_value is not None:
            outputs += ((key_states, value_states), )
        return outputs  # attn_output, (attn_probs), (past_key_value)
    
class TinyLMLayer(nn.Module):
    def __init__(self, config: TinyLMConfig):
        super().__init__()
        self.attention = GroupQueryAttention(
            hidden_size=config.hidden_size,
            num_query_heads=config.num_attention_heads,
            num_kv_heads=config.num_attention_heads // 2,
            dropout_prob=config.dropout_prob
        )
        self.norm1 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            ACT2FN[config.hidden_act],
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout_prob)
        )
        self.norm2 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        attn_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions
        )
        attn_output = attn_outputs[0]
        hidden_states = hidden_states + attn_output
        hidden_states = self.norm1(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = hidden_states + ffn_output
        hidden_states = self.norm2(hidden_states)
        outputs = (hidden_states, )
        if output_attentions:
            outputs += (attn_outputs[1], )
        if past_key_value is not None:
            outputs += (attn_outputs[2], )
        return outputs  # hidden_states, (attn_probs), (past_key_value)
    
class TinyLanguageModel(nn.Module):
    def __init__(self, config: TinyLMConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList([TinyLMLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.init_weights()

    def init_weights(self):
        init_range = 0.02
        self.embed_tokens.weight.data.uniform_(-init_range, init_range)
        self.position_embeddings.weight.data.uniform_(-init_range, init_range)
        for layer in self.layers:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.uniform_(-init_range, init_range)
                    if module.bias is not None:
                        module.bias.data.zero_()

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
                output_attentions: bool = False,
                output_hidden_states: bool = False) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[List[Tuple[torch.Tensor]]]]:
        bsz, seq_len = input_ids.size()
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        hidden_states = self.embed_tokens(input_ids) + self.position_embeddings(position_ids)
        hidden_states = self.dropout(hidden_states)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if output_hidden_states:
                all_hidden_states += (hidden_states, )
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions += (layer_outputs[1], )
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states, )
        outputs = (hidden_states, )
        if output_hidden_states:
            outputs += (all_hidden_states, )
        if output_attentions:
            outputs += (all_attentions, )
        return outputs  # hidden_states, (all_hidden_states), (all_attentions)
