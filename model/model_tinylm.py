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
                 bos_token_id: int = 1,
                 eos_token_id: int = 2,
                 vocab_size: int = 6400,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 num_kv_heads: int = 4,
                 intermediate_size: int = 3072,
                 hidden_act: str = "gelu",
                 max_position_embeddings: int = 512,
                 rms_norm_eps: float = 1e-5,
                 dropout_prob: float = 0.0,
                 rope_theta: int = 1000000,
                 inference_rope_scaling: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.dropout_prob = dropout_prob
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None


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

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the magic sauce for GQA. 
    It repeats the KV heads n_rep times so they match the number of Query heads.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class GroupQueryAttention(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_query_heads: int,
                 num_kv_heads: int,
                 dropout_prob: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_query_heads = num_query_heads
        self.head_dim = hidden_size // num_query_heads
        self.num_kv_heads = num_kv_heads
        assert self.head_dim * num_query_heads == hidden_size, "hidden_size must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = dropout_prob
    def forward(self,
                hidden_states: torch.Tensor,
                position_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, seq_len, self.num_query_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_query_heads // self.num_kv_heads)
        value_states = repeat_kv(value_states, self.num_query_heads // self.num_kv_heads)

        output = F.scaled_dot_product_attention(query_states, 
                                                key_states, 
                                                value_states, 
                                                dropout_p=self.dropout if self.training else 0.0, 
                                                is_causal=True) # 这里加入了causal mask
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        output = self.out_proj(output)
        return output


class SwiGLU(nn.Module):
    def __init__(self, config: TinyLMConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class TinyLMLayer(nn.Module):
    def __init__(self, config: TinyLMConfig):
        super().__init__()
        self.attention = GroupQueryAttention(
            hidden_size=config.hidden_size,
            num_query_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            dropout_prob=config.dropout_prob
        )
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = SwiGLU(config)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,
                hidden_states: torch.Tensor,
                position_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        # RMSNorm -> Attention -> Add -> RMSNorm -> FFN -> Add
        normed_states = self.norm1(hidden_states)
        attn_output = self.attention(
            normed_states,
            position_embeddings=position_embeddings
        )
        hidden_states = hidden_states + attn_output
        normed_states = self.norm2(hidden_states)
        ffn_output = self.ffn(normed_states)
        hidden_states = hidden_states + ffn_output
        return hidden_states
    



def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


class TinyLM(nn.Module):
    def __init__(self, config: TinyLMConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TinyLMLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, 
                                                    rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.size()
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)
        position_embeddings = (self.freqs_cos[:seq_len, :], self.freqs_sin[:seq_len, :])
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class TinyLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = TinyLMConfig

    def __init__(self, config: TinyLMConfig):
        super().__init__(config)
        self.model = TinyLM(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, 
                input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Union[Tuple, CausalLMOutputWithPast]:
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_states,
        )