import copy
from typing import Optional, Any, Union, Callable
from .moe import MoE
import torch
import warnings
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch import nn

__all__ = ['Transformer', 'TransformerEncoder', 'TransformerDecoder', 'TransformerEncoderLayer', 'TransformerDecoderLayer']

def _generate_square_subsequent_mask(
        sz: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
) -> Tensor:
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )

def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            return src_size[0]
        else:
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal

class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_embed, n_embed)
        )

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 bias=True, device=None, dtype=None, num_experts=8):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               bias=bias, batch_first=batch_first,
                                               **factory_kwargs)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.moe_layer = MoE(dim=d_model, num_experts=num_experts, balance_loss_coef=1e-2,
                             gating_top_n=2, capacity_factor_train=2,
                             experts=nn.ModuleList([FeedForward(d_model) for _ in range(num_experts)]))

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            x, sa_aux_loss = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x, ff_aux_loss = self._ff_block(self.norm2(x))
        else:
            x, sa_aux_loss = self._sa_block(x, src_mask, src_key_padding_mask)
            x, ff_aux_loss = self._ff_block(x)
            x = self.norm1(x)
            x = self.norm2(x)
        return x, sa_aux_loss + ff_aux_loss

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x), torch.tensor(0.0, device=x.device)

    def _ff_block(self, x):
        x, aux_loss,_,_ = self.moe_layer(x)
        return self.dropout2(x), aux_loss


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 bias=True, device=None, dtype=None, num_experts=8):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               bias=bias, **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                    bias=bias, **factory_kwargs)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.moe_layer = MoE(dim=d_model, num_experts=num_experts, balance_loss_coef=1e-2,
                             gating_top_n=2, capacity_factor_train=2,
                             experts=nn.ModuleList([FeedForward(d_model) for _ in range(num_experts)]))

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = tgt
        if self.norm_first:
            x, sa_aux_loss = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x, mha_aux_loss = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x, ff_aux_loss = self._ff_block(self.norm3(x))
        else:
            x, sa_aux_loss = self._sa_block(x, tgt_mask, tgt_key_padding_mask)
            x, mha_aux_loss = self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
            x, ff_aux_loss = self._ff_block(x)
            x = self.norm1(x)
            x = self.norm2(x)
            x = self.norm3(x)
        return x, sa_aux_loss + mha_aux_loss + ff_aux_loss

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x), torch.tensor(0.0, device=x.device)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout2(x), torch.tensor(0.0, device=x.device)

    def _ff_block(self, x):
        x, aux_loss,_,_ = self.moe_layer(x)
        return self.dropout3(x), aux_loss


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        total_aux_loss = torch.tensor(0.0, device=src.device)
        for mod in self.layers:
            output, aux_loss = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            total_aux_loss += aux_loss
        if self.norm is not None:
            output = self.norm(output)
        return output, total_aux_loss


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        total_aux_loss = torch.tensor(0.0, device=tgt.device)
        for mod in self.layers:
            output, aux_loss = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask)
            total_aux_loss += aux_loss
        if self.norm is not None:
            output = self.norm(output)
        return output, total_aux_loss


