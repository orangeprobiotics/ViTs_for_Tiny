import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt


class FullAttention(nn.Module):
    def __init__(self, scale=None, factor=5, attention_dropout=0.1, is_LSA=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.is_LSA = is_LSA

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        if self.is_LSA:
            device = torch.device('cuda:0')
            scale = nn.Parameter(scale * torch.ones(H)).to(device)
            mask = torch.eye(L, S)
            mask = torch.nonzero((mask == 1), as_tuple=False)
        else:
            mask = None
        if mask is None:
            scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale
        else:
            scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale.unsqueeze(0).unsqueeze(-1).unsqueeze(
                -1).expand((B, H, 1, 1))
            scores[:, :, mask[:, 0], mask[:, 1]] = -987654321

        A = self.dropout(torch.softmax(scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous(), A


class ProbAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1, is_LSA=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.is_LSA = is_LSA

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, E = V.shape
        V_sum = V.mean(dim=-2)
        contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        return contex
    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, E = V.shape
        attn = torch.softmax(scores, dim=-1)
        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        return context_in, attn

    def forward(self, queries, keys, values):
        B, L_Q, H, E = queries.shape
        _, L_K, _, _ = keys.shape
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)  ln(L_K)比较小，需要乘一个常数factor
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        scale = self.scale or 1. / sqrt(E)
        if self.is_LSA:
            device = torch.device('cuda:0')
            scale = nn.Parameter(scale * torch.ones(H)).to(device)
            mask = torch.eye(u, L_K)
            mask = torch.nonzero((mask == 1), as_tuple=False)
        else:
            mask = None
        if mask is None:
            scores_top = scores_top * scale
        else:
            scores_top = scores_top * scale.unsqueeze(0).unsqueeze(-1).unsqueeze(
                -1).expand((B, H, 1, 1))
            scores_top[:, :, mask[:, 0], mask[:, 1]] = -987654321
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q)
        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, qkv_bias=False):
        super(AttentionLayer, self).__init__()
        # 不给d_keys传入值d_keys的值就是d_model // n_heads
        d_keys = d_keys or (d_model // n_heads)  # 如果d_model=512并且采用默认n_heads=8时，d_keys=64
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads, bias=qkv_bias)
        self.key_value_projection = nn.Linear(d_model, d_keys * n_heads * 2, bias=qkv_bias)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, x):
        B, N, C = x.shape
        queries = self.query_projection(x).view(B, N, self.n_heads, C // self.n_heads)
        kv = self.key_value_projection(x).reshape(B, -1, 2, self.n_heads, C // self.n_heads).permute(2, 0, 1, 3, 4)
        keys, values = kv[0], kv[1]

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
        )
        out = out.view(B, N, -1)
        return self.out_projection(out), attn
