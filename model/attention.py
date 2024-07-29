import torch
import math, copy
import torch.nn.functional as F
import torch.nn as nn

from utils import clones


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # 对于为True的部分填充为-1e9，不太清楚为什么不可以为0
    p_attn = F.softmax(scores, dim=-1) # dim=-1表示softmax将按照最后一个维度为轴进行操作，比如二维，
    # 那么沿着列进行操作，就是对每一行进行softmax。理解上最终结果是对query的每一行与key的相关性做归一化处理
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 保证可以整除
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # 4个线性层分别用于query、key、value和多头注意力的线性变换
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 这里理解是直接将8个head的权重合并为一个nn.Linear()层，从这里理解默认原始输入的维度为d_model
        # 这里应该是使用了广播的机制
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  # tensor.view()方法用于改变tensor形状
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)  # 分别为QKV和QK的结果

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)  # 调用
        return self.linears[-1](x)  # 返回
