from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

# attention block
# transformer.h.0.ln_1.weight torch.Size([768])
# transformer.h.0.ln_1.bias torch.Size([768])
# transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
# transformer.h.0.attn.c_attn.bias torch.Size([2304])
# transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
# transformer.h.0.attn.c_proj.bias torch.Size([768])
# transformer.h.0.ln_2.weight torch.Size([768])
# transformer.h.0.ln_2.bias torch.Size([768])
# transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
# transformer.h.0.mlp.c_fc.bias torch.Size([3072])
# transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
# transformer.h.0.mlp.c_proj.bias torch.Size([768])


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU()
        self.c_porj = nn.Linear(4*config.n_embd, config.n_embd)
    def forward(self, x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_porj(x)
        return x

# 这部分也可以看transformers的实现,写的也很不错
class CrossSelfAttention(nn.Module):
    # transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
    # transformer.h.0.attn.c_attn.bias torch.Size([2304])
    # transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
    # transformer.h.0.attn.c_proj.bias torch.Size([768])
    def __init__(self, config):
        super().__init__()
        # 防止embedd不能整除head导致维度出错
        assert config.n_embd % config.n_head == 0
        # x->(q,k,v)
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        # output
        self.c_proj=nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        # 它的核心作用是创建一个 “看不见未来” 的挡板，专业上称为因果掩码（Causal Mask）
        self.register_buffer('bias',torch.zeros(config.n_head,config.n_embd))
    def forward(self,x):
        # B, T, C 分别代表了 “批量大小”、“序列长度” 和 “特征维度”
        B, T, C = x.size()
        qkv=self.c_attn(x)
        # 用一个大w计算,速度更快,然后再分割
        # B, T不变,liner做的仅在最后一维操作,因此在最后一维分割即可
        q,k,v = qkv.split(3, dim=-1)
        # multi head
        q=q.view(B,T,self.n_head,self.n_embd//self.n_head)
        k=k.view(B,T,self.n_head,self.n_embd//self.n_head)
        v=v.view(B,T,self.n_head,self.n_embd//self.n_head)

        # att=q@kT /sqrt(dim_k)
        att=torch.matmul(q,k.transpose(-2,-1))/(self.n_embd//self.n_head)
        #  att.masked_fill -inf,使得softmax时值为0
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att_score = F.softmax(att, dim=-1)

        y=torch.matmul(att_score,v)
        # re-assemble all head outputs side by
        y=y.transpose(1,2).contiguous().view(B,T,C)
        y=self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn=CrossSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)
    def forward(self,x):
        # gradient可以随着res路径向下
        x=self.attn(self.ln_1(x))+x
        x=self.mlp(self.ln_2(x))+x
        return x

class GPT(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        # 模仿transformers中的结构,用nn.ModuleDict构建模型块,forward时再调用即可
        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.n_embd, config.n_embd),
            # ModuleList可以按数字访问,和play.ipynb中保持一致
            h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            # gpt2论文指出最后的输出前要加layernorm
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        # 输出头,直接用Liner
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size,bias=False)