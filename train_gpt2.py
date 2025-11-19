from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


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
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_porj(x)
        return x


# 这部分也可以看transformers的实现,写的也很不错
class CrossSelfAttention(nn.Module):
    # transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
    # transformer.h.0.attn.c_attn.bias torch.Size([2304])
    # transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
    # transformer.h.0.attn.c_proj.bias torch.Size([768])
    def __init__(self, config: GPTConfig):
        super().__init__()
        # 防止embedd不能整除head导致维度出错
        assert config.n_embd % config.n_head == 0
        # x->(q,k,v)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        # 它的核心作用是创建一个 “看不见未来” 的挡板，专业上称为因果掩码（Causal Mask）
        # torch.tril:下三角,会把矩阵中上三角部分的元素全部变成 0，只保留对角线和下三角部分的 1
        # .view把它从 (block_size, block_size) 变成 (1, 1, block_size, block_size),为了和模型的输入维度对齐,方便后续通过广播（broadcasting）适配任意批量大小和头数
        # register_buffer是PyTorch的一个方法，意思是：把这个矩阵（掩码）注册为模型的一个 “缓冲区”（不是模型参数，不会被训练更新）。
        # 给它起个名字叫"bias"（虽然它本质是掩码，但沿用了OpenAI / HuggingFace的命名习惯，方便兼容）。
        # 作用：让这个掩码成为模型的一部分，每次调用模型时都能直接使用，不用重新创建。
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (n_embd)
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        B, T, C = x.size()
        qkv = self.c_attn(x)
        # 用一个大w计算,速度更快,然后再分割
        # B, T不变,liner做的仅在最后一维操作,因此在最后一维分割即可
        q, k, v = qkv.split(self.n_embd, dim=-1)
        # multi head
        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)  # (B,nh,T,ns)
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)  # (B,nh,T,ns)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)  # (B,nh,T,ns)

        # att=q@kT /sqrt(dim_k)
        # 对于更高维度的张量，torch.matmul 执行的是批量矩阵乘法
        att = torch.matmul(q, k.transpose(-2, -1)) / (self.n_embd // self.n_head)
        #  att.masked_fill -inf,使得softmax时值为0
        # self.bias[:, :, :T, :T] 取出和当前序列长度 T 匹配的掩码部分,
        # masked_fill(条件, 值)：把掩码中为 0 的位置，对应的注意力分数改成 -inf（负无穷）
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att_score = F.softmax(att, dim=-1)

        y = torch.matmul(att_score, v)  # (B,nh,T,T) * (B,nh,T,hs) -> (B,nh,T,hs)
        # re-assemble all head outputs side by
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CrossSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # gradient可以随着res路径向下
        x = self.attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # 模仿transformers中的结构,用nn.ModuleDict构建模型块,forward时再调用即可
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            # ModuleList可以按数字访问,和play.ipynb中保持一致
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # gpt2论文指出最后的输出前要加layernorm
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        # 输出头,直接用Liner
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    @classmethod
    def from_pretrained(cls,model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        # n_layer, n_head and n_embd are determined from model_type
        # 根据你传入的model_type变量，从字典中取出对应的参数
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        # 为模型超参数创造状态字典
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        #将除了bias的keys读取进来(bias是attention的掩码,之前讲过)
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init huggingface/transformers model
        model_hf=GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        #copy while ensuring all the parameters are aligned and match in names and shapes
        sd_keys_hf=sd_hf.keys()
        # endswith()方法用于判断字符串是否以指定的后缀结束,这里是前面提到的mask部分,由于不参与模型训练,因此忽略
        sd_keys_hf=[k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]# ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)

        # OpenAI的checkpoint使用的是 “Conv1D” 模块，但我们只想使用普通的
        # Linear这就意味着，在导入这些权重时，我们必须对它们进行转置操作
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            # 需要转置处理的特殊weight
            if any(k.endswith(w) for w in transposed):
                # 确保转置后的源权重形状与目标权重形状匹配
                print(k, sd_hf[k].shape)
                print(k, sd[k].shape)
                assert sd_hf[k].shape[::-1] == sd[k].shape,f"{sd_hf[k].shape[::-1]} != {sd[k].shape},{k}"
                with torch.no_grad():
                    # # 3. 将源权重转置后，复制到目标模型的参数中
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape, f"{sd_hf[k].shape} != {sd[k].shape},{k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model



model=GPT.from_pretrained('gpt2')
print("ok")