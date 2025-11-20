import math
import time
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
        x = self.c_proj(x)
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
        # set signal wo scale
        self.c_proj.NANOGPT_SCALE_INIT=1
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
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.n_embd // self.n_head)
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

        # 根据gpt2论文,wte和lm_head的权重是绑定的,
        # 因此这里将二者直接使用指针引用,从而实现参数共享
        # nn.Linear与nn.embedding 在权重上是转置关系,
        # 搜一下相关资料就知道为什么了(token 理解为one-hot,结果计算embedding矩阵的一行,因此和liner运算是转置关系
        self.transformer.wte.weight = self.lm_head.weight



    def _init_weights(self,module):
    # 根据代码提示,liner和embed用的标准差0.02的正态分布
    # wpe用的0.01正态分布初始化
        if isinstance(module, nn.Linear):
            std=0.02
            # 残差会积累标准差,因此要在残差处进行缩放,具体看gpt2论文
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std*=(2*self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, target=None):
        # b*t个token,经过了tokenizer
        B, T = idx.size()
        # 超出最大序列长度则报错
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # 位置编码,为每个位置带上自己的位置索引,然后进行pos_embed
        # device=idx.device 避免数据在不同的device上
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        # emb+pos
        pos_emb = self.transformer.wpe(pos)  # (T,n_embed)
        emb = self.transformer.wte(idx)  # (B,T,n_embed)
        x = emb + pos_emb  # (B,T,n_embed)

        for block in self.transformer.h:
            x = block(x)
        # gpt2论文中要求输出前加一个LayerNorm
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        loss = None
        if target is not None:
            # celoss不接受多维输入,因此需要将多维的输入拉平
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        # n_layer, n_head and n_embd are determined from model_type
        # 根据你传入的model_type变量，从字典中取出对应的参数
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        # 为模型超参数创造状态字典
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # 将除了bias的keys读取进来(bias是attention的掩码,之前讲过)
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        # endswith()方法用于判断字符串是否以指定的后缀结束,这里是前面提到的mask部分,由于不参与模型训练,因此忽略
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
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
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"{sd_hf[k].shape[::-1]} != {sd[k].shape},{k}"
                with torch.no_grad():
                    # # 3. 将源权重转置后，复制到目标模型的参数中
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape, f"{sd_hf[k].shape} != {sd[k].shape},{k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model


class DataLoaderLite:
    def __init__(self, B, T):
        self.B, self.T = B, T
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
            # print(text[:1000])
        encoding = tiktoken.get_encoding('gpt2')
        self.tokens = encoding.encode(text)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        tokens = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = torch.tensor(tokens[:-1]).view(B, T)
        y = torch.tensor(tokens[1:]).view(B, T)
        self.current_position += B * T
        # 超界限则reset
        if self.current_position + B * T >= len(self.tokens):
            self.current_position = 0
        # 按batch跳token
        return x, y


# Use a more general method
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data loading
import tiktoken

enc = tiktoken.get_encoding("gpt2")
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    # print(text[:1000])
encoding = tiktoken.get_encoding('gpt2')
token = encoding.encode(text[:1000])
B, T = 2, 1024  # using for debug
# 初始化,用buf不断滑动窗口得到输入和输出来求loss


num_return_sequences = 5
max_length = 30
# model = GPT.from_pretrained('gpt2')
# torch已经帮我们做好了随机初始化
model = GPT(GPTConfig())
print("ok")
# 设置float32的运算精度
'''
“highest”:按完整fp32计算
"high":如果存在可用的快速矩阵算法,则按tf32 or 将每个 f32视为两个bf16进行运算 否则和highest模式一致
"medium":：若存在内部使用bf16的快速矩阵乘法算法,则使用bfloat16,不存在则按high进行计算
剩余部分建议看文档
'''
# 而且即使运算速度最高提升了8倍,实际运算吞吐量从6000t/s ->8000t/s 主要原因还是受内存速率影响
torch.set_float32_matmul_precision('high')
model.to(device)
# model.eval()

# 随机初始化的loss=10.9650
p_token = 1 / 50257
# 交叉熵计算:-ln P
print(-math.log(p_token))
# 和随机预测的loss很接近

# AdamW将优化过程中使用的针对网络权重的衰减项（或者叫正则项）从loss中单独拿了出来，
# 不参与Adam中一二阶动量的计算

# optim
print(device)
optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler = torch.GradScaler()
# 循环训练
data = DataLoaderLite(B, T)
for i in range(50):
    t0=time.time()
    optim.zero_grad()
    x, y = data.next_batch()
    # 仅在读取时调用至GPU,减少显存消耗
    x, y = x.to(device), y.to(device)
    # using amp to accelerate training
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        logits, loss = model(x, y)
    print(f"step{i}", 'loss=', loss.item())
    # loss.backward()
    scaler.scale(loss).backward()
    # optim.step()
    scaler.step(optim)
    scaler.update()

    # 阻塞cpu流,用于预估一次训练的耗时
    # 因为运行计算过后,哪怕gpu未完成计算,cpu也会继续运行后续代码
    # 因此使用torch.cuda.synchronize()阻塞cpu流
    torch.cuda.synchronize()
    print(f'step{i} use time: {(time.time()-t0)}s')
    print(f'{data.B * data.T / (time.time() - t0):.1f}token/s')

# 下面是之前生成的代码,这里计算loss时忽略
import sys;

sys.exit(0)

# tokenizer encode
import tiktoken

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
# 得到5个相同起始的输入
tokens = tokens.unsqueeze(0).repeat(num_return_sequences)
x = tokens.to(device)

# 生成
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)  # (b,t,vocab_size)

        # 取出最后一个字的词表概率用于计算
        logits = logits[:, -1, :]  # (b,vocab_size)
        # 在最后一维求softmax
        probs = F.softmax(logits, dim=-1)  # (b,vocab_size)

        # topk,用于temp高时实现非贪心搜索
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)  # (b,50)

        # torch.multinomial根据topk probs的概率，在50个选项里随机抽1个，概率大的词更容易被抽到；
        # idx是每行选中的位置（0~49之间），形状为(5, 1)。
        ix = torch.multinomial(topk_probs, num_samples=1)

        # 取出真正的token ID
        xcol = torch.gather(topk_indices, dim=-1, index=ix)  # (b,1)
        # 将新的词拼接在原来的输入x后面
        x = torch.cat((x, xcol), dim=1)
# 输出解码结果
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
