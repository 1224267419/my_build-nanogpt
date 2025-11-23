#TODO fix device and device_type

import math
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
import numpy as np


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
        self.c_proj.NANOGPT_SCALE_INIT = 1
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

        # 得到qkv矩阵
        qkv = self.c_attn(x)
        # 用一个大w计算,速度更快,然后再分割
        # B, T不变,liner做的仅在最后一维操作,因此在最后一维分割即可
        q, k, v = qkv.split(self.n_embd, dim=-1)
        # multi head
        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)  # (B,nh,T,ns)
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)  # (B,nh,T,ns)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)  # (B,nh,T,ns)

        # #att=q@kT /sqrt(dim_k)
        # #对于更高维度的张量，torch.matmul 执行的是批量矩阵乘法
        # att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.n_embd // self.n_head)
        # #  att.masked_fill -inf,使得softmax时值为0
        # # self.bias[:, :, :T, :T] 取出和当前序列长度 T 匹配的掩码部分,
        # # masked_fill(条件, 值)：把掩码中为 0 的位置，对应的注意力分数改成 -inf（负无穷）
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # y = F.softmax(att, dim=-1)
        # y = torch.matmul(y, v)  # (B,nh,T,T) * (B,nh,T,hs) -> (B,nh,T,hs)

        # FlashAttention,一行就可以调用完成
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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

    def _init_weights(self, module):
        # 根据代码提示,liner和embed用的标准差0.02的正态分布
        # wpe用的0.01正态分布初始化
        if isinstance(module, nn.Linear):
            std = 0.02
            # 残差会积累标准差,因此要在残差处进行缩放,具体看gpt2论文
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
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

    def configure_optimizer(self, weight_decay, lr, device):
        # start with all the candidate parameters (that require grad)
        param_dict = {n: p for n, p in self.named_parameters()}  # 取出所有参数
        param_dict = {n: p for n, p in param_dict.items() if p.requires_grad}  # 要求梯度的参数
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]  # only liner and embedding decay
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            # 一个list嵌套dict,从而实现对多组参数实现不同的优化方式
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # numel = number of elements 的缩写,元素个数
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        # fused version could run faster than the non-fused, so we'll default to it.
        # use one instead of lots of kernel to optimize weights,so could get rid of a lot of overhead
        # following is to check if the fused AdamW is available or not
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")

        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


class DataLoaderLite0:
    def __init__(self, B, T, process_rank, num_processes):
        self.B, self.T = B, T
        self.process_rank, self.num_processes = process_rank, num_processes
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
            # print(text[:1000])
        encoding = tiktoken.get_encoding('gpt2')
        self.tokens = encoding.encode(text)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        if master_process:
            print(f"loaded {len(self.tokens)} tokens")
        # 分布式数据加载,确保每个进程只加载自己需要的数据且数据不重合
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        tokens = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = torch.tensor(tokens[:-1]).view(B, T)
        y = torch.tensor(tokens[1:]).view(B, T)
        self.current_position += self.B * self.T * self.process_rank
        # 超界限则reset
        if self.current_position + self.B * self.T * self.process_rank >= len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        # 按batch跳token
        return x, y


class DataLoaderLite1:
    def __init__(self, B, T, process_rank, num_processes, split):
        def load_tokens(filename):
            npt = np.load(filename)
            npt = npt.astype(np.int32)
            ptt = torch.tensor(npt, dtype=torch.long)
            return ptt

        self.B, self.T = B, T
        self.process_rank, self.num_processes = process_rank, num_processes
        assert split in ['train', 'val']
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")

        # get shard data
        data_root = "./edu_fineweb10B"
        shards = os.listdir(data_root)
        # 数据分为 tarin 和 val 两个部分,取出想要的部分
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split{split}"
        if master_process:
            print(f"1 epoch = {len(shards)} shards for split {split}")

        # 最后三行用一个函数处理,避免重复,且可以用于value模式
        # 最后三行用一个函数处理,避免重复,且可以用于value模式
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        tokens = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = torch.tensor(tokens[:-1]).view(B, T)
        y = torch.tensor(tokens[1:]).view(B, T)
        self.current_position += self.B * self.T * self.process_rank
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


# -----------------------------------------------------------------------------


# Use a more general method
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs: #8个单独进程,8个gpu
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# 让程序只看到 2、3 两张卡
# CUDA_VISIBLE_DEVICES=2,3 python train.py
# 程序中：
#   "cuda:0" 实际是物理卡 2
#   "cuda:1" 实际是物理卡 3

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    # 初始化进程组，使当前进程加入分布式通信（如 NCCL、Gloo 等）
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')

    # RANK 全局进程 ID（0~world_size-1
    # LOCAL_RANK  当前节点（机器）内的 GPU ID
    # WORLD_SIZE  总进程数（= 总 GPU 数）

    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    # 仅cuda0为master process 用于输出日志等信息,其它卡仅train
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288  # 2**19, ~0.5M, in number of tokens
B = 1  # micro batch size
T = 1024  # sequence length
assert total_batch_size % (
            B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# data loading
import tiktoken

# enc = tiktoken.get_encoding("gpt2")
# with open('input.txt', 'r', encoding='utf-8') as f:
#     text = f.read()
#     # print(text[:1000])
# encoding = tiktoken.get_encoding('gpt2')
# token = encoding.encode(text[:1000])


total_batch_size = 524288  # 2**19, ~0.5M, in number of tokens
assert total_batch_size % (
            B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
B, T = 2, 1024  # using for debug
use_compile = 0
# 初始化,用buf不断滑动窗口得到输入和输出来求loss


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
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model
if compile:
    model = torch.compile(model)
# model.eval()

# AdamW将优化过程中使用的针对网络权重的衰减项（或者叫正则项）从loss中单独拿了出来，
# 不参与Adam中一二阶动量的计算

# optim
print(device)
# 调整训练超参数(来自gpt3论文
max_lr = 6e-4
min_lr = max_lr * 0.1
# 以下step来自gpt3原文
warmup_steps = 715  # 375e6//(2**19)
max_steps = 19703  # 10e9//(2**19)


def get_lr(step):
    if step <= warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    # between,use cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    # cosine decay
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# 在模型内部定义optimiser,从而实现更多自定义功能
# optim = raw_model.configure_optimizer(lr=3e-4,betas=(0.9,0.95),eps=1e-8)
# optim=model.configure_optimizers(weight_decay=0.1,learning_rate=max_lr,device= device)
#  使用ddp_model进行前向和反向传播,优化时应优化原始模型
optimizer = raw_model.configure_optimizer(weight_decay=0.1, lr=6e-4, device=device)

# log
log_dir = 'log.txt'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'log.txt')
with open(log_file, 'a') as f:
    pass

# 梯度裁剪
scaler = torch.GradScaler()
# 循环训练
# data = DataLoaderLite0(B, T, process_rank=ddp_rank, num_processes=ddp_world_size)
train_loader = DataLoaderLite1(B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite1(B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

for step in range(max_steps):
    last_step = (max_steps - 1 == step)

    t0 = time.time()
    # origin data val loss
    if step % 250 == 0:
        model = model.eval()
        val_loader.reset()
        generate(model, ddp_rank)
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_step = 20
            for _ in range(val_loss_step):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                # 不用反向传播,低精度速度更快
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                val_loss_accum += (loss / val_loss_step).detach()
        if ddp:
            # 求平均用于输出
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"step {step} val loss: {val_loss / val_loss_step}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            # save model checkpoint
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # val的第二部分,做出一些output以观察状况,一般用于debug
    # and False用于屏蔽这部分,调用时删去即可
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile) and False:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        # torch.Generator随机数生成器,不影响全局seed的前提下操作随机数
        # 随后随机数的生成默认使用该 torch.Generator 实例
        # 如果你用多张显卡训练，每张卡（rank 不同）生成的随机数会略有不同，防止所有显卡打印出一模一样的文本。
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen)  # (B, T, vocab_size)
                    # take the logits at the last position
                    logits = logits[:, -1, :]  # (B, vocab_size)
                    # get the probabilities
                    probs = F.softmax(logits, dim=-1)
                    # do top-k sampling of 50 (huggingface pipeline default)
                    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # select a token from the top-k probabilities
                    # note: multinomial does not demand the input to sum to 1
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                    # append to the sequence
                    xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    from hellaswag import render_example, iterate_examples

    # val第三部分,用hellaswag value model
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0  # 已评估的 HellaSwag 验证集样本的总数量

        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            # 在一个多卡（DDP）环境中，不需要每张显卡都跑完整的 10,000 条测试数据。这里使用了取模运算 (%) 将数据集切分
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                # 根据模型输出的 logits，计算 4 个候选项中哪一个生成的概率（Normalized Probability）最大，作为模型的预测结果
                pred_norm = get_most_likely_row(tokens, mask, logits)

            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        # 同步数据：因为数据被分散到不同显卡上处理，每张卡只知道自己的正确率。
        # 聚合结果：使用 dist.all_reduce 将所有显卡的 num_total 和 num_correct_norm 相加。操作完成后，每张显卡都会得到全局的总样本数和总正确数。
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)

            # 运行时程序会暂停，等待所有显卡都运行到这一行。
            # 计算出的总和写回给每一张显卡。
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        # 主进程输出：只有主进程（Master Process，通常是 Rank 0）负责打印最终的准确率并将结果写入 log.txt 文件，用于后续绘制训练曲线或监控模型性能。
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = data.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # using amp to accelerate training
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            # 这部分等价于with ddp.no_sync():, 但更简单
            # ddp就是通过查看model.require_backward_grad_sync来确认是否需要同步梯度

            # 配合梯度累积，在每个step中,只在最后一次 micro-step 同步梯度,节省通信开销
            # 通信次数从grad_accum_steps次 → 减少为1次
            # 但结果仍然等价于：所有micro - batch的梯度在各卡累加 + 最后一次同步
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        scaler.scale(loss).backward()
    if ddp:
        # 把各进程的 loss 做 all-reduce 取平均,仅用于log
        # 梯度同步是 DDP 帮你在 backward 时做的，
        # all_reduce(loss_accum) 只是为了日志统计，不影响训练结果。
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # import code;code.interact(local=locals()) #截断运行,可以用于调试
    # 限制loss范数,避免梯度爆炸
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    # 遍历优化器中的每一组参数，并为它们设置一个新的学习率,从而实现动态lr
    # 也可以让模型的不同块具有不同的lr
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    scaler.step(optimizer)
    scaler.update()
    # 阻塞cpu流,用于预估一次训练的耗时
    # 因为运行计算过后,哪怕gpu未完成计算,cpu也会继续运行后续代码
    # 因此使用torch.cuda.synchronize()阻塞cpu流
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0  # time difference in seconds
    tokens_processed = data.B * data.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(
            f"step {step:4d} | loss: {loss_accum:.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

# 下面是之前生成的代码,这里计算loss时忽略
import sys;

sys.exit(0)

# 随机初始化的loss=10.9650
p_token = 1 / 50257
# 交叉熵计算:-ln P
print(-math.log(p_token))
# 和随机预测的loss很接近
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
for step in range(num_return_sequences):
    tokens = x[step, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
