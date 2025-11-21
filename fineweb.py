import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset 
from tqdm import tqdm 

# ------------------------------------------
# 配置部分
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard

# ------------------------------------------
# 全局变量与函数
# 注意：在 Windows 下，子进程会重新运行这部分代码。
# tiktoken 的初始化速度很快，保留在全局是可接受的。
# 如果放在 main 里面，子进程因为看不到 enc 变量会报错。

try:
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>'] 
except Exception as e:
    print(f"Tokenizer init failed: {e}")

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    # 确保在这个函数里使用的变量（如 enc, eot）在子进程的全局作用域里是存在的
    tokens = [eot] 
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# ------------------------------------------
# 主程序入口
if __name__ == "__main__":
    # 这一行不仅是规范，在 Windows 多进程中是必须的
    # 它可以防止子进程陷入无限递归生成新进程的死循环
    mp.freeze_support() 

    # 1. 创建缓存目录 (移入 main)
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # 2. 下载/加载数据集 (移入 main，防止子进程重复加载！)
    print("Loading dataset...")
    # 如果你是加载本地文件夹，请确保路径正确
    # fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
    fw = load_dataset("./edu_data", split="train") 

    # 3. 设置进程数
    nprocs = max(1, os.cpu_count() // 2)
    print(f"Processing with {nprocs} processes...")

    # 4. 开始处理
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        # pool.imap 会在主进程迭代 fw，并将数据分发给子进程，避免了子进程加载 fw
        for tokens in pool.imap(tokenize, fw, chunksize=16):

            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])
    
    print("Done.")