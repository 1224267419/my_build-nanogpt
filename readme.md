从0开始构建gpt2,按视频教程一步步进行

### 25-11-18
[play.ipynb](play.ipynb)加载来自transformers的gpt2模型,并根据官方demo输出流模型的参数,实现了利用模型进行输出

25-11-19: 实现了dataloader,模型预测和训练,并得知为什么要在残差中乘上 (n**-0.5) :见play中的实验
```python
import torch
x=torch.zeros(768)
n=100
for i in range(n):
    x+=torch.randn(768)
# 由于残差的影响,多次残差后标准差会发生偏移,nearly sqrt(n)
print(x.std ())
# 通过对x缩放来使得标准差与n的平方根相同
print(x.std()/n**0.5)

x=torch.zeros(768)
n=100
for i in range(n):
    x+=torch.randn(768)*n**-0.5
print(x.std())
```
使得方差缩放回1左右

### 11-20:  计算与精度

`torch.set_float32_matmul_precision('high')`可以用于设置运算精度,在带来微量误差的同时,提升运算速度,但很多时候仍受到内存(显存)速率影响

tf32和fp16具有相同的尾数长度,仅阶码部分存在差异,
tf32和bf16具有相同的阶码长度,仅尾数长度不同,更适合fp32和tf32截断尾数进行运算,从而提升运算效率

![image-20251120104643063](./readme.assets/image-20251120104643063.png)

`torch.amp.autocast` 是 PyTorch 提供的一个工具，用于自动处理混合精度训练中的[数值类型](https://so.csdn.net/so/search?q=数值类型&spm=1001.2101.3001.7020)选择，使得计算能在尽量减少精度损失的同时，提升性能。

1. 当代码进入 `autocast` 上下文时，PyTorch 会自动将相关操作（如矩阵乘法、卷积等）切换到 **低精度浮点数**（通常为 `float16` 或 `bfloat16`），以提高计算速度和节省显存。
2. 计算完成后，`autocast` 会退出上下文并将所有变量恢复到 **默认精度**（通常是 `float32`）。这对于梯度计算和权重更新至关重要，因为在低精度下进行梯度计算可能会导致数值不稳定或精度损失。
3. 在 `autocast` 内部进行的前向传播计算使用低精度（`float16` 或 `bfloat16`），但 **梯度计算和权重更新** 操作仍然在 **`float32`** 精度下进行，以保证数值稳定性。

```python
# Creates model and optimizer in default precision
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

# Creates a GradScaler once at the beginning of training.
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()

        # Runs the forward pass with autocasting.
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.混合精度专用,用于避免上下溢
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()
```

amp中会使用fp16/32的模块见[链接](https://docs.pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16)
一般而言,**乘法不易受精度影响**,而指数运算影响较大,具体看链接吧

`import code;code.interact(local=locals())`使用这段代码可以截断运行并保存运行状态,然后打开一个ipython**解释器**，让你可以实时查看和操作当前作用域内的变量与函数。