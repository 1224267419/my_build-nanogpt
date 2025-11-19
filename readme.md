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