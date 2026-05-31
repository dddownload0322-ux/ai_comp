# Nibble Neural Compression Prototype

这个工具把每个字节拆成两个 4-bit token，也就是高 4 bit 在前、低 4 bit 在后。模型用长度为 `M` 的 token 上下文预测下一个 token 的 16 类概率分布。

模型结构固定为 4 层全连接：

```text
M -> 16 -> 16 -> 16 -> 16
```

前三个线性层后接 ReLU，最后一层输出 16 个 logits。

## 1. 训练

下面命令里的 `py` 可以替换成你的 Python 解释器路径；当前机器上的 PyTorch 解释器路径类似 `C:\Users\xia\AppData\Local\Programs\Python\Python312\python.exe`。

```powershell
py train.py --input .\enwik8\enwik8 --context 64 --epochs 1 --steps-per-epoch 10000 --batch-size 4096 --model-out .\model.pt
```

训练脚本会随机采样上下文窗口。`--max-bytes` 可用于快速小样本测试。

## 2. 推理与带宽统计

只统计推理带宽：

```powershell
py infer.py --input .\enwik8\enwik8 --model .\model.pt --batch-size 65536
```

同时保存每个被预测 token 的量化频率分布：

```powershell
py infer.py --input .\enwik8\enwik8 --model .\model.pt --prob-out .\enwik8.freq.npy --prob-format freq-u16
```

注意：概率表大小约为 `(2 * 原始字节数 - M) * 16 * 2` 字节。对 100 MB 文件会非常大。

## 3. 压缩与压缩比

直接用模型边推理边 arithmetic/range coding：

```powershell
py compress.py --input .\enwik8\enwik8 --model .\model.pt --output .\enwik8.nnrc
```

也可以使用推理脚本保存的概率表：

```powershell
py compress.py --input .\enwik8\enwik8 --prob-in .\enwik8.freq.npy --prob-format freq-u16 --output .\enwik8.nnrc
```

快速估算熵编码长度，不写出 range-coded payload：

```powershell
py compress.py --input .\enwik8\enwik8 --model .\model.pt --estimate-only
```

压缩文件中会保存头部和前 `M` 个 seed token；其余 token 由模型概率驱动 arithmetic coder 编码。解码时需要相同的模型 checkpoint。
