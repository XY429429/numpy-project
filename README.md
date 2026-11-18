# 项目 5：用 NumPy 复现 OpenCLIP ViT-B/32 图像编码器

本项目在**只使用 NumPy**（不导入 PyTorch / OpenCLIP）的条件下，复现 OpenCLIP ViT-B/32 图像塔的完整前向计算：

```
[B, 3, 224, 224]  ->  [B, 49, 3072]  ->  [B, 50, 768]
                  ->  12 × Transformer block
                  ->  CLS [B, 768]  ->  [B, 512] 单位向量
```

课程提供固定的 152 个视觉张量、`helpers.py`、预处理脚本和逐阶段参考数组；学生完成 `image_encoder.py` 中的 TODO-01—06。

## 我的实现

| TODO | 位置 | 内容 |
| --- | --- | --- |
| **TODO-01** | `ImageEncoder.patchify` | `[B,3,224,224] -> [B,49,3072]`，7×7 patch，patch 内按 C-H-W 展平 |
| **TODO-02** | `MultiHeadAttention.split_heads` | `[B,N,D] -> [B,H,N,D/H]`，D=768，H=12，d=64 |
| **TODO-03a** | `MultiHeadAttention.__call__` | `scores = QKᵀ / sqrt(d)`，`[B,12,50,50]` |
| **TODO-03b/c** | `MultiHeadAttention.__call__` | 系数加权 V、合并 12 个 head、输出投影 |
| **TODO-04a/b** | `TransformerBlock.__call__` | 两条残差 + `GELU(uW₁ᵀ+b₁)W₂ᵀ+b₂` |
| **TODO-05a/b** | `ImageEncoder.embed` | 无偏置 patch 投影、拼接 CLS、加位置编码 |
| **TODO-06** | `ImageEncoder.__call__` | 取 CLS、post-LayerNorm、投影到 512 维、L2 单位化 |

报告中的插图用**红色**标出了以上本人实现的部分：`report/figures/architecture.pdf`。

## GitHub

仓库地址：<https://github.com/XY429429/numpy-project>

## 环境

- Python 3.9+（本项目实测：Python 3.14.0 + NumPy 2.5.5，项目自带 `.venv`）
- 依赖见 `requirements.txt`：NumPy、Pillow、imageio
- 不需要 GPU、PyTorch、OpenCLIP 包或网络

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 复现

课程的大体积数据不放进 Git 仓库，请先从课程附件复制以下内容：

```
weights/image_encoder.npz        # 约 326 MB，152 个 float32 视觉张量
assets/                          # 10 张图片 + reference_outputs.npz + manifests
```

然后运行：

```bash
# 1) 逐阶段数值检查（assets -> preprocess -> patches -> tokens
#    -> block0 -> last_tokens -> output -> batch）
python check.py --stage all

# 2) 用 10 张课程图片生成 [10,512] 输出
python run.py --output results/output_tokens.npz

# 3) 也可以指定自己的图片
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

`check.py` 会校验权重和参考文件的 SHA-256；文件不对会立即失败，不会出现“看起来在跑、其实读错文件”的静默错误。参考数组只用于检查，绝不会作为编码器输入。

### 预期输出

`results/check_all.log` 是本地运行的完整日志。关键结果：

| 阶段 | 形状 | max abs | mean abs | atol | 结果 |
| --- | --- | --- | --- | --- | --- |
| preprocess | (10,3,224,224) | 0 | 0 | 2e-6 | PASS |
| patches | (2,49,3072) | 0 | 0 | 2e-4 | PASS |
| tokens | (2,50,768) | 4.005e-05 | 1.289e-06 | 2e-4 | PASS |
| block0 | (2,50,768) | 7.063e-06 | 3.250e-07 | 2e-4 | PASS |
| last_tokens | (2,50,768) | 4.768e-05 | 2.696e-06 | 2e-4 | PASS |
| output | (10,512) | 5.439e-07 | 8.311e-08 | 2e-5 | PASS |
| batch | (3,512) | 4.619e-07 | 7.632e-08 | 2e-5 | PASS |

`batch` 阶段使用顺序 `[1,0,1]`，包含重复图片，用来验证编码器不依赖缓存或输入顺序。最终输出每行的 L2 范数为 1（rtol/atol = 2e-4）。

## 目录结构

```
.
├── image_encoder.py       # 六个 TODO 的实现（核心代码，含详细注释）
├── helpers.py             # 课程提供：LayerNorm / GELU / softmax / unit_vector
├── preprocess_image.py    # 课程提供：resize / center crop / normalize
├── check.py               # 课程提供：逐阶段参考检查
├── run.py                 # 课程提供：命令行入口
├── runtime.py             # 课程提供：路径与数据加载
├── config.json            # 模型结构常量
├── weights/               # 课程提供的大体积张量（不提交到 Git）
├── assets/                # 课程提供的图片与参考数组（不提交到 Git）
├── results/               # 本次运行的日志与输出
├── report/                # LaTeX 报告、插图和 Overleaf 压缩包
├── README.course.md       # 课程原始 README（保留备查）
└── README.md              # 本文件
```

## 实现要点

### 1. patch 切分（TODO-01）

```python
x = images.reshape(B, 3, 7, 32, 7, 32)      # (B, C, G_h, P_h, G_w, P_w)
x = x.transpose(0, 2, 4, 1, 3, 5)            # (B, G_h, G_w, C, P_h, P_w)
patches = x.reshape(B, 49, 3072)             # 49 = 7*7, 3072 = 3*32*32
```

`224 = 7 × 32`。转置后按 `(G_h, G_w)` 行优先展平，等价于“从左到右、再从上到下”的 patch 顺序；patch 内部按 C-H-W 展平，与 `visual.conv1.weight` 的 `[768,3,32,32]` 布局一致，因此 `patches @ patch_weight.T` 与 stride 32、kernel 32 的卷积数值等价。

### 2. 多头注意力（TODO-02/03）

```
[B,50,768] -> [B,50,12,64] -> [B,12,50,64]
scores = QKᵀ / sqrt(64)          # [B,12,50,50]
context = softmax(scores) @ V    # [B,12,50,64]
-> [B,50,12,64] -> [B,50,768] -> out_proj
```

课程权重按 `[out,in]` 存储，所以所有线性层都用 `weight.T`；只有 `visual.proj` 已经是 `[in,out]=[768,512]`，直接使用。

### 3. 一个真实 bug：NumPy 2 的 dtype 提升

六个 TODO 写完后，`patches` 和 `tokens` 通过，但 `block0` 报 `use float32`。原因不是数值错，而是 dtype：

```python
np.sqrt(self.head_dim)          # np.float64(8.0)，NumPy 标量
float32_array / np.float64      # NumPy 2 (NEP 50) -> float64
```

`scores` 变成 float64 后，沿 `coefficients -> context -> 合并 -> 输出投影 -> 残差 -> MLP` 一路传播，12 层之后整个输出都是 float64。修复：

```python
scale = np.float32(np.sqrt(self.head_dim))
scores = (q @ k.transpose(0, 1, 3, 2)) / scale
```

或者用 Python 标准库的 `math.sqrt`（Python float 在 NumPy 2 中是弱标量）。修复后所有阶段通过，最终 `output` 的 max abs 为 5.44e-07。

## 报告与 Overleaf

- `report/main.pdf`：已编译的 5 页版本（封面 + 4 页正文，满足“少于 5 页，不包括封面”）。
- `report/main.tex`：XeLaTeX 源码；插图是 `report/figures/architecture.pdf`（矢量）。
- `report/report-overleaf.zip`：可直接上传 Overleaf，编译器选择 **XeLaTeX**。

报告中用红色高亮本人实现的部分，并包含逐阶段误差表、一个形状变化（patchify）和一个真实 bug（NumPy dtype）的完整定位过程。

## 贡献与帮助

本人完成 TODO-01—06 的实现、调试和数值验证；使用 Claude Code 辅助定位 NumPy dtype 问题、生成插图与 LaTeX 排版。所有代码与数值结果均在本地独立运行验证。

## 说明

`weights/`、`assets/` 中的课程数据仅用于本课程作业，不随仓库分发；请从课程附件获取。课程提供的 `helpers.py`、`check.py`、`run.py`、`preprocess_image.py` 等文件的版权归课程/原作者所有。
