<h1 align="center">AI-Vision-Service</h1>

<p align="center">
🌐 Language Switch | <a href="./README_CN.md">中文文档</a> / <a href="./README.md">English Documentation</a>
</p>

一个专注于视觉语言模型部署优化与推理加速的实践项目，集成了主流多模态模型（Flux、Qwen2.5-VL），并通过 `torch.compile`、`Flash Attention` 等技术实现高效推理。

## 项目特点

- **多模型支持**：集成文本生成图像（Flux）和图像生成文本（Qwen2.5-VL）两大核心能力

- **推理加速优化**：

  - 基于 `torch.compile` 实现算子融合与计算图优化

  - 采用 Flash Attention 提升注意力机制计算效率

  - 优化设备到主机（D2H）数据传输瓶颈

- **批量处理支持**：提供单任务与批处理两种接口，适配不同场景需求

- **性能可观测**：集成 PyTorch 性能分析工具（PyTorch Profiler），支持性能分析与瓶颈定位

## 支持模型

| 模型类型   | 模型名称               | 功能描述           |
| ---------- | ---------------------- | ------------------ |
| text2image | FLUX.1-dev             | 文本生成图像       |
| image2text | Qwen2.5-VL-3B-Instruct | 图像理解与文本生成 |

## 环境搭建

### 依赖要求

- Python ≥ 3.10

- CUDA 12.0+（推荐，用于 GPU 加速）

- 其他依赖详见 `pyproject.toml` 文件

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/fyh2001/AI-Vision-Service.git
cd AI-Vision-Service

# 安装依赖（推荐使用 uv 工具加速）
pip install uv
uv sync
```

### 快速开始

#### 启动服务

```bash
# 启动文本生成图像服务
MODEL_TYPE=text2image uvicorn src.api.batch_task:app --host 0.0.0.0 --port 8000

# 启动图像生成文本服务
MODEL_TYPE=image2text uvicorn src.api.batch_task:app --host 0.0.0.0 --port 8000
```

#### API 调用示例

##### 文本生成图像（调用示例）

```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A beautiful sunset over a calm ocean"}'
```

##### 图像生成文本（调用示例）

```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"images": ["outputs/image_0.png"], "prompt": "Describe this image"}'
```

## 优化心得

以下是我在技术实践过程中记录的优化方案与分析方法。若想循序渐进理解完整的优化流程，建议按以下顺序阅读：`/AI-Vision-Service/docs`

1. [基于 `torch.compile` 的 Flux 模型优化](./docs/flux/torch_compile.md)  
   — 介绍如何为 Flux 文本生成图像模型调优 `torch.compile` 参数（如 `mode="reduce-overhead"`），并附推理速度提升约 20% 的测试结果。

2. [基于 `torch.compile` 的 Qwen2.5-VL 模型优化](./docs/qwen2_5_vl/torch_compile.md)  
   — 聚焦 Qwen2.5-VL 图像生成文本模型的算子融合效果与编译开销优化，附带其推理速度提升约 10% 的实测数据。

3. [Qwen2.5-VL 的 D2H 传输优化](./docs/qwen2_5_vl/attention_d2h.md)  
   — 讲解如何解决设备到主机（D2H）传输瓶颈：通过避免在注意力计算中频繁进行 GPU 与 CPU 张量转换，减少数据传输耗时。

4. [基于 FlashAttention 的 Qwen2.5-VL 模型优化](./docs/qwen2_5_vl/flash_attention.md)  
   — 说明如何用 FlashAttention 2 替代标准注意力机制，包括安装步骤以及内存占用降低、推理延迟缩短的测试结果。
