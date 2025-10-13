# AI-Vision-Service

A practical project focused on visual language model deployment optimization and inference acceleration, integrating mainstream multimodal models (Flux, Qwen2.5-VL) with efficient inference through technologies like `torch.compile` 、`Flash Attention` and so on.

## Project Features

- **Multi-model Support**: Integrates two core capabilities: text-to-image (Flux) and image-to-text (Qwen2.5-VL)

- **Inference Acceleration Optimization**:

  - Achieves operator fusion and computation graph optimization with torch.compile

  - Improves attention mechanism efficiency using Flash Attention

  - Optimizes D2H (device-to-host) data transmission bottlenecks

- **Batch Processing **Support\*\*: Provides both single-task and batch processing interfaces to adapt to different scenario requirements

- **Performance Observability**: Integrates PyTorch Profiler for performance analysis and bottleneck identification

## Supported Models

| Model Type | Model Name             | Description                             |
| ---------- | ---------------------- | --------------------------------------- |
| text2image | FLUX.1-dev             | Text-to-image generation                |
| image2text | Qwen2.5-VL-3B-Instruct | Image understanding and text generation |

## Setup

### Dependency Requirements

- Python >= 3.10

- CUDA 12.0+ (recommended for GPU acceleration)

- Other dependencies are listed in pyproject.toml

### Installation

```bash
# Clone the repository
git clone https://github.com/fyh2001/AI-Vision-Service.git
cd AI-Vision-Service

# Install dependencies (recommended to use uv for acceleration)
pip install uv
uv sync
```

### Quick Start

#### Launch the Service

```bash
# Start text-to-image service
MODEL_TYPE=text2image uvicorn src.api.batch_task:app --host 0.0.0.0 --port 8000

# Start image-to-text service
MODEL_TYPE=image2text uvicorn src.api.batch_task:app --host 0.0.0.0 --port 8000
```

#### API Call Examples

Text-to-Image

```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A beautiful sunset over a calm ocean"}'
```

Image-to-Text

```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"images": ["outputs/image_0.png"], "prompt": "Describe this image"}'
```

## Optimization Insights

Below are the optimization strategies and analytical methods I documented throughout my technical practice. For a step-by-step understanding of the optimization process, I recommend reading them in this order:

1. [Flux Model Optimized with `torch.compile`](./docs/flux/torch_compile.md)

   - Covers how to tune torch.compile parameters (e.g., mode="reduce-overhead") for the Flux text-to-image model, plus test results showing ~20% inference speedup.

2. [Qwen2.5-VL optimized by `torch.compile`](./docs/qwen2_5_vl/torch_compile.md)

   - Focuses on operator fusion effects and compilation overhead reduction for the Qwen2.5-VL image-to-text model, with data on its ~10% speed improvement.

3. [D2H Transmission Optimization for Qwen2.5-VL](./docs/qwen2_5_vl/attention_d2h.md)

   - Explains how to fix device-to-host (D2H) bottlenecks: avoiding frequent GPU-to-CPU tensor conversions in attention calculations to cut data transfer time.

4. [Qwen2.5-VL Optimized with `FlashAttention`](./docs/qwen2_5_vl/flash_attention.md)

   - Shows how to replace standard attention with FlashAttention 2, including steps for installation and tests on memory usage reduction and latency improvement.
