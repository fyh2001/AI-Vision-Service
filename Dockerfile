# 基础镜像：CUDA 12.2 + cuDNN
FROM --platform=linux/amd64 nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04


# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install uv

COPY flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl /app/
COPY pyproject.toml /app/
COPY uv.lock /app/

RUN uv python install 3.10
RUN uv sync
RUN uv venv

# 使用 uv 安装 flash-attn
RUN uv add flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation

# 复制项目文件到容器
COPY . .

ENV PYTHONPATH=/app

# 默认启动命令（根据你的项目修改）
CMD ["uv", "run", "python", "src/models/t2i.py"]
