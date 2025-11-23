FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

# 首先安装 NumPy 1.24.3
RUN pip install --no-cache-dir numpy==1.24.3

# 安装其他基础依赖
RUN pip install --no-cache-dir \
    pillow==10.0.1 \
    pydantic==2.5.0

# 安装 PyTorch
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 安装 OpenCV 和其他依赖
RUN pip install --no-cache-dir \
    opencv-python==4.8.1.78 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    python-multipart==0.0.6 \
    lapx>=0.5.2

# 安装 ultralytics
RUN pip install --no-cache-dir ultralytics==8.0.186

# 最后再次强制安装正确的 NumPy 版本，覆盖任何升级
RUN pip install --no-cache-dir --force-reinstall numpy==1.24.3

COPY . .

RUN mkdir -p uploaded_images uploaded_videos model scripts

EXPOSE 8000