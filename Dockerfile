FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set noninteractive installation to avoid timezone configuration prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install PyTorch 2.5.1 with CUDA 12.4 support
RUN pip3 install --no-cache-dir torch==2.5.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip3 install --no-cache-dir \
    timm \
    tqdm \
    scikit-learn \
    pandas \
    matplotlib \
    umap-learn \
    pyyaml

# Copy the code
COPY . .

# Set up entrypoint
CMD ["python", "main.py", "--config", "/app/config/config.yaml"]