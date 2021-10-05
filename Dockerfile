FROM ubuntu:20.04

# Install needed packages
# Needed so that tzdata install will be non-interactive
# https://stackoverflow.com/questions/44331836/apt-get-install-tzdata-noninteractive
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \
  apt install -y \
    cmake \
    curl \
    g++ \
    git \
    libssl-dev \
    pkg-config \
    python3-dev
#    libgtest-dev \
#    wget \
#    unzip \
#    libtinfo-dev \
#    libz-dev \
#    libcurl4-openssl-dev \
#    libopenblas-dev \
#    sudo \
#    libclang-dev \
#    lsb-release \
#    wget \
#    software-properties-common \
#    python3-pip \

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Build Glenside with all features
WORKDIR /root
ADD glenside glenside
WORKDIR /root/glenside
RUN --mount=type=ssh NUM_JOBS=2 cargo build --no-default-features --features "tvm"