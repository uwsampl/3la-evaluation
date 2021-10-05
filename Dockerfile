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
    libclang-dev \
    libssl-dev \
    pkg-config \
    python3-dev \
    sudo \
    wget
#    libgtest-dev \
#    unzip \
#    libtinfo-dev \
#    libz-dev \
#    libcurl4-openssl-dev \
#    libopenblas-dev \
#    lsb-release \
#    software-properties-common \
#    python3-pip \

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Install LLVM
RUN wget https://apt.llvm.org/llvm.sh
RUN chmod +x llvm.sh
RUN sudo ./llvm.sh 10

# Needed by TVM Rust build process
ENV LLVM_CONFIG_PATH=/usr/lib/llvm-10/bin/llvm-config

# Build TVM with Rust bindings 
# THIS MUST BE KEPT UP-TO-DATE WITH WHATEVER TVM VERSION GLENSIDE IS USING!
# This is a frustrating feature of the Dockerfile. We could potentially rely on
# the TVM Rust crate to build TVM, which mostly works but I ran into some
# library-loading issues. Additionally, I'm not sure that method will install
# the needed Python bindings and update PYTHONPATH.
WORKDIR /root
ADD 3la-tvm tvm
WORKDIR /root/tvm
RUN echo 'set(USE_LLVM $ENV{LLVM_CONFIG_PATH})' >> config.cmake
RUN echo 'set(USE_RPC ON)' >> config.cmake
RUN echo 'set(USE_SORT ON)' >> config.cmake
RUN echo 'set(USE_GRAPH_RUNTIME ON)' >> config.cmake
RUN echo 'set(USE_BLAS openblas)' >> config.cmake
RUN echo 'set(CMAKE_CXX_STANDARD 14)' >> config.cmake
RUN echo 'set(CMAKE_CXX_STANDARD_REQUIRED ON)' >> config.cmake
RUN echo 'set(CMAKE_CXX_EXTENSIONS OFF)' >> config.cmake
#RUN echo 'set(CMAKE_BUILD_TYPE Debug)' >> config.cmake
ARG TVM_BUILD_JOBS=2
RUN bash -c \
     "mkdir -p build && \
     cd build && \
     cmake .. && \
     make -j${TVM_BUILD_JOBS}"

# Help the system find the libtvm library and TVM Python library
ENV TVM_HOME=/root/tvm
ENV PYTHONPATH="$TVM_HOME/python:$TVM_HOME/topi/python:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="$TVM_HOME/build/"

# Set up Python
RUN pip3 install --upgrade pip
COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

# Build Glenside with all features
WORKDIR /root
ADD glenside glenside
WORKDIR /root/glenside
RUN --mount=type=ssh cargo build --no-default-features --features "tvm"