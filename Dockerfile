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
    libcurl4-openssl-dev \
    libgtest-dev \
    libopenblas-dev \
    libssl-dev \
    libtinfo-dev \
    libz-dev \
    lsb-release \
    pkg-config \
    python3-dev \
    python3-pip \
    software-properties-common \
    sudo \
    unzip \
    wget

# Set up MLPerf inference. (Downloads take a while, so this should be up near
# the top.)
 the pipeline)
WORKDIR /root
ADD ./inference ./inference
WORKDIR /root/inference/language/bert/
RUN mkdir build
RUN	cp ../../mlperf.conf build/mlperf.conf
# Make wget quiet so as to not spam the output.
RUN echo "verbose = off" >> /root/.wgetrc
RUN make download_data
RUN make download_model

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Install LLVM
WORKDIR /root
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
# Note the --ignore-libllvm, necessary for fixing Rust bindings as mentioned
# here:
# https://discuss.tvm.apache.org/t/python-debugger-segfaults-with-tvm/843/9
RUN echo 'set(USE_LLVM "$ENV{LLVM_CONFIG_PATH} --ignore-libllvm")' >> config.cmake
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
# --ignore-installed to fix a weird bug when installing PyYAML, see:
# https://github.com/pypa/pip/issues/5247.
RUN pip3 install -r requirements.txt --ignore-installed

# Build Glenside with all features
WORKDIR /root
ADD glenside glenside
WORKDIR /root/glenside
ENV GLENSIDE_HOME=/root/glenside
RUN --mount=type=ssh cargo build --no-default-features --features "tvm"

WORKDIR /root
ADD run.sh run.sh

WORKDIR /root
ADD ./bert_onnx.py ./bert_onnx.py

WORKDIR /root
ADD ./e2e ./e2e

WORKDIR /root
ADD ./flexmatch ./flexmatch
# need to have flexmatch in the environment
ENV PYTHONPATH="/root/flexmatch/:${PYTHONPATH}"
