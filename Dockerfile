##
## Build ILA models and their SystemC simulator (from 3la-integrate)
##
FROM ubuntu:bionic as ilabuilder
LABEL stage=intermediate

# var
ENV WORK_ROOT /root
ENV VIRTUAL_ENV 3laEnv
ENV BUILD_PREF $WORK_ROOT/$VIRTUAL_ENV
RUN mkdir -p $BUILD_PREF

# need to make this if no virtualenv
RUN mkdir -p $BUILD_PREF/bin

# required packages
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install --yes --no-install-recommends \
    bison \
    build-essential \
    ca-certificates \
    flex \
    gcc-5 \
    g++-5 \
    git \
    libz3-dev \
    openssh-client \
    python3 \
    python3-pip \
    wget \
    z3 \
    && rm -rf /var/lib/apt/lists/*

# setup local build via virtualenv
#WORKDIR $WORK_ROOT
#RUN pip3 install virtualenv
#RUN virtualenv $VIRTUAL_ENV

# cmake
ENV CMAKE_DIR $WORK_ROOT/cmake-3.19.2-Linux-x86_64
WORKDIR $WORK_ROOT
RUN wget https://github.com/Kitware/CMake/releases/download/v3.19.2/cmake-3.19.2-Linux-x86_64.tar.gz \
  && tar zxvf cmake-3.19.2-Linux-x86_64.tar.gz

# SystemC
ENV SYSC_DIR $WORK_ROOT/systemc-2.3.3
WORKDIR $WORK_ROOT
RUN wget https://accellera.org/images/downloads/standards/systemc/systemc-2.3.3.tar.gz && \
  tar zxvf systemc-2.3.3.tar.gz && \
  cd $SYSC_DIR && \
  mkdir -p build && \
  cd $SYSC_DIR/build && \
  $CMAKE_DIR/bin/cmake $SYSC_DIR -DCMAKE_INSTALL_PREFIX=$BUILD_PREF -DCMAKE_CXX_STANDARD=11 && \
  make -j"$(nproc)" && \
  make install 

# boost
ENV BOOST_DIR $WORK_ROOT/boost_1_75_0
WORKDIR $WORK_ROOT
RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.75.0/source/boost_1_75_0.tar.gz && \
  tar zxvf boost_1_75_0.tar.gz && \
  cd $BOOST_DIR && \
  ./bootstrap.sh --prefix=$BUILD_PREF && \
  ./b2 --with-chrono --with-math --with-system install -j"$(nproc)" || :

# to access private repo
ARG SSH_KEY
RUN eval "$(ssh-agent -s)" && \
  mkdir -p /root/.ssh/ && \
  echo "$SSH_KEY" > /root/.ssh/id_rsa && \
  chmod -R 600 /root/.ssh/ && \
  ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

# 3la_sim_testbench
ENV SIM_TEST_DIR $WORK_ROOT/3la_sim_testbench
WORKDIR $WORK_ROOT
RUN git clone --depth=1 --branch extended_spad git@github.com:LeeOHzzZ/3la_sim_testbench.git $SIM_TEST_DIR && \
  cd $SIM_TEST_DIR && \
  git submodule init && \
  git submodule update tool/numcpp

# 3la_ILA_tensor_op
ENV ILA_TENSOR_OP_DIR $WORK_ROOT/3la_ILA_tensor_op
WORKDIR $WORK_ROOT
RUN git clone --depth=1 --branch sim_mapping git@github.com:LeeOHzzZ/3la_ILA_tensor_op.git $ILA_TENSOR_OP_DIR

# ILAng
ENV ILANG_DIR $WORK_ROOT/ILAng
WORKDIR $WORK_ROOT
RUN git clone --depth=1 https://github.com/PrincetonUniversity/ILAng.git $ILANG_DIR
WORKDIR $ILANG_DIR
# Branch: ilator_opt
RUN git fetch origin ilator_opt && git checkout 7de6fd9f78999845644326e462bcb723daf60b6f
RUN mkdir -p build 
WORKDIR $ILANG_DIR/build
# RUN $CMAKE_DIR/bin/cmake $ILANG_DIR -DCMAKE_INSTALL_PREFIX=$BUILD_PREF && \
RUN $CMAKE_DIR/bin/cmake $ILANG_DIR && \
    make -j"$(nproc)" && \
    make install 

# vta-ila
ENV VTA_ILA_DIR $WORK_ROOT/vta-ila
WORKDIR $WORK_ROOT
ADD https://api.github.com/repos/LeeOHzzZ/vta-ila/git/refs/heads/master vtaila_version.json
RUN git clone --depth=1 https://github.com/LeeOHzzZ/vta-ila.git $VTA_ILA_DIR
WORKDIR $VTA_ILA_DIR
RUN git fetch origin alu_mul && git checkout 41a12ae4b5a29e6139bc5dcbfa4c726502897338
RUN mkdir -p build
WORKDIR $VTA_ILA_DIR/build
RUN $CMAKE_DIR/bin/cmake $VTA_ILA_DIR -DCMAKE_PREFIX_PATH=$BUILD_PREF && \
    make -j"$(nproc)" && \
    ./vta

# vta-ila simulator
ENV VTA_SIM_DIR $VTA_ILA_DIR/build/sim_model
RUN cp $SIM_TEST_DIR/vta/sim_driver.cc $VTA_SIM_DIR/app/main.cc
RUN cp $SIM_TEST_DIR/vta/uninterpreted_func.cc $VTA_SIM_DIR/extern/uninterpreted_func.cc
WORKDIR $VTA_SIM_DIR
RUN mkdir -p build
WORKDIR $VTA_SIM_DIR/build
RUN HEADER="-isystem$SIM_TEST_DIR/vta/ap_include" && \
    $CMAKE_DIR/bin/cmake $VTA_SIM_DIR \
      -DCMAKE_PREFIX_PATH=$BUILD_PREF \
      -DCMAKE_CXX_FLAGS=$HEADER && \
    make -j"$(nproc)"

# HLSCNN
ENV HLSCNN_DIR $WORK_ROOT/HLSCNN_Accel
WORKDIR $WORK_ROOT
RUN git clone git@github.com:ttambe/HLSCNN_Accel.git $HLSCNN_DIR
WORKDIR $HLSCNN_DIR
RUN git submodule update --init --recursive

# hlscnn-ila
ENV CNN_ILA_DIR $WORK_ROOT/hlscnn-ila
WORKDIR $WORK_ROOT
RUN git clone --depth=1 --branch extended_spad https://github.com/PrincetonUniversity/hlscnn-ila.git $CNN_ILA_DIR
WORKDIR $CNN_ILA_DIR
RUN mkdir -p build
WORKDIR $CNN_ILA_DIR/build
RUN $CMAKE_DIR/bin/cmake $CNN_ILA_DIR -DCMAKE_PREFIX_PATH=$BUILD_PREF && \
    make -j"$(nproc)" && \
    ./hlscnn

# HlsCnn-ila simulator: sim_driver 
ENV CNN_SIM_DIR $CNN_ILA_DIR/build/sim_model
RUN cp $SIM_TEST_DIR/hlscnn/sim_driver.cc $CNN_SIM_DIR/app/main.cc
RUN cp $SIM_TEST_DIR/hlscnn/uninterpreted_func.cc $CNN_SIM_DIR/extern/uninterpreted_func.cc
WORKDIR $CNN_SIM_DIR
RUN mkdir -p build
WORKDIR $CNN_SIM_DIR/build
RUN HEADER0="-isystem$SIM_TEST_DIR/ac_include" && \
    HEADER1="-isystem$HLSCNN_DIR/cmod/include" && \
    HEADER2="-isystem$HLSCNN_DIR/cmod/harvard/top" && \
    $CMAKE_DIR/bin/cmake $CNN_SIM_DIR \
      -DCMAKE_PREFIX_PATH=$BUILD_PREF \
      -DCMAKE_CXX_STANDARD=11 \
      -DCMAKE_CXX_COMPILER=g++-5 \
      -DCMAKE_CXX_FLAGS="$HEADER0 $HEADER1 $HEADER2" && \
    make -j"$(nproc)"
RUN cp hlscnn $BUILD_PREF/bin/hlscnn_sim_driver

# FlexNLP
ENV FLEX_NLP_DIR $WORK_ROOT/FlexNLP
WORKDIR $WORK_ROOT
RUN git clone --depth=1 git@github.com:ttambe/FlexNLP.git $FLEX_NLP_DIR
WORKDIR $FLEX_NLP_DIR
RUN git submodule update --init --recursive

# flexnlp-ila
ENV FLEX_ILA_DIR $WORK_ROOT/flexnlp-ila
WORKDIR $WORK_ROOT
ADD https://api.github.com/repos/PrincetonUniversity/flexnlp-ila/git/refs/heads/master flexila_version.json
RUN git clone --depth=1 https://github.com/PrincetonUniversity/flexnlp-ila.git $FLEX_ILA_DIR
WORKDIR $FLEX_ILA_DIR
RUN mkdir -p build
WORKDIR $FLEX_ILA_DIR/build
RUN $CMAKE_DIR/bin/cmake $FLEX_ILA_DIR -DCMAKE_PREFIX_PATH=$BUILD_PREF && \
    make -j"$(nproc)" && \
    ./flex

# FlexNLP-ila simulator
ENV FLEX_SIM_DIR $FLEX_ILA_DIR/build/sim_model
RUN cp $SIM_TEST_DIR/flexnlp/sim_driver/sim_driver.cc $FLEX_SIM_DIR/app/main.cc
RUN cp $SIM_TEST_DIR/flexnlp/sim_driver/uninterpreted_func.cc $FLEX_SIM_DIR/extern/uninterpreted_func.cc
WORKDIR $FLEX_SIM_DIR
RUN mkdir -p build
WORKDIR $FLEX_SIM_DIR/build
RUN HEADER0="-isystem$SIM_TEST_DIR/ac_include" && \
    HEADER1="-isystem$FLEX_NLP_DIR/cmod/include" && \
    HEADER2="-isystem$FLEX_NLP_DIR/matchlib/cmod/include" && \
    HEADER3="-isystem$FLEX_NLP_DIR/matchlib/rapidjson/include" && \
    HEADER4="-isystem$FLEX_NLP_DIR/matchlib/connections/include" && \
    DEF0="-DSC_INCLUDE_DYNAMIC_PROCESSES" && \
    DEF1="-DCONNECTIONS_ACCURATE_SIM" && \
    DEF2="-DHLS_CATAPULT" && \
    $CMAKE_DIR/bin/cmake $FLEX_SIM_DIR \
      -DCMAKE_PREFIX_PATH=$BUILD_PREF \
      -DCMAKE_CXX_STANDARD=11 \
      -DCMAKE_CXX_COMPILER=g++-5 \
      -DCMAKE_CXX_FLAGS="$HEADER0 $HEADER1 $HEADER2 $HEADER3 $HEADER4 $DEF0 $DEF1 $DEF2" && \
    make -j"$(nproc)"
# RUN cp flex $BUILD_PREF/bin/flexnlp_ila_sim_driver

# asm_sim_driver
WORKDIR $FLEX_SIM_DIR/build
RUN cp $SIM_TEST_DIR/flexnlp/sim_driver/asm_sim_driver.cc $FLEX_SIM_DIR/app/main.cc
RUN make -j"$(nproc)"
RUN cp flex $BUILD_PREF/bin/flex_asm_sim_driver

# adpfloat_to_float
WORKDIR $SIM_TEST_DIR/flexnlp/sim_driver/tool
RUN g++-5 adpfloat_to_float.cc -o adpfloat_to_float \
    -I/root/3laEnv/include \
    -I/root/3la_sim_testbench/ac_include \
    -I/root/FlexNLP/cmod/include \
    -I/root/FlexNLP/matchlib/cmod/include \
    -I/root/FlexNLP/matchlib/rapidjson/include \
    -I/root/FlexNLP/matchlib/connections/include \
    -DSC_INCLUDE_DYNAMIC_PROCESSES -DCONNECTIONS_ACCURATE_SIM -DHLS_CATAPULT \
    -std=c++11 -lstdc++ -lsystemc -lm -lpthread \
    -L/root/3laEnv/lib
RUN cp adpfloat_to_float $BUILD_PREF/bin/adpfloat_to_float

# float_to_adpfloat
WORKDIR $SIM_TEST_DIR/flexnlp/sim_driver/tool
RUN g++-5 float_to_adpfloat.cc -o float_to_adpfloat \
    -I/root/3laEnv/include \
    -I/root/3la_sim_testbench/ac_include \
    -I/root/FlexNLP/cmod/include \
    -I/root/FlexNLP/matchlib/cmod/include \
    -I/root/FlexNLP/matchlib/rapidjson/include \
    -I/root/FlexNLP/matchlib/connections/include \
    -I/root/3la_sim_testbench/tool/numcpp/include \
    -DSC_INCLUDE_DYNAMIC_PROCESSES -DCONNECTIONS_ACCURATE_SIM -DHLS_CATAPULT \
    -std=c++11 -lstdc++ -lsystemc -lm -lpthread \
    -L/root/3laEnv/lib
RUN cp float_to_adpfloat $BUILD_PREF/bin/float_to_adpfloat

##
## Deployment
##
FROM ubuntu:bionic

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
    libedit-dev \
    libgtest-dev \
    libopenblas-dev \
    libssl-dev \
    libtinfo-dev \
    libxml2-dev \
    libz-dev \
    lsb-release \
    pkg-config \
    python3.7 \
    python3.7-dev \
    python3.7-distutils \
    software-properties-common \
    sudo \
    unzip \
    wget \
    zlib1g-dev

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

# Need to ensure 18.04 will use 3.7
# Copied from https://stackoverflow.com/a/58562728
# Makes weird stuff happen in the LLVM install so leaving until later
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN curl -s https://bootstrap.pypa.io/pip/3.7/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

# setup env
ENV VIRTUAL_ENV 3laEnv
ENV BUILD_PREF /root/$VIRTUAL_ENV
#COPY --from=ilabuilder $BUILD_PREF/pyvenv.cfg $BUILD_PREF/pyvenv.cfg
COPY --from=ilabuilder $BUILD_PREF/bin $BUILD_PREF/bin
COPY --from=ilabuilder $BUILD_PREF/lib $BUILD_PREF/lib
COPY --from=ilabuilder /root/3la_ILA_tensor_op /root/3la_ILA_tensor_op

# init
WORKDIR /root
ENV PY_3LA_DRIVER=/root/3la_ILA_tensor_op/
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/root/3laEnv/lib"
ENV PATH="/root/3laEnv/bin/:$PATH"
# RUN echo "source /root/$VIRTUAL_ENV/bin/activate" >> init.sh
# RUN echo "export PYTHONPATH=/root/3la-tvm/python:${PYTHONPATH}" >> init.sh
# RUN echo "export PY_3LA_DRIVER=/root/3la_ILA_tensor_op/" >> init.sh
# RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/3laEnv/lib" >> init.sh
# CMD echo "run 'source init.sh' to start" && /bin/bash
#RUN . /root/$VIRTUAL_ENV/bin/activate

# Set up MLPerf inference. (Downloads take a while, so this should be up near
# the top.)
WORKDIR /root
ADD ./inference ./inference
WORKDIR /root/inference/language/bert/
RUN mkdir build
RUN	cp ../../mlperf.conf build/mlperf.conf
# Make wget quiet so as to not spam the output.
RUN echo "verbose = off" >> /root/.wgetrc
# We don't use BERT, so disable this.
# RUN make download_data
# RUN make download_model

# Install Boost manually to get latest version.
WORKDIR /root
RUN wget -q https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.gz \
    && tar xf boost_1_79_0.tar.gz \
    && cd boost_1_79_0 \
    && ./bootstrap.sh \
    && ./b2 install

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
RUN echo 'set(USE_LLVM "$ENV{LLVM_CONFIG_PATH} --ignore-libllvm")' >> config.cmake && \
  echo 'set(USE_RPC ON)' >> config.cmake && \
  echo 'set(USE_SORT ON)' >> config.cmake && \
  echo 'set(USE_GRAPH_RUNTIME ON)' >> config.cmake && \
  echo 'set(USE_BLAS openblas)' >> config.cmake && \
  echo 'set(CMAKE_CXX_STANDARD 14)' >> config.cmake && \
  echo 'set(CMAKE_CXX_STANDARD_REQUIRED ON)' >> config.cmake && \
  echo 'set(CMAKE_CXX_EXTENSIONS OFF)' >> config.cmake && \
  echo 'set(USE_VTA_FSIM OFF)' >> config.cmake && \
  echo 'set(USE_ILAVTA_CODEGEN ON)' >> config.cmake && \
  echo 'set(USE_ILAFLEX_CODEGEN ON)' >> config.cmake && \
  echo 'set(USE_ILACNN_CODEGEN ON)' >> config.cmake && \
  bash -c \
     "mkdir -p build && \
     cd build && \
     cmake .. && \
     make -j$(nproc)"
#RUN echo 'set(CMAKE_BUILD_TYPE Debug)' >> config.cmake
#ARG TVM_BUILD_JOBS=2

# Help the system find the libtvm library and TVM Python library
ENV TVM_HOME=/root/tvm
ENV VTA_HW_PATH=$TVM_HOME/3rdparty/vta-hw
ENV PYTHONPATH="$TVM_HOME/python:$TVM_HOME/vta/python:$TVM_HOME/topi/python:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TVM_HOME/build/"

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
WORKDIR /root/flexmatch/flexmatch
ENV FLEXMATCH_HOME=/root/flexmatch
RUN --mount=type=ssh cargo build

# need to have flexmatch in the environment
ENV PYTHONPATH="/root/flexmatch/:${PYTHONPATH}"
WORKDIR /root

WORKDIR /root
ADD ./models ./models
ADD ./count_ops.py ./count_ops.py
ADD ./op_summary.py ./op_summary.py
ADD ./diffing_models_from_different_frameworks ./diffing_models_from_different_frameworks
