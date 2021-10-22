#!/bin/sh
set -e

# Test the exact matcher in TVM.
# TODO(@gussmith23) Add more invocations of TVM tests.
TVM_FFI=ctypes python3 -m pytest -v tvm/tests/python/relay/test_exact_matcher.py

# Test the 3LA version of Glenside.
cargo test --manifest-path glenside/Cargo.toml --no-default-features --features "tvm"

# Test import of BERT into Relay.
python3 bert_onnx.py

# Test end-to-end run of ResMLP
cd e2e/resmlp
python3 trial.py --num-images 5 --use-accelerators
python3 digest.py
cd ../..
