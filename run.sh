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
# TODO(@gussmith23) Lowering num-images to 1. We should have a "long" and
# "short" setting.
cd e2e/resmlp
python3 trial.py --num-images 1 --use-accelerators
python3 digest.py
cd ../..
