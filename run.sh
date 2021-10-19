#!/bin/sh

# Test the 3LA version of Glenside.
cargo test --manifest-path glenside/Cargo.toml --no-default-features --features "tvm"

# Test import of BERT into Relay.
python3 bert_onnx.py

# Test end-to-end run of ResMLP
pushd e2e/resmlp
python3 trial.py --num-images 5
python3 digest.py
popd
