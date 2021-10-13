#!/bin/sh

# Test the 3LA version of Glenside.
cargo test --manifest-path glenside/Cargo.toml --no-default-features --features "tvm"

# Test import of BERT into Relay.
python3 bert_onnx.py