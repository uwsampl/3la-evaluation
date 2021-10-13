#!/bin/sh

# Test the 3LA version of Glenside.
cd glenside && cargo test --no-default-features --features "tvm"