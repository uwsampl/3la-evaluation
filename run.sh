#!/bin/sh
set -e

# Use this flag to set the type of eval you want to run.
# 0: fast (minutes)
# 1: medium/nightly (<=2 hours)
# 2: full
# Defaults to full.
export EVAL_TYPE="${EVAL_TYPE:-2}"
[ "$EVAL_TYPE" -eq "0" ] || [ "$EVAL_TYPE" -eq "1" ] || [ "$EVAL_TYPE" -eq "2" ] || exit 1

# Test the exact matcher in TVM.
# TODO(@gussmith23) Add more invocations of TVM tests.
TVM_FFI=ctypes python3 -m pytest -v tvm/tests/python/relay/test_exact_matcher.py

# Test the 3LA version of Glenside.
cargo test --manifest-path glenside/Cargo.toml --no-default-features --features "tvm"

# Test import of BERT into Relay.
# We don't actually use BERT, so we'll disable this for now.
# python3 bert_onnx.py

# Test end-to-end run of ResMLP 
# TODO(@gussmith23) Disabling this for now, as even 1 image doesn't terminate.
# TODO(@gussmith23) Add "fast", "nightly", and "full" eval settings (perhaps in
# the Dockerfile? Perhaps as an env var?)
cd e2e/resmlp
# TODO(@gussmith23 @slyubomirsky) Before re-enabling this, please filter the
# output of the simulator e.g. using grep. It spams the logs.
if [ "$EVAL_TYPE" -ge "1" ]
then
  python3 trial.py --num-images 1 --use-accelerators
fi
# python3 digest.py
cd ../..

#count relay ops in models
python3 count_ops.py

cd flexmatch/tests 
./get_table_stats.sh
cd ../..
