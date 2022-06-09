"""
Adapted from the word language model in pytorch/examples
https://github.com/pytorch/examples/blob/master/word_language_model/
"""
import argparse
import math

import tvm
from tvm import relay
from tvm import runtime

# from tvm.contrib import graph_executor

import numpy as np
import torch

import data
from data import Corpus

BATCH_SIZE = 1
BPTT = 35

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(torch.device("cpu"))


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def compile_into_tvm(mod):
    exe = relay.vm.compile(mod, "llvm")
    vm = runtime.vm.VirtualMachine(exe, tvm.cpu(0))
    return vm


def execute_tvm_model(vm, relay_params, data):
    params = {k: v for k, v in relay_params.items()}
    params["data"] = data.numpy().astype("int64")
    params["hidden"] = (
        np.zeros((1, BATCH_SIZE, 128)).astype("float32"),
        np.zeros((1, BATCH_SIZE, 128)).astype("float32")
    )
    ret = vm.invoke("main", **params)
    return torch.from_numpy(ret[0].asnumpy())


def execute_torch_model(model, data):
    hidden = model.init_hidden(BATCH_SIZE)
    with torch.no_grad():
        output, hidden = model(data, hidden)
        hidden = repackage_hidden(hidden)
    return output


def compute_perplexity(outputs, targets):
    criterion = torch.nn.NLLLoss()
    total_loss = 0.0
    for i in range(len(outputs)):
        total_loss += len(outputs[i]) * criterion(outputs[i], targets[i]).item()
    return math.exp(total_loss / (len(outputs) - 1))


def get_batch(source, i):
    seq_len = min(BPTT, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def main(num_batches, base_prog_filename, annotated_prog_filename, torch_filename, params_filename, use_accelerators):
    # copying the testing settings used in the word_language_model example in pytorch/examples
    corpus = Corpus("./data/wikitext-2")
    val_data = batchify(corpus.valid, BATCH_SIZE)

    # torch_model = torch.load(torch_filename, map_location=torch.device("cpu"))

    with open(base_prog_filename, "r") as fp:
        base_relay_mod = tvm.parser.fromtext(fp.read())
    with open(params_filename, "rb") as fp:
        relay_params = relay.load_param_dict(fp.read())

    base_relay = compile_into_tvm(base_relay_mod)
    accel_relay = None
    if use_accelerators:
        with open(annotated_prog_filename, "r") as fp:
            annotated_relay_mod = tvm.parser.fromtext(fp.read())
        accel_relay = compile_into_tvm(annotated_relay_mod)

    base_loss = 0.0
    accel_loss = 0.0
    torch_loss = 0.0

    criterion = torch.nn.NLLLoss()
    for i in range(0, num_batches*BPTT, BPTT):
        data, target = get_batch(val_data, i)
        # torch_out = execute_torch_model(torch_model, data)
        base_out = execute_tvm_model(base_relay, relay_params, data)

        # torch_loss += len(data) * criterion(torch_out, target).item()
        base_loss += len(data) * criterion(base_out, target).item()

        if use_accelerators:
            accel_out = execute_tvm_model(accel_relay, relay_params, data)
            accel_loss += len(accel_out) * criterion(accel_out, target).item()

    # print(f"Total Torch loss: {torch_loss}")
    print(f"Total Relay loss: {base_loss}")
    if use_accelerators:
        print(f"Total accelerator loss: {accel_loss}")

    # torch_perp = math.exp(torch_loss / (num_batches*BPTT - 1))
    base_perp = math.exp(base_loss / (num_batches*BPTT - 1))
    # print(f"Torch perplexity: {torch_perp}")
    print(f"Default Relay perplexity: {base_perp}")

    if use_accelerators:
        accel_perp = math.exp(accel_loss / (num_batches*BPTT - 1))
        print(f"Relay with accelerators perplexity: {accel_perp}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-batches", type=int, default=10, help="Must be >1")
    parser.add_argument("--base-model", type=str, default="lstm_model.relay")
    parser.add_argument("--annotated-model", type=str, default="annotated_lstm.relay")
    parser.add_argument("--params-file", type=str, default="lstm_model.params")
    parser.add_argument("--torch-model", type=str, default="lstm_model.pt")
    parser.add_argument("--use-accelerators", action="store_true")
    args = parser.parse_args()

    if args.n_batches <= 1:
        print("Num batches must be >1")
        exit(1)

    main(args.n_batches, args.base_model, args.annotated_model, args.torch_model, args.params_file, args.use_accelerators)
