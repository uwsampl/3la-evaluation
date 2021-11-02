import argparse
import csv
import json
import os
import subprocess
import time

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from model import ResMLP

import tvm
from tvm import relay
import tvm.testing
from tvm.contrib import graph_executor
from tvm.relay import ExprMutator

from megraph.language import RecExprCompiler

# also adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
INPUT_PREFIX = "input"
BATCH_SIZE = 32
DIFF_THRESHOLD = 1e-5

def load_data(shuffle=True):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=shuffle, num_workers=2)
    return testloader


def init_net(params_path):
    net = ResMLP(image_size=32,
                 patch_size=16,
                 dim=512,
                 depth=12,
                 num_classes=len(CLASSES))
    if not os.path.exists(params_path):
        raise Exception("Missing trained model!")
    net.load_state_dict(torch.load(params_path, map_location=torch.device('cpu')))
    return net


def get_trace(net):
    # need an input in order to do a trace.
    # in this case, we prepare a random image of the right size
    rand_images = torch.from_numpy(
        np.random.randn(BATCH_SIZE, 3, 32, 32).astype("float32")
    )
    trace = torch.jit.trace(net, [rand_images])
    if isinstance(net, torch.nn.Module):
        trace = trace.float().eval()
    return trace


def import_into_relay(net):
    trace = get_trace(net)
    input_names = [f"{INPUT_PREFIX}_0"]
    input_shapes = [(f"{INPUT_PREFIX}_0", (BATCH_SIZE, 3, 32, 32))]
    custom_convert_map = {}
    mod, params = relay.frontend.from_pytorch(trace, input_shapes, custom_convert_map)
    for arg in mod["main"].params[: len(input_names)]:
        assert arg.name_hint in input_names
    return mod, params


def compile_into_tvm(mod, params):
    with tvm.transform.PassContext(opt_level=0):
        relay_graph, relay_lib, relay_params = relay.build(
            mod, target="llvm", params=params
        )
    relay_model = graph_executor.create(relay_graph, relay_lib, tvm.cpu(0))
    relay_model.set_input(**relay_params)
    return relay_model


def compile_into_glenside(net):
    mod, params = import_into_relay(net)
    # weirdness due to hardcoded directories in the Glenside ResMLP test

    # need to rename variables with dots in their names
    # also the text format does not like variable names that start with numbers, so let's fix those
    def rename(name):
        new_name = name
        if name[0].isdigit():
            new_name = f"v{new_name}"
        if "." in new_name:
            new_name = new_name.replace(".", "_")
        return new_name

    class RenameMutator(ExprMutator):
        def __init__(self):
            super().__init__()
            self.var_map = {}

        def visit_var(self, var):
            if var in self.var_map:
                return self.var_map[var]

            new_name = rename(var.name_hint)
            if new_name != var.name_hint:
                new_var = relay.Var(new_name, type_annotation=var.type_annotation)
                self.var_map[var] = new_var
                return new_var
            return var

    # rename params accordingly
    new_params = {rename(k): v for k, v in params.items()}

    mutator = RenameMutator()
    mod = tvm.ir.IRModule.from_expr(mutator.visit(mod["main"]))
    # restore type annotations
    mod = relay.transform.InferType()(mod)

    # need to write the model to a file
    flexmatch_home = os.environ["FLEXMATCH_HOME"]
    model_path = os.path.abspath(os.path.join(os.getcwd(), "resmlp.relay"))
    with open(model_path, "w") as fp:
        fp.write(mod.astext())

    # maybe we could make this a proper API to avoid calling Python from in here
    start = time.time()
    flexmatch_tests = os.path.join(flexmatch_home, "tests")
    rewrite_rules = ["linear-rewrites"]
    if use_im2col:
        rewrite_rules.append("im2col-rewrites")
    subprocess.run(["python3", "run_eqsat.py", model_path, "resmlp",
                    *rewrite_rules],
                   cwd=flexmatch_tests)
    rewrites_path = os.path.join(flexmatch_tests, "resmlp-rewritten.json")
    data_path = os.path.join(flexmatch_tests, "resmlp-data.json")
    end = time.time()
    print(f"Glenside search time: {end-start}")

    start = time.time()
    target_path = os.path.join(os.getcwd(), "resmlp-rewritten.relay")
    subprocess.run(["python3", "compile_model.py", model_path, target_path,
                    rewrites_path, data_path,
                    *rewrite_rules],
                   cwd=flexmatch_tests)

    with open(target_path, "r") as fp:
        new_mod_text = fp.read()

    new_mod = tvm.parser.fromtext(new_mod_text)
    end = time.time()
    print(f"RecExpr to Relay conversion time: {end-start}")
    return new_mod, new_params


def execute_tvm_model(relay_model, images):
    relay_model.set_input(f"{INPUT_PREFIX}_0", images)
    relay_model.run()
    return torch.from_numpy(
        relay_model.get_output(0).asnumpy()
    )


def assert_shapes_match(tru, est):
    if tru.shape != est.shape:
        msg = "Output shapes {} and {} don't match"
        raise AssertionError(msg.format(tru.shape, est.shape))


def dump_to_csv(filename, fieldnames, data):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in data:
            writer.writerow(r)


def compare_on_data(net, testloader, num_images, use_accelerators, use_im2col):
    mod, params = import_into_relay(net)
    compile_start = time.time()
    relay_model = compile_into_tvm(mod, params)
    compile_end = time.time()
    print(f"Relay compile time: {compile_end - compile_start}")

    if use_accelerators:
        accel_mod, accel_params = compile_into_glenside(net)
        accel_model = compile_into_tvm(accel_mod, accel_params)

    # numerical results:
    #   number of differing elements (PT vs Relay)
    #   max difference magnitude (PT vs Relay)
    #   number of different elements (Relay vs accelerators)
    #   max different magnitude (Relay vs accelerators)
    #   PT exec time
    #   Relay exec time
    #   Accel time
    numerical_results = []

    # prediction results:
    #        Correct prediction
    #        PT prediction
    #        Relay prediction
    #        Accelerated Prediction (eventually)
    #        PT correct?
    #        Relay correct?
    #        Accelerated correct?
    #        Relay faithful to PT?
    #        Accerated faithful to Relay?
    pred_results = []

    device = torch.device("cpu")

    i = 0
    with torch.no_grad():
        for data in testloader:
            if i >= num_images:
                break

            pt_start = time.time()
            images, labels = data[0].to(device), data[1].to(device)
            pt_outputs = net(images)
            _, pt_preds = torch.max(pt_outputs, 1)
            pt_end = time.time()
            pt_time = pt_end - pt_start

            relay_start = time.time()
            relay_outputs = execute_tvm_model(relay_model, images)
            assert_shapes_match(pt_outputs, relay_outputs)
            _, relay_preds = torch.max(relay_outputs, 1)
            relay_end = time.time()
            relay_time = relay_end - relay_start

            accel_outputs = None
            # using a string None because of CSV serialization
            accel_preds = ["None"] * len(relay_preds)
            if use_accelerators:
                accel_start = time.time()
                accel_outputs = execute_tvm_model(accel_model, images)
                _, accel_preds = torch.max(accel_outputs, 1)
                accel_end = time.time()
                accel_time = accel_end - accel_start

            pt_relay_diff = torch.abs(torch.flatten(pt_outputs - relay_outputs))
            relay_accel_diff = None
            if use_accelerators:
                relay_accel_diff = torch.abs(torch.flatten(accel_outputs - relay_outputs))

            numerical_results.append({
                "n_diff_pt_relay": len(pt_relay_diff >= DIFF_THRESHOLD),
                "max_diff_pt_relay": torch.max(pt_relay_diff).item(),
                "n_diff_relay_accel": "None" if not use_accelerators else len(relay_accel_diff >= DIFF_THRESHOLD),
                "max_diff_relay_accel": "None" if not use_accelerators else torch.max(relay_accel_diff).item(),
                "pt_time": pt_time,
                "relay_time": relay_time,
                "accel_time": "None" if not use_accelerators else accel_time
            })

            for label, pt_pred, relay_pred, accel_pred in zip(labels, pt_preds, relay_preds, accel_preds):
                relay_faithful = (pt_pred == relay_pred).item()
                pt_correct = (pt_pred == label).item()
                relay_correct = (relay_pred == label).item()
                accel_correct = "None" if not use_accelerators else (accel_pred == label).item()
                accel_faithful = "None" if not use_accelerators else (accel_pred == relay_pred).item()
                pred_results.append({
                    "label": int(label),
                    "pt": int(pt_pred),
                    "relay": int(relay_pred),
                    "accel": "None" if not use_accelerators else int(accel_pred),
                    "pt_correct": pt_correct,
                    "relay_correct": relay_correct,
                    "accel_correct": accel_correct,
                    "relay_faithful": relay_faithful,
                    "accel_faithful": accel_faithful
                })
            i += 1

    numerical_fieldnames = ["n_diff_pt_relay", "max_diff_pt_relay", "n_diff_relay_accel", "max_diff_relay_accel", "pt_time", "relay_time", "accel_time"]
    pred_fieldnames = ["label", "pt", "relay", "accel",
                       "pt_correct", "relay_correct", "accel_correct",
                       "relay_faithful", "accel_faithful"]

    dump_to_csv("numerical.csv", numerical_fieldnames, numerical_results)
    dump_to_csv("pred.csv", pred_fieldnames, pred_results)


def main(params_path, num_images, use_accelerators, shuffle, use_im2col):
    testloader = load_data(shuffle=shuffle)
    net = init_net(params_path)
    compare_on_data(net, testloader, num_images, use_accelerators, use_im2col)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--use-accelerators", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--params-path", type=str, default="./cifar_net.pth")
    parser.add_argument("--use-im2col", action="store_true")
    args = parser.parse_args()

    main(args.params_path, args.num_images, args.use_accelerators, not args.no_shuffle, args.use_im2col)
