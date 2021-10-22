import argparse
import csv
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
    net.load_state_dict(torch.load(params_path))
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
    with tvm.transform.PassContext(opt_level=3):
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
    # copied from flexmatch/demo/get_relay_model to avoid tight coupling with a demo script
    class RenameMutator(ExprMutator):
        def __init__(self):
            super().__init__()
            self.var_map = {}

        def visit_var(self, var):
            if var in self.var_map:
                return self.var_map[var]

            if "." in var.name_hint:
                new_name = var.name_hint.replace(".", "_")
                new_var = relay.Var(new_name, type_annotation=var.type_annotation)
                self.var_map[var] = new_var
                return new_var
            return var

    mutator = RenameMutator()
    mod["main"] = mutator.visit(mod["main"])
    # restore type annotations
    mod = relay.transform.InferType()(mod)
    # dump to glenside/models/resmlp.relay
    # TODO: Don't use a hardcoded directory in the test file...
    glenside_home = os.environ["GLENSIDE_HOME"]
    with open(os.path.join(glenside_home, "models", "resmlp.relay"), "w") as fp:
        fp.write(mod.astext())

    # now we invoke the glenside resmlp test to apply rewrites
    start = time.time()
    subprocess.run(["cargo", "test", "test_resmlp"], cwd=glenside_home)
    end = time.time()
    print(f"Glenside total time: {end - start}")
    result_file = os.path.join(glenside_home, "models", "resmlp_dump.json")
    if not os.path.exists(result_file):
        raise Exception("No rewrite results given")

    # now we take the JSON dump and compile it back into Relay
    with open(result_file, "r") as fp:
        recexpr_json = json.load(fp)

    compiler = RecExprCompiler({
        "flex-linear": "ilaflex.linear"
    }, {
        "flex-linear": "ilaflex"
    })
    shape_dict = {}
    for arg in mod["main"].params:
        shape_dict[arg.name_hint] = tuple(arg.type_annotation.shape)

    start = time.time()
    expr = compiler.to_relay_expr(recexpr_json, shape_dict)
    mod = tvm.ir.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    end = time.time()
    print(f"RecExpr to Relay conversion time: {end-start}")
    return mod, params


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


def compare_on_data(net, testloader, num_images, use_accelerators):
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
                relay_accel_diff = torch.abs(torch.flatten(accel_outpus - relay_outputs))

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


def main(params_path, num_images, use_accelerators, shuffle):
    testloader = load_data(shuffle=True)
    net = init_net(params_path)
    compare_on_data(net, testloader, num_images, use_accelerators)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--use-accelerators", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--params-path", type=str, default="./cifar_net.pth")
    args = parser.parse_args()

    main(args.params_path, args.num_images, args.use_accelerators, not args.no_shuffle)
