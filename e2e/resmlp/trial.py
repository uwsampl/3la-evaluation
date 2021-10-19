import argparse
import csv
import os
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


def compile_into_tvm(net):
    trace = get_trace(net)
    input_names = [f"{INPUT_PREFIX}_0"]
    input_shapes = [(f"{INPUT_PREFIX}_0", (BATCH_SIZE, 3, 32, 32))]
    custom_convert_map = 0
    mod, params = relay.frontend.from_pytorch(trace, input_shapes, custom_convert_map)
    for arg in mod["main"].params[: len(input_names)]:
        assert arg.name_hint in input_names

    with tvm.transform.PassContext(opt_level=3):
        relay_graph, relay_lib, relay_params = relay.build(
            mod, target="llvm", params=params
        )
    relay_model = graph_executor.create(relay_graph, relay_lib, tvm.cpu(0))
    relay_model.set_input(**relay_params)
    return relay_model
    # return relay_graph, relay_lib, relay_params


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
    compile_start = time.time()
    relay_model = compile_into_tvm(net)
    compile_end = time.time()
    print(f"Relay compile time: {compile_end - compile_start}")

    if use_accelerators:
        print(f"Accelerator compile time: ")

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
                raise NotImplementedError("Not implemented!")

            pt_relay_diff = torch.abs(torch.flatten(pt_outputs - relay_outputs))
            relay_accel_diff = None
            if use_accelerators:
                raise NotImplementedError("Not implemented")

            numerical_results.append({
                "n_diff_pt_relay": len(pt_relay_diff >= DIFF_THRESHOLD),
                "max_diff_pt_relay": torch.max(pt_relay_diff).item(),
                "n_diff_relay_accel": "None" if not use_accelerators else len(relay_accel_diff >= DIFF_THRESHOLD),
                "max_diff_relay_accel": "None" if not use_accelerators else torch.max(relay_accel_diff).item(),
                "pt_time": pt_time,
                "relay_time": relay_time,
                "accel_time": "None" if not use_accelerators else 0
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

"""
dataiter = iter(testloader)
images, labels = dataiter.next()
# need an input in order to do a trace
trace = torch.jit.trace(net, [images])
if isinstance(net, torch.nn.Module):
    trace = trace.float().eval()

# copied directly from the ResMLP test
baseline_input = [images]
with torch.no_grad():
    baseline_outputs = (net(images.clone()).numpy(), )
custom_convert_map = {}

input_names = ["input{}".format(idx) for idx, inp in enumerate(baseline_input)]
input_shapes = list(zip(input_names, [inp.shape for inp in baseline_input]))
mod, params = relay.frontend.from_pytorch(trace, input_shapes, custom_convert_map)
for arg in mod["main"].params[: len(input_names)]:
    assert arg.name_hint in input_names
compiled_input = dict(
    zip(input_names, [inp.clone().cpu().numpy() for inp in baseline_input])
)


with tvm.transform.PassContext(opt_level=3):
    for target, dev in tvm.testing.enabled_targets():
        relay_graph, relay_lib, relay_params = relay.build(
            mod, target=target, params=params
        )
        relay_model = graph_executor.create(relay_graph, relay_lib, dev)
        relay_model.set_input(**relay_params)
        for name, inp in compiled_input.items():
            relay_model.set_input(name, inp)
        relay_model.run()

        for i, baseline_output in enumerate(baseline_outputs):
            compiled_output = relay_model.get_output(i).asnumpy()

            assert_shapes_match(baseline_output, compiled_output)
            tvm.testing.assert_allclose(
                baseline_output, compiled_output, rtol=1e-5, atol=1e-5
            )
"""
