'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


"""
Testing MobileNetV2
"""

from typing_extensions import OrderedDict

import torch
import numpy as np
import tvm
from tvm import relay
from tvm import runtime
from tvm.contrib import graph_executor
from tvm.relay.op.contrib import ilacnn, ilaflex
from tvm.relay.testing import annotate_exact_matches

import json, subprocess, os
from fxp_convert import quantize_

def quantize_conv2d_hlscnn(net):
    for name, m in net.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            if m.groups == 1:
                # 8 bit weight quantization
                # m.weight = nn.Parameter(quantize_(m.weight, 2, 6))
                # 16 bit weight quantization
                m.weight = nn.Parameter(quantize_(m.weight, 2, 14))
    return net

def get_trace(net):
    rand_images = torch.from_numpy(
        np.random.randn(1, 3, 32, 32).astype("float32")
    )
    net.eval()
    trace = torch.jit.trace(net, [rand_images])
    if isinstance(net, torch.nn.Module):
        trace = trace.float().eval()
    return trace

def annotate_conv2d_with_exact_matcher(mod):
    def conv2d_layer_definition():
        x = relay.var("input")
        y = relay.var("weight")
        return relay.Function([x, y], relay.nn.conv2d(x, y))

    def non_grouped(conv2d):
        assert isinstance(conv2d, relay.Call)
        if "groups" not in conv2d.attrs.keys():
            return True
        return conv2d.attrs.groups == 1

    conv_pattern = conv2d_layer_definition().body
    mod["main"] = ilacnn.remove_padding(mod["main"])
    mod = relay.transform.InferType()(mod)
    mod["main"] = ilacnn.chan_splitting(mod["main"])
    main_mut = annotate_exact_matches(mod["main"], conv_pattern, "ilacnn", "ilacnn.conv2d", non_grouped)
    mod_mut = tvm.IRModule.from_expr(main_mut)

    # print(mod_mut) 
    return mod_mut


def validate_with_3la(relay_model, task_id, batch_size):
    """
    Load data from the files and run validation using 3LA compiler stack
    """
    # get result output path from the environment
    # result_output_path = "/home/yl29/gpfs/mobilenetv2_3la_results_0.2_16bit_wgt"
    result_output_path = os.environ["RESULT_OUTPUT_PATH"]
    subprocess.run(["mkdir", "-p", result_output_path])
    print(f"result output path is {result_output_path}")
    
    dataset_file_path = "/home/yl29/gpfs/cifar10_dataset_0.2"
    with open(f"{dataset_file_path}/batch_{task_id}.json", "r") as fin:
        jd = json.load(fin)
    data_list = jd["data"]
    outputs = []
    for b in range(batch_size):
        image = np.asarray_chkfinite([data_list[b]]).astype("float32")
        relay_model.set_input(f"{INPUT_PREFIX}_0", image)
        relay_model.run()
        ret = relay_model.get_output(0).asnumpy().tolist()
        outputs.append(ret[0])
    result_dump = {
        "batch_id": task_id,
        "label": jd["label"],
        "result": outputs,
    }
    with open(f"{result_output_path}/batch_{task_id}_result.json", "w") as fout:
        json.dump(result_dump, fout, indent=4)


def run_3la(net, task_id, batch_size):
    # export model to relay

    # # quantize the conv2d weights of the model
    # net = quantize_conv2d_hlscnn(net)

    trace = get_trace(net)    
    input_names = [f"{INPUT_PREFIX}_0"]
    input_shapes = [(f"{INPUT_PREFIX}_0", (1, 3, 32, 32))]
    custom_convert_map = 0
    mod, params = relay.frontend.from_pytorch(trace, input_shapes, custom_convert_map)
    for arg in mod["main"].params[: len(input_names)]:
        assert arg.name_hint in input_names
    # BYOC doesn't work here    
    # # annotated for 3la backends
    # mod = relay.transform.InferType()(mod)
    # mod["main"] = ilacnn.chan_splitting(mod["main"])
    # mod["main"] = ilacnn.remove_padding(mod["main"])
    # ilacnn_pt = ilacnn.pattern_table()
    # mod = tvm.relay.transform.MergeComposite(ilacnn_pt)(mod)

    ilaflex_pt = ilaflex.pattern_table()
    mod = annotate_conv2d_with_exact_matcher(mod)
    mod = tvm.relay.transform.MergeComposite(ilaflex_pt)(mod)
    mod = tvm.relay.transform.AnnotateTarget(["ilaflex"])(mod)
    mod = tvm.relay.transform.PartitionGraph()(mod)
    with tvm.transform.PassContext():
        relay_graph, relay_lib, relay_params = relay.build(
            mod, target="llvm", params=params
        )
    relay_model = graph_executor.create(relay_graph, relay_lib, tvm.cpu())
    relay_model.set_input(**relay_params)
    validate_with_3la(relay_model, task_id, batch_size=batch_size)

if __name__ == "__main__":
    # dataset parameters
    batch_size = 16
    
    net = MobileNetV2()
    param_state_dict = torch.load(
        "/home/yl29/slurm_job/test_mobilenet_3la/models/final_mobilenet_cifar10_400_epochs.pth",
        map_location=torch.device("cpu")
    )
    # rename the state_dict
    new_param_state_dict = OrderedDict()
    for k, data in param_state_dict.items():
        new_param_state_dict[k[7:]] = data
    net.load_state_dict(new_param_state_dict)

    INPUT_PREFIX = "input"

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    run_3la(net, task_id, batch_size)
