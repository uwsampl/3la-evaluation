from mxnet.gluon.data import dataset
import tvm
from tvm import relay, runtime
from tvm.relay.op.contrib import ilacnn, ilaflex

import mxnet as mx
from gluoncv.model_zoo import get_model

import numpy as np
import subprocess, json, timeit, os

def validate_with_3la(mod, vm, params, task_id):
    """
    Load data from the files and run validation using 3LA compiler stack
    """
    batch_size = 16
    dataset_file_path = "/home/yl29/gpfs/cifar10_dataset_0.2"
    # result_output_path = "/home/yl29/gpfs/cifar10_3la_results_0.2_16bit_wgt"
    # result_output_path = "/home/yl29/gpfs/resnet_3la_results_8bit_wgt_stride_1"
    # result_output_path = "/home/yl29/gpfs/resnet_3la_results_16bit_wgt"
    result_output_path = "/home/yl29/gpfs/3la_e2e_results/resnet_3la_results_8bit_wgt"
    
    subprocess.run(["mkdir", "-p", result_output_path])

    with open(f"{dataset_file_path}/batch_{task_id}.json", "r") as fin:
        jd = json.load(fin)
    data_list = jd["data"]
    outputs = []
    for b in range(batch_size):
        data_1_batch = np.asarray_chkfinite([data_list[b]]).astype("float32")
        params["data"] = tvm.nd.array(data_1_batch)
        args = [params[var.name_hint] for var in mod["main"].params]
        ret = vm.invoke("main", *args).asnumpy().tolist()
        outputs.append(ret[0])
    result_dump = {
        "batch_id": task_id,
        "label": jd["label"],
        "result": outputs,
    }
    with open(f"{result_output_path}/batch_{task_id}_result.json", "w") as fout:
        json.dump(result_dump, fout, indent=4)
    

def run_3la(task_id):
    ctx = [mx.cpu()]
    net = get_model("cifar_resnet20_v2", classes=10, pretrained=True)
    # export model to relay
    mod, params = relay.frontend.from_mxnet(net, {"data": (1, 3, 32, 32)})
    print(mod)
    print("successfully import model to relay!")
    # annotated for 3la backends
    mod["main"] = ilacnn.remove_padding(mod["main"])
    ilacnn_pt = ilacnn.pattern_table()
    ilaflex_pt = ilaflex.pattern_table()
    mod = tvm.relay.transform.MergeComposite(ilacnn_pt)(mod)
    mod = tvm.relay.transform.MergeComposite(ilaflex_pt)(mod)
    mod = tvm.relay.transform.AnnotateTarget(["ilacnn", "ilaflex"])(mod)
    mod = tvm.relay.transform.PartitionGraph()(mod)

    with tvm.transform.PassContext(opt_level=3):
        device = tvm.cpu()
        target = "llvm"
        exe = relay.vm.compile(mod, target)
        vm = runtime.vm.VirtualMachine(exe, device)
        start_time = timeit.default_timer()
        validate_with_3la(mod, vm, params, task_id)
        end_time = timeit.default_timer()

        print(f"validation time for 1 batch: {end_time - start_time:.2}s")


if __name__ == "__main__":
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    run_3la(task_id)


