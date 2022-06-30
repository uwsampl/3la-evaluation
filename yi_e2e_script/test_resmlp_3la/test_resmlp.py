import tvm
from tvm import relay, runtime
from tvm.contrib import graph_executor
from tvm.relay.op.contrib import ilacnn, ilaflex

import numpy as np
import timeit, os, json, subprocess


total_val_data_batch = 125
dataset_file_path = "/home/yl29/gpfs/cifar10_dataset_0.2"
batch_size = 16
INPUT_PREFIX = "input"
result_output_path = "/home/yl29/gpfs/3la_e2e_results/resmlp_3la"

def load_resmlp_from_text(fpath):
    with open(fpath, "r") as fi:
        mod = tvm.parser.fromtext(fi.read())
    print(f"loaded relay model from {fpath}")
    return mod

def validate_with_3la(relay_model, task_id, batch_size):
    """
    Load data from the files and run validation using 3LA compiler stack
    """
    subprocess.run(["mkdir", "-p", result_output_path])
    
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


def run_3la():
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    mod = load_resmlp_from_text("/home/yl29/slurm_job/test_resmlp_3la/models/resmlp-rewritten.relay")
    with open("/home/yl29/slurm_job/test_resmlp_3la/models/resmlp.params", "rb") as fin:
        params = relay.load_param_dict(fin.read())
    with tvm.transform.PassContext(opt_level=3):
        relay_graph, relay_lib, relay_params = relay.build(
            mod, target="llvm", params=params
        )
    relay_model = graph_executor.create(relay_graph, relay_lib, tvm.cpu())
    relay_model.set_input(**relay_params)

    validate_with_3la(relay_model, task_id, batch_size=batch_size)


if __name__ == "__main__":
    run_3la()




