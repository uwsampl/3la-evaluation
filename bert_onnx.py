import tvm
from tvm import relay
import onnx

path = "/root/inference/language/bert/build/data/bert_tf_v1_1_large_fp32_384_v2/model.onnx"
onnx_model = onnx.load(path)

mod, params = relay.frontend.from_onnx(onnx_model, {})

print(mod)
