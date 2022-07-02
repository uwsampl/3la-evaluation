import sys
import tvm
import os
import pathlib
from tvm import relay
import tvm.relay.testing
from op_summary import count_all_ops, count_all_overloads, count_all_ops_in_overloads
from e2e.resmlp.trial import import_into_relay
from e2e.resmlp.trial import init_net
from tvm.relay.testing.exact_matcher import deduplicate_vars, check_compiler_call
from tvm.relay.testing import annotate_exact_matches


def linear_body(data, weight, bias):
    return relay.nn.bias_add(relay.nn.dense(data, weight), bias)


def linear_layer_definition():
    input_var = relay.Var("a")
    weight_var = relay.Var("b")
    bias_var = relay.Var("c")
    return relay.Function([input_var, weight_var, bias_var],
                          linear_body(input_var, weight_var, bias_var))


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ENET_DIR = os.path.join(TEST_DIR, "models/efficientnet/EfficientNet")
PARAMS_FILE = os.path.join(
    ENET_DIR, "0.3358-imagenet-efficientnet-b0-47-best.params")


def callback(expr):
    assert isinstance(expr, relay.Call)
    assert expr.op.name == "nn.conv2d"
    # print(expr)
    # print('\n\n\n')
    if "groups" not in expr.attrs.keys():
        return True
    return expr.attrs.groups == 1


def flexasr_pattern(mod):
    linear_pattern = linear_layer_definition().body
    main_func = mod["main"]
    match_bias_add_dense = annotate_exact_matches(
        main_func, linear_pattern, "ilaflex", "ilaflex.linear")
    return match_bias_add_dense


def hlscnn_pattern(mod):
    x = relay.Var("x")
    y = relay.Var("y")
    main_func = mod["main"]
    conv2d = relay.Function([x, y], relay.nn.conv2d(x, y))
    match_conv2d = annotate_exact_matches(
        main_func, conv2d.body, "", "", callback=callback)
    return match_conv2d


def vta_pattern(mod):
    main_func = mod["main"]
    x = relay.Var("x")
    y = relay.Var("y")
    dense = relay.Function([x, y], relay.nn.dense(x, y))
    match_dense = annotate_exact_matches(main_func, dense.body, "", "")
    bias_add = relay.Function([x, y], relay.nn.bias_add(x, y))
    match_bias_add = annotate_exact_matches(match_dense, bias_add.body, "", "")
    return match_bias_add


def efficientnet2():
    print("EFFICIENTNET")
    # FlexASR
    with open("./models/efficientnet/efficientnet.relay", "r") as fp:
        mod = tvm.parser.fromtext(fp.read())
    print("total:", count_all_ops(mod))
    mod["main"] = flexasr_pattern(mod)
    print(count_all_overloads(mod))
    # HLSCNN
    with open("./models/efficientnet/efficientnet.relay", "r") as fp:
        mod = tvm.parser.fromtext(fp.read())
    x = relay.Var("x")
    y = relay.Var("y")
    main_func = mod["main"]
    conv2d = relay.Function([x, y], relay.nn.conv2d(x, y))
    match_conv2d = annotate_exact_matches(
        main_func, conv2d.body, "", "", callback=callback)
    mod["main"] = match_conv2d
    print(count_all_overloads(mod))
    with open("./models/efficientnet/efficientnet.relay", "r") as fp:
        mod = tvm.parser.fromtext(fp.read())
    # vta
    main_func = mod["main"]
    x = relay.Var("x")
    y = relay.Var("y")
    dense = relay.Function([x, y], relay.nn.dense(x, y))
    match_dense = annotate_exact_matches(main_func, dense.body, "", "")
    bias_add = relay.Function([x, y], relay.nn.bias_add(x, y))
    match_bias = annotate_exact_matches(match_dense, bias_add.body, "", "")
    mod["main"] = match_bias
    print(count_all_overloads(mod))


def mobilenetv2():
    print("MOBILENET V2")
    with open("./models/mobilenetv2/mobilenet.relay", "r") as fp:
        mod = tvm.parser.fromtext(fp.read())
    mod = relay.transform.SimplifyInference()(mod)
    print("total:", count_all_ops(mod))
    # FlexASR
    linear_pattern = linear_layer_definition().body
    main_func = mod["main"]
    mod["main"] = annotate_exact_matches(
        main_func, linear_pattern, "ilaflex", "ilaflex.linear")
    print(count_all_overloads(mod))
    # HLSCNN
    with open("./models/mobilenetv2/mobilenet.relay", "r") as fp:
        mod = tvm.parser.fromtext(fp.read())
    mod = relay.transform.SimplifyInference()(mod)
    x = relay.Var("x")
    y = relay.Var("y")
    main_func = mod["main"]
    conv2d = relay.Function([x, y], relay.nn.conv2d(x, y))
    match_conv2d = annotate_exact_matches(
        main_func, conv2d.body, "", "", callback=callback)
    mod["main"] = match_conv2d
    print(count_all_overloads(mod))
    # VTA
    with open("./models/mobilenetv2/mobilenet.relay", "r") as fp:
        mod = tvm.parser.fromtext(fp.read())
    mod = relay.transform.SimplifyInference()(mod)
    main_func = mod["main"]
    x = relay.Var("x")
    y = relay.Var("y")
    dense = relay.Function([x, y], relay.nn.dense(x, y))
    match_dense = annotate_exact_matches(main_func, dense.body, "", "")
    bias_add = relay.Function([x, y], relay.nn.bias_add(x, y))
    match_bias = annotate_exact_matches(match_dense, bias_add.body, "", "")
    mod["main"] = match_bias
    print(count_all_overloads(mod))


def resmlp():
    print("RESMLP")
    # This is a hack to make the ResMLP model load correctly.
    sys.path.append(os.path.join(os.path.dirname(__file__), 'e2e', 'resmlp'))
    net = init_net("./e2e/resmlp/cifar_net.pth")
    # with open("./models/res_mlp/resmlp.relay", "r") as fp:
    #     mod = tvm.parser.fromtext(fp.read()
    mod, params = import_into_relay(net)
    print("total:", count_all_ops(mod))
    mod["main"] = flexasr_pattern(mod)
    print(count_all_overloads(mod))

    mod, params = import_into_relay(net)
    mod["main"] = hlscnn_pattern(mod)
    print(count_all_overloads(mod))

    mod, params = import_into_relay(net)
    mod["main"] = vta_pattern(mod)
    print(count_all_overloads(mod))


def resnet20():
    print("RESNET")
    with open("./models/resnet20/resnet20.relay", "r") as fp:
        glob = fp.read()
        mod = tvm.parser.fromtext(glob)
    mod = relay.transform.SimplifyInference()(mod)
    print("total:", count_all_ops(mod))
    mod["main"] = flexasr_pattern(mod)
    print(count_all_ops_in_overloads(mod))

    with open("./models/resnet20/resnet20.relay", "r") as fp:
        mod = tvm.parser.fromtext(fp.read())
    mod["main"] = hlscnn_pattern(mod)
    print(count_all_ops_in_overloads(mod))

    with open("./models/resnet20/resnet20.relay", "r") as fp:
        mod = tvm.parser.fromtext(fp.read())
    mod = relay.transform.SimplifyInference()(mod)
    mod["main"] = vta_pattern(mod)
    print(count_all_ops_in_overloads(mod))


def transformer():
    print("TRANSFORMER")
    with open("./models/transformer/transformer.relay", "r") as fp:
        mod = tvm.parser.fromtext(fp.read())
    print("total:", count_all_ops(mod))
    mod["main"] = flexasr_pattern(mod)
    print(count_all_overloads(mod))
    with open("./models/transformer/transformer.relay", "r") as fp:
        mod = tvm.parser.fromtext(fp.read())
    mod["main"] = hlscnn_pattern(mod)
    print(count_all_overloads(mod))
    with open("./models/transformer/transformer.relay", "r") as fp:
        mod = tvm.parser.fromtext(fp.read())
    mod["main"] = vta_pattern(mod)
    print(count_all_overloads(mod))


def lstm2():
    print("LSTM")
    with open("./models/lstm/lstm_model.relay", "r") as fp:
        mod = tvm.parser.fromtext(fp.read())
    print("total:", count_all_ops(mod))
    with open("./models/lstm/lstm_pattern.relay", "r") as fp:
        lmod = tvm.parser.fromtext(fp.read())
    main_func = mod["main"]
    match_lstm = annotate_exact_matches(main_func, lmod["main"].body, "", "")
    lmod["main"] = match_lstm
    print(count_all_overloads(lmod))
    with open("./models/lstm/lstm_model.relay", "r") as fp:
        mod = tvm.parser.fromtext(fp.read())

    mod["main"] = hlscnn_pattern(mod)
    print(count_all_overloads(mod))
    with open("./models/lstm/lstm_model.relay", "r") as fp:
        mod = tvm.parser.fromtext(fp.read())
    mod["main"] = vta_pattern(mod)
    print(count_all_overloads(mod))


def resnet50_from_different_frameworks():
    for framework in ['tf', 'pytorch', 'onnx']:
        print(f"RESNET50 from {framework}")
        with open(pathlib.Path(__file__).parent.resolve() / "diffing_models_from_different_frameworks" / f"resnet50_simplifyinference_from_{framework}.relay", "r") as fp:
            glob = fp.read()
            mod = tvm.parser.fromtext(glob)
        print("total:", count_all_ops(mod))
        for pattern in [flexasr_pattern, hlscnn_pattern, vta_pattern]:
            mod = tvm.IRModule({'main': pattern(mod)})
            print(count_all_ops_in_overloads(mod))


transformer()
efficientnet2()
lstm2()
mobilenetv2()
resmlp()
resnet20()
resnet50_from_different_frameworks()
