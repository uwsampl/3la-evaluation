import sys
import tvm
from tvm import relay
from models.end_to_end_speech_to_text import build_relay_module
from models.end_to_end_speech_to_text import load_weights
import op_summary
from op_summary import count_all_ops, count_all_overloads, count_all_ops_in_overloads
import e2e.resmlp
from e2e.resmlp.trial import import_into_relay
from e2e.resmlp.trial import init_net
from e2e.resmlp.linear_rewrite import LinearLayerRewriter
from tvm.relay.testing.exact_matcher import deduplicate_vars, check_compiler_call
from tvm.relay.testing import annotate_exact_matches
#linear layer function
def linear_body(data, weight, bias):
    return relay.nn.bias_add(relay.nn.dense(data, weight), bias)
def linear_layer_definition():
    input_var = relay.Var("a")
    weight_var = relay.Var("b")
    bias_var = relay.Var("c")
    return relay.Function([input_var, weight_var, bias_var],
                          linear_body(input_var, weight_var, bias_var))



def resmlp():
    net = init_net("./e2e/resmlp/cifar_net.pth")
    mod, params = import_into_relay(net)
    # print(count_all_ops(mod))
    #VTA
    rewriter = LinearLayerRewriter()
    main_func = mod["main"]
    x = relay.Var("x")
    y = relay.Var("y")
    dense = relay.Function([x, y], relay.nn.dense(x, y))
    match_dense = annotate_exact_matches(main_func, dense.body, "", "")
    bias_add = relay.Function([x, y], relay.nn.bias_add(x, y))
    match_bias = annotate_exact_matches(match_dense, bias_add.body, "", "")
    mod["main"] = match_bias

    print(count_all_overloads(mod))
    
    print(count_all_ops_in_overloads(mod))

    # FlexASR
    mod, params = import_into_relay(net)
    linear_pattern = linear_layer_definition().body
    main_func = mod["main"] 
    mod["main"]  = annotate_exact_matches(main_func, linear_pattern, "ilaflex", "ilaflex.linear")
    print(count_all_overloads(mod))
    print(count_all_ops_in_overloads(mod))

resmlp()


