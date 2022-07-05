"""Legalize variable names for Relay modules imported from other frameworsk. 

Turn this into a script, or just edit the vars below."""
import re
import tvm
from tvm import relay

FILENAME = "resnet50_simplifyinference_from_tf.relay"
MAIN_FUNC_DEF_LINE = 2

file = open(FILENAME).read()
funcdef = file.splitlines()[MAIN_FUNC_DEF_LINE]


def f(code, find, replacewith):
    return code.replace(find, replacewith)


newcode = file
matches = re.findall(r'%[:a-zA-Z_\./0-9]*: ', funcdef)
for match in matches:
    newcode = f(newcode, match[:-2],
                match[:-2].replace('/', '_').replace('.', '_').replace(':', '_'))

tvm.parser.fromtext(newcode)
mod = tvm.parser.fromtext(newcode)
mod = relay.transform.InferType()(mod)

print(mod)
