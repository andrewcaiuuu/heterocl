import heterocl as hcl
from hcl_mlir.exceptions import APIError
import numpy as np
import torch 
import torch.nn as nn

hcl.init()
I = hcl.placeholder((3, 3, 32, 32), "I")
W = hcl.placeholder((64, 3, 3, 3), "W")

def conv2d(I, W):
    r = hcl.reduce_axis(0, 3)
    c = hcl.reduce_axis(0, 3)
    dout_r = hcl.reduce_axis(0, 3)
    dout_c = hcl.reduce_axis(0, 64)
    return hcl.compute((3, 64, 30, 30), 
            lambda i, j, y, x: hcl.sum(I[i, j, y+r, x+c]*W[dout_r, dout_c, r, c], axis=[dout_r, dout_c, r,c]), "O")

s = hcl.create_schedule([I, W], conv2d)
print(s.device_module)
f = hcl.build(s)

# input = torch.randn(3, 3, 32, 32)
input = np.random.randint(10, size=(3, 3, 32, 32)).astype(np.float32)
weight = torch.ones(64, 3, 3, 3)

m = nn.Conv2d(3, 64, 3, stride=1, padding=0, bias=False)
m.weight.data = weight
# print("weight")
# print(m.weight.data)

hcl_I = hcl.asarray(input)
hcl_W = hcl.asarray(m.weight.data.detach().numpy())
hcl_O = hcl.asarray(np.zeros((3, 64, 30, 30)))
f(hcl_I, hcl_W, hcl_O)
print("result")
print(hcl_O.asnumpy())
print("golden")
golden = m(torch.from_numpy(input))
print(golden)
assert np.allclose(golden.detach().numpy(), hcl_O.asnumpy())
