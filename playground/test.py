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
    ch = hcl.reduce_axis(0, 3)
    O = hcl.compute((3, 64, 30, 30), 
        lambda i, j, y, x: hcl.sum(I[i, ch, y+r, x+c]*W[j, ch, r, c],axis=[ch, r,c]), "O")
    return O


s = hcl.create_schedule([I, W], conv2d)
s_O = conv2d.O
x_out, x_in = s[s_O].split(s_O.axis[1], factor=8)
print(s.device_module)
f = hcl.build(s)

input = np.random.randint(10, size=(3, 3, 32, 32)).astype(np.float32)
# weight = torch.ones(64, 3, 3, 3)
weight = np.random.randint(10, size=(64, 3, 3, 3)).astype(np.float32)

m = nn.Conv2d(3, 64, 3, stride=1, padding=0, bias=False)
m.weight.data = torch.from_numpy(weight)
golden = m(torch.from_numpy(input))

print("weight")
print(m.weight.data.detach().numpy())

hcl_I = hcl.asarray(input)
hcl_W = hcl.asarray(m.weight.data.detach().numpy())
hcl_O = hcl.asarray(np.zeros((3, 64, 30, 30)))
f(hcl_I, hcl_W, hcl_O)

print("hcl")
print(hcl_O.asnumpy())

print("golden")
print(golden.detach().numpy())


assert np.allclose(golden.detach().numpy(), hcl_O.asnumpy())