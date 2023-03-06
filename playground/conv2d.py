import heterocl as hcl
from hcl_mlir.exceptions import APIError
import numpy as np
import torch 
import torch.nn as nn

hcl.init()
A = hcl.placeholder((6, 6), "A")
F = hcl.placeholder((3, 3), "F")

def conv2d(A, F):
    r = hcl.reduce_axis(0, 3)
    c = hcl.reduce_axis(0, 3)
    return hcl.compute((4, 4),
            lambda y, x: hcl.sum(A[y+r, x+c]*F[r, c], axis=[r, c]), "B")

s = hcl.create_schedule([A, F], conv2d)


# input = torch.randn(6, 6)
input = np.random.randint(10, size=(6, 6)).astype(np.float32)
weight = torch.ones(1, 1, 3, 3)
m = nn.Conv2d(1, 1, 3, stride=1, padding=0, bias=False)
m.weight.data = weight

# start of new
# hcl.init()
# I = hcl.placeholder((3, 3, 32, 32), "I")
# W = hcl.placeholder((64, 3, 3, 3), "W")

# I = hcl.placeholder((32, 32), "I")
# W = hcl.placeholder((3, 3), "W")

# def conv2d(I, W):
#     r = hcl.reduce_axis(0, 3)
#     c = hcl.reduce_axis(0, 3)
#     dout_r = hcl.reduce_axis(0, 3)
#     dout_c = hcl.reduce_axis(0, 64)
#     return hcl.compute((3, 64, 30, 30), 
#             lambda i, j, y, x: hcl.sum(I[i, j, y+r, x+c]*W[dout_r, dout_c, r, c], axis=[dout_r, dout_c, r,c]), "O")

# s = hcl.create_schedule([I, W], conv2d)

# f = hcl.build(s)

# input = torch.randn(3, 3, 32, 32)
# weight = torch.ones(64, 3, 3, 3)

# m = nn.Conv2d(3, 64, 3, stride=1, padding=0)
# m.weight.data = weight

# O = np.zeros((3, 64, 30, 30), dtype=np.float32)
print("weight")
print(m.weight.data)
kernel = m.weight.data.detach().numpy()[0,0,...]
print("input")
print(input)
print("kernel")
print(kernel)
hcl_I = hcl.asarray(input)
hcl_W = hcl.asarray(kernel)
hcl_O = hcl.asarray(np.zeros((4, 4)))
print("I")
print(hcl_I.asnumpy())
print("W")
print(hcl_W.asnumpy())
f = hcl.build(s)
f(hcl_I, hcl_W, hcl_O)
print("O")
print(hcl_O.asnumpy())
print("golden")
golden = m(torch.from_numpy(input)[None, None, ...])
print(golden)
# golden = golden.detach().numpy()[0,0, ...]
# print(np.amax(golden - O))
assert np.allclose(golden.detach().numpy(), hcl_O.asnumpy())
