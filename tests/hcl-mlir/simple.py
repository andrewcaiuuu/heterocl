import heterocl as hcl
import os
import sys
import numpy as np

A = hcl.placeholder((10,), "A")
def kernel(A):
    B = hcl.compute((10,), lambda x: A[x])
    return B

s = hcl.create_schedule([A], kernel)
print(hcl.lower(s))