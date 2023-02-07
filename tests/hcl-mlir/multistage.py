import heterocl as hcl
import os
import sys
import numpy as np

A = hcl.placeholder((32, 32), "A")
def kernel(A):
    B = hcl.compute(A.shape, lambda i, j: A[i, j] + 1, "B")
    C = hcl.compute(A.shape, lambda i, j: A[i, j] + 1, "C")
    D = hcl.compute(A.shape, lambda i, j: B[i, j] + C[i, j], "D")
    return D
s = hcl.create_schedule([A], kernel)
print(hcl.lower(s))
# mod = hcl.build(s)
