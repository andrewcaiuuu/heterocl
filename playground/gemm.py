import heterocl as hcl
from hcl_mlir.exceptions import APIError
import numpy as np

# 0. Initialize the data type
#    If `hcl.init` is not called, hcl.Int() will be used
hcl.init(hcl.Float())

# 1. Declare placeholders for the input
A = hcl.placeholder((32, 32), "A")
B = hcl.placeholder((32, 32), "B")

# 2. Declare the algorithm
def gemm(A, B):
    k = hcl.reduce_axis(0, 32, "k")
    C = hcl.compute((32, 32), lambda i, j:
            hcl.sum(A[i, k] * B[k, j], axis=k), "C")
    return C

# 3. Create schedule
s = hcl.create_schedule([A, B], gemm)
op_C = gemm.C
x_out, x_in = s[op_C].split(op_C.axis[0], factor=8)
y_out, y_in = s[op_C].split(op_C.axis[1], factor=8)
s[op_C].reorder(x_out, y_out, x_in, y_in)
f = hcl.build(s)
print(s.device_module)

# 5. Execute the module
#    HeteroCL needs to use the destination passing style,
#    i.e., the output of the function is passed as the last argument
A = np.random.randint(10, size=(32, 32)).astype(np.float32)
B = np.random.randint(10, size=(32, 32)).astype(np.float32)
# print(A)
C = np.zeros((32, 32), dtype=np.float32)
hcl_A = hcl.asarray(A)
hcl_B = hcl.asarray(B)
hcl_C = hcl.asarray(C)
f(hcl_A, hcl_B, hcl_C)
golden = np.matmul(A, B)
assert np.allclose(golden, hcl_C.asnumpy())