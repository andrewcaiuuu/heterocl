import numpy as np
import heterocl as hcl
import tvm
import os


hcl.config.init_dtype = "float32"


def PreCompute_BitReverse_Pattern(L):
  """
  Parameters
  ----------
  L : int
    Length of the input vector.

  Returns
  -------
  IndexTable : vector of int
    Permutation pattern.
  """
  bit_width = int(np.log2(L))
  IndexTable = np.zeros((L), dtype='int')
  for i in range(L):
    b = '{:0{width}b}'.format(i, width=bit_width)
    IndexTable[i] = int(b[::-1], 2)
  return IndexTable


def build_fft(X_real, X_imag, IndexTable):
  L = X_real.shape[0]
  if np.log2(L) % 1 > 0:
    raise ValueError("Length of input vector (1d tensor) must be power of 2")
  num_stages = int(np.log2(L))

  # bit reverse permutation
  F_real = hcl.compute(X_real.shape, lambda i: X_real[IndexTable[i]], name='F_real')
  F_imag = hcl.compute(X_imag.shape, lambda i: X_imag[IndexTable[i]], name='F_imag')

  def FFT(F_real, F_imag):
    one = hcl.local(1, dtype="int32")
    with hcl.for_(0, num_stages) as stage:
      DFTpts = one[0] << (stage + 1)
      numBF = DFTpts / 2
      e = -2 * np.pi / DFTpts
      a = hcl.local(0)
      with hcl.for_(0, numBF) as j:
        c = hcl.local(tvm.cos(a[0]))
        s = hcl.local(tvm.sin(a[0]))
        a[0] = a[0] + e
        with hcl.for_(j, L + DFTpts - 1, DFTpts) as i:
          i_lower = i + numBF
          temp_r = hcl.local(F_real[i_lower] * c - F_imag[i_lower] * s)
          temp_i = hcl.local(F_imag[i_lower] * c + F_real[i_lower] * s)
          F_real[i_lower] = F_real[i] - temp_r[0]
          F_imag[i_lower] = F_imag[i] - temp_i[0]
          F_real[i] = F_real[i] + temp_r[0]
          F_imag[i] = F_imag[i] + temp_i[0]

  with hcl.stage() as Out:
    FFT(F_real, F_imag)

  s = hcl.create_schedule(Out)
  # print hcl.lower(s, [X_real, X_imag, IndexTable, F_real, F_imag])
  f = hcl.build(s, [X_real, X_imag, IndexTable, F_real, F_imag],
          target='merlinc')
  return f

L = 1024
X_real = hcl.placeholder((L,))
X_imag = hcl.placeholder((L,))
IndexTable = hcl.placeholder((L,), dtype="int32")

code = build_fft(X_real, X_imag, IndexTable)
with open('fft_kernel.cpp', 'w') as f:
    f.write(code)

# Prepare input
data_file = open('input.dat', 'w')
x_real_np = np.random.random((L))
data_file.write('\t'.join([str(n) for n in x_real_np.tolist()]))
data_file.write('\n')
x_imag_np = np.random.random((L))
data_file.write('\t'.join([str(n) for n in x_imag_np.tolist()]))
data_file.write('\n')
x_np = x_real_np + 1j * x_imag_np
data_file.write('\t'.join([str(n) for n in PreCompute_BitReverse_Pattern(L).tolist()]))
data_file.close()

# Prepare reference
out_np = np.fft.fft(x_np)
out_real_np = out_np.real
out_imag_np = out_np.imag

# Here we use gcc to evaluate the functionality
os.system('g++ -std=c++11 fft_host.cpp fft_kernel.cpp')
os.system('./a.out input.dat')

# Read output
with open('output.dat', 'r') as f:
    out_real_hcl = [float(n) for n in f.readline().split('\t')]
    out_imag_hcl = [float(n) for n in f.readline().split('\t')]

    np.testing.assert_allclose(out_real_np, np.array(out_real_hcl),
            rtol=1e-03, atol=1e-4)
    np.testing.assert_allclose(out_imag_np, np.array(out_imag_hcl),
            rtol=1e-03, atol=1e-4)

os.system('rm a.out *.dat')
print "Success."
