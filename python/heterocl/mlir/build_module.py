import io

import hcl_mlir
from hcl_mlir import (get_context, get_insertion_point, get_location,
                      set_insertion_point)

from mlir import passmanager
from mlir.execution_engine import *
from mlir.ir import *

from .base import get_module, get_top_function


def build(schedule, target=None, name="top", stmt=None):
    """Build the executable according to the schedule and target.
    """
    new_inputs = []
    for input_tensor in schedule.inputs:  # should be hcl_mlir.TensorOp
        new_inputs.append(input_tensor)

    # apply the schedule and lower
    func = get_top_function()
    hcl_mlir.loop_transformation(func.operation)
    get_module().dump()

    if target == "vhls":
        return build_fpga_kernel(schedule, new_inputs, target, name, stmt)
    else:
        return build_llvm(schedule, new_inputs, target, name, stmt)


def build_fpga_kernel(schedule, inputs, target=None, name="top", stmt=None):
    # generate code
    buf = io.StringIO()
    hcl_mlir.emit_hlscpp(get_module(), buf)
    buf.seek(0)
    return buf.read()


def lowerToLLVM(module):
    import mlir.conversions
    pm = passmanager.PassManager.parse(
        "reconcile-unrealized-casts")
    pm.run(module)
    return module


def build_llvm(schedule, inputs, target=None, name="top", stmt=None):
    with get_context() as ctx, get_location():
        # mod = get_module()
        func = get_top_function()
        func.attributes['llvm.emit_c_interface'] = UnitAttr.get()
        print("\n\nBefore Lowering: ")
        get_module().dump()
        hcl_mlir.lower_hcl_to_llvm(get_module(), ctx)
        # lowerToLLVM(get_module())
        print("lowered.")
        print("\n\nAfter Lowering: ")
        get_module().dump()
        execution_engine = ExecutionEngine(get_module())
        execution_engine.invoke(name)
        print("Execution success")
