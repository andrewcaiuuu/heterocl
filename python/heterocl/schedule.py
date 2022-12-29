import functools
import warnings

import hcl_mlir
from hcl_mlir import GlobalInsertionPoint
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.dialects import func as func_d
from hcl_mlir.ir import *
from hcl_mlir.exceptions import *

from .devices import Device, DevMemoryPair
from .context import (BreakFlag, ImperativeLoopDepth, ImperativeLoopNestCount,
                      NestedStageLevel, StageName, UniqueName, StageAttachGlobal,
                      get_context, get_location, set_context, exit_context)
from .dfg import DataflowGraph
from .utils import get_extra_type_hints, remove_moved_attr, get_src_loc, hcl_dtype_to_mlir
from .ast import ast
from .ast.ir_builder import IRBuilder

# By default, Python ignores deprecation warnings.
# we have to enable it to see the warning.
warnings.simplefilter('always', DeprecationWarning)

def create_schedule_from_ast(_ast, inputs, func, name):
    """Create a schedule from an intermediate representation.
    Also used by creating schedule from scheme.
    """
    s = Schedule(name, inputs, func)
    s._ast = _ast
    create_stage_pass = CreateStage(s.ast, s)
    create_stage_pass.apply()
    return s

def build_schedule(inputs, func=None, name=""):
    """Build a schedule for compute optimizations.
    inputs: list of Tensor
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    top_func = ast.FuncOp("top", inputs, [], loc)
    top_func.level = 0
    if func is None:
        # All operations have inserted in scope!
        outputs = list()
        for op in ast.scope.pop():
            top_func.body.append(op)
        if len(top_func.body) == 0:
            raise APIError("received an empty algorithm specification, no operations present")
    else:
        ast.scope.pop()
        ast.scope.push(top_func.body)
        ret = func(*inputs)
        if ret is None:
            outputs = list()
        elif isinstance(ret, tuple):
            outputs = list(ret)
        else:
            outputs = [ret]
    top_func.return_tensors.extend(outputs)
    _ast = ast.AST(top_func)
    # print(_ast)
    s = create_schedule_from_ast(_ast, inputs, func, name)
    return s

def build_schedule_old(inputs, func=None, name=""):
    """Create a schedule for compute optimizations.
    inputs: list of Tensor
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    new_inputs = []
    for tensor in inputs:
        if not isinstance(tensor.op, hcl_mlir.TensorOp) and len(tensor.op.inputs) != 0:
            raise RuntimeError("Inputs are not roots!")
        new_inputs.append(tensor)
    inputs = new_inputs
    # initialization
    GlobalInsertionPoint.clear()
    set_context()

    # create actual HCL IR nodes
    if name == "":
        if func != None:
            name = func.__name__
        else:
            name = UniqueName.get("schedule")
    sch = Schedule(name, inputs, func)

    # build IR
    with get_context() as ctx, get_location() as loc:
        # create actual IR reference
        func_op = sch.device_top
        for placeholder, arg in zip(inputs, func_op.entry_block.arguments):
            placeholder.op.update_op(arg)

        # execute all fcompute and generate inner IR nodes
        # 1) func is hcl.compute: IR nodes not build inplace (default)
        # 2) func is defined by imperative DSL: IR nodes build inplace
        hcl_mlir.flags.BIT_OP = False
        if func != None:  # can build function directly
            """
            When having code like
            def kernel(A):
                A[0][4] = 1
            It should automatically enable in-place building
            """
            hcl_mlir.enable_build_inplace()
            ret = func(*inputs)
            hcl_mlir.disable_build_inplace()
        else:
            ret = None
            # traverse forward in AST to build IR

            def topological_sort(roots):
                lst = []
                output_tensor = []
                working_set = roots.copy()
                while len(working_set) != 0:
                    node = working_set.pop(0)
                    lst.append(node)
                    if len(node.uses) == 0:  # also get the output tensors
                        output_tensor.append(node)
                    for use in node.uses:
                        flags = [
                            in_tensor in lst for in_tensor in use.op.inputs]
                        if sum(flags) == len(use.op.inputs):
                            working_set.append(use)
                return lst, output_tensor

            order, ret = topological_sort(inputs)
            # Unwrap the stage's output tensor
            # The Tensor wrapping around ComputeOp/TensorOp acts as a container
            # The ComputeOp's output Tensor is the actual returned result
            ret = [t.op.output for t in ret if not isinstance(
                t.op, hcl_mlir.TensorOp)]
            for tensor in order:
                if not isinstance(tensor.op, hcl_mlir.TensorOp):
                    tensor.build()
        if hcl_mlir.flags.BIT_OP:
            sch.device_top.attributes["bit"] = UnitAttr.get()

        if ret is not None:
            outputs = []
            if isinstance(ret, (list, tuple)):
                outputs = list(ret)
            else:
                outputs.append(ret)
            # recompute the function type
            return_types = [v.memref_type for v in outputs]
            function_type = FunctionType.get(
                inputs=func_op.type.inputs, results=return_types)
            func_op.attributes["function_type"] = TypeAttr.get(function_type)
            otypes = "".join(
                [get_extra_type_hints(v.op.dtype) for v in outputs])
            func_op.attributes["otypes"] = StringAttr.get(otypes)

            # create block terminator
            new_outputs = []
            for output in outputs:
                new_outputs.append(output.result)
            sch.DataflowGraph.set_leaves(outputs)
            assert len(new_outputs) == len(outputs)
            ret_op = func_d.ReturnOp(
                new_outputs, ip=GlobalInsertionPoint.get())
            GlobalInsertionPoint.restore()

            # let the later schedule nodes insert before ret_op
            #   compute1
            #   compute2
            #   schedule1 # inserted _before_ the point
            #   ret_op    <- InsertionPoint
            GlobalInsertionPoint.save(InsertionPoint(ret_op))
        else:  # there's no return value
            function_type = FunctionType.get(
                inputs=func_op.type.inputs, results=[])
            func_op.attributes["function_type"] = TypeAttr.get(function_type)
            func_op.attributes["otypes"] = StringAttr.get("")
            # create block terminator
            ret_op = func_d.ReturnOp([], ip=GlobalInsertionPoint.get())
            GlobalInsertionPoint.restore()
            GlobalInsertionPoint.save(InsertionPoint(ret_op))

    # let each stage's output be an attribute of the function
    if StageAttachGlobal.get():
        if func != None:
            func.__dict__.clear()
            for op, stage in Stage._mapping:
                if op is not None:
                    func.__setattr__(op.name, op)

    exit_context()
    remove_moved_attr(sch.device_module)
    return sch


def customize(inputs, func=None, name=""):
    try:
        return build_schedule(inputs, func, name)
    except Exception as e:
        raise e
    finally:
        ast.scope.reset()
        Schedule._FuncDefs.clear()


def create_schedule(inputs, func=None, name=""):
    """Create a schedule for compute optimizations.
    inputs: list of Tensor
    """
    return customize(inputs, func, name)


class Partition(object):
    Complete = 0
    Block = 1
    Cyclic = 2


class Schedule(object):
    """Create a compute schedule
    """
    _TopFunction = None
    _CurrentSchedule = None
    _ast = None #TODO(Niansong): consider removing this
    _FuncDefs = dict() #TODO(Niansong): add reset logic to this
    

    def __init__(self, name, inputs, func=None):
        self.name = name
        self.lowered = False
        # Device-agnostic module:
        # used for transformation
        self._device_module = None
        self._device_top = None

        # Device-aware module:
        # used for generating host & xcel code
        self._host_module = None
        self._xcel_module = None
        self._host_top = None
        self._xcel_top = None
        self._host_ret = None
        self._xcel_ret = None

        self._ast = None

        # Used by Stages to refer to the current schedule
        Schedule._CurrentSchedule = self
        Schedule._TopFunction = func

        # External module:
        # used for generating other backend codes
        self._extern_module = None
        self._extern_top = None

        # Clear stage mapping
        Stage._mapping.clear()

    @property
    def device_module(self):
        return self._device_module

    @property
    def device_top(self):
        return self._device_top

    @property
    def host_module(self):
        return self._host_module

    @property
    def host_top(self):
        return self._host_top

    @property
    def xcel_module(self):
        return self._xcel_module

    @property
    def xcel_top(self):
        return self._xcel_top

    @property
    def extern_module(self):
        return self._extern_module

    @property
    def extern_top(self):
        return self._extern_top

    @property
    def ast(self):
        return self._ast

    def set_lowered(self):
        self.lowered = True

    def is_lowered(self):
        return self.lowered

    def __getitem__(self, target):
        """Return a Stage
        """
        if isinstance(target, Stage):
            return target
        for op, stage in Stage._mapping:
            if op.name == target.name:
                return stage
        raise APIError("Cannot find stage: " + target.name)

    def partition(self, target, partition_type=Partition.Complete, dim=0, factor=0):
        """Partition a Tensor into smaller Tensors or even registers
        """
        if self.is_lowered():
            raise APIError(".partition() must be called before lowering")
        if partition_type > 2:
            raise HCLValueError("Invalid partition type")
        if dim < 0:
            raise HCLValueError("Invalid dimension")
        if factor < 0:
            raise HCLValueError("Invalid factor")

        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        if partition_type == Partition.Complete:
            partition_type = 0
        elif partition_type == Partition.Block:
            partition_type = 1
        elif partition_type == Partition.Cyclic:
            partition_type = 2
        else:
            raise HCLValueError("Not supported partition type")
        partition_op = ast.PartitionOp(target, partition_type, dim, factor, loc)
        self.ast.top_func.body.append(partition_op)


    def replace(self, src, dst):
        """Replace a Tensor with another Tensor
        """
        if self.is_lowered():
            raise APIError(".replace() must be called before lowering")
        
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        replace_op = ast.ReplaceOp(src, dst, loc)
        self.ast.top_func.body.append(replace_op)


    def reshape(self, target, shape):
        """Reshape a Tensor to a specified new shape
        """
        if self.is_lowered():
            raise APIError(".reshape() must be called before lowering")
        ori_size = functools.reduce(lambda a, b: a*b, target.shape, 1)
        new_size = functools.reduce(lambda a, b: a*b, shape, 1)
        if ori_size != new_size:
            raise RuntimeError(
                "The reshaped tensor should have the same total size with the original tensor")
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        reshape_op = ast.ReshapeOp(target, shape, loc)
        self.ast.top_func.body.append(reshape_op)

    def reform(self, target, layout):
        """Change the layout of a tensor
        """
        if self.is_lowered():
            raise APIError(".reform() must be called before lowering")
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        reform_op = ast.ReformOp(target, layout, loc)
        self.ast.top_func.body.append(reform_op)

    def reuse_at(self, target, parent, axis, name=None):
        if self.is_lowered():
            raise APIError(".reuse_at() must be called before lowering")
        if not isinstance(axis, ast.LoopHandle):
            raise DTypeError("reuse_at() got invalid axis of type {}".format(type(axis)))
        if not isinstance(target, (ast.AllocOp, ast.ReuseAtOp)):
            raise DTypeError("reuse_at() got invalid target of type {}".format(type(target)))
        
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        reuse_at_op = ast.ReuseAtOp(target, axis, loc)
        self.ast.top_func.body.append(reuse_at_op)
        return reuse_at_op
    
    def buffer_at(self, target, parent, axis, name=None):
        """Create a write buffer reusing the output of current stage"""
        if self.is_lowered():
            raise APIError(".buffer_at() must be called before lowering")
        
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        buffer_at_op = ast.BufferAtOp(target, axis, loc)
        self.ast.top_func.body.append(buffer_at_op)
        return buffer_at_op


    def to(self, tensor, dst=None, fifo_depth=-1):
        if self.is_lowered():
            raise APIError(".to() must be called before lowering")
        # host-device data movement
        if isinstance(dst, (Device, DevMemoryPair)):
            # only do annotation not mutation here
            # code change happens when building the module
            if not isinstance(tensor, list):
                tensor = [tensor]
            for t in tensor:
                t.device = dst
        # inter-stage data movement
        elif isinstance(dst, Stage):
            #TODO(Niansong): deal with this later
            try:
                tensor = tensor.tensor
            except (AttributeError, ValueError):
                try:
                    tensor = tensor._op
                except AttributeError:
                    pass
            if not isinstance(tensor, OpResult):
                tensor = tensor.result
            with get_context() as ctx, get_location() as loc:
                # automatically set dataflow pragma
                self.device_top.attributes["dataflow"] = UnitAttr.get()
                i32 = IntegerType.get_signless(32)
                fifo_depth = IntegerAttr.get(i32, fifo_depth)
                # do .to() scheduling
                to_op = hcl_d.InterKernelToOp(
                    tensor, dst.stage_handle.result, fifo_depth=fifo_depth, ip=GlobalInsertionPoint.get())

    def outline(self, *stage_list, unify=False):
        """Outline stages as a function

        e.g., s.outline([s0,s1], [s2], [s3,s4])
        """
        if self.is_lowered():
            raise APIError(".outline() must be called before lowering")
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        results = []
        for i, stages in enumerate(stage_list):
            if isinstance(stages, list):
                handles = [stage.stage_handle for stage in stages]
                names = [stage.name for stage in stages]
            else:
                handles = [stages.stage_handle]
                names = [stages.name]
            
            outline_op = ast.OutlineOp(handles, loc)
            self.ast.top_func.body.append(outline_op)
            if unify and i > 0:
                outline_op.unify = results[0].name
            else:
                results.append(StageFunction(names))
        return results if len(results) > 1 else results[0]


class StageFunction(object):
    """
    Looks like stage function is just to have a separate
    MLIR module for creation of execution engine
    """
    def __init__(self, name=None):
        if not isinstance(name, list):
            name = [name]
        self.name = "Stage"
        for n in name:
            self.name += "_" + n
        self.module = None

    def build(self, schedule):
        set_context()
        with get_context() as ctx, get_location() as loc:
            new_module = Module.create(loc)
            # just a placeholder for inserting the function
            top = func_d.FuncOp(name="top", type=FunctionType.get(
                inputs=[], results=[]), ip=InsertionPoint(new_module.body))
            for op in schedule.device_module.body.operations:
                if str(op.name) == "\"{}\"".format(self.name):
                    op.move_before(top)
                    op.attributes["bit"] = UnitAttr.get()
                    break
            else:
                raise APIError("Stage {} not found".format(self.name))
            top.operation.erase()
        self.module = new_module
        return new_module


class Stage(object):
    """A Stage represents schedule for one operation.
    """

    """ 
    obsolete note:
    Stage._mapping is a list of (Tensor, Stage) tuples
    or (Stage, Stage) tuples to keep track of all stages
    and their corresponding tensors. 
    For compute and mutate, we attach (Tensor, Stage) tuples
    For update and imperative, we attach (Stage, Stage) tuples
    """
    _mapping = []

    def __init__(self, name=None):
        if name is None:
            name = UniqueName.get("stage")
        self.name = name
        self.tensor = None
        self.stage_handle = None
        self.ip = None
        # Imperative stage attaches axes to Stage object
        self.axis = list()


    def reorder(self, *args):
        """reorder the arguments in the specified order.
        """
        schedule = Schedule._CurrentSchedule
        if schedule.is_lowered():
            raise APIError(".reorder() must be called before lowering")
        args = list(args)
        for i in range(0, len(args)):
            if isinstance(args[i], int):
                args[i] = self.tensor.axis[args[i]]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        reorder_op = ast.ReorderOp(args, loc)
        schedule.ast.top_func.body.append(reorder_op)

    def split(self, parent, factor=None, nparts=None, mode="transform"):
        """Split the stage either by factor providing outer scope, or both
        """
        schedule = Schedule._CurrentSchedule
        if schedule.is_lowered():
            raise APIError(".split() must be called before lowering")
        if nparts != None or mode != "transform":
            raise HCLNotImplementedError("nparts={}, mode={} not supported".format(nparts, mode))
        if isinstance(parent, int):
            parent = self.tensor.axis[parent]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        split_op = ast.SplitOp(self.stage_handle, parent, factor, loc)
        schedule.ast.top_func.body.append(split_op)
        return split_op.results[0], split_op.results[1]

    def tile(self, x_parent, y_parent, x_factor, y_factor):
        """ Perform tiling on two dimensions
        """
        schedule = Schedule._CurrentSchedule
        if schedule.is_lowered():
            raise APIError(".tile() must be called before lowering")
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        tile_op = ast.TileOp(self.stage_handle, x_parent, y_parent, x_factor, y_factor, loc)
        schedule.ast.top_func.body.append(tile_op)
        return tile_op.results[0], tile_op.results[1], tile_op.results[2], tile_op.results[3]

    def pipeline(self, var, initiation_interval=1):
        """Pipeline the iteration.
        """
        schedule = Schedule._CurrentSchedule
        if schedule.is_lowered():
            raise APIError(".pipeline() must be called before lowering")
        if isinstance(var, int):
            var = self.tensor.axis[var]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        pipeline_op = ast.PipelineOp(var, initiation_interval, loc)
        schedule.ast.top_func.body.append(pipeline_op)

    def unroll(self, var, factor=0):
        """Unroll the iteration.
        """
        schedule = Schedule._CurrentSchedule
        if schedule.is_lowered():
            raise APIError(".unroll() must be called before lowering")
        if isinstance(var, int):
            var = self.tensor.axis[var]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        unroll_op = ast.UnrollOp(var, factor, loc)
        schedule.ast.top_func.body.append(unroll_op)

    def parallel(self, var):
        """Parallelize the iteration.
        """
        schedule = Schedule._CurrentSchedule
        if schedule.is_lowered():
            raise APIError(".parallel() must be called before lowering")
        if isinstance(var, int):
            var = self.tensor.axis[var]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        parallel_op = ast.ParallelOp(var, loc)
        schedule.ast.top_func.body.append(parallel_op)

    def fuse(self, *args):
        """Fuse multiple consecutive iteration variables into a single iteration variable.
        """
        schedule = Schedule._CurrentSchedule
        if schedule.is_lowered():
            raise APIError(".fuse() must be called before lowering")
        assert len(args) >= 1, "Length of the arguments must be >=1 for fuse."
        args = list(args)
        for i in range(0, len(args)):
            if isinstance(args[i], int):
                args[i] = self.tensor.axis[args[i]]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        fuse_op = ast.FuseOp(args, loc)
        schedule.ast.top_func.body.append(fuse_op)
        return fuse_op

    def compute_at(self, parent, axis):
        """Attach the stage at parent's scope
        """
        schedule = Schedule._CurrentSchedule
        if schedule.is_lowered():
            raise APIError(".compute_at() must be called before lowering")
        if isinstance(axis, int):
            axis = parent.tensor.axis[axis]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        compute_at_op = ast.ComputeAtOp(self.stage_handle, parent.stage_handle, axis, loc)
        schedule.ast.top_func.body.append(compute_at_op)

    def outline(self, axis=None, unify=None):
        """Outline a stage as a function
        """
        schedule = Schedule._CurrentSchedule
        if schedule.is_lowered():
            raise APIError(".outline() must be called before lowering")
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        outline_op = ast.OutlineOp([self.stage_handle], loc)
        schedule.ast.top_func.body.append(outline_op)
        if axis is not None:
            if isinstance(axis, str):
                outline_op.axis = axis
            else:
                outline_op.axis = axis.loop_name
        if unify is not None:
            outline_op.unify = unify.name
            return unify
        else:
            return StageFunction(self.name)        

    def systolic(self):
        """Wrap the current stage as a systolic array
        """
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        systolic_op = ast.SystolicOp(self.tensor, loc)
        schedule = Schedule._CurrentSchedule
        schedule.ast.top_func.body.append(systolic_op)

    def __enter__(self):
        HCLDeprecationWarning(
            "hcl.Stage() is deprecated, please remove it.").warn()

    def __exit__(self, ptype, value, trace):
        pass


class CreateStage(object):
    """Create HeteroCL stages

    This pass does three things:
    1. Create stage and loop handles and set tensor.axis for all stage's tensors
    2. Attach tensors to Python functions as attributes
    3. Create a mapping from tensor to stage in Schedule
    """
    def __init__(self, _ast, sch):
        self._ast = _ast
        self.sch = sch

    def apply(self):
        """Pass entry point"""
        top_func = self._ast.top_func
        self.visit(top_func)

    def visit(self, op):
        self.create_stage(op)
        if hasattr(op, "body") and op.body is not None:
            for op in op.body:
                # recursively visit the body
                self.visit(op)
    
    def create_stage(self, op):
        if isinstance(op, ast.ComputeOp):
            self.create_compute_stage(op)
        elif isinstance(op, ast.ForOp):
            self.create_imperative_stage(op)

    def create_compute_stage(self, op : ast.ComputeOp):
        # Create stage and attach attributes
        stage = Stage(op.name)
        tensor = op.tensor if op.kind == "compute" else op.aux_tensor
        stage.tensor = tensor
        top_func = Schedule._TopFunction
        if op.kind == "compute":
            Stage._mapping.append((tensor, stage))
            if top_func is not None:
                top_func.__setattr__(op.name, op.tensor)
        elif op.kind == "update":
            stage.__setattr__(op.tensor.name, tensor)
            Stage._mapping.append((stage, stage))
            if top_func is not None:
                top_func.__setattr__(op.name, stage)
        else:
            # TODO: Mutate
            pass
        
        # create handles
        stage_hdl = ast.OpHandle(op.name, op.loc)
        stage.stage_handle = stage_hdl
        for iter_var in op.iter_vars + op.reduce_vars:
            loop_hdl = ast.LoopHandle(stage_hdl, iter_var.name, op.loc)
            tensor.axis.append(loop_hdl)

    def create_imperative_stage(self, op : ast.ForOp):
        if op.tag is None:
            return
        # create stage and attach attributes
        stage = Stage(op.tag)
        Stage._mapping.append((stage, stage))
        top_func = Schedule._TopFunction
        if top_func is not None:
            top_func.__setattr__(op.tag, stage)        
        
        # create handles
        nested_for_loops = [op]
        def get_nested_for_loops(op):
            for body_op in op.body:
                if isinstance(body_op, ast.ForOp):
                    nested_for_loops.append(body_op)
                    get_nested_for_loops(body_op)
        get_nested_for_loops(op)
        stage_hdl = ast.OpHandle(op.tag, op.loc)
        stage.stage_handle = stage_hdl
        for loop in nested_for_loops:
            loop_hdl = ast.LoopHandle(stage_hdl, loop.name, op.loc)
            stage.axis.append(loop_hdl)
            setattr(stage, loop.name, loop_hdl)

class CreateStage_deprecated(object):
    """Create HeteroCL stages, stage and loop handles.

    This pass does three things:
    1. Create stage and loop handles and set tensor.axis for all stage's tensors
    2. Attach tensors to Python functions as attributes
    3. Create a mapping from tensor to stage in Schedule
    """

    def __init__(self, intermediate, schedule):
        super().__init__("create_stage", intermediate)
        self.sch = schedule
        self.ip = InsertionPoint.at_block_terminator(self.ast.top_func.ir_op.entry_block)


    def visit(self, op):
        self.create_stage(op)
        if hasattr(op, "body") and op.body is not None:
            for op in op.body:
                # recursively visit the body
                self.visit(op)

    def create_stage(self, op):
        if isinstance(op, ast.ComputeOp):
            self.create_compute_stage(op)
        elif isinstance(op, ast.ForOp):
            self.create_imperative_stage(op)
        else:
            pass
            # raise HCLNotImplementedError("create_stage method not implemented for op type: " + type(op))


    def create_compute_stage(self, op : ast.ComputeOp):
        tensor = op.tensor if op.kind == "compute" else op.aux_tensor
        # Step 1: create stage and loop handles
        with get_context(), get_location():
            stage_hdl = hcl_d.CreateOpHandleOp(StringAttr.get(op.name), ip=self.ip)
            for iter_var in op.iter_vars:
                loop_hdl = hcl_d.CreateLoopHandleOp(stage_hdl.result, StringAttr.get(iter_var.name), ip=self.ip)
                tensor.axis.append(loop_hdl)
            for reduce_var in op.reduce_vars:
                loop_hdl = hcl_d.CreateLoopHandleOp(stage_hdl.result, StringAttr.get(reduce_var.name), ip=self.ip)
                tensor.axis.append(loop_hdl)

        # Step 2: attach tensors to top Python function
        top_func = Schedule._TopFunction
        if top_func is not None:
            top_func.__setattr__(tensor.name, tensor)

        # Step 3: create a mapping from tensor to stage
        stage = Stage(op.name)
        stage.ip = self.ip
        stage.tensor = tensor
        stage.stage_handle = stage_hdl
        Stage._mapping.append((tensor, stage))

    def create_imperative_stage(self, op : ast.ForOp):
        if op.tag is None:
            return

        nested_for_loops = [op]
        def get_nested_for_loops(op):
            for body_op in op.body:
                if isinstance(body_op, ast.ForOp):
                    nested_for_loops.append(body_op)
                    get_nested_for_loops(body_op)
        get_nested_for_loops(op)
        
        # Step 1: create a stage
        stage = Stage(op.tag)
        stage.ip = self.ip
        
        # Step 2: create stage and loop handles
        with get_context(), get_location():
            stage_hdl = hcl_d.CreateOpHandleOp(StringAttr.get(op.tag), ip=self.ip)
            stage.stage_handle = stage_hdl
            for l in nested_for_loops:
                loop_hdl = hcl_d.CreateLoopHandleOp(stage_hdl.result, StringAttr.get(l.name), ip=self.ip)
                stage.axis.append(loop_hdl)

        # Step 3: attach stage to top function
        top_func = Schedule._TopFunction
        if top_func is not None:
            top_func.__setattr__(op.tag, stage)

        # TODO: Mapping?

    def apply(self):
        """Pass entry point"""
        top_func = self.ast.top_func
        self.visit(top_func)