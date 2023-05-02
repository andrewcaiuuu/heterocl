from ..ast import ast
from .pass_manager import Pass
from hcl_mlir.exceptions import *

class ExpandFunc(Pass):
    """ Convert all funcop into nested funcop """
    def __init__(self):
        super().__init__("expand_func")
        self._ast = None
        self.subfuncs = []

    def visit(self, op):
        if isinstance(op, ast.FuncOp) and op.name == "top":
            self.expand_func(op)
            # print("SUBFUNCS: ", self.subfuncs)
            # print("ORIGINAL BODY: ", op.body)
            op.body = []
            for subfunc in self.subfuncs:
                call_op = ast.CallOp(subfunc.name, subfunc.args, subfunc.return_tensors, subfunc.loc)
                op.body.append(call_op)
                
    
    def apply(self, _ast):
        """Pass entry point"""
        self._ast = _ast
        for op in _ast.region:
            self.visit(op)
        return _ast

    def expand_func(self, scope):
        i = 0
        for op in scope.body:
            # print("EXPAND_FUNC GOT OP: ", op)
            if isinstance(op, ast.ComputeOp):
                lower_func_op = ast.FuncOp(f"sub_func{i}", op.input_tensors, [op], op.loc)
                lower_func_op.level = 1
                self.update_level(lower_func_op)
                self._ast.region.insert(1, lower_func_op)
                self.subfuncs.append(lower_func_op)
                i += 1
        return

        