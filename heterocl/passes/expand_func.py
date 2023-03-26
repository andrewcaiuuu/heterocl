from ..ast import ast
from .pass_manager import Pass
from hcl_mlir.exceptions import *

class ExpandFunc(Pass):
    """ Convert all funcop into nested funcop """
    def __init__(self):
        super().__init__("expand_func")

    def visit(self, op):
        print("hi")
        if isinstance(op, ast.FuncOp):
            print("THIS IS A FUNCOP: ", op)
            self.expand_func(op)
        if isinstance(op, ast.StoreOp):
            print(op)
            if op.value is not None:
                self.visit(op.value)
            if hasattr(op, "body") and op.body is not None:
                for body_op in op.body:
                    self.visit(body_op)
                self.expand_func(op)
    
    def apply(self, _ast):
        """Pass entry point"""
        for op in _ast.region:
            self.visit(op)

        return _ast

    def expand_func(self, scope):
        print("FUNCOP BODY: ", scope.body)
        computeops = list()
        for op in scope.body:
            if isinstance(op, ast.ComputeOp):
                computeops.append([op])
        return

        