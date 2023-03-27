# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ..ast import ast
from .pass_manager import Pass
from hcl_mlir.exceptions import *


class PromoteFunc(Pass):
    """
    Move all function ops to global scope.
    """

    def __init__(self):
        #print("SHOULD CALL PROMOTE CONSTRUCTOR")
        super().__init__("promote_func")
        self._ast = None

    def apply(self, _ast):
        #print("SHOULD CALL PROMOTE APPLY")
        """Pass entry point"""
        self._ast = _ast
        #print("_AST: ", _ast)
        self.visit(_ast.region)
        return _ast

    def visit(self, region):
        #print("SHOULD CALL PROMOTE VISIT")
        for op in region:
            if isinstance(op, ast.FuncOp):
                self.promote_func(op, region)
            if hasattr(op, "body") and op.body is not None:
                self.visit(op.body)

    def promote_func(self, op, region):
        #print("SHOULD CALL PROMOTE PROMOTE_FUNC")
        if op in self._ast.region:
            # already promoted
            return

        op.level = 0
        self.update_level(op)
        region.remove(op)
        self._ast.region.insert(0, op)
