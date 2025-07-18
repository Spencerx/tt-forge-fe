# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn.functional
from ..interface import PyEltwiseUnaryOp
from loguru import logger
from ..common import to_torch_operands
from ....forgeglobal import TILE_DIM
from ....tensor import forge_dataformat_to_pytorch_dtype
import numpy as np
from forge.op.eval.common import calculate_tile_size
from .reciprocal import Reciprocal


class Sqrt(PyEltwiseUnaryOp):
    @classmethod
    def create(cls):
        self = cls("sqrt")
        return self

    def eval(self, tensors):
        assert len(tensors) == 1, "Sqrt should have one input"
        shape = tensors[0].shape
        original_types = [o.dtype for o in tensors]

        ret = torch.sqrt(tensors[0])

        if ret.dtype != original_types[0]:
            ret = ret.type(original_types[0])

        return ret

    def shape(self, tensor_shapes):
        assert len(tensor_shapes) == 1, "Sqrt should have one input"
        shape = tensor_shapes[0]
        return shape, []

    def backward(self, ac, operand, inputs, output, grad):
        assert len(inputs) == 1, "Sqrt should have one input"
        assert operand == 0, "Invalid operand index"
        rec = ac.op(Reciprocal.create(), (output,))
        mult = ac.op("multiply", (rec, ac.constant(0.5)))
        return ac.op("multiply", (mult, grad))

    def initial_flops_estimate(self, tensor_shapes):
        flops = 0
        output_shape = self.shape(tensor_shapes)[0]
        flops = np.prod(output_shape)

        return flops
