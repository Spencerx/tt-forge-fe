# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import importlib
from types import ModuleType
from functools import lru_cache
from .cosine import Cosine
from .ethernet_datacopy import EthernetDatacopy
from .reciprocal import Reciprocal
from .tanh import Tanh
from .nop import Nop
from .buffer import Buffer
from .tilizer import Tilizer
from .clip import Clip
from .cumulativesum import CumulativeSum
from .argmax import Argmax
from .convolution import Conv2d
from .convolution import Conv2dTranspose
from .pooling import MaxPool2d
from .cast import Cast
from .pad import Pad
from .kv_cache import UpdateCache
from .kv_cache import FillCache

op_to_module_map = {
    "add": "eltwise_binary",
    "cast": Cast,
    "divide": "eltwise_binary",
    "remainder": "eltwise_binary",
    "subtract": "eltwise_binary",
    "maximum": "eltwise_binary",
    "minimum": "eltwise_binary",
    "multiply": "eltwise_binary",
    "heaviside": "eltwise_binary",
    "power": "eltwise_binary",
    "greater": "eltwise_binary",
    "greater_equal": "eltwise_binary",
    "less": "eltwise_binary",
    "less_equal": "eltwise_binary",
    "equal": "eltwise_binary",
    "not_equal": "eltwise_binary",
    "logical_and": "eltwise_binary",
    "exp": "eltwise_unary",
    "reciprocal": Reciprocal,
    "nop": Nop,
    "buffer": Buffer,
    "sqrt": "eltwise_unary",
    "relu": "eltwise_unary",
    "leaky_relu": "eltwise_unary",
    "gelu": "eltwise_unary",
    "log": "eltwise_unary",
    "sigmoid": "eltwise_unary",
    "clip": Clip,
    "cosine": Cosine,
    "sine": "eltwise_unary",
    "atan": "eltwise_unary",
    "tanh": Tanh,
    "cumsum": CumulativeSum,
    "argmax": Argmax,
    "logical_not": "eltwise_unary",
    "dropout": "eltwise_unary",
    "pow": "eltwise_unary",
    "tilizer": Tilizer,
    "erf": "eltwise_unary",
    "conv_sum": "eltwise_nary",
    "concatenate": "eltwise_nary",
    "where": "eltwise_nary",
    "index_copy": "eltwise_nary",
    "interleave": "eltwise_nary",
    "stack": "eltwise_nary",
    "matmul": "matmul",
    "sparse_matmul": "matmul",
    "depthwise": "depthwise",
    "embedding": "embedding",
    "embedding_bw": "embedding_bw",
    "ethernet_datacopy": EthernetDatacopy,
    "transpose": "tm",
    "adv_index": "tm",
    "reshape": "tm",
    "index": "tm",
    "select": "tm",
    "gather": "tm",
    "hslice": "tm",
    "hstack": "tm",
    "vslice": "tm",
    "vstack": "tm",
    "broadcast": "tm",
    "repeat": "tm",
    "repeat_interleave": "tm",
    "conv2d_depthwise_weights": "tm",
    "conv2d_depthwise_weights_bw": "tm",
    "conv2d_grouped_weights": "tm",
    "conv2d_grouped_weights_bw": "tm",
    "conv2d_prestride_act": "tm",
    "conv2d_prestride_weights": "tm",
    "pad_tile": "tm",
    "narrow": "tm",
    "pad": Pad,
    "unsqueeze": "tm",
    "squeeze": "tm",
    "pixel_shuffle": "tm",
    "forge_pad": "tm",
    "forge_unpad": "tm",
    "reduce_avg": "reduce",
    "reduce_sum": "reduce",
    "reduce_max": "reduce",
    "conv2d": Conv2d,
    "conv2d_transpose": Conv2dTranspose,
    "conv3d": "convolution",
    "max_pool1d": "pooling",
    "max_pool2d": MaxPool2d,
    "max_pool3d": "pooling",
    "avg_pool1d": "pooling",
    "avg_pool2d": "pooling",
    "avg_pool3d": "pooling",
    "resize1d": "resize",
    "resize2d": "resize",
    "resize3d": "resize",
    "upsample2d": "resize",
    "downsample2d": "resize",
    "dram_queue": "dram_queue",
    "softmax": "nn",
    "log_softmax": "nn",
    "softmax_bw": "nn",
    "mask": "mask",
    "layernorm": "nn",
    "layernorm_bw": "nn",
    "batchnorm": "nn",
    "quantize": "quantize",
    "forge_quantize": "quantize",
    "dequantize": "quantize",
    "requantize": "quantize",
    "forge_requantize": "quantize",
    "forge_dequantize": "quantize",
    "update_cache": UpdateCache,
    "fill_cache": FillCache,
}


def has_newstyle_interface(op_name):
    return type(op_to_module_map[op_name]) is not str


def is_tm(op_type):
    module_name_or_cls = op_to_module_map[op_type.op]
    if type(module_name_or_cls) is str:
        return module_name_or_cls == "tm"
    else:
        return module_name_or_cls(op_type).is_tm()


def is_eltwise(op_type):
    module_name_or_cls = op_to_module_map[op_type.op]
    if type(module_name_or_cls) is str:
        return "eltwise" in module_name_or_cls
    else:
        return module_name_or_cls(op_type).is_eltwise()


def is_eltwise_binary(op_type):
    module_name_or_cls = op_to_module_map[op_type.op]
    if type(module_name_or_cls) is str:
        return "eltwise_binary" == module_name_or_cls
    else:
        return module_name_or_cls(op_type).is_eltwise_binary()


def is_eltwise_unary(op_type):
    module_name_or_cls = op_to_module_map[op_type.op]
    if type(module_name_or_cls) is str:
        return "eltwise_unary" in module_name_or_cls
    else:
        return module_name_or_cls(op_type).is_eltwise_unary()


def is_eltwise_nary(op_type):
    module_name_or_cls = op_to_module_map[op_type.op]
    if type(module_name_or_cls) is str:
        return "eltwise_nary" in module_name_or_cls
    else:
        return module_name_or_cls(op_type).is_eltwise_nary()


@lru_cache(maxsize=len(op_to_module_map))
def _get_module_or_class(op_name):
    assert op_name in op_to_module_map, f"Forge op module not defined for {op_name}"
    module_name_or_cls = op_to_module_map[op_name]
    if type(module_name_or_cls) is str:
        return importlib.import_module("." + module_name_or_cls, package="forge.op.eval.forge")
    else:
        return module_name_or_cls


def empty_function(*inputs):
    pass


def get_f_forge_backward(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        return lambda *inputs: module_or_class.backward(op_type.op, op_type.attr, *inputs)
    else:
        return module_or_class(op_type).backward


def get_f_forge_shape(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        return lambda *inputs: module_or_class.shape(op_type.op, op_type.attr, *inputs)
    else:
        return module_or_class(op_type).shape


def get_f_forge_eval(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        return lambda *inputs: module_or_class.eval(op_type.op, op_type.attr, *inputs)
    else:
        return module_or_class(op_type).eval


def get_f_forge_decompose(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        if hasattr(module_or_class, "decompose"):
            return lambda *inputs: module_or_class.decompose(op_type.op, op_type.attr, *inputs)
        else:
            return empty_function
    else:
        return module_or_class(op_type).decompose


def get_f_forge_decompose_post_autograd(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        if hasattr(module_or_class, "decompose_post_autograd"):
            return lambda *inputs: module_or_class.decompose_post_autograd(op_type.op, op_type.attr, *inputs)
        else:
            return empty_function
    else:
        return module_or_class(op_type).decompose_post_autograd


def get_f_forge_decompose_post_optimize(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        if hasattr(module_or_class, "decompose_post_optimize"):
            return lambda *inputs: module_or_class.decompose_post_optimize(op_type.op, op_type.attr, *inputs)
        else:
            return empty_function
    else:
        return module_or_class(op_type).decompose_post_optimize


def get_f_forge_initial_flops_estimate(op_type):
    module_or_class = _get_module_or_class(op_type.op)
    if isinstance(module_or_class, ModuleType):
        if hasattr(module_or_class, "initial_flops_estimate"):
            return lambda *inputs: module_or_class.initial_flops_estimate(op_type.op, op_type.attr, *inputs)
        else:
            return empty_function
    else:
        return module_or_class(op_type).initial_flops_estimate
