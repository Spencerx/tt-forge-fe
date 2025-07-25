# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import forge
import forge.op
from forge import ForgeModule

from loguru import logger
import torch

from forge import Tensor, compile
from forge.verify.verify import verify
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.config import VerifyConfig
from forge.forge_property_utils import (
    record_forge_op_name,
    record_op_model_names,
    record_forge_op_args,
    record_single_op_operands_info,
)
import pytest


class Maxpool2D0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=3,
            stride=2,
            padding=[0, 0, 0, 0],
            dilation=1,
            ceil_mode=True,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=3,
            stride=2,
            padding=[0, 0, 0, 0],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=3,
            stride=2,
            padding=[1, 1, 1, 1],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=3,
            stride=1,
            padding=[1, 1, 1, 1],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=2,
            stride=2,
            padding=[0, 0, 0, 0],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=3,
            stride=1,
            padding=[1, 1, 1, 1],
            dilation=1,
            ceil_mode=True,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=2,
            stride=2,
            padding=[0, 0, 0, 0],
            dilation=1,
            ceil_mode=True,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=5,
            stride=1,
            padding=[2, 2, 2, 2],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=8,
            stride=6,
            padding=[0, 0, 0, 0],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=9,
            stride=1,
            padding=[4, 4, 4, 4],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=13,
            stride=1,
            padding=[6, 6, 6, 6],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=28,
            stride=26,
            padding=[0, 0, 0, 0],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=14,
            stride=13,
            padding=[0, 0, 0, 0],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=1,
            stride=2,
            padding=[0, 0, 0, 0],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=3,
            stride=2,
            padding=[0, 0, 1, 1],
            dilation=1,
            ceil_mode=False,
            channel_last=1,
        )
        return maxpool2d_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Maxpool2D0,
        [((1, 256, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_v1_vovnet39_obj_det_torchhub",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 512, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_v1_vovnet39_obj_det_torchhub",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 768, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_v1_vovnet39_obj_det_torchhub",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 64, 112, 112), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 64, 112, 112), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_resnet_50_img_cls_hf",
                "onnx_densenet_densenet169_img_cls_torchvision",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "onnx_wideresnet_wide_resnet50_2_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "onnx_wideresnet_wide_resnet101_2_img_cls_timm",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_resnet_50_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 192, 55, 55), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D3,
        [((1, 192, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "1",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 192, 27, 27), torch.float32)],
        {
            "model_names": ["onnx_alexnet_base_img_cls_torchhub", "pd_alexnet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D3,
        [((1, 256, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "1",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 480, 27, 27), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D3,
        [((1, 480, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "1",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D3,
        [((1, 512, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "1",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Maxpool2D3,
            [((1, 528, 13, 13), torch.float32)],
            {
                "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
                "pcc": 0.99,
                "args": {
                    "kernel_size": "3",
                    "stride": "1",
                    "padding": "[1, 1, 1, 1]",
                    "dilation": "1",
                    "ceil_mode": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_FATAL @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/generic/generic_pools.cpp:96: sw_parallel_config.has_value()"
            )
        ],
    ),
    (
        Maxpool2D1,
        [((1, 832, 13, 13), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D3,
        [((1, 832, 6, 6), torch.float32)],
        {
            "model_names": ["pd_googlenet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "1",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 64, 55, 55), torch.bfloat16)],
        {
            "model_names": ["pt_alexnet_base_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 64, 55, 55), torch.bfloat16)],
        {
            "model_names": ["pt_alexnet_base_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 192, 27, 27), torch.bfloat16)],
        {
            "model_names": ["pt_alexnet_base_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 192, 27, 27), torch.bfloat16)],
        {
            "model_names": ["pt_alexnet_base_img_cls_torchhub", "pt_rcnn_base_obj_det_torchvision_rect_0"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 256, 13, 13), torch.bfloat16)],
        {
            "model_names": ["pt_alexnet_base_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 256, 13, 13), torch.bfloat16)],
        {
            "model_names": ["pt_alexnet_base_img_cls_torchhub", "pt_rcnn_base_obj_det_torchvision_rect_0"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 16, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_autoencoder_conv_img_enc_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Maxpool2D4,
            [((1, 4, 14, 14), torch.bfloat16)],
            {
                "model_names": ["pt_autoencoder_conv_img_enc_github"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
                "args": {
                    "kernel_size": "2",
                    "stride": "2",
                    "padding": "[0, 0, 0, 0]",
                    "dilation": "1",
                    "ceil_mode": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_FATAL @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/generic/generic_pools.cpp:96: sw_parallel_config.has_value()"
            )
        ],
    ),
    (
        Maxpool2D2,
        [((1, 96, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet161_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 64, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet18_img_cls_torchvision",
                "pt_resnet_resnet34_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_unet_qubvel_img_seg_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_densenet_densenet121_hf_xray_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 64, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 64, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 32, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_dla_dla60_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 128, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 128, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_vovnet_vovnet27s_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D5,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "1",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_vovnet_vovnet27s_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 512, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_vgg_vgg13_obj_det_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D5,
        [((1, 512, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "1",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 64, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 64, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_rcnn_base_obj_det_torchvision_rect_0"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 128, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 256, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_monodle_base_obj_det_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 64, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 128, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_dla_dla46_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 192, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D5,
        [((1, 192, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "1",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 480, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D5,
        [((1, 480, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "1",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Maxpool2D5,
            [((1, 528, 14, 14), torch.bfloat16)],
            {
                "model_names": ["pt_googlenet_base_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
                "args": {
                    "kernel_size": "3",
                    "stride": "1",
                    "padding": "[1, 1, 1, 1]",
                    "dilation": "1",
                    "ceil_mode": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_FATAL @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/generic/generic_pools.cpp:96: sw_parallel_config.has_value()"
            )
        ],
    ),
    (
        Maxpool2D6,
        [((1, 832, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D5,
        [((1, 832, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_googlenet_base_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "1",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 64, 147, 147), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 192, 71, 71), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 384, 35, 35), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 1024, 17, 17), torch.bfloat16)],
        {
            "model_names": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 64, 24, 24), torch.bfloat16)],
        {
            "model_names": ["pt_mnist_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 64, 160, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet18_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 64, 160, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 64, 96, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 64, 240, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_retinanet_retinanet_rn101fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn152fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn18fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn34fpn_obj_det_hf",
                "pt_retinanet_retinanet_rn50fpn_obj_det_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 64, 150, 150), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_resnet50_base_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 32, 256, 256), torch.bfloat16)],
        {
            "model_names": ["pt_unet_base_img_seg_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 64, 128, 128), torch.bfloat16)],
        {
            "model_names": ["pt_unet_base_img_seg_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 128, 64, 64), torch.bfloat16)],
        {
            "model_names": ["pt_unet_base_img_seg_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 256, 32, 32), torch.bfloat16)],
        {
            "model_names": ["pt_unet_base_img_seg_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 64, 224, 224), torch.bfloat16)],
        {
            "model_names": [
                "pt_unet_carvana_base_img_seg_github",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg13_obj_det_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 128, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_unet_carvana_base_img_seg_github",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg13_obj_det_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 256, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_unet_carvana_base_img_seg_github",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg13_obj_det_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 256, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_vovnet_vovnet57_img_cls_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 512, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_unet_carvana_base_img_seg_github",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg13_obj_det_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 512, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_vovnet_vovnet57_img_cls_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 768, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_vovnet_vovnet57_img_cls_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 128, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10n_obj_det_github", "pt_yolov8_yolov8n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 320, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10x_obj_det_github", "pt_yolov8_yolov8x_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 512, 10, 10), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_320x320"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 512, 15, 15), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 384, 10, 10), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_320x320"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 384, 15, 15), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 128, 10, 10), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_320x320"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 256, 15, 15), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_640x640"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 640, 15, 15), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 640, 20, 20), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_640x640"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 384, 14, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 128, 14, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 256, 14, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6s_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 256, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_world_default_obj_det_github",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D8,
        [((1, 256, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "8",
                "stride": "6",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D9,
        [((1, 256, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_s_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "9",
                "stride": "1",
                "padding": "[4, 4, 4, 4]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D10,
        [((1, 256, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_s_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "13",
                "stride": "1",
                "padding": "[6, 6, 6, 6]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D11,
        [((1, 256, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "28",
                "stride": "26",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D12,
        [((1, 256, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "14",
                "stride": "13",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 128, 13, 13), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D9,
        [((1, 128, 13, 13), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "9",
                "stride": "1",
                "padding": "[4, 4, 4, 4]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D10,
        [((1, 128, 13, 13), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "13",
                "stride": "1",
                "padding": "[6, 6, 6, 6]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 640, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D9,
        [((1, 640, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "9",
                "stride": "1",
                "padding": "[4, 4, 4, 4]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D10,
        [((1, 640, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_x_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "13",
                "stride": "1",
                "padding": "[6, 6, 6, 6]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 384, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_vovnet_vovnet27s_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 512, 20, 20), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 128, 15, 15), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 512, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D9,
        [((1, 512, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "9",
                "stride": "1",
                "padding": "[4, 4, 4, 4]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D10,
        [((1, 512, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "13",
                "stride": "1",
                "padding": "[6, 6, 6, 6]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 384, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D9,
        [((1, 384, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "9",
                "stride": "1",
                "padding": "[4, 4, 4, 4]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D10,
        [((1, 384, 20, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_m_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "13",
                "stride": "1",
                "padding": "[6, 6, 6, 6]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D13,
        [((1, 256, 8, 8), torch.bfloat16)],
        {
            "model_names": ["pt_fpn_base_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "1",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D10,
        [((1, 512, 15, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "13",
                "stride": "1",
                "padding": "[6, 6, 6, 6]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D9,
        [((1, 512, 15, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "9",
                "stride": "1",
                "padding": "[4, 4, 4, 4]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 512, 15, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v4_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 640, 10, 10), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 128, 147, 147), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 256, 74, 74), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    pytest.param(
        (
            Maxpool2D2,
            [((1, 728, 37, 37), torch.bfloat16)],
            {
                "model_names": ["pt_xception_xception_img_cls_timm"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
                "args": {
                    "kernel_size": "3",
                    "stride": "2",
                    "padding": "[1, 1, 1, 1]",
                    "dilation": "1",
                    "ceil_mode": "False",
                    "channel_last": "0",
                },
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_FATAL @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/generic/generic_pools.cpp:96: sw_parallel_config.has_value()"
            )
        ],
    ),
    (
        Maxpool2D2,
        [((1, 1024, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_xception_xception_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 384, 20, 20), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 256, 10, 10), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_320x320"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 512, 14, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 128, 159, 159), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 256, 79, 79), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 256, 39, 39), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 128, 79, 79), torch.bfloat16)],
        {
            "model_names": ["pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D14,
        [((1, 112, 109, 64), torch.float32)],
        {
            "model_names": ["tf_resnet_resnet50_img_cls_keras"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "1",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla102_visual_bb_torchvision",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla102_visual_bb_torchvision",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 384, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "True",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 64, 512, 512), torch.bfloat16)],
        {
            "model_names": ["pt_vgg19_unet_default_sem_seg_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 128, 256, 256), torch.bfloat16)],
        {
            "model_names": ["pt_vgg19_unet_default_sem_seg_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 256, 128, 128), torch.bfloat16)],
        {
            "model_names": ["pt_vgg19_unet_default_sem_seg_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 512, 64, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vgg19_unet_default_sem_seg_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 128, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 256, 40, 40), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D7,
        [((1, 192, 13, 13), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D9,
        [((1, 192, 13, 13), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "9",
                "stride": "1",
                "padding": "[4, 4, 4, 4]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D10,
        [((1, 192, 13, 13), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "kernel_size": "13",
                "stride": "1",
                "padding": "[6, 6, 6, 6]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 64, 55, 55), torch.float32)],
        {
            "model_names": ["onnx_alexnet_base_img_cls_torchhub", "pd_alexnet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 256, 13, 13), torch.float32)],
        {
            "model_names": ["onnx_alexnet_base_img_cls_torchhub", "pd_alexnet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_dla_dla102_visual_bb_torchvision",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_dla_dla46_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla102_visual_bb_torchvision",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 64, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla46_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 64, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla46_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 128, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla46_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 96, 112, 112), torch.float32)],
        {
            "model_names": ["onnx_densenet_densenet161_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 32, 256, 256), torch.float32)],
        {
            "model_names": ["onnx_unet_base_img_seg_torchhub"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_names": ["onnx_unet_base_img_seg_torchhub"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_unet_base_img_seg_torchhub"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 256, 32, 32), torch.float32)],
        {
            "model_names": ["onnx_unet_base_img_seg_torchhub"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_dla_dla34_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 256, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla34_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 64, 214, 320), torch.float32)],
        {
            "model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 64, 160, 160), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv3_ssd_resnet50_img_cls_torchvision"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 288, 2, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D14,
        [((1, 112, 112, 64), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "1",
            },
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("MaxPool2d")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name == "model_names":
            record_op_model_names(metadata_value)
        elif metadata_name == "args":
            record_forge_op_args(metadata_value)
        else:
            logger.warning(
                "No utility function available in forge property handler to record %s property", metadata_name
            )

    max_int = 1000
    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype, max_int=max_int)
        for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)

    for name, parameter in framework_model._parameters.items():
        parameter_tensor = Tensor.create_torch_tensor(
            shape=parameter.shape.get_pytorch_shape(), dtype=parameter.pt_data_format, max_int=max_int
        )
        framework_model.set_parameter(name, parameter_tensor)

    for name, constant in framework_model._constants.items():
        constant_tensor = Tensor.create_torch_tensor(
            shape=constant.shape.get_pytorch_shape(), dtype=constant.pt_data_format, max_int=max_int
        )
        framework_model.set_constant(name, constant_tensor)

    record_single_op_operands_info(framework_model, inputs)

    compiler_cfg = forge.config.CompilerConfig()
    if "default_df_override" in metadata.keys():
        compiler_cfg.default_df_override = forge.DataFormat.from_json(metadata["default_df_override"])

    compiled_model = compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
