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


class Matmul0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul0.weight_1",
            forge.Parameter(*(1024, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul0.weight_1"))
        return matmul_output_1


class Matmul1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, matmul_input_0, matmul_input_1):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, matmul_input_1)
        return matmul_output_1


class Matmul2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul2.weight_1",
            forge.Parameter(*(1024, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul2.weight_1"))
        return matmul_output_1


class Matmul3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul3.weight_1",
            forge.Parameter(*(4096, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul3.weight_1"))
        return matmul_output_1


class Matmul4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul4.weight_1", forge.Parameter(*(64, 64), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul4.weight_1"))
        return matmul_output_1


class Matmul5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul5.weight_1",
            forge.Parameter(*(64, 256), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul5.weight_1"))
        return matmul_output_1


class Matmul6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul6.weight_1",
            forge.Parameter(*(256, 64), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul6.weight_1"))
        return matmul_output_1


class Matmul7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul7.weight_1",
            forge.Parameter(*(128, 128), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul7.weight_1"))
        return matmul_output_1


class Matmul8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul8.weight_1",
            forge.Parameter(*(128, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul8.weight_1"))
        return matmul_output_1


class Matmul9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul9.weight_1",
            forge.Parameter(*(512, 128), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul9.weight_1"))
        return matmul_output_1


class Matmul10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul10.weight_1",
            forge.Parameter(*(320, 320), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul10.weight_1"))
        return matmul_output_1


class Matmul11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul11.weight_1",
            forge.Parameter(*(320, 1280), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul11.weight_1"))
        return matmul_output_1


class Matmul12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul12.weight_1",
            forge.Parameter(*(1280, 320), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul12.weight_1"))
        return matmul_output_1


class Matmul13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul13.weight_1",
            forge.Parameter(*(512, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul13.weight_1"))
        return matmul_output_1


class Matmul14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul14.weight_1",
            forge.Parameter(*(512, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul14.weight_1"))
        return matmul_output_1


class Matmul15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul15.weight_1",
            forge.Parameter(*(2048, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul15.weight_1"))
        return matmul_output_1


class Matmul16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul16_const_1", shape=(1, 1, 44), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul16_const_1"))
        return matmul_output_1


class Matmul17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul17_const_1", shape=(1, 1, 39), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul17_const_1"))
        return matmul_output_1


class Matmul18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul18_const_1", shape=(1, 1, 522), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul18_const_1"))
        return matmul_output_1


class Matmul19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul19_const_1", shape=(1, 1, 128), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul19_const_1"))
        return matmul_output_1


class Matmul20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul20_const_1", shape=(1, 1, 107), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul20_const_1"))
        return matmul_output_1


class Matmul21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul21_const_1", shape=(1, 1, 4), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul21_const_1"))
        return matmul_output_1


class Matmul22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul22_const_1", shape=(1, 1, 10), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul22_const_1"))
        return matmul_output_1


class Matmul23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul23_const_1", shape=(1, 1, 8), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul23_const_1"))
        return matmul_output_1


class Matmul24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul24_const_1", shape=(1, 1, 135), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul24_const_1"))
        return matmul_output_1


class Matmul25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul25_const_1", shape=(1, 1, 256), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul25_const_1"))
        return matmul_output_1


class Matmul26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul26_const_1", shape=(1, 1, 12), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul26_const_1"))
        return matmul_output_1


class Matmul27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul27_const_1", shape=(1, 1, 11), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul27_const_1"))
        return matmul_output_1


class Matmul28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul28_const_1", shape=(1, 1, 5), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul28_const_1"))
        return matmul_output_1


class Matmul29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul29_const_1", shape=(1, 1, 13), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul29_const_1"))
        return matmul_output_1


class Matmul30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul30_const_1", shape=(1, 1, 35), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul30_const_1"))
        return matmul_output_1


class Matmul31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul31_const_1", shape=(1, 1, 29), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul31_const_1"))
        return matmul_output_1


class Matmul32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul32_const_1", shape=(1, 1, 38), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul32_const_1"))
        return matmul_output_1


class Matmul33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul33.weight_1",
            forge.Parameter(*(768, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul33.weight_1"))
        return matmul_output_1


class Matmul34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul34.weight_1",
            forge.Parameter(*(768, 3072), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul34.weight_1"))
        return matmul_output_1


class Matmul35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul35.weight_1",
            forge.Parameter(*(3072, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul35.weight_1"))
        return matmul_output_1


class Matmul36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul36.weight_1",
            forge.Parameter(*(768, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul36.weight_1"))
        return matmul_output_1


class Matmul37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul37.weight_1",
            forge.Parameter(*(1024, 1000), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul37.weight_1"))
        return matmul_output_1


class Matmul38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul38.weight_1",
            forge.Parameter(*(1152, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul38.weight_1"))
        return matmul_output_1


class Matmul39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul39.weight_1",
            forge.Parameter(*(1280, 1000), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul39.weight_1"))
        return matmul_output_1


class Matmul40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul40_const_1", shape=(12, 24), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul40_const_1"))
        return matmul_output_1


class Matmul41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul41_const_1", shape=(12, 72), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul41_const_1"))
        return matmul_output_1


class Matmul42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul42_const_1", shape=(4, 24), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul42_const_1"))
        return matmul_output_1


class Matmul43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul43_const_1", shape=(4, 72), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul43_const_1"))
        return matmul_output_1


class Matmul44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul44_const_0", shape=(80, 512), dtype=torch.bfloat16)

    def forward(self, matmul_input_1):
        matmul_output_1 = forge.op.Matmul("", self.get_constant("matmul44_const_0"), matmul_input_1)
        return matmul_output_1


class Matmul45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul45.weight_1",
            forge.Parameter(*(2048, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul45.weight_1"))
        return matmul_output_1


class Matmul46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul46.weight_1",
            forge.Parameter(*(2048, 8192), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul46.weight_1"))
        return matmul_output_1


class Matmul47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul47.weight_1",
            forge.Parameter(*(8192, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul47.weight_1"))
        return matmul_output_1


class Matmul48(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul48.weight_1",
            forge.Parameter(*(2048, 128256), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul48.weight_1"))
        return matmul_output_1


class Matmul49(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul49_const_1", shape=(1, 1, 6), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul49_const_1"))
        return matmul_output_1


class Matmul50(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul50.weight_1",
            forge.Parameter(*(128, 312), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul50.weight_1"))
        return matmul_output_1


class Matmul51(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul51.weight_1",
            forge.Parameter(*(312, 312), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul51.weight_1"))
        return matmul_output_1


class Matmul52(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul52.weight_1",
            forge.Parameter(*(312, 1248), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul52.weight_1"))
        return matmul_output_1


class Matmul53(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul53.weight_1",
            forge.Parameter(*(1248, 312), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul53.weight_1"))
        return matmul_output_1


class Matmul54(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul54.weight_1",
            forge.Parameter(*(312, 128), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul54.weight_1"))
        return matmul_output_1


class Matmul55(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul55.weight_1",
            forge.Parameter(*(2048, 1000), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul55.weight_1"))
        return matmul_output_1


class Matmul56(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul56.weight_1",
            forge.Parameter(*(512, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul56.weight_1"))
        return matmul_output_1


class Matmul57(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul57.weight_1",
            forge.Parameter(*(320, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul57.weight_1"))
        return matmul_output_1


class Matmul58(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul58.weight_1",
            forge.Parameter(*(128, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul58.weight_1"))
        return matmul_output_1


class Matmul59(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul59.weight_1",
            forge.Parameter(*(64, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul59.weight_1"))
        return matmul_output_1


class Matmul60(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul60.weight_1",
            forge.Parameter(*(384, 384), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul60.weight_1"))
        return matmul_output_1


class Matmul61(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul61.weight_1",
            forge.Parameter(*(384, 1536), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul61.weight_1"))
        return matmul_output_1


class Matmul62(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul62.weight_1",
            forge.Parameter(*(1536, 384), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul62.weight_1"))
        return matmul_output_1


class Matmul63(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul63.weight_1",
            forge.Parameter(*(768, 30522), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul63.weight_1"))
        return matmul_output_1


class Matmul64(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul64.weight_1",
            forge.Parameter(*(512, 256), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul64.weight_1"))
        return matmul_output_1


class Matmul65(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul65.weight_1",
            forge.Parameter(*(320, 256), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul65.weight_1"))
        return matmul_output_1


class Matmul66(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul66.weight_1",
            forge.Parameter(*(128, 256), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul66.weight_1"))
        return matmul_output_1


class Matmul67(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul67.weight_1",
            forge.Parameter(*(512, 1000), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul67.weight_1"))
        return matmul_output_1


class Matmul68(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul68.weight_1",
            forge.Parameter(*(32, 32), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul68.weight_1"))
        return matmul_output_1


class Matmul69(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul69.weight_1",
            forge.Parameter(*(32, 128), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul69.weight_1"))
        return matmul_output_1


class Matmul70(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul70.weight_1",
            forge.Parameter(*(128, 32), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul70.weight_1"))
        return matmul_output_1


class Matmul71(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul71.weight_1",
            forge.Parameter(*(160, 160), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul71.weight_1"))
        return matmul_output_1


class Matmul72(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul72.weight_1",
            forge.Parameter(*(160, 640), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul72.weight_1"))
        return matmul_output_1


class Matmul73(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul73.weight_1",
            forge.Parameter(*(640, 160), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul73.weight_1"))
        return matmul_output_1


class Matmul74(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul74.weight_1",
            forge.Parameter(*(256, 256), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul74.weight_1"))
        return matmul_output_1


class Matmul75(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul75.weight_1",
            forge.Parameter(*(256, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul75.weight_1"))
        return matmul_output_1


class Matmul76(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul76.weight_1",
            forge.Parameter(*(1024, 256), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul76.weight_1"))
        return matmul_output_1


class Matmul77(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul77_const_1", shape=(1, 1, 588), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul77_const_1"))
        return matmul_output_1


class Matmul78(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul78.weight_1",
            forge.Parameter(*(96, 96), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul78.weight_1"))
        return matmul_output_1


class Matmul79(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul79.weight_1",
            forge.Parameter(*(512, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul79.weight_1"))
        return matmul_output_1


class Matmul80(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul80.weight_1",
            forge.Parameter(*(96, 384), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul80.weight_1"))
        return matmul_output_1


class Matmul81(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul81.weight_1",
            forge.Parameter(*(384, 96), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul81.weight_1"))
        return matmul_output_1


class Matmul82(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul82.weight_1",
            forge.Parameter(*(384, 192), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul82.weight_1"))
        return matmul_output_1


class Matmul83(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul83.weight_1",
            forge.Parameter(*(192, 192), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul83.weight_1"))
        return matmul_output_1


class Matmul84(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul84.weight_1",
            forge.Parameter(*(512, 6), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul84.weight_1"))
        return matmul_output_1


class Matmul85(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul85.weight_1",
            forge.Parameter(*(192, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul85.weight_1"))
        return matmul_output_1


class Matmul86(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul86.weight_1",
            forge.Parameter(*(768, 192), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul86.weight_1"))
        return matmul_output_1


class Matmul87(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul87.weight_1",
            forge.Parameter(*(768, 384), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul87.weight_1"))
        return matmul_output_1


class Matmul88(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul88.weight_1",
            forge.Parameter(*(512, 12), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul88.weight_1"))
        return matmul_output_1


class Matmul89(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul89.weight_1",
            forge.Parameter(*(1536, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul89.weight_1"))
        return matmul_output_1


class Matmul90(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul90.weight_1",
            forge.Parameter(*(512, 24), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul90.weight_1"))
        return matmul_output_1


class Matmul91(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul91.weight_1",
            forge.Parameter(*(256, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul91.weight_1"))
        return matmul_output_1


class Matmul92(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul92.weight_1",
            forge.Parameter(*(2048, 256), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul92.weight_1"))
        return matmul_output_1


class Matmul93(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul93.weight_1",
            forge.Parameter(*(256, 92), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul93.weight_1"))
        return matmul_output_1


class Matmul94(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul94.weight_1",
            forge.Parameter(*(256, 4), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul94.weight_1"))
        return matmul_output_1


class Matmul95(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul95.weight_1",
            forge.Parameter(*(120, 360), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul95.weight_1"))
        return matmul_output_1


class Matmul96(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul96.weight_1",
            forge.Parameter(*(120, 120), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul96.weight_1"))
        return matmul_output_1


class Matmul97(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul97.weight_1",
            forge.Parameter(*(120, 240), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul97.weight_1"))
        return matmul_output_1


class Matmul98(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul98.weight_1",
            forge.Parameter(*(240, 120), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul98.weight_1"))
        return matmul_output_1


class Matmul99(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul99.weight_1",
            forge.Parameter(*(120, 6625), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul99.weight_1"))
        return matmul_output_1


class Matmul100(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul100.weight_1",
            forge.Parameter(*(120, 97), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul100.weight_1"))
        return matmul_output_1


class Matmul101(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul101.weight_1",
            forge.Parameter(*(160, 256), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul101.weight_1"))
        return matmul_output_1


class Matmul102(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul102.weight_1",
            forge.Parameter(*(32, 256), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul102.weight_1"))
        return matmul_output_1


class Matmul103(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul103.weight_1",
            forge.Parameter(*(9216, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul103.weight_1"))
        return matmul_output_1


class Matmul104(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul104.weight_1",
            forge.Parameter(*(4096, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul104.weight_1"))
        return matmul_output_1


class Matmul105(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul105.weight_1",
            forge.Parameter(*(4096, 1000), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul105.weight_1"))
        return matmul_output_1


class Matmul106(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul106_const_0", shape=(1, 48), dtype=torch.float32)

    def forward(self, matmul_input_1):
        matmul_output_1 = forge.op.Matmul("", self.get_constant("matmul106_const_0"), matmul_input_1)
        return matmul_output_1


class Matmul107(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul107.weight_1",
            forge.Parameter(*(96, 6625), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul107.weight_1"))
        return matmul_output_1


class Matmul108(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul108.weight_1",
            forge.Parameter(*(96, 97), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul108.weight_1"))
        return matmul_output_1


class Matmul109(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul109_const_1", shape=(1, 1, 334), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul109_const_1"))
        return matmul_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Matmul0,
        [((384, 1024), torch.float32)],
        {"model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 384, 64), torch.float32), ((16, 64, 384), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 384, 384), torch.float32), ((16, 384, 64), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((1, 384, 1024), torch.float32)],
        {"model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul3,
        [((1, 384, 4096), torch.float32)],
        {"model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((384, 1024), torch.float32), ((1024, 1), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul4,
        [((1, 16384, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul4,
        [((256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 16384, 64), torch.float32), ((1, 64, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 16384, 256), torch.float32), ((1, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul5,
        [((1, 16384, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul6,
        [((1, 16384, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul7,
        [((1, 4096, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul7,
        [((256, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((2, 4096, 64), torch.float32), ((2, 64, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((2, 4096, 256), torch.float32), ((2, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul7,
        [((4096, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul8,
        [((1, 4096, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul9,
        [((1, 4096, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul10,
        [((1, 1024, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul10,
        [((256, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((5, 1024, 64), torch.float32), ((5, 64, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((5, 1024, 256), torch.float32), ((5, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul10,
        [((1024, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul11,
        [((1, 1024, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul12,
        [((1, 1024, 1280), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul13,
        [((256, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((8, 256, 64), torch.float32), ((8, 64, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((8, 256, 256), torch.float32), ((8, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul14,
        [((1, 256, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul15,
        [((1, 256, 2048), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 512), torch.float32), ((512, 1000), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_mit_b2_img_cls_hf", "onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99},
    ),
    (
        Matmul0,
        [((197, 1024), torch.float32)],
        {"model_names": ["onnx_vit_base_google_vit_large_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 197, 64), torch.float32), ((16, 64, 197), torch.float32)],
        {"model_names": ["onnx_vit_base_google_vit_large_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 197, 197), torch.float32), ((16, 197, 64), torch.float32)],
        {"model_names": ["onnx_vit_base_google_vit_large_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 197, 1024), torch.float32)],
        {"model_names": ["onnx_vit_base_google_vit_large_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul3,
        [((1, 197, 4096), torch.float32)],
        {"model_names": ["onnx_vit_base_google_vit_large_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1024), torch.float32), ((1024, 1000), torch.float32)],
        {
            "model_names": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "onnx_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "onnx_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "onnx_densenet_densenet121_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((44, 3072), torch.float32), ((3072, 3072), torch.float32)],
        {"model_names": ["pt_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul16,
        [((1, 64, 1), torch.float32)],
        {"model_names": ["pt_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((44, 3072), torch.float32), ((3072, 1024), torch.float32)],
        {"model_names": ["pt_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((24, 44, 128), torch.float32), ((24, 128, 44), torch.float32)],
        {"model_names": ["pt_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((24, 44, 44), torch.float32), ((24, 44, 128), torch.float32)],
        {"model_names": ["pt_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((44, 3072), torch.float32), ((3072, 8192), torch.float32)],
        {"model_names": ["pt_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 44, 8192), torch.float32), ((8192, 3072), torch.float32)],
        {"model_names": ["pt_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 44, 3072), torch.float32), ((3072, 128256), torch.float32)],
        {"model_names": ["pt_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 4096), torch.float32), ((4096, 4096), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul17,
        [((1, 64, 1), torch.float32)],
        {
            "model_names": [
                "pt_deepseek_deepseek_math_7b_instruct_qa_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 39, 128), torch.float32), ((32, 128, 39), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 39, 39), torch.float32), ((32, 39, 128), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 4096), torch.float32), ((4096, 11008), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 39, 11008), torch.float32), ((11008, 4096), torch.float32)],
            {"model_names": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 39, 4096), torch.float32), ((4096, 102400), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((522, 3072), torch.float32), ((3072, 3072), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul18,
        [((1, 128, 1), torch.float32)],
        {
            "model_names": [
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((522, 3072), torch.float32), ((3072, 1024), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 522, 256), torch.float32), ((12, 256, 522), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 522, 522), torch.float32), ((12, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((522, 3072), torch.float32), ((3072, 9216), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 522, 9216), torch.float32), ((9216, 3072), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 522, 3072), torch.float32), ((3072, 131072), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((522, 3072), torch.float32), ((3072, 23040), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 522, 23040), torch.float32), ((23040, 3072), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((128, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {
            "model_names": [
                "pt_gemma_google_gemma_2b_text_gen_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Matmul19, [((1, 128, 1), torch.float32)], {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99}),
    (
        Matmul1,
        [((128, 2048), torch.float32), ((2048, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 128, 256), torch.float32), ((8, 256, 128), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 128, 128), torch.float32), ((8, 128, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((128, 2048), torch.float32), ((2048, 16384), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 128, 16384), torch.float32), ((16384, 2048), torch.float32)],
            {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 128, 2048), torch.float32), ((2048, 256000), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((107, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul20,
        [((1, 128, 1), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf", "pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((107, 2048), torch.float32), ((2048, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 107, 256), torch.float32), ((8, 256, 107), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 107, 107), torch.float32), ((8, 107, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((107, 2048), torch.float32), ((2048, 16384), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 107, 16384), torch.float32), ((16384, 2048), torch.float32)],
            {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 107, 2048), torch.float32), ((2048, 256000), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((107, 3072), torch.float32), ((3072, 4096), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 107, 256), torch.float32), ((16, 256, 107), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 107, 107), torch.float32), ((16, 107, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((107, 4096), torch.float32), ((4096, 3072), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((107, 3072), torch.float32), ((3072, 24576), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 107, 24576), torch.float32), ((24576, 3072), torch.float32)],
            {"model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 107, 3072), torch.float32), ((3072, 256000), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 2560), torch.float32), ((2560, 2560), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((20, 256, 128), torch.float32), ((20, 128, 256), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 256, 256), torch.float32), ((20, 256, 128), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 2560), torch.float32), ((2560, 10240), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 256, 10240), torch.float32), ((10240, 2560), torch.float32)],
            {
                "model_names": [
                    "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
                    "pt_phi2_microsoft_phi_2_clm_hf",
                    "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 256, 2560), torch.float32), ((2560, 50257), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((5, 2560), torch.float32), ((2560, 2560), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 5, 128), torch.float32), ((20, 128, 5), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 5, 5), torch.float32), ((20, 5, 128), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 5, 2560), torch.float32), ((2560, 10240), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 5, 10240), torch.float32), ((10240, 2560), torch.float32)],
            {"model_names": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 5, 2560), torch.float32), ((2560, 2), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 2048), torch.bfloat16), ((2048, 1000), torch.bfloat16)],
            {
                "model_names": [
                    "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                    "pt_hrnet_hrnet_w32_pose_estimation_timm",
                    "pt_hrnet_hrnet_w40_pose_estimation_timm",
                    "pt_hrnet_hrnet_w44_pose_estimation_timm",
                    "pt_hrnet_hrnet_w48_pose_estimation_timm",
                    "pt_hrnet_hrnet_w64_pose_estimation_timm",
                    "pt_hrnet_hrnet_w18_pose_estimation_timm",
                    "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                    "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
                    "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                    "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                    "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                    "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                    "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                    "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                    "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                    "pt_resnet_50_img_cls_hf",
                    "pt_resnet_resnet101_img_cls_torchvision",
                    "pt_resnet_resnet50_img_cls_torchvision",
                    "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                    "pt_resnext_resnext14_32x4d_img_cls_osmr",
                    "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                    "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                    "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                    "pt_xception_xception65_img_cls_timm",
                    "pt_xception_xception71_tf_in1k_img_cls_timm",
                    "pt_hrnet_hrnet_w30_pose_estimation_timm",
                    "pt_resnet_resnet152_img_cls_torchvision",
                    "pt_xception_xception41_img_cls_timm",
                    "pt_xception_xception71_img_cls_timm",
                    "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                    "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                    "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                    "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                    "pt_resnext_resnext101_64x4d_img_cls_osmr",
                    "pt_resnext_resnext50_32x4d_img_cls_osmr",
                    "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                    "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                    "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                    "pt_resnet_50_img_cls_timm",
                    "pt_xception_xception_img_cls_timm",
                    "pt_resnext_resnext26_32x4d_img_cls_osmr",
                    "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                    "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                    "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((128, 4096), torch.float32), ((4096, 4096), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_llama_7b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul19,
        [((1, 64, 1), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_llama_7b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 128, 128), torch.float32), ((32, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_llama_7b_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((128, 4096), torch.float32), ((4096, 11008), torch.float32)],
        {"model_names": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 128, 11008), torch.float32), ((11008, 4096), torch.float32)],
            {"model_names": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 128, 4096), torch.float32), ((4096, 32000), torch.float32)],
        {
            "model_names": ["pt_llama3_huggyllama_llama_7b_clm_hf", "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((4, 4096), torch.float32), ((4096, 4096), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul21,
        [((1, 64, 1), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 4, 128), torch.float32), ((32, 128, 4), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 4, 4), torch.float32), ((32, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((4, 4096), torch.float32), ((4096, 11008), torch.float32)],
        {"model_names": ["pt_llama3_huggyllama_llama_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 4, 11008), torch.float32), ((11008, 4096), torch.float32)],
            {"model_names": ["pt_llama3_huggyllama_llama_7b_seq_cls_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((4, 4096), torch.float32), ((4096, 2), torch.float32)],
        {"model_names": ["pt_llama3_huggyllama_llama_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((128, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((128, 4096), torch.float32), ((4096, 14336), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 128, 14336), torch.float32), ((14336, 4096), torch.float32)],
            {
                "model_names": [
                    "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                    "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 128, 4096), torch.float32), ((4096, 128256), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((4, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((4, 4096), torch.float32), ((4096, 14336), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 4, 14336), torch.float32), ((14336, 4096), torch.float32)],
            {
                "model_names": [
                    "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                    "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                    "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                    "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 4, 4096), torch.float32), ((4096, 2), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((128, 3072), torch.float32), ((3072, 3072), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((128, 3072), torch.float32), ((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((24, 128, 128), torch.float32), ((24, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((128, 3072), torch.float32), ((3072, 8192), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 8192), torch.float32), ((8192, 3072), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 3072), torch.float32), ((3072, 128256), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((4, 3072), torch.float32), ((3072, 3072), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((4, 3072), torch.float32), ((3072, 1024), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((24, 4, 128), torch.float32), ((24, 128, 4), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((24, 4, 4), torch.float32), ((24, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((4, 3072), torch.float32), ((3072, 8192), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 4, 8192), torch.float32), ((8192, 3072), torch.float32)],
            {
                "model_names": [
                    "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                    "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 4, 3072), torch.float32), ((3072, 2), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((10, 4096), torch.float32), ((4096, 4096), torch.float32)],
        {"model_names": ["pt_ministral_ministral_ministral_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul22,
        [((1, 64, 1), torch.float32)],
        {"model_names": ["pt_ministral_ministral_ministral_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((10, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {"model_names": ["pt_ministral_ministral_ministral_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 10, 128), torch.float32), ((32, 128, 10), torch.float32)],
        {"model_names": ["pt_ministral_ministral_ministral_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 10, 10), torch.float32), ((32, 10, 128), torch.float32)],
        {"model_names": ["pt_ministral_ministral_ministral_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((10, 4096), torch.float32), ((4096, 14336), torch.float32)],
        {"model_names": ["pt_ministral_ministral_ministral_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 10, 14336), torch.float32), ((14336, 4096), torch.float32)],
            {"model_names": ["pt_ministral_ministral_ministral_3b_instruct_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 10, 4096), torch.float32), ((4096, 32000), torch.float32)],
        {"model_names": ["pt_ministral_ministral_ministral_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 4096), torch.float32), ((4096, 4096), torch.float32)],
        {"model_names": ["pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul23,
        [((1, 64, 1), torch.float32)],
        {"model_names": ["pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {"model_names": ["pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 8, 128), torch.float32), ((32, 128, 8), torch.float32)],
        {"model_names": ["pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 8, 8), torch.float32), ((32, 8, 128), torch.float32)],
        {"model_names": ["pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 4096), torch.float32), ((4096, 12288), torch.float32)],
        {"model_names": ["pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 8, 12288), torch.float32), ((12288, 4096), torch.float32)],
            {"model_names": ["pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 8, 4096), torch.float32), ((4096, 131072), torch.float32)],
        {"model_names": ["pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((135, 4096), torch.float32), ((4096, 4096), torch.float32)],
        {"model_names": ["pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul24,
        [((1, 64, 1), torch.float32)],
        {"model_names": ["pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((135, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {"model_names": ["pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 135, 128), torch.float32), ((32, 128, 135), torch.float32)],
        {"model_names": ["pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 135, 135), torch.float32), ((32, 135, 128), torch.float32)],
        {"model_names": ["pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((135, 4096), torch.float32), ((4096, 14336), torch.float32)],
        {"model_names": ["pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 135, 14336), torch.float32), ((14336, 4096), torch.float32)],
            {"model_names": ["pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 135, 4096), torch.float32), ((4096, 32768), torch.float32)],
        {"model_names": ["pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul25,
        [((1, 16, 1), torch.float32)],
        {
            "model_names": [
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 256, 80), torch.float32), ((32, 80, 256), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf", "pt_phi2_microsoft_phi_2_pytdml_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 256, 256), torch.float32), ((32, 256, 80), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf", "pt_phi2_microsoft_phi_2_pytdml_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 2560), torch.float32), ((2560, 10240), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf", "pt_phi2_microsoft_phi_2_pytdml_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 2560), torch.float32), ((2560, 51200), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf", "pt_phi2_microsoft_phi_2_pytdml_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 2560), torch.float32), ((2560, 2560), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul26,
        [((1, 16, 1), torch.float32)],
        {
            "model_names": [
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 12, 80), torch.float32), ((32, 80, 12), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 12, 12), torch.float32), ((32, 12, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 2560), torch.float32), ((2560, 10240), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 12, 10240), torch.float32), ((10240, 2560), torch.float32)],
            {
                "model_names": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 12, 2560), torch.float32), ((2560, 2), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((11, 2560), torch.float32), ((2560, 2560), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_seq_cls_hf", "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul27,
        [((1, 16, 1), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_seq_cls_hf", "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 11, 80), torch.float32), ((32, 80, 11), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_seq_cls_hf", "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 11, 11), torch.float32), ((32, 11, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_seq_cls_hf", "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((11, 2560), torch.float32), ((2560, 10240), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_seq_cls_hf", "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 11, 10240), torch.float32), ((10240, 2560), torch.float32)],
            {
                "model_names": ["pt_phi2_microsoft_phi_2_seq_cls_hf", "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 11, 2560), torch.float32), ((2560, 2), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_seq_cls_hf", "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256, 3072), torch.float32), ((3072, 9216), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul25,
        [((1, 48, 1), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 256, 96), torch.float32), ((32, 96, 256), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 256, 256), torch.float32), ((32, 256, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((256, 3072), torch.float32), ((3072, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((256, 3072), torch.float32), ((3072, 8192), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256, 8192), torch.float32), ((8192, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256, 3072), torch.float32), ((3072, 32064), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 5, 3072), torch.float32), ((3072, 9216), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul28,
        [((1, 48, 1), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 5, 96), torch.float32), ((32, 96, 5), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 5, 5), torch.float32), ((32, 5, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((5, 3072), torch.float32), ((3072, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((5, 3072), torch.float32), ((3072, 8192), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 5, 8192), torch.float32), ((8192, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 5, 3072), torch.float32), ((3072, 2), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 13, 3072), torch.float32), ((3072, 9216), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul29,
        [((1, 48, 1), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 13, 96), torch.float32), ((32, 96, 13), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 13, 13), torch.float32), ((32, 13, 96), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((13, 3072), torch.float32), ((3072, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((13, 3072), torch.float32), ((3072, 8192), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 13, 8192), torch.float32), ((8192, 3072), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 13, 3072), torch.float32), ((3072, 2), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 5, 5120), torch.float32), ((5120, 7680), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99},
    ),
    (Matmul28, [((1, 64, 1), torch.float32)], {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99}),
    (
        Matmul1,
        [((40, 5, 128), torch.float32), ((40, 128, 5), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((40, 5, 5), torch.float32), ((40, 5, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((5, 5120), torch.float32), ((5120, 5120), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((5, 5120), torch.float32), ((5120, 17920), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 5, 17920), torch.float32), ((17920, 5120), torch.float32)],
            {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((5, 5120), torch.float32), ((5120, 2), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 12, 5120), torch.float32), ((5120, 7680), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99},
    ),
    (Matmul26, [((1, 64, 1), torch.float32)], {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99}),
    (
        Matmul1,
        [((40, 12, 128), torch.float32), ((40, 128, 12), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((40, 12, 12), torch.float32), ((40, 12, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 5120), torch.float32), ((5120, 5120), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 5120), torch.float32), ((5120, 17920), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 12, 17920), torch.float32), ((17920, 5120), torch.float32)],
            {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 12, 5120), torch.float32), ((5120, 2), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((35, 1536), torch.float32), ((1536, 1536), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul30,
        [((1, 64, 1), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((35, 1536), torch.float32), ((1536, 256), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 35, 128), torch.float32), ((12, 128, 35), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 35, 35), torch.float32), ((12, 35, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((35, 1536), torch.float32), ((1536, 8960), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 35, 8960), torch.float32), ((8960, 1536), torch.float32)],
            {
                "model_names": [
                    "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                    "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 35, 1536), torch.float32), ((1536, 151936), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((35, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((35, 2048), torch.float32), ((2048, 256), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 35, 128), torch.float32), ((16, 128, 35), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 35, 35), torch.float32), ((16, 35, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((35, 2048), torch.float32), ((2048, 11008), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 35, 11008), torch.float32), ((11008, 2048), torch.float32)],
            {
                "model_names": [
                    "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                    "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 35, 2048), torch.float32), ((2048, 151936), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((35, 3584), torch.float32), ((3584, 3584), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((35, 3584), torch.float32), ((3584, 512), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((28, 35, 128), torch.float32), ((28, 128, 35), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((28, 35, 35), torch.float32), ((28, 35, 128), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((35, 3584), torch.float32), ((3584, 18944), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 35, 18944), torch.float32), ((18944, 3584), torch.float32)],
            {
                "model_names": [
                    "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                    "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 35, 3584), torch.float32), ((3584, 152064), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((29, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul31,
        [((1, 64, 1), torch.float32)],
        {
            "model_names": [
                "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_instruct_1m_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((29, 2048), torch.float32), ((2048, 256), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 29, 128), torch.float32), ((16, 128, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 29, 29), torch.float32), ((16, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 2048), torch.float32), ((2048, 11008), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 29, 11008), torch.float32), ((11008, 2048), torch.float32)],
            {"model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 29, 2048), torch.float32), ((2048, 151936), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 2048), torch.float32), ((2048, 256), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 39, 128), torch.float32), ((16, 128, 39), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 39, 39), torch.float32), ((16, 39, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 2048), torch.float32), ((2048, 11008), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 39, 11008), torch.float32), ((11008, 2048), torch.float32)],
            {"model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 39, 2048), torch.float32), ((2048, 151936), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 3584), torch.float32), ((3584, 3584), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_7b_instruct_1m_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((29, 3584), torch.float32), ((3584, 512), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_7b_instruct_1m_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((28, 29, 128), torch.float32), ((28, 128, 29), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_7b_instruct_1m_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((28, 29, 29), torch.float32), ((28, 29, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_7b_instruct_1m_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((29, 3584), torch.float32), ((3584, 18944), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_7b_instruct_1m_clm_hf"],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 29, 18944), torch.float32), ((18944, 3584), torch.float32)],
            {
                "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_7b_instruct_1m_clm_hf"],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 29, 3584), torch.float32), ((3584, 152064), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_7b_instruct_1m_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((39, 3584), torch.float32), ((3584, 3584), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 3584), torch.float32), ((3584, 512), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((28, 39, 128), torch.float32), ((28, 128, 39), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((28, 39, 39), torch.float32), ((28, 39, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 3584), torch.float32), ((3584, 18944), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 39, 18944), torch.float32), ((18944, 3584), torch.float32)],
            {"model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 39, 3584), torch.float32), ((3584, 152064), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((38, 3584), torch.float32), ((3584, 3584), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_math_7b_clm_hf"], "pcc": 0.99},
    ),
    (Matmul32, [((1, 64, 1), torch.float32)], {"model_names": ["pt_qwen_v2_qwen_qwen2_5_math_7b_clm_hf"], "pcc": 0.99}),
    (
        Matmul1,
        [((38, 3584), torch.float32), ((3584, 512), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_math_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((28, 38, 128), torch.float32), ((28, 128, 38), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_math_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((28, 38, 38), torch.float32), ((28, 38, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_math_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((38, 3584), torch.float32), ((3584, 18944), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_math_7b_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 38, 18944), torch.float32), ((18944, 3584), torch.float32)],
            {"model_names": ["pt_qwen_v2_qwen_qwen2_5_math_7b_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 38, 3584), torch.float32), ((3584, 152064), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_math_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((13, 3584), torch.float32), ((3584, 3584), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99},
    ),
    (Matmul29, [((1, 64, 1), torch.float32)], {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99}),
    (
        Matmul1,
        [((13, 3584), torch.float32), ((3584, 512), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((28, 13, 128), torch.float32), ((28, 128, 13), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((28, 13, 13), torch.float32), ((28, 13, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((13, 3584), torch.float32), ((3584, 18944), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 13, 18944), torch.float32), ((18944, 3584), torch.float32)],
            {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 13, 3584), torch.float32), ((3584, 2), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((513, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 513, 64), torch.float32), ((12, 64, 513), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 513, 513), torch.float32), ((12, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((61, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 61, 64), torch.float32), ((12, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 61, 61), torch.float32), ((12, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((61, 768), torch.float32), ((768, 2048), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 61, 2048), torch.float32), ((2048, 768), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 513, 64), torch.float32), ((12, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 513, 61), torch.float32), ((12, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((513, 768), torch.float32), ((768, 2048), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 513, 2048), torch.float32), ((2048, 768), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 513, 768), torch.float32), ((768, 32128), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((513, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 513, 64), torch.float32), ((16, 64, 513), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 513, 513), torch.float32), ((16, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((61, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 61, 64), torch.float32), ((16, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 61, 61), torch.float32), ((16, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((61, 1024), torch.float32), ((1024, 2816), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 61, 2816), torch.float32), ((2816, 1024), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 513, 64), torch.float32), ((16, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 513, 61), torch.float32), ((16, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((513, 1024), torch.float32), ((1024, 2816), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 513, 2816), torch.float32), ((2816, 1024), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 513, 1024), torch.float32), ((1024, 32128), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_large_text_gen_hf", "pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((101, 1280), torch.float32), ((1280, 1280), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 101, 64), torch.float32), ((20, 64, 101), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 101, 101), torch.float32), ((20, 101, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 101, 1280), torch.float32), ((1280, 1280), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1500, 1280), torch.float32), ((1280, 1280), torch.float32)],
        {
            "model_names": [
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((20, 1500, 64), torch.float32), ((20, 64, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 1500, 1500), torch.float32), ((20, 1500, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 1280), torch.float32), ((1280, 5120), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 5120), torch.float32), ((5120, 1280), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 101, 64), torch.float32), ((20, 64, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 101, 1500), torch.float32), ((20, 1500, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 101, 1280), torch.float32), ((1280, 5120), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 101, 5120), torch.float32), ((5120, 1280), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 101, 1280), torch.float32), ((1280, 51865), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul33,
        [((8, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 8, 64), torch.float32), ((12, 64, 8), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 8, 8), torch.float32), ((12, 8, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul34,
        [((1, 8, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul35,
        [((1, 8, 3072), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul33,
        [((1, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_bert_base_japanese_seq_cls_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul36,
        [((1, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_bert_base_japanese_seq_cls_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul33,
        [((11, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 11, 64), torch.float32), ((12, 64, 11), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 11, 11), torch.float32), ((12, 11, 64), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul34,
        [((1, 11, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul35,
        [((1, 11, 3072), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul37,
        [((1, 1024), torch.float32)],
        {
            "model_names": [
                "pd_googlenet_base_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (Matmul38, [((1, 1152), torch.float32)], {"model_names": ["pd_googlenet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Matmul39,
        [((1, 1280), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 128), torch.float32), ((128, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((128, 768), torch.float32), ((768, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v2_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_albert_base_v1_token_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 128, 64), torch.float32), ((12, 64, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v2_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_albert_base_v1_token_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 128, 128), torch.float32), ((12, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v2_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_albert_base_v1_token_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 768), torch.float32), ((768, 3072), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v2_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_albert_base_v1_token_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 3072), torch.float32), ((3072, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v2_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_albert_base_v1_token_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_albert_base_v2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 768), torch.float32), ((768, 128), torch.float32)],
        {"model_names": ["pt_albert_base_v2_mlm_hf", "pt_albert_base_v1_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 128), torch.float32), ((128, 30000), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 128), torch.float32), ((128, 2048), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 128, 128), torch.float32), ((16, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 2048), torch.float32), ((2048, 128), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v1_mlm_hf", "pt_albert_xlarge_v2_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 128), torch.float32), ((128, 4096), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((64, 128, 64), torch.float32), ((64, 64, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((64, 128, 128), torch.float32), ((64, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 4096), torch.float32), ((4096, 16384), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 128, 16384), torch.float32), ((16384, 4096), torch.float32)],
            {
                "model_names": [
                    "pt_albert_xxlarge_v2_mlm_hf",
                    "pt_albert_xxlarge_v1_token_cls_hf",
                    "pt_albert_xxlarge_v1_mlm_hf",
                    "pt_albert_xxlarge_v2_token_cls_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 128, 4096), torch.float32), ((4096, 128), torch.float32)],
        {"model_names": ["pt_albert_xxlarge_v2_mlm_hf", "pt_albert_xxlarge_v1_mlm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 9216), torch.bfloat16), ((9216, 4096), torch.bfloat16)],
            {
                "model_names": [
                    "pt_alexnet_base_img_cls_osmr",
                    "pt_alexnet_base_img_cls_torchhub",
                    "pt_rcnn_base_obj_det_torchvision_rect_0",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 4096), torch.bfloat16), ((4096, 4096), torch.bfloat16)],
            {
                "model_names": [
                    "pt_alexnet_base_img_cls_osmr",
                    "pt_alexnet_base_img_cls_torchhub",
                    "pt_rcnn_base_obj_det_torchvision_rect_0",
                    "pt_vgg_bn_vgg19_obj_det_osmr",
                    "pt_vgg_vgg11_bn_img_cls_torchvision",
                    "pt_vgg_vgg13_bn_img_cls_torchvision",
                    "pt_vgg_vgg16_bn_img_cls_torchvision",
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
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 4096), torch.bfloat16), ((4096, 1000), torch.bfloat16)],
            {
                "model_names": [
                    "pt_alexnet_base_img_cls_osmr",
                    "pt_alexnet_base_img_cls_torchhub",
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
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((256, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 256, 64), torch.float32), ((16, 64, 256), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 256, 256), torch.float32), ((16, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_names": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {
            "model_names": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf", "pt_xglm_facebook_xglm_564m_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 768), torch.float32), ((768, 768), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 16, 64), torch.float32), ((12, 64, 16), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 16, 16), torch.float32), ((12, 16, 64), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 16, 768), torch.float32), ((768, 3072), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 16, 3072), torch.float32), ((3072, 768), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 768), torch.float32), ((768, 768), torch.float32)],
        {
            "model_names": [
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((256, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256, 1024), torch.float32), ((1024, 51200), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((197, 768), torch.bfloat16), ((768, 768), torch.bfloat16)],
            {
                "model_names": [
                    "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                    "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                    "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                    "pt_vit_vit_b_16_img_cls_torchvision",
                    "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((12, 197, 64), torch.bfloat16), ((12, 64, 197), torch.bfloat16)],
        {
            "model_names": [
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((12, 197, 197), torch.bfloat16), ((12, 197, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 197, 768), torch.bfloat16), ((768, 3072), torch.bfloat16)],
            {
                "model_names": [
                    "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                    "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                    "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                    "pt_vit_vit_b_16_img_cls_torchvision",
                    "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 197, 3072), torch.bfloat16), ((3072, 768), torch.bfloat16)],
            {
                "model_names": [
                    "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                    "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                    "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                    "pt_vit_vit_b_16_img_cls_torchvision",
                    "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 768), torch.bfloat16), ((768, 1000), torch.bfloat16)],
            {
                "model_names": [
                    "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                    "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                    "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                    "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                    "pt_swin_swin_t_img_cls_torchvision",
                    "pt_vit_vit_b_32_img_cls_torchvision",
                    "pt_swin_swin_s_img_cls_torchvision",
                    "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                    "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                    "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                    "pt_vit_vit_b_16_img_cls_torchvision",
                    "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                    "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((197, 384), torch.bfloat16), ((384, 384), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((6, 197, 64), torch.bfloat16), ((6, 64, 197), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((6, 197, 197), torch.bfloat16), ((6, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 197, 384), torch.bfloat16), ((384, 1536), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 197, 1536), torch.bfloat16), ((1536, 384), torch.bfloat16)],
            {
                "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 384), torch.bfloat16), ((384, 1000), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 2208), torch.bfloat16), ((2208, 1000), torch.bfloat16)],
            {
                "model_names": ["pt_densenet_densenet161_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1664), torch.bfloat16), ((1664, 1000), torch.bfloat16)],
            {
                "model_names": ["pt_densenet_densenet169_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1280), torch.bfloat16), ((1280, 1000), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b0_img_cls_timm",
                    "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                    "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                    "pt_ghostnet_ghostnet_100_img_cls_timm",
                    "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                    "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                    "pt_vit_vit_h_14_img_cls_torchvision",
                    "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                    "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                    "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                    "pt_mobilenetv2_basic_img_cls_torchhub",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1792), torch.bfloat16), ((1792, 1000), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                    "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                    "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b4_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1280), torch.bfloat16), ((1280, 21843), torch.bfloat16)],
            {
                "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 19200, 64), torch.bfloat16), ((64, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((300, 64), torch.bfloat16), ((64, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 19200, 64), torch.bfloat16), ((1, 64, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 19200, 300), torch.bfloat16), ((1, 300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 19200, 64), torch.bfloat16), ((64, 256), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 19200, 256), torch.bfloat16), ((256, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 4800, 128), torch.bfloat16), ((128, 128), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((300, 128), torch.bfloat16), ((128, 128), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((2, 4800, 64), torch.bfloat16), ((2, 64, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((2, 4800, 300), torch.bfloat16), ((2, 300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((4800, 128), torch.bfloat16), ((128, 128), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 4800, 128), torch.bfloat16), ((128, 512), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 4800, 512), torch.bfloat16), ((512, 128), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 1200, 320), torch.bfloat16), ((320, 320), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((300, 320), torch.bfloat16), ((320, 320), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((5, 1200, 64), torch.bfloat16), ((5, 64, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((5, 1200, 300), torch.bfloat16), ((5, 300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1200, 320), torch.bfloat16), ((320, 320), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 1200, 320), torch.bfloat16), ((320, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1200, 1280), torch.bfloat16), ((1280, 320), torch.bfloat16)],
            {
                "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((300, 512), torch.bfloat16), ((512, 512), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((8, 300, 64), torch.bfloat16), ((8, 64, 300), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((8, 300, 300), torch.bfloat16), ((8, 300, 64), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 300, 512), torch.bfloat16), ((512, 2048), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 300, 2048), torch.bfloat16), ((2048, 512), torch.bfloat16)],
            {
                "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1024), torch.bfloat16), ((1024, 1000), torch.bfloat16)],
            {
                "model_names": [
                    "pt_googlenet_base_img_cls_torchvision",
                    "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                    "pt_mlp_mixer_mixer_l32_224_img_cls_timm",
                    "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                    "pt_swin_swin_b_img_cls_torchvision",
                    "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                    "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
                    "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                    "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                    "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                    "pt_vit_vit_l_32_img_cls_torchvision",
                    "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                    "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                    "pt_vovnet_vovnet39_img_cls_osmr",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                    "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                    "pt_vit_vit_l_16_img_cls_torchvision",
                    "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                    "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                    "pt_vovnet_vovnet57_img_cls_osmr",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((5, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 5, 64), torch.float32), ((12, 64, 5), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 5, 5), torch.float32), ((12, 5, 64), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 5, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 5, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 5, 768), torch.float32), ((768, 2), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1536), torch.bfloat16), ((1536, 1000), torch.bfloat16)],
            {
                "model_names": [
                    "pt_inception_v4_img_cls_osmr",
                    "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                    "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                    "pt_inception_inception_v4_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((4, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul21,
        [((1, 32, 1), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((4, 2048), torch.float32), ((2048, 512), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 4, 64), torch.float32), ((32, 64, 4), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 4, 4), torch.float32), ((32, 4, 64), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((4, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 4, 8192), torch.float32), ((8192, 2048), torch.float32)],
            {
                "model_names": [
                    "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                    "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 4, 2048), torch.float32), ((2048, 2), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 257, 768), torch.bfloat16), ((768, 2304), torch.bfloat16)],
            {
                "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((12, 257, 64), torch.bfloat16), ((12, 64, 257), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((12, 257, 257), torch.bfloat16), ((12, 257, 64), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((257, 768), torch.bfloat16), ((768, 768), torch.bfloat16)],
            {
                "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 257, 768), torch.bfloat16), ((768, 3072), torch.bfloat16)],
            {
                "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 257, 3072), torch.bfloat16), ((3072, 768), torch.bfloat16)],
            {
                "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 27, 257), torch.bfloat16), ((1, 257, 768), torch.bfloat16)],
        {
            "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 27, 768), torch.bfloat16), ((768, 38), torch.bfloat16)],
            {
                "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 27, 768), torch.bfloat16), ((768, 50257), torch.bfloat16)],
            {
                "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 27, 768), torch.bfloat16), ((768, 30522), torch.bfloat16)],
            {
                "model_names": ["pt_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 768, 196), torch.bfloat16), ((196, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 768, 384), torch.bfloat16), ((384, 196), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 196, 768), torch.bfloat16), ((768, 3072), torch.bfloat16)],
            {
                "model_names": [
                    "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                    "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                    "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                    "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                    "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 196, 3072), torch.bfloat16), ((3072, 768), torch.bfloat16)],
            {
                "model_names": [
                    "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                    "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                    "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                    "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                    "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 768), torch.bfloat16), ((768, 21843), torch.bfloat16)],
            {
                "model_names": ["pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 768), torch.bfloat16), ((768, 11221), torch.bfloat16)],
            {
                "model_names": ["pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 768, 49), torch.bfloat16), ((49, 384), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_b32_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((1, 768, 384), torch.bfloat16), ((384, 49), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_b32_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 49, 768), torch.bfloat16), ((768, 3072), torch.bfloat16)],
            {
                "model_names": [
                    "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                    "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 49, 3072), torch.bfloat16), ((3072, 768), torch.bfloat16)],
            {
                "model_names": [
                    "pt_mlp_mixer_mixer_b32_224_img_cls_timm",
                    "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 1024, 196), torch.bfloat16), ((196, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 1024, 512), torch.bfloat16), ((512, 196), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 196, 1024), torch.bfloat16), ((1024, 4096), torch.bfloat16)],
            {
                "model_names": [
                    "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                    "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 196, 4096), torch.bfloat16), ((4096, 1024), torch.bfloat16)],
            {
                "model_names": [
                    "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                    "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 1024, 49), torch.bfloat16), ((49, 512), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((1, 1024, 512), torch.bfloat16), ((512, 49), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 49, 1024), torch.bfloat16), ((1024, 4096), torch.bfloat16)],
            {
                "model_names": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 49, 4096), torch.bfloat16), ((4096, 1024), torch.bfloat16)],
            {
                "model_names": ["pt_mlp_mixer_mixer_l32_224_img_cls_timm"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 9216), torch.bfloat16), ((9216, 128), torch.bfloat16)],
            {"model_names": ["pt_mnist_base_img_cls_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 128), torch.bfloat16), ((128, 10), torch.bfloat16)],
        {"model_names": ["pt_mnist_base_img_cls_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1280), torch.bfloat16), ((1280, 1001), torch.bfloat16)],
            {
                "model_names": [
                    "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                    "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                    "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 960), torch.bfloat16), ((960, 1280), torch.bfloat16)],
            {
                "model_names": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 576), torch.bfloat16), ((576, 1024), torch.bfloat16)],
            {
                "model_names": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 512), torch.bfloat16), ((512, 1000), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_ssd_resnet18_img_cls_torchvision",
                "pt_resnet_resnet18_img_cls_torchvision",
                "pt_resnet_resnet34_img_cls_torchvision",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_mlp_mixer_base_img_cls_github",
                "pt_mlp_mixer_mixer_s16_224_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_mlp_mixer_mixer_s32_224_img_cls_timm",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((150, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 150, 64), torch.float32), ((12, 64, 150), torch.float32)],
        {"model_names": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 150, 150), torch.float32), ((12, 150, 64), torch.float32)],
        {"model_names": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul33,
        [((150, 768), torch.float32)],
        {"model_names": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul34,
        [((150, 768), torch.float32)],
        {"model_names": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul35,
        [((150, 3072), torch.float32)],
        {"model_names": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 72), torch.float32), ((72, 512), torch.float32)],
        {"model_names": ["pt_nbeats_generic_basis_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 512), torch.float32), ((512, 512), torch.float32)],
        {"model_names": ["pt_nbeats_generic_basis_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 512), torch.float32), ((512, 96), torch.float32)],
        {"model_names": ["pt_nbeats_generic_basis_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 72), torch.float32), ((72, 2048), torch.float32)],
        {"model_names": ["pt_nbeats_seasionality_basis_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_nbeats_seasionality_basis_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 2048), torch.float32), ((2048, 48), torch.float32)],
        {"model_names": ["pt_nbeats_seasionality_basis_clm_hf"], "pcc": 0.99},
    ),
    (Matmul40, [((1024, 12), torch.float32)], {"model_names": ["pt_nbeats_seasionality_basis_clm_hf"], "pcc": 0.99}),
    (Matmul41, [((1024, 12), torch.float32)], {"model_names": ["pt_nbeats_seasionality_basis_clm_hf"], "pcc": 0.99}),
    (
        Matmul1,
        [((1024, 72), torch.float32), ((72, 256), torch.float32)],
        {"model_names": ["pt_nbeats_trend_basis_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 256), torch.float32), ((256, 256), torch.float32)],
        {"model_names": ["pt_nbeats_trend_basis_clm_hf", "pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 256), torch.float32), ((256, 8), torch.float32)],
        {"model_names": ["pt_nbeats_trend_basis_clm_hf"], "pcc": 0.99},
    ),
    (Matmul42, [((1024, 4), torch.float32)], {"model_names": ["pt_nbeats_trend_basis_clm_hf"], "pcc": 0.99}),
    (Matmul43, [((1024, 4), torch.float32)], {"model_names": ["pt_nbeats_trend_basis_clm_hf"], "pcc": 0.99}),
    pytest.param(
        (
            Matmul1,
            [((1, 1, 1024), torch.bfloat16), ((1024, 1024), torch.bfloat16)],
            {
                "model_names": [
                    "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                    "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                    "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 512, 1024), torch.bfloat16), ((1024, 322), torch.bfloat16)],
            {
                "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((3025, 322), torch.bfloat16), ((322, 322), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 512, 322), torch.bfloat16), ((1, 322, 3025), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 512, 3025), torch.bfloat16), ((1, 3025, 322), torch.bfloat16)],
            {
                "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 512, 322), torch.bfloat16), ((322, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 512, 1024), torch.bfloat16), ((1024, 1024), torch.bfloat16)],
            {
                "model_names": [
                    "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                    "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                    "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((512, 1024), torch.bfloat16), ((1024, 1024), torch.bfloat16)],
            {
                "model_names": [
                    "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                    "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                    "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((8, 512, 128), torch.bfloat16), ((8, 128, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((8, 512, 512), torch.bfloat16), ((8, 512, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1, 1024), torch.bfloat16), ((1, 1024, 512), torch.bfloat16)],
            {
                "model_names": [
                    "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                    "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                    "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 1, 512), torch.bfloat16), ((1, 512, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1, 1024), torch.bfloat16), ((1024, 1000), torch.bfloat16)],
            {
                "model_names": [
                    "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                    "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                    "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 512, 1024), torch.bfloat16), ((1024, 261), torch.bfloat16)],
            {
                "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((50176, 261), torch.bfloat16), ((261, 261), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 512, 261), torch.bfloat16), ((1, 261, 50176), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 512, 50176), torch.bfloat16), ((1, 50176, 261), torch.bfloat16)],
            {
                "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 512, 261), torch.bfloat16), ((261, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 512, 1024), torch.bfloat16), ((1024, 512), torch.bfloat16)],
            {
                "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 50176, 256), torch.bfloat16), ((256, 256), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((50176, 512), torch.bfloat16), ((512, 512), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 512, 512), torch.bfloat16), ((1, 512, 50176), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 512, 50176), torch.bfloat16), ((1, 50176, 512), torch.bfloat16)],
            {
                "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 512, 512), torch.bfloat16), ((512, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((5, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul28,
        [((1, 16, 1), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 5, 64), torch.float32), ((32, 64, 5), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 5, 5), torch.float32), ((32, 5, 64), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((5, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_seq_cls_hf", "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 5, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 5, 2048), torch.float32), ((2048, 2), torch.float32)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_token_cls_hf", "pt_phi1_microsoft_phi_1_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 12, 64), torch.float32), ((32, 64, 12), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_token_cls_hf", "pt_phi1_microsoft_phi_1_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 12, 12), torch.float32), ((32, 12, 64), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_token_cls_hf", "pt_phi1_microsoft_phi_1_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_token_cls_hf", "pt_phi1_microsoft_phi_1_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 12, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_token_cls_hf", "pt_phi1_microsoft_phi_1_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 12, 2048), torch.float32), ((2048, 2), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_token_cls_hf", "pt_phi1_microsoft_phi_1_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((39, 1536), torch.float32), ((1536, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 1536), torch.float32), ((1536, 256), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 39, 128), torch.float32), ((12, 128, 39), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 39, 39), torch.float32), ((12, 39, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 1536), torch.float32), ((1536, 8960), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 39, 8960), torch.float32), ((8960, 1536), torch.float32)],
            {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 39, 1536), torch.float32), ((1536, 151936), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 4096), torch.bfloat16), ((4096, 2), torch.bfloat16)],
        {"model_names": ["pt_rcnn_base_obj_det_torchvision_rect_0"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1088), torch.bfloat16), ((1088, 1000), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1296), torch.bfloat16), ((1296, 1000), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_facebook_regnet_y_064_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 2016), torch.bfloat16), ((2016, 1000), torch.bfloat16)],
            {
                "model_names": [
                    "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                    "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 2240), torch.bfloat16), ((2240, 1000), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_facebook_regnet_y_120_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 400), torch.bfloat16), ((400, 1000), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 672), torch.bfloat16), ((672, 1000), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_x_800mf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1920), torch.bfloat16), ((1920, 1000), torch.bfloat16)],
            {
                "model_names": [
                    "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                    "pt_densenet_densenet201_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 3024), torch.bfloat16), ((3024, 1000), torch.bfloat16)],
            {
                "model_names": [
                    "pt_regnet_regnet_y_16gf_img_cls_torchvision",
                    "pt_regnet_facebook_regnet_y_160_img_cls_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 888), torch.bfloat16), ((888, 1000), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 3712), torch.bfloat16), ((3712, 1000), torch.bfloat16)],
            {
                "model_names": [
                    "pt_regnet_regnet_y_32gf_img_cls_torchvision",
                    "pt_regnet_facebook_regnet_y_320_img_cls_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1512), torch.bfloat16), ((1512, 1000), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 440), torch.bfloat16), ((440, 1000), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 784), torch.bfloat16), ((784, 1000), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 768), torch.float32), ((768, 3), torch.float32)],
        {
            "model_names": [
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 768), torch.float32), ((768, 768), torch.float32)],
        {
            "model_names": [
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 768), torch.float32), ((768, 250002), torch.float32)],
        {"model_names": ["pt_roberta_xlm_roberta_base_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 16384, 64), torch.bfloat16), ((64, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((256, 64), torch.bfloat16), ((64, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 16384, 64), torch.bfloat16), ((1, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 16384, 256), torch.bfloat16), ((1, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 16384, 64), torch.bfloat16), ((64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 16384, 256), torch.bfloat16), ((256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 4096, 128), torch.bfloat16), ((128, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((256, 128), torch.bfloat16), ((128, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((2, 4096, 64), torch.bfloat16), ((2, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((2, 4096, 256), torch.bfloat16), ((2, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((4096, 128), torch.bfloat16), ((128, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 4096, 128), torch.bfloat16), ((128, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 4096, 512), torch.bfloat16), ((512, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 1024, 320), torch.bfloat16), ((320, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((256, 320), torch.bfloat16), ((320, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((5, 1024, 64), torch.bfloat16), ((5, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((5, 1024, 256), torch.bfloat16), ((5, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1024, 320), torch.bfloat16), ((320, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 1024, 320), torch.bfloat16), ((320, 1280), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1024, 1280), torch.bfloat16), ((1280, 320), torch.bfloat16)],
            {
                "model_names": [
                    "pt_segformer_nvidia_mit_b2_img_cls_hf",
                    "pt_segformer_nvidia_mit_b4_img_cls_hf",
                    "pt_segformer_nvidia_mit_b5_img_cls_hf",
                    "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_mit_b1_img_cls_hf",
                    "pt_segformer_nvidia_mit_b3_img_cls_hf",
                    "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((256, 512), torch.bfloat16), ((512, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((8, 256, 64), torch.bfloat16), ((8, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((8, 256, 256), torch.bfloat16), ((8, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 256, 512), torch.bfloat16), ((512, 2048), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 256, 2048), torch.bfloat16), ((2048, 512), torch.bfloat16)],
            {
                "model_names": [
                    "pt_segformer_nvidia_mit_b2_img_cls_hf",
                    "pt_segformer_nvidia_mit_b4_img_cls_hf",
                    "pt_segformer_nvidia_mit_b5_img_cls_hf",
                    "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_mit_b1_img_cls_hf",
                    "pt_segformer_nvidia_mit_b3_img_cls_hf",
                    "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 16384, 32), torch.bfloat16), ((32, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((256, 32), torch.bfloat16), ((32, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 16384, 32), torch.bfloat16), ((1, 32, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 16384, 256), torch.bfloat16), ((1, 256, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 16384, 32), torch.bfloat16), ((32, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 16384, 128), torch.bfloat16), ((128, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 4096, 64), torch.bfloat16), ((64, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((2, 4096, 32), torch.bfloat16), ((2, 32, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((2, 4096, 256), torch.bfloat16), ((2, 256, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((4096, 64), torch.bfloat16), ((64, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 4096, 64), torch.bfloat16), ((64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 4096, 256), torch.bfloat16), ((256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 1024, 160), torch.bfloat16), ((160, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((256, 160), torch.bfloat16), ((160, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((5, 1024, 32), torch.bfloat16), ((5, 32, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((5, 1024, 256), torch.bfloat16), ((5, 256, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1024, 160), torch.bfloat16), ((160, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 1024, 160), torch.bfloat16), ((160, 640), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1024, 640), torch.bfloat16), ((640, 160), torch.bfloat16)],
            {
                "model_names": [
                    "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_mit_b0_img_cls_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((256, 256), torch.bfloat16), ((256, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((8, 256, 32), torch.bfloat16), ((8, 32, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((8, 256, 256), torch.bfloat16), ((8, 256, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 256, 256), torch.bfloat16), ((256, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 256, 1024), torch.bfloat16), ((1024, 256), torch.bfloat16)],
            {
                "model_names": [
                    "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                    "pt_segformer_nvidia_mit_b0_img_cls_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 256, 256), torch.bfloat16), ((256, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 1024, 160), torch.bfloat16), ((160, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 16384, 32), torch.bfloat16), ((32, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 256, 512), torch.bfloat16), ((512, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_mlp_mixer_base_img_cls_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 1024, 320), torch.bfloat16), ((320, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 4096, 128), torch.bfloat16), ((128, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 256, 512), torch.bfloat16), ((512, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 1024, 320), torch.bfloat16), ((320, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 4096, 128), torch.bfloat16), ((128, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 16384, 64), torch.bfloat16), ((64, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((2, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 1, 64), torch.float32), ((32, 64, 1), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 1, 1), torch.float32), ((32, 1, 64), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((26, 768), torch.float32), ((768, 768), torch.float32)],
        {
            "model_names": [
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((24, 13, 64), torch.float32), ((24, 64, 13), torch.float32)],
        {
            "model_names": [
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((24, 13, 13), torch.float32), ((24, 13, 64), torch.float32)],
        {
            "model_names": [
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((26, 768), torch.float32), ((768, 3072), torch.float32)],
        {
            "model_names": [
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((26, 3072), torch.float32), ((3072, 768), torch.float32)],
        {
            "model_names": [
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((26, 768), torch.float32), ((768, 1024), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((26, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 1, 64), torch.float32), ((32, 64, 13), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 1, 13), torch.float32), ((32, 13, 64), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 1024), torch.float32), ((1024, 2048), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((3136, 96), torch.bfloat16), ((96, 96), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((192, 49, 32), torch.bfloat16), ((192, 32, 49), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((192, 49, 49), torch.bfloat16), ((192, 49, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 3136, 96), torch.bfloat16), ((96, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 3136, 384), torch.bfloat16), ((384, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 784, 384), torch.bfloat16), ((384, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((784, 192), torch.bfloat16), ((192, 192), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((96, 49, 32), torch.bfloat16), ((96, 32, 49), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((96, 49, 49), torch.bfloat16), ((96, 49, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 784, 192), torch.bfloat16), ((192, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 784, 768), torch.bfloat16), ((768, 192), torch.bfloat16)],
            {
                "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 196, 768), torch.bfloat16), ((768, 384), torch.bfloat16)],
            {
                "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((196, 384), torch.bfloat16), ((384, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((48, 49, 32), torch.bfloat16), ((48, 32, 49), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((48, 49, 49), torch.bfloat16), ((48, 49, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 196, 384), torch.bfloat16), ((384, 1536), torch.bfloat16)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 196, 1536), torch.bfloat16), ((1536, 384), torch.bfloat16)],
            {
                "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 49, 1536), torch.bfloat16), ((1536, 768), torch.bfloat16)],
            {
                "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((49, 768), torch.bfloat16), ((768, 768), torch.bfloat16)],
            {
                "model_names": [
                    "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                    "pt_swin_swin_t_img_cls_torchvision",
                    "pt_swin_swin_s_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((24, 49, 32), torch.bfloat16), ((24, 32, 49), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((24, 49, 49), torch.bfloat16), ((24, 49, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_swin_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((3136, 128), torch.bfloat16), ((128, 384), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((256, 49, 32), torch.bfloat16), ((256, 32, 49), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((256, 49, 49), torch.bfloat16), ((256, 49, 32), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((3136, 128), torch.bfloat16), ((128, 128), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((3136, 128), torch.bfloat16), ((128, 512), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((3136, 512), torch.bfloat16), ((512, 128), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((784, 512), torch.bfloat16), ((512, 256), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((784, 256), torch.bfloat16), ((256, 768), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((128, 49, 32), torch.bfloat16), ((128, 32, 49), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((128, 49, 49), torch.bfloat16), ((128, 49, 32), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((784, 256), torch.bfloat16), ((256, 256), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((784, 256), torch.bfloat16), ((256, 1024), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    pytest.param(
        (
            Matmul1,
            [((784, 1024), torch.bfloat16), ((1024, 256), torch.bfloat16)],
            {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((196, 1024), torch.bfloat16), ((1024, 512), torch.bfloat16)],
            {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((196, 512), torch.bfloat16), ((512, 1536), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((64, 49, 32), torch.bfloat16), ((64, 32, 49), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((64, 49, 49), torch.bfloat16), ((64, 49, 32), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((196, 512), torch.bfloat16), ((512, 512), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((196, 512), torch.bfloat16), ((512, 2048), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    pytest.param(
        (
            Matmul1,
            [((196, 2048), torch.bfloat16), ((2048, 512), torch.bfloat16)],
            {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((49, 2048), torch.bfloat16), ((2048, 1024), torch.bfloat16)],
            {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((49, 1024), torch.bfloat16), ((1024, 3072), torch.bfloat16)],
            {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((32, 49, 32), torch.bfloat16), ((32, 32, 49), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((32, 49, 49), torch.bfloat16), ((32, 49, 32), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    pytest.param(
        (
            Matmul1,
            [((49, 1024), torch.bfloat16), ((1024, 1024), torch.bfloat16)],
            {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((49, 1024), torch.bfloat16), ((1024, 4096), torch.bfloat16)],
            {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((49, 4096), torch.bfloat16), ((4096, 1024), torch.bfloat16)],
            {"model_names": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((3136, 96), torch.bfloat16), ((96, 288), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((3136, 96), torch.bfloat16), ((96, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((3136, 384), torch.bfloat16), ((384, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((784, 384), torch.bfloat16), ((384, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((784, 192), torch.bfloat16), ((192, 576), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((784, 192), torch.bfloat16), ((192, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((784, 768), torch.bfloat16), ((768, 192), torch.bfloat16)],
            {
                "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((196, 768), torch.bfloat16), ((768, 384), torch.bfloat16)],
            {
                "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((196, 384), torch.bfloat16), ((384, 1152), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((196, 384), torch.bfloat16), ((384, 1536), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((196, 1536), torch.bfloat16), ((1536, 384), torch.bfloat16)],
            {
                "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((49, 1536), torch.bfloat16), ((1536, 768), torch.bfloat16)],
            {
                "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((49, 768), torch.bfloat16), ((768, 2304), torch.bfloat16)],
            {
                "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((49, 768), torch.bfloat16), ((768, 3072), torch.bfloat16)],
            {
                "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((49, 3072), torch.bfloat16), ((3072, 768), torch.bfloat16)],
            {
                "model_names": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((4096, 128), torch.float32), ((128, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 64, 32), torch.float32), ((256, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((225, 2), torch.float32), ((2, 512), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_b_img_cls_torchvision",
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((225, 512), torch.float32), ((512, 4), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 64, 64), torch.float32), ((256, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((4096, 128), torch.float32), ((128, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((4096, 128), torch.float32), ((128, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((4096, 512), torch.float32), ((512, 128), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 512), torch.float32), ((512, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 256), torch.float32), ((256, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((128, 64, 32), torch.float32), ((128, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((225, 512), torch.float32), ((512, 8), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((128, 64, 64), torch.float32), ((128, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 256), torch.float32), ((256, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 1024), torch.float32), ((1024, 256), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 1024), torch.float32), ((1024, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision", "pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 512), torch.float32), ((512, 1536), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((64, 64, 32), torch.float32), ((64, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((225, 512), torch.float32), ((512, 16), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((64, 64, 64), torch.float32), ((64, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 512), torch.float32), ((512, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 512), torch.float32), ((512, 2048), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 2048), torch.float32), ((2048, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((64, 2048), torch.float32), ((2048, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((64, 1024), torch.float32), ((1024, 3072), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 64, 32), torch.float32), ((32, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((225, 512), torch.float32), ((512, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 64, 64), torch.float32), ((32, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((64, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((64, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((64, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((4096, 96), torch.float32), ((96, 288), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((192, 64, 32), torch.float32), ((192, 32, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((225, 512), torch.float32), ((512, 3), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((192, 64, 64), torch.float32), ((192, 64, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((4096, 96), torch.float32), ((96, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((4096, 96), torch.float32), ((96, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((4096, 384), torch.float32), ((384, 96), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1024, 384), torch.float32), ((384, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1024, 192), torch.float32), ((192, 576), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((96, 64, 32), torch.float32), ((96, 32, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((225, 512), torch.float32), ((512, 6), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((96, 64, 64), torch.float32), ((96, 64, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1024, 192), torch.float32), ((192, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1024, 192), torch.float32), ((192, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1024, 768), torch.float32), ((768, 192), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((256, 768), torch.float32), ((768, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((256, 384), torch.float32), ((384, 1152), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((48, 64, 32), torch.float32), ((48, 32, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((225, 512), torch.float32), ((512, 12), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((48, 64, 64), torch.float32), ((48, 64, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((256, 384), torch.float32), ((384, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((256, 384), torch.float32), ((384, 1536), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((256, 1536), torch.float32), ((1536, 384), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((64, 1536), torch.float32), ((1536, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((64, 768), torch.float32), ((768, 2304), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((24, 64, 32), torch.float32), ((24, 32, 64), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((225, 512), torch.float32), ((512, 24), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((24, 64, 64), torch.float32), ((24, 64, 32), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((64, 768), torch.float32), ((768, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((64, 768), torch.float32), ((768, 3072), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((64, 3072), torch.float32), ((3072, 768), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 768), torch.float32), ((768, 1000), torch.float32)],
        {
            "model_names": [
                "pt_swin_swin_v2_s_img_cls_torchvision",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "onnx_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 61, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {"model_names": ["pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 61, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {"model_names": ["pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 513, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {"model_names": ["pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 513, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {"model_names": ["pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 25088), torch.bfloat16), ((25088, 4096), torch.bfloat16)],
            {
                "model_names": [
                    "pt_vgg_bn_vgg19_obj_det_osmr",
                    "pt_vgg_vgg11_bn_img_cls_torchvision",
                    "pt_vgg_vgg13_bn_img_cls_torchvision",
                    "pt_vgg_vgg16_bn_img_cls_torchvision",
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
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((197, 1024), torch.bfloat16), ((1024, 1024), torch.bfloat16)],
            {
                "model_names": [
                    "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                    "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                    "pt_vit_vit_l_16_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((16, 197, 64), torch.bfloat16), ((16, 64, 197), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((16, 197, 197), torch.bfloat16), ((16, 197, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 197, 1024), torch.bfloat16), ((1024, 4096), torch.bfloat16)],
            {
                "model_names": [
                    "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                    "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                    "pt_vit_vit_l_16_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 197, 4096), torch.bfloat16), ((4096, 1024), torch.bfloat16)],
            {
                "model_names": [
                    "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                    "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                    "pt_vit_vit_l_16_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((50, 768), torch.bfloat16), ((768, 2304), torch.bfloat16)],
            {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((12, 50, 64), torch.bfloat16), ((12, 64, 50), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((12, 50, 50), torch.bfloat16), ((12, 50, 64), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    pytest.param(
        (
            Matmul1,
            [((50, 768), torch.bfloat16), ((768, 768), torch.bfloat16)],
            {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 50, 768), torch.bfloat16), ((768, 3072), torch.bfloat16)],
            {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 50, 3072), torch.bfloat16), ((3072, 768), torch.bfloat16)],
            {"model_names": ["pt_vit_vit_b_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1370, 1280), torch.bfloat16), ((1280, 3840), torch.bfloat16)],
            {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((16, 1370, 80), torch.bfloat16), ((16, 80, 1370), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    pytest.param(
        (
            Matmul1,
            [((16, 1370, 1370), torch.bfloat16), ((16, 1370, 80), torch.bfloat16)],
            {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1370, 1280), torch.bfloat16), ((1280, 1280), torch.bfloat16)],
            {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1370, 1280), torch.bfloat16), ((1280, 5120), torch.bfloat16)],
            {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1370, 5120), torch.bfloat16), ((5120, 1280), torch.bfloat16)],
            {"model_names": ["pt_vit_vit_h_14_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((101, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 101, 64), torch.float32), ((16, 64, 101), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 101, 101), torch.float32), ((16, 101, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 101, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1500, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 1500, 64), torch.float32), ((16, 64, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 1500, 1500), torch.float32), ((16, 1500, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 101, 64), torch.float32), ((16, 64, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 101, 1500), torch.float32), ((16, 1500, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 101, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 101, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 101, 1024), torch.float32), ((1024, 51865), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {
            "model_names": [
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 256, 128), torch.float32), ((16, 128, 256), torch.float32)],
        {"model_names": ["pt_xglm_facebook_xglm_1_7b_clm_hf", "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 256, 256), torch.float32), ((16, 256, 128), torch.float32)],
        {"model_names": ["pt_xglm_facebook_xglm_1_7b_clm_hf", "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {"model_names": ["pt_xglm_facebook_xglm_1_7b_clm_hf", "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {
            "model_names": [
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256, 2048), torch.float32), ((2048, 256008), torch.float32)],
        {"model_names": ["pt_xglm_facebook_xglm_1_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 400, 32), torch.bfloat16), ((2, 32, 400), torch.bfloat16)],
        {"model_names": ["pt_yolov10_yolov10n_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((2, 64, 400), torch.bfloat16), ((2, 400, 400), torch.bfloat16)],
        {"model_names": ["pt_yolov10_yolov10n_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((5, 400, 32), torch.bfloat16), ((5, 32, 400), torch.bfloat16)],
        {"model_names": ["pt_yolov10_yolov10x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((5, 64, 400), torch.bfloat16), ((5, 400, 400), torch.bfloat16)],
        {"model_names": ["pt_yolov10_yolov10x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul44,
        [((512, 128), torch.bfloat16)],
        {"model_names": ["pt_yolo_world_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul44,
        [((512, 64), torch.bfloat16)],
        {"model_names": ["pt_yolo_world_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((1, 80, 512), torch.bfloat16), ((512, 256), torch.bfloat16)],
        {"model_names": ["pt_yolo_world_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((1, 27, 256), torch.bfloat16), ((256, 256), torch.bfloat16)],
        {"model_names": ["pt_yolo_world_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((8, 80, 32), torch.bfloat16), ((8, 32, 27), torch.bfloat16)],
        {"model_names": ["pt_yolo_world_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((80, 256), torch.bfloat16), ((256, 512), torch.bfloat16)],
        {"model_names": ["pt_yolo_world_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((80, 512), torch.bfloat16), ((512, 128), torch.bfloat16)],
        {"model_names": ["pt_yolo_world_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((80, 512), torch.bfloat16), ((512, 256), torch.bfloat16)],
        {"model_names": ["pt_yolo_world_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul45,
        [((128, 2048), torch.float32)],
        {
            "model_names": [
                "onnx_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "onnx_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul15,
        [((128, 2048), torch.float32)],
        {
            "model_names": [
                "onnx_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "onnx_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 128, 64), torch.float32), ((32, 64, 128), torch.float32)],
        {
            "model_names": [
                "onnx_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "onnx_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 128, 128), torch.float32), ((32, 128, 64), torch.float32)],
        {
            "model_names": [
                "onnx_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "onnx_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul46,
        [((128, 2048), torch.float32)],
        {
            "model_names": [
                "onnx_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "onnx_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul47,
        [((1, 128, 8192), torch.float32)],
        {
            "model_names": [
                "onnx_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "onnx_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul48,
        [((1, 128, 2048), torch.float32)],
        {
            "model_names": [
                "onnx_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "onnx_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((6, 4544), torch.float32), ((4544, 18176), torch.float32)],
        {"model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 6, 18176), torch.float32), ((18176, 4544), torch.float32)],
            {"model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((6, 4544), torch.float32), ((4544, 4672), torch.float32)],
        {"model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul49,
        [((1, 32, 1), torch.float32)],
        {
            "model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf", "pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((71, 6, 64), torch.float32), ((1, 64, 6), torch.float32)],
        {"model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((71, 6, 6), torch.float32), ((1, 6, 64), torch.float32)],
        {"model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 4544), torch.float32), ((4544, 4544), torch.float32)],
        {"model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 6, 4544), torch.float32), ((4544, 65024), torch.float32)],
        {"model_names": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (Matmul50, [((1, 11, 128), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (Matmul51, [((11, 312), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (
        Matmul1,
        [((12, 11, 26), torch.float32), ((12, 26, 11), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 11, 11), torch.float32), ((12, 11, 26), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (Matmul52, [((1, 11, 312), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (Matmul53, [((1, 11, 1248), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (Matmul54, [((1, 11, 312), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (
        Matmul1,
        [((1, 11, 128), torch.float32), ((128, 21128), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul55,
        [((1, 2048), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
                "pd_resnet_50_img_cls_paddlemodels",
                "jax_resnet_50_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 128), torch.float32), ((128, 1024), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((128, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 128, 64), torch.float32), ((16, 64, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 128, 128), torch.float32), ((16, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 1024), torch.float32), ((1024, 2), torch.float32)],
        {"model_names": ["pt_albert_large_v2_token_cls_hf", "pt_albert_large_v1_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((128, 768), torch.float32), ((768, 1), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 768), torch.float32), ((768, 1), torch.float32)],
            {
                "model_names": [
                    "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                    "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 2304), torch.bfloat16), ((2304, 1000), torch.bfloat16)],
            {
                "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((256, 768), torch.bfloat16), ((768, 512), torch.bfloat16)],
            {"model_names": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 256, 256), torch.bfloat16), ((256, 512), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((1, 512, 196), torch.bfloat16), ((196, 256), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((1, 512, 256), torch.bfloat16), ((256, 196), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((1, 196, 512), torch.bfloat16), ((512, 2048), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 196, 2048), torch.bfloat16), ((2048, 512), torch.bfloat16)],
            {
                "model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1024), torch.bfloat16), ((1024, 1001), torch.bfloat16)],
            {
                "model_names": ["pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((32, 256, 64), torch.float32), ((32, 64, 256), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf", "pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 256, 256), torch.float32), ((32, 256, 64), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf", "pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf", "pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 2048), torch.float32), ((2048, 51200), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 7392), torch.bfloat16), ((7392, 1000), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 256), torch.bfloat16), ((256, 1000), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    pytest.param(
        (
            Matmul1,
            [((50, 1024), torch.bfloat16), ((1024, 3072), torch.bfloat16)],
            {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((16, 50, 64), torch.bfloat16), ((16, 64, 50), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((16, 50, 50), torch.bfloat16), ((16, 50, 64), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    pytest.param(
        (
            Matmul1,
            [((50, 1024), torch.bfloat16), ((1024, 1024), torch.bfloat16)],
            {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 50, 1024), torch.bfloat16), ((1024, 4096), torch.bfloat16)],
            {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 50, 4096), torch.bfloat16), ((4096, 1024), torch.bfloat16)],
            {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul56,
        [((1, 256, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul57,
        [((1, 1024, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul58,
        [((1, 4096, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul59,
        [((1, 16384, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul33,
        [((9, 768), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 9, 64), torch.float32), ((12, 64, 9), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 9, 9), torch.float32), ((12, 9, 64), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul34,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul35,
        [((1, 9, 3072), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1024), torch.bfloat16), ((1024, 18), torch.bfloat16)],
            {
                "model_names": ["pt_densenet_densenet121_hf_xray_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 2560), torch.bfloat16), ((2560, 1000), torch.bfloat16)],
            {
                "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 512, 49), torch.bfloat16), ((49, 256), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((1, 512, 256), torch.bfloat16), ((256, 49), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((1, 49, 512), torch.bfloat16), ((512, 2048), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 49, 2048), torch.bfloat16), ((2048, 512), torch.bfloat16)],
            {
                "model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 768), torch.bfloat16), ((768, 1001), torch.bfloat16)],
            {
                "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((6, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 6, 64), torch.float32), ((16, 64, 6), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 6, 6), torch.float32), ((16, 6, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 1024), torch.float32), ((1024, 2816), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 6, 2816), torch.float32), ((2816, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 6, 1024), torch.float32), ((1024, 151936), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 1536), torch.float32), ((1536, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 1536), torch.float32), ((1536, 256), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 29, 128), torch.float32), ((12, 128, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 29, 29), torch.float32), ((12, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 1536), torch.float32), ((1536, 8960), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 29, 8960), torch.float32), ((8960, 1536), torch.float32)],
            {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 29, 1536), torch.float32), ((1536, 151936), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 2520), torch.bfloat16), ((2520, 1000), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1008), torch.bfloat16), ((1008, 1000), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 61, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 61, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 513, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 513, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 2048), torch.float32), ((2048, 1000), torch.float32)],
        {
            "model_names": [
                "onnx_xception_xception71_tf_in1k_img_cls_timm",
                "onnx_resnet_50_img_cls_hf",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_hrnet_hrnetv2_w30_img_cls_osmr",
                "onnx_xception_xception65_img_cls_timm",
                "onnx_hrnet_hrnetv2_w18_img_cls_osmr",
                "onnx_hrnet_hrnetv2_w44_img_cls_osmr",
                "onnx_hrnet_hrnetv2_w64_img_cls_osmr",
                "onnx_wideresnet_wide_resnet50_2_img_cls_timm",
                "onnx_hrnet_hrnetv2_w48_img_cls_osmr",
                "onnx_wideresnet_wide_resnet101_2_img_cls_timm",
                "onnx_hrnet_hrnet_w18_small_v2_img_cls_osmr",
                "onnx_mobilenetv3_ssd_resnet50_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 2048), torch.float32), ((2048, 2), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf", "pt_albert_xlarge_v1_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((197, 192), torch.bfloat16), ((192, 192), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((3, 197, 64), torch.bfloat16), ((3, 64, 197), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((3, 197, 197), torch.bfloat16), ((3, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((1, 197, 192), torch.bfloat16), ((192, 768), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 197, 768), torch.bfloat16), ((768, 192), torch.bfloat16)],
            {
                "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 192), torch.bfloat16), ((192, 1000), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1024), torch.bfloat16), ((1024, 9), torch.bfloat16)],
            {
                "model_names": ["pt_mobilenetv1_basic_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((29, 896), torch.float32), ((896, 896), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul31,
        [((1, 32, 1), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf", "pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((29, 896), torch.float32), ((896, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 29, 64), torch.float32), ((14, 64, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 29, 29), torch.float32), ((14, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 896), torch.float32), ((896, 4864), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 29, 4864), torch.float32), ((4864, 896), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 29, 896), torch.float32), ((896, 151936), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((197, 768), torch.bfloat16), ((768, 2304), torch.bfloat16)],
            {"model_names": ["pt_vit_vit_b_16_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((197, 1024), torch.bfloat16), ((1024, 3072), torch.bfloat16)],
            {"model_names": ["pt_vit_vit_l_16_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 256, 1024), torch.float32), ((1024, 256008), torch.float32)],
        {"model_names": ["pt_xglm_facebook_xglm_564m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul33,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 9, 768), torch.float32), ((768, 21128), torch.float32)],
        {"model_names": ["pd_bert_chinese_roberta_base_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 768), torch.float32), ((768, 28996), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 576), torch.float32), ((576, 1024), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1280), torch.float32), ((1280, 1000), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "onnx_efficientnet_efficientnet_lite0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_ghostnet_ghostnet_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 9, 768), torch.float32), ((768, 30522), torch.float32)],
        {"model_names": ["pd_bert_bert_base_uncased_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul60,
        [((13, 384), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 13, 32), torch.float32), ((12, 32, 13), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 13, 13), torch.float32), ((12, 13, 32), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul61,
        [((1, 13, 384), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul62,
        [((1, 13, 1536), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 384), torch.float32), ((384, 384), torch.float32)],
        {
            "model_names": [
                "onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Matmul33, [((1, 11, 768), torch.float32)], {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99}),
    (
        Matmul1,
        [((1, 11, 768), torch.float32), ((768, 21128), torch.float32)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 768), torch.float32), ((768, 2), torch.float32)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 32, 64), torch.float32), ((32, 64, 32), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 32, 32), torch.float32), ((32, 32, 64), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 2048), torch.float32), ((2048, 2), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 32, 512), torch.float32), ((512, 1024), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 32, 64), torch.float32), ((16, 64, 32), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 32, 32), torch.float32), ((16, 32, 64), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 1024), torch.float32), ((1024, 512), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 512), torch.float32), ((512, 2), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 29, 64), torch.float32), ((16, 64, 29), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 29, 29), torch.float32), ((16, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 1024), torch.float32), ((1024, 2816), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 29, 2816), torch.float32), ((2816, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 29, 1024), torch.float32), ((1024, 151936), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 896), torch.float32), ((896, 896), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul17,
        [((1, 32, 1), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 896), torch.float32), ((896, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 39, 64), torch.float32), ((14, 64, 39), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 39, 39), torch.float32), ((14, 39, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 896), torch.float32), ((896, 4864), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 39, 4864), torch.float32), ((4864, 896), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 39, 896), torch.float32), ((896, 151936), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 912), torch.bfloat16), ((912, 1000), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 512), torch.float32), ((512, 384), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 1, 64), torch.float32), ((6, 64, 1), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((6, 1, 1), torch.float32), ((6, 1, 64), torch.float32)],
        {
            "model_names": [
                "pt_t5_google_flan_t5_small_text_gen_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 384), torch.float32), ((384, 512), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((61, 512), torch.float32), ((512, 384), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 61, 64), torch.float32), ((6, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 61, 61), torch.float32), ((6, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((61, 384), torch.float32), ((384, 512), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((61, 512), torch.float32), ((512, 1024), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 61, 1024), torch.float32), ((1024, 512), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 1, 64), torch.float32), ((6, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 1, 61), torch.float32), ((6, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 512), torch.float32), ((512, 1024), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 1024), torch.float32), ((1024, 512), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 512), torch.float32), ((512, 32128), torch.float32)],
        {"model_names": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1664), torch.float32), ((1664, 1000), torch.float32)],
        {"model_names": ["onnx_densenet_densenet169_img_cls_torchvision"], "pcc": 0.99},
    ),
    (Matmul33, [((128, 768), torch.float32)], {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99}),
    (Matmul34, [((1, 128, 768), torch.float32)], {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99}),
    (Matmul35, [((1, 128, 3072), torch.float32)], {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99}),
    (Matmul33, [((1, 128, 768), torch.float32)], {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99}),
    (Matmul63, [((1, 128, 768), torch.float32)], {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99}),
    pytest.param(
        (
            Matmul1,
            [((1, 9216), torch.float32), ((9216, 4096), torch.float32)],
            {"model_names": ["onnx_alexnet_base_img_cls_torchhub"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 4096), torch.float32), ((4096, 4096), torch.float32)],
        {"model_names": ["onnx_alexnet_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 4096), torch.float32), ((4096, 1000), torch.float32)],
        {"model_names": ["onnx_alexnet_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Matmul60,
        [((197, 384), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 197, 64), torch.float32), ((6, 64, 197), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 197, 197), torch.float32), ((6, 197, 64), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul61,
        [((1, 197, 384), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul62,
        [((1, 197, 1536), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 384), torch.float32), ((384, 1000), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1024), torch.float32), ((1024, 9), torch.float32)],
        {"model_names": ["onnx_mobilenetv1_mobilenet_v1_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul64,
        [((1, 256, 512), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul65,
        [((1, 1024, 320), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul66,
        [((1, 4096, 128), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul36,
        [((1, 9, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_uncased_qa_padlenlp", "pd_ernie_1_0_qa_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul67,
        [((1, 512), torch.float32)],
        {"model_names": ["pd_resnet_18_img_cls_paddlemodels", "pd_resnet_34_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 4096), torch.float32), ((4096, 2), torch.float32)],
        {"model_names": ["pt_albert_xxlarge_v1_token_cls_hf", "pt_albert_xxlarge_v2_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 768), torch.float32), ((768, 30522), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_uncased_mlm_hf", "pt_bert_bert_base_uncased_mlm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256, 2048), torch.float32), ((2048, 50257), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 32, 64), torch.float32), ((12, 64, 32), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 32, 32), torch.float32), ((12, 32, 64), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 768), torch.float32), ((768, 1), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((35, 896), torch.float32), ((896, 896), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul30,
        [((1, 32, 1), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((35, 896), torch.float32), ((896, 128), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 35, 64), torch.float32), ((14, 64, 35), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 35, 35), torch.float32), ((14, 35, 64), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((35, 896), torch.float32), ((896, 4864), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 35, 4864), torch.float32), ((4864, 896), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 35, 896), torch.float32), ((896, 151936), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 1536), torch.float32), ((1536, 1536), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((48, 1, 64), torch.float32), ((48, 64, 1), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((48, 1, 1), torch.float32), ((48, 1, 64), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((26, 768), torch.float32), ((768, 1536), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((26, 1536), torch.float32), ((1536, 1536), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((48, 1, 64), torch.float32), ((48, 64, 13), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((48, 1, 13), torch.float32), ((48, 13, 64), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 1536), torch.float32), ((1536, 6144), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 6144), torch.float32), ((6144, 1536), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 1536), torch.float32), ((1536, 2048), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((201, 768), torch.bfloat16), ((768, 768), torch.bfloat16)],
            {
                "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((12, 201, 64), torch.bfloat16), ((12, 64, 201), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul1,
        [((12, 201, 201), torch.bfloat16), ((12, 201, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 201, 768), torch.bfloat16), ((768, 3072), torch.bfloat16)],
            {
                "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 201, 3072), torch.bfloat16), ((3072, 768), torch.bfloat16)],
            {
                "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 768), torch.bfloat16), ((768, 768), torch.bfloat16)],
            {
                "model_names": [
                    "pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf",
                    "pt_vilt_dandelin_vilt_b32_mlm_mlm_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 768), torch.bfloat16), ((768, 1536), torch.bfloat16)],
            {
                "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1536), torch.bfloat16), ((1536, 3129), torch.bfloat16)],
            {
                "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 2208), torch.float32), ((2208, 1000), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1536), torch.float32), ((1536, 1000), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul68,
        [((1, 16384, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul68,
        [((256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 16384, 32), torch.float32), ((1, 32, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 16384, 256), torch.float32), ((1, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul69,
        [((1, 16384, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul70,
        [((1, 16384, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul4,
        [((1, 4096, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((2, 4096, 32), torch.float32), ((2, 32, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((2, 4096, 256), torch.float32), ((2, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul4,
        [((4096, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul5,
        [((1, 4096, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul6,
        [((1, 4096, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul71,
        [((1, 1024, 160), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul71,
        [((256, 160), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((5, 1024, 32), torch.float32), ((5, 32, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((5, 1024, 256), torch.float32), ((5, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul71,
        [((1024, 160), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul72,
        [((1, 1024, 160), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul73,
        [((1, 1024, 640), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul74,
        [((256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((8, 256, 32), torch.float32), ((8, 32, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((8, 256, 256), torch.float32), ((8, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul75,
        [((1, 256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul76,
        [((1, 256, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256), torch.float32), ((256, 1000), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 400, 32), torch.float32), ((2, 32, 400), torch.float32)],
        {"model_names": ["onnx_yolov10_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 64, 400), torch.float32), ((2, 400, 400), torch.float32)],
        {"model_names": ["onnx_yolov10_default_obj_det_github"], "pcc": 0.99},
    ),
    (Matmul33, [((14, 768), torch.float32)], {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99}),
    (
        Matmul1,
        [((12, 14, 64), torch.float32), ((12, 64, 14), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp", "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 14, 14), torch.float32), ((12, 14, 64), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp", "pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul34,
        [((1, 14, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul35,
        [((1, 14, 3072), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul36,
        [((1, 14, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 768), torch.float32), ((768, 2), torch.float32)],
        {"model_names": ["pt_albert_base_v1_token_cls_hf", "pt_albert_base_v2_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 1024), torch.float32), ((1024, 128), torch.float32)],
        {"model_names": ["pt_albert_large_v2_mlm_hf", "pt_albert_large_v1_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 14, 128), torch.float32), ((128, 768), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 14, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 14, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 768), torch.float32), ((768, 1), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((588, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul77,
        [((1, 64, 1), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 588, 128), torch.float32), ((16, 128, 588), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 588, 588), torch.float32), ((16, 588, 128), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((588, 2048), torch.float32), ((2048, 5504), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 588, 5504), torch.float32), ((5504, 2048), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 588, 2048), torch.float32), ((2048, 32256), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 768), torch.float32), ((768, 119547), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_multilingual_cased_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 512), torch.float32), ((512, 512), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 1, 64), torch.float32), ((8, 64, 1), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 1, 1), torch.float32), ((8, 1, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 512), torch.float32), ((512, 512), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1500, 512), torch.float32), ((512, 512), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 1500, 64), torch.float32), ((8, 64, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 1500, 1500), torch.float32), ((8, 1500, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 512), torch.float32), ((512, 2048), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 2048), torch.float32), ((2048, 512), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 1, 64), torch.float32), ((8, 64, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 1, 1500), torch.float32), ((8, 1500, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 512), torch.float32), ((512, 2048), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 2048), torch.float32), ((2048, 512), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 512), torch.float32), ((512, 51865), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul78,
        [((4096, 96), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Matmul79,
        [((225, 512), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Matmul80,
        [((1, 4096, 96), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Matmul81,
        [((1, 4096, 384), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Matmul82,
        [((1024, 384), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Matmul83,
        [((1024, 192), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Matmul84,
        [((225, 512), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Matmul85,
        [((1, 1024, 192), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Matmul86,
        [((1, 1024, 768), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Matmul87,
        [((256, 768), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Matmul60,
        [((256, 384), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Matmul88,
        [((225, 512), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Matmul61,
        [((1, 256, 384), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Matmul62,
        [((1, 256, 1536), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Matmul89,
        [((64, 1536), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Matmul33,
        [((64, 768), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Matmul90,
        [((225, 512), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Matmul34,
        [((1, 64, 768), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Matmul35,
        [((1, 64, 3072), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Matmul36,
        [((1, 11, 768), torch.float32)],
        {"model_names": ["pd_bert_chinese_roberta_base_qa_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 384), torch.float32), ((384, 384), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1500, 384), torch.float32), ((384, 384), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 1500, 64), torch.float32), ((6, 64, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 1500, 1500), torch.float32), ((6, 1500, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 384), torch.float32), ((384, 1536), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 1536), torch.float32), ((1536, 384), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 1, 64), torch.float32), ((6, 64, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 1, 1500), torch.float32), ((6, 1500, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 384), torch.float32), ((384, 1536), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 1536), torch.float32), ((1536, 384), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 384), torch.float32), ((384, 51865), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul33,
        [((6, 768), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 6, 64), torch.float32), ((12, 64, 6), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 6, 6), torch.float32), ((12, 6, 64), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul34,
        [((1, 6, 768), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul35,
        [((1, 6, 3072), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul33,
        [((197, 768), torch.float32)],
        {
            "model_names": [
                "onnx_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 197, 64), torch.float32), ((12, 64, 197), torch.float32)],
        {
            "model_names": [
                "onnx_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 197, 197), torch.float32), ((12, 197, 64), torch.float32)],
        {
            "model_names": [
                "onnx_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul34,
        [((1, 197, 768), torch.float32)],
        {
            "model_names": [
                "onnx_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul35,
        [((1, 197, 3072), torch.float32)],
        {
            "model_names": [
                "onnx_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1408), torch.float32), ((1408, 1000), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1792), torch.float32), ((1792, 1000), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_140_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul33,
        [((15, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 15, 64), torch.float32), ((12, 64, 15), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 15, 15), torch.float32), ((12, 15, 64), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul34,
        [((1, 15, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul35,
        [((1, 15, 3072), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 9, 128), torch.float32), ((128, 768), torch.float32)],
        {"model_names": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((9, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 9, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_names": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 9, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_names": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((384, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 384, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 384, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((14, 512), torch.float32), ((512, 512), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 7, 64), torch.float32), ((16, 64, 7), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 7, 7), torch.float32), ((16, 7, 64), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 512), torch.float32), ((512, 2048), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 2048), torch.float32), ((2048, 512), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 768), torch.float32), ((768, 9), torch.float32)],
        {"model_names": ["pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 2048), torch.float32), ((2048, 50272), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 1280), torch.float32), ((1280, 1280), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 2, 64), torch.float32), ((20, 64, 2), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 2, 2), torch.float32), ((20, 2, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 2, 1280), torch.float32), ((1280, 1280), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 2, 64), torch.float32), ((20, 64, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 2, 1500), torch.float32), ((20, 1500, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 2, 1280), torch.float32), ((1280, 5120), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 2, 5120), torch.float32), ((5120, 1280), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 2, 1280), torch.float32), ((1280, 51866), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Matmul74,
        [((100, 256), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 100, 32), torch.float32), ((8, 32, 100), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((100, 256), torch.float32), ((256, 256), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 100, 100), torch.float32), ((8, 100, 32), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Matmul74,
        [((1, 100, 256), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Matmul74,
        [((280, 256), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 280, 32), torch.float32), ((8, 32, 280), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Matmul74,
        [((1, 280, 256), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 280, 280), torch.float32), ((8, 280, 32), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Matmul91,
        [((1, 280, 256), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Matmul92,
        [((1, 280, 2048), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 100, 32), torch.float32), ((8, 32, 280), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 100, 280), torch.float32), ((8, 280, 32), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Matmul91,
        [((1, 100, 256), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Matmul92,
        [((1, 100, 2048), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Matmul93,
        [((100, 256), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Matmul94,
        [((1, 100, 256), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (Matmul33, [((10, 768), torch.float32)], {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99}),
    (
        Matmul1,
        [((12, 10, 64), torch.float32), ((12, 64, 10), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 10, 10), torch.float32), ((12, 10, 64), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul34,
        [((1, 10, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul35,
        [((1, 10, 3072), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul33,
        [((1, 10, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 10, 768), torch.float32), ((768, 32000), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul95,
        [((1, 12, 120), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((8, 12, 15), torch.float32), ((8, 15, 12), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((8, 12, 12), torch.float32), ((8, 12, 15), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul96,
        [((12, 120), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul97,
        [((1, 12, 120), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul98,
        [((1, 12, 240), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul99,
        [((1, 12, 120), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Matmul100,
        [((1, 12, 120), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((384, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 384, 64), torch.float32), ((12, 64, 384), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 384, 384), torch.float32), ((12, 384, 64), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 384, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 384, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((384, 768), torch.float32), ((768, 1), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((522, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((522, 2048), torch.float32), ((2048, 1024), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 522, 256), torch.float32), ((8, 256, 522), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 522, 522), torch.float32), ((8, 522, 256), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((522, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 522, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 522, 2048), torch.float32), ((2048, 131072), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((7, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 7, 64), torch.float32), ((12, 64, 7), torch.float32)],
        {"model_names": ["pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 7, 7), torch.float32), ((12, 7, 64), torch.float32)],
        {"model_names": ["pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul33,
        [((7, 768), torch.float32)],
        {"model_names": ["pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul34,
        [((7, 768), torch.float32)],
        {"model_names": ["pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul35,
        [((7, 3072), torch.float32)],
        {"model_names": ["pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((7, 768), torch.float32), ((768, 2), torch.float32)],
        {"model_names": ["pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 5, 128), torch.float32), ((16, 128, 5), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 5, 5), torch.float32), ((16, 5, 128), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 5, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul19,
        [((1, 32, 1), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((128, 2048), torch.float32), ((2048, 512), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((128, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 2048), torch.float32), ((2048, 128256), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 512), torch.float32), ((512, 1024), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 512), torch.float32), ((512, 50272), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 512), torch.float32), ((512, 1), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 2048, 768), torch.float32), ((768, 256), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 1280), torch.float32), ((1280, 256), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2048, 768), torch.float32), ((768, 256), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 256, 32), torch.float32), ((8, 32, 2048), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2048, 768), torch.float32), ((768, 1280), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 256, 2048), torch.float32), ((8, 2048, 160), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 1280), torch.float32), ((1280, 1280), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 1280), torch.float32), ((1280, 1280), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 1280), torch.float32), ((1280, 256), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 256, 256), torch.float32), ((8, 256, 160), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 2048, 32), torch.float32), ((8, 32, 256), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 1280), torch.float32), ((1280, 768), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 2048, 256), torch.float32), ((8, 256, 96), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2048, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 2048, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2048, 768), torch.float32), ((768, 262), torch.float32)],
        {"model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((204, 768), torch.bfloat16), ((768, 768), torch.bfloat16)],
            {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((12, 204, 64), torch.bfloat16), ((12, 64, 204), torch.bfloat16)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul1,
        [((12, 204, 204), torch.bfloat16), ((12, 204, 64), torch.bfloat16)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 204, 768), torch.bfloat16), ((768, 3072), torch.bfloat16)],
            {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 204, 3072), torch.bfloat16), ((3072, 768), torch.bfloat16)],
            {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul74,
        [((1, 256, 256), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul101,
        [((1, 1024, 160), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul102,
        [((1, 16384, 32), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 9, 768), torch.float32), ((768, 18000), torch.float32)],
        {"model_names": ["pd_ernie_1_0_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 768), torch.float32), ((768, 768), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_gpt_gpt2_text_gen_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 256, 64), torch.float32), ((12, 64, 256), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_gpt_gpt2_text_gen_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 256, 256), torch.float32), ((12, 256, 64), torch.float32)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_gpt_gpt2_text_gen_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 768), torch.float32), ((768, 50257), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf", "pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 1, 64), torch.float32), ((12, 64, 1), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 1, 1), torch.float32), ((12, 1, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1500, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 1500, 64), torch.float32), ((12, 64, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 1500, 1500), torch.float32), ((12, 1500, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 1, 64), torch.float32), ((12, 64, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 1, 1500), torch.float32), ((12, 1500, 64), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 768), torch.float32), ((768, 51865), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul83,
        [((197, 192), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((3, 197, 64), torch.float32), ((3, 64, 197), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((3, 197, 197), torch.float32), ((3, 197, 64), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul85,
        [((1, 197, 192), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul86,
        [((1, 197, 768), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 192), torch.float32), ((192, 1000), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul103,
            [((1, 9216), torch.float32)],
            {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (Matmul104, [((1, 4096), torch.float32)], {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (Matmul105, [((1, 4096), torch.float32)], {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Matmul1,
        [((1, 288), torch.float32), ((288, 192), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul106,
        [((48, 192), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 48), torch.float32), ((48, 192), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 96), torch.float32), ((96, 192), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul107,
        [((1, 25, 96), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 784), torch.float32), ((784, 128), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128), torch.float32), ((128, 64), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 64), torch.float32), ((64, 12), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 12), torch.float32), ((12, 3), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 3), torch.float32), ((3, 12), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 12), torch.float32), ((12, 64), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 64), torch.float32), ((64, 128), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128), torch.float32), ((128, 784), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 1024), torch.float32), ((1024, 9), torch.float32)],
        {"model_names": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 1408), torch.bfloat16), ((1408, 1000), torch.bfloat16)],
            {
                "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (Matmul33, [((256, 768), torch.float32)], {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99}),
    (Matmul34, [((256, 768), torch.float32)], {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99}),
    (Matmul35, [((256, 3072), torch.float32)], {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99}),
    (
        Matmul1,
        [((256, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 768), torch.float32), ((768, 50272), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 768), torch.float32), ((768, 2), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((513, 512), torch.float32), ((512, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 513, 64), torch.float32), ((8, 64, 513), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 513, 513), torch.float32), ((8, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((61, 512), torch.float32), ((512, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 61, 64), torch.float32), ((8, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 61, 61), torch.float32), ((8, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 61, 512), torch.float32), ((512, 2048), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 61, 2048), torch.float32), ((2048, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 513, 64), torch.float32), ((8, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 513, 61), torch.float32), ((8, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 513, 512), torch.float32), ((512, 2048), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 513, 2048), torch.float32), ((2048, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 513, 512), torch.float32), ((512, 32128), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul108,
        [((1, 25, 96), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 32, 1536), torch.float32), ((1536, 4608), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 32, 96), torch.float32), ((16, 96, 32), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 32, 32), torch.float32), ((16, 32, 96), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 1536), torch.float32), ((1536, 1536), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 32, 1536), torch.float32), ((1536, 6144), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 32, 6144), torch.float32), ((6144, 1536), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 32, 1536), torch.float32), ((1536, 250880), torch.float32)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 334, 4096), torch.float32), ((4096, 12288), torch.float32)],
        {"model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99},
    ),
    (Matmul109, [((1, 16, 1), torch.float32)], {"model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99}),
    (
        Matmul1,
        [((64, 334, 64), torch.float32), ((64, 64, 334), torch.float32)],
        {"model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((64, 334, 334), torch.float32), ((64, 334, 64), torch.float32)],
        {"model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((334, 4096), torch.float32), ((4096, 4096), torch.float32)],
        {"model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 334, 4096), torch.float32), ((4096, 16384), torch.float32)],
        {"model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 334, 16384), torch.float32), ((16384, 4096), torch.float32)],
            {"model_names": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((32, 2048), torch.float32), ((2048, 1), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((64, 1, 64), torch.float32), ((64, 64, 1), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((64, 1, 1), torch.float32), ((64, 1, 64), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((26, 768), torch.float32), ((768, 2048), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((26, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((64, 1, 64), torch.float32), ((64, 64, 13), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((64, 1, 13), torch.float32), ((64, 13, 64), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Matmul")

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
