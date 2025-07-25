# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.vision.yolo.model_utils.yolo_utils import (
    YoloWrapper,
    load_yolo_model_and_image,
)

variants = ["yolov10x", "yolov10n"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_yolov10(variant):

    if variant in ["yolov10x"]:
        group = ModelGroup.RED
        priority = ModelPriority.P1
    else:
        group = ModelGroup.GENERALITY
        priority = ModelPriority.P2

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.YOLOV10,
        variant=variant,
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
        group=group,
        priority=priority,
    )

    # Load  model and input
    model, image_tensor = load_yolo_model_and_image(
        f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{variant}.pt"
    )
    framework_model = YoloWrapper(model).to(torch.bfloat16)
    image_tensor = image_tensor.to(torch.bfloat16)
    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[image_tensor],
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify([image_tensor], framework_model, compiled_model)
