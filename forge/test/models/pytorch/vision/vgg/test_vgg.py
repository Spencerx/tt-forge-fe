# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import timm
import torch
from loguru import logger
from PIL import Image
from pytorchcv.model_provider import get_model as ptcv_get_model
from third_party.tt_forge_models.tools.utils import get_file
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
from vgg_pytorch import VGG

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from test.models.models_utils import print_cls_results
from test.models.pytorch.vision.vision_utils.utils import load_vision_model_and_input
from test.utils import download_model

variants = [
    pytest.param("vgg11"),
    pytest.param("vgg13"),
    pytest.param("vgg16"),
    pytest.param("vgg19"),
    pytest.param("bn_vgg19"),
    pytest.param("bn_vgg19b"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_vgg_osmr_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VGG,
        variant=variant,
        source=Source.OSMR,
        task=Task.OBJECT_DETECTION,
    )

    framework_model = download_model(ptcv_get_model, variant, pretrained=True).to(torch.bfloat16)
    framework_model.eval()

    # Image preprocessing
    try:
        file_path = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
        input_image = Image.open(file_path).convert("RGB")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        input_batch = torch.rand(1, 3, 224, 224)

    inputs = [input_batch.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    pcc = 0.99

    if variant in ["vgg16", "bn_vgg19"]:
        pcc = 0.98
    elif variant == "vgg19":
        pcc = 0.97
    elif variant == "bn_vgg19b":
        pcc = 0.96

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])


@pytest.mark.nightly
def test_vgg_19_hf_pytorch():

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VGG,
        variant="19",
        source=Source.HUGGINGFACE,
        task=Task.OBJECT_DETECTION,
    )

    """
    # https://pypi.org/project/vgg-pytorch/
    # Variants:
    vgg11, vgg11_bn
    vgg13, vgg13_bn
    vgg16, vgg16_bn
    vgg19, vgg19_bn
    """
    framework_model = download_model(VGG.from_pretrained, "vgg19").to(torch.bfloat16)
    framework_model.eval()

    # Image preprocessing
    try:
        file_path = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
        input_image = Image.open(file_path).convert("RGB")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        input_batch = torch.rand(1, 3, 224, 224)

    inputs = [input_batch.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    fw_out, co_out = verify(inputs, framework_model, compiled_model)

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])


def preprocess_timm_model(model_name):
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    try:
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        file_path = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
        img = Image.open(file_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # transform and add batch dimension
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)

    return model, img_tensor


@pytest.mark.nightly
def test_vgg_bn19_timm_pytorch():

    variant = "vgg19_bn"

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VGG,
        variant="vgg19_bn",
        source=Source.TIMM,
        task=Task.OBJECT_DETECTION,
    )

    torch.multiprocessing.set_sharing_strategy("file_system")
    framework_model, image_tensor = download_model(preprocess_timm_model, variant)

    inputs = [image_tensor.to(torch.bfloat16)]
    framework_model.to(torch.bfloat16)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    fw_out, co_out = verify(inputs, framework_model, compiled_model)

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])


@pytest.mark.nightly
def test_vgg_bn19_torchhub_pytorch():

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VGG,
        variant="vgg19_bn",
        source=Source.TORCH_HUB,
        task=Task.OBJECT_DETECTION,
    )

    framework_model = download_model(torch.hub.load, "pytorch/vision:v0.10.0", "vgg19_bn", pretrained=True).to(
        torch.bfloat16
    )
    framework_model.eval()

    # Image preprocessing
    try:
        file_path = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
        input_image = Image.open(file_path).convert("RGB")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        input_batch = torch.rand(1, 3, 224, 224)

    inputs = [input_batch.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    fw_out, co_out = verify(inputs, framework_model, compiled_model)

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])


variants_with_weights = {
    "vgg11": "VGG11_Weights",
    "vgg11_bn": "VGG11_BN_Weights",
    "vgg13": "VGG13_Weights",
    "vgg13_bn": "VGG13_BN_Weights",
    "vgg16": "VGG16_Weights",
    "vgg16_bn": "VGG16_BN_Weights",
    "vgg19": "VGG19_Weights",
}

variants = [
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_vgg_torchvision(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VGG,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Load model and input
    weight_name = variants_with_weights[variant]
    framework_model, inputs = load_vision_model_and_input(variant, "classification", weight_name)
    framework_model.to(torch.bfloat16)
    inputs = [inputs[0].to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    pcc = 0.99
    if variant in ["vgg16_bn", "vgg13_bn"]:
        pcc = 0.98

    # Model Verificatiogn and inference
    fw_out, co_out = verify(
        inputs, framework_model, compiled_model, verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc))
    )

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])
