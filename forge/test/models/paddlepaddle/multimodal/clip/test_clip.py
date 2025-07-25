# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import paddle
from paddlenlp.transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTextModel

from forge.tvm_calls.forge_utils import paddle_trace
import forge
from PIL import Image
from forge.verify.verify import verify
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.config import VerifyConfig
from third_party.tt_forge_models.tools.utils import get_file

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties

variants = ["openai/clip-vit-base-patch16"]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_clip_text(variant):
    # Record Forge properties
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.CLIPTEXT,
        variant=variant,
        source=Source.PADDLENLP,
        task=Task.TEXT_ENCODING,
    )

    # Load model and processor
    model = CLIPTextModel.from_pretrained(variant)
    processor = CLIPProcessor.from_pretrained(variant)

    # Prepare inputs
    text = "a photo of cats in bed"
    inputs = processor(text=text, return_tensors="pd", padding=True)
    inputs = [inputs["input_ids"]]

    # Compile model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_clip_vision(variant):
    # Record Forge properties
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.CLIPVISION,
        variant=variant,
        source=Source.PADDLENLP,
        task=Task.IMAGE_ENCODING,
    )

    # Load model and processor
    model = CLIPVisionModel.from_pretrained(variant)
    processor = CLIPProcessor.from_pretrained(variant)

    # Prepare inputs
    input_image = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
    image = Image.open(str(input_image))
    inputs = processor(images=image, return_tensors="pd")
    inputs = [inputs["pixel_values"]]

    # Compile model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify
    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.97)))


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
@pytest.mark.xfail
def test_clip(variant):
    # Record Forge properties
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.CLIP,
        variant=variant,
        source=Source.PADDLENLP,
        task=Task.IMAGE_TEXT_PAIRING,
    )

    # Load model and processor
    model = CLIPModel.from_pretrained(variant)
    processor = CLIPProcessor.from_pretrained(variant)

    # Prepare inputs
    text = [
        "a photo of cats in bed",
        "a photo of dog in snow",
    ]
    input_image = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
    image = Image.open(str(input_image))
    inputs = processor(images=image, text=text, return_tensors="pd")
    inputs = [inputs["input_ids"], inputs["pixel_values"]]

    # Test framework model
    outputs = model(*inputs)

    image_embed = outputs.image_embeds
    text_embeds = outputs.text_embeds

    image_embed = paddle.nn.functional.normalize(image_embed, axis=-1)
    text_embeds = paddle.nn.functional.normalize(text_embeds, axis=-1)

    similarities = paddle.matmul(text_embeds, image_embed.T)
    similarities = similarities.squeeze().numpy()

    for t, sim in zip(text, similarities):
        print(f"{t}: similarity = {sim:.4f}")

    # Compile model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify
    verify(inputs, framework_model, compiled_model)
