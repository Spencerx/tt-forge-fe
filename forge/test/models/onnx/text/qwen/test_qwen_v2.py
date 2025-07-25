# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import onnx
from transformers import AutoModelForCausalLM, AutoTokenizer

import forge
from forge.verify.verify import verify
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties

from test.models.models_utils import TextModelWrapper


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "Qwen/Qwen2.5-0.5B",
        ),
        pytest.param(
            "Qwen/Qwen2.5-1.5B",
        ),
        pytest.param(
            "Qwen/Qwen2.5-3B",
        ),
        pytest.param(
            "Qwen/Qwen2.5-0.5B-Instruct",
        ),
        pytest.param(
            "Qwen/Qwen2.5-1.5B-Instruct",
        ),
        pytest.param(
            "Qwen/Qwen2.5-3B-Instruct",
        ),
    ],
)
def test_qwen_clm_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.QWENV2,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    if variant in ["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct"]:
        pytest.xfail(reason="Segmentation Fault")
    else:
        pytest.xfail(reason="Requires multi-chip support")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(variant, use_cache=False)
    framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)
    framework_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(variant)

    prompt = "Give me a short introduction to large language models."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize and generate
    tokenized_inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Get input_ids and attention_mask
    input_ids = tokenized_inputs["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]

    inputs = [input_ids, attention_mask]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/model.onnx"
    torch.onnx.export(framework_model, tuple(inputs), onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)

    # passing model file instead of model proto due to size of the model(>2GB) - #https://github.com/onnx/onnx/issues/3775#issuecomment-943416925
    onnx.checker.check_model(onnx_path)
    framework_model = forge.OnnxModule(module_name, onnx_model, onnx_path)

    # Compile model
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
    )
