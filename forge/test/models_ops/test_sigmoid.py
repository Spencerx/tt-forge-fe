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


class Sigmoid0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, sigmoid_input_0):
        sigmoid_output_1 = forge.op.Sigmoid("", sigmoid_input_0)
        return sigmoid_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Sigmoid0,
        [((1, 44, 8192), torch.float32)],
        {"model_names": ["pt_cogito_deepcogito_cogito_v1_preview_llama_3b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 39, 11008), torch.float32)],
        {
            "model_names": [
                "pt_deepseek_deepseek_math_7b_instruct_qa_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 522, 9216), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 522, 23040), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 11008), torch.float32)],
        {"model_names": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 4, 11008), torch.float32)],
        {"model_names": ["pt_llama3_huggyllama_llama_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 14336), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 4, 14336), torch.float32)],
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
        Sigmoid0,
        [((1, 128, 8192), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf",
                "onnx_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "onnx_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 4, 8192), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 10, 14336), torch.float32)],
        {"model_names": ["pt_ministral_ministral_ministral_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 8, 12288), torch.float32)],
        {"model_names": ["pt_ministral_mistralai_ministral_8b_instruct_2410_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 135, 14336), torch.float32)],
        {"model_names": ["pt_mistral_mistralai_mistral_7b_instruct_v0_3_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 256, 8192), torch.float32)],
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
        Sigmoid0,
        [((1, 5, 8192), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 13, 8192), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Sigmoid0, [((1, 5, 17920), torch.float32)], {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99}),
    (
        Sigmoid0,
        [((1, 12, 17920), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 35, 8960), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 35, 11008), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 35, 18944), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Sigmoid0, [((1, 29, 11008), torch.float32)], {"model_names": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99}),
    (
        Sigmoid0,
        [((1, 29, 18944), torch.float32)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_7b_instruct_1m_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 39, 18944), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 38, 18944), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_math_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 13, 18944), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 8, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 4, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 6, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 240, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 10, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 240, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 240, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 480, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 20, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 480, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 672, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 28, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 672, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 672, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1152, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 48, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1152, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1280, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 16, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 16, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1920, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b1_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b1_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1920, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b1_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 48, 160, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 12, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 24, 160, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 24, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 160, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 336, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 14, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 336, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 336, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 672, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 960, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 40, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 960, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 960, 10, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1632, 10, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 68, 1, 1), torch.bfloat16)],
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
    (
        Sigmoid0,
        [((1, 1632, 1, 1), torch.bfloat16)],
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
    (
        Sigmoid0,
        [((1, 2688, 10, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 112, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 2688, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1792, 10, 10), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 24, 144, 144), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 72, 72), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 72, 72), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 36, 36), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 36, 36), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 18, 18), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 512, 18, 18), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 512, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 768, 18, 18), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 768, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 960, 18, 18), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 960, 9, 9), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1632, 9, 9), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1792, 9, 9), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 24, 150, 150), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 75, 75), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 75, 75), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 38, 38), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 38, 38), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 512, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 768, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 960, 19, 19), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1536, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1536, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1280, 10, 10), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 2, 30, 40), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 2, 60, 80), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 2, 120, 160), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1, 480, 640), torch.bfloat16)],
        {
            "model_names": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1, 320, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1, 192, 640), torch.bfloat16)],
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
        },
    ),
    (
        Sigmoid0,
        [((1, 39, 8960), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_040_img_cls_hf",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1088, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 288, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 576, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_064_img_cls_hf",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1296, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_064_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 168, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_080_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 448, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "pt_regnet_facebook_regnet_y_120_img_cls_hf",
                "pt_regnet_regnet_y_16gf_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 896, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_080_img_cls_hf",
                "pt_regnet_facebook_regnet_y_120_img_cls_hf",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 2016, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_080_img_cls_hf", "pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 224, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_facebook_regnet_y_120_img_cls_hf",
                "pt_regnet_regnet_y_16gf_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 2240, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_120_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1232, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_16gf_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 3024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_16gf_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 120, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 888, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 232, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 696, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1392, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 3712, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
                "pt_regnet_facebook_regnet_y_320_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 72, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 216, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1512, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 104, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 208, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 440, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 320, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 784, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 720, 60, 80), torch.bfloat16)],
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
        },
    ),
    (
        Sigmoid0,
        [((1, 720, 30, 40), torch.bfloat16)],
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
        },
    ),
    (
        Sigmoid0,
        [((1, 720, 15, 20), torch.bfloat16)],
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
        },
    ),
    (
        Sigmoid0,
        [((1, 720, 8, 10), torch.bfloat16)],
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
        },
    ),
    (
        Sigmoid0,
        [((1, 720, 4, 5), torch.bfloat16)],
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
        },
    ),
    (
        Sigmoid0,
        [((1, 4, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 8, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 16, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 3, 64, 64), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 6, 64, 64), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 12, 64, 64), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 24, 64, 64), torch.float32)],
        {
            "model_names": ["pt_swin_swin_v2_s_img_cls_torchvision", "pt_swin_swin_v2_t_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1, 256, 256), torch.bfloat16)],
        {"model_names": ["pt_unet_base_img_seg_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 16, 320, 320), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10n_obj_det_github", "pt_yolov8_yolov8n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 160, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 16, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10n_obj_det_github", "pt_yolov8_yolov8n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolov10_yolov10n_obj_det_github", "pt_yolov8_yolov8n_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 8400), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 320, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolov8_yolov8x_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 160, 160, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolov8_yolov8x_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 160, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolov8_yolov8x_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 320, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolov8_yolov8x_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 160, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolov8_yolov8x_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 640, 80, 80), torch.bfloat16)],
        {"model_names": ["pt_yolov10_yolov10x_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 640, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolov8_yolov8x_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 320, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolov8_yolov8x_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 640, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolov8_yolov8x_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 320, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolov8_yolov8x_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 160, 160), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "onnx_yolov10_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "onnx_yolov10_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 512, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1024, 10, 10), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 512, 10, 10), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 255, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 255, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 255, 10, 10), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 240, 240), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 120, 120), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 120, 120), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 60, 60), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 60, 60), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 512, 30, 30), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 256, 30, 30), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1024, 15, 15), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 512, 15, 15), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 255, 60, 60), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 255, 30, 30), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 255, 15, 15), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 48, 160, 160), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 48, 80, 80), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 192, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 40, 40), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 384, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 768, 10, 10), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 384, 10, 10), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 48, 240, 240), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 96, 120, 120), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 48, 120, 120), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 192, 60, 60), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 96, 60, 60), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 384, 30, 30), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 192, 30, 30), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 768, 15, 15), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 384, 15, 15), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 16, 160, 160), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 16, 80, 80), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 40, 40), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 10, 10), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 10, 10), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 240, 240), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 120, 120), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 60, 60), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 30, 30), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 15, 15), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 320, 320), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 160, 160), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 255, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 240, 240), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 160, 120, 120), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 80, 120, 120), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 320, 60, 60), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 160, 60, 60), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 640, 30, 30), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 320, 30, 30), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1280, 15, 15), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 640, 15, 15), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 80, 320, 320), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 160, 160, 160), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 80, 160, 160), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 320, 80, 80), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 160, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 640, 40, 40), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 320, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1280, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 640, 20, 20), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 56, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 192, 28, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 384, 14, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 80, 56, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 28, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 14, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 56, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 64, 28, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 128, 14, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6n_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 64, 56, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6s_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 28, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6s_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 14, 20), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6s_obj_det_torchhub", "pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 320, 320), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_world_default_obj_det_github", "pt_yolox_yolox_s_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 160, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_world_default_obj_det_github",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 512, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_world_default_obj_det_github",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 4, 40, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_world_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 2, 80, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_world_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 8, 20, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_world_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 16, 208, 208), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 16, 104, 104), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 32, 104, 104), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 32, 52, 52), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 64, 52, 52), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 64, 26, 26), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 128, 26, 26), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 128, 13, 13), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 256, 13, 13), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 1, 52, 52), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_nano_obj_det_torchhub", "pt_yolox_yolox_tiny_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 52, 52), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_nano_obj_det_torchhub", "pt_yolox_yolox_tiny_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1, 26, 26), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_nano_obj_det_torchhub", "pt_yolox_yolox_tiny_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 26, 26), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_nano_obj_det_torchhub", "pt_yolox_yolox_tiny_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 13, 13), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_nano_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 1, 13, 13), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_nano_obj_det_torchhub", "pt_yolox_yolox_tiny_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 80, 13, 13), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_nano_obj_det_torchhub", "pt_yolox_yolox_tiny_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1, 80, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1, 40, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1, 20, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1280, 20, 20), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_x_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 40, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 24, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 288, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 288, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 576, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 816, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 34, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 816, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 816, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1392, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 58, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 2304, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 2304, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1536, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b3_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 48, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 336, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 336, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 960, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 960, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1632, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 2688, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1792, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b4_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 56, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 56, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 240, 56, 56), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 432, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 18, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 432, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 432, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 864, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 36, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 864, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1200, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 50, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1200, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1200, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 2064, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 86, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 2064, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 3456, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 3456, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 528, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1056, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 2904, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 7392, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 320, 320), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 160, 160), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 512, 40, 40), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1024, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5l_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 16, 240, 240), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 16, 120, 120), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 60, 60), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 30, 30), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 15, 15), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 320, 320), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 128, 160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 256, 80, 80), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 512, 40, 40), torch.bfloat16)],
        {
            "model_names": ["pt_yolox_yolox_l_obj_det_torchhub", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1024, 20, 20), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 48, 320, 320), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 96, 160, 160), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 96, 80, 80), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 384, 40, 40), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 768, 20, 20), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 384, 20, 20), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 192, 20, 20), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_m_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 18), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet121_hf_xray_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 64, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 288, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 480, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1344, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1344, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1344, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 3840, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 160, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 3840, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 2560, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (Sigmoid0, [((1, 6, 2816), torch.float32)], {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99}),
    (Sigmoid0, [((1, 29, 8960), torch.float32)], {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99}),
    (
        Sigmoid0,
        [((1, 80, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 160, 40, 40), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 320, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1280, 10, 10), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 640, 10, 10), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5x_img_cls_torchhub_320x320"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 384, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 384, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 384, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 768, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1056, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 44, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1056, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1824, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 76, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1824, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 3072, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 3072, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 2048, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b5_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (Sigmoid0, [((1, 29, 4864), torch.float32)], {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Sigmoid0,
        [((1, 48, 320, 320), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 96, 160, 160), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 192, 80, 80), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 384, 40, 40), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 768, 20, 20), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5m_img_cls_torchhub_640x640"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 64, 224, 320), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 128, 112, 160), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 64, 112, 160), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 256, 56, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 128, 56, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 512, 28, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 256, 28, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 1024, 14, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 512, 14, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 256, 160, 160), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 512, 80, 80), torch.bfloat16)],
        {"model_names": ["pt_yolov9_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 29, 2816), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 39, 4864), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1, 512, 512), torch.bfloat16)],
        {"model_names": ["pt_vgg19_unet_default_sem_seg_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 16, 320, 320), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 32, 640, 640), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 255, 160, 160), torch.float32)],
        {"model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 24, 208, 208), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 48, 104, 104), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 24, 104, 104), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 96, 52, 52), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 48, 52, 52), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 192, 26, 26), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 96, 26, 26), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 384, 13, 13), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 192, 13, 13), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 96, 13, 13), torch.bfloat16)],
        {"model_names": ["pt_yolox_yolox_tiny_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Sigmoid0,
        [((1, 48, 224, 224), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 12, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 48, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 24, 224, 224), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 6, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 24, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 224, 224), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 112, 112), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 240, 112, 112), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 10, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 240, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 240, 56, 56), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 384, 56, 56), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 16, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 384, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 384, 28, 28), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 768, 28, 28), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 768, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1056, 28, 28), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 44, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1056, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1056, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1824, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 76, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1824, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 3072, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 128, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 3072, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 2048, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 35, 4864), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 40, 144, 144), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 40, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 24, 144, 144), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 144, 144), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 72, 72), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 72, 72), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 8, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 192, 36, 36), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 288, 36, 36), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 288, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 288, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 576, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 576, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 816, 18, 18), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 34, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 816, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 816, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1392, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 58, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1392, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 2304, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 2304, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1536, 9, 9), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (Sigmoid0, [((1, 1, 256, 256), torch.float32)], {"model_names": ["onnx_unet_base_img_seg_torchhub"], "pcc": 0.99}),
    (
        Sigmoid0,
        [((1, 80, 40, 40), torch.float32)],
        {"model_names": ["onnx_yolov10_default_obj_det_github", "onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 80, 20, 20), torch.float32)],
        {"model_names": ["onnx_yolov10_default_obj_det_github", "onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 80, 8400), torch.float32)],
        {"model_names": ["onnx_yolov10_default_obj_det_github", "onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1, 448, 448), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 588, 5504), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((3, 64, 64), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((6, 64, 64), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((12, 64, 64), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((24, 64, 64), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 4, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 96, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 64, 64), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 288, 32, 32), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 288, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 528, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 22, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 528, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 720, 16, 16), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 30, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 720, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 720, 8, 8), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1248, 8, 8), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 52, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1248, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 2112, 8, 8), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 88, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 2112, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1408, 8, 8), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2a_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 24, 160, 160), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 160, 160), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 80, 80), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 336, 40, 40), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 14, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 336, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 336, 20, 20), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 672, 20, 20), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 28, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 672, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 960, 20, 20), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 960, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 960, 10, 10), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1632, 10, 10), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 68, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1632, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 2688, 10, 10), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 112, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 2688, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1792, 10, 10), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((2, 7, 2048), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 100, 4), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 32, 112, 112), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 96, 112, 112), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 96, 56, 56), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 56, 56), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 28, 28), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 240, 28, 28), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 240, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 480, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 20, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 480, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 672, 14, 14), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 672, 7, 7), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1152, 7, 7), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1152, 1, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 1280, 7, 7), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b0_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 60, 1, 12), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 120, 1, 12), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
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
        Sigmoid0,
        [((1, 480, 1, 12), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 522, 8192), torch.float32)],
        {"model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 60, 60), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 144, 30, 30), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 240, 30, 30), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 240, 15, 15), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 480, 15, 15), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 672, 15, 15), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 672, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1152, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1920, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 80, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1920, 1, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 1280, 8, 8), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Sigmoid0,
        [((1, 48), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Sigmoid0,
        [((1, 528, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 22, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 720, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 30, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 720, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 720, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1248, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 52, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1248, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 2112, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 88, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 2112, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1408, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 48, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 24, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 144, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 240, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 384, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 768, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1056, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 1824, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 3072, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Sigmoid0,
        [((1, 2048, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Sigmoid")

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
