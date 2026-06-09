# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for torch.compile fusion passes.

Each test verifies that a model produces identical greedy outputs when
a fusion pass is enabled versus when it is disabled.  Dummy weights are
used so the tests can run without downloading large checkpoints; the
comparison is still meaningful because fusions must be numerically
equivalent to the unfused path.
"""

import json

import pytest

from tests.utils import compare_two_settings, create_new_process_for_each_test
from vllm.config import CompilationMode
from vllm.platforms import current_platform

# --------------------------------------------------------------------------- #
# Models and their reduced hf_overrides for running quickly.                  #
# --------------------------------------------------------------------------- #

LLAMA_FP8_MODEL = "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8"

# Shrink the model so it fits within CI memory limits.
LLAMA_FP8_HF_OVERRIDES = {
    "num_hidden_layers": 4,
    "hidden_size": 512,
    "intermediate_size": 800,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
}


def _common_args(hf_overrides: dict) -> list[str]:
    return [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "512",
        "--max-num-seqs",
        "4",
        "--load-format",
        "dummy",
        "--hf-overrides",
        json.dumps(hf_overrides),
    ]


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


@create_new_process_for_each_test()
@pytest.mark.parametrize(
    "pass_config_overrides",
    [
        {"fuse_norm_quant": True},
        {"fuse_act_quant": True},
        {"fuse_attn_quant": True},
        {"fuse_norm_quant": True, "fuse_act_quant": True, "fuse_attn_quant": True},
    ],
    ids=["rms_quant", "act_quant", "attn_quant", "all_quant"],
)
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="CUDA/ROCm required"
)
def test_fp8_fusion_correctness(pass_config_overrides: dict):
    """Fused FP8 passes must produce the same greedy output as no fusions."""
    common = _common_args(LLAMA_FP8_HF_OVERRIDES)

    fused_config = {
        "mode": CompilationMode.VLLM_COMPILE,
        "pass_config": pass_config_overrides,
    }

    fused_args = [
        *common,
        "--compilation_config",
        json.dumps(fused_config),
    ]

    # Baseline: eager mode (no torch.compile, no fusions).
    baseline_args = [*common, "--enforce-eager"]

    compare_two_settings(
        LLAMA_FP8_MODEL,
        fused_args,
        baseline_args,
        method="generate",
        force_v1_runner=True,
    )


@create_new_process_for_each_test()
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="CUDA/ROCm required"
)
def test_norm_rope_fusion_correctness():
    """QK-norm + RoPE fusion must produce the same output as the unfused path."""
    # Qwen3 uses QK-norm so it exercises the norm_rope_fusion pass.
    model = "Qwen/Qwen3-0.6B"
    common = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "512",
        "--max-num-seqs",
        "4",
        "--load-format",
        "dummy",
        "--hf-overrides",
        json.dumps({"num_hidden_layers": 4}),
    ]

    fused_config = {
        "mode": CompilationMode.VLLM_COMPILE,
        "pass_config": {"enable_qk_norm_rope_fusion": True},
    }

    fused_args = [
        *common,
        "--compilation_config",
        json.dumps(fused_config),
    ]

    baseline_args = [*common, "--enforce-eager"]

    compare_two_settings(
        model,
        fused_args,
        baseline_args,
        method="generate",
        force_v1_runner=True,
    )
