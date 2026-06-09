# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backward-compatibility shim.

All FlashInfer utilities now live in ``vllm.utils.flashinfer``.
Import from there directly.
"""
from vllm.utils.flashinfer import (  # noqa: F401
    FlashinferMoeBackend,
    activation_to_flashinfer_int,
    activation_to_flashinfer_type,
    align_fp4_moe_weights_for_fi,
    align_fp8_moe_weights_for_fi,
    align_trtllm_fp4_moe_hidden_dim_for_fi,
    convert_moe_weights_to_flashinfer_trtllm_block_layout,
    get_flashinfer_moe_backend,
    is_flashinfer_supporting_global_sf,
    prepare_fp8_moe_layer_for_fi,
    rotate_weights_for_fi_trtllm_fp8_per_tensor_moe,
    swap_w13_to_w31,
)

