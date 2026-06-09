# Agent Result: Issue #31414

## Root Cause

Two separate modules contained FlashInfer-related utilities with no clear separation of concerns:

- `vllm.utils.flashinfer` - availability checks, lazy import wrappers, custom ops
- `vllm.model_executor.layers.quantization.utils.flashinfer_utils` - MoE weight manipulation utilities

Both modules served as "FlashInfer utility" modules, causing confusion about which to import from.

## Change Made

All MoE weight utility functions were moved from
`vllm.model_executor.layers.quantization.utils.flashinfer_utils`
into `vllm.utils.flashinfer`, which is the canonical single location for all FlashInfer utilities in vLLM.

Functions/classes moved to `vllm.utils.flashinfer`:
- `FlashinferMoeBackend` (Enum)
- `vllm.utils.flashinfer.activation_to_flashinfer_int`
- `vllm.utils.flashinfer.activation_to_flashinfer_type`
- `vllm.utils.flashinfer.swap_w13_to_w31`
- `vllm.utils.flashinfer.rotate_weights_for_fi_trtllm_fp8_per_tensor_moe`
- `vllm.utils.flashinfer.get_flashinfer_moe_backend`
- `vllm.utils.flashinfer.is_flashinfer_supporting_global_sf`
- `vllm.utils.flashinfer.convert_moe_weights_to_flashinfer_trtllm_block_layout`
- `vllm.utils.flashinfer.align_fp4_moe_weights_for_fi`
- `vllm.utils.flashinfer.align_trtllm_fp4_moe_hidden_dim_for_fi`
- `vllm.utils.flashinfer.align_fp8_moe_weights_for_fi`
- `vllm.utils.flashinfer.prepare_fp8_moe_layer_for_fi`

`flashinfer_utils.py` was replaced with a backward-compat re-export shim so all existing callers continue to work without changes.

The `MoEActivation` import was made lazy (inside the function body) to avoid a `utils -> model_executor` module-level layering violation.

## Testing

- All existing callers (in `vllm/model_executor/layers/fused_moe/`, `vllm/model_executor/layers/quantization/`) continue to work via the backward-compat shim.
- Import identity check: objects imported from the shim are the same objects as those imported directly from `vllm.utils.flashinfer`.
- Functional tests: `FlashinferMoeBackend`, `swap_w13_to_w31`, `is_flashinfer_supporting_global_sf`, `align_fp8_moe_weights_for_fi` (no-op and padding cases) all behave correctly.

## Lint

`ruff check` passes with no errors on both changed files after auto-fix of import sort order.
