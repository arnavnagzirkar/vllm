# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MoEKernelOracle class hierarchy."""
import pytest

from vllm.model_executor.layers.fused_moe.oracle.base import MoEKernelOracle


def test_moe_kernel_oracle_is_abstract():
    """MoEKernelOracle cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MoEKernelOracle()


def test_incomplete_subclass_cannot_be_instantiated():
    """A subclass that does not implement all abstract methods raises TypeError."""

    class IncompleteOracle(MoEKernelOracle):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteOracle()


def test_oracle_subclass_hierarchy():
    """Each concrete oracle is a subclass of MoEKernelOracle."""
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoEKernelOracle
    from vllm.model_executor.layers.fused_moe.oracle.int8 import Int8MoEKernelOracle
    from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
        WNA16MoEKernelOracle,
    )
    from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import MxFp8MoEKernelOracle
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoEKernelOracle
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        UnquantizedMoEKernelOracle,
    )
    from vllm.model_executor.layers.fused_moe.oracle.w4a8 import W4A8MoEKernelOracle
    from vllm.model_executor.layers.fused_moe.oracle.w4a8_int8 import (
        W4A8Int8MoEKernelOracle,
    )

    oracle_classes = [
        Fp8MoEKernelOracle,
        Int8MoEKernelOracle,
        WNA16MoEKernelOracle,
        MxFp8MoEKernelOracle,
        NvFp4MoEKernelOracle,
        UnquantizedMoEKernelOracle,
        W4A8MoEKernelOracle,
        W4A8Int8MoEKernelOracle,
    ]
    for cls in oracle_classes:
        assert issubclass(cls, MoEKernelOracle), (
            f"{cls.__name__} is not a subclass of MoEKernelOracle"
        )


def test_mxfp8_oracle_inherits_from_fp8():
    """MxFp8MoEKernelOracle inherits from Fp8MoEKernelOracle."""
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoEKernelOracle
    from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import MxFp8MoEKernelOracle

    assert issubclass(MxFp8MoEKernelOracle, Fp8MoEKernelOracle)


def test_mxfp4_oracle_is_oracle_subclass():
    """MxFp4MoEKernelOracle is a subclass of MoEKernelOracle."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import MxFp4MoEKernelOracle

    assert issubclass(MxFp4MoEKernelOracle, MoEKernelOracle)


def test_concrete_oracle_implements_abstract_methods():
    """Each concrete oracle class implements the required abstract methods."""
    import inspect

    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoEKernelOracle
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        UnquantizedMoEKernelOracle,
    )

    for cls in (Fp8MoEKernelOracle, UnquantizedMoEKernelOracle):
        # If any abstract methods remain unimplemented, inspect will flag them.
        abstract_methods = {
            name
            for name, member in inspect.getmembers(cls)
            if getattr(member, "__isabstractmethod__", False)
        }
        assert not abstract_methods, (
            f"{cls.__name__} still has unimplemented abstract methods: "
            f"{abstract_methods}"
        )


def test_default_methods_raise_not_implemented():
    """convert_to_kernel_format and make_moe_quant_config raise by default."""
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        UnquantizedMoEKernelOracle,
    )

    oracle = UnquantizedMoEKernelOracle()
    with pytest.raises(NotImplementedError):
        oracle.make_moe_quant_config()
