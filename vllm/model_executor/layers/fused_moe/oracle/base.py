# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from typing import Any

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)


class MoEKernelOracle(ABC):
    """Abstract base class for MoE kernel oracles.

    An oracle is responsible for:
    - Selecting the appropriate backend and expert class for a given config.
    - Converting weights to the format expected by the selected kernel.
    - Creating the FusedMoEQuantConfig for the selected kernel.
    - Constructing the FusedMoEKernel.

    Subclasses must implement :meth:`select_moe_backend` and
    :meth:`make_moe_kernel`. The weight-conversion and quant-config methods
    are optional - the default implementations raise ``NotImplementedError``
    so that oracles which do not need them can simply omit them.
    """

    @abstractmethod
    def select_moe_backend(
        self,
        config: FusedMoEConfig,
        **kwargs: Any,
    ) -> tuple[Any, type[mk.FusedMoEExperts] | None]:
        """Select the MoE backend and expert class for the given config.

        Returns a ``(backend, experts_cls)`` tuple.  ``experts_cls`` may be
        ``None`` for backends that are not yet migrated to the modular-kernel
        structure.
        """
        raise NotImplementedError

    def convert_to_kernel_format(self, **kwargs: Any) -> Any:
        """Convert weights to the format expected by the selected kernel.

        Not all oracles require weight conversion.  The default raises
        ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement convert_to_kernel_format."
        )

    def make_moe_quant_config(self, **kwargs: Any) -> FusedMoEQuantConfig:
        """Create the FusedMoEQuantConfig for this oracle.

        Not all oracles need a separate quant config.  The default raises
        ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement make_moe_quant_config."
        )

    @abstractmethod
    def make_moe_kernel(
        self,
        moe_quant_config: FusedMoEQuantConfig,
        moe_config: FusedMoEConfig,
        experts_cls: type[mk.FusedMoEExperts],
        **kwargs: Any,
    ) -> mk.FusedMoEKernel:
        """Construct and return the FusedMoEKernel."""
        raise NotImplementedError
