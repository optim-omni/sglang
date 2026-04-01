"""Fused residual + hidden_states * scale kernel."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_scale_kernel(
    residual_ptr, hidden_ptr, output_ptr,
    scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    residual = tl.load(residual_ptr + offsets, mask=mask)
    hidden = tl.load(hidden_ptr + offsets, mask=mask)
    output = residual + hidden * scale
    tl.store(output_ptr + offsets, output, mask=mask)


def fused_add_scale(residual: torch.Tensor, hidden: torch.Tensor, scale: float) -> torch.Tensor:
    """Compute residual + hidden * scale in a single kernel."""
    output = torch.empty_like(residual)
    n_elements = residual.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _fused_add_scale_kernel[grid](
        residual, hidden, output,
        scale, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
