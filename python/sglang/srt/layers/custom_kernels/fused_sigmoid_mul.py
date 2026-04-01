"""Fused x * sigmoid(gate) kernel."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_sigmoid_mul_kernel(
    x_ptr, gate_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    output = x * tl.sigmoid(gate)
    tl.store(output_ptr + offsets, output.to(tl.float16), mask=mask)


def fused_sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compute x * sigmoid(gate) in a single kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _fused_sigmoid_mul_kernel[grid](
        x, gate, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
