import torch
import enum
import math
from typing import Type, Tuple, Callable, Optional, Literal
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from cutlass.cute.runtime import from_dlpack

def assume_strides_aligned(t):
    """Assume all strides except the last are divisible by 128 bits.

    Python int strides (e.g., stride=0 from GQA expand) are kept as-is
    since they're static and don't need alignment assumptions.
    """
    divby = 128 // t.element_type.width
    strides = tuple(s if isinstance(s, int) else cute.assume(s, divby=divby) for s in t.stride[:-1])
    return (*strides, t.stride[-1])

def assume_tensor_aligned(t):
    """Rebuild a tensor with 128-bit aligned stride assumptions. Passes through None."""
    if t is None:
        return None
    return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=assume_strides_aligned(t)))

def to_cute_tensor(t, assumed_align=16, leading_dim=-1, fully_dynamic=False, enable_tvm_ffi=True):
    """Convert torch tensor to cute tensor for TVM FFI. leading_dim=-1 defaults to t.ndim-1."""
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=enable_tvm_ffi)
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)

@cute.kernel
def cute_kernel(
    mQ: cute.Tensor,
    mK: cute.Tensor=None,
    mV: cute.Tensor=None,
    mO: cute.Tensor=None,
    N: cute.Uint32 = 0,
    ):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    thread_idx = bidx * bdim + tidx
    if thread_idx < N:
        mO[thread_idx] = mQ[thread_idx] + cute.BFloat16(1)
    # output = input + 1
    return 

@cute.jit
def launch_kernel(
    mQ: cute.Tensor,
    mK: cute.Tensor=None,
    mV: cute.Tensor=None,
    mO: cute.Tensor=None,
    N :cute.Uint32 = 0,
    ):
    q_dtype = mQ.element_type
    # k_dtype = mK.element_type
    print(q_dtype)
    print(q_dtype.width)

    q_layout_transpose = [1, 0, 2]
    print(f'{mQ=}')
    mQ = assume_tensor_aligned(mQ)
    print(f'assume_tensor_aligned {mQ.stride=}')
    mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=q_layout_transpose))
    print(f'compile select {mQ.layout.stride=}')
    cute.printf(f'running select {mQ.layout.stride=}')
    # Create a 2D identity layout with shape (4,4)
    layout = cute.make_identity_layout((4,4))     # stride=(1@0,1@1)
    print(f'2D identity layout: {layout.shape=}, {layout.stride=}')
    # Create a 3D identity layout
    layout = cute.make_identity_layout((32,16,8)) # stride=(1@0,1@1,1@2)
    cute.printf(f'3D identity layout: {layout.shape=}, {layout.stride=}')
    num_threads_per_block = 256
    grid_dim = cute.ceil_div(cute.Int32(N), num_threads_per_block), 1, 1
    cute_kernel(
        mQ=mQ,
        mO=mO,
        N=N,
    ).launch(
        grid=grid_dim,
        block=(num_threads_per_block, 1, 1),
    )

if __name__ == "__main__":
    torch.manual_seed(13)
    shape = [2, 4, 8]
    input = torch.ones(shape, dtype=torch.bfloat16).cuda()
    output = torch.empty(shape, dtype=torch.bfloat16).cuda()
    print(f'torch tensor: {input}')
    input_tensor = to_cute_tensor(input.detach())
    output_tensor = to_cute_tensor(output.detach())
    kernel_cache = cute.compile(
        launch_kernel,
        mQ=input_tensor,
        mO=output_tensor,
        N=input.numel(),
    )
    launch_kernel(input_tensor, mO=output_tensor, N=input.numel())
    print(output)