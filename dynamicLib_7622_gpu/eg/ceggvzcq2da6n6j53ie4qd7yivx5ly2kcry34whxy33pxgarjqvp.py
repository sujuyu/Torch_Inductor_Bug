
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/6l/c6lx3w7annboscvqny67w4ma2bdvnqe6mjzfugpdmrcc5tukmxwf.py
# Source Nodes: [user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm], Original ATen: [aten.native_layer_norm]
# user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm => add_2, add_3, mul_10, mul_11, rsqrt, sub_2, var_mean
triton_per_fused_native_layer_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 16, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 16.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r1 + (16*x0)), tmp27, rmask & xmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/iz/cizybixqsgvvri3ke7q52qbj5qratl63u35kyrl4smnth2ndx7zv.py
# Source Nodes: [where], Original ATen: [aten.scalar_tensor, aten.where]
# where => full_default_1, where
triton_poi_fused_scalar_tensor_where_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_scalar_tensor_where_1', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 176)
    x2 = xindex
    tmp2 = tl.load(in_ptr0 + ((x1 // 50)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp0 = x1 % 50
    tmp1 = tmp0.to(tl.float64)
    tmp3 = tl.full([1], 50.0, tl.float64)
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = tmp1 < tmp4
    tmp7 = 0.0
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/gp/cgpqnanhf645zczmn6pagexw7azue7q675cburz7ymzk673uie23.py
# Source Nodes: [user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm], Original ATen: [aten.native_layer_norm]
# user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm => var_mean_1
triton_per_fused_native_layer_norm_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[131072, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_2', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 16, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/fe/cfenprznzdpqnb443odi6czvwvnqbowbp5pvrlmisbuo2mhoq6dj.py
# Source Nodes: [where_1], Original ATen: [aten.scalar_tensor, aten.where]
# where_1 => full_default_3, where_1
triton_poi_fused_scalar_tensor_where_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_scalar_tensor_where_3', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 172)
    x2 = xindex
    tmp2 = tl.load(in_ptr0 + ((x1 // 50)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp0 = x1 % 50
    tmp1 = tmp0.to(tl.float64)
    tmp3 = tl.full([1], 50.0, tl.float64)
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = tmp1 < tmp4
    tmp7 = 0.0
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/gr/cgrbp2uhbknxl6c2uixmuezofacno5b2jtf7dynjfvhd772dsrlj.py
# Source Nodes: [logical_or, lt, sum_1, sum_3, to, to_2], Original ATen: [aten._to_copy, aten.logical_or, aten.lt, aten.sum]
# logical_or => logical_or
# lt => lt
# sum_1 => sum_1
# sum_3 => sum_3
# to => convert_element_type
# to_2 => convert_element_type_2
triton_per_fused__to_copy_logical_or_lt_sum_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp64', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_logical_or_lt_sum_4', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 50
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp2 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = r1
    tmp1 = tmp0.to(tl.float64)
    tmp3 = tl.full([1, 1], 50.0, tl.float64)
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = tmp1 < tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = 0.0
    tmp12 = tmp10 == tmp11
    tmp13 = tmp5 | tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp18, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/mn/cmnj3yukewo5k4o3ipliituwh2tjvgz2btwb2eagw5xhu2gjozht.py
# Source Nodes: [leaky_relu], Original ATen: [aten.leaky_relu]
# leaky_relu => gt, mul_1, where_2
triton_poi_fused_leaky_relu_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 176
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.01
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/nm/cnmen3auol7xyadxsofsuqgkixttzyezhiypyfuf24a5ntennm3w.py
# Source Nodes: [mul, pow_1, sum_4], Original ATen: [aten.mul, aten.pow, aten.sum]
# mul => mul_3
# pow_1 => pow_1
# sum_4 => sum_4
triton_per_fused_mul_pow_sum_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_pow_sum_6', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 50
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x1 = (xindex // 176)
    x0 = xindex % 176
    x3 = xindex
    tmp2 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x0 + (176*r2) + (8800*x1)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0 + (176*r2) + (8800*x1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = r2
    tmp1 = tmp0.to(tl.float64)
    tmp3 = tl.full([1, 1], 50.0, tl.float64)
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = tmp1 < tmp4
    tmp7 = 0.0
    tmp8 = tmp6 == tmp7
    tmp9 = tmp5 | tmp8
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp9, tmp14, tmp7)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp21 = 1 / tmp20
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr1 + (x0 + (1872*x1)), tmp22, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/ix/cixdsy45epigtua7hnnb2trqacs6zuo7dswwuahb3l5xhtzttbsb.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone
triton_poi_fused_clone_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[65536, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 50
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = (yindex // 128)
    y4 = yindex % 128
    y1 = (yindex // 16) % 8
    y0 = yindex % 16
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (y4 + (128*x3) + (6400*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y1 + (8*x3) + (400*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y1 + (8*x3) + (400*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 16.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3 + (50*y5)), tmp13, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/c7/cc7luc6ehgsuslee67ue4mhxjmp54fqnhxn5pm7r3hw67qpcpita.py
# Source Nodes: [softmax, truediv], Original ATen: [aten._softmax, aten.div]
# softmax => amax, div_1, exp, sub_4, sum_7
# truediv => div
triton_per_fused__softmax_div_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_div_8', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 50
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (50*x0)), rmask & xmask, other=0.0)
    tmp1 = 4.0
    tmp2 = tmp0 / tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = tmp8 / tmp12
    tl.store(out_ptr2 + (r1 + (50*x0)), tmp13, rmask & xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/l6/cl65xknsmhvdzvpivttvbp6f3osgfs6abk7h623mgjpqnufnowgi.py
# Source Nodes: [matmul_1], Original ATen: [aten.clone]
# matmul_1 => clone_1
triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 50
    x2 = (xindex // 800) % 8
    x3 = (xindex // 6400)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*x2) + (128*x1) + (6400*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (16*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/et/cetwnsq7peyp6mbxhn3synhkfsx2acgkvdpqup6mlp3vx6echqnh.py
# Source Nodes: [leaky_relu_1], Original ATen: [aten.leaky_relu]
# leaky_relu_1 => gt_1, mul_5, where_4
triton_poi_fused_leaky_relu_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 172
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.01
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/ti/ctiqpdlpgaiduxafg6qyu7cxfydu4g6aep4cm3sra4idbsog63rw.py
# Source Nodes: [mul_1, pow_2, sum_6], Original ATen: [aten.mul, aten.pow, aten.sum]
# mul_1 => mul_7
# pow_2 => pow_2
# sum_6 => sum_6
triton_per_fused_mul_pow_sum_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_pow_sum_11', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 50
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x1 = (xindex // 172)
    x0 = xindex % 172
    x3 = xindex
    tmp2 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x0 + (172*r2) + (8600*x1)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0 + (172*r2) + (8600*x1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = r2
    tmp1 = tmp0.to(tl.float64)
    tmp3 = tl.full([1, 1], 50.0, tl.float64)
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = tmp1 < tmp4
    tmp7 = 0.0
    tmp8 = tmp6 == tmp7
    tmp9 = tmp5 | tmp8
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp9, tmp14, tmp7)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp21 = 1 / tmp20
    tmp22 = tmp19 * tmp21
    tl.store(out_ptr1 + (x0 + (1872*x1)), tmp22, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/jq/cjqzf4pkgzdrvr43psci747zbqvjpauec7a2eh52g5cssxwlnktf.py
# Source Nodes: [cat], Original ATen: [aten.cat]
# cat => cat
triton_poi_fused_cat_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_12', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 204
    x1 = (xindex // 204)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (1872*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/42/c42dvfe3p6etsydyccf4pzgcvjkxe3ylssnr24pmc2pkj5mpl2c2.py
# Source Nodes: [cat], Original ATen: [aten.cat]
# cat => cat
triton_poi_fused_cat_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_13', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 220
    x1 = (xindex // 220)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (1872*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/ce/cces32z2rzbxenwigzpoeg6mjxel7jmcly646dik76zzsgnkrysw.py
# Source Nodes: [cat], Original ATen: [aten.cat]
# cat => cat
triton_poi_fused_cat_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_14', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 688
    x1 = (xindex // 688)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (1872*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/rl/crli2g66nylbgqwqnwls5js4wo76vfpiofboqfonjwldqzljnlq3.py
# Source Nodes: [cat], Original ATen: [aten.cat]
# cat => cat
triton_poi_fused_cat_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_15', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (1872*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/o2/co2bh6ja7u5ef56pyz7oesplbsy5nvnwjvfhdxlzw5srqrneklvc.py
# Source Nodes: [cat], Original ATen: [aten.cat]
# cat => cat
triton_poi_fused_cat_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_16', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 156
    x1 = (xindex // 156)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (1872*x1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/po/cpocpqlx2hjyw32sszurlvvg3e7s5lyd577vxrui6ymwmcqvam2r.py
# Source Nodes: [user_model_ctr_net_tower_layers_0_act, user_model_ctr_net_tower_layers_0_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# user_model_ctr_net_tower_layers_0_act => gt_2, mul_27, where_6
# user_model_ctr_net_tower_layers_0_norm => add_10, add_11, mul_24, mul_25, mul_26, reciprocal, sqrt, sub_8
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/vr/cvrxhs4yiyundsifp3klji3euekjwsgnjsxdet5i6jay3v53kgzt.py
# Source Nodes: [user_model_ctr_net_tower_layers_1_act, user_model_ctr_net_tower_layers_1_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# user_model_ctr_net_tower_layers_1_act => gt_3, mul_31, where_7
# user_model_ctr_net_tower_layers_1_norm => add_12, add_13, mul_28, mul_29, mul_30, reciprocal_1, sqrt_1, sub_9
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/yy/cyyvalbqnff5omgmlnxer3muf3xlwusgrwxmgqidozscbgli5hjw.py
# Source Nodes: [user_model_ctr_net_tower_layers_2_act, user_model_ctr_net_tower_layers_2_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# user_model_ctr_net_tower_layers_2_act => gt_4, mul_35, where_8
# user_model_ctr_net_tower_layers_2_norm => add_14, add_15, mul_32, mul_33, mul_34, reciprocal_2, sqrt_2, sub_10
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = 0.0
    tmp16 = tmp14 > tmp15
    tmp17 = 0.01
    tmp18 = tmp14 * tmp17
    tmp19 = tl.where(tmp16, tmp14, tmp18)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/6t/c6tzaqb7ibfjag6y77mfdkven4ehtn43m6lhaxkskry2zhoa445q.py
# Source Nodes: [cat_1], Original ATen: [aten.cat]
# cat_1 => cat_1
triton_poi_fused_cat_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_20', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'f0131f321b9fa5e5db96058d255dfc3dd9f0e55f5b20a486e3ce37d6a635e46d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 568
    x1 = (xindex // 568)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 204, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (204*x1)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 220, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-204) + x0 + (16*x1)), tmp11 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 396, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-220) + x0 + (1872*x1)), tmp18 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1], 568, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-396) + x0 + (1872*x1)), tmp22 & xmask, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tl.store(out_ptr0 + (x2), tmp30, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1 = args
    args.clear()
    arg122_1_size = arg122_1.size()
    s0 = arg122_1_size[0]
    assert_size_stride(arg122_1, (s0, 688), (688, 1))
    assert_size_stride(arg123_1, (s0, 50, 176), (8800, 176, 1))
    assert_size_stride(arg124_1, (s0, 1), (1, 1))
    assert_size_stride(arg125_1, (s0, 220), (220, 1))
    assert_size_stride(arg126_1, (s0, 50, 172), (8600, 172, 1))
    assert_size_stride(arg127_1, (s0, 1), (1, 1))
    assert_size_stride(arg128_1, (s0, 156), (156, 1))
    assert_size_stride(arg129_1, (s0, 16), (16, 1))
    assert_size_stride(arg130_1, (s0, 204), (204, 1))

    for kernel in globals().values():
        if isinstance(kernel, torch._inductor.triton_heuristics.CachingAutotuner):
            kernel.cuda_kernel_saved = False
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((s0, 128), (128, 1), torch.float32)
        # Source Nodes: [user_model_multi_h_attens_item_clk_seq_ta_proj_q_linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(reinterpret_tensor(L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_q_linear_bias, (s0, 128), (0, 1), 0), reinterpret_tensor(arg128_1, (s0, 156), (156, 1), 0), reinterpret_tensor(L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_q_linear_weight, (156, 128), (1, 156), 0), alpha=1, beta=1, out=buf0)
        buf24 = empty_strided_cuda((s0, 1, 8, 16), (128, 128, 16, 1), torch.float32)
        # Source Nodes: [user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_0_xnumel = 8*s0
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_0.run(buf0, L__self___user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm_weight, L__self___user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm_bias, buf24, triton_per_fused_native_layer_norm_0_xnumel, 16, grid=grid(triton_per_fused_native_layer_norm_0_xnumel), stream=stream0)
        buf4 = empty_strided_cuda((50*s0, 176), (176, 1), torch.float32)
        # Source Nodes: [where], Original ATen: [aten.scalar_tensor, aten.where]
        triton_poi_fused_scalar_tensor_where_1_xnumel = 8800*s0
        triton_poi_fused_scalar_tensor_where_1.run(arg124_1, arg123_1, buf4, triton_poi_fused_scalar_tensor_where_1_xnumel, grid=grid(triton_poi_fused_scalar_tensor_where_1_xnumel), stream=stream0)
        del arg123_1
        buf5 = empty_strided_cuda((50*s0, 128), (128, 1), torch.float32)
        # Source Nodes: [user_model_multi_h_attens_item_clk_seq_ta_proj_k_linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(reinterpret_tensor(L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_k_linear_bias, (50*s0, 128), (0, 1), 0), reinterpret_tensor(buf4, (50*s0, 176), (176, 1), 0), reinterpret_tensor(L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_k_linear_weight, (176, 128), (1, 176), 0), alpha=1, beta=1, out=buf5)
        buf6 = empty_strided_cuda((s0, 50, 8, 1), (400, 8, 1, 400*s0), torch.float32)
        buf7 = empty_strided_cuda((s0, 50, 8, 1), (400, 8, 1, 400*s0), torch.float32)
        # Source Nodes: [user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_2_xnumel = 400*s0
        triton_per_fused_native_layer_norm_2.run(buf5, buf6, buf7, triton_per_fused_native_layer_norm_2_xnumel, 16, grid=grid(triton_per_fused_native_layer_norm_2_xnumel), stream=stream0)
        buf9 = buf0; del buf0  # reuse
        # Source Nodes: [user_model_multi_h_attens_query_seq_ta_proj_q_linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(reinterpret_tensor(L__self___user_model_multi_h_attens_query_seq_ta_proj_q_linear_bias, (s0, 128), (0, 1), 0), reinterpret_tensor(arg128_1, (s0, 156), (156, 1), 0), reinterpret_tensor(L__self___user_model_multi_h_attens_query_seq_ta_proj_q_linear_weight, (156, 128), (1, 156), 0), alpha=1, beta=1, out=buf9)
        buf39 = empty_strided_cuda((s0, 1, 8, 16), (128, 128, 16, 1), torch.float32)
        # Source Nodes: [user_model_multi_h_attens_query_seq_ta_q_layer_norm], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_0_xnumel = 8*s0
        triton_per_fused_native_layer_norm_0.run(buf9, L__self___user_model_multi_h_attens_query_seq_ta_q_layer_norm_weight, L__self___user_model_multi_h_attens_query_seq_ta_q_layer_norm_bias, buf39, triton_per_fused_native_layer_norm_0_xnumel, 16, grid=grid(triton_per_fused_native_layer_norm_0_xnumel), stream=stream0)
        del buf9
        buf13 = empty_strided_cuda((50*s0, 172), (172, 1), torch.float32)
        # Source Nodes: [where_1], Original ATen: [aten.scalar_tensor, aten.where]
        triton_poi_fused_scalar_tensor_where_3_xnumel = 8600*s0
        triton_poi_fused_scalar_tensor_where_3.run(arg127_1, arg126_1, buf13, triton_poi_fused_scalar_tensor_where_3_xnumel, grid=grid(triton_poi_fused_scalar_tensor_where_3_xnumel), stream=stream0)
        del arg126_1
        buf14 = empty_strided_cuda((50*s0, 128), (128, 1), torch.float32)
        # Source Nodes: [user_model_multi_h_attens_query_seq_ta_proj_k_linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(reinterpret_tensor(L__self___user_model_multi_h_attens_query_seq_ta_proj_k_linear_bias, (50*s0, 128), (0, 1), 0), reinterpret_tensor(buf13, (50*s0, 172), (172, 1), 0), reinterpret_tensor(L__self___user_model_multi_h_attens_query_seq_ta_proj_k_linear_weight, (172, 128), (1, 172), 0), alpha=1, beta=1, out=buf14)
        buf15 = empty_strided_cuda((s0, 50, 8, 1), (400, 8, 1, 400*s0), torch.float32)
        buf16 = empty_strided_cuda((s0, 50, 8, 1), (400, 8, 1, 400*s0), torch.float32)
        # Source Nodes: [user_model_multi_h_attens_query_seq_ta_k_layer_norm], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_2_xnumel = 400*s0
        triton_per_fused_native_layer_norm_2.run(buf14, buf15, buf16, triton_per_fused_native_layer_norm_2_xnumel, 16, grid=grid(triton_per_fused_native_layer_norm_2_xnumel), stream=stream0)
        buf18 = empty_strided_cuda((s0, 1), (1, s0), torch.float32)
        buf23 = empty_strided_cuda((s0, 1), (1, s0), torch.float32)
        # Source Nodes: [logical_or, lt, sum_1, sum_3, to, to_2], Original ATen: [aten._to_copy, aten.logical_or, aten.lt, aten.sum]
        triton_per_fused__to_copy_logical_or_lt_sum_4.run(arg124_1, buf18, buf23, s0, 50, grid=grid(s0), stream=stream0)
        buf19 = empty_strided_cuda((50*s0, 176), (176, 1), torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf4, (50*s0, 176), (176, 1), 0), reinterpret_tensor(L__self___user_model_feedforwards_item_clk_seq_fc1_linear_weight, (176, 176), (1, 176), 0), out=buf19)
        buf20 = reinterpret_tensor(buf19, (s0, 50, 176), (8800, 176, 1), 0); del buf19  # reuse
        # Source Nodes: [leaky_relu], Original ATen: [aten.leaky_relu]
        triton_poi_fused_leaky_relu_5_xnumel = 8800*s0
        triton_poi_fused_leaky_relu_5.run(buf20, L__self___user_model_feedforwards_item_clk_seq_fc1_linear_bias, triton_poi_fused_leaky_relu_5_xnumel, grid=grid(triton_poi_fused_leaky_relu_5_xnumel), stream=stream0)
        buf21 = empty_strided_cuda((50*s0, 176), (176, 1), torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (50*s0, 176), (176, 1), 0), reinterpret_tensor(L__self___user_model_feedforwards_item_clk_seq_fc2_linear_weight, (176, 176), (1, 176), 0), out=buf21)
        del buf20
        buf56 = empty_strided_cuda((s0, 1872), (1872, 1), torch.float32)
        buf51 = reinterpret_tensor(buf56, (s0, 176), (1872, 1), 1112)  # alias
        # Source Nodes: [mul, pow_1, sum_4], Original ATen: [aten.mul, aten.pow, aten.sum]
        triton_per_fused_mul_pow_sum_6_xnumel = 176*s0
        triton_per_fused_mul_pow_sum_6.run(arg124_1, buf18, buf21, L__self___user_model_feedforwards_item_clk_seq_fc2_linear_bias, buf4, buf23, buf51, triton_per_fused_mul_pow_sum_6_xnumel, 50, grid=grid(triton_per_fused_mul_pow_sum_6_xnumel), stream=stream0)
        del arg124_1
        del buf21
        buf25 = empty_strided_cuda((s0, 8, 16, 50), (6400, 800, 50, 1), torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_7_ynumel = 128*s0
        triton_poi_fused_clone_7.run(buf5, buf6, buf7, L__self___user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm_weight, L__self___user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm_bias, buf25, triton_poi_fused_clone_7_ynumel, 50, grid=grid(triton_poi_fused_clone_7_ynumel, 50), stream=stream0)
        buf26 = reinterpret_tensor(buf7, (8*s0, 1, 50), (50, 50, 1), 0); del buf7  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf24, (8*s0, 1, 16), (16, 0, 1), 0), reinterpret_tensor(buf25, (8*s0, 16, 50), (800, 50, 1), 0), out=buf26)
        buf30 = reinterpret_tensor(buf6, (s0, 8, 1, 50), (400, 50, 50, 1), 0); del buf6  # reuse
        # Source Nodes: [softmax, truediv], Original ATen: [aten._softmax, aten.div]
        triton_per_fused__softmax_div_8_xnumel = 8*s0
        triton_per_fused__softmax_div_8.run(buf26, buf30, triton_per_fused__softmax_div_8_xnumel, 50, grid=grid(triton_per_fused__softmax_div_8_xnumel), stream=stream0)
        del buf26
        buf29 = reinterpret_tensor(buf25, (50*s0, 128), (128, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf4, (50*s0, 176), (176, 1), 0), reinterpret_tensor(L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_v_linear_weight, (176, 128), (1, 176), 0), out=buf29)
        del buf4
        buf31 = reinterpret_tensor(buf5, (s0, 8, 50, 16), (6400, 800, 16, 1), 0); del buf5  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 6400*s0
        triton_poi_fused_clone_9.run(buf29, L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_v_linear_bias, buf31, triton_poi_fused_clone_9_xnumel, grid=grid(triton_poi_fused_clone_9_xnumel), stream=stream0)
        del buf29
        buf32 = reinterpret_tensor(buf24, (8*s0, 1, 16), (16, 16, 1), 0); del buf24  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf30, (8*s0, 1, 50), (50, 0, 1), 0), reinterpret_tensor(buf31, (8*s0, 50, 16), (800, 16, 1), 0), out=buf32)
        del buf30
        buf33 = buf23; del buf23  # reuse
        buf38 = buf18; del buf18  # reuse
        # Source Nodes: [logical_or_1, lt_1, sum_2, sum_5, to_1, to_3], Original ATen: [aten._to_copy, aten.logical_or, aten.lt, aten.sum]
        triton_per_fused__to_copy_logical_or_lt_sum_4.run(arg127_1, buf33, buf38, s0, 50, grid=grid(s0), stream=stream0)
        buf34 = empty_strided_cuda((50*s0, 172), (172, 1), torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf13, (50*s0, 172), (172, 1), 0), reinterpret_tensor(L__self___user_model_feedforwards_query_seq_fc1_linear_weight, (172, 172), (1, 172), 0), out=buf34)
        buf35 = reinterpret_tensor(buf34, (s0, 50, 172), (8600, 172, 1), 0); del buf34  # reuse
        # Source Nodes: [leaky_relu_1], Original ATen: [aten.leaky_relu]
        triton_poi_fused_leaky_relu_10_xnumel = 8600*s0
        triton_poi_fused_leaky_relu_10.run(buf35, L__self___user_model_feedforwards_query_seq_fc1_linear_bias, triton_poi_fused_leaky_relu_10_xnumel, grid=grid(triton_poi_fused_leaky_relu_10_xnumel), stream=stream0)
        buf36 = empty_strided_cuda((50*s0, 172), (172, 1), torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf35, (50*s0, 172), (172, 1), 0), reinterpret_tensor(L__self___user_model_feedforwards_query_seq_fc2_linear_weight, (172, 172), (1, 172), 0), out=buf36)
        del buf35
        buf53 = reinterpret_tensor(buf56, (s0, 172), (1872, 1), 1416)  # alias
        # Source Nodes: [mul_1, pow_2, sum_6], Original ATen: [aten.mul, aten.pow, aten.sum]
        triton_per_fused_mul_pow_sum_11_xnumel = 172*s0
        triton_per_fused_mul_pow_sum_11.run(arg127_1, buf33, buf36, L__self___user_model_feedforwards_query_seq_fc2_linear_bias, buf13, buf38, buf53, triton_per_fused_mul_pow_sum_11_xnumel, 50, grid=grid(triton_per_fused_mul_pow_sum_11_xnumel), stream=stream0)
        del arg127_1
        del buf36
        buf40 = reinterpret_tensor(buf31, (s0, 8, 16, 50), (6400, 800, 50, 1), 0); del buf31  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_7_ynumel = 128*s0
        triton_poi_fused_clone_7.run(buf14, buf15, buf16, L__self___user_model_multi_h_attens_query_seq_ta_k_layer_norm_weight, L__self___user_model_multi_h_attens_query_seq_ta_k_layer_norm_bias, buf40, triton_poi_fused_clone_7_ynumel, 50, grid=grid(triton_poi_fused_clone_7_ynumel, 50), stream=stream0)
        buf41 = reinterpret_tensor(buf16, (8*s0, 1, 50), (50, 50, 1), 0); del buf16  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf39, (8*s0, 1, 16), (16, 0, 1), 0), reinterpret_tensor(buf40, (8*s0, 16, 50), (800, 50, 1), 0), out=buf41)
        buf45 = reinterpret_tensor(buf15, (s0, 8, 1, 50), (400, 50, 50, 1), 0); del buf15  # reuse
        # Source Nodes: [softmax_1, truediv_1], Original ATen: [aten._softmax, aten.div]
        triton_per_fused__softmax_div_8_xnumel = 8*s0
        triton_per_fused__softmax_div_8.run(buf41, buf45, triton_per_fused__softmax_div_8_xnumel, 50, grid=grid(triton_per_fused__softmax_div_8_xnumel), stream=stream0)
        del buf41
        buf44 = reinterpret_tensor(buf40, (50*s0, 128), (128, 1), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf13, (50*s0, 172), (172, 1), 0), reinterpret_tensor(L__self___user_model_multi_h_attens_query_seq_ta_proj_v_linear_weight, (172, 128), (1, 172), 0), out=buf44)
        del buf13
        buf46 = reinterpret_tensor(buf14, (s0, 8, 50, 16), (6400, 800, 16, 1), 0); del buf14  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_9_xnumel = 6400*s0
        triton_poi_fused_clone_9.run(buf44, L__self___user_model_multi_h_attens_query_seq_ta_proj_v_linear_bias, buf46, triton_poi_fused_clone_9_xnumel, grid=grid(triton_poi_fused_clone_9_xnumel), stream=stream0)
        del buf44
        buf47 = reinterpret_tensor(buf39, (8*s0, 1, 16), (16, 16, 1), 0); del buf39  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf45, (8*s0, 1, 50), (50, 0, 1), 0), reinterpret_tensor(buf46, (8*s0, 50, 16), (800, 16, 1), 0), out=buf47)
        del buf45
        del buf46
        buf48 = reinterpret_tensor(buf56, (s0, 204), (1872, 1), 0)  # alias
        # Source Nodes: [cat], Original ATen: [aten.cat]
        triton_poi_fused_cat_12_xnumel = 204*s0
        triton_poi_fused_cat_12.run(arg130_1, buf48, triton_poi_fused_cat_12_xnumel, grid=grid(triton_poi_fused_cat_12_xnumel), stream=stream0)
        buf49 = reinterpret_tensor(buf56, (s0, 220), (1872, 1), 204)  # alias
        # Source Nodes: [cat], Original ATen: [aten.cat]
        triton_poi_fused_cat_13_xnumel = 220*s0
        triton_poi_fused_cat_13.run(arg125_1, buf49, triton_poi_fused_cat_13_xnumel, grid=grid(triton_poi_fused_cat_13_xnumel), stream=stream0)
        del arg125_1
        buf50 = reinterpret_tensor(buf56, (s0, 688), (1872, 1), 424)  # alias
        # Source Nodes: [cat], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 688*s0
        triton_poi_fused_cat_14.run(arg122_1, buf50, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg122_1
        buf52 = reinterpret_tensor(buf56, (s0, 128), (1872, 1), 1288)  # alias
        # Source Nodes: [cat], Original ATen: [aten.cat]
        triton_poi_fused_cat_15_xnumel = 128*s0
        triton_poi_fused_cat_15.run(buf32, buf52, triton_poi_fused_cat_15_xnumel, grid=grid(triton_poi_fused_cat_15_xnumel), stream=stream0)
        buf54 = reinterpret_tensor(buf56, (s0, 128), (1872, 1), 1588)  # alias
        # Source Nodes: [cat], Original ATen: [aten.cat]
        triton_poi_fused_cat_15_xnumel = 128*s0
        triton_poi_fused_cat_15.run(buf47, buf54, triton_poi_fused_cat_15_xnumel, grid=grid(triton_poi_fused_cat_15_xnumel), stream=stream0)
        buf55 = reinterpret_tensor(buf56, (s0, 156), (1872, 1), 1716)  # alias
        # Source Nodes: [cat], Original ATen: [aten.cat]
        triton_poi_fused_cat_16_xnumel = 156*s0
        triton_poi_fused_cat_16.run(arg128_1, buf55, triton_poi_fused_cat_16_xnumel, grid=grid(triton_poi_fused_cat_16_xnumel), stream=stream0)
        del arg128_1
        del buf48
        del buf49
        del buf50
        del buf52
        del buf54
        del buf55
        buf57 = empty_strided_cuda((s0, 512), (512, 1), torch.float32)
        # Source Nodes: [user_model_ctr_net_tower_layers_0_fc], Original ATen: [aten.mm]
        extern_kernels.mm(buf56, reinterpret_tensor(getattr_L__self___user_model_ctr_net_tower_layers___0___fc_weight, (1872, 512), (1, 1872), 0), out=buf57)
        buf58 = buf57; del buf57  # reuse
        buf59 = buf58; del buf58  # reuse
        # Source Nodes: [user_model_ctr_net_tower_layers_0_act, user_model_ctr_net_tower_layers_0_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel = 512*s0
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17.run(buf59, getattr_L__self___user_model_ctr_net_tower_layers___0___norm_running_mean, getattr_L__self___user_model_ctr_net_tower_layers___0___norm_running_var, getattr_L__self___user_model_ctr_net_tower_layers___0___norm_weight, getattr_L__self___user_model_ctr_net_tower_layers___0___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel), stream=stream0)
        buf60 = empty_strided_cuda((s0, 256), (256, 1), torch.float32)
        # Source Nodes: [user_model_ctr_net_tower_layers_0_act, user_model_ctr_net_tower_layers_1_fc], Original ATen: [aten.leaky_relu, aten.mm]
        extern_kernels.mm(buf59, reinterpret_tensor(getattr_L__self___user_model_ctr_net_tower_layers___1___fc_weight, (512, 256), (1, 512), 0), out=buf60)
        buf61 = buf60; del buf60  # reuse
        buf62 = buf61; del buf61  # reuse
        # Source Nodes: [user_model_ctr_net_tower_layers_1_act, user_model_ctr_net_tower_layers_1_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel = 256*s0
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18.run(buf62, getattr_L__self___user_model_ctr_net_tower_layers___1___norm_running_mean, getattr_L__self___user_model_ctr_net_tower_layers___1___norm_running_var, getattr_L__self___user_model_ctr_net_tower_layers___1___norm_weight, getattr_L__self___user_model_ctr_net_tower_layers___1___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel), stream=stream0)
        buf63 = reinterpret_tensor(buf47, (s0, 128), (128, 1), 0); del buf47  # reuse
        # Source Nodes: [user_model_ctr_net_tower_layers_1_act, user_model_ctr_net_tower_layers_2_fc], Original ATen: [aten.leaky_relu, aten.mm]
        extern_kernels.mm(buf62, reinterpret_tensor(getattr_L__self___user_model_ctr_net_tower_layers___2___fc_weight, (256, 128), (1, 256), 0), out=buf63)
        buf64 = buf63; del buf63  # reuse
        buf71 = buf64; del buf64  # reuse
        # Source Nodes: [user_model_ctr_net_tower_layers_2_act, user_model_ctr_net_tower_layers_2_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel = 128*s0
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19.run(buf71, getattr_L__self___user_model_ctr_net_tower_layers___2___norm_running_mean, getattr_L__self___user_model_ctr_net_tower_layers___2___norm_running_var, getattr_L__self___user_model_ctr_net_tower_layers___2___norm_weight, getattr_L__self___user_model_ctr_net_tower_layers___2___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel), stream=stream0)
        buf65 = empty_strided_cuda((s0, 568), (568, 1), torch.float32)
        # Source Nodes: [cat_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_20_xnumel = 568*s0
        triton_poi_fused_cat_20.run(arg130_1, arg129_1, buf51, buf53, buf65, triton_poi_fused_cat_20_xnumel, grid=grid(triton_poi_fused_cat_20_xnumel), stream=stream0)
        del arg129_1
        del arg130_1
        del buf51
        del buf53
        buf66 = buf62; del buf62  # reuse
        # Source Nodes: [cat_1, user_model_bias_net_tower_layers_0_fc], Original ATen: [aten.cat, aten.mm]
        extern_kernels.mm(buf65, reinterpret_tensor(getattr_L__self___user_model_bias_net_tower_layers___0___fc_weight, (568, 256), (1, 568), 0), out=buf66)
        del buf65
        buf67 = buf66; del buf66  # reuse
        buf68 = buf67; del buf67  # reuse
        # Source Nodes: [user_model_bias_net_tower_layers_0_act, user_model_bias_net_tower_layers_0_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel = 256*s0
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18.run(buf68, getattr_L__self___user_model_bias_net_tower_layers___0___norm_running_mean, getattr_L__self___user_model_bias_net_tower_layers___0___norm_running_var, getattr_L__self___user_model_bias_net_tower_layers___0___norm_weight, getattr_L__self___user_model_bias_net_tower_layers___0___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel), stream=stream0)
        buf69 = reinterpret_tensor(buf32, (s0, 128), (128, 1), 0); del buf32  # reuse
        # Source Nodes: [user_model_bias_net_tower_layers_0_act, user_model_bias_net_tower_layers_1_fc], Original ATen: [aten.leaky_relu, aten.mm]
        extern_kernels.mm(buf68, reinterpret_tensor(getattr_L__self___user_model_bias_net_tower_layers___1___fc_weight, (256, 128), (1, 256), 0), out=buf69)
        buf70 = buf69; del buf69  # reuse
        buf72 = buf70; del buf70  # reuse
        # Source Nodes: [user_model_bias_net_tower_layers_1_act, user_model_bias_net_tower_layers_1_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel = 128*s0
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19.run(buf72, getattr_L__self___user_model_bias_net_tower_layers___1___norm_running_mean, getattr_L__self___user_model_bias_net_tower_layers___1___norm_running_var, getattr_L__self___user_model_bias_net_tower_layers___1___norm_weight, getattr_L__self___user_model_bias_net_tower_layers___1___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel), stream=stream0)
        buf73 = reinterpret_tensor(buf38, (s0, 1), (1, 1), 0); del buf38  # reuse
        # Source Nodes: [add_2, user_model_bias_net_tower_layers_1_act, user_model_ctr_net_tower_layers_2_act], Original ATen: [aten.add, aten.leaky_relu]
        extern_kernels._mm_plus_mm(buf71, reinterpret_tensor(L__self___user_model_ctr_head_weight, (128, 1), (1, 128), 0), buf72, reinterpret_tensor(L__self___user_model_ctr_bias_head_weight, (128, 1), (1, 128), 0), out=buf73)
        buf74 = buf59; del buf59  # reuse
        # Source Nodes: [user_model_click_net_tower_layers_0_fc], Original ATen: [aten.mm]
        extern_kernels.mm(buf56, reinterpret_tensor(getattr_L__self___user_model_click_net_tower_layers___0___fc_weight, (1872, 512), (1, 1872), 0), out=buf74)
        buf75 = buf74; del buf74  # reuse
        buf76 = buf75; del buf75  # reuse
        # Source Nodes: [user_model_click_net_tower_layers_0_act, user_model_click_net_tower_layers_0_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel = 512*s0
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17.run(buf76, getattr_L__self___user_model_click_net_tower_layers___0___norm_running_mean, getattr_L__self___user_model_click_net_tower_layers___0___norm_running_var, getattr_L__self___user_model_click_net_tower_layers___0___norm_weight, getattr_L__self___user_model_click_net_tower_layers___0___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel), stream=stream0)
        buf77 = buf68; del buf68  # reuse
        # Source Nodes: [user_model_click_net_tower_layers_0_act, user_model_click_net_tower_layers_1_fc], Original ATen: [aten.leaky_relu, aten.mm]
        extern_kernels.mm(buf76, reinterpret_tensor(getattr_L__self___user_model_click_net_tower_layers___1___fc_weight, (512, 256), (1, 512), 0), out=buf77)
        buf78 = buf77; del buf77  # reuse
        buf79 = buf78; del buf78  # reuse
        # Source Nodes: [user_model_click_net_tower_layers_1_act, user_model_click_net_tower_layers_1_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel = 256*s0
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18.run(buf79, getattr_L__self___user_model_click_net_tower_layers___1___norm_running_mean, getattr_L__self___user_model_click_net_tower_layers___1___norm_running_var, getattr_L__self___user_model_click_net_tower_layers___1___norm_weight, getattr_L__self___user_model_click_net_tower_layers___1___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel), stream=stream0)
        buf80 = buf71; del buf71  # reuse
        # Source Nodes: [user_model_click_net_tower_layers_1_act, user_model_click_net_tower_layers_2_fc], Original ATen: [aten.leaky_relu, aten.mm]
        extern_kernels.mm(buf79, reinterpret_tensor(getattr_L__self___user_model_click_net_tower_layers___2___fc_weight, (256, 128), (1, 256), 0), out=buf80)
        buf81 = buf80; del buf80  # reuse
        buf82 = buf81; del buf81  # reuse
        # Source Nodes: [user_model_click_net_tower_layers_2_act, user_model_click_net_tower_layers_2_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel = 128*s0
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19.run(buf82, getattr_L__self___user_model_click_net_tower_layers___2___norm_running_mean, getattr_L__self___user_model_click_net_tower_layers___2___norm_running_var, getattr_L__self___user_model_click_net_tower_layers___2___norm_weight, getattr_L__self___user_model_click_net_tower_layers___2___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel), stream=stream0)
        buf83 = reinterpret_tensor(buf33, (s0, 1), (1, 1), 0); del buf33  # reuse
        # Source Nodes: [add_3, user_model_click_net_tower_layers_2_act], Original ATen: [aten.add, aten.leaky_relu]
        extern_kernels._mm_plus_mm(buf82, reinterpret_tensor(L__self___user_model_click_head_weight, (128, 1), (1, 128), 0), buf72, reinterpret_tensor(L__self___user_model_click_bias_head_weight, (128, 1), (1, 128), 0), out=buf83)
        buf84 = buf76; del buf76  # reuse
        # Source Nodes: [user_model_page_net_tower_layers_0_fc], Original ATen: [aten.mm]
        extern_kernels.mm(buf56, reinterpret_tensor(getattr_L__self___user_model_page_net_tower_layers___0___fc_weight, (1872, 512), (1, 1872), 0), out=buf84)
        buf85 = buf84; del buf84  # reuse
        buf86 = buf85; del buf85  # reuse
        # Source Nodes: [user_model_page_net_tower_layers_0_act, user_model_page_net_tower_layers_0_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel = 512*s0
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17.run(buf86, getattr_L__self___user_model_page_net_tower_layers___0___norm_running_mean, getattr_L__self___user_model_page_net_tower_layers___0___norm_running_var, getattr_L__self___user_model_page_net_tower_layers___0___norm_weight, getattr_L__self___user_model_page_net_tower_layers___0___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel), stream=stream0)
        buf87 = buf79; del buf79  # reuse
        # Source Nodes: [user_model_page_net_tower_layers_0_act, user_model_page_net_tower_layers_1_fc], Original ATen: [aten.leaky_relu, aten.mm]
        extern_kernels.mm(buf86, reinterpret_tensor(getattr_L__self___user_model_page_net_tower_layers___1___fc_weight, (512, 256), (1, 512), 0), out=buf87)
        buf88 = buf87; del buf87  # reuse
        buf89 = buf88; del buf88  # reuse
        # Source Nodes: [user_model_page_net_tower_layers_1_act, user_model_page_net_tower_layers_1_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel = 256*s0
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18.run(buf89, getattr_L__self___user_model_page_net_tower_layers___1___norm_running_mean, getattr_L__self___user_model_page_net_tower_layers___1___norm_running_var, getattr_L__self___user_model_page_net_tower_layers___1___norm_weight, getattr_L__self___user_model_page_net_tower_layers___1___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel), stream=stream0)
        buf90 = buf82; del buf82  # reuse
        # Source Nodes: [user_model_page_net_tower_layers_1_act, user_model_page_net_tower_layers_2_fc], Original ATen: [aten.leaky_relu, aten.mm]
        extern_kernels.mm(buf89, reinterpret_tensor(getattr_L__self___user_model_page_net_tower_layers___2___fc_weight, (256, 128), (1, 256), 0), out=buf90)
        buf91 = buf90; del buf90  # reuse
        buf92 = buf91; del buf91  # reuse
        # Source Nodes: [user_model_page_net_tower_layers_2_act, user_model_page_net_tower_layers_2_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel = 128*s0
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19.run(buf92, getattr_L__self___user_model_page_net_tower_layers___2___norm_running_mean, getattr_L__self___user_model_page_net_tower_layers___2___norm_running_var, getattr_L__self___user_model_page_net_tower_layers___2___norm_weight, getattr_L__self___user_model_page_net_tower_layers___2___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel), stream=stream0)
        buf93 = empty_strided_cuda((s0, 1), (1, 1), torch.float32)
        # Source Nodes: [add_4, user_model_page_net_tower_layers_2_act], Original ATen: [aten.add, aten.leaky_relu]
        extern_kernels._mm_plus_mm(buf92, reinterpret_tensor(L__self___user_model_page_head_weight, (128, 1), (1, 128), 0), buf72, reinterpret_tensor(L__self___user_model_page_bias_head_weight, (128, 1), (1, 128), 0), out=buf93)
        buf94 = buf86; del buf86  # reuse
        # Source Nodes: [user_model_pay_net_tower_layers_0_fc], Original ATen: [aten.mm]
        extern_kernels.mm(buf56, reinterpret_tensor(getattr_L__self___user_model_pay_net_tower_layers___0___fc_weight, (1872, 512), (1, 1872), 0), out=buf94)
        del buf56
        buf95 = buf94; del buf94  # reuse
        buf96 = buf95; del buf95  # reuse
        # Source Nodes: [user_model_pay_net_tower_layers_0_act, user_model_pay_net_tower_layers_0_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel = 512*s0
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17.run(buf96, getattr_L__self___user_model_pay_net_tower_layers___0___norm_running_mean, getattr_L__self___user_model_pay_net_tower_layers___0___norm_running_var, getattr_L__self___user_model_pay_net_tower_layers___0___norm_weight, getattr_L__self___user_model_pay_net_tower_layers___0___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel), stream=stream0)
        buf97 = buf89; del buf89  # reuse
        # Source Nodes: [user_model_pay_net_tower_layers_0_act, user_model_pay_net_tower_layers_1_fc], Original ATen: [aten.leaky_relu, aten.mm]
        extern_kernels.mm(buf96, reinterpret_tensor(getattr_L__self___user_model_pay_net_tower_layers___1___fc_weight, (512, 256), (1, 512), 0), out=buf97)
        del buf96
        buf98 = buf97; del buf97  # reuse
        buf99 = buf98; del buf98  # reuse
        # Source Nodes: [user_model_pay_net_tower_layers_1_act, user_model_pay_net_tower_layers_1_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel = 256*s0
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18.run(buf99, getattr_L__self___user_model_pay_net_tower_layers___1___norm_running_mean, getattr_L__self___user_model_pay_net_tower_layers___1___norm_running_var, getattr_L__self___user_model_pay_net_tower_layers___1___norm_weight, getattr_L__self___user_model_pay_net_tower_layers___1___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel), stream=stream0)
        buf100 = buf92; del buf92  # reuse
        # Source Nodes: [user_model_pay_net_tower_layers_1_act, user_model_pay_net_tower_layers_2_fc], Original ATen: [aten.leaky_relu, aten.mm]
        extern_kernels.mm(buf99, reinterpret_tensor(getattr_L__self___user_model_pay_net_tower_layers___2___fc_weight, (256, 128), (1, 256), 0), out=buf100)
        del buf99
        buf101 = buf100; del buf100  # reuse
        buf102 = buf101; del buf101  # reuse
        # Source Nodes: [user_model_pay_net_tower_layers_2_act, user_model_pay_net_tower_layers_2_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel = 128*s0
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19.run(buf102, getattr_L__self___user_model_pay_net_tower_layers___2___norm_running_mean, getattr_L__self___user_model_pay_net_tower_layers___2___norm_running_var, getattr_L__self___user_model_pay_net_tower_layers___2___norm_weight, getattr_L__self___user_model_pay_net_tower_layers___2___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel), stream=stream0)
        buf103 = empty_strided_cuda((s0, 1), (1, 1), torch.float32)
        # Source Nodes: [add_5, user_model_pay_net_tower_layers_2_act], Original ATen: [aten.add, aten.leaky_relu]
        extern_kernels._mm_plus_mm(buf102, reinterpret_tensor(L__self___user_model_pay_head_weight, (128, 1), (1, 128), 0), buf72, reinterpret_tensor(L__self___user_model_pay_bias_head_weight, (128, 1), (1, 128), 0), out=buf103)
        del buf102
        del buf72

    for kernel in globals().values():
        if isinstance(kernel, torch._inductor.triton_heuristics.CachingAutotuner):
            if not kernel.cuda_kernel_saved:
                if len(kernel.launchers) == 0:
                    kernel.precompile()
                kernel.save_cuda_kernel(
                    grid=(0, 0, 0),   # use dummy grid
                    stream="stream",  # use dummy stream
                    launcher=kernel.launchers[0],
                )
    return (buf73, buf83, buf93, buf103, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    global L__self___user_model_feedforwards_item_clk_seq_fc1_linear_weight
    L__self___user_model_feedforwards_item_clk_seq_fc1_linear_weight = rand_strided((176, 176), (176, 1), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_feedforwards_item_clk_seq_fc1_linear_bias
    L__self___user_model_feedforwards_item_clk_seq_fc1_linear_bias = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_feedforwards_item_clk_seq_fc2_linear_weight
    L__self___user_model_feedforwards_item_clk_seq_fc2_linear_weight = rand_strided((176, 176), (176, 1), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_feedforwards_item_clk_seq_fc2_linear_bias
    L__self___user_model_feedforwards_item_clk_seq_fc2_linear_bias = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_feedforwards_query_seq_fc1_linear_weight
    L__self___user_model_feedforwards_query_seq_fc1_linear_weight = rand_strided((172, 172), (172, 1), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_feedforwards_query_seq_fc1_linear_bias
    L__self___user_model_feedforwards_query_seq_fc1_linear_bias = rand_strided((172, ), (1, ), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_feedforwards_query_seq_fc2_linear_weight
    L__self___user_model_feedforwards_query_seq_fc2_linear_weight = rand_strided((172, 172), (172, 1), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_feedforwards_query_seq_fc2_linear_bias
    L__self___user_model_feedforwards_query_seq_fc2_linear_bias = rand_strided((172, ), (1, ), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_q_linear_weight
    L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_q_linear_weight = rand_strided((128, 156), (156, 1), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_q_linear_bias
    L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_q_linear_bias = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_k_linear_weight
    L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_k_linear_weight = rand_strided((128, 176), (176, 1), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_k_linear_bias
    L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_k_linear_bias = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_v_linear_weight
    L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_v_linear_weight = rand_strided((128, 176), (176, 1), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_v_linear_bias
    L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_v_linear_bias = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm_weight
    L__self___user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm_weight = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm_bias
    L__self___user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm_bias = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm_weight
    L__self___user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm_weight = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm_bias
    L__self___user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm_bias = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_query_seq_ta_proj_q_linear_weight
    L__self___user_model_multi_h_attens_query_seq_ta_proj_q_linear_weight = rand_strided((128, 156), (156, 1), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_query_seq_ta_proj_q_linear_bias
    L__self___user_model_multi_h_attens_query_seq_ta_proj_q_linear_bias = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_query_seq_ta_proj_k_linear_weight
    L__self___user_model_multi_h_attens_query_seq_ta_proj_k_linear_weight = rand_strided((128, 172), (172, 1), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_query_seq_ta_proj_k_linear_bias
    L__self___user_model_multi_h_attens_query_seq_ta_proj_k_linear_bias = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_query_seq_ta_proj_v_linear_weight
    L__self___user_model_multi_h_attens_query_seq_ta_proj_v_linear_weight = rand_strided((128, 172), (172, 1), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_query_seq_ta_proj_v_linear_bias
    L__self___user_model_multi_h_attens_query_seq_ta_proj_v_linear_bias = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_query_seq_ta_q_layer_norm_weight
    L__self___user_model_multi_h_attens_query_seq_ta_q_layer_norm_weight = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_query_seq_ta_q_layer_norm_bias
    L__self___user_model_multi_h_attens_query_seq_ta_q_layer_norm_bias = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_query_seq_ta_k_layer_norm_weight
    L__self___user_model_multi_h_attens_query_seq_ta_k_layer_norm_weight = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_multi_h_attens_query_seq_ta_k_layer_norm_bias
    L__self___user_model_multi_h_attens_query_seq_ta_k_layer_norm_bias = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_ctr_net_tower_layers___0___fc_weight
    getattr_L__self___user_model_ctr_net_tower_layers___0___fc_weight = rand_strided((512, 1872), (1872, 1), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_ctr_net_tower_layers___0___norm_weight
    getattr_L__self___user_model_ctr_net_tower_layers___0___norm_weight = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_ctr_net_tower_layers___0___norm_bias
    getattr_L__self___user_model_ctr_net_tower_layers___0___norm_bias = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_ctr_net_tower_layers___1___fc_weight
    getattr_L__self___user_model_ctr_net_tower_layers___1___fc_weight = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_ctr_net_tower_layers___1___norm_weight
    getattr_L__self___user_model_ctr_net_tower_layers___1___norm_weight = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_ctr_net_tower_layers___1___norm_bias
    getattr_L__self___user_model_ctr_net_tower_layers___1___norm_bias = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_ctr_net_tower_layers___2___fc_weight
    getattr_L__self___user_model_ctr_net_tower_layers___2___fc_weight = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_ctr_net_tower_layers___2___norm_weight
    getattr_L__self___user_model_ctr_net_tower_layers___2___norm_weight = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_ctr_net_tower_layers___2___norm_bias
    getattr_L__self___user_model_ctr_net_tower_layers___2___norm_bias = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_click_net_tower_layers___0___fc_weight
    getattr_L__self___user_model_click_net_tower_layers___0___fc_weight = rand_strided((512, 1872), (1872, 1), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_click_net_tower_layers___0___norm_weight
    getattr_L__self___user_model_click_net_tower_layers___0___norm_weight = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_click_net_tower_layers___0___norm_bias
    getattr_L__self___user_model_click_net_tower_layers___0___norm_bias = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_click_net_tower_layers___1___fc_weight
    getattr_L__self___user_model_click_net_tower_layers___1___fc_weight = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_click_net_tower_layers___1___norm_weight
    getattr_L__self___user_model_click_net_tower_layers___1___norm_weight = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_click_net_tower_layers___1___norm_bias
    getattr_L__self___user_model_click_net_tower_layers___1___norm_bias = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_click_net_tower_layers___2___fc_weight
    getattr_L__self___user_model_click_net_tower_layers___2___fc_weight = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_click_net_tower_layers___2___norm_weight
    getattr_L__self___user_model_click_net_tower_layers___2___norm_weight = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_click_net_tower_layers___2___norm_bias
    getattr_L__self___user_model_click_net_tower_layers___2___norm_bias = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_page_net_tower_layers___0___fc_weight
    getattr_L__self___user_model_page_net_tower_layers___0___fc_weight = rand_strided((512, 1872), (1872, 1), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_page_net_tower_layers___0___norm_weight
    getattr_L__self___user_model_page_net_tower_layers___0___norm_weight = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_page_net_tower_layers___0___norm_bias
    getattr_L__self___user_model_page_net_tower_layers___0___norm_bias = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_page_net_tower_layers___1___fc_weight
    getattr_L__self___user_model_page_net_tower_layers___1___fc_weight = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_page_net_tower_layers___1___norm_weight
    getattr_L__self___user_model_page_net_tower_layers___1___norm_weight = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_page_net_tower_layers___1___norm_bias
    getattr_L__self___user_model_page_net_tower_layers___1___norm_bias = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_page_net_tower_layers___2___fc_weight
    getattr_L__self___user_model_page_net_tower_layers___2___fc_weight = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_page_net_tower_layers___2___norm_weight
    getattr_L__self___user_model_page_net_tower_layers___2___norm_weight = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_page_net_tower_layers___2___norm_bias
    getattr_L__self___user_model_page_net_tower_layers___2___norm_bias = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_pay_net_tower_layers___0___fc_weight
    getattr_L__self___user_model_pay_net_tower_layers___0___fc_weight = rand_strided((512, 1872), (1872, 1), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_pay_net_tower_layers___0___norm_weight
    getattr_L__self___user_model_pay_net_tower_layers___0___norm_weight = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_pay_net_tower_layers___0___norm_bias
    getattr_L__self___user_model_pay_net_tower_layers___0___norm_bias = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_pay_net_tower_layers___1___fc_weight
    getattr_L__self___user_model_pay_net_tower_layers___1___fc_weight = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_pay_net_tower_layers___1___norm_weight
    getattr_L__self___user_model_pay_net_tower_layers___1___norm_weight = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_pay_net_tower_layers___1___norm_bias
    getattr_L__self___user_model_pay_net_tower_layers___1___norm_bias = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_pay_net_tower_layers___2___fc_weight
    getattr_L__self___user_model_pay_net_tower_layers___2___fc_weight = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_pay_net_tower_layers___2___norm_weight
    getattr_L__self___user_model_pay_net_tower_layers___2___norm_weight = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_pay_net_tower_layers___2___norm_bias
    getattr_L__self___user_model_pay_net_tower_layers___2___norm_bias = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_bias_net_tower_layers___0___fc_weight
    getattr_L__self___user_model_bias_net_tower_layers___0___fc_weight = rand_strided((256, 568), (568, 1), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_bias_net_tower_layers___0___norm_weight
    getattr_L__self___user_model_bias_net_tower_layers___0___norm_weight = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_bias_net_tower_layers___0___norm_bias
    getattr_L__self___user_model_bias_net_tower_layers___0___norm_bias = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_bias_net_tower_layers___1___fc_weight
    getattr_L__self___user_model_bias_net_tower_layers___1___fc_weight = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_bias_net_tower_layers___1___norm_weight
    getattr_L__self___user_model_bias_net_tower_layers___1___norm_weight = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_bias_net_tower_layers___1___norm_bias
    getattr_L__self___user_model_bias_net_tower_layers___1___norm_bias = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_ctr_head_weight
    L__self___user_model_ctr_head_weight = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_click_head_weight
    L__self___user_model_click_head_weight = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_page_head_weight
    L__self___user_model_page_head_weight = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_pay_head_weight
    L__self___user_model_pay_head_weight = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_ctr_bias_head_weight
    L__self___user_model_ctr_bias_head_weight = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_click_bias_head_weight
    L__self___user_model_click_bias_head_weight = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_page_bias_head_weight
    L__self___user_model_page_bias_head_weight = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    global L__self___user_model_pay_bias_head_weight
    L__self___user_model_pay_bias_head_weight = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_ctr_net_tower_layers___0___norm_running_mean
    getattr_L__self___user_model_ctr_net_tower_layers___0___norm_running_mean = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_ctr_net_tower_layers___0___norm_running_var
    getattr_L__self___user_model_ctr_net_tower_layers___0___norm_running_var = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_ctr_net_tower_layers___1___norm_running_mean
    getattr_L__self___user_model_ctr_net_tower_layers___1___norm_running_mean = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_ctr_net_tower_layers___1___norm_running_var
    getattr_L__self___user_model_ctr_net_tower_layers___1___norm_running_var = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_ctr_net_tower_layers___2___norm_running_mean
    getattr_L__self___user_model_ctr_net_tower_layers___2___norm_running_mean = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_ctr_net_tower_layers___2___norm_running_var
    getattr_L__self___user_model_ctr_net_tower_layers___2___norm_running_var = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_click_net_tower_layers___0___norm_running_mean
    getattr_L__self___user_model_click_net_tower_layers___0___norm_running_mean = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_click_net_tower_layers___0___norm_running_var
    getattr_L__self___user_model_click_net_tower_layers___0___norm_running_var = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_click_net_tower_layers___1___norm_running_mean
    getattr_L__self___user_model_click_net_tower_layers___1___norm_running_mean = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_click_net_tower_layers___1___norm_running_var
    getattr_L__self___user_model_click_net_tower_layers___1___norm_running_var = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_click_net_tower_layers___2___norm_running_mean
    getattr_L__self___user_model_click_net_tower_layers___2___norm_running_mean = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_click_net_tower_layers___2___norm_running_var
    getattr_L__self___user_model_click_net_tower_layers___2___norm_running_var = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_page_net_tower_layers___0___norm_running_mean
    getattr_L__self___user_model_page_net_tower_layers___0___norm_running_mean = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_page_net_tower_layers___0___norm_running_var
    getattr_L__self___user_model_page_net_tower_layers___0___norm_running_var = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_page_net_tower_layers___1___norm_running_mean
    getattr_L__self___user_model_page_net_tower_layers___1___norm_running_mean = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_page_net_tower_layers___1___norm_running_var
    getattr_L__self___user_model_page_net_tower_layers___1___norm_running_var = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_page_net_tower_layers___2___norm_running_mean
    getattr_L__self___user_model_page_net_tower_layers___2___norm_running_mean = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_page_net_tower_layers___2___norm_running_var
    getattr_L__self___user_model_page_net_tower_layers___2___norm_running_var = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_pay_net_tower_layers___0___norm_running_mean
    getattr_L__self___user_model_pay_net_tower_layers___0___norm_running_mean = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_pay_net_tower_layers___0___norm_running_var
    getattr_L__self___user_model_pay_net_tower_layers___0___norm_running_var = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_pay_net_tower_layers___1___norm_running_mean
    getattr_L__self___user_model_pay_net_tower_layers___1___norm_running_mean = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_pay_net_tower_layers___1___norm_running_var
    getattr_L__self___user_model_pay_net_tower_layers___1___norm_running_var = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_pay_net_tower_layers___2___norm_running_mean
    getattr_L__self___user_model_pay_net_tower_layers___2___norm_running_mean = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_pay_net_tower_layers___2___norm_running_var
    getattr_L__self___user_model_pay_net_tower_layers___2___norm_running_var = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_bias_net_tower_layers___0___norm_running_mean
    getattr_L__self___user_model_bias_net_tower_layers___0___norm_running_mean = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_bias_net_tower_layers___0___norm_running_var
    getattr_L__self___user_model_bias_net_tower_layers___0___norm_running_var = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_bias_net_tower_layers___1___norm_running_mean
    getattr_L__self___user_model_bias_net_tower_layers___1___norm_running_mean = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    global getattr_L__self___user_model_bias_net_tower_layers___1___norm_running_var
    getattr_L__self___user_model_bias_net_tower_layers___1___norm_running_var = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((321, 688), (688, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((321, 50, 176), (8800, 176, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((321, 1), (1, 1), device='cuda:0', dtype=torch.float64)
    arg125_1 = rand_strided((321, 220), (220, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((321, 50, 172), (8600, 172, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((321, 1), (1, 1), device='cuda:0', dtype=torch.float64)
    arg128_1 = rand_strided((321, 156), (156, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((321, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((321, 204), (204, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
