from ctypes import c_void_p, c_long, c_int
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
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, grid_combo_kernels, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch._inductor.kernel.bmm

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/fm/cfmj2rybzejtqyrqokrlv3v7bve5q4uqz4q3d4wvonjbrjmqswp5.py
# Topologically Sorted Source Nodes: [user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm => add_253, add_254, mul_187, mul_188, rsqrt, sub_81, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_28, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_28, %getitem_1), kwargs = {})
#   %add_253 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_253,), kwargs = {})
#   %mul_187 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %rsqrt), kwargs = {})
#   %mul_188 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_187, %l__self___user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm_weight), kwargs = {})
#   %add_254 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_188, %l__self___user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm_bias), kwargs = {})
triton_per_fused_native_layer_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True}
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
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 16, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
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
    tl.store(out_ptr2 + (r1 + (16*x0)), tmp27, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/4k/c4klftg74xd22dfcegyz4qzniaxfxdul5ismk3ac6mjhtv3j27rh.py
# Topologically Sorted Source Nodes: [where], Original ATen: [aten.scalar_tensor, aten.where]
# Source node to ATen node mapping:
#   where => full_default_1, where
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%view_2, %view_1, %full_default_1), kwargs = {})
triton_poi_fused_scalar_tensor_where_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_scalar_tensor_where_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 176)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((x1 // 50)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp1 = tl.full([1], 50.0, tl.float64)
    tmp2 = triton_helpers.minimum(tmp0, tmp1)
    tmp3 = (x2 // 176) % 50
    tmp4 = tmp3.to(tl.float64)
    tmp5 = tmp4 < tmp2
    tmp7 = 0.0
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/lx/clxhixd53bmknv4ofrzqvvhxd3bervjfc6pbihja35ckcexixua6.py
# Topologically Sorted Source Nodes: [user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm => var_mean_1
# Graph fragment:
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_29, [3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_layer_norm_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[131072, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 4, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True}
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
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 16, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/pa/cpafh3iopr22xn2arqldkphlfi47ai4yloiifdx4ptxxfrm7xer6.py
# Topologically Sorted Source Nodes: [where_1], Original ATen: [aten.scalar_tensor, aten.where]
# Source node to ATen node mapping:
#   where_1 => full_default_3, where_1
# Graph fragment:
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%view_6, %view_5, %full_default_3), kwargs = {})
triton_poi_fused_scalar_tensor_where_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_scalar_tensor_where_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 172)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((x1 // 50)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp1 = tl.full([1], 50.0, tl.float64)
    tmp2 = triton_helpers.minimum(tmp0, tmp1)
    tmp3 = (x2 // 172) % 50
    tmp4 = tmp3.to(tl.float64)
    tmp5 = tmp4 < tmp2
    tmp7 = 0.0
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/oy/coyjxip4s6xvtcg2t65ph3jndq756xmifu6l3oxu7y5qyksgxmln.py
# Topologically Sorted Source Nodes: [lt, to, sum_1, eq, tile, logical_or, to_2, sum_3], Original ATen: [aten.lt, aten._to_copy, aten.sum, aten.eq, aten.repeat, aten.logical_or]
# Source node to ATen node mapping:
#   eq => eq_16
#   logical_or => logical_or
#   lt => lt
#   sum_1 => sum_1
#   sum_3 => sum_3
#   tile => repeat
#   to => convert_element_type
#   to_2 => convert_element_type_2
# Graph fragment:
#   %lt : [num_users=3] = call_function[target=torch.ops.aten.lt.Tensor](args = (%expand, %expand_1), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%lt, torch.float32), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type, [1], True), kwargs = {})
#   %eq_16 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%sum_1, 0), kwargs = {})
#   %repeat : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%eq_16, [1, 50]), kwargs = {})
#   %logical_or : [num_users=2] = call_function[target=torch.ops.aten.logical_or.default](args = (%lt, %repeat), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%logical_or, torch.float32), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_2, [1], True), kwargs = {})
triton_per_fused__to_copy_eq_logical_or_lt_repeat_sum_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp64', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_eq_logical_or_lt_repeat_sum_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True}
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
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 50.0, tl.float64)
    tmp2 = triton_helpers.minimum(tmp0, tmp1)
    tmp3 = r1
    tmp4 = tmp3.to(tl.float64)
    tmp5 = tmp4 < tmp2
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


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/vt/cvtehqg5j7mmiffkvcqk4ch6apbkg2sp54zgmznaw4nlxt7wb2tb.py
# Topologically Sorted Source Nodes: [leaky_relu], Original ATen: [aten.leaky_relu]
# Source node to ATen node mapping:
#   leaky_relu => gt, mul_58, where_2
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_9, 0), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_9, 0.01), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %view_9, %mul_58), kwargs = {})
triton_poi_fused_leaky_relu_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
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


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/i2/ci2xsuz5hmbhdp6op2nnf4sykpv2zmuz6hsmprxn4ecqa27pj3gh.py
# Topologically Sorted Source Nodes: [sum_4, tile_2, pow_1, mul], Original ATen: [aten.sum, aten.repeat, aten.pow, aten.mul]
# Source node to ATen node mapping:
#   mul => mul_95
#   pow_1 => pow_1
#   sum_4 => sum_4
#   tile_2 => repeat_2
# Graph fragment:
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_14, [1]), kwargs = {})
#   %repeat_2 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%sum_3, [1, 176]), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%repeat_2, -1), kwargs = {})
#   %mul_95 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_4, %pow_1), kwargs = {})
triton_per_fused_mul_pow_repeat_sum_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_pow_repeat_sum_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True}
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
    x1 = (xindex // 176)
    r2 = rindex
    x0 = xindex % 176
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x0 + (176*r2) + (8800*x1)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0 + (176*r2) + (8800*x1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 50.0, tl.float64)
    tmp2 = triton_helpers.minimum(tmp0, tmp1)
    tmp3 = r2
    tmp4 = tmp3.to(tl.float64)
    tmp5 = tmp4 < tmp2
    tmp7 = 0.0
    tmp8 = tmp6 == tmp7
    tmp9 = tmp5 | tmp8
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp9, tmp14, tmp7)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp21 = tl.full([1, 1], 1, tl.int32)
    tmp22 = tmp21 / tmp20
    tmp23 = tmp19 * tmp22
    tl.store(out_ptr1 + (x0 + (1872*x1)), tmp23, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/ak/cak5b7ifbyysgpn6c4plcw4cu6dwygnrwa7semwtfe7p7vo3ijno.py
# Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 50
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
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


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/hx/chxchkplnzg7hsc46onnfcqm6yzunklrcmvmjkqecv4st64w5wsb.py
# Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   matmul => bmm
# Graph fragment:
#   %bmm : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_31, %view_32), kwargs = {})
triton_tem_fused_bmm_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=5,
    num_warps=4,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_bmm_8', 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
)
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16
    A = arg_A
    B = arg_B

    M = 1
    N = 50
    K = 16

    stride_aq = 16
    stride_am = 0
    stride_ak = 1

    stride_bq = 800
    stride_bk = 50
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (50*idx_m) + (50*idx_q)
    tl.store(out_ptr0 + (tl.broadcast_to(idx_n + (50*idx_q), acc.shape)), acc, mask)
''', device_str='cuda')
meta0 = {'GROUP_M': 8, 'EVEN_K': True, 'ALLOW_TF32': False, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 16}


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/ji/cji6genlqlzfsvy7mazgz5dxtagx7gskl2ny2a5wxmn3a4b7nbin.py
# Topologically Sorted Source Nodes: [softmax], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   softmax => div_1, exp, sum_7
# Graph fragment:
#   %mul_tensor_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_33, 1), kwargs = {})
#   %amax_default_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_1, [-1], True), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_1, %amax_default_1), kwargs = {})
#   %div_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor_1, 4.0), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor_1,), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_7), kwargs = {})
triton_per_fused__softmax_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True}
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
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = 0.25
    tmp9 = tmp7 * tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r1 + (50*x0)), tmp15, rmask & xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/d5/cd55pdmtorhezi5mv7veyzpcewxicut3wuipam43bwjl5qib5br6.py
# Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_1 => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_7,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
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


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/ot/cotxopk6uk7guirjrna2nfny4ffqcc74hi7uztz2fxntzor5jqhs.py
# Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   matmul_1 => bmm_1
# Graph fragment:
#   %bmm_1 : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_34, %view_35), kwargs = {})
triton_tem_fused_bmm_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=4,
    num_warps=1,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_bmm_11', 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
)
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 32
    A = arg_A
    B = arg_B

    M = 1
    N = 16
    K = 50

    stride_aq = 50
    stride_am = 0
    stride_ak = 1

    stride_bq = 800
    stride_bk = 16
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (16*idx_m) + (16*idx_q)
    tl.store(out_ptr0 + (tl.broadcast_to(idx_n + (16*idx_q), acc.shape)), acc, mask)
''', device_str='cuda')
meta1 = {'GROUP_M': 8, 'EVEN_K': False, 'ALLOW_TF32': False, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 32}


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/3p/c3przjdjlik7lmyll44jybubjecan53xx6p2uhtfargn4vpvgsl7.py
# Topologically Sorted Source Nodes: [leaky_relu_1], Original ATen: [aten.leaky_relu]
# Source node to ATen node mapping:
#   leaky_relu_1 => gt_1, mul_108, where_4
# Graph fragment:
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_16, 0), kwargs = {})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_16, 0.01), kwargs = {})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %view_16, %mul_108), kwargs = {})
triton_poi_fused_leaky_relu_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
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


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/r2/cr2bzfsu7zlnmu74s6vf5fnysuwuhxgpcom7tsaz4267anppe7np.py
# Topologically Sorted Source Nodes: [sum_6, tile_3, pow_2, mul_1], Original ATen: [aten.sum, aten.repeat, aten.pow, aten.mul]
# Source node to ATen node mapping:
#   mul_1 => mul_145
#   pow_2 => pow_2
#   sum_6 => sum_6
#   tile_3 => repeat_3
# Graph fragment:
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_21, [1]), kwargs = {})
#   %repeat_3 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%sum_5, [1, 172]), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%repeat_3, -1), kwargs = {})
#   %mul_145 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_6, %pow_2), kwargs = {})
triton_per_fused_mul_pow_repeat_sum_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_pow_repeat_sum_13', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True}
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
    x1 = (xindex // 172)
    r2 = rindex
    x0 = xindex % 172
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x0 + (172*r2) + (8600*x1)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0 + (172*r2) + (8600*x1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 50.0, tl.float64)
    tmp2 = triton_helpers.minimum(tmp0, tmp1)
    tmp3 = r2
    tmp4 = tmp3.to(tl.float64)
    tmp5 = tmp4 < tmp2
    tmp7 = 0.0
    tmp8 = tmp6 == tmp7
    tmp9 = tmp5 | tmp8
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp9, tmp14, tmp7)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp21 = tl.full([1, 1], 1, tl.int32)
    tmp22 = tmp21 / tmp20
    tmp23 = tmp19 * tmp22
    tl.store(out_ptr1 + (x0 + (1872*x1)), tmp23, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/ig/cigrm7gpd3gsz3kswi7nvyvqriz76snzpm2mleawnhj4dbriofvr.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=4] = call_function[target=torch.ops.aten.cat.default](args = ([%view_56, %view_57, %view_58, %view_59, %view_60, %view_61, %view_62, %view_63], 1), kwargs = {})
triton_poi_fused_cat_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_14', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
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


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/qi/cqiwarv76v36pxm4m4casilxrt2hjladpat5tizmhz6erjqghw6f.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=4] = call_function[target=torch.ops.aten.cat.default](args = ([%view_56, %view_57, %view_58, %view_59, %view_60, %view_61, %view_62, %view_63], 1), kwargs = {})
triton_poi_fused_cat_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_15', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
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


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/p6/cp6by25mg5c4a7momxhwymgdcody5aigysc2bakgsxi2nef4n2g6.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=4] = call_function[target=torch.ops.aten.cat.default](args = ([%view_56, %view_57, %view_58, %view_59, %view_60, %view_61, %view_62, %view_63], 1), kwargs = {})
triton_poi_fused_cat_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_16', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
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


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/e4/ce4cbztwdxd6uuipvtwqzhlzuhhmsnjwgewabe4yjherij3fsjd3.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=4] = call_function[target=torch.ops.aten.cat.default](args = ([%view_56, %view_57, %view_58, %view_59, %view_60, %view_61, %view_62, %view_63], 1), kwargs = {})
triton_poi_fused_cat_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_17', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
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


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/6p/c6pisv3oxpachlkp5tciwos7ckihswphewxj2kuz7g4krlgjlgb4.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=4] = call_function[target=torch.ops.aten.cat.default](args = ([%view_56, %view_57, %view_58, %view_59, %view_60, %view_61, %view_62, %view_63], 1), kwargs = {})
triton_poi_fused_cat_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_18', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
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


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/7b/c7bpqehsga6xvg2pcslli4z6siemsg7idp33i3s3xqxmqs4jnpl6.py
# Topologically Sorted Source Nodes: [user_model_ctr_net_tower_layers_0_norm, user_model_ctr_net_tower_layers_0_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   user_model_ctr_net_tower_layers_0_act => gt_2, mul_504, where_6
#   user_model_ctr_net_tower_layers_0_norm => add_623, add_624, mul_499, mul_500, mul_501, reciprocal, sqrt, sub_173
# Graph fragment:
#   %sub_173 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mm, %getattr_l__self___user_model_ctr_net_tower_layers___0___norm_running_mean), kwargs = {})
#   %add_623 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getattr_l__self___user_model_ctr_net_tower_layers___0___norm_running_var, 0.001), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_623,), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul_499 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %mul_500 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_173, %mul_499), kwargs = {})
#   %mul_501 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_500, %getattr_l__self___user_model_ctr_net_tower_layers___0___norm_weight), kwargs = {})
#   %add_624 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_501, %getattr_l__self___user_model_ctr_net_tower_layers___0___norm_bias), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_624, 0), kwargs = {})
#   %mul_504 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_624, 0.01), kwargs = {})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %add_624, %mul_504), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
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
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/lz/clz6bz5fsw7r7et4zmts3kmkh32racid7c7abjvuzxonumbziitm.py
# Topologically Sorted Source Nodes: [user_model_ctr_net_tower_layers_1_norm, user_model_ctr_net_tower_layers_1_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   user_model_ctr_net_tower_layers_1_act => gt_3, mul_515, where_7
#   user_model_ctr_net_tower_layers_1_norm => add_634, add_635, mul_510, mul_511, mul_512, reciprocal_1, sqrt_1, sub_177
# Graph fragment:
#   %sub_177 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mm_1, %getattr_l__self___user_model_ctr_net_tower_layers___1___norm_running_mean), kwargs = {})
#   %add_634 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getattr_l__self___user_model_ctr_net_tower_layers___1___norm_running_var, 0.001), kwargs = {})
#   %sqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_634,), kwargs = {})
#   %reciprocal_1 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_510 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %mul_511 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_177, %mul_510), kwargs = {})
#   %mul_512 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_511, %getattr_l__self___user_model_ctr_net_tower_layers___1___norm_weight), kwargs = {})
#   %add_635 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_512, %getattr_l__self___user_model_ctr_net_tower_layers___1___norm_bias), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_635, 0), kwargs = {})
#   %mul_515 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_635, 0.01), kwargs = {})
#   %where_7 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %add_635, %mul_515), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
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
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/vk/cvkz2rfkazercfxsdgw7snsyk4zwt6nwnrtymvhipavw3iltu34o.py
# Topologically Sorted Source Nodes: [user_model_ctr_net_tower_layers_2_norm, user_model_ctr_net_tower_layers_2_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   user_model_ctr_net_tower_layers_2_act => gt_4, mul_526, where_8
#   user_model_ctr_net_tower_layers_2_norm => add_645, add_646, mul_521, mul_522, mul_523, reciprocal_2, sqrt_2, sub_181
# Graph fragment:
#   %sub_181 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mm_2, %getattr_l__self___user_model_ctr_net_tower_layers___2___norm_running_mean), kwargs = {})
#   %add_645 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getattr_l__self___user_model_ctr_net_tower_layers___2___norm_running_var, 0.001), kwargs = {})
#   %sqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_645,), kwargs = {})
#   %reciprocal_2 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_2,), kwargs = {})
#   %mul_521 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 1), kwargs = {})
#   %mul_522 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_181, %mul_521), kwargs = {})
#   %mul_523 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_522, %getattr_l__self___user_model_ctr_net_tower_layers___2___norm_weight), kwargs = {})
#   %add_646 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_523, %getattr_l__self___user_model_ctr_net_tower_layers___2___norm_bias), kwargs = {})
#   %gt_4 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_646, 0), kwargs = {})
#   %mul_526 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_646, 0.01), kwargs = {})
#   %where_8 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %add_646, %mul_526), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
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
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.01
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, xmask)
''', device_str='cuda')


# kernel path: /home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu_torch260/ql/cqlgpp5cpaskarye2xch272l53vinlaolekwkjuj3okjcfyxzy6b.py
# Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_64, %view_65, %view_66, %view_67], 1), kwargs = {})
triton_poi_fused_cat_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=72, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_22', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '462733BE51DCFE9C4ED20AC6F112342FD094BF18D44EDA526B784174EE96A530', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True},
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
    tmp5 = tl.load(in_ptr0 + ((204*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 220, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + ((16*x1) + ((-204) + x0)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 396, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + ((1872*x1) + ((-220) + x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 568, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + ((1872*x1) + ((-396) + x0)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tl.store(out_ptr0 + (x2), tmp22, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1 = args
    args.clear()
    arg122_1_size = arg122_1.size()
    s10 = arg122_1_size[0]
    assert_size_stride(arg122_1, (s10, 688), (688, 1))
    assert_size_stride(arg123_1, (s10, 50, 176), (8800, 176, 1))
    assert_size_stride(arg124_1, (s10, 1), (1, 1))
    assert_size_stride(arg125_1, (s10, 220), (220, 1))
    assert_size_stride(arg126_1, (s10, 50, 172), (8600, 172, 1))
    assert_size_stride(arg127_1, (s10, 1), (1, 1))
    assert_size_stride(arg128_1, (s10, 156), (156, 1))
    assert_size_stride(arg129_1, (s10, 16), (16, 1))
    assert_size_stride(arg130_1, (s10, 204), (204, 1))

    for kernel in globals().values():
        if isinstance(kernel, torch._inductor.runtime.triton_heuristics.CachingAutotuner):
            kernel.cuda_kernel_saved = False
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((s10, 128), (128, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.addmm(reinterpret_tensor(L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_q_linear_bias, (s10, 128), (0, 1), 0), arg128_1, reinterpret_tensor(L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_q_linear_weight, (156, 128), (1, 156), 0), alpha=1, beta=1, out=buf0)
        buf24 = empty_strided_cuda((s10, 1, 8, 16), (128, 1, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_0_xnumel = 8*s10
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_0.run(buf0, L__self___user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm_weight, L__self___user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm_bias, buf24, triton_per_fused_native_layer_norm_0_xnumel, 16, grid=grid(triton_per_fused_native_layer_norm_0_xnumel), stream=stream0)
        buf4 = empty_strided_cuda((50*s10, 176), (176, 1), torch.float32)
        # Topologically Sorted Source Nodes: [where], Original ATen: [aten.scalar_tensor, aten.where]
        triton_poi_fused_scalar_tensor_where_1_xnumel = 8800*s10
        triton_poi_fused_scalar_tensor_where_1.run(arg124_1, arg123_1, buf4, triton_poi_fused_scalar_tensor_where_1_xnumel, grid=grid(triton_poi_fused_scalar_tensor_where_1_xnumel), stream=stream0)
        del arg123_1
        buf5 = empty_strided_cuda((50*s10, 128), (128, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.addmm(reinterpret_tensor(L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_k_linear_bias, (50*s10, 128), (0, 1), 0), buf4, reinterpret_tensor(L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_k_linear_weight, (176, 128), (1, 176), 0), alpha=1, beta=1, out=buf5)
        buf6 = empty_strided_cuda((s10, 50, 8, 1), (400, 8, 1, 400*s10), torch.float32)
        buf7 = empty_strided_cuda((s10, 50, 8, 1), (400, 8, 1, 400*s10), torch.float32)
        # Topologically Sorted Source Nodes: [user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_2_xnumel = 400*s10
        triton_per_fused_native_layer_norm_2.run(buf5, buf6, buf7, triton_per_fused_native_layer_norm_2_xnumel, 16, grid=grid(triton_per_fused_native_layer_norm_2_xnumel), stream=stream0)
        buf9 = buf0; del buf0  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.addmm(reinterpret_tensor(L__self___user_model_multi_h_attens_query_seq_ta_proj_q_linear_bias, (s10, 128), (0, 1), 0), arg128_1, reinterpret_tensor(L__self___user_model_multi_h_attens_query_seq_ta_proj_q_linear_weight, (156, 128), (1, 156), 0), alpha=1, beta=1, out=buf9)
        buf39 = empty_strided_cuda((s10, 1, 8, 16), (128, 1, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [user_model_multi_h_attens_query_seq_ta_q_layer_norm], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_0_xnumel = 8*s10
        triton_per_fused_native_layer_norm_0.run(buf9, L__self___user_model_multi_h_attens_query_seq_ta_q_layer_norm_weight, L__self___user_model_multi_h_attens_query_seq_ta_q_layer_norm_bias, buf39, triton_per_fused_native_layer_norm_0_xnumel, 16, grid=grid(triton_per_fused_native_layer_norm_0_xnumel), stream=stream0)
        del buf9
        buf13 = empty_strided_cuda((50*s10, 172), (172, 1), torch.float32)
        # Topologically Sorted Source Nodes: [where_1], Original ATen: [aten.scalar_tensor, aten.where]
        triton_poi_fused_scalar_tensor_where_3_xnumel = 8600*s10
        triton_poi_fused_scalar_tensor_where_3.run(arg127_1, arg126_1, buf13, triton_poi_fused_scalar_tensor_where_3_xnumel, grid=grid(triton_poi_fused_scalar_tensor_where_3_xnumel), stream=stream0)
        del arg126_1
        buf14 = empty_strided_cuda((50*s10, 128), (128, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.addmm(reinterpret_tensor(L__self___user_model_multi_h_attens_query_seq_ta_proj_k_linear_bias, (50*s10, 128), (0, 1), 0), buf13, reinterpret_tensor(L__self___user_model_multi_h_attens_query_seq_ta_proj_k_linear_weight, (172, 128), (1, 172), 0), alpha=1, beta=1, out=buf14)
        buf15 = empty_strided_cuda((s10, 50, 8, 1), (400, 8, 1, 400*s10), torch.float32)
        buf16 = empty_strided_cuda((s10, 50, 8, 1), (400, 8, 1, 400*s10), torch.float32)
        # Topologically Sorted Source Nodes: [user_model_multi_h_attens_query_seq_ta_k_layer_norm], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_2_xnumel = 400*s10
        triton_per_fused_native_layer_norm_2.run(buf14, buf15, buf16, triton_per_fused_native_layer_norm_2_xnumel, 16, grid=grid(triton_per_fused_native_layer_norm_2_xnumel), stream=stream0)
        buf18 = empty_strided_cuda((s10, 1), (1, s10), torch.float32)
        buf23 = empty_strided_cuda((s10, 1), (1, s10), torch.float32)
        # Topologically Sorted Source Nodes: [lt, to, sum_1, eq, tile, logical_or, to_2, sum_3], Original ATen: [aten.lt, aten._to_copy, aten.sum, aten.eq, aten.repeat, aten.logical_or]
        triton_per_fused__to_copy_eq_logical_or_lt_repeat_sum_4.run(arg124_1, buf18, buf23, s10, 50, grid=grid(s10), stream=stream0)
        buf19 = empty_strided_cuda((50*s10, 176), (176, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf4, reinterpret_tensor(L__self___user_model_feedforwards_item_clk_seq_fc1_linear_weight, (176, 176), (1, 176), 0), out=buf19)
        buf20 = reinterpret_tensor(buf19, (s10, 50, 176), (8800, 176, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [leaky_relu], Original ATen: [aten.leaky_relu]
        triton_poi_fused_leaky_relu_5_xnumel = 8800*s10
        triton_poi_fused_leaky_relu_5.run(buf20, L__self___user_model_feedforwards_item_clk_seq_fc1_linear_bias, triton_poi_fused_leaky_relu_5_xnumel, grid=grid(triton_poi_fused_leaky_relu_5_xnumel), stream=stream0)
        buf21 = empty_strided_cuda((50*s10, 176), (176, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (50*s10, 176), (176, 1), 0), reinterpret_tensor(L__self___user_model_feedforwards_item_clk_seq_fc2_linear_weight, (176, 176), (1, 176), 0), out=buf21)
        del buf20
        buf56 = empty_strided_cuda((s10, 1872), (1872, 1), torch.float32)
        buf51 = reinterpret_tensor(buf56, (s10, 176), (1872, 1), 1112)  # alias
        # Topologically Sorted Source Nodes: [sum_4, tile_2, pow_1, mul], Original ATen: [aten.sum, aten.repeat, aten.pow, aten.mul]
        triton_per_fused_mul_pow_repeat_sum_6_xnumel = 176*s10
        triton_per_fused_mul_pow_repeat_sum_6.run(arg124_1, buf18, buf21, L__self___user_model_feedforwards_item_clk_seq_fc2_linear_bias, buf4, buf23, buf51, triton_per_fused_mul_pow_repeat_sum_6_xnumel, 50, grid=grid(triton_per_fused_mul_pow_repeat_sum_6_xnumel), stream=stream0)
        del arg124_1
        del buf21
        buf25 = empty_strided_cuda((s10, 8, 16, 50), (6400, 800, 50, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_7_ynumel = 128*s10
        triton_poi_fused_clone_7.run(buf5, buf6, buf7, L__self___user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm_weight, L__self___user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm_bias, buf25, triton_poi_fused_clone_7_ynumel, 50, grid=grid(triton_poi_fused_clone_7_ynumel, 50), stream=stream0)
        buf26 = reinterpret_tensor(buf7, (8*s10, 1, 50), (50, 50, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        triton_tem_fused_bmm_8.run(buf24, buf25, buf26, grid=torch._inductor.kernel.bmm.bmm_grid(8*s10, 1, 50, meta0), stream=stream0)
        buf30 = reinterpret_tensor(buf6, (s10, 8, 1, 50), (400, 50, 50, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [softmax], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9_xnumel = 8*s10
        triton_per_fused__softmax_9.run(buf26, buf30, triton_per_fused__softmax_9_xnumel, 50, grid=grid(triton_per_fused__softmax_9_xnumel), stream=stream0)
        del buf26
        buf29 = reinterpret_tensor(buf25, (50*s10, 128), (128, 1), 0); del buf25  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf4, reinterpret_tensor(L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_v_linear_weight, (176, 128), (1, 176), 0), out=buf29)
        del buf4
        buf31 = reinterpret_tensor(buf5, (s10, 8, 50, 16), (6400, 800, 16, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_10_xnumel = 6400*s10
        triton_poi_fused_clone_10.run(buf29, L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_v_linear_bias, buf31, triton_poi_fused_clone_10_xnumel, grid=grid(triton_poi_fused_clone_10_xnumel), stream=stream0)
        del buf29
        buf32 = reinterpret_tensor(buf24, (8*s10, 1, 16), (16, 16, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        triton_tem_fused_bmm_11.run(buf30, buf31, buf32, grid=torch._inductor.kernel.bmm.bmm_grid(8*s10, 1, 16, meta1), stream=stream0)
        del buf30
        buf33 = buf23; del buf23  # reuse
        buf38 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [lt_1, to_1, sum_2, eq_1, tile_1, logical_or_1, to_3, sum_5], Original ATen: [aten.lt, aten._to_copy, aten.sum, aten.eq, aten.repeat, aten.logical_or]
        triton_per_fused__to_copy_eq_logical_or_lt_repeat_sum_4.run(arg127_1, buf33, buf38, s10, 50, grid=grid(s10), stream=stream0)
        buf34 = empty_strided_cuda((50*s10, 172), (172, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf13, reinterpret_tensor(L__self___user_model_feedforwards_query_seq_fc1_linear_weight, (172, 172), (1, 172), 0), out=buf34)
        buf35 = reinterpret_tensor(buf34, (s10, 50, 172), (8600, 172, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [leaky_relu_1], Original ATen: [aten.leaky_relu]
        triton_poi_fused_leaky_relu_12_xnumel = 8600*s10
        triton_poi_fused_leaky_relu_12.run(buf35, L__self___user_model_feedforwards_query_seq_fc1_linear_bias, triton_poi_fused_leaky_relu_12_xnumel, grid=grid(triton_poi_fused_leaky_relu_12_xnumel), stream=stream0)
        buf36 = empty_strided_cuda((50*s10, 172), (172, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf35, (50*s10, 172), (172, 1), 0), reinterpret_tensor(L__self___user_model_feedforwards_query_seq_fc2_linear_weight, (172, 172), (1, 172), 0), out=buf36)
        del buf35
        buf53 = reinterpret_tensor(buf56, (s10, 172), (1872, 1), 1416)  # alias
        # Topologically Sorted Source Nodes: [sum_6, tile_3, pow_2, mul_1], Original ATen: [aten.sum, aten.repeat, aten.pow, aten.mul]
        triton_per_fused_mul_pow_repeat_sum_13_xnumel = 172*s10
        triton_per_fused_mul_pow_repeat_sum_13.run(arg127_1, buf33, buf36, L__self___user_model_feedforwards_query_seq_fc2_linear_bias, buf13, buf38, buf53, triton_per_fused_mul_pow_repeat_sum_13_xnumel, 50, grid=grid(triton_per_fused_mul_pow_repeat_sum_13_xnumel), stream=stream0)
        del arg127_1
        del buf36
        buf40 = reinterpret_tensor(buf31, (s10, 8, 16, 50), (6400, 800, 50, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_7_ynumel = 128*s10
        triton_poi_fused_clone_7.run(buf14, buf15, buf16, L__self___user_model_multi_h_attens_query_seq_ta_k_layer_norm_weight, L__self___user_model_multi_h_attens_query_seq_ta_k_layer_norm_bias, buf40, triton_poi_fused_clone_7_ynumel, 50, grid=grid(triton_poi_fused_clone_7_ynumel, 50), stream=stream0)
        buf41 = reinterpret_tensor(buf16, (8*s10, 1, 50), (50, 50, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        triton_tem_fused_bmm_8.run(buf39, buf40, buf41, grid=torch._inductor.kernel.bmm.bmm_grid(8*s10, 1, 50, meta0), stream=stream0)
        buf45 = reinterpret_tensor(buf15, (s10, 8, 1, 50), (400, 50, 50, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [softmax_1], Original ATen: [aten._softmax]
        triton_per_fused__softmax_9_xnumel = 8*s10
        triton_per_fused__softmax_9.run(buf41, buf45, triton_per_fused__softmax_9_xnumel, 50, grid=grid(triton_per_fused__softmax_9_xnumel), stream=stream0)
        del buf41
        buf44 = reinterpret_tensor(buf40, (50*s10, 128), (128, 1), 0); del buf40  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf13, reinterpret_tensor(L__self___user_model_multi_h_attens_query_seq_ta_proj_v_linear_weight, (172, 128), (1, 172), 0), out=buf44)
        del buf13
        buf46 = reinterpret_tensor(buf14, (s10, 8, 50, 16), (6400, 800, 16, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_10_xnumel = 6400*s10
        triton_poi_fused_clone_10.run(buf44, L__self___user_model_multi_h_attens_query_seq_ta_proj_v_linear_bias, buf46, triton_poi_fused_clone_10_xnumel, grid=grid(triton_poi_fused_clone_10_xnumel), stream=stream0)
        del buf44
        buf47 = reinterpret_tensor(buf39, (8*s10, 1, 16), (16, 16, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        triton_tem_fused_bmm_11.run(buf45, buf46, buf47, grid=torch._inductor.kernel.bmm.bmm_grid(8*s10, 1, 16, meta1), stream=stream0)
        del buf45
        del buf46
        buf48 = reinterpret_tensor(buf56, (s10, 204), (1872, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 204*s10
        triton_poi_fused_cat_14.run(arg130_1, buf48, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        buf49 = reinterpret_tensor(buf56, (s10, 220), (1872, 1), 204)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        triton_poi_fused_cat_15_xnumel = 220*s10
        triton_poi_fused_cat_15.run(arg125_1, buf49, triton_poi_fused_cat_15_xnumel, grid=grid(triton_poi_fused_cat_15_xnumel), stream=stream0)
        del arg125_1
        buf50 = reinterpret_tensor(buf56, (s10, 688), (1872, 1), 424)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        triton_poi_fused_cat_16_xnumel = 688*s10
        triton_poi_fused_cat_16.run(arg122_1, buf50, triton_poi_fused_cat_16_xnumel, grid=grid(triton_poi_fused_cat_16_xnumel), stream=stream0)
        del arg122_1
        buf52 = reinterpret_tensor(buf56, (s10, 128), (1872, 1), 1288)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        triton_poi_fused_cat_17_xnumel = 128*s10
        triton_poi_fused_cat_17.run(buf32, buf52, triton_poi_fused_cat_17_xnumel, grid=grid(triton_poi_fused_cat_17_xnumel), stream=stream0)
        buf54 = reinterpret_tensor(buf56, (s10, 128), (1872, 1), 1588)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        triton_poi_fused_cat_17_xnumel = 128*s10
        triton_poi_fused_cat_17.run(buf47, buf54, triton_poi_fused_cat_17_xnumel, grid=grid(triton_poi_fused_cat_17_xnumel), stream=stream0)
        buf55 = reinterpret_tensor(buf56, (s10, 156), (1872, 1), 1716)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        triton_poi_fused_cat_18_xnumel = 156*s10
        triton_poi_fused_cat_18.run(arg128_1, buf55, triton_poi_fused_cat_18_xnumel, grid=grid(triton_poi_fused_cat_18_xnumel), stream=stream0)
        del arg128_1
        del buf48
        del buf49
        del buf50
        del buf52
        del buf54
        del buf55
        buf57 = empty_strided_cuda((s10, 512), (512, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf56, reinterpret_tensor(getattr_L__self___user_model_ctr_net_tower_layers___0___fc_weight, (1872, 512), (1, 1872), 0), out=buf57)
        buf58 = buf57; del buf57  # reuse
        buf59 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [user_model_ctr_net_tower_layers_0_norm, user_model_ctr_net_tower_layers_0_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel = 512*s10
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19.run(buf59, getattr_L__self___user_model_ctr_net_tower_layers___0___norm_running_mean, getattr_L__self___user_model_ctr_net_tower_layers___0___norm_running_var, getattr_L__self___user_model_ctr_net_tower_layers___0___norm_weight, getattr_L__self___user_model_ctr_net_tower_layers___0___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel), stream=stream0)
        buf60 = empty_strided_cuda((s10, 256), (256, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf59, reinterpret_tensor(getattr_L__self___user_model_ctr_net_tower_layers___1___fc_weight, (512, 256), (1, 512), 0), out=buf60)
        buf61 = buf60; del buf60  # reuse
        buf62 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [user_model_ctr_net_tower_layers_1_norm, user_model_ctr_net_tower_layers_1_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20_xnumel = 256*s10
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20.run(buf62, getattr_L__self___user_model_ctr_net_tower_layers___1___norm_running_mean, getattr_L__self___user_model_ctr_net_tower_layers___1___norm_running_var, getattr_L__self___user_model_ctr_net_tower_layers___1___norm_weight, getattr_L__self___user_model_ctr_net_tower_layers___1___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20_xnumel), stream=stream0)
        buf63 = reinterpret_tensor(buf47, (s10, 128), (128, 1), 0); del buf47  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf62, reinterpret_tensor(getattr_L__self___user_model_ctr_net_tower_layers___2___fc_weight, (256, 128), (1, 256), 0), out=buf63)
        buf64 = buf63; del buf63  # reuse
        buf71 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [user_model_ctr_net_tower_layers_2_norm, user_model_ctr_net_tower_layers_2_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21_xnumel = 128*s10
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21.run(buf71, getattr_L__self___user_model_ctr_net_tower_layers___2___norm_running_mean, getattr_L__self___user_model_ctr_net_tower_layers___2___norm_running_var, getattr_L__self___user_model_ctr_net_tower_layers___2___norm_weight, getattr_L__self___user_model_ctr_net_tower_layers___2___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21_xnumel), stream=stream0)
        buf65 = empty_strided_cuda((s10, 568), (568, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        triton_poi_fused_cat_22_xnumel = 568*s10
        triton_poi_fused_cat_22.run(arg130_1, arg129_1, buf51, buf53, buf65, triton_poi_fused_cat_22_xnumel, grid=grid(triton_poi_fused_cat_22_xnumel), stream=stream0)
        del arg129_1
        del arg130_1
        del buf51
        del buf53
        buf66 = buf62; del buf62  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf65, reinterpret_tensor(getattr_L__self___user_model_bias_net_tower_layers___0___fc_weight, (568, 256), (1, 568), 0), out=buf66)
        del buf65
        buf67 = buf66; del buf66  # reuse
        buf68 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [user_model_bias_net_tower_layers_0_norm, user_model_bias_net_tower_layers_0_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20_xnumel = 256*s10
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20.run(buf68, getattr_L__self___user_model_bias_net_tower_layers___0___norm_running_mean, getattr_L__self___user_model_bias_net_tower_layers___0___norm_running_var, getattr_L__self___user_model_bias_net_tower_layers___0___norm_weight, getattr_L__self___user_model_bias_net_tower_layers___0___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20_xnumel), stream=stream0)
        buf69 = reinterpret_tensor(buf32, (s10, 128), (128, 1), 0); del buf32  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf68, reinterpret_tensor(getattr_L__self___user_model_bias_net_tower_layers___1___fc_weight, (256, 128), (1, 256), 0), out=buf69)
        buf70 = buf69; del buf69  # reuse
        buf72 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [user_model_bias_net_tower_layers_1_norm, user_model_bias_net_tower_layers_1_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21_xnumel = 128*s10
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21.run(buf72, getattr_L__self___user_model_bias_net_tower_layers___1___norm_running_mean, getattr_L__self___user_model_bias_net_tower_layers___1___norm_running_var, getattr_L__self___user_model_bias_net_tower_layers___1___norm_weight, getattr_L__self___user_model_bias_net_tower_layers___1___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21_xnumel), stream=stream0)
        buf73 = reinterpret_tensor(buf38, (s10, 1), (1, 1), 0); del buf38  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels._mm_plus_mm(buf71, reinterpret_tensor(L__self___user_model_ctr_head_weight, (128, 1), (1, 128), 0), buf72, reinterpret_tensor(L__self___user_model_ctr_bias_head_weight, (128, 1), (1, 128), 0), out=buf73)
        buf74 = buf59; del buf59  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf56, reinterpret_tensor(getattr_L__self___user_model_click_net_tower_layers___0___fc_weight, (1872, 512), (1, 1872), 0), out=buf74)
        buf75 = buf74; del buf74  # reuse
        buf76 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [user_model_click_net_tower_layers_0_norm, user_model_click_net_tower_layers_0_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel = 512*s10
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19.run(buf76, getattr_L__self___user_model_click_net_tower_layers___0___norm_running_mean, getattr_L__self___user_model_click_net_tower_layers___0___norm_running_var, getattr_L__self___user_model_click_net_tower_layers___0___norm_weight, getattr_L__self___user_model_click_net_tower_layers___0___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel), stream=stream0)
        buf77 = buf68; del buf68  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf76, reinterpret_tensor(getattr_L__self___user_model_click_net_tower_layers___1___fc_weight, (512, 256), (1, 512), 0), out=buf77)
        buf78 = buf77; del buf77  # reuse
        buf79 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [user_model_click_net_tower_layers_1_norm, user_model_click_net_tower_layers_1_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20_xnumel = 256*s10
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20.run(buf79, getattr_L__self___user_model_click_net_tower_layers___1___norm_running_mean, getattr_L__self___user_model_click_net_tower_layers___1___norm_running_var, getattr_L__self___user_model_click_net_tower_layers___1___norm_weight, getattr_L__self___user_model_click_net_tower_layers___1___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20_xnumel), stream=stream0)
        buf80 = buf71; del buf71  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf79, reinterpret_tensor(getattr_L__self___user_model_click_net_tower_layers___2___fc_weight, (256, 128), (1, 256), 0), out=buf80)
        buf81 = buf80; del buf80  # reuse
        buf82 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [user_model_click_net_tower_layers_2_norm, user_model_click_net_tower_layers_2_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21_xnumel = 128*s10
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21.run(buf82, getattr_L__self___user_model_click_net_tower_layers___2___norm_running_mean, getattr_L__self___user_model_click_net_tower_layers___2___norm_running_var, getattr_L__self___user_model_click_net_tower_layers___2___norm_weight, getattr_L__self___user_model_click_net_tower_layers___2___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21_xnumel), stream=stream0)
        buf83 = reinterpret_tensor(buf33, (s10, 1), (1, 1), 0); del buf33  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels._mm_plus_mm(buf82, reinterpret_tensor(L__self___user_model_click_head_weight, (128, 1), (1, 128), 0), buf72, reinterpret_tensor(L__self___user_model_click_bias_head_weight, (128, 1), (1, 128), 0), out=buf83)
        buf84 = buf76; del buf76  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf56, reinterpret_tensor(getattr_L__self___user_model_page_net_tower_layers___0___fc_weight, (1872, 512), (1, 1872), 0), out=buf84)
        buf85 = buf84; del buf84  # reuse
        buf86 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [user_model_page_net_tower_layers_0_norm, user_model_page_net_tower_layers_0_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel = 512*s10
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19.run(buf86, getattr_L__self___user_model_page_net_tower_layers___0___norm_running_mean, getattr_L__self___user_model_page_net_tower_layers___0___norm_running_var, getattr_L__self___user_model_page_net_tower_layers___0___norm_weight, getattr_L__self___user_model_page_net_tower_layers___0___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel), stream=stream0)
        buf87 = buf79; del buf79  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf86, reinterpret_tensor(getattr_L__self___user_model_page_net_tower_layers___1___fc_weight, (512, 256), (1, 512), 0), out=buf87)
        buf88 = buf87; del buf87  # reuse
        buf89 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [user_model_page_net_tower_layers_1_norm, user_model_page_net_tower_layers_1_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20_xnumel = 256*s10
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20.run(buf89, getattr_L__self___user_model_page_net_tower_layers___1___norm_running_mean, getattr_L__self___user_model_page_net_tower_layers___1___norm_running_var, getattr_L__self___user_model_page_net_tower_layers___1___norm_weight, getattr_L__self___user_model_page_net_tower_layers___1___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20_xnumel), stream=stream0)
        buf90 = buf82; del buf82  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf89, reinterpret_tensor(getattr_L__self___user_model_page_net_tower_layers___2___fc_weight, (256, 128), (1, 256), 0), out=buf90)
        buf91 = buf90; del buf90  # reuse
        buf92 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [user_model_page_net_tower_layers_2_norm, user_model_page_net_tower_layers_2_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21_xnumel = 128*s10
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21.run(buf92, getattr_L__self___user_model_page_net_tower_layers___2___norm_running_mean, getattr_L__self___user_model_page_net_tower_layers___2___norm_running_var, getattr_L__self___user_model_page_net_tower_layers___2___norm_weight, getattr_L__self___user_model_page_net_tower_layers___2___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21_xnumel), stream=stream0)
        buf93 = empty_strided_cuda((s10, 1), (1, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels._mm_plus_mm(buf92, reinterpret_tensor(L__self___user_model_page_head_weight, (128, 1), (1, 128), 0), buf72, reinterpret_tensor(L__self___user_model_page_bias_head_weight, (128, 1), (1, 128), 0), out=buf93)
        buf94 = buf86; del buf86  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf56, reinterpret_tensor(getattr_L__self___user_model_pay_net_tower_layers___0___fc_weight, (1872, 512), (1, 1872), 0), out=buf94)
        del buf56
        buf95 = buf94; del buf94  # reuse
        buf96 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [user_model_pay_net_tower_layers_0_norm, user_model_pay_net_tower_layers_0_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel = 512*s10
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19.run(buf96, getattr_L__self___user_model_pay_net_tower_layers___0___norm_running_mean, getattr_L__self___user_model_pay_net_tower_layers___0___norm_running_var, getattr_L__self___user_model_pay_net_tower_layers___0___norm_weight, getattr_L__self___user_model_pay_net_tower_layers___0___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel), stream=stream0)
        buf97 = buf89; del buf89  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf96, reinterpret_tensor(getattr_L__self___user_model_pay_net_tower_layers___1___fc_weight, (512, 256), (1, 512), 0), out=buf97)
        del buf96
        buf98 = buf97; del buf97  # reuse
        buf99 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [user_model_pay_net_tower_layers_1_norm, user_model_pay_net_tower_layers_1_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20_xnumel = 256*s10
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20.run(buf99, getattr_L__self___user_model_pay_net_tower_layers___1___norm_running_mean, getattr_L__self___user_model_pay_net_tower_layers___1___norm_running_var, getattr_L__self___user_model_pay_net_tower_layers___1___norm_weight, getattr_L__self___user_model_pay_net_tower_layers___1___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_20_xnumel), stream=stream0)
        buf100 = buf92; del buf92  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf99, reinterpret_tensor(getattr_L__self___user_model_pay_net_tower_layers___2___fc_weight, (256, 128), (1, 256), 0), out=buf100)
        del buf99
        buf101 = buf100; del buf100  # reuse
        buf102 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [user_model_pay_net_tower_layers_2_norm, user_model_pay_net_tower_layers_2_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21_xnumel = 128*s10
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21.run(buf102, getattr_L__self___user_model_pay_net_tower_layers___2___norm_running_mean, getattr_L__self___user_model_pay_net_tower_layers___2___norm_running_var, getattr_L__self___user_model_pay_net_tower_layers___2___norm_weight, getattr_L__self___user_model_pay_net_tower_layers___2___norm_bias, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21_xnumel, grid=grid(triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_21_xnumel), stream=stream0)
        buf103 = empty_strided_cuda((s10, 1), (1, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels._mm_plus_mm(buf102, reinterpret_tensor(L__self___user_model_pay_head_weight, (128, 1), (1, 128), 0), buf72, reinterpret_tensor(L__self___user_model_pay_bias_head_weight, (128, 1), (1, 128), 0), out=buf103)
        del buf102
        del buf72

    for kernel in globals().values():
        if isinstance(kernel, torch._inductor.runtime.triton_heuristics.CachingAutotuner):
            if not kernel.cuda_kernel_saved:
                if len(kernel.launchers) == 0:
                    kernel.precompile()
                kernel.save_gpu_kernel(
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
