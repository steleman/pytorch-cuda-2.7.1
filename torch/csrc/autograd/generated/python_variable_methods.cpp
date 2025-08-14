#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// @generated from ../tools/autograd/templates/python_variable_methods.cpp

#include <Python.h>

// Undefine the copysign macro so that at::copysign works as intended with MSVC
// https://github.com/python/cpython/blob/c60394c7fc9cc09b16e9675a3eeb5844b6d8523f/PC/pyconfig.h#L196
#ifdef _MSC_VER
#undef copysign
#endif // _MSC_VER

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/Size.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/autograd/utils/error_messages.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/jit/frontend/tracer.h"
#ifdef USE_CUDA
#include "torch/csrc/cuda/Event.h"
#endif
#include "torch/csrc/utils/device_lazy_init.h"
#include <torch/csrc/utils/numpy_stub.h>
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/python_numbers.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/tensor_apply.h"
#include "torch/csrc/utils/tensor_list.h"
#include "torch/csrc/utils/tensor_new.h"
#include "torch/csrc/utils/tensor_numpy.h"
#include "torch/csrc/utils/tensor_types.h"
#include "torch/csrc/autograd/generated/python_return_types.h"

#include <ATen/core/Tensor.h>
#include <ATen/FuncTorchTLS.h>
#include "c10/core/Stream.h"

#include <optional>
#include <stdexcept>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/and.h>
#include <ATen/ops/and.h>
#include <ATen/ops/lshift.h>
#include <ATen/ops/or.h>
#include <ATen/ops/rshift.h>
#include <ATen/ops/xor.h>
#include <ATen/ops/lshift.h>
#include <ATen/ops/or.h>
#include <ATen/ops/rshift.h>
#include <ATen/ops/xor.h>
#include <ATen/ops/_addmm_activation.h>
#include <ATen/ops/_autocast_to_full_precision.h>
#include <ATen/ops/_autocast_to_reduced_precision.h>
#include <ATen/ops/_coalesced.h>
#include <ATen/ops/_conj.h>
#include <ATen/ops/_conj_physical.h>
#include <ATen/ops/_dimI.h>
#include <ATen/ops/_dimV.h>
#include <ATen/ops/_indices.h>
#include <ATen/ops/_is_all_true.h>
#include <ATen/ops/_is_any_true.h>
#include <ATen/ops/_is_zerotensor.h>
#include <ATen/ops/_lazy_clone.h>
#include <ATen/ops/_neg_view.h>
#include <ATen/ops/_nested_tensor_size.h>
#include <ATen/ops/_nested_tensor_storage_offsets.h>
#include <ATen/ops/_nested_tensor_strides.h>
#include <ATen/ops/_nnz.h>
#include <ATen/ops/_sparse_mask_projection.h>
#include <ATen/ops/_to_dense.h>
#include <ATen/ops/_to_sparse.h>
#include <ATen/ops/_to_sparse_bsc.h>
#include <ATen/ops/_to_sparse_bsr.h>
#include <ATen/ops/_to_sparse_csc.h>
#include <ATen/ops/_to_sparse_csr.h>
#include <ATen/ops/_values.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/absolute.h>
#include <ATen/ops/absolute.h>
#include <ATen/ops/acos.h>
#include <ATen/ops/acos.h>
#include <ATen/ops/acosh.h>
#include <ATen/ops/acosh.h>
#include <ATen/ops/add.h>
#include <ATen/ops/add.h>
#include <ATen/ops/addbmm.h>
#include <ATen/ops/addbmm.h>
#include <ATen/ops/addcdiv.h>
#include <ATen/ops/addcdiv.h>
#include <ATen/ops/addcmul.h>
#include <ATen/ops/addcmul.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/addmv.h>
#include <ATen/ops/addmv.h>
#include <ATen/ops/addr.h>
#include <ATen/ops/addr.h>
#include <ATen/ops/adjoint.h>
#include <ATen/ops/align_as.h>
#include <ATen/ops/align_to.h>
#include <ATen/ops/all.h>
#include <ATen/ops/allclose.h>
#include <ATen/ops/amax.h>
#include <ATen/ops/amin.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/angle.h>
#include <ATen/ops/any.h>
#include <ATen/ops/arccos.h>
#include <ATen/ops/arccos.h>
#include <ATen/ops/arccosh.h>
#include <ATen/ops/arccosh.h>
#include <ATen/ops/arcsin.h>
#include <ATen/ops/arcsin.h>
#include <ATen/ops/arcsinh.h>
#include <ATen/ops/arcsinh.h>
#include <ATen/ops/arctan.h>
#include <ATen/ops/arctan2.h>
#include <ATen/ops/arctan2.h>
#include <ATen/ops/arctan.h>
#include <ATen/ops/arctanh.h>
#include <ATen/ops/arctanh.h>
#include <ATen/ops/argmax.h>
#include <ATen/ops/argmin.h>
#include <ATen/ops/argsort.h>
#include <ATen/ops/argwhere.h>
#include <ATen/ops/as_strided.h>
#include <ATen/ops/as_strided.h>
#include <ATen/ops/as_strided_scatter.h>
#include <ATen/ops/asin.h>
#include <ATen/ops/asin.h>
#include <ATen/ops/asinh.h>
#include <ATen/ops/asinh.h>
#include <ATen/ops/atan.h>
#include <ATen/ops/atan2.h>
#include <ATen/ops/atan2.h>
#include <ATen/ops/atan.h>
#include <ATen/ops/atanh.h>
#include <ATen/ops/atanh.h>
#include <ATen/ops/baddbmm.h>
#include <ATen/ops/baddbmm.h>
#include <ATen/ops/bernoulli.h>
#include <ATen/ops/bernoulli.h>
#include <ATen/ops/bincount.h>
#include <ATen/ops/bitwise_and.h>
#include <ATen/ops/bitwise_and.h>
#include <ATen/ops/bitwise_left_shift.h>
#include <ATen/ops/bitwise_left_shift.h>
#include <ATen/ops/bitwise_not.h>
#include <ATen/ops/bitwise_not.h>
#include <ATen/ops/bitwise_or.h>
#include <ATen/ops/bitwise_or.h>
#include <ATen/ops/bitwise_right_shift.h>
#include <ATen/ops/bitwise_right_shift.h>
#include <ATen/ops/bitwise_xor.h>
#include <ATen/ops/bitwise_xor.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/broadcast_to.h>
#include <ATen/ops/cauchy.h>
#include <ATen/ops/ccol_indices.h>
#include <ATen/ops/ceil.h>
#include <ATen/ops/ceil.h>
#include <ATen/ops/chalf.h>
#include <ATen/ops/cholesky.h>
#include <ATen/ops/cholesky_inverse.h>
#include <ATen/ops/cholesky_solve.h>
#include <ATen/ops/chunk.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/clamp_max.h>
#include <ATen/ops/clamp_max.h>
#include <ATen/ops/clamp_min.h>
#include <ATen/ops/clamp_min.h>
#include <ATen/ops/clip.h>
#include <ATen/ops/clip.h>
#include <ATen/ops/clone.h>
#include <ATen/ops/coalesce.h>
#include <ATen/ops/col_indices.h>
#include <ATen/ops/conj.h>
#include <ATen/ops/conj_physical.h>
#include <ATen/ops/conj_physical.h>
#include <ATen/ops/copysign.h>
#include <ATen/ops/copysign.h>
#include <ATen/ops/corrcoef.h>
#include <ATen/ops/cos.h>
#include <ATen/ops/cos.h>
#include <ATen/ops/cosh.h>
#include <ATen/ops/cosh.h>
#include <ATen/ops/count_nonzero.h>
#include <ATen/ops/cov.h>
#include <ATen/ops/cross.h>
#include <ATen/ops/crow_indices.h>
#include <ATen/ops/cummax.h>
#include <ATen/ops/cummin.h>
#include <ATen/ops/cumprod.h>
#include <ATen/ops/cumprod.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/deg2rad.h>
#include <ATen/ops/deg2rad.h>
#include <ATen/ops/dense_dim.h>
#include <ATen/ops/dequantize.h>
#include <ATen/ops/det.h>
#include <ATen/ops/detach.h>
#include <ATen/ops/detach.h>
#include <ATen/ops/diag.h>
#include <ATen/ops/diag_embed.h>
#include <ATen/ops/diagflat.h>
#include <ATen/ops/diagonal.h>
#include <ATen/ops/diagonal_scatter.h>
#include <ATen/ops/diff.h>
#include <ATen/ops/digamma.h>
#include <ATen/ops/digamma.h>
#include <ATen/ops/dist.h>
#include <ATen/ops/div.h>
#include <ATen/ops/div.h>
#include <ATen/ops/divide.h>
#include <ATen/ops/divide.h>
#include <ATen/ops/dot.h>
#include <ATen/ops/dsplit.h>
#include <ATen/ops/eq.h>
#include <ATen/ops/eq.h>
#include <ATen/ops/equal.h>
#include <ATen/ops/erf.h>
#include <ATen/ops/erf.h>
#include <ATen/ops/erfc.h>
#include <ATen/ops/erfc.h>
#include <ATen/ops/erfinv.h>
#include <ATen/ops/erfinv.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/exp2.h>
#include <ATen/ops/exp2.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/expand.h>
#include <ATen/ops/expand_as.h>
#include <ATen/ops/expm1.h>
#include <ATen/ops/expm1.h>
#include <ATen/ops/exponential.h>
#include <ATen/ops/fill.h>
#include <ATen/ops/fill_diagonal.h>
#include <ATen/ops/fix.h>
#include <ATen/ops/fix.h>
#include <ATen/ops/flatten.h>
#include <ATen/ops/flip.h>
#include <ATen/ops/fliplr.h>
#include <ATen/ops/flipud.h>
#include <ATen/ops/float_power.h>
#include <ATen/ops/float_power.h>
#include <ATen/ops/floor.h>
#include <ATen/ops/floor.h>
#include <ATen/ops/floor_divide.h>
#include <ATen/ops/floor_divide.h>
#include <ATen/ops/fmax.h>
#include <ATen/ops/fmin.h>
#include <ATen/ops/fmod.h>
#include <ATen/ops/fmod.h>
#include <ATen/ops/frac.h>
#include <ATen/ops/frac.h>
#include <ATen/ops/frexp.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/gcd.h>
#include <ATen/ops/gcd.h>
#include <ATen/ops/ge.h>
#include <ATen/ops/ge.h>
#include <ATen/ops/geometric.h>
#include <ATen/ops/geqrf.h>
#include <ATen/ops/ger.h>
#include <ATen/ops/greater.h>
#include <ATen/ops/greater.h>
#include <ATen/ops/greater_equal.h>
#include <ATen/ops/greater_equal.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/hardshrink.h>
#include <ATen/ops/heaviside.h>
#include <ATen/ops/heaviside.h>
#include <ATen/ops/histc.h>
#include <ATen/ops/histogram.h>
#include <ATen/ops/hsplit.h>
#include <ATen/ops/hypot.h>
#include <ATen/ops/hypot.h>
#include <ATen/ops/i0.h>
#include <ATen/ops/i0.h>
#include <ATen/ops/igamma.h>
#include <ATen/ops/igamma.h>
#include <ATen/ops/igammac.h>
#include <ATen/ops/igammac.h>
#include <ATen/ops/index_add.h>
#include <ATen/ops/index_add.h>
#include <ATen/ops/index_copy.h>
#include <ATen/ops/index_copy.h>
#include <ATen/ops/index_fill.h>
#include <ATen/ops/index_fill.h>
#include <ATen/ops/index_put.h>
#include <ATen/ops/index_put.h>
#include <ATen/ops/index_reduce.h>
#include <ATen/ops/index_reduce.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/indices.h>
#include <ATen/ops/inner.h>
#include <ATen/ops/int_repr.h>
#include <ATen/ops/inverse.h>
#include <ATen/ops/is_coalesced.h>
#include <ATen/ops/is_complex.h>
#include <ATen/ops/is_conj.h>
#include <ATen/ops/is_distributed.h>
#include <ATen/ops/is_floating_point.h>
#include <ATen/ops/is_inference.h>
#include <ATen/ops/is_neg.h>
#include <ATen/ops/is_nonzero.h>
#include <ATen/ops/is_pinned.h>
#include <ATen/ops/is_same_size.h>
#include <ATen/ops/is_set_to.h>
#include <ATen/ops/is_signed.h>
#include <ATen/ops/isclose.h>
#include <ATen/ops/isfinite.h>
#include <ATen/ops/isinf.h>
#include <ATen/ops/isnan.h>
#include <ATen/ops/isneginf.h>
#include <ATen/ops/isposinf.h>
#include <ATen/ops/isreal.h>
#include <ATen/ops/istft.h>
#include <ATen/ops/kron.h>
#include <ATen/ops/kthvalue.h>
#include <ATen/ops/lcm.h>
#include <ATen/ops/lcm.h>
#include <ATen/ops/ldexp.h>
#include <ATen/ops/ldexp.h>
#include <ATen/ops/le.h>
#include <ATen/ops/le.h>
#include <ATen/ops/lerp.h>
#include <ATen/ops/lerp.h>
#include <ATen/ops/less.h>
#include <ATen/ops/less.h>
#include <ATen/ops/less_equal.h>
#include <ATen/ops/less_equal.h>
#include <ATen/ops/lgamma.h>
#include <ATen/ops/lgamma.h>
#include <ATen/ops/log.h>
#include <ATen/ops/log10.h>
#include <ATen/ops/log10.h>
#include <ATen/ops/log1p.h>
#include <ATen/ops/log1p.h>
#include <ATen/ops/log2.h>
#include <ATen/ops/log2.h>
#include <ATen/ops/log.h>
#include <ATen/ops/log_normal.h>
#include <ATen/ops/log_softmax.h>
#include <ATen/ops/logaddexp.h>
#include <ATen/ops/logaddexp2.h>
#include <ATen/ops/logcumsumexp.h>
#include <ATen/ops/logdet.h>
#include <ATen/ops/logical_and.h>
#include <ATen/ops/logical_and.h>
#include <ATen/ops/logical_not.h>
#include <ATen/ops/logical_not.h>
#include <ATen/ops/logical_or.h>
#include <ATen/ops/logical_or.h>
#include <ATen/ops/logical_xor.h>
#include <ATen/ops/logical_xor.h>
#include <ATen/ops/logit.h>
#include <ATen/ops/logit.h>
#include <ATen/ops/logsumexp.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/lu_solve.h>
#include <ATen/ops/masked_fill.h>
#include <ATen/ops/masked_fill.h>
#include <ATen/ops/masked_scatter.h>
#include <ATen/ops/masked_scatter.h>
#include <ATen/ops/masked_select.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/matrix_exp.h>
#include <ATen/ops/matrix_power.h>
#include <ATen/ops/max.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/median.h>
#include <ATen/ops/min.h>
#include <ATen/ops/minimum.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/mode.h>
#include <ATen/ops/moveaxis.h>
#include <ATen/ops/movedim.h>
#include <ATen/ops/msort.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/multinomial.h>
#include <ATen/ops/multiply.h>
#include <ATen/ops/multiply.h>
#include <ATen/ops/mv.h>
#include <ATen/ops/mvlgamma.h>
#include <ATen/ops/mvlgamma.h>
#include <ATen/ops/nan_to_num.h>
#include <ATen/ops/nan_to_num.h>
#include <ATen/ops/nanmean.h>
#include <ATen/ops/nanmedian.h>
#include <ATen/ops/nanquantile.h>
#include <ATen/ops/nansum.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/narrow_copy.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/negative.h>
#include <ATen/ops/negative.h>
#include <ATen/ops/new_empty.h>
#include <ATen/ops/new_empty_strided.h>
#include <ATen/ops/new_full.h>
#include <ATen/ops/new_ones.h>
#include <ATen/ops/new_zeros.h>
#include <ATen/ops/nextafter.h>
#include <ATen/ops/nextafter.h>
#include <ATen/ops/nonzero_static.h>
#include <ATen/ops/norm.h>
#include <ATen/ops/normal.h>
#include <ATen/ops/not_equal.h>
#include <ATen/ops/not_equal.h>
#include <ATen/ops/orgqr.h>
#include <ATen/ops/ormqr.h>
#include <ATen/ops/outer.h>
#include <ATen/ops/permute.h>
#include <ATen/ops/pin_memory.h>
#include <ATen/ops/pinverse.h>
#include <ATen/ops/polygamma.h>
#include <ATen/ops/polygamma.h>
#include <ATen/ops/positive.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/prelu.h>
#include <ATen/ops/prod.h>
#include <ATen/ops/put.h>
#include <ATen/ops/put.h>
#include <ATen/ops/q_per_channel_axis.h>
#include <ATen/ops/q_per_channel_scales.h>
#include <ATen/ops/q_per_channel_zero_points.h>
#include <ATen/ops/q_scale.h>
#include <ATen/ops/q_zero_point.h>
#include <ATen/ops/qr.h>
#include <ATen/ops/qscheme.h>
#include <ATen/ops/quantile.h>
#include <ATen/ops/rad2deg.h>
#include <ATen/ops/rad2deg.h>
#include <ATen/ops/random.h>
#include <ATen/ops/ravel.h>
#include <ATen/ops/reciprocal.h>
#include <ATen/ops/reciprocal.h>
#include <ATen/ops/record_stream.h>
#include <ATen/ops/refine_names.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/remainder.h>
#include <ATen/ops/remainder.h>
#include <ATen/ops/rename.h>
#include <ATen/ops/rename.h>
#include <ATen/ops/renorm.h>
#include <ATen/ops/renorm.h>
#include <ATen/ops/repeat.h>
#include <ATen/ops/repeat_interleave.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/reshape_as.h>
#include <ATen/ops/resize.h>
#include <ATen/ops/resize_as.h>
#include <ATen/ops/resize_as_sparse.h>
#include <ATen/ops/resolve_conj.h>
#include <ATen/ops/resolve_neg.h>
#include <ATen/ops/retain_grad.h>
#include <ATen/ops/roll.h>
#include <ATen/ops/rot90.h>
#include <ATen/ops/round.h>
#include <ATen/ops/round.h>
#include <ATen/ops/row_indices.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/scatter.h>
#include <ATen/ops/scatter.h>
#include <ATen/ops/scatter_add.h>
#include <ATen/ops/scatter_add.h>
#include <ATen/ops/scatter_reduce.h>
#include <ATen/ops/scatter_reduce.h>
#include <ATen/ops/select.h>
#include <ATen/ops/select_scatter.h>
#include <ATen/ops/sgn.h>
#include <ATen/ops/sgn.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/sign.h>
#include <ATen/ops/sign.h>
#include <ATen/ops/signbit.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/sinc.h>
#include <ATen/ops/sinc.h>
#include <ATen/ops/sinh.h>
#include <ATen/ops/sinh.h>
#include <ATen/ops/slice_inverse.h>
#include <ATen/ops/slice_scatter.h>
#include <ATen/ops/slogdet.h>
#include <ATen/ops/smm.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/sparse_dim.h>
#include <ATen/ops/sparse_mask.h>
#include <ATen/ops/sparse_resize.h>
#include <ATen/ops/sparse_resize_and_clear.h>
#include <ATen/ops/split.h>
#include <ATen/ops/split_with_sizes.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/square.h>
#include <ATen/ops/square.h>
#include <ATen/ops/squeeze.h>
#include <ATen/ops/squeeze.h>
#include <ATen/ops/sspaddmm.h>
#include <ATen/ops/std.h>
#include <ATen/ops/stft.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/subtract.h>
#include <ATen/ops/subtract.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/sum_to_size.h>
#include <ATen/ops/svd.h>
#include <ATen/ops/swapaxes.h>
#include <ATen/ops/swapaxes.h>
#include <ATen/ops/swapdims.h>
#include <ATen/ops/swapdims.h>
#include <ATen/ops/t.h>
#include <ATen/ops/t.h>
#include <ATen/ops/take.h>
#include <ATen/ops/take_along_dim.h>
#include <ATen/ops/tan.h>
#include <ATen/ops/tan.h>
#include <ATen/ops/tanh.h>
#include <ATen/ops/tanh.h>
#include <ATen/ops/tensor_split.h>
#include <ATen/ops/tile.h>
#include <ATen/ops/to_dense.h>
#include <ATen/ops/to_mkldnn.h>
#include <ATen/ops/to_padded_tensor.h>
#include <ATen/ops/to_sparse.h>
#include <ATen/ops/to_sparse_bsc.h>
#include <ATen/ops/to_sparse_bsr.h>
#include <ATen/ops/to_sparse_csc.h>
#include <ATen/ops/to_sparse_csr.h>
#include <ATen/ops/topk.h>
#include <ATen/ops/trace.h>
#include <ATen/ops/transpose.h>
#include <ATen/ops/transpose.h>
#include <ATen/ops/triangular_solve.h>
#include <ATen/ops/tril.h>
#include <ATen/ops/tril.h>
#include <ATen/ops/triu.h>
#include <ATen/ops/triu.h>
#include <ATen/ops/true_divide.h>
#include <ATen/ops/true_divide.h>
#include <ATen/ops/trunc.h>
#include <ATen/ops/trunc.h>
#include <ATen/ops/type_as.h>
#include <ATen/ops/unbind.h>
#include <ATen/ops/unflatten.h>
#include <ATen/ops/unfold.h>
#include <ATen/ops/uniform.h>
#include <ATen/ops/unsafe_chunk.h>
#include <ATen/ops/unsafe_split.h>
#include <ATen/ops/unsafe_split_with_sizes.h>
#include <ATen/ops/unsqueeze.h>
#include <ATen/ops/unsqueeze.h>
#include <ATen/ops/values.h>
#include <ATen/ops/var.h>
#include <ATen/ops/vdot.h>
#include <ATen/ops/view.h>
#include <ATen/ops/view_as.h>
#include <ATen/ops/vsplit.h>
#include <ATen/ops/where.h>
#include <ATen/ops/xlogy.h>
#include <ATen/ops/xlogy.h>
#include <ATen/ops/zero.h>
#include <ATen/ops/_local_scalar_dense.h>
#endif

using at::device_of;
using at::OptionalDeviceGuard;
using at::Scalar;
using at::ScalarType;
using at::Tensor;
using c10::Stream;
using namespace torch::autograd::utils;

namespace torch::autograd {

static PyObject * THPVariable__is_view(PyObject *self, PyObject* args)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "_is_view", args);
  }
  auto& self_ = THPVariable_Unpack(self);
  if (self_.is_view()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// implemented on the python object bc no support for first-class functions in native_functions.yaml
// See: ATen/native/README.md for more context
static PyObject * THPVariable_apply_(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    auto args = py::make_tuple(py::handle(arg));
    return handle_torch_function(self, "apply_", args.ptr());
  }
  auto& self_ = THPVariable_Unpack(self);
  if (self_.requires_grad()) {
    throw std::runtime_error(
        "Can't call apply_() on Variable that requires grad. Use "
        "var.detach().apply_() instead.");
  }
  return THPVariable_Wrap(torch::utils::apply_(self_, arg));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_size(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "size(int64_t? dim=None)",
    "size(Dimname dim)",
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  if (r.idx == 0) {
    if (!r.toInt64Optional(0).has_value()) {
      return THPSize_NewFromSymSizes(self_);
    }
    if (jit::tracer::isTracing()) {
      // will error out if a tensor has symints
      return wrap(jit::tracer::getSizeOf(self_, r.toInt64(0)));
    } else {
      return torch::toPyObject(self_.sym_size(r.toInt64(0)));
    }
  } else if (r.idx == 1) {
    if (jit::tracer::isTracing()) {
      TORCH_INTERNAL_ASSERT(false, "NYI: Named tensors w/ JIT");
    }
    return wrap(self_.size(r.dimname(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_stride(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "stride(int64_t? dim=None)",
    "stride(Dimname dim)",
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  if (r.idx == 0) {
    if (r.toInt64Optional(0).has_value()) {
      return torch::toPyObject(self_.sym_stride(r.toInt64(0)));
    }
    // yes, this is called strides in ATen.
    at::SymIntArrayRef strides = self_.sym_strides();
    // we can't do the normal wrapping here because IntArrayRef maps to both
    // torch.Size and tuple in python
    // TODO: consider factoring this out
    THPObjectPtr tuple(PyTuple_New(static_cast<Py_ssize_t>(strides.size())));
    if (!tuple) throw python_error();
    for (size_t i = 0; i != strides.size(); i++) {
      PyObject* s = torch::toPyObject(strides[i]);
      if (!s) throw python_error();
      PyTuple_SET_ITEM(tuple.get(), i, s);
    }
    return tuple.release();
  } else if (r.idx == 1) {
    return wrap(self_.stride(r.dimname(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_get_device(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self_)) {
    return handle_torch_function(self_, "get_device", args, nullptr);
  }
  auto& self = THPVariable_Unpack(self_);
  return wrap(self.get_device());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_has_names(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self_)) {
    return handle_torch_function(self_, "has_names", args);
  }
  auto& self = THPVariable_Unpack(self_);
  return wrap(self.has_names());
  END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_data_ptr(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self_)) {
    return handle_torch_function(self_, "data_ptr", args);
  }
  auto& self = THPVariable_Unpack(self_);
  return wrap(self.data_ptr());
  END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_storage_offset(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self_)) {
    return handle_torch_function(self_, "storage_offset");
  }
  auto& self = THPVariable_Unpack(self_);
  return py::cast(self.sym_storage_offset()).release().ptr();
  END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_dim(PyObject* self, PyObject* args)
{
   HANDLE_TH_ERRORS
   if (check_has_torch_function(self)) {
     return handle_torch_function(self, "dim", args);
   }
   auto& self_ = THPVariable_Unpack(self);
   return THPUtils_packInt64(self_.dim());
   END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_numel(PyObject* self, PyObject* args)
{
   HANDLE_TH_ERRORS
   if (check_has_torch_function(self)) {
     return handle_torch_function(self, "numel", args);
   }
   auto& self_ = THPVariable_Unpack(self);
   if (jit::tracer::isTracing()) {
     return wrap(jit::tracer::getNumelOf(self_));
   } else {
     return py::cast(self_.sym_numel()).release().ptr();
   }
   END_HANDLE_TH_ERRORS
}

static Tensor dispatch_contiguous(const Tensor & self, at::MemoryFormat memory_format) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.contiguous(memory_format);
}

static PyObject * THPVariable_contiguous(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "contiguous(*, MemoryFormat memory_format=contiguous_format)",
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto& self_ = THPVariable_Unpack(self);
  auto memory_format = r.memoryformat(0);
  // avoids touching the GIL or current device if self is already contiguous
  if (self_.is_contiguous(memory_format)) {
    // NOTE: this logic is duplicated from VariableType.cpp. Since we need to
    // record this call to contiguous() in the trace regardless of whether
    // we actually call contiguous here, we need to record this information
    // manually.
    if (jit::tracer::isTracing()) {
      const auto& tracer_state = jit::tracer::getTracingState();
      auto op_name = c10::Symbol::fromQualString("aten::contiguous");
      auto node = tracer_state->createNode(op_name, /*num_outputs=*/0);
      jit::tracer::recordSourceLocation(node);
      jit::tracer::addInputs(node, "self", self_);
      jit::tracer::addInputs(node, "memory_format", memory_format);
      tracer_state->insertNode(node);
      jit::tracer::addOutput(node, self_);
    }
    Py_INCREF(self);
    return self;
  }
  return THPVariable_Wrap(dispatch_contiguous(self_, memory_format));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_copy_(const Tensor & self, const Tensor & other, bool non_blocking) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.copy_(other, non_blocking);
}

 static PyObject * THPVariable_copy_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "copy_(Tensor other, bool non_blocking=False)",
    "copy_(Tensor other, bool async=False)|deprecated"
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  return THPVariable_Wrap(dispatch_copy_(self_, r.tensor(0), r.toBool(1)));
  END_HANDLE_TH_ERRORS
}

template<typename T>
static T dispatch_to(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  TORCH_CHECK_VALUE(self.sym_numel() == 1, "only one element tensors can be converted to Python scalars");
  return self.template item<T>();
}

static PyObject * THPVariable_float_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "__float__", args);
  }
  jit::tracer::warn("Converting a tensor to a Python float", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto& self_ = THPVariable_Unpack(self);
  return wrap(dispatch_to<double>(self_));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_complex_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "__complex__", args);
  }
  jit::tracer::warn("Converting a tensor to a Python complex", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto& self_ = THPVariable_Unpack(self);
  return wrap(dispatch_to<c10::complex<double>>(self_));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_integral_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "__int__", args);
  }
  jit::tracer::warn("Converting a tensor to a Python integer", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto& self_ = THPVariable_Unpack(self);
  if (isFloatingType(self_.scalar_type())) {
    // we can't dispatch to item<int64_t> here because we want to avoid ATen overflow checks;
    // the python integral type (long in python2) can't overflow.
    return THPUtils_packDoubleAsInt(dispatch_to<double>(self_));
  } else {
    return wrap(dispatch_to<int64_t>(self_));
  }
  END_HANDLE_TH_ERRORS
}

// This is the __index__ function in Python which is similar to __int__, but
// called when used as a slice.
static PyObject * THPVariable_index_scalar(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "__index__", args);
  }
  auto& self_ = THPVariable_Unpack(self);
  // TODO: change the condition to `self_.dim() != 0` once we expose scalars
  // in PyTorch.
  if (!isIntegralType(self_.scalar_type(), /*includeBool=*/true) || self_.sym_numel() != 1) {
    throw TypeError("only integer tensors of a single element can be converted to an index");
  }
  return wrap(dispatch_to<int64_t>(self_));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_invert(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.bitwise_not();
}

static PyObject * THPVariable_invert(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "__invert__", args);
  }
  auto& self_ = THPVariable_Unpack(self);
  if (!isIntegralType(self_.scalar_type(), /*includeBool=*/true)) {
    throw TypeError("~ (operator.invert) is only implemented on integer and Boolean-type tensors");
  }
  return THPVariable_Wrap(dispatch_invert(self_));
  END_HANDLE_TH_ERRORS
}

static Tensor dispatch_to(const Tensor & self, Device device, bool non_blocking, bool copy, std::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // NOTE: this is where we record aten::to in the graph during tracing. However, the behavior of aten::to
  // is different with respect to TensorOptions fields that are not present: aten::to inherits fields that
  // are missing from the self argument while the tracer assumes that they should be populated with the
  // default values (eg. float for scalar type). By explicitly copying over the tensor options here we fully
  // specify all tensor options and thus record the proper trace
  return self.to(self.options().device(device).memory_format(optional_memory_format), non_blocking, copy);
}

static Tensor dispatch_to(const Tensor & self, bool non_blocking, bool copy, std::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  return self.to(self.options().memory_format(optional_memory_format), non_blocking, copy);
}

static Tensor dispatch_to(const Tensor & self, ScalarType dtype, bool non_blocking, bool copy, std::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // TODO: Make this call the TensorOptions version, maybe?
  return self.to(dtype, non_blocking, copy, optional_memory_format);
}

static Tensor dispatch_to(const Tensor & self, Device device, ScalarType dtype, bool non_blocking, bool copy, std::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // TODO: Make this call the TensorOptions version, maybe?
  return self.to(device, dtype, non_blocking, copy, optional_memory_format);
}

static PyObject * THPVariable_cpu(PyObject* self, PyObject* args, PyObject* kwargs)
{
   HANDLE_TH_ERRORS
   static PythonArgParser parser({
     "cpu(*, MemoryFormat? memory_format=None)"
   });
   auto& self_ = THPVariable_Unpack(self);
   ParsedArgs<1> parsed_args;
   auto r = parser.parse(self, args, kwargs, parsed_args);

   if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
    }

   auto opt_memory_format = r.memoryformatOptional(0);
   return THPVariable_Wrap(dispatch_to(self_, at::Device(at::DeviceType::CPU), false, false, opt_memory_format));
   END_HANDLE_TH_ERRORS
}

static Tensor dispatch_nonzero(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.nonzero();
}

static std::vector<Tensor> dispatch_nonzero_numpy(const Tensor & self) {
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.nonzero_numpy();
}

static PyObject * THPVariable_nonzero(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nonzero()",
    "nonzero(*, bool as_tuple)",
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  if (r.idx == 0 || (r.idx == 1 && !r.toBool(0))) {
    return wrap(dispatch_nonzero(self_));
  } else {
    return wrap(dispatch_nonzero_numpy(self_));
  }
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_cuda(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cuda(Device? device=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "cuda(Device? device=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto device = r.isNone(0) ? at::Device(at::DeviceType::CUDA) : r.device(0);
  auto opt_memory_format = r.memoryformatOptional(2);
  TORCH_CHECK(device.is_cuda(), "Invalid device, must be cuda device");
  torch::utils::device_lazy_init(at::kCUDA);
  return THPVariable_Wrap(dispatch_to(self_, device, r.toBool(1), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_mtia(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mtia(Device? device=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "mtia(Device? device=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if (r.has_torch_function()) {
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto device = r.isNone(0) ? at::Device(at::DeviceType::MTIA) : r.device(0);
  auto opt_memory_format = r.memoryformatOptional(2);
  TORCH_CHECK(device.is_mtia(), "Invalid device, must be MTIA device");
  torch::utils::device_lazy_init(at::kMTIA);
  return THPVariable_Wrap(dispatch_to(self_, device, r.toBool(1), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_xpu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "xpu(Device? device=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "xpu(Device? device=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if (r.has_torch_function()) {
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto device = r.isNone(0) ? at::Device(at::DeviceType::XPU) : r.device(0);
  auto opt_memory_format = r.memoryformatOptional(2);
  TORCH_CHECK(device.is_xpu(), "Invalid device, must be xpu device");
  torch::utils::device_lazy_init(at::kXPU);
  return THPVariable_Wrap(dispatch_to(self_, device, r.toBool(1), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_ipu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ipu(Device? device=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "ipu(Device? device=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if (r.has_torch_function()) {
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto device = r.isNone(0) ? at::Device(at::DeviceType::IPU) : r.device(0);
  auto opt_memory_format = r.memoryformatOptional(2);
  TORCH_CHECK(device.is_ipu(), "Invalid device, must be ipu device");
  return THPVariable_Wrap(dispatch_to(self_, device, r.toBool(1), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_to_type(PyObject* self, ScalarType scalarType, std::optional<c10::MemoryFormat> optional_memory_format) {
  HANDLE_TH_ERRORS
  auto& self_ = THPVariable_Unpack(self);
  return THPVariable_Wrap(dispatch_to(self_, scalarType, false, false, optional_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_byte(PyObject* self, PyObject* args, PyObject* kwargs)  {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "byte(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Byte, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_char(PyObject* self, PyObject* args, PyObject* kwargs)  {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "char(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Char, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_double(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "double(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Double, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_float(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "float(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Float, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_cdouble(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cdouble(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::ComplexDouble, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_cfloat(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cfloat(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::ComplexFloat, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_half(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "half(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Half, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_int(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "int(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Int, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_long(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "long(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Long, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_short(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "short(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Short, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_bool(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bool(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::Bool, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_bfloat16(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bfloat16(*, MemoryFormat? memory_format=None)"
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto opt_memory_format = r.memoryformatOptional(0);
  return THPVariable_to_type(self, ScalarType::BFloat16, opt_memory_format);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_element_size(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "element_size", args);
  }
  auto& self_ = THPVariable_Unpack(self);
  return THPUtils_packInt64(self_.element_size());
  END_HANDLE_TH_ERRORS
}

// implemented on the python object bc PyObjects not declarable in native_functions.yaml
// See: ATen/native/README.md for more context
static PyObject * THPVariable_numpy(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "numpy(*, bool force=False)"
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if (r.has_torch_function()) {
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  jit::tracer::warn("Converting a tensor to a NumPy array", jit::tracer::WARN_PYTHON_DATAFLOW);
  return torch::utils::tensor_to_numpy(self_, r.toBool(0));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_requires_grad_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "requires_grad_(bool requires_grad=True)",
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  // temporary hack to improve functorch UX.
  const auto& functorch_tls = at::functorch::functorchTLSAccessor();
  if (functorch_tls) {
    functorch_tls->checkSupportsInplaceRequiresGrad();
  }

  auto requires_grad = r.toBool(0);
  // should we throw if requires_grad is true?  var.requires_grad = True throws here
  // but it's nice to let this be a no-op.
  if (!self_.is_leaf() && !requires_grad) {
    throw std::runtime_error(autograd::utils::requires_grad_leaf_error(requires_grad));
  }
  if (requires_grad && ! isDifferentiableType(at::typeMetaToScalarType(self_.dtype()))) {
    throw std::runtime_error("only Tensors of floating point dtype can require gradients");
  }
  self_.set_requires_grad(requires_grad);
  return THPVariable_Wrap(self_);
  END_HANDLE_TH_ERRORS
}

inline bool dispatch_is_contiguous(const Tensor & self, MemoryFormat memory_format) {
  return self.is_contiguous(memory_format);
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_is_contiguous(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_contiguous(*, MemoryFormat memory_format=contiguous_format)",
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self_, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self_, args, kwargs, PyObject_Type(self_), "torch.Tensor");
  }

  auto memory_format = r.memoryformat(0);
  auto& self = THPVariable_Unpack(self_);
  return wrap(dispatch_is_contiguous(self, memory_format));
  END_HANDLE_TH_ERRORS
}

// implemented on the python object to avoid dispatch overhead
static PyObject * THPVariable_item(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "item", args);
  }
  jit::tracer::warn("Converting a tensor to a Python number", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto& self_ = THPVariable_Unpack(self);
  auto dispatch_item_ = [](const Tensor& self) -> at::Scalar {
    pybind11::gil_scoped_release no_gil;
    return self.item();
  };
  return py::cast(dispatch_item_(self_)).release().ptr();
  END_HANDLE_TH_ERRORS
}

// implemented on the python object bc no support for first class functions in native_functions.yaml
// See: ATen/native/README.md for more context
static PyObject * THPVariable_map_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({ "map_(Tensor other, PyObject* callable)" });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  Variable other = r.tensor(0);
  if (self_.requires_grad() || other.requires_grad()) {
    throw std::runtime_error(
        "Can't call map_() on Variable that requires grad. Use "
        "var.detach().map_() instead.");
  }
  TORCH_CHECK(
      !self_.unsafeGetTensorImpl()->is_python_dispatch() && !other.unsafeGetTensorImpl()->is_python_dispatch(),
      ".map_ is not supported for tensor subclasses.");

  return THPVariable_Wrap(torch::utils::map_(self_, other, r.pyobject(1)));
  END_HANDLE_TH_ERRORS
}

// implemented on the python object bc no support for first class functions in native_functions.yaml
// See: ATen/native/README.md for more context
static PyObject * THPVariable_map2_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({ "map2_(Tensor x, Tensor y, PyObject* callable)" });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  Variable x = r.tensor(0);
  Variable y = r.tensor(1);
  if (self_.requires_grad() || x.requires_grad() || y.requires_grad()) {
    throw std::runtime_error(
        "Can't call map2_() on Variable that requires grad. Use "
        "var.detach().map2_() instead.");
  }
  TORCH_CHECK(
      !x.unsafeGetTensorImpl()->is_python_dispatch() && !y.unsafeGetTensorImpl()->is_python_dispatch(),
      ".map2_ is not supported for tensor subclasses.");
  return THPVariable_Wrap(torch::utils::map2_(self_, x, y, r.pyobject(2)));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "new", args, kwargs);
  }
  auto& self_ = THPVariable_Unpack(self);
  OptionalDeviceGuard device_guard(device_of(self_));
  return THPVariable_Wrap(torch::utils::legacy_tensor_new(legacyExtractDispatchKey(self_), self_.scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_new_tensor(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "new_tensor", args, kwargs);
  }
  auto& self_ = THPVariable_Unpack(self);
  OptionalDeviceGuard device_guard(device_of(self_));
  return THPVariable_Wrap(torch::utils::new_tensor(legacyExtractDispatchKey(self_), self_.scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_storage(PyObject* self, PyObject* arg)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "untyped_storage");
  }
  auto& self_ = THPVariable_Unpack(self);
  return createPyObject(self_.storage());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_to(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "to(Device device=None, ScalarType dtype=None, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
    "to(ScalarType dtype, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
    "to(Tensor tensor, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
  });
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);
  if (r.has_torch_function()) {
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  auto parsed = parse_to_conversion(r, /*allow_copy*/ true);
  auto& device = std::get<0>(parsed);
  auto& scalarType = std::get<1>(parsed);
  auto non_blocking = std::get<2>(parsed);
  auto copy = std::get<3>(parsed);
  auto opt_memory_format = std::get<4>(parsed);
  auto& self_ = THPVariable_Unpack(self);
  torch::utils::maybe_initialize_device(device);
  if (!device && !scalarType && !copy && !opt_memory_format.has_value()) {
    Py_INCREF(self);
    return self;
  } else if (!device && !scalarType) {
    return THPVariable_Wrap(
        dispatch_to(self_, non_blocking, copy, opt_memory_format));
  } else if (!device) {
    return THPVariable_Wrap(dispatch_to(self_, *scalarType, non_blocking, copy, opt_memory_format));
  } else if (!scalarType) {
    return THPVariable_Wrap(dispatch_to(self_, *device, non_blocking, copy, opt_memory_format));
  } else {
    return THPVariable_Wrap(dispatch_to(self_, *device, *scalarType, non_blocking, copy, opt_memory_format));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// implemented on the python object b/c arbitrarily nested list not declarable in native_functions.yaml
// See: ATen/native/README.md for more context
static PyObject * THPVariable_tolist(PyObject* self, PyObject* args)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function(self, "tolist", args);
  }
  jit::tracer::warn("Converting a tensor to a Python list", jit::tracer::WARN_PYTHON_DATAFLOW);
  auto self_ = THPVariable_Unpack(self);
  return torch::utils::tensor_to_list(self_);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_type(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "type(PyObject* dtype=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "type(PyObject* dtype=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  auto& self_ = THPVariable_Unpack(self);
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);

  if(r.has_torch_function()){
    return handle_torch_function(r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  if (r.isNone(0)) {
    return THPUtils_packString(torch::utils::options_to_string(self_.options()));
  }
  auto obj = r.pyobject(0);
  auto opt_memory_format = r.memoryformatOptional(2);
  std::string type_name;
  bool is_dtype = false;
  if (PyType_Check(obj)) {
    if (obj == THPVariableClass) {
      type_name = "torch.Tensor";
    } else {
      type_name = ((PyTypeObject*)obj)->tp_name;
    }
  } else if (THPUtils_checkString(obj)) {
    type_name = THPUtils_unpackString(obj);
  } else if (THPDtype_Check(obj)) {
    is_dtype = true;
  } else {
    throw TypeError("dtype must be a type, str, or dtype object");
  }
  Device device = self_.device();
  if (is_dtype) {
    auto scalar_type = r.scalartype(0);
    return THPVariable_Wrap(dispatch_to(self_, scalar_type, /*non_blocking=*/ r.toBool(1), /*copy=*/ false, opt_memory_format));
  }
  at::TensorOptions options = torch::utils::options_from_string(type_name);
  auto scalar_type = at::typeMetaToScalarType(options.dtype());
  auto device_type = options.device().type();
  if (device_type != device.type()) {
    device = at::Device(device_type);
  }
  torch::utils::maybe_initialize_device(device);
  return THPVariable_Wrap(dispatch_to(self_, device, scalar_type, /*non_blocking=*/ r.toBool(1), /*copy=*/ false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

// generated methods start here

\
// __and__
static PyObject * THPVariable___and__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "__and__(Tensor other)",
    "__and__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch___and__ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__and__(other);
      };
      return wrap(dispatch___and__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch___and__ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__and__(other);
      };
      return wrap(dispatch___and__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __iand__
static PyObject * THPVariable___iand__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "__iand__(Tensor other)",
    "__iand__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::__iand__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch___iand__ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__iand__(other);
      };
      return wrap(dispatch___iand__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__iand__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch___iand__ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__iand__(other);
      };
      return wrap(dispatch___iand__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __ilshift__
static PyObject * THPVariable___ilshift__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "__ilshift__(Tensor other)",
    "__ilshift__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::__ilshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch___ilshift__ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__ilshift__(other);
      };
      return wrap(dispatch___ilshift__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__ilshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch___ilshift__ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__ilshift__(other);
      };
      return wrap(dispatch___ilshift__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __ior__
static PyObject * THPVariable___ior__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "__ior__(Tensor other)",
    "__ior__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::__ior__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch___ior__ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__ior__(other);
      };
      return wrap(dispatch___ior__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__ior__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch___ior__ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__ior__(other);
      };
      return wrap(dispatch___ior__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __irshift__
static PyObject * THPVariable___irshift__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "__irshift__(Tensor other)",
    "__irshift__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::__irshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch___irshift__ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__irshift__(other);
      };
      return wrap(dispatch___irshift__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__irshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch___irshift__ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__irshift__(other);
      };
      return wrap(dispatch___irshift__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __ixor__
static PyObject * THPVariable___ixor__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "__ixor__(Tensor other)",
    "__ixor__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::__ixor__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch___ixor__ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__ixor__(other);
      };
      return wrap(dispatch___ixor__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__ixor__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch___ixor__ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__ixor__(other);
      };
      return wrap(dispatch___ixor__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __lshift__
static PyObject * THPVariable___lshift__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "__lshift__(Tensor other)",
    "__lshift__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::__lshift__.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch___lshift__ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__lshift__(other);
      };
      return wrap(dispatch___lshift__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch___lshift__ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__lshift__(other);
      };
      return wrap(dispatch___lshift__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __or__
static PyObject * THPVariable___or__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "__or__(Tensor other)",
    "__or__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::__or__.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch___or__ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__or__(other);
      };
      return wrap(dispatch___or__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__or__.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch___or__ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__or__(other);
      };
      return wrap(dispatch___or__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __rshift__
static PyObject * THPVariable___rshift__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "__rshift__(Tensor other)",
    "__rshift__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::__rshift__.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch___rshift__ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__rshift__(other);
      };
      return wrap(dispatch___rshift__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__rshift__.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch___rshift__ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__rshift__(other);
      };
      return wrap(dispatch___rshift__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __xor__
static PyObject * THPVariable___xor__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "__xor__(Tensor other)",
    "__xor__(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::__xor__.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch___xor__ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__xor__(other);
      };
      return wrap(dispatch___xor__(self, _r.tensor(0)));
    }
    case 1: {
      // aten::__xor__.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch___xor__ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__xor__(other);
      };
      return wrap(dispatch___xor__(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _addmm_activation
static PyObject * THPVariable__addmm_activation(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "_addmm_activation(Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::_addmm_activation(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False) -> Tensor
  
  auto dispatch__addmm_activation = [](const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, bool use_gelu) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._addmm_activation(mat1, mat2, beta, alpha, use_gelu);
  };
  return wrap(dispatch__addmm_activation(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _autocast_to_full_precision
static PyObject * THPVariable__autocast_to_full_precision(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "_autocast_to_full_precision(bool cuda_enabled, bool cpu_enabled)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::_autocast_to_full_precision(Tensor(a) self, bool cuda_enabled, bool cpu_enabled) -> Tensor(a)
  
  auto dispatch__autocast_to_full_precision = [](const at::Tensor & self, bool cuda_enabled, bool cpu_enabled) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._autocast_to_full_precision(cuda_enabled, cpu_enabled);
  };
  return wrap(dispatch__autocast_to_full_precision(self, _r.toBool(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _autocast_to_reduced_precision
static PyObject * THPVariable__autocast_to_reduced_precision(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "_autocast_to_reduced_precision(bool cuda_enabled, bool cpu_enabled, ScalarType cuda_dtype, ScalarType cpu_dtype)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::_autocast_to_reduced_precision(Tensor(a) self, bool cuda_enabled, bool cpu_enabled, ScalarType cuda_dtype, ScalarType cpu_dtype) -> Tensor(a)
  
  auto dispatch__autocast_to_reduced_precision = [](const at::Tensor & self, bool cuda_enabled, bool cpu_enabled, at::ScalarType cuda_dtype, at::ScalarType cpu_dtype) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._autocast_to_reduced_precision(cuda_enabled, cpu_enabled, cuda_dtype, cpu_dtype);
  };
  return wrap(dispatch__autocast_to_reduced_precision(self, _r.toBool(0), _r.toBool(1), _r.scalartype(2), _r.scalartype(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _coalesced_
static PyObject * THPVariable__coalesced_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "_coalesced_(bool coalesced)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::_coalesced_(Tensor(a!) self, bool coalesced) -> Tensor(a!)
  
  auto dispatch__coalesced_ = [](const at::Tensor & self, bool coalesced) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._coalesced_(coalesced);
  };
  return wrap(dispatch__coalesced_(self, _r.toBool(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _conj
static PyObject * THPVariable__conj(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "_conj");
  }
  // aten::_conj(Tensor(a) self) -> Tensor(a)
  
  auto dispatch__conj = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._conj();
  };
  return wrap(dispatch__conj(self));
  END_HANDLE_TH_ERRORS
}

// _conj_physical
static PyObject * THPVariable__conj_physical(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "_conj_physical");
  }
  // aten::_conj_physical(Tensor self) -> Tensor
  
  auto dispatch__conj_physical = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._conj_physical();
  };
  return wrap(dispatch__conj_physical(self));
  END_HANDLE_TH_ERRORS
}

// _dimI
static PyObject * THPVariable__dimI(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "_dimI");
  }
  // aten::_dimI(Tensor self) -> int
  
  auto dispatch__dimI = [](const at::Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return self._dimI();
  };
  return wrap(dispatch__dimI(self));
  END_HANDLE_TH_ERRORS
}

// _dimV
static PyObject * THPVariable__dimV(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "_dimV");
  }
  // aten::_dimV(Tensor self) -> int
  
  auto dispatch__dimV = [](const at::Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return self._dimV();
  };
  return wrap(dispatch__dimV(self));
  END_HANDLE_TH_ERRORS
}

// _indices
static PyObject * THPVariable__indices(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "_indices");
  }
  // aten::_indices(Tensor(a) self) -> Tensor(a)
  
  auto dispatch__indices = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._indices();
  };
  return wrap(dispatch__indices(self));
  END_HANDLE_TH_ERRORS
}

// _is_all_true
static PyObject * THPVariable__is_all_true(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "_is_all_true");
  }
  // aten::_is_all_true(Tensor self) -> Tensor
  
  auto dispatch__is_all_true = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._is_all_true();
  };
  return wrap(dispatch__is_all_true(self));
  END_HANDLE_TH_ERRORS
}

// _is_any_true
static PyObject * THPVariable__is_any_true(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "_is_any_true");
  }
  // aten::_is_any_true(Tensor self) -> Tensor
  
  auto dispatch__is_any_true = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._is_any_true();
  };
  return wrap(dispatch__is_any_true(self));
  END_HANDLE_TH_ERRORS
}

// _is_zerotensor
static PyObject * THPVariable__is_zerotensor(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "_is_zerotensor");
  }
  // aten::_is_zerotensor(Tensor self) -> bool
  
  auto dispatch__is_zerotensor = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self._is_zerotensor();
  };
  return wrap(dispatch__is_zerotensor(self));
  END_HANDLE_TH_ERRORS
}

// _lazy_clone
static PyObject * THPVariable__lazy_clone(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "_lazy_clone");
  }
  // aten::_lazy_clone(Tensor self) -> Tensor
  
  auto dispatch__lazy_clone = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._lazy_clone();
  };
  return wrap(dispatch__lazy_clone(self));
  END_HANDLE_TH_ERRORS
}

// _neg_view
static PyObject * THPVariable__neg_view(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "_neg_view");
  }
  // aten::_neg_view(Tensor(a) self) -> Tensor(a)
  
  auto dispatch__neg_view = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._neg_view();
  };
  return wrap(dispatch__neg_view(self));
  END_HANDLE_TH_ERRORS
}

// _nested_tensor_size
static PyObject * THPVariable__nested_tensor_size(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "_nested_tensor_size");
  }
  // aten::_nested_tensor_size(Tensor self) -> Tensor
  
  auto dispatch__nested_tensor_size = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._nested_tensor_size();
  };
  return wrap(dispatch__nested_tensor_size(self));
  END_HANDLE_TH_ERRORS
}

// _nested_tensor_storage_offsets
static PyObject * THPVariable__nested_tensor_storage_offsets(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "_nested_tensor_storage_offsets");
  }
  // aten::_nested_tensor_storage_offsets(Tensor self) -> Tensor
  
  auto dispatch__nested_tensor_storage_offsets = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._nested_tensor_storage_offsets();
  };
  return wrap(dispatch__nested_tensor_storage_offsets(self));
  END_HANDLE_TH_ERRORS
}

// _nested_tensor_strides
static PyObject * THPVariable__nested_tensor_strides(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "_nested_tensor_strides");
  }
  // aten::_nested_tensor_strides(Tensor self) -> Tensor
  
  auto dispatch__nested_tensor_strides = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._nested_tensor_strides();
  };
  return wrap(dispatch__nested_tensor_strides(self));
  END_HANDLE_TH_ERRORS
}

// _nnz
static PyObject * THPVariable__nnz(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "_nnz");
  }
  // aten::_nnz(Tensor self) -> int
  
  auto dispatch__nnz = [](const at::Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return self._nnz();
  };
  return wrap(dispatch__nnz(self));
  END_HANDLE_TH_ERRORS
}

// _sparse_mask_projection
static PyObject * THPVariable__sparse_mask_projection(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "_sparse_mask_projection(Tensor mask, bool accumulate_matches=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::_sparse_mask_projection(Tensor self, Tensor mask, bool accumulate_matches=False) -> Tensor
  
  auto dispatch__sparse_mask_projection = [](const at::Tensor & self, const at::Tensor & mask, bool accumulate_matches) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._sparse_mask_projection(mask, accumulate_matches);
  };
  return wrap(dispatch__sparse_mask_projection(self, _r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _to_dense
static PyObject * THPVariable__to_dense(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "_to_dense(ScalarType? dtype=None, bool? masked_grad=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::_to_dense(Tensor self, ScalarType? dtype=None, bool? masked_grad=None) -> Tensor
  
  auto dispatch__to_dense = [](const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<bool> masked_grad) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._to_dense(dtype, masked_grad);
  };
  return wrap(dispatch__to_dense(self, _r.scalartypeOptional(0), _r.toBoolOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _to_sparse
static PyObject * THPVariable__to_sparse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "_to_sparse(*, Layout? layout=None, IntArrayRef[2]? blocksize=None, int64_t? dense_dim=None)",
    "_to_sparse(int64_t sparse_dim)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_to_sparse(Tensor self, *, Layout? layout=None, int[2]? blocksize=None, int? dense_dim=None) -> Tensor
      
      auto dispatch__to_sparse = [](const at::Tensor & self, ::std::optional<at::Layout> layout, at::OptionalIntArrayRef blocksize, ::std::optional<int64_t> dense_dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self._to_sparse(layout, blocksize, dense_dim);
      };
      return wrap(dispatch__to_sparse(self, _r.layoutOptional(0), _r.intlistOptional(1), _r.toInt64Optional(2)));
    }
    case 1: {
      // aten::_to_sparse.sparse_dim(Tensor self, int sparse_dim) -> Tensor
      
      auto dispatch__to_sparse = [](const at::Tensor & self, int64_t sparse_dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self._to_sparse(sparse_dim);
      };
      return wrap(dispatch__to_sparse(self, _r.toInt64(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _to_sparse_bsc
static PyObject * THPVariable__to_sparse_bsc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "_to_sparse_bsc(IntArrayRef[2] blocksize, int64_t? dense_dim=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::_to_sparse_bsc(Tensor self, int[2] blocksize, int? dense_dim=None) -> Tensor
  
  auto dispatch__to_sparse_bsc = [](const at::Tensor & self, at::IntArrayRef blocksize, ::std::optional<int64_t> dense_dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._to_sparse_bsc(blocksize, dense_dim);
  };
  return wrap(dispatch__to_sparse_bsc(self, _r.intlist(0), _r.toInt64Optional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _to_sparse_bsr
static PyObject * THPVariable__to_sparse_bsr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "_to_sparse_bsr(IntArrayRef[2] blocksize, int64_t? dense_dim=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::_to_sparse_bsr(Tensor self, int[2] blocksize, int? dense_dim=None) -> Tensor
  
  auto dispatch__to_sparse_bsr = [](const at::Tensor & self, at::IntArrayRef blocksize, ::std::optional<int64_t> dense_dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._to_sparse_bsr(blocksize, dense_dim);
  };
  return wrap(dispatch__to_sparse_bsr(self, _r.intlist(0), _r.toInt64Optional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _to_sparse_csc
static PyObject * THPVariable__to_sparse_csc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "_to_sparse_csc(int64_t? dense_dim=None)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::_to_sparse_csc(Tensor self, int? dense_dim=None) -> Tensor
  
  auto dispatch__to_sparse_csc = [](const at::Tensor & self, ::std::optional<int64_t> dense_dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._to_sparse_csc(dense_dim);
  };
  return wrap(dispatch__to_sparse_csc(self, _r.toInt64Optional(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _to_sparse_csr
static PyObject * THPVariable__to_sparse_csr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "_to_sparse_csr(int64_t? dense_dim=None)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::_to_sparse_csr(Tensor self, int? dense_dim=None) -> Tensor
  
  auto dispatch__to_sparse_csr = [](const at::Tensor & self, ::std::optional<int64_t> dense_dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._to_sparse_csr(dense_dim);
  };
  return wrap(dispatch__to_sparse_csr(self, _r.toInt64Optional(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _values
static PyObject * THPVariable__values(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "_values");
  }
  // aten::_values(Tensor(a) self) -> Tensor(a)
  
  auto dispatch__values = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._values();
  };
  return wrap(dispatch__values(self));
  END_HANDLE_TH_ERRORS
}

// abs
static PyObject * THPVariable_abs(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "abs");
  }
  // aten::abs(Tensor self) -> Tensor
  
  auto dispatch_abs = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.abs();
  };
  return wrap(dispatch_abs(self));
  END_HANDLE_TH_ERRORS
}

// abs_
static PyObject * THPVariable_abs_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "abs_");
  }
  // aten::abs_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_abs_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.abs_();
  };
  return wrap(dispatch_abs_(self));
  END_HANDLE_TH_ERRORS
}

// absolute
static PyObject * THPVariable_absolute(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "absolute");
  }
  // aten::absolute(Tensor self) -> Tensor
  
  auto dispatch_absolute = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.absolute();
  };
  return wrap(dispatch_absolute(self));
  END_HANDLE_TH_ERRORS
}

// absolute_
static PyObject * THPVariable_absolute_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "absolute_");
  }
  // aten::absolute_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_absolute_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.absolute_();
  };
  return wrap(dispatch_absolute_(self));
  END_HANDLE_TH_ERRORS
}

// acos
static PyObject * THPVariable_acos(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "acos");
  }
  // aten::acos(Tensor self) -> Tensor
  
  auto dispatch_acos = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.acos();
  };
  return wrap(dispatch_acos(self));
  END_HANDLE_TH_ERRORS
}

// acos_
static PyObject * THPVariable_acos_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "acos_");
  }
  // aten::acos_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_acos_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.acos_();
  };
  return wrap(dispatch_acos_(self));
  END_HANDLE_TH_ERRORS
}

// acosh
static PyObject * THPVariable_acosh(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "acosh");
  }
  // aten::acosh(Tensor self) -> Tensor
  
  auto dispatch_acosh = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.acosh();
  };
  return wrap(dispatch_acosh(self));
  END_HANDLE_TH_ERRORS
}

// acosh_
static PyObject * THPVariable_acosh_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "acosh_");
  }
  // aten::acosh_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_acosh_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.acosh_();
  };
  return wrap(dispatch_acosh_(self));
  END_HANDLE_TH_ERRORS
}

\
// add
static PyObject * THPVariable_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "add(Scalar alpha, Tensor other)|deprecated",
    "add(Tensor other, *, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::add(Tensor self, Scalar alpha, Tensor other) -> Tensor
      
      auto dispatch_add = [](const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.add(other, alpha);
      };
      return wrap(dispatch_add(self, _r.scalar(0), _r.tensor(1)));
    }
    case 1: {
      // aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
      
      auto dispatch_add = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.add(other, alpha);
      };
      return wrap(dispatch_add(self, _r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// add_
static PyObject * THPVariable_add_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "add_(Scalar alpha, Tensor other)|deprecated",
    "add_(Tensor other, *, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::add_(Tensor(a!) self, Scalar alpha, Tensor other) -> Tensor(a!)
      
      auto dispatch_add_ = [](const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.add_(other, alpha);
      };
      return wrap(dispatch_add_(self, _r.scalar(0), _r.tensor(1)));
    }
    case 1: {
      // aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
      
      auto dispatch_add_ = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.add_(other, alpha);
      };
      return wrap(dispatch_add_(self, _r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addbmm
static PyObject * THPVariable_addbmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "addbmm(Scalar beta, Scalar alpha, Tensor batch1, Tensor batch2)|deprecated",
    "addbmm(Scalar beta, Tensor batch1, Tensor batch2)|deprecated",
    "addbmm(Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addbmm(Scalar beta, Tensor self, Scalar alpha, Tensor batch1, Tensor batch2) -> Tensor
      
      auto dispatch_addbmm = [](const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & batch1, const at::Tensor & batch2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addbmm(batch1, batch2, beta, alpha);
      };
      return wrap(dispatch_addbmm(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::addbmm(Scalar beta, Tensor self, Tensor batch1, Tensor batch2) -> Tensor
      
      auto dispatch_addbmm = [](const at::Scalar & beta, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addbmm(batch1, batch2, beta, 1);
      };
      return wrap(dispatch_addbmm(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      
      auto dispatch_addbmm = [](const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addbmm(batch1, batch2, beta, alpha);
      };
      return wrap(dispatch_addbmm(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addbmm_
static PyObject * THPVariable_addbmm_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "addbmm_(Scalar beta, Scalar alpha, Tensor batch1, Tensor batch2)|deprecated",
    "addbmm_(Scalar beta, Tensor batch1, Tensor batch2)|deprecated",
    "addbmm_(Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addbmm_(Scalar beta, Tensor(a!) self, Scalar alpha, Tensor batch1, Tensor batch2) -> Tensor(a!)
      
      auto dispatch_addbmm_ = [](const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & batch1, const at::Tensor & batch2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addbmm_(batch1, batch2, beta, alpha);
      };
      return wrap(dispatch_addbmm_(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::addbmm_(Scalar beta, Tensor(a!) self, Tensor batch1, Tensor batch2) -> Tensor(a!)
      
      auto dispatch_addbmm_ = [](const at::Scalar & beta, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addbmm_(batch1, batch2, beta, 1);
      };
      return wrap(dispatch_addbmm_(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      
      auto dispatch_addbmm_ = [](const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addbmm_(batch1, batch2, beta, alpha);
      };
      return wrap(dispatch_addbmm_(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addcdiv
static PyObject * THPVariable_addcdiv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "addcdiv(Scalar value, Tensor tensor1, Tensor tensor2)|deprecated",
    "addcdiv(Tensor tensor1, Tensor tensor2, *, Scalar value=1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addcdiv(Tensor self, Scalar value, Tensor tensor1, Tensor tensor2) -> Tensor
      
      auto dispatch_addcdiv = [](const at::Tensor & self, const at::Scalar & value, const at::Tensor & tensor1, const at::Tensor & tensor2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addcdiv(tensor1, tensor2, value);
      };
      return wrap(dispatch_addcdiv(self, _r.scalar(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
      
      auto dispatch_addcdiv = [](const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addcdiv(tensor1, tensor2, value);
      };
      return wrap(dispatch_addcdiv(self, _r.tensor(0), _r.tensor(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addcdiv_
static PyObject * THPVariable_addcdiv_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "addcdiv_(Scalar value, Tensor tensor1, Tensor tensor2)|deprecated",
    "addcdiv_(Tensor tensor1, Tensor tensor2, *, Scalar value=1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addcdiv_(Tensor(a!) self, Scalar value, Tensor tensor1, Tensor tensor2) -> Tensor(a!)
      
      auto dispatch_addcdiv_ = [](const at::Tensor & self, const at::Scalar & value, const at::Tensor & tensor1, const at::Tensor & tensor2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addcdiv_(tensor1, tensor2, value);
      };
      return wrap(dispatch_addcdiv_(self, _r.scalar(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
      
      auto dispatch_addcdiv_ = [](const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addcdiv_(tensor1, tensor2, value);
      };
      return wrap(dispatch_addcdiv_(self, _r.tensor(0), _r.tensor(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addcmul
static PyObject * THPVariable_addcmul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "addcmul(Scalar value, Tensor tensor1, Tensor tensor2)|deprecated",
    "addcmul(Tensor tensor1, Tensor tensor2, *, Scalar value=1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addcmul(Tensor self, Scalar value, Tensor tensor1, Tensor tensor2) -> Tensor
      
      auto dispatch_addcmul = [](const at::Tensor & self, const at::Scalar & value, const at::Tensor & tensor1, const at::Tensor & tensor2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addcmul(tensor1, tensor2, value);
      };
      return wrap(dispatch_addcmul(self, _r.scalar(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
      
      auto dispatch_addcmul = [](const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addcmul(tensor1, tensor2, value);
      };
      return wrap(dispatch_addcmul(self, _r.tensor(0), _r.tensor(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addcmul_
static PyObject * THPVariable_addcmul_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "addcmul_(Scalar value, Tensor tensor1, Tensor tensor2)|deprecated",
    "addcmul_(Tensor tensor1, Tensor tensor2, *, Scalar value=1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addcmul_(Tensor(a!) self, Scalar value, Tensor tensor1, Tensor tensor2) -> Tensor(a!)
      
      auto dispatch_addcmul_ = [](const at::Tensor & self, const at::Scalar & value, const at::Tensor & tensor1, const at::Tensor & tensor2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addcmul_(tensor1, tensor2, value);
      };
      return wrap(dispatch_addcmul_(self, _r.scalar(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
      
      auto dispatch_addcmul_ = [](const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addcmul_(tensor1, tensor2, value);
      };
      return wrap(dispatch_addcmul_(self, _r.tensor(0), _r.tensor(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addmm
static PyObject * THPVariable_addmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "addmm(Scalar beta, Scalar alpha, Tensor mat1, Tensor mat2)|deprecated",
    "addmm(Scalar beta, Tensor mat1, Tensor mat2)|deprecated",
    "addmm(Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addmm(Scalar beta, Tensor self, Scalar alpha, Tensor mat1, Tensor mat2) -> Tensor
      
      auto dispatch_addmm = [](const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & mat1, const at::Tensor & mat2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmm(mat1, mat2, beta, alpha);
      };
      return wrap(dispatch_addmm(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::addmm(Scalar beta, Tensor self, Tensor mat1, Tensor mat2) -> Tensor
      
      auto dispatch_addmm = [](const at::Scalar & beta, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmm(mat1, mat2, beta, 1);
      };
      return wrap(dispatch_addmm(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      
      auto dispatch_addmm = [](const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmm(mat1, mat2, beta, alpha);
      };
      return wrap(dispatch_addmm(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addmm_
static PyObject * THPVariable_addmm_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "addmm_(Scalar beta, Scalar alpha, Tensor mat1, Tensor mat2)|deprecated",
    "addmm_(Scalar beta, Tensor mat1, Tensor mat2)|deprecated",
    "addmm_(Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addmm_(Scalar beta, Tensor(a!) self, Scalar alpha, Tensor mat1, Tensor mat2) -> Tensor(a!)
      
      auto dispatch_addmm_ = [](const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & mat1, const at::Tensor & mat2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmm_(mat1, mat2, beta, alpha);
      };
      return wrap(dispatch_addmm_(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::addmm_(Scalar beta, Tensor(a!) self, Tensor mat1, Tensor mat2) -> Tensor(a!)
      
      auto dispatch_addmm_ = [](const at::Scalar & beta, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmm_(mat1, mat2, beta, 1);
      };
      return wrap(dispatch_addmm_(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      
      auto dispatch_addmm_ = [](const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmm_(mat1, mat2, beta, alpha);
      };
      return wrap(dispatch_addmm_(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addmv
static PyObject * THPVariable_addmv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "addmv(Scalar beta, Scalar alpha, Tensor mat, Tensor vec)|deprecated",
    "addmv(Scalar beta, Tensor mat, Tensor vec)|deprecated",
    "addmv(Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addmv(Scalar beta, Tensor self, Scalar alpha, Tensor mat, Tensor vec) -> Tensor
      
      auto dispatch_addmv = [](const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & mat, const at::Tensor & vec) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmv(mat, vec, beta, alpha);
      };
      return wrap(dispatch_addmv(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::addmv(Scalar beta, Tensor self, Tensor mat, Tensor vec) -> Tensor
      
      auto dispatch_addmv = [](const at::Scalar & beta, const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmv(mat, vec, beta, 1);
      };
      return wrap(dispatch_addmv(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      
      auto dispatch_addmv = [](const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmv(mat, vec, beta, alpha);
      };
      return wrap(dispatch_addmv(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addmv_
static PyObject * THPVariable_addmv_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "addmv_(Scalar beta, Scalar alpha, Tensor mat, Tensor vec)|deprecated",
    "addmv_(Scalar beta, Tensor mat, Tensor vec)|deprecated",
    "addmv_(Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addmv_(Scalar beta, Tensor(a!) self, Scalar alpha, Tensor mat, Tensor vec) -> Tensor(a!)
      
      auto dispatch_addmv_ = [](const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & mat, const at::Tensor & vec) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmv_(mat, vec, beta, alpha);
      };
      return wrap(dispatch_addmv_(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::addmv_(Scalar beta, Tensor(a!) self, Tensor mat, Tensor vec) -> Tensor(a!)
      
      auto dispatch_addmv_ = [](const at::Scalar & beta, const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmv_(mat, vec, beta, 1);
      };
      return wrap(dispatch_addmv_(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      
      auto dispatch_addmv_ = [](const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmv_(mat, vec, beta, alpha);
      };
      return wrap(dispatch_addmv_(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addr
static PyObject * THPVariable_addr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "addr(Scalar beta, Scalar alpha, Tensor vec1, Tensor vec2)|deprecated",
    "addr(Scalar beta, Tensor vec1, Tensor vec2)|deprecated",
    "addr(Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addr(Scalar beta, Tensor self, Scalar alpha, Tensor vec1, Tensor vec2) -> Tensor
      
      auto dispatch_addr = [](const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & vec1, const at::Tensor & vec2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addr(vec1, vec2, beta, alpha);
      };
      return wrap(dispatch_addr(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::addr(Scalar beta, Tensor self, Tensor vec1, Tensor vec2) -> Tensor
      
      auto dispatch_addr = [](const at::Scalar & beta, const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addr(vec1, vec2, beta, 1);
      };
      return wrap(dispatch_addr(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      
      auto dispatch_addr = [](const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addr(vec1, vec2, beta, alpha);
      };
      return wrap(dispatch_addr(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addr_
static PyObject * THPVariable_addr_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "addr_(Scalar beta, Scalar alpha, Tensor vec1, Tensor vec2)|deprecated",
    "addr_(Scalar beta, Tensor vec1, Tensor vec2)|deprecated",
    "addr_(Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addr_(Scalar beta, Tensor(a!) self, Scalar alpha, Tensor vec1, Tensor vec2) -> Tensor(a!)
      
      auto dispatch_addr_ = [](const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & vec1, const at::Tensor & vec2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addr_(vec1, vec2, beta, alpha);
      };
      return wrap(dispatch_addr_(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::addr_(Scalar beta, Tensor(a!) self, Tensor vec1, Tensor vec2) -> Tensor(a!)
      
      auto dispatch_addr_ = [](const at::Scalar & beta, const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addr_(vec1, vec2, beta, 1);
      };
      return wrap(dispatch_addr_(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      
      auto dispatch_addr_ = [](const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addr_(vec1, vec2, beta, alpha);
      };
      return wrap(dispatch_addr_(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// adjoint
static PyObject * THPVariable_adjoint(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "adjoint");
  }
  // aten::adjoint(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_adjoint = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.adjoint();
  };
  return wrap(dispatch_adjoint(self));
  END_HANDLE_TH_ERRORS
}

// align_as
static PyObject * THPVariable_align_as(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "align_as(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::align_as(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_align_as = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.align_as(other);
  };
  return wrap(dispatch_align_as(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// align_to
static PyObject * THPVariable_align_to(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "align_to(DimnameList names)",
    "align_to(DimnameList order, int64_t ellipsis_idx)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::align_to(Tensor(a) self, Dimname[] names) -> Tensor(a)
      
      auto dispatch_align_to = [](const at::Tensor & self, at::DimnameList names) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.align_to(names);
      };
      return wrap(dispatch_align_to(self, _r.dimnamelist(0)));
    }
    case 1: {
      // aten::align_to.ellipsis_idx(Tensor(a) self, Dimname[] order, int ellipsis_idx) -> Tensor(a)
      
      auto dispatch_align_to = [](const at::Tensor & self, at::DimnameList order, int64_t ellipsis_idx) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.align_to(order, ellipsis_idx);
      };
      return wrap(dispatch_align_to(self, _r.dimnamelist(0), _r.toInt64(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// all
static PyObject * THPVariable_all(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "all()",
    "all(IntArrayRef? dim=None, bool keepdim=False)",
    "all(int64_t dim, bool keepdim=False)",
    "all(Dimname dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::all(Tensor self) -> Tensor
      
      auto dispatch_all = [](const at::Tensor & self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.all();
      };
      return wrap(dispatch_all(self));
    }
    case 1: {
      // aten::all.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor
      
      auto dispatch_all = [](const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.all(dim, keepdim);
      };
      return wrap(dispatch_all(self, _r.intlistOptional(0), _r.toBool(1)));
    }
    case 2: {
      // aten::all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
      
      auto dispatch_all = [](const at::Tensor & self, int64_t dim, bool keepdim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.all(dim, keepdim);
      };
      return wrap(dispatch_all(self, _r.toInt64(0), _r.toBool(1)));
    }
    case 3: {
      // aten::all.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor
      
      auto dispatch_all = [](const at::Tensor & self, at::Dimname dim, bool keepdim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.all(dim, keepdim);
      };
      return wrap(dispatch_all(self, _r.dimname(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// allclose
static PyObject * THPVariable_allclose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "allclose(Tensor other, double rtol=1e-05, double atol=1e-08, bool equal_nan=False)",
  }, /*traceable=*/false);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool
  
  auto dispatch_allclose = [](const at::Tensor & self, const at::Tensor & other, double rtol, double atol, bool equal_nan) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.allclose(other, rtol, atol, equal_nan);
  };
  return wrap(dispatch_allclose(self, _r.tensor(0), _r.toDouble(1), _r.toDouble(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// amax
static PyObject * THPVariable_amax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "amax(IntArrayRef[1] dim=None, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
  
  auto dispatch_amax = [](const at::Tensor & self, at::IntArrayRef dim, bool keepdim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.amax(dim, keepdim);
  };
  return wrap(dispatch_amax(self, _r.intlist(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// amin
static PyObject * THPVariable_amin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "amin(IntArrayRef[1] dim=None, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
  
  auto dispatch_amin = [](const at::Tensor & self, at::IntArrayRef dim, bool keepdim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.amin(dim, keepdim);
  };
  return wrap(dispatch_amin(self, _r.intlist(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// aminmax
static PyObject * THPVariable_aminmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_aminmax_structseq();
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "aminmax(*, int64_t? dim=None, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::aminmax(Tensor self, *, int? dim=None, bool keepdim=False) -> (Tensor min, Tensor max)
  
  auto dispatch_aminmax = [](const at::Tensor & self, ::std::optional<int64_t> dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.aminmax(dim, keepdim);
  };
  return wrap(NamedTuple, dispatch_aminmax(self, _r.toInt64Optional(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// angle
static PyObject * THPVariable_angle(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "angle");
  }
  // aten::angle(Tensor self) -> Tensor
  
  auto dispatch_angle = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.angle();
  };
  return wrap(dispatch_angle(self));
  END_HANDLE_TH_ERRORS
}

\
// any
static PyObject * THPVariable_any(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "any()",
    "any(IntArrayRef? dim=None, bool keepdim=False)",
    "any(int64_t dim, bool keepdim=False)",
    "any(Dimname dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::any(Tensor self) -> Tensor
      
      auto dispatch_any = [](const at::Tensor & self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.any();
      };
      return wrap(dispatch_any(self));
    }
    case 1: {
      // aten::any.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor
      
      auto dispatch_any = [](const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.any(dim, keepdim);
      };
      return wrap(dispatch_any(self, _r.intlistOptional(0), _r.toBool(1)));
    }
    case 2: {
      // aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
      
      auto dispatch_any = [](const at::Tensor & self, int64_t dim, bool keepdim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.any(dim, keepdim);
      };
      return wrap(dispatch_any(self, _r.toInt64(0), _r.toBool(1)));
    }
    case 3: {
      // aten::any.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor
      
      auto dispatch_any = [](const at::Tensor & self, at::Dimname dim, bool keepdim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.any(dim, keepdim);
      };
      return wrap(dispatch_any(self, _r.dimname(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// arccos
static PyObject * THPVariable_arccos(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "arccos");
  }
  // aten::arccos(Tensor self) -> Tensor
  
  auto dispatch_arccos = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arccos();
  };
  return wrap(dispatch_arccos(self));
  END_HANDLE_TH_ERRORS
}

// arccos_
static PyObject * THPVariable_arccos_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "arccos_");
  }
  // aten::arccos_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_arccos_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arccos_();
  };
  return wrap(dispatch_arccos_(self));
  END_HANDLE_TH_ERRORS
}

// arccosh
static PyObject * THPVariable_arccosh(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "arccosh");
  }
  // aten::arccosh(Tensor self) -> Tensor
  
  auto dispatch_arccosh = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arccosh();
  };
  return wrap(dispatch_arccosh(self));
  END_HANDLE_TH_ERRORS
}

// arccosh_
static PyObject * THPVariable_arccosh_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "arccosh_");
  }
  // aten::arccosh_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_arccosh_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arccosh_();
  };
  return wrap(dispatch_arccosh_(self));
  END_HANDLE_TH_ERRORS
}

// arcsin
static PyObject * THPVariable_arcsin(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "arcsin");
  }
  // aten::arcsin(Tensor self) -> Tensor
  
  auto dispatch_arcsin = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arcsin();
  };
  return wrap(dispatch_arcsin(self));
  END_HANDLE_TH_ERRORS
}

// arcsin_
static PyObject * THPVariable_arcsin_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "arcsin_");
  }
  // aten::arcsin_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_arcsin_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arcsin_();
  };
  return wrap(dispatch_arcsin_(self));
  END_HANDLE_TH_ERRORS
}

// arcsinh
static PyObject * THPVariable_arcsinh(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "arcsinh");
  }
  // aten::arcsinh(Tensor self) -> Tensor
  
  auto dispatch_arcsinh = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arcsinh();
  };
  return wrap(dispatch_arcsinh(self));
  END_HANDLE_TH_ERRORS
}

// arcsinh_
static PyObject * THPVariable_arcsinh_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "arcsinh_");
  }
  // aten::arcsinh_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_arcsinh_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arcsinh_();
  };
  return wrap(dispatch_arcsinh_(self));
  END_HANDLE_TH_ERRORS
}

// arctan
static PyObject * THPVariable_arctan(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "arctan");
  }
  // aten::arctan(Tensor self) -> Tensor
  
  auto dispatch_arctan = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arctan();
  };
  return wrap(dispatch_arctan(self));
  END_HANDLE_TH_ERRORS
}

// arctan2
static PyObject * THPVariable_arctan2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "arctan2(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::arctan2(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_arctan2 = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arctan2(other);
  };
  return wrap(dispatch_arctan2(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// arctan2_
static PyObject * THPVariable_arctan2_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "arctan2_(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::arctan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  
  auto dispatch_arctan2_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arctan2_(other);
  };
  return wrap(dispatch_arctan2_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// arctan_
static PyObject * THPVariable_arctan_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "arctan_");
  }
  // aten::arctan_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_arctan_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arctan_();
  };
  return wrap(dispatch_arctan_(self));
  END_HANDLE_TH_ERRORS
}

// arctanh
static PyObject * THPVariable_arctanh(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "arctanh");
  }
  // aten::arctanh(Tensor self) -> Tensor
  
  auto dispatch_arctanh = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arctanh();
  };
  return wrap(dispatch_arctanh(self));
  END_HANDLE_TH_ERRORS
}

// arctanh_
static PyObject * THPVariable_arctanh_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "arctanh_");
  }
  // aten::arctanh_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_arctanh_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arctanh_();
  };
  return wrap(dispatch_arctanh_(self));
  END_HANDLE_TH_ERRORS
}

// argmax
static PyObject * THPVariable_argmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "argmax(int64_t? dim=None, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
  
  auto dispatch_argmax = [](const at::Tensor & self, ::std::optional<int64_t> dim, bool keepdim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.argmax(dim, keepdim);
  };
  return wrap(dispatch_argmax(self, _r.toInt64Optional(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// argmin
static PyObject * THPVariable_argmin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "argmin(int64_t? dim=None, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
  
  auto dispatch_argmin = [](const at::Tensor & self, ::std::optional<int64_t> dim, bool keepdim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.argmin(dim, keepdim);
  };
  return wrap(dispatch_argmin(self, _r.toInt64Optional(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// argsort
static PyObject * THPVariable_argsort(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "argsort(*, bool stable, int64_t dim=-1, bool descending=False)",
    "argsort(int64_t dim=-1, bool descending=False)",
    "argsort(Dimname dim, bool descending=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::argsort.stable(Tensor self, *, bool stable, int dim=-1, bool descending=False) -> Tensor
      
      auto dispatch_argsort = [](const at::Tensor & self, bool stable, int64_t dim, bool descending) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.argsort(stable, dim, descending);
      };
      return wrap(dispatch_argsort(self, _r.toBool(0), _r.toInt64(1), _r.toBool(2)));
    }
    case 1: {
      // aten::argsort(Tensor self, int dim=-1, bool descending=False) -> Tensor
      
      auto dispatch_argsort = [](const at::Tensor & self, int64_t dim, bool descending) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.argsort(dim, descending);
      };
      return wrap(dispatch_argsort(self, _r.toInt64(0), _r.toBool(1)));
    }
    case 2: {
      // aten::argsort.dimname(Tensor self, Dimname dim, bool descending=False) -> Tensor
      
      auto dispatch_argsort = [](const at::Tensor & self, at::Dimname dim, bool descending) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.argsort(dim, descending);
      };
      return wrap(dispatch_argsort(self, _r.dimname(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// argwhere
static PyObject * THPVariable_argwhere(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "argwhere");
  }
  // aten::argwhere(Tensor self) -> Tensor
  
  auto dispatch_argwhere = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.argwhere();
  };
  return wrap(dispatch_argwhere(self));
  END_HANDLE_TH_ERRORS
}

// as_strided
static PyObject * THPVariable_as_strided(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "as_strided(SymIntArrayRef size, SymIntArrayRef stride, SymInt? storage_offset=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)
  
  auto dispatch_as_strided = [](const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<c10::SymInt> storage_offset) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.as_strided_symint(size, stride, storage_offset);
  };
  return wrap(dispatch_as_strided(self, _r.symintlist(0), _r.symintlist(1), _r.toSymIntOptional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// as_strided_
static PyObject * THPVariable_as_strided_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "as_strided_(SymIntArrayRef size, SymIntArrayRef stride, SymInt? storage_offset=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::as_strided_(Tensor(a!) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a!)
  
  auto dispatch_as_strided_ = [](const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<c10::SymInt> storage_offset) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.as_strided__symint(size, stride, storage_offset);
  };
  return wrap(dispatch_as_strided_(self, _r.symintlist(0), _r.symintlist(1), _r.toSymIntOptional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// as_strided_scatter
static PyObject * THPVariable_as_strided_scatter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "as_strided_scatter(Tensor src, SymIntArrayRef size, SymIntArrayRef stride, SymInt? storage_offset=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::as_strided_scatter(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor
  
  auto dispatch_as_strided_scatter = [](const at::Tensor & self, const at::Tensor & src, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<c10::SymInt> storage_offset) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.as_strided_scatter_symint(src, size, stride, storage_offset);
  };
  return wrap(dispatch_as_strided_scatter(self, _r.tensor(0), _r.symintlist(1), _r.symintlist(2), _r.toSymIntOptional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// asin
static PyObject * THPVariable_asin(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "asin");
  }
  // aten::asin(Tensor self) -> Tensor
  
  auto dispatch_asin = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.asin();
  };
  return wrap(dispatch_asin(self));
  END_HANDLE_TH_ERRORS
}

// asin_
static PyObject * THPVariable_asin_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "asin_");
  }
  // aten::asin_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_asin_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.asin_();
  };
  return wrap(dispatch_asin_(self));
  END_HANDLE_TH_ERRORS
}

// asinh
static PyObject * THPVariable_asinh(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "asinh");
  }
  // aten::asinh(Tensor self) -> Tensor
  
  auto dispatch_asinh = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.asinh();
  };
  return wrap(dispatch_asinh(self));
  END_HANDLE_TH_ERRORS
}

// asinh_
static PyObject * THPVariable_asinh_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "asinh_");
  }
  // aten::asinh_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_asinh_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.asinh_();
  };
  return wrap(dispatch_asinh_(self));
  END_HANDLE_TH_ERRORS
}

// atan
static PyObject * THPVariable_atan(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "atan");
  }
  // aten::atan(Tensor self) -> Tensor
  
  auto dispatch_atan = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.atan();
  };
  return wrap(dispatch_atan(self));
  END_HANDLE_TH_ERRORS
}

// atan2
static PyObject * THPVariable_atan2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "atan2(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::atan2(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_atan2 = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.atan2(other);
  };
  return wrap(dispatch_atan2(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// atan2_
static PyObject * THPVariable_atan2_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "atan2_(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  
  auto dispatch_atan2_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.atan2_(other);
  };
  return wrap(dispatch_atan2_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// atan_
static PyObject * THPVariable_atan_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "atan_");
  }
  // aten::atan_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_atan_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.atan_();
  };
  return wrap(dispatch_atan_(self));
  END_HANDLE_TH_ERRORS
}

// atanh
static PyObject * THPVariable_atanh(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "atanh");
  }
  // aten::atanh(Tensor self) -> Tensor
  
  auto dispatch_atanh = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.atanh();
  };
  return wrap(dispatch_atanh(self));
  END_HANDLE_TH_ERRORS
}

// atanh_
static PyObject * THPVariable_atanh_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "atanh_");
  }
  // aten::atanh_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_atanh_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.atanh_();
  };
  return wrap(dispatch_atanh_(self));
  END_HANDLE_TH_ERRORS
}

\
// baddbmm
static PyObject * THPVariable_baddbmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "baddbmm(Scalar beta, Scalar alpha, Tensor batch1, Tensor batch2)|deprecated",
    "baddbmm(Scalar beta, Tensor batch1, Tensor batch2)|deprecated",
    "baddbmm(Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::baddbmm(Scalar beta, Tensor self, Scalar alpha, Tensor batch1, Tensor batch2) -> Tensor
      
      auto dispatch_baddbmm = [](const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & batch1, const at::Tensor & batch2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.baddbmm(batch1, batch2, beta, alpha);
      };
      return wrap(dispatch_baddbmm(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::baddbmm(Scalar beta, Tensor self, Tensor batch1, Tensor batch2) -> Tensor
      
      auto dispatch_baddbmm = [](const at::Scalar & beta, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.baddbmm(batch1, batch2, beta, 1);
      };
      return wrap(dispatch_baddbmm(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      
      auto dispatch_baddbmm = [](const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.baddbmm(batch1, batch2, beta, alpha);
      };
      return wrap(dispatch_baddbmm(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// baddbmm_
static PyObject * THPVariable_baddbmm_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "baddbmm_(Scalar beta, Scalar alpha, Tensor batch1, Tensor batch2)|deprecated",
    "baddbmm_(Scalar beta, Tensor batch1, Tensor batch2)|deprecated",
    "baddbmm_(Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::baddbmm_(Scalar beta, Tensor(a!) self, Scalar alpha, Tensor batch1, Tensor batch2) -> Tensor(a!)
      
      auto dispatch_baddbmm_ = [](const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & batch1, const at::Tensor & batch2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.baddbmm_(batch1, batch2, beta, alpha);
      };
      return wrap(dispatch_baddbmm_(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::baddbmm_(Scalar beta, Tensor(a!) self, Tensor batch1, Tensor batch2) -> Tensor(a!)
      
      auto dispatch_baddbmm_ = [](const at::Scalar & beta, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.baddbmm_(batch1, batch2, beta, 1);
      };
      return wrap(dispatch_baddbmm_(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      
      auto dispatch_baddbmm_ = [](const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.baddbmm_(batch1, batch2, beta, alpha);
      };
      return wrap(dispatch_baddbmm_(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bernoulli
static PyObject * THPVariable_bernoulli(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "bernoulli(*, Generator? generator=None)",
    "bernoulli(double p, *, Generator? generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor
      
      auto dispatch_bernoulli = [](const at::Tensor & self, ::std::optional<at::Generator> generator) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bernoulli(generator);
      };
      return wrap(dispatch_bernoulli(self, _r.generator(0)));
    }
    case 1: {
      // aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> Tensor
      
      auto dispatch_bernoulli = [](const at::Tensor & self, double p, ::std::optional<at::Generator> generator) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bernoulli(p, generator);
      };
      return wrap(dispatch_bernoulli(self, _r.toDouble(0), _r.generator(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bernoulli_
static PyObject * THPVariable_bernoulli_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "bernoulli_(Tensor p, *, Generator? generator=None)",
    "bernoulli_(double p=0.5, *, Generator? generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)
      
      auto dispatch_bernoulli_ = [](const at::Tensor & self, const at::Tensor & p, ::std::optional<at::Generator> generator) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bernoulli_(p, generator);
      };
      return wrap(dispatch_bernoulli_(self, _r.tensor(0), _r.generator(1)));
    }
    case 1: {
      // aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)
      
      auto dispatch_bernoulli_ = [](const at::Tensor & self, double p, ::std::optional<at::Generator> generator) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bernoulli_(p, generator);
      };
      return wrap(dispatch_bernoulli_(self, _r.toDouble(0), _r.generator(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// bincount
static PyObject * THPVariable_bincount(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "bincount(Tensor? weights=None, int64_t minlength=0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor
  
  auto dispatch_bincount = [](const at::Tensor & self, const ::std::optional<at::Tensor> & weights, int64_t minlength) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.bincount(weights, minlength);
  };
  return wrap(dispatch_bincount(self, _r.optionalTensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bitwise_and
static PyObject * THPVariable_bitwise_and(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "bitwise_and(Tensor other)",
    "bitwise_and(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_bitwise_and = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_and(other);
      };
      return wrap(dispatch_bitwise_and(self, _r.tensor(0)));
    }
    case 1: {
      // aten::bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_bitwise_and = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_and(other);
      };
      return wrap(dispatch_bitwise_and(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bitwise_and_
static PyObject * THPVariable_bitwise_and_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "bitwise_and_(Tensor other)",
    "bitwise_and_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::bitwise_and_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_bitwise_and_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_and_(other);
      };
      return wrap(dispatch_bitwise_and_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::bitwise_and_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_bitwise_and_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_and_(other);
      };
      return wrap(dispatch_bitwise_and_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bitwise_left_shift
static PyObject * THPVariable_bitwise_left_shift(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "bitwise_left_shift(Tensor other)",
    "bitwise_left_shift(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_bitwise_left_shift = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_left_shift(other);
      };
      return wrap(dispatch_bitwise_left_shift(self, _r.tensor(0)));
    }
    case 1: {
      // aten::bitwise_left_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_bitwise_left_shift = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_left_shift(other);
      };
      return wrap(dispatch_bitwise_left_shift(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bitwise_left_shift_
static PyObject * THPVariable_bitwise_left_shift_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "bitwise_left_shift_(Tensor other)",
    "bitwise_left_shift_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::bitwise_left_shift_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_bitwise_left_shift_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_left_shift_(other);
      };
      return wrap(dispatch_bitwise_left_shift_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::bitwise_left_shift_.Tensor_Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_bitwise_left_shift_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_left_shift_(other);
      };
      return wrap(dispatch_bitwise_left_shift_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// bitwise_not
static PyObject * THPVariable_bitwise_not(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "bitwise_not");
  }
  // aten::bitwise_not(Tensor self) -> Tensor
  
  auto dispatch_bitwise_not = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.bitwise_not();
  };
  return wrap(dispatch_bitwise_not(self));
  END_HANDLE_TH_ERRORS
}

// bitwise_not_
static PyObject * THPVariable_bitwise_not_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "bitwise_not_");
  }
  // aten::bitwise_not_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_bitwise_not_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.bitwise_not_();
  };
  return wrap(dispatch_bitwise_not_(self));
  END_HANDLE_TH_ERRORS
}

\
// bitwise_or
static PyObject * THPVariable_bitwise_or(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "bitwise_or(Tensor other)",
    "bitwise_or(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_bitwise_or = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_or(other);
      };
      return wrap(dispatch_bitwise_or(self, _r.tensor(0)));
    }
    case 1: {
      // aten::bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_bitwise_or = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_or(other);
      };
      return wrap(dispatch_bitwise_or(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bitwise_or_
static PyObject * THPVariable_bitwise_or_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "bitwise_or_(Tensor other)",
    "bitwise_or_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::bitwise_or_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_bitwise_or_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_or_(other);
      };
      return wrap(dispatch_bitwise_or_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::bitwise_or_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_bitwise_or_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_or_(other);
      };
      return wrap(dispatch_bitwise_or_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bitwise_right_shift
static PyObject * THPVariable_bitwise_right_shift(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "bitwise_right_shift(Tensor other)",
    "bitwise_right_shift(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_bitwise_right_shift = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_right_shift(other);
      };
      return wrap(dispatch_bitwise_right_shift(self, _r.tensor(0)));
    }
    case 1: {
      // aten::bitwise_right_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_bitwise_right_shift = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_right_shift(other);
      };
      return wrap(dispatch_bitwise_right_shift(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bitwise_right_shift_
static PyObject * THPVariable_bitwise_right_shift_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "bitwise_right_shift_(Tensor other)",
    "bitwise_right_shift_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::bitwise_right_shift_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_bitwise_right_shift_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_right_shift_(other);
      };
      return wrap(dispatch_bitwise_right_shift_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::bitwise_right_shift_.Tensor_Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_bitwise_right_shift_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_right_shift_(other);
      };
      return wrap(dispatch_bitwise_right_shift_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bitwise_xor
static PyObject * THPVariable_bitwise_xor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "bitwise_xor(Tensor other)",
    "bitwise_xor(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_bitwise_xor = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_xor(other);
      };
      return wrap(dispatch_bitwise_xor(self, _r.tensor(0)));
    }
    case 1: {
      // aten::bitwise_xor.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_bitwise_xor = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_xor(other);
      };
      return wrap(dispatch_bitwise_xor(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bitwise_xor_
static PyObject * THPVariable_bitwise_xor_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "bitwise_xor_(Tensor other)",
    "bitwise_xor_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::bitwise_xor_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_bitwise_xor_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_xor_(other);
      };
      return wrap(dispatch_bitwise_xor_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::bitwise_xor_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_bitwise_xor_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bitwise_xor_(other);
      };
      return wrap(dispatch_bitwise_xor_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// bmm
static PyObject * THPVariable_bmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "bmm(Tensor mat2)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::bmm(Tensor self, Tensor mat2) -> Tensor
  
  auto dispatch_bmm = [](const at::Tensor & self, const at::Tensor & mat2) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.bmm(mat2);
  };
  return wrap(dispatch_bmm(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// broadcast_to
static PyObject * THPVariable_broadcast_to(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "broadcast_to(SymIntArrayRef size)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::broadcast_to(Tensor(a) self, SymInt[] size) -> Tensor(a)
  
  auto dispatch_broadcast_to = [](const at::Tensor & self, c10::SymIntArrayRef size) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.broadcast_to_symint(size);
  };
  return wrap(dispatch_broadcast_to(self, _r.symintlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cauchy_
static PyObject * THPVariable_cauchy_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "cauchy_(double median=0, double sigma=1, *, Generator? generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)
  
  auto dispatch_cauchy_ = [](const at::Tensor & self, double median, double sigma, ::std::optional<at::Generator> generator) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cauchy_(median, sigma, generator);
  };
  return wrap(dispatch_cauchy_(self, _r.toDouble(0), _r.toDouble(1), _r.generator(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ccol_indices
static PyObject * THPVariable_ccol_indices(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "ccol_indices");
  }
  // aten::ccol_indices(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_ccol_indices = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.ccol_indices();
  };
  return wrap(dispatch_ccol_indices(self));
  END_HANDLE_TH_ERRORS
}

// ceil
static PyObject * THPVariable_ceil(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "ceil");
  }
  // aten::ceil(Tensor self) -> Tensor
  
  auto dispatch_ceil = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.ceil();
  };
  return wrap(dispatch_ceil(self));
  END_HANDLE_TH_ERRORS
}

// ceil_
static PyObject * THPVariable_ceil_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "ceil_");
  }
  // aten::ceil_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_ceil_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.ceil_();
  };
  return wrap(dispatch_ceil_(self));
  END_HANDLE_TH_ERRORS
}

// chalf
static PyObject * THPVariable_chalf(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "chalf(*, MemoryFormat? memory_format=None)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::chalf(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
  
  auto dispatch_chalf = [](const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.chalf(memory_format);
  };
  return wrap(dispatch_chalf(self, _r.memoryformatOptional(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cholesky
static PyObject * THPVariable_cholesky(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "cholesky(bool upper=False)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::cholesky(Tensor self, bool upper=False) -> Tensor
  
  auto dispatch_cholesky = [](const at::Tensor & self, bool upper) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cholesky(upper);
  };
  return wrap(dispatch_cholesky(self, _r.toBool(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cholesky_inverse
static PyObject * THPVariable_cholesky_inverse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "cholesky_inverse(bool upper=False)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::cholesky_inverse(Tensor self, bool upper=False) -> Tensor
  
  auto dispatch_cholesky_inverse = [](const at::Tensor & self, bool upper) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cholesky_inverse(upper);
  };
  return wrap(dispatch_cholesky_inverse(self, _r.toBool(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cholesky_solve
static PyObject * THPVariable_cholesky_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "cholesky_solve(Tensor input2, bool upper=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor
  
  auto dispatch_cholesky_solve = [](const at::Tensor & self, const at::Tensor & input2, bool upper) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cholesky_solve(input2, upper);
  };
  return wrap(dispatch_cholesky_solve(self, _r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// chunk
static PyObject * THPVariable_chunk(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "chunk(int64_t chunks, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::chunk(Tensor(a -> *) self, int chunks, int dim=0) -> Tensor(a)[]
  
  auto dispatch_chunk = [](const at::Tensor & self, int64_t chunks, int64_t dim) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.chunk(chunks, dim);
  };
  return wrap(dispatch_chunk(self, _r.toInt64(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// clamp
static PyObject * THPVariable_clamp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "clamp(Tensor? min=None, Tensor? max=None)",
    "clamp(Scalar? min=None, Scalar? max=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
      
      auto dispatch_clamp = [](const at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clamp(min, max);
      };
      return wrap(dispatch_clamp(self, _r.optionalTensor(0), _r.optionalTensor(1)));
    }
    case 1: {
      // aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
      
      auto dispatch_clamp = [](const at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clamp(min, max);
      };
      return wrap(dispatch_clamp(self, _r.scalarOptional(0), _r.scalarOptional(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// clamp_
static PyObject * THPVariable_clamp_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "clamp_(Tensor? min=None, Tensor? max=None)",
    "clamp_(Scalar? min=None, Scalar? max=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::clamp_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)
      
      auto dispatch_clamp_ = [](const at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clamp_(min, max);
      };
      return wrap(dispatch_clamp_(self, _r.optionalTensor(0), _r.optionalTensor(1)));
    }
    case 1: {
      // aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)
      
      auto dispatch_clamp_ = [](const at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clamp_(min, max);
      };
      return wrap(dispatch_clamp_(self, _r.scalarOptional(0), _r.scalarOptional(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// clamp_max
static PyObject * THPVariable_clamp_max(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "clamp_max(Tensor max)",
    "clamp_max(Scalar max)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::clamp_max.Tensor(Tensor self, Tensor max) -> Tensor
      
      auto dispatch_clamp_max = [](const at::Tensor & self, const at::Tensor & max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clamp_max(max);
      };
      return wrap(dispatch_clamp_max(self, _r.tensor(0)));
    }
    case 1: {
      // aten::clamp_max(Tensor self, Scalar max) -> Tensor
      
      auto dispatch_clamp_max = [](const at::Tensor & self, const at::Scalar & max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clamp_max(max);
      };
      return wrap(dispatch_clamp_max(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// clamp_max_
static PyObject * THPVariable_clamp_max_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "clamp_max_(Tensor max)",
    "clamp_max_(Scalar max)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::clamp_max_.Tensor(Tensor(a!) self, Tensor max) -> Tensor(a!)
      
      auto dispatch_clamp_max_ = [](const at::Tensor & self, const at::Tensor & max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clamp_max_(max);
      };
      return wrap(dispatch_clamp_max_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)
      
      auto dispatch_clamp_max_ = [](const at::Tensor & self, const at::Scalar & max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clamp_max_(max);
      };
      return wrap(dispatch_clamp_max_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// clamp_min
static PyObject * THPVariable_clamp_min(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "clamp_min(Tensor min)",
    "clamp_min(Scalar min)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::clamp_min.Tensor(Tensor self, Tensor min) -> Tensor
      
      auto dispatch_clamp_min = [](const at::Tensor & self, const at::Tensor & min) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clamp_min(min);
      };
      return wrap(dispatch_clamp_min(self, _r.tensor(0)));
    }
    case 1: {
      // aten::clamp_min(Tensor self, Scalar min) -> Tensor
      
      auto dispatch_clamp_min = [](const at::Tensor & self, const at::Scalar & min) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clamp_min(min);
      };
      return wrap(dispatch_clamp_min(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// clamp_min_
static PyObject * THPVariable_clamp_min_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "clamp_min_(Tensor min)",
    "clamp_min_(Scalar min)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::clamp_min_.Tensor(Tensor(a!) self, Tensor min) -> Tensor(a!)
      
      auto dispatch_clamp_min_ = [](const at::Tensor & self, const at::Tensor & min) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clamp_min_(min);
      };
      return wrap(dispatch_clamp_min_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)
      
      auto dispatch_clamp_min_ = [](const at::Tensor & self, const at::Scalar & min) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clamp_min_(min);
      };
      return wrap(dispatch_clamp_min_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// clip
static PyObject * THPVariable_clip(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "clip(Tensor? min=None, Tensor? max=None)",
    "clip(Scalar? min=None, Scalar? max=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::clip.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
      
      auto dispatch_clip = [](const at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clip(min, max);
      };
      return wrap(dispatch_clip(self, _r.optionalTensor(0), _r.optionalTensor(1)));
    }
    case 1: {
      // aten::clip(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
      
      auto dispatch_clip = [](const at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clip(min, max);
      };
      return wrap(dispatch_clip(self, _r.scalarOptional(0), _r.scalarOptional(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// clip_
static PyObject * THPVariable_clip_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "clip_(Tensor? min=None, Tensor? max=None)",
    "clip_(Scalar? min=None, Scalar? max=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::clip_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)
      
      auto dispatch_clip_ = [](const at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clip_(min, max);
      };
      return wrap(dispatch_clip_(self, _r.optionalTensor(0), _r.optionalTensor(1)));
    }
    case 1: {
      // aten::clip_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)
      
      auto dispatch_clip_ = [](const at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clip_(min, max);
      };
      return wrap(dispatch_clip_(self, _r.scalarOptional(0), _r.scalarOptional(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// clone
static PyObject * THPVariable_clone(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "clone(*, MemoryFormat? memory_format=None)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
  
  auto dispatch_clone = [](const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.clone(memory_format);
  };
  return wrap(dispatch_clone(self, _r.memoryformatOptional(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// coalesce
static PyObject * THPVariable_coalesce(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "coalesce");
  }
  // aten::coalesce(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_coalesce = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.coalesce();
  };
  return wrap(dispatch_coalesce(self));
  END_HANDLE_TH_ERRORS
}

// col_indices
static PyObject * THPVariable_col_indices(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "col_indices");
  }
  // aten::col_indices(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_col_indices = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.col_indices();
  };
  return wrap(dispatch_col_indices(self));
  END_HANDLE_TH_ERRORS
}

// conj
static PyObject * THPVariable_conj(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "conj");
  }
  // aten::conj(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_conj = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.conj();
  };
  return wrap(dispatch_conj(self));
  END_HANDLE_TH_ERRORS
}

// conj_physical
static PyObject * THPVariable_conj_physical(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "conj_physical");
  }
  // aten::conj_physical(Tensor self) -> Tensor
  
  auto dispatch_conj_physical = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.conj_physical();
  };
  return wrap(dispatch_conj_physical(self));
  END_HANDLE_TH_ERRORS
}

// conj_physical_
static PyObject * THPVariable_conj_physical_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "conj_physical_");
  }
  // aten::conj_physical_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_conj_physical_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.conj_physical_();
  };
  return wrap(dispatch_conj_physical_(self));
  END_HANDLE_TH_ERRORS
}

\
// copysign
static PyObject * THPVariable_copysign(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "copysign(Tensor other)",
    "copysign(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::copysign.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_copysign = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.copysign(other);
      };
      return wrap(dispatch_copysign(self, _r.tensor(0)));
    }
    case 1: {
      // aten::copysign.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_copysign = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.copysign(other);
      };
      return wrap(dispatch_copysign(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// copysign_
static PyObject * THPVariable_copysign_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "copysign_(Tensor other)",
    "copysign_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::copysign_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_copysign_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.copysign_(other);
      };
      return wrap(dispatch_copysign_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::copysign_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_copysign_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.copysign_(other);
      };
      return wrap(dispatch_copysign_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// corrcoef
static PyObject * THPVariable_corrcoef(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "corrcoef");
  }
  // aten::corrcoef(Tensor self) -> Tensor
  
  auto dispatch_corrcoef = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.corrcoef();
  };
  return wrap(dispatch_corrcoef(self));
  END_HANDLE_TH_ERRORS
}

// cos
static PyObject * THPVariable_cos(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "cos");
  }
  // aten::cos(Tensor self) -> Tensor
  
  auto dispatch_cos = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cos();
  };
  return wrap(dispatch_cos(self));
  END_HANDLE_TH_ERRORS
}

// cos_
static PyObject * THPVariable_cos_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "cos_");
  }
  // aten::cos_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_cos_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cos_();
  };
  return wrap(dispatch_cos_(self));
  END_HANDLE_TH_ERRORS
}

// cosh
static PyObject * THPVariable_cosh(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "cosh");
  }
  // aten::cosh(Tensor self) -> Tensor
  
  auto dispatch_cosh = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cosh();
  };
  return wrap(dispatch_cosh(self));
  END_HANDLE_TH_ERRORS
}

// cosh_
static PyObject * THPVariable_cosh_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "cosh_");
  }
  // aten::cosh_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_cosh_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cosh_();
  };
  return wrap(dispatch_cosh_(self));
  END_HANDLE_TH_ERRORS
}

\
// count_nonzero
static PyObject * THPVariable_count_nonzero(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "count_nonzero(int64_t? dim=None)",
    "count_nonzero(IntArrayRef dim)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::count_nonzero(Tensor self, int? dim=None) -> Tensor
      
      auto dispatch_count_nonzero = [](const at::Tensor & self, ::std::optional<int64_t> dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.count_nonzero(dim);
      };
      return wrap(dispatch_count_nonzero(self, _r.toInt64Optional(0)));
    }
    case 1: {
      // aten::count_nonzero.dim_IntList(Tensor self, int[] dim) -> Tensor
      
      auto dispatch_count_nonzero = [](const at::Tensor & self, at::IntArrayRef dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.count_nonzero(dim);
      };
      return wrap(dispatch_count_nonzero(self, _r.intlist(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cov
static PyObject * THPVariable_cov(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "cov(*, int64_t correction=1, Tensor? fweights=None, Tensor? aweights=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::cov(Tensor self, *, int correction=1, Tensor? fweights=None, Tensor? aweights=None) -> Tensor
  
  auto dispatch_cov = [](const at::Tensor & self, int64_t correction, const ::std::optional<at::Tensor> & fweights, const ::std::optional<at::Tensor> & aweights) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cov(correction, fweights, aweights);
  };
  return wrap(dispatch_cov(self, _r.toInt64(0), _r.optionalTensor(1), _r.optionalTensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cross
static PyObject * THPVariable_cross(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "cross(Tensor other, int64_t? dim=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor
  
  auto dispatch_cross = [](const at::Tensor & self, const at::Tensor & other, ::std::optional<int64_t> dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cross(other, dim);
  };
  return wrap(dispatch_cross(self, _r.tensor(0), _r.toInt64Optional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// crow_indices
static PyObject * THPVariable_crow_indices(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "crow_indices");
  }
  // aten::crow_indices(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_crow_indices = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.crow_indices();
  };
  return wrap(dispatch_crow_indices(self));
  END_HANDLE_TH_ERRORS
}

\
// cummax
static PyObject * THPVariable_cummax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_cummax_structseq();
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "cummax(int64_t dim)",
    "cummax(Dimname dim)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)
      
      auto dispatch_cummax = [](const at::Tensor & self, int64_t dim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.cummax(dim);
      };
      return wrap(NamedTuple, dispatch_cummax(self, _r.toInt64(0)));
    }
    case 1: {
      // aten::cummax.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)
      
      auto dispatch_cummax = [](const at::Tensor & self, at::Dimname dim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.cummax(dim);
      };
      return wrap(NamedTuple, dispatch_cummax(self, _r.dimname(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// cummin
static PyObject * THPVariable_cummin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_cummin_structseq();
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "cummin(int64_t dim)",
    "cummin(Dimname dim)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::cummin(Tensor self, int dim) -> (Tensor values, Tensor indices)
      
      auto dispatch_cummin = [](const at::Tensor & self, int64_t dim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.cummin(dim);
      };
      return wrap(NamedTuple, dispatch_cummin(self, _r.toInt64(0)));
    }
    case 1: {
      // aten::cummin.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)
      
      auto dispatch_cummin = [](const at::Tensor & self, at::Dimname dim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.cummin(dim);
      };
      return wrap(NamedTuple, dispatch_cummin(self, _r.dimname(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// cumprod
static PyObject * THPVariable_cumprod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "cumprod(int64_t dim, *, ScalarType? dtype=None)",
    "cumprod(Dimname dim, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_cumprod = [](const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.cumprod(dim, dtype);
      };
      return wrap(dispatch_cumprod(self, _r.toInt64(0), _r.scalartypeOptional(1)));
    }
    case 1: {
      // aten::cumprod.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_cumprod = [](const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.cumprod(dim, dtype);
      };
      return wrap(dispatch_cumprod(self, _r.dimname(0), _r.scalartypeOptional(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// cumprod_
static PyObject * THPVariable_cumprod_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "cumprod_(int64_t dim, *, ScalarType? dtype=None)",
    "cumprod_(Dimname dim, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::cumprod_(Tensor(a!) self, int dim, *, ScalarType? dtype=None) -> Tensor(a!)
      
      auto dispatch_cumprod_ = [](const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.cumprod_(dim, dtype);
      };
      return wrap(dispatch_cumprod_(self, _r.toInt64(0), _r.scalartypeOptional(1)));
    }
    case 1: {
      // aten::cumprod_.dimname(Tensor(a!) self, Dimname dim, *, ScalarType? dtype=None) -> Tensor(a!)
      
      auto dispatch_cumprod_ = [](const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.cumprod_(dim, dtype);
      };
      return wrap(dispatch_cumprod_(self, _r.dimname(0), _r.scalartypeOptional(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// cumsum
static PyObject * THPVariable_cumsum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "cumsum(int64_t dim, *, ScalarType? dtype=None)",
    "cumsum(Dimname dim, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_cumsum = [](const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.cumsum(dim, dtype);
      };
      return wrap(dispatch_cumsum(self, _r.toInt64(0), _r.scalartypeOptional(1)));
    }
    case 1: {
      // aten::cumsum.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_cumsum = [](const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.cumsum(dim, dtype);
      };
      return wrap(dispatch_cumsum(self, _r.dimname(0), _r.scalartypeOptional(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// cumsum_
static PyObject * THPVariable_cumsum_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "cumsum_(int64_t dim, *, ScalarType? dtype=None)",
    "cumsum_(Dimname dim, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::cumsum_(Tensor(a!) self, int dim, *, ScalarType? dtype=None) -> Tensor(a!)
      
      auto dispatch_cumsum_ = [](const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.cumsum_(dim, dtype);
      };
      return wrap(dispatch_cumsum_(self, _r.toInt64(0), _r.scalartypeOptional(1)));
    }
    case 1: {
      // aten::cumsum_.dimname(Tensor(a!) self, Dimname dim, *, ScalarType? dtype=None) -> Tensor(a!)
      
      auto dispatch_cumsum_ = [](const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.cumsum_(dim, dtype);
      };
      return wrap(dispatch_cumsum_(self, _r.dimname(0), _r.scalartypeOptional(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// deg2rad
static PyObject * THPVariable_deg2rad(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "deg2rad");
  }
  // aten::deg2rad(Tensor self) -> Tensor
  
  auto dispatch_deg2rad = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.deg2rad();
  };
  return wrap(dispatch_deg2rad(self));
  END_HANDLE_TH_ERRORS
}

// deg2rad_
static PyObject * THPVariable_deg2rad_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "deg2rad_");
  }
  // aten::deg2rad_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_deg2rad_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.deg2rad_();
  };
  return wrap(dispatch_deg2rad_(self));
  END_HANDLE_TH_ERRORS
}

// dense_dim
static PyObject * THPVariable_dense_dim(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "dense_dim");
  }
  // aten::dense_dim(Tensor self) -> int
  
  auto dispatch_dense_dim = [](const at::Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return self.dense_dim();
  };
  return wrap(dispatch_dense_dim(self));
  END_HANDLE_TH_ERRORS
}

// dequantize
static PyObject * THPVariable_dequantize(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "dequantize");
  }
  // aten::dequantize.self(Tensor self) -> Tensor
  
  auto dispatch_dequantize = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.dequantize();
  };
  return wrap(dispatch_dequantize(self));
  END_HANDLE_TH_ERRORS
}

// det
static PyObject * THPVariable_det(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "det");
  }
  // aten::det(Tensor self) -> Tensor
  
  auto dispatch_det = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.det();
  };
  return wrap(dispatch_det(self));
  END_HANDLE_TH_ERRORS
}

// detach
static PyObject * THPVariable_detach(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "detach");
  }
  // aten::detach(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_detach = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.detach();
  };
  return wrap(dispatch_detach(self));
  END_HANDLE_TH_ERRORS
}

// detach_
static PyObject * THPVariable_detach_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "detach_");
  }
  // aten::detach_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_detach_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.detach_();
  };
  return wrap(dispatch_detach_(self));
  END_HANDLE_TH_ERRORS
}

// diag
static PyObject * THPVariable_diag(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "diag(int64_t diagonal=0)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::diag(Tensor self, int diagonal=0) -> Tensor
  
  auto dispatch_diag = [](const at::Tensor & self, int64_t diagonal) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.diag(diagonal);
  };
  return wrap(dispatch_diag(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// diag_embed
static PyObject * THPVariable_diag_embed(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "diag_embed(int64_t offset=0, int64_t dim1=-2, int64_t dim2=-1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor
  
  auto dispatch_diag_embed = [](const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.diag_embed(offset, dim1, dim2);
  };
  return wrap(dispatch_diag_embed(self, _r.toInt64(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// diagflat
static PyObject * THPVariable_diagflat(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "diagflat(int64_t offset=0)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::diagflat(Tensor self, int offset=0) -> Tensor
  
  auto dispatch_diagflat = [](const at::Tensor & self, int64_t offset) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.diagflat(offset);
  };
  return wrap(dispatch_diagflat(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// diagonal
static PyObject * THPVariable_diagonal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "diagonal(*, Dimname outdim, Dimname dim1, Dimname dim2, int64_t offset=0)",
    "diagonal(int64_t offset=0, int64_t dim1=0, int64_t dim2=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::diagonal.Dimname(Tensor(a) self, *, Dimname outdim, Dimname dim1, Dimname dim2, int offset=0) -> Tensor(a)
      
      auto dispatch_diagonal = [](const at::Tensor & self, at::Dimname outdim, at::Dimname dim1, at::Dimname dim2, int64_t offset) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.diagonal(outdim, dim1, dim2, offset);
      };
      return wrap(dispatch_diagonal(self, _r.dimname(0), _r.dimname(1), _r.dimname(2), _r.toInt64(3)));
    }
    case 1: {
      // aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)
      
      auto dispatch_diagonal = [](const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.diagonal(offset, dim1, dim2);
      };
      return wrap(dispatch_diagonal(self, _r.toInt64(0), _r.toInt64(1), _r.toInt64(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// diagonal_scatter
static PyObject * THPVariable_diagonal_scatter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "diagonal_scatter(Tensor src, int64_t offset=0, int64_t dim1=0, int64_t dim2=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::diagonal_scatter(Tensor self, Tensor src, int offset=0, int dim1=0, int dim2=1) -> Tensor
  
  auto dispatch_diagonal_scatter = [](const at::Tensor & self, const at::Tensor & src, int64_t offset, int64_t dim1, int64_t dim2) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.diagonal_scatter(src, offset, dim1, dim2);
  };
  return wrap(dispatch_diagonal_scatter(self, _r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toInt64(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// diff
static PyObject * THPVariable_diff(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "diff(int64_t n=1, int64_t dim=-1, Tensor? prepend=None, Tensor? append=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::diff(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None) -> Tensor
  
  auto dispatch_diff = [](const at::Tensor & self, int64_t n, int64_t dim, const ::std::optional<at::Tensor> & prepend, const ::std::optional<at::Tensor> & append) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.diff(n, dim, prepend, append);
  };
  return wrap(dispatch_diff(self, _r.toInt64(0), _r.toInt64(1), _r.optionalTensor(2), _r.optionalTensor(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// digamma
static PyObject * THPVariable_digamma(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "digamma");
  }
  // aten::digamma(Tensor self) -> Tensor
  
  auto dispatch_digamma = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.digamma();
  };
  return wrap(dispatch_digamma(self));
  END_HANDLE_TH_ERRORS
}

// digamma_
static PyObject * THPVariable_digamma_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "digamma_");
  }
  // aten::digamma_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_digamma_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.digamma_();
  };
  return wrap(dispatch_digamma_(self));
  END_HANDLE_TH_ERRORS
}

// dist
static PyObject * THPVariable_dist(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "dist(Tensor other, Scalar p=2)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::dist(Tensor self, Tensor other, Scalar p=2) -> Tensor
  
  auto dispatch_dist = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & p) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.dist(other, p);
  };
  return wrap(dispatch_dist(self, _r.tensor(0), _r.scalar(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// div
static PyObject * THPVariable_div(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "div(Tensor other)",
    "div(Tensor other, *, c10::string_view? rounding_mode)",
    "div(Scalar other, *, c10::string_view? rounding_mode)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::div.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_div = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.div(other);
      };
      return wrap(dispatch_div(self, _r.tensor(0)));
    }
    case 1: {
      // aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
      
      auto dispatch_div = [](const at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.div(other, rounding_mode);
      };
      return wrap(dispatch_div(self, _r.tensor(0), _r.stringViewOptional(1)));
    }
    case 2: {
      // aten::div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor
      
      auto dispatch_div = [](const at::Tensor & self, const at::Scalar & other, ::std::optional<c10::string_view> rounding_mode) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.div(other, rounding_mode);
      };
      return wrap(dispatch_div(self, _r.scalar(0), _r.stringViewOptional(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// div_
static PyObject * THPVariable_div_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "div_(Tensor other)",
    "div_(Tensor other, *, c10::string_view? rounding_mode)",
    "div_(Scalar other, *, c10::string_view? rounding_mode)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_div_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.div_(other);
      };
      return wrap(dispatch_div_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::div_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)
      
      auto dispatch_div_ = [](const at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.div_(other, rounding_mode);
      };
      return wrap(dispatch_div_(self, _r.tensor(0), _r.stringViewOptional(1)));
    }
    case 2: {
      // aten::div_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)
      
      auto dispatch_div_ = [](const at::Tensor & self, const at::Scalar & other, ::std::optional<c10::string_view> rounding_mode) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.div_(other, rounding_mode);
      };
      return wrap(dispatch_div_(self, _r.scalar(0), _r.stringViewOptional(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// divide
static PyObject * THPVariable_divide(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "divide(Tensor other)",
    "divide(Tensor other, *, c10::string_view? rounding_mode)",
    "divide(Scalar other)",
    "divide(Scalar other, *, c10::string_view? rounding_mode)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::divide.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_divide = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.divide(other);
      };
      return wrap(dispatch_divide(self, _r.tensor(0)));
    }
    case 1: {
      // aten::divide.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
      
      auto dispatch_divide = [](const at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.divide(other, rounding_mode);
      };
      return wrap(dispatch_divide(self, _r.tensor(0), _r.stringViewOptional(1)));
    }
    case 2: {
      // aten::divide.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_divide = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.divide(other);
      };
      return wrap(dispatch_divide(self, _r.scalar(0)));
    }
    case 3: {
      // aten::divide.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor
      
      auto dispatch_divide = [](const at::Tensor & self, const at::Scalar & other, ::std::optional<c10::string_view> rounding_mode) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.divide(other, rounding_mode);
      };
      return wrap(dispatch_divide(self, _r.scalar(0), _r.stringViewOptional(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// divide_
static PyObject * THPVariable_divide_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "divide_(Tensor other)",
    "divide_(Tensor other, *, c10::string_view? rounding_mode)",
    "divide_(Scalar other)",
    "divide_(Scalar other, *, c10::string_view? rounding_mode)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_divide_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.divide_(other);
      };
      return wrap(dispatch_divide_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::divide_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)
      
      auto dispatch_divide_ = [](const at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.divide_(other, rounding_mode);
      };
      return wrap(dispatch_divide_(self, _r.tensor(0), _r.stringViewOptional(1)));
    }
    case 2: {
      // aten::divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_divide_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.divide_(other);
      };
      return wrap(dispatch_divide_(self, _r.scalar(0)));
    }
    case 3: {
      // aten::divide_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)
      
      auto dispatch_divide_ = [](const at::Tensor & self, const at::Scalar & other, ::std::optional<c10::string_view> rounding_mode) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.divide_(other, rounding_mode);
      };
      return wrap(dispatch_divide_(self, _r.scalar(0), _r.stringViewOptional(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// dot
static PyObject * THPVariable_dot(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "dot(Tensor tensor)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::dot(Tensor self, Tensor tensor) -> Tensor
  
  auto dispatch_dot = [](const at::Tensor & self, const at::Tensor & tensor) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.dot(tensor);
  };
  return wrap(dispatch_dot(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// dsplit
static PyObject * THPVariable_dsplit(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "dsplit(int64_t sections)",
    "dsplit(IntArrayRef indices)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::dsplit.int(Tensor(a -> *) self, int sections) -> Tensor(a)[]
      
      auto dispatch_dsplit = [](const at::Tensor & self, int64_t sections) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.dsplit(sections);
      };
      return wrap(dispatch_dsplit(self, _r.toInt64(0)));
    }
    case 1: {
      // aten::dsplit.array(Tensor(a -> *) self, int[] indices) -> Tensor(a)[]
      
      auto dispatch_dsplit = [](const at::Tensor & self, at::IntArrayRef indices) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.dsplit(indices);
      };
      return wrap(dispatch_dsplit(self, _r.intlist(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// eq
static PyObject * THPVariable_eq(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "eq(Tensor other)",
    "eq(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::eq.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_eq = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.eq(other);
      };
      return wrap(dispatch_eq(self, _r.tensor(0)));
    }
    case 1: {
      // aten::eq.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_eq = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.eq(other);
      };
      return wrap(dispatch_eq(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// eq_
static PyObject * THPVariable_eq_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "eq_(Tensor other)",
    "eq_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::eq_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_eq_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.eq_(other);
      };
      return wrap(dispatch_eq_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::eq_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_eq_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.eq_(other);
      };
      return wrap(dispatch_eq_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// equal
static PyObject * THPVariable_equal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "equal(Tensor other)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::equal(Tensor self, Tensor other) -> bool
  
  auto dispatch_equal = [](const at::Tensor & self, const at::Tensor & other) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.equal(other);
  };
  return wrap(dispatch_equal(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// erf
static PyObject * THPVariable_erf(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "erf");
  }
  // aten::erf(Tensor self) -> Tensor
  
  auto dispatch_erf = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.erf();
  };
  return wrap(dispatch_erf(self));
  END_HANDLE_TH_ERRORS
}

// erf_
static PyObject * THPVariable_erf_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "erf_");
  }
  // aten::erf_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_erf_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.erf_();
  };
  return wrap(dispatch_erf_(self));
  END_HANDLE_TH_ERRORS
}

// erfc
static PyObject * THPVariable_erfc(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "erfc");
  }
  // aten::erfc(Tensor self) -> Tensor
  
  auto dispatch_erfc = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.erfc();
  };
  return wrap(dispatch_erfc(self));
  END_HANDLE_TH_ERRORS
}

// erfc_
static PyObject * THPVariable_erfc_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "erfc_");
  }
  // aten::erfc_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_erfc_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.erfc_();
  };
  return wrap(dispatch_erfc_(self));
  END_HANDLE_TH_ERRORS
}

// erfinv
static PyObject * THPVariable_erfinv(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "erfinv");
  }
  // aten::erfinv(Tensor self) -> Tensor
  
  auto dispatch_erfinv = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.erfinv();
  };
  return wrap(dispatch_erfinv(self));
  END_HANDLE_TH_ERRORS
}

// erfinv_
static PyObject * THPVariable_erfinv_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "erfinv_");
  }
  // aten::erfinv_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_erfinv_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.erfinv_();
  };
  return wrap(dispatch_erfinv_(self));
  END_HANDLE_TH_ERRORS
}

// exp
static PyObject * THPVariable_exp(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "exp");
  }
  // aten::exp(Tensor self) -> Tensor
  
  auto dispatch_exp = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.exp();
  };
  return wrap(dispatch_exp(self));
  END_HANDLE_TH_ERRORS
}

// exp2
static PyObject * THPVariable_exp2(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "exp2");
  }
  // aten::exp2(Tensor self) -> Tensor
  
  auto dispatch_exp2 = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.exp2();
  };
  return wrap(dispatch_exp2(self));
  END_HANDLE_TH_ERRORS
}

// exp2_
static PyObject * THPVariable_exp2_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "exp2_");
  }
  // aten::exp2_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_exp2_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.exp2_();
  };
  return wrap(dispatch_exp2_(self));
  END_HANDLE_TH_ERRORS
}

// exp_
static PyObject * THPVariable_exp_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "exp_");
  }
  // aten::exp_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_exp_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.exp_();
  };
  return wrap(dispatch_exp_(self));
  END_HANDLE_TH_ERRORS
}

// expand
static PyObject * THPVariable_expand(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "expand(SymIntArrayRef size, *, bool implicit=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)
  
  auto dispatch_expand = [](const at::Tensor & self, c10::SymIntArrayRef size, bool implicit) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.expand_symint(size, implicit);
  };
  return wrap(dispatch_expand(self, _r.symintlist(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// expand_as
static PyObject * THPVariable_expand_as(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "expand_as(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)
  
  auto dispatch_expand_as = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.expand_as(other);
  };
  return wrap(dispatch_expand_as(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// expm1
static PyObject * THPVariable_expm1(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "expm1");
  }
  // aten::expm1(Tensor self) -> Tensor
  
  auto dispatch_expm1 = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.expm1();
  };
  return wrap(dispatch_expm1(self));
  END_HANDLE_TH_ERRORS
}

// expm1_
static PyObject * THPVariable_expm1_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "expm1_");
  }
  // aten::expm1_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_expm1_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.expm1_();
  };
  return wrap(dispatch_expm1_(self));
  END_HANDLE_TH_ERRORS
}

// exponential_
static PyObject * THPVariable_exponential_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "exponential_(double lambd=1, *, Generator? generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::exponential_(Tensor(a!) self, float lambd=1, *, Generator? generator=None) -> Tensor(a!)
  
  auto dispatch_exponential_ = [](const at::Tensor & self, double lambd, ::std::optional<at::Generator> generator) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.exponential_(lambd, generator);
  };
  return wrap(dispatch_exponential_(self, _r.toDouble(0), _r.generator(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// fill_
static PyObject * THPVariable_fill_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "fill_(Tensor value)",
    "fill_(Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)
      
      auto dispatch_fill_ = [](const at::Tensor & self, const at::Tensor & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.fill_(value);
      };
      return wrap(dispatch_fill_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
      
      auto dispatch_fill_ = [](const at::Tensor & self, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.fill_(value);
      };
      return wrap(dispatch_fill_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fill_diagonal_
static PyObject * THPVariable_fill_diagonal_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "fill_diagonal_(Scalar fill_value, bool wrap=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::fill_diagonal_(Tensor(a!) self, Scalar fill_value, bool wrap=False) -> Tensor(a!)
  
  auto dispatch_fill_diagonal_ = [](const at::Tensor & self, const at::Scalar & fill_value, bool wrap) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.fill_diagonal_(fill_value, wrap);
  };
  return wrap(dispatch_fill_diagonal_(self, _r.scalar(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fix
static PyObject * THPVariable_fix(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "fix");
  }
  // aten::fix(Tensor self) -> Tensor
  
  auto dispatch_fix = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.fix();
  };
  return wrap(dispatch_fix(self));
  END_HANDLE_TH_ERRORS
}

// fix_
static PyObject * THPVariable_fix_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "fix_");
  }
  // aten::fix_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_fix_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.fix_();
  };
  return wrap(dispatch_fix_(self));
  END_HANDLE_TH_ERRORS
}

\
// flatten
static PyObject * THPVariable_flatten(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "flatten(int64_t start_dim, int64_t end_dim, Dimname out_dim)",
    "flatten(int64_t start_dim=0, int64_t end_dim=-1)",
    "flatten(Dimname start_dim, Dimname end_dim, Dimname out_dim)",
    "flatten(DimnameList dims, Dimname out_dim)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::flatten.named_out_dim(Tensor(a) self, int start_dim, int end_dim, Dimname out_dim) -> Tensor(a)
      
      auto dispatch_flatten = [](const at::Tensor & self, int64_t start_dim, int64_t end_dim, at::Dimname out_dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.flatten(start_dim, end_dim, out_dim);
      };
      return wrap(dispatch_flatten(self, _r.toInt64(0), _r.toInt64(1), _r.dimname(2)));
    }
    case 1: {
      // aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)
      
      auto dispatch_flatten = [](const at::Tensor & self, int64_t start_dim, int64_t end_dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.flatten(start_dim, end_dim);
      };
      return wrap(dispatch_flatten(self, _r.toInt64(0), _r.toInt64(1)));
    }
    case 2: {
      // aten::flatten.using_names(Tensor(a) self, Dimname start_dim, Dimname end_dim, Dimname out_dim) -> Tensor(a)
      
      auto dispatch_flatten = [](const at::Tensor & self, at::Dimname start_dim, at::Dimname end_dim, at::Dimname out_dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.flatten(start_dim, end_dim, out_dim);
      };
      return wrap(dispatch_flatten(self, _r.dimname(0), _r.dimname(1), _r.dimname(2)));
    }
    case 3: {
      // aten::flatten.DimnameList(Tensor(a) self, Dimname[] dims, Dimname out_dim) -> Tensor(a)
      
      auto dispatch_flatten = [](const at::Tensor & self, at::DimnameList dims, at::Dimname out_dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.flatten(dims, out_dim);
      };
      return wrap(dispatch_flatten(self, _r.dimnamelist(0), _r.dimname(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// flip
static PyObject * THPVariable_flip(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "flip(IntArrayRef dims)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::flip(Tensor self, int[] dims) -> Tensor
  
  auto dispatch_flip = [](const at::Tensor & self, at::IntArrayRef dims) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.flip(dims);
  };
  return wrap(dispatch_flip(self, _r.intlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fliplr
static PyObject * THPVariable_fliplr(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "fliplr");
  }
  // aten::fliplr(Tensor self) -> Tensor
  
  auto dispatch_fliplr = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.fliplr();
  };
  return wrap(dispatch_fliplr(self));
  END_HANDLE_TH_ERRORS
}

// flipud
static PyObject * THPVariable_flipud(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "flipud");
  }
  // aten::flipud(Tensor self) -> Tensor
  
  auto dispatch_flipud = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.flipud();
  };
  return wrap(dispatch_flipud(self));
  END_HANDLE_TH_ERRORS
}

\
// float_power
static PyObject * THPVariable_float_power(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "float_power(Tensor exponent)",
    "float_power(Scalar exponent)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::float_power.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
      
      auto dispatch_float_power = [](const at::Tensor & self, const at::Tensor & exponent) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.float_power(exponent);
      };
      return wrap(dispatch_float_power(self, _r.tensor(0)));
    }
    case 1: {
      // aten::float_power.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
      
      auto dispatch_float_power = [](const at::Tensor & self, const at::Scalar & exponent) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.float_power(exponent);
      };
      return wrap(dispatch_float_power(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// float_power_
static PyObject * THPVariable_float_power_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "float_power_(Tensor exponent)",
    "float_power_(Scalar exponent)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::float_power_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)
      
      auto dispatch_float_power_ = [](const at::Tensor & self, const at::Tensor & exponent) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.float_power_(exponent);
      };
      return wrap(dispatch_float_power_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::float_power_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)
      
      auto dispatch_float_power_ = [](const at::Tensor & self, const at::Scalar & exponent) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.float_power_(exponent);
      };
      return wrap(dispatch_float_power_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// floor
static PyObject * THPVariable_floor(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "floor");
  }
  // aten::floor(Tensor self) -> Tensor
  
  auto dispatch_floor = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.floor();
  };
  return wrap(dispatch_floor(self));
  END_HANDLE_TH_ERRORS
}

// floor_
static PyObject * THPVariable_floor_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "floor_");
  }
  // aten::floor_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_floor_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.floor_();
  };
  return wrap(dispatch_floor_(self));
  END_HANDLE_TH_ERRORS
}

\
// floor_divide
static PyObject * THPVariable_floor_divide(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "floor_divide(Tensor other)",
    "floor_divide(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::floor_divide(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_floor_divide = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.floor_divide(other);
      };
      return wrap(dispatch_floor_divide(self, _r.tensor(0)));
    }
    case 1: {
      // aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_floor_divide = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.floor_divide(other);
      };
      return wrap(dispatch_floor_divide(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// floor_divide_
static PyObject * THPVariable_floor_divide_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "floor_divide_(Tensor other)",
    "floor_divide_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::floor_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_floor_divide_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.floor_divide_(other);
      };
      return wrap(dispatch_floor_divide_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::floor_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_floor_divide_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.floor_divide_(other);
      };
      return wrap(dispatch_floor_divide_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fmax
static PyObject * THPVariable_fmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "fmax(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::fmax(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_fmax = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.fmax(other);
  };
  return wrap(dispatch_fmax(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fmin
static PyObject * THPVariable_fmin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "fmin(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::fmin(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_fmin = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.fmin(other);
  };
  return wrap(dispatch_fmin(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// fmod
static PyObject * THPVariable_fmod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "fmod(Tensor other)",
    "fmod(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_fmod = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.fmod(other);
      };
      return wrap(dispatch_fmod(self, _r.tensor(0)));
    }
    case 1: {
      // aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_fmod = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.fmod(other);
      };
      return wrap(dispatch_fmod(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// fmod_
static PyObject * THPVariable_fmod_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "fmod_(Tensor other)",
    "fmod_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::fmod_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_fmod_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.fmod_(other);
      };
      return wrap(dispatch_fmod_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::fmod_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_fmod_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.fmod_(other);
      };
      return wrap(dispatch_fmod_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// frac
static PyObject * THPVariable_frac(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "frac");
  }
  // aten::frac(Tensor self) -> Tensor
  
  auto dispatch_frac = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.frac();
  };
  return wrap(dispatch_frac(self));
  END_HANDLE_TH_ERRORS
}

// frac_
static PyObject * THPVariable_frac_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "frac_");
  }
  // aten::frac_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_frac_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.frac_();
  };
  return wrap(dispatch_frac_(self));
  END_HANDLE_TH_ERRORS
}

// frexp
static PyObject * THPVariable_frexp(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_frexp_structseq();
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "frexp");
  }
  // aten::frexp.Tensor(Tensor self) -> (Tensor mantissa, Tensor exponent)
  
  auto dispatch_frexp = [](const at::Tensor & self) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.frexp();
  };
  return wrap(NamedTuple, dispatch_frexp(self));
  END_HANDLE_TH_ERRORS
}

\
// gather
static PyObject * THPVariable_gather(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "gather(int64_t dim, Tensor index, *, bool sparse_grad=False)",
    "gather(Dimname dim, Tensor index, *, bool sparse_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
      
      auto dispatch_gather = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.gather(dim, index, sparse_grad);
      };
      return wrap(dispatch_gather(self, _r.toInt64(0), _r.tensor(1), _r.toBool(2)));
    }
    case 1: {
      // aten::gather.dimname(Tensor self, Dimname dim, Tensor index, *, bool sparse_grad=False) -> Tensor
      
      auto dispatch_gather = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index, bool sparse_grad) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.gather(dim, index, sparse_grad);
      };
      return wrap(dispatch_gather(self, _r.dimname(0), _r.tensor(1), _r.toBool(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// gcd
static PyObject * THPVariable_gcd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "gcd(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::gcd(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_gcd = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.gcd(other);
  };
  return wrap(dispatch_gcd(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// gcd_
static PyObject * THPVariable_gcd_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "gcd_(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::gcd_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  
  auto dispatch_gcd_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.gcd_(other);
  };
  return wrap(dispatch_gcd_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// ge
static PyObject * THPVariable_ge(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "ge(Tensor other)",
    "ge(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::ge.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_ge = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.ge(other);
      };
      return wrap(dispatch_ge(self, _r.tensor(0)));
    }
    case 1: {
      // aten::ge.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_ge = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.ge(other);
      };
      return wrap(dispatch_ge(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// ge_
static PyObject * THPVariable_ge_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "ge_(Tensor other)",
    "ge_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::ge_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_ge_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.ge_(other);
      };
      return wrap(dispatch_ge_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::ge_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_ge_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.ge_(other);
      };
      return wrap(dispatch_ge_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// geometric_
static PyObject * THPVariable_geometric_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "geometric_(double p, *, Generator? generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
  
  auto dispatch_geometric_ = [](const at::Tensor & self, double p, ::std::optional<at::Generator> generator) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.geometric_(p, generator);
  };
  return wrap(dispatch_geometric_(self, _r.toDouble(0), _r.generator(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// geqrf
static PyObject * THPVariable_geqrf(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_geqrf_structseq();
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "geqrf");
  }
  // aten::geqrf(Tensor self) -> (Tensor a, Tensor tau)
  
  auto dispatch_geqrf = [](const at::Tensor & self) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.geqrf();
  };
  return wrap(NamedTuple, dispatch_geqrf(self));
  END_HANDLE_TH_ERRORS
}

// ger
static PyObject * THPVariable_ger(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "ger(Tensor vec2)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::ger(Tensor self, Tensor vec2) -> Tensor
  
  auto dispatch_ger = [](const at::Tensor & self, const at::Tensor & vec2) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.ger(vec2);
  };
  return wrap(dispatch_ger(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// greater
static PyObject * THPVariable_greater(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "greater(Tensor other)",
    "greater(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::greater.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_greater = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.greater(other);
      };
      return wrap(dispatch_greater(self, _r.tensor(0)));
    }
    case 1: {
      // aten::greater.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_greater = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.greater(other);
      };
      return wrap(dispatch_greater(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// greater_
static PyObject * THPVariable_greater_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "greater_(Tensor other)",
    "greater_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::greater_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_greater_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.greater_(other);
      };
      return wrap(dispatch_greater_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::greater_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_greater_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.greater_(other);
      };
      return wrap(dispatch_greater_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// greater_equal
static PyObject * THPVariable_greater_equal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "greater_equal(Tensor other)",
    "greater_equal(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::greater_equal.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_greater_equal = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.greater_equal(other);
      };
      return wrap(dispatch_greater_equal(self, _r.tensor(0)));
    }
    case 1: {
      // aten::greater_equal.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_greater_equal = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.greater_equal(other);
      };
      return wrap(dispatch_greater_equal(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// greater_equal_
static PyObject * THPVariable_greater_equal_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "greater_equal_(Tensor other)",
    "greater_equal_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::greater_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_greater_equal_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.greater_equal_(other);
      };
      return wrap(dispatch_greater_equal_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::greater_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_greater_equal_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.greater_equal_(other);
      };
      return wrap(dispatch_greater_equal_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// gt
static PyObject * THPVariable_gt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "gt(Tensor other)",
    "gt(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::gt.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_gt = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.gt(other);
      };
      return wrap(dispatch_gt(self, _r.tensor(0)));
    }
    case 1: {
      // aten::gt.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_gt = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.gt(other);
      };
      return wrap(dispatch_gt(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// gt_
static PyObject * THPVariable_gt_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "gt_(Tensor other)",
    "gt_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::gt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_gt_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.gt_(other);
      };
      return wrap(dispatch_gt_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::gt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_gt_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.gt_(other);
      };
      return wrap(dispatch_gt_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hardshrink
static PyObject * THPVariable_hardshrink(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "hardshrink(Scalar lambd=0.5)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor
  
  auto dispatch_hardshrink = [](const at::Tensor & self, const at::Scalar & lambd) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.hardshrink(lambd);
  };
  return wrap(dispatch_hardshrink(self, _r.scalar(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// heaviside
static PyObject * THPVariable_heaviside(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "heaviside(Tensor values)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::heaviside(Tensor self, Tensor values) -> Tensor
  
  auto dispatch_heaviside = [](const at::Tensor & self, const at::Tensor & values) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.heaviside(values);
  };
  return wrap(dispatch_heaviside(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// heaviside_
static PyObject * THPVariable_heaviside_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "heaviside_(Tensor values)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::heaviside_(Tensor(a!) self, Tensor values) -> Tensor(a!)
  
  auto dispatch_heaviside_ = [](const at::Tensor & self, const at::Tensor & values) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.heaviside_(values);
  };
  return wrap(dispatch_heaviside_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// histc
static PyObject * THPVariable_histc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "histc(int64_t bins=100, Scalar min=0, Scalar max=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor
  
  auto dispatch_histc = [](const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.histc(bins, min, max);
  };
  return wrap(dispatch_histc(self, _r.toInt64(0), _r.scalar(1), _r.scalar(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// histogram
static PyObject * THPVariable_histogram(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_histogram_structseq();
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "histogram(Tensor bins, *, Tensor? weight=None, bool density=False)",
    "histogram(int64_t bins=100, *, ArrayRef<double>? range=None, Tensor? weight=None, bool density=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::histogram.bins_tensor(Tensor self, Tensor bins, *, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor bin_edges)
      
      auto dispatch_histogram = [](const at::Tensor & self, const at::Tensor & bins, const ::std::optional<at::Tensor> & weight, bool density) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.histogram(bins, weight, density);
      };
      return wrap(NamedTuple, dispatch_histogram(self, _r.tensor(0), _r.optionalTensor(1), _r.toBool(2)));
    }
    case 1: {
      // aten::histogram.bin_ct(Tensor self, int bins=100, *, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor bin_edges)
      
      auto dispatch_histogram = [](const at::Tensor & self, int64_t bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.histogram(bins, range, weight, density);
      };
      return wrap(NamedTuple, dispatch_histogram(self, _r.toInt64(0), _r.doublelistOptional(1), _r.optionalTensor(2), _r.toBool(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// hsplit
static PyObject * THPVariable_hsplit(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "hsplit(int64_t sections)",
    "hsplit(IntArrayRef indices)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::hsplit.int(Tensor(a -> *) self, int sections) -> Tensor(a)[]
      
      auto dispatch_hsplit = [](const at::Tensor & self, int64_t sections) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.hsplit(sections);
      };
      return wrap(dispatch_hsplit(self, _r.toInt64(0)));
    }
    case 1: {
      // aten::hsplit.array(Tensor(a -> *) self, int[] indices) -> Tensor(a)[]
      
      auto dispatch_hsplit = [](const at::Tensor & self, at::IntArrayRef indices) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.hsplit(indices);
      };
      return wrap(dispatch_hsplit(self, _r.intlist(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hypot
static PyObject * THPVariable_hypot(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "hypot(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::hypot(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_hypot = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.hypot(other);
  };
  return wrap(dispatch_hypot(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hypot_
static PyObject * THPVariable_hypot_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "hypot_(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::hypot_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  
  auto dispatch_hypot_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.hypot_(other);
  };
  return wrap(dispatch_hypot_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// i0
static PyObject * THPVariable_i0(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "i0");
  }
  // aten::i0(Tensor self) -> Tensor
  
  auto dispatch_i0 = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.i0();
  };
  return wrap(dispatch_i0(self));
  END_HANDLE_TH_ERRORS
}

// i0_
static PyObject * THPVariable_i0_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "i0_");
  }
  // aten::i0_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_i0_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.i0_();
  };
  return wrap(dispatch_i0_(self));
  END_HANDLE_TH_ERRORS
}

// igamma
static PyObject * THPVariable_igamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "igamma(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::igamma(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_igamma = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.igamma(other);
  };
  return wrap(dispatch_igamma(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// igamma_
static PyObject * THPVariable_igamma_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "igamma_(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::igamma_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  
  auto dispatch_igamma_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.igamma_(other);
  };
  return wrap(dispatch_igamma_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// igammac
static PyObject * THPVariable_igammac(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "igammac(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::igammac(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_igammac = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.igammac(other);
  };
  return wrap(dispatch_igammac(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// igammac_
static PyObject * THPVariable_igammac_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "igammac_(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::igammac_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  
  auto dispatch_igammac_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.igammac_(other);
  };
  return wrap(dispatch_igammac_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// index_add
static PyObject * THPVariable_index_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "index_add(int64_t dim, Tensor index, Tensor source, *, Scalar alpha=1)",
    "index_add(Dimname dim, Tensor index, Tensor source, *, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::index_add(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor
      
      auto dispatch_index_add = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_add(dim, index, source, alpha);
      };
      return wrap(dispatch_index_add(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2), _r.scalar(3)));
    }
    case 1: {
      // aten::index_add.dimname(Tensor self, Dimname dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor
      
      auto dispatch_index_add = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_add(dim, index, source, alpha);
      };
      return wrap(dispatch_index_add(self, _r.dimname(0), _r.tensor(1), _r.tensor(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// index_add_
static PyObject * THPVariable_index_add_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "index_add_(int64_t dim, Tensor index, Tensor source, *, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor(a!)
  
  auto dispatch_index_add_ = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.index_add_(dim, index, source, alpha);
  };
  return wrap(dispatch_index_add_(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2), _r.scalar(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// index_copy
static PyObject * THPVariable_index_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "index_copy(int64_t dim, Tensor index, Tensor source)",
    "index_copy(Dimname dim, Tensor index, Tensor source)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
      
      auto dispatch_index_copy = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_copy(dim, index, source);
      };
      return wrap(dispatch_index_copy(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::index_copy.dimname(Tensor self, Dimname dim, Tensor index, Tensor source) -> Tensor
      
      auto dispatch_index_copy = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_copy(dim, index, source);
      };
      return wrap(dispatch_index_copy(self, _r.dimname(0), _r.tensor(1), _r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// index_copy_
static PyObject * THPVariable_index_copy_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "index_copy_(int64_t dim, Tensor index, Tensor source)",
    "index_copy_(Dimname dim, Tensor index, Tensor source)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
      
      auto dispatch_index_copy_ = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_copy_(dim, index, source);
      };
      return wrap(dispatch_index_copy_(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::index_copy_.dimname(Tensor(a!) self, Dimname dim, Tensor index, Tensor source) -> Tensor(a!)
      
      auto dispatch_index_copy_ = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_copy_(dim, index, source);
      };
      return wrap(dispatch_index_copy_(self, _r.dimname(0), _r.tensor(1), _r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// index_fill
static PyObject * THPVariable_index_fill(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "index_fill(int64_t dim, Tensor index, Tensor value)",
    "index_fill(Dimname dim, Tensor index, Tensor value)",
    "index_fill(int64_t dim, Tensor index, Scalar value)",
    "index_fill(Dimname dim, Tensor index, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::index_fill.int_Tensor(Tensor self, int dim, Tensor index, Tensor value) -> Tensor
      
      auto dispatch_index_fill = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill(dim, index, value);
      };
      return wrap(dispatch_index_fill(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::index_fill.Dimname_Tensor(Tensor self, Dimname dim, Tensor index, Tensor value) -> Tensor
      
      auto dispatch_index_fill = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill(dim, index, value);
      };
      return wrap(dispatch_index_fill(self, _r.dimname(0), _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
      
      auto dispatch_index_fill = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill(dim, index, value);
      };
      return wrap(dispatch_index_fill(self, _r.toInt64(0), _r.tensor(1), _r.scalar(2)));
    }
    case 3: {
      // aten::index_fill.Dimname_Scalar(Tensor self, Dimname dim, Tensor index, Scalar value) -> Tensor
      
      auto dispatch_index_fill = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill(dim, index, value);
      };
      return wrap(dispatch_index_fill(self, _r.dimname(0), _r.tensor(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// index_fill_
static PyObject * THPVariable_index_fill_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "index_fill_(int64_t dim, Tensor index, Tensor value)",
    "index_fill_(Dimname dim, Tensor index, Tensor value)",
    "index_fill_(int64_t dim, Tensor index, Scalar value)",
    "index_fill_(Dimname dim, Tensor index, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::index_fill_.int_Tensor(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)
      
      auto dispatch_index_fill_ = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill_(dim, index, value);
      };
      return wrap(dispatch_index_fill_(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::index_fill_.Dimname_Tensor(Tensor(a!) self, Dimname dim, Tensor index, Tensor value) -> Tensor(a!)
      
      auto dispatch_index_fill_ = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill_(dim, index, value);
      };
      return wrap(dispatch_index_fill_(self, _r.dimname(0), _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::index_fill_.int_Scalar(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)
      
      auto dispatch_index_fill_ = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill_(dim, index, value);
      };
      return wrap(dispatch_index_fill_(self, _r.toInt64(0), _r.tensor(1), _r.scalar(2)));
    }
    case 3: {
      // aten::index_fill_.Dimname_Scalar(Tensor(a!) self, Dimname dim, Tensor index, Scalar value) -> Tensor(a!)
      
      auto dispatch_index_fill_ = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill_(dim, index, value);
      };
      return wrap(dispatch_index_fill_(self, _r.dimname(0), _r.tensor(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// index_put
static PyObject * THPVariable_index_put(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "index_put(c10::List<::std::optional<Tensor>> indices, Tensor values, bool accumulate=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
  
  auto dispatch_index_put = [](const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.index_put(indices, values, accumulate);
  };
  return wrap(dispatch_index_put(self, _r.list_of_optional_tensors(0), _r.tensor(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// index_put_
static PyObject * THPVariable_index_put_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "index_put_(c10::List<::std::optional<Tensor>> indices, Tensor values, bool accumulate=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)
  
  auto dispatch_index_put_ = [](const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.index_put_(indices, values, accumulate);
  };
  return wrap(dispatch_index_put_(self, _r.list_of_optional_tensors(0), _r.tensor(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// index_reduce
static PyObject * THPVariable_index_reduce(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "index_reduce(int64_t dim, Tensor index, Tensor source, c10::string_view reduce, *, bool include_self=True)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::index_reduce(Tensor self, int dim, Tensor index, Tensor source, str reduce, *, bool include_self=True) -> Tensor
  
  auto dispatch_index_reduce = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, c10::string_view reduce, bool include_self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.index_reduce(dim, index, source, reduce, include_self);
  };
  return wrap(dispatch_index_reduce(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2), _r.stringView(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// index_reduce_
static PyObject * THPVariable_index_reduce_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "index_reduce_(int64_t dim, Tensor index, Tensor source, c10::string_view reduce, *, bool include_self=True)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::index_reduce_(Tensor(a!) self, int dim, Tensor index, Tensor source, str reduce, *, bool include_self=True) -> Tensor(a!)
  
  auto dispatch_index_reduce_ = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, c10::string_view reduce, bool include_self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.index_reduce_(dim, index, source, reduce, include_self);
  };
  return wrap(dispatch_index_reduce_(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2), _r.stringView(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// index_select
static PyObject * THPVariable_index_select(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "index_select(int64_t dim, Tensor index)",
    "index_select(Dimname dim, Tensor index)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::index_select(Tensor self, int dim, Tensor index) -> Tensor
      
      auto dispatch_index_select = [](const at::Tensor & self, int64_t dim, const at::Tensor & index) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_select(dim, index);
      };
      return wrap(dispatch_index_select(self, _r.toInt64(0), _r.tensor(1)));
    }
    case 1: {
      // aten::index_select.dimname(Tensor self, Dimname dim, Tensor index) -> Tensor
      
      auto dispatch_index_select = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_select(dim, index);
      };
      return wrap(dispatch_index_select(self, _r.dimname(0), _r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// indices
static PyObject * THPVariable_indices(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "indices");
  }
  // aten::indices(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_indices = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.indices();
  };
  return wrap(dispatch_indices(self));
  END_HANDLE_TH_ERRORS
}

// inner
static PyObject * THPVariable_inner(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "inner(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::inner(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_inner = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.inner(other);
  };
  return wrap(dispatch_inner(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// int_repr
static PyObject * THPVariable_int_repr(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "int_repr");
  }
  // aten::int_repr(Tensor self) -> Tensor
  
  auto dispatch_int_repr = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.int_repr();
  };
  return wrap(dispatch_int_repr(self));
  END_HANDLE_TH_ERRORS
}

// inverse
static PyObject * THPVariable_inverse(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "inverse");
  }
  // aten::inverse(Tensor self) -> Tensor
  
  auto dispatch_inverse = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.inverse();
  };
  return wrap(dispatch_inverse(self));
  END_HANDLE_TH_ERRORS
}

// is_coalesced
static PyObject * THPVariable_is_coalesced(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "is_coalesced");
  }
  // aten::is_coalesced(Tensor self) -> bool
  
  auto dispatch_is_coalesced = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_coalesced();
  };
  return wrap(dispatch_is_coalesced(self));
  END_HANDLE_TH_ERRORS
}

// is_complex
static PyObject * THPVariable_is_complex(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "is_complex");
  }
  // aten::is_complex(Tensor self) -> bool
  
  auto dispatch_is_complex = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_complex();
  };
  return wrap(dispatch_is_complex(self));
  END_HANDLE_TH_ERRORS
}

// is_conj
static PyObject * THPVariable_is_conj(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "is_conj");
  }
  // aten::is_conj(Tensor self) -> bool
  
  auto dispatch_is_conj = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_conj();
  };
  return wrap(dispatch_is_conj(self));
  END_HANDLE_TH_ERRORS
}

// is_distributed
static PyObject * THPVariable_is_distributed(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "is_distributed");
  }
  // aten::is_distributed(Tensor self) -> bool
  
  auto dispatch_is_distributed = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_distributed();
  };
  return wrap(dispatch_is_distributed(self));
  END_HANDLE_TH_ERRORS
}

// is_floating_point
static PyObject * THPVariable_is_floating_point(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "is_floating_point");
  }
  // aten::is_floating_point(Tensor self) -> bool
  
  auto dispatch_is_floating_point = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_floating_point();
  };
  return wrap(dispatch_is_floating_point(self));
  END_HANDLE_TH_ERRORS
}

// is_inference
static PyObject * THPVariable_is_inference(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "is_inference");
  }
  // aten::is_inference(Tensor self) -> bool
  
  auto dispatch_is_inference = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_inference();
  };
  return wrap(dispatch_is_inference(self));
  END_HANDLE_TH_ERRORS
}

// is_neg
static PyObject * THPVariable_is_neg(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "is_neg");
  }
  // aten::is_neg(Tensor self) -> bool
  
  auto dispatch_is_neg = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_neg();
  };
  return wrap(dispatch_is_neg(self));
  END_HANDLE_TH_ERRORS
}

// is_nonzero
static PyObject * THPVariable_is_nonzero(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "is_nonzero");
  }
  // aten::is_nonzero(Tensor self) -> bool
  
  auto dispatch_is_nonzero = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_nonzero();
  };
  return wrap(dispatch_is_nonzero(self));
  END_HANDLE_TH_ERRORS
}

// is_pinned
static PyObject * THPVariable_is_pinned(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "is_pinned(Device? device=None)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::is_pinned(Tensor self, Device? device=None) -> bool
  
  auto dispatch_is_pinned = [](const at::Tensor & self, ::std::optional<at::Device> device) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_pinned(device);
  };
  return wrap(dispatch_is_pinned(self, _r.deviceOptional(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_same_size
static PyObject * THPVariable_is_same_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "is_same_size(Tensor other)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::is_same_size(Tensor self, Tensor other) -> bool
  
  auto dispatch_is_same_size = [](const at::Tensor & self, const at::Tensor & other) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_same_size(other);
  };
  return wrap(dispatch_is_same_size(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_set_to
static PyObject * THPVariable_is_set_to(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "is_set_to(Tensor tensor)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::is_set_to(Tensor self, Tensor tensor) -> bool
  
  auto dispatch_is_set_to = [](const at::Tensor & self, const at::Tensor & tensor) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_set_to(tensor);
  };
  return wrap(dispatch_is_set_to(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_signed
static PyObject * THPVariable_is_signed(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "is_signed");
  }
  // aten::is_signed(Tensor self) -> bool
  
  auto dispatch_is_signed = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_signed();
  };
  return wrap(dispatch_is_signed(self));
  END_HANDLE_TH_ERRORS
}

// isclose
static PyObject * THPVariable_isclose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "isclose(Tensor other, double rtol=1e-05, double atol=1e-08, bool equal_nan=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor
  
  auto dispatch_isclose = [](const at::Tensor & self, const at::Tensor & other, double rtol, double atol, bool equal_nan) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.isclose(other, rtol, atol, equal_nan);
  };
  return wrap(dispatch_isclose(self, _r.tensor(0), _r.toDouble(1), _r.toDouble(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// isfinite
static PyObject * THPVariable_isfinite(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "isfinite");
  }
  // aten::isfinite(Tensor self) -> Tensor
  
  auto dispatch_isfinite = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.isfinite();
  };
  return wrap(dispatch_isfinite(self));
  END_HANDLE_TH_ERRORS
}

// isinf
static PyObject * THPVariable_isinf(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "isinf");
  }
  // aten::isinf(Tensor self) -> Tensor
  
  auto dispatch_isinf = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.isinf();
  };
  return wrap(dispatch_isinf(self));
  END_HANDLE_TH_ERRORS
}

// isnan
static PyObject * THPVariable_isnan(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "isnan");
  }
  // aten::isnan(Tensor self) -> Tensor
  
  auto dispatch_isnan = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.isnan();
  };
  return wrap(dispatch_isnan(self));
  END_HANDLE_TH_ERRORS
}

// isneginf
static PyObject * THPVariable_isneginf(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "isneginf");
  }
  // aten::isneginf(Tensor self) -> Tensor
  
  auto dispatch_isneginf = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.isneginf();
  };
  return wrap(dispatch_isneginf(self));
  END_HANDLE_TH_ERRORS
}

// isposinf
static PyObject * THPVariable_isposinf(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "isposinf");
  }
  // aten::isposinf(Tensor self) -> Tensor
  
  auto dispatch_isposinf = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.isposinf();
  };
  return wrap(dispatch_isposinf(self));
  END_HANDLE_TH_ERRORS
}

// isreal
static PyObject * THPVariable_isreal(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "isreal");
  }
  // aten::isreal(Tensor self) -> Tensor
  
  auto dispatch_isreal = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.isreal();
  };
  return wrap(dispatch_isreal(self));
  END_HANDLE_TH_ERRORS
}

// istft
static PyObject * THPVariable_istft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "istft(int64_t n_fft, int64_t? hop_length=None, int64_t? win_length=None, Tensor? window=None, bool center=True, bool normalized=False, bool? onesided=None, int64_t? length=None, bool return_complex=False)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::istft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool center=True, bool normalized=False, bool? onesided=None, int? length=None, bool return_complex=False) -> Tensor
  
  auto dispatch_istft = [](const at::Tensor & self, int64_t n_fft, ::std::optional<int64_t> hop_length, ::std::optional<int64_t> win_length, const ::std::optional<at::Tensor> & window, bool center, bool normalized, ::std::optional<bool> onesided, ::std::optional<int64_t> length, bool return_complex) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.istft(n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex);
  };
  return wrap(dispatch_istft(self, _r.toInt64(0), _r.toInt64Optional(1), _r.toInt64Optional(2), _r.optionalTensor(3), _r.toBool(4), _r.toBool(5), _r.toBoolOptional(6), _r.toInt64Optional(7), _r.toBool(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// kron
static PyObject * THPVariable_kron(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "kron(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::kron(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_kron = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.kron(other);
  };
  return wrap(dispatch_kron(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// kthvalue
static PyObject * THPVariable_kthvalue(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_kthvalue_structseq();
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "kthvalue(int64_t k, int64_t dim=-1, bool keepdim=False)",
    "kthvalue(int64_t k, Dimname dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
      
      auto dispatch_kthvalue = [](const at::Tensor & self, int64_t k, int64_t dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.kthvalue(k, dim, keepdim);
      };
      return wrap(NamedTuple, dispatch_kthvalue(self, _r.toInt64(0), _r.toInt64(1), _r.toBool(2)));
    }
    case 1: {
      // aten::kthvalue.dimname(Tensor self, int k, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
      
      auto dispatch_kthvalue = [](const at::Tensor & self, int64_t k, at::Dimname dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.kthvalue(k, dim, keepdim);
      };
      return wrap(NamedTuple, dispatch_kthvalue(self, _r.toInt64(0), _r.dimname(1), _r.toBool(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// lcm
static PyObject * THPVariable_lcm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "lcm(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::lcm(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_lcm = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.lcm(other);
  };
  return wrap(dispatch_lcm(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// lcm_
static PyObject * THPVariable_lcm_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "lcm_(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::lcm_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  
  auto dispatch_lcm_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.lcm_(other);
  };
  return wrap(dispatch_lcm_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ldexp
static PyObject * THPVariable_ldexp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "ldexp(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::ldexp.Tensor(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_ldexp = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.ldexp(other);
  };
  return wrap(dispatch_ldexp(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ldexp_
static PyObject * THPVariable_ldexp_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "ldexp_(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::ldexp_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  
  auto dispatch_ldexp_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.ldexp_(other);
  };
  return wrap(dispatch_ldexp_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// le
static PyObject * THPVariable_le(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "le(Tensor other)",
    "le(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::le.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_le = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.le(other);
      };
      return wrap(dispatch_le(self, _r.tensor(0)));
    }
    case 1: {
      // aten::le.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_le = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.le(other);
      };
      return wrap(dispatch_le(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// le_
static PyObject * THPVariable_le_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "le_(Tensor other)",
    "le_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::le_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_le_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.le_(other);
      };
      return wrap(dispatch_le_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::le_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_le_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.le_(other);
      };
      return wrap(dispatch_le_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// lerp
static PyObject * THPVariable_lerp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "lerp(Tensor end, Tensor weight)",
    "lerp(Tensor end, Scalar weight)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor
      
      auto dispatch_lerp = [](const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.lerp(end, weight);
      };
      return wrap(dispatch_lerp(self, _r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor
      
      auto dispatch_lerp = [](const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.lerp(end, weight);
      };
      return wrap(dispatch_lerp(self, _r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// lerp_
static PyObject * THPVariable_lerp_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "lerp_(Tensor end, Tensor weight)",
    "lerp_(Tensor end, Scalar weight)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::lerp_.Tensor(Tensor(a!) self, Tensor end, Tensor weight) -> Tensor(a!)
      
      auto dispatch_lerp_ = [](const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.lerp_(end, weight);
      };
      return wrap(dispatch_lerp_(self, _r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::lerp_.Scalar(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)
      
      auto dispatch_lerp_ = [](const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.lerp_(end, weight);
      };
      return wrap(dispatch_lerp_(self, _r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// less
static PyObject * THPVariable_less(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "less(Tensor other)",
    "less(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::less.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_less = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.less(other);
      };
      return wrap(dispatch_less(self, _r.tensor(0)));
    }
    case 1: {
      // aten::less.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_less = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.less(other);
      };
      return wrap(dispatch_less(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// less_
static PyObject * THPVariable_less_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "less_(Tensor other)",
    "less_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::less_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_less_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.less_(other);
      };
      return wrap(dispatch_less_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::less_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_less_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.less_(other);
      };
      return wrap(dispatch_less_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// less_equal
static PyObject * THPVariable_less_equal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "less_equal(Tensor other)",
    "less_equal(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::less_equal.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_less_equal = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.less_equal(other);
      };
      return wrap(dispatch_less_equal(self, _r.tensor(0)));
    }
    case 1: {
      // aten::less_equal.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_less_equal = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.less_equal(other);
      };
      return wrap(dispatch_less_equal(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// less_equal_
static PyObject * THPVariable_less_equal_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "less_equal_(Tensor other)",
    "less_equal_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::less_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_less_equal_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.less_equal_(other);
      };
      return wrap(dispatch_less_equal_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::less_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_less_equal_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.less_equal_(other);
      };
      return wrap(dispatch_less_equal_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// lgamma
static PyObject * THPVariable_lgamma(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "lgamma");
  }
  // aten::lgamma(Tensor self) -> Tensor
  
  auto dispatch_lgamma = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.lgamma();
  };
  return wrap(dispatch_lgamma(self));
  END_HANDLE_TH_ERRORS
}

// lgamma_
static PyObject * THPVariable_lgamma_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "lgamma_");
  }
  // aten::lgamma_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_lgamma_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.lgamma_();
  };
  return wrap(dispatch_lgamma_(self));
  END_HANDLE_TH_ERRORS
}

// log
static PyObject * THPVariable_log(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "log");
  }
  // aten::log(Tensor self) -> Tensor
  
  auto dispatch_log = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log();
  };
  return wrap(dispatch_log(self));
  END_HANDLE_TH_ERRORS
}

// log10
static PyObject * THPVariable_log10(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "log10");
  }
  // aten::log10(Tensor self) -> Tensor
  
  auto dispatch_log10 = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log10();
  };
  return wrap(dispatch_log10(self));
  END_HANDLE_TH_ERRORS
}

// log10_
static PyObject * THPVariable_log10_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "log10_");
  }
  // aten::log10_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_log10_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log10_();
  };
  return wrap(dispatch_log10_(self));
  END_HANDLE_TH_ERRORS
}

// log1p
static PyObject * THPVariable_log1p(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "log1p");
  }
  // aten::log1p(Tensor self) -> Tensor
  
  auto dispatch_log1p = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log1p();
  };
  return wrap(dispatch_log1p(self));
  END_HANDLE_TH_ERRORS
}

// log1p_
static PyObject * THPVariable_log1p_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "log1p_");
  }
  // aten::log1p_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_log1p_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log1p_();
  };
  return wrap(dispatch_log1p_(self));
  END_HANDLE_TH_ERRORS
}

// log2
static PyObject * THPVariable_log2(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "log2");
  }
  // aten::log2(Tensor self) -> Tensor
  
  auto dispatch_log2 = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log2();
  };
  return wrap(dispatch_log2(self));
  END_HANDLE_TH_ERRORS
}

// log2_
static PyObject * THPVariable_log2_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "log2_");
  }
  // aten::log2_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_log2_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log2_();
  };
  return wrap(dispatch_log2_(self));
  END_HANDLE_TH_ERRORS
}

// log_
static PyObject * THPVariable_log_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "log_");
  }
  // aten::log_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_log_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log_();
  };
  return wrap(dispatch_log_(self));
  END_HANDLE_TH_ERRORS
}

// log_normal_
static PyObject * THPVariable_log_normal_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "log_normal_(double mean=1, double std=2, *, Generator? generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator? generator=None) -> Tensor(a!)
  
  auto dispatch_log_normal_ = [](const at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log_normal_(mean, std, generator);
  };
  return wrap(dispatch_log_normal_(self, _r.toDouble(0), _r.toDouble(1), _r.generator(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// log_softmax
static PyObject * THPVariable_log_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "log_softmax(int64_t dim, ScalarType? dtype=None)",
    "log_softmax(Dimname dim, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_log_softmax = [](const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.log_softmax(dim, dtype);
      };
      return wrap(dispatch_log_softmax(self, _r.toInt64(0), _r.scalartypeOptional(1)));
    }
    case 1: {
      // aten::log_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_log_softmax = [](const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.log_softmax(dim, dtype);
      };
      return wrap(dispatch_log_softmax(self, _r.dimname(0), _r.scalartypeOptional(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logaddexp
static PyObject * THPVariable_logaddexp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "logaddexp(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::logaddexp(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_logaddexp = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logaddexp(other);
  };
  return wrap(dispatch_logaddexp(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logaddexp2
static PyObject * THPVariable_logaddexp2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "logaddexp2(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::logaddexp2(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_logaddexp2 = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logaddexp2(other);
  };
  return wrap(dispatch_logaddexp2(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// logcumsumexp
static PyObject * THPVariable_logcumsumexp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "logcumsumexp(int64_t dim)",
    "logcumsumexp(Dimname dim)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::logcumsumexp(Tensor self, int dim) -> Tensor
      
      auto dispatch_logcumsumexp = [](const at::Tensor & self, int64_t dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.logcumsumexp(dim);
      };
      return wrap(dispatch_logcumsumexp(self, _r.toInt64(0)));
    }
    case 1: {
      // aten::logcumsumexp.dimname(Tensor self, Dimname dim) -> Tensor
      
      auto dispatch_logcumsumexp = [](const at::Tensor & self, at::Dimname dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.logcumsumexp(dim);
      };
      return wrap(dispatch_logcumsumexp(self, _r.dimname(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logdet
static PyObject * THPVariable_logdet(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "logdet");
  }
  // aten::logdet(Tensor self) -> Tensor
  
  auto dispatch_logdet = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logdet();
  };
  return wrap(dispatch_logdet(self));
  END_HANDLE_TH_ERRORS
}

// logical_and
static PyObject * THPVariable_logical_and(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "logical_and(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::logical_and(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_logical_and = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logical_and(other);
  };
  return wrap(dispatch_logical_and(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logical_and_
static PyObject * THPVariable_logical_and_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "logical_and_(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::logical_and_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  
  auto dispatch_logical_and_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logical_and_(other);
  };
  return wrap(dispatch_logical_and_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logical_not
static PyObject * THPVariable_logical_not(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "logical_not");
  }
  // aten::logical_not(Tensor self) -> Tensor
  
  auto dispatch_logical_not = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logical_not();
  };
  return wrap(dispatch_logical_not(self));
  END_HANDLE_TH_ERRORS
}

// logical_not_
static PyObject * THPVariable_logical_not_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "logical_not_");
  }
  // aten::logical_not_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_logical_not_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logical_not_();
  };
  return wrap(dispatch_logical_not_(self));
  END_HANDLE_TH_ERRORS
}

// logical_or
static PyObject * THPVariable_logical_or(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "logical_or(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::logical_or(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_logical_or = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logical_or(other);
  };
  return wrap(dispatch_logical_or(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logical_or_
static PyObject * THPVariable_logical_or_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "logical_or_(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::logical_or_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  
  auto dispatch_logical_or_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logical_or_(other);
  };
  return wrap(dispatch_logical_or_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logical_xor
static PyObject * THPVariable_logical_xor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "logical_xor(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::logical_xor(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_logical_xor = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logical_xor(other);
  };
  return wrap(dispatch_logical_xor(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logical_xor_
static PyObject * THPVariable_logical_xor_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "logical_xor_(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::logical_xor_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  
  auto dispatch_logical_xor_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logical_xor_(other);
  };
  return wrap(dispatch_logical_xor_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logit
static PyObject * THPVariable_logit(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "logit(double? eps=None)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::logit(Tensor self, float? eps=None) -> Tensor
  
  auto dispatch_logit = [](const at::Tensor & self, ::std::optional<double> eps) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logit(eps);
  };
  return wrap(dispatch_logit(self, _r.toDoubleOptional(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logit_
static PyObject * THPVariable_logit_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "logit_(double? eps=None)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::logit_(Tensor(a!) self, float? eps=None) -> Tensor(a!)
  
  auto dispatch_logit_ = [](const at::Tensor & self, ::std::optional<double> eps) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logit_(eps);
  };
  return wrap(dispatch_logit_(self, _r.toDoubleOptional(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// logsumexp
static PyObject * THPVariable_logsumexp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "logsumexp(IntArrayRef[1] dim, bool keepdim=False)",
    "logsumexp(DimnameList[1] dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
      
      auto dispatch_logsumexp = [](const at::Tensor & self, at::IntArrayRef dim, bool keepdim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.logsumexp(dim, keepdim);
      };
      return wrap(dispatch_logsumexp(self, _r.intlist(0), _r.toBool(1)));
    }
    case 1: {
      // aten::logsumexp.names(Tensor self, Dimname[1] dim, bool keepdim=False) -> Tensor
      
      auto dispatch_logsumexp = [](const at::Tensor & self, at::DimnameList dim, bool keepdim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.logsumexp(dim, keepdim);
      };
      return wrap(dispatch_logsumexp(self, _r.dimnamelist(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// lt
static PyObject * THPVariable_lt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "lt(Tensor other)",
    "lt(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::lt.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_lt = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.lt(other);
      };
      return wrap(dispatch_lt(self, _r.tensor(0)));
    }
    case 1: {
      // aten::lt.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_lt = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.lt(other);
      };
      return wrap(dispatch_lt(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// lt_
static PyObject * THPVariable_lt_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "lt_(Tensor other)",
    "lt_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::lt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_lt_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.lt_(other);
      };
      return wrap(dispatch_lt_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::lt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_lt_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.lt_(other);
      };
      return wrap(dispatch_lt_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// lu_solve
static PyObject * THPVariable_lu_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "lu_solve(Tensor LU_data, Tensor LU_pivots)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor
  
  auto dispatch_lu_solve = [](const at::Tensor & self, const at::Tensor & LU_data, const at::Tensor & LU_pivots) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.lu_solve(LU_data, LU_pivots);
  };
  return wrap(dispatch_lu_solve(self, _r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// masked_fill
static PyObject * THPVariable_masked_fill(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "masked_fill(Tensor mask, Tensor value)",
    "masked_fill(Tensor mask, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor
      
      auto dispatch_masked_fill = [](const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.masked_fill(mask, value);
      };
      return wrap(dispatch_masked_fill(self, _r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
      
      auto dispatch_masked_fill = [](const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.masked_fill(mask, value);
      };
      return wrap(dispatch_masked_fill(self, _r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// masked_fill_
static PyObject * THPVariable_masked_fill_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "masked_fill_(Tensor mask, Tensor value)",
    "masked_fill_(Tensor mask, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)
      
      auto dispatch_masked_fill_ = [](const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.masked_fill_(mask, value);
      };
      return wrap(dispatch_masked_fill_(self, _r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)
      
      auto dispatch_masked_fill_ = [](const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.masked_fill_(mask, value);
      };
      return wrap(dispatch_masked_fill_(self, _r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// masked_scatter
static PyObject * THPVariable_masked_scatter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "masked_scatter(Tensor mask, Tensor source)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor
  
  auto dispatch_masked_scatter = [](const at::Tensor & self, const at::Tensor & mask, const at::Tensor & source) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.masked_scatter(mask, source);
  };
  return wrap(dispatch_masked_scatter(self, _r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// masked_scatter_
static PyObject * THPVariable_masked_scatter_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "masked_scatter_(Tensor mask, Tensor source)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)
  
  auto dispatch_masked_scatter_ = [](const at::Tensor & self, const at::Tensor & mask, const at::Tensor & source) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.masked_scatter_(mask, source);
  };
  return wrap(dispatch_masked_scatter_(self, _r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// masked_select
static PyObject * THPVariable_masked_select(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "masked_select(Tensor mask)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::masked_select(Tensor self, Tensor mask) -> Tensor
  
  auto dispatch_masked_select = [](const at::Tensor & self, const at::Tensor & mask) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.masked_select(mask);
  };
  return wrap(dispatch_masked_select(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// matmul
static PyObject * THPVariable_matmul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "matmul(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::matmul(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_matmul = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.matmul(other);
  };
  return wrap(dispatch_matmul(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// matrix_exp
static PyObject * THPVariable_matrix_exp(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "matrix_exp");
  }
  // aten::matrix_exp(Tensor self) -> Tensor
  
  auto dispatch_matrix_exp = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.matrix_exp();
  };
  return wrap(dispatch_matrix_exp(self));
  END_HANDLE_TH_ERRORS
}

// matrix_power
static PyObject * THPVariable_matrix_power(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "matrix_power(int64_t n)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::matrix_power(Tensor self, int n) -> Tensor
  
  auto dispatch_matrix_power = [](const at::Tensor & self, int64_t n) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.matrix_power(n);
  };
  return wrap(dispatch_matrix_power(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// max
static PyObject * THPVariable_max(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_max_structseq();
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "max()",
    "max(Tensor other)",
    "max(int64_t dim, bool keepdim=False)",
    "max(Dimname dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::max(Tensor self) -> Tensor
      
      auto dispatch_max = [](const at::Tensor & self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.max();
      };
      return wrap(dispatch_max(self));
    }
    case 1: {
      // aten::max.other(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_max = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.max(other);
      };
      return wrap(dispatch_max(self, _r.tensor(0)));
    }
    case 2: {
      // aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
      
      auto dispatch_max = [](const at::Tensor & self, int64_t dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.max(dim, keepdim);
      };
      return wrap(NamedTuple, dispatch_max(self, _r.toInt64(0), _r.toBool(1)));
    }
    case 3: {
      // aten::max.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
      
      auto dispatch_max = [](const at::Tensor & self, at::Dimname dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.max(dim, keepdim);
      };
      return wrap(NamedTuple, dispatch_max(self, _r.dimname(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// maximum
static PyObject * THPVariable_maximum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "maximum(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::maximum(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_maximum = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.maximum(other);
  };
  return wrap(dispatch_maximum(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// mean
static PyObject * THPVariable_mean(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "mean(*, ScalarType? dtype=None)",
    "mean(IntArrayRef[1]? dim, bool keepdim=False, *, ScalarType? dtype=None)",
    "mean(DimnameList[1] dim, bool keepdim=False, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_mean = [](const at::Tensor & self, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.mean(dtype);
      };
      return wrap(dispatch_mean(self, _r.scalartypeOptional(0)));
    }
    case 1: {
      // aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_mean = [](const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.mean(dim, keepdim, dtype);
      };
      return wrap(dispatch_mean(self, _r.intlistOptional(0), _r.toBool(1), _r.scalartypeOptional(2)));
    }
    case 2: {
      // aten::mean.names_dim(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_mean = [](const at::Tensor & self, at::DimnameList dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.mean(dim, keepdim, dtype);
      };
      return wrap(dispatch_mean(self, _r.dimnamelist(0), _r.toBool(1), _r.scalartypeOptional(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// median
static PyObject * THPVariable_median(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_median_structseq();
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "median()",
    "median(int64_t dim, bool keepdim=False)",
    "median(Dimname dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::median(Tensor self) -> Tensor
      
      auto dispatch_median = [](const at::Tensor & self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.median();
      };
      return wrap(dispatch_median(self));
    }
    case 1: {
      // aten::median.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
      
      auto dispatch_median = [](const at::Tensor & self, int64_t dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.median(dim, keepdim);
      };
      return wrap(NamedTuple, dispatch_median(self, _r.toInt64(0), _r.toBool(1)));
    }
    case 2: {
      // aten::median.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
      
      auto dispatch_median = [](const at::Tensor & self, at::Dimname dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.median(dim, keepdim);
      };
      return wrap(NamedTuple, dispatch_median(self, _r.dimname(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// min
static PyObject * THPVariable_min(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_min_structseq();
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "min()",
    "min(Tensor other)",
    "min(int64_t dim, bool keepdim=False)",
    "min(Dimname dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::min(Tensor self) -> Tensor
      
      auto dispatch_min = [](const at::Tensor & self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.min();
      };
      return wrap(dispatch_min(self));
    }
    case 1: {
      // aten::min.other(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_min = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.min(other);
      };
      return wrap(dispatch_min(self, _r.tensor(0)));
    }
    case 2: {
      // aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
      
      auto dispatch_min = [](const at::Tensor & self, int64_t dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.min(dim, keepdim);
      };
      return wrap(NamedTuple, dispatch_min(self, _r.toInt64(0), _r.toBool(1)));
    }
    case 3: {
      // aten::min.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
      
      auto dispatch_min = [](const at::Tensor & self, at::Dimname dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.min(dim, keepdim);
      };
      return wrap(NamedTuple, dispatch_min(self, _r.dimname(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// minimum
static PyObject * THPVariable_minimum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "minimum(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::minimum(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_minimum = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.minimum(other);
  };
  return wrap(dispatch_minimum(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mm
static PyObject * THPVariable_mm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "mm(Tensor mat2)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::mm(Tensor self, Tensor mat2) -> Tensor
  
  auto dispatch_mm = [](const at::Tensor & self, const at::Tensor & mat2) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.mm(mat2);
  };
  return wrap(dispatch_mm(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// mode
static PyObject * THPVariable_mode(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_mode_structseq();
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "mode(int64_t dim=-1, bool keepdim=False)",
    "mode(Dimname dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
      
      auto dispatch_mode = [](const at::Tensor & self, int64_t dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.mode(dim, keepdim);
      };
      return wrap(NamedTuple, dispatch_mode(self, _r.toInt64(0), _r.toBool(1)));
    }
    case 1: {
      // aten::mode.dimname(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
      
      auto dispatch_mode = [](const at::Tensor & self, at::Dimname dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.mode(dim, keepdim);
      };
      return wrap(NamedTuple, dispatch_mode(self, _r.dimname(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// moveaxis
static PyObject * THPVariable_moveaxis(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "moveaxis(int64_t source, int64_t destination)",
    "moveaxis(IntArrayRef source, IntArrayRef destination)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::moveaxis.int(Tensor(a) self, int source, int destination) -> Tensor(a)
      
      auto dispatch_moveaxis = [](const at::Tensor & self, int64_t source, int64_t destination) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.moveaxis(source, destination);
      };
      return wrap(dispatch_moveaxis(self, _r.toInt64(0), _r.toInt64(1)));
    }
    case 1: {
      // aten::moveaxis.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)
      
      auto dispatch_moveaxis = [](const at::Tensor & self, at::IntArrayRef source, at::IntArrayRef destination) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.moveaxis(source, destination);
      };
      return wrap(dispatch_moveaxis(self, _r.intlist(0), _r.intlist(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// movedim
static PyObject * THPVariable_movedim(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "movedim(int64_t source, int64_t destination)",
    "movedim(IntArrayRef source, IntArrayRef destination)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::movedim.int(Tensor(a) self, int source, int destination) -> Tensor(a)
      
      auto dispatch_movedim = [](const at::Tensor & self, int64_t source, int64_t destination) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.movedim(source, destination);
      };
      return wrap(dispatch_movedim(self, _r.toInt64(0), _r.toInt64(1)));
    }
    case 1: {
      // aten::movedim.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)
      
      auto dispatch_movedim = [](const at::Tensor & self, at::IntArrayRef source, at::IntArrayRef destination) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.movedim(source, destination);
      };
      return wrap(dispatch_movedim(self, _r.intlist(0), _r.intlist(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// msort
static PyObject * THPVariable_msort(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "msort");
  }
  // aten::msort(Tensor self) -> Tensor
  
  auto dispatch_msort = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.msort();
  };
  return wrap(dispatch_msort(self));
  END_HANDLE_TH_ERRORS
}

// mul
static PyObject * THPVariable_mul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "mul(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::mul.Tensor(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_mul = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.mul(other);
  };
  return wrap(dispatch_mul(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mul_
static PyObject * THPVariable_mul_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "mul_(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
  
  auto dispatch_mul_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.mul_(other);
  };
  return wrap(dispatch_mul_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// multinomial
static PyObject * THPVariable_multinomial(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "multinomial(int64_t num_samples, bool replacement=False, *, Generator? generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor
  
  auto dispatch_multinomial = [](const at::Tensor & self, int64_t num_samples, bool replacement, ::std::optional<at::Generator> generator) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.multinomial(num_samples, replacement, generator);
  };
  return wrap(dispatch_multinomial(self, _r.toInt64(0), _r.toBool(1), _r.generator(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// multiply
static PyObject * THPVariable_multiply(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "multiply(Tensor other)",
    "multiply(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::multiply.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_multiply = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.multiply(other);
      };
      return wrap(dispatch_multiply(self, _r.tensor(0)));
    }
    case 1: {
      // aten::multiply.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_multiply = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.multiply(other);
      };
      return wrap(dispatch_multiply(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// multiply_
static PyObject * THPVariable_multiply_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "multiply_(Tensor other)",
    "multiply_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::multiply_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_multiply_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.multiply_(other);
      };
      return wrap(dispatch_multiply_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::multiply_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_multiply_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.multiply_(other);
      };
      return wrap(dispatch_multiply_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mv
static PyObject * THPVariable_mv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "mv(Tensor vec)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::mv(Tensor self, Tensor vec) -> Tensor
  
  auto dispatch_mv = [](const at::Tensor & self, const at::Tensor & vec) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.mv(vec);
  };
  return wrap(dispatch_mv(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mvlgamma
static PyObject * THPVariable_mvlgamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "mvlgamma(int64_t p)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::mvlgamma(Tensor self, int p) -> Tensor
  
  auto dispatch_mvlgamma = [](const at::Tensor & self, int64_t p) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.mvlgamma(p);
  };
  return wrap(dispatch_mvlgamma(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mvlgamma_
static PyObject * THPVariable_mvlgamma_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "mvlgamma_(int64_t p)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::mvlgamma_(Tensor(a!) self, int p) -> Tensor(a!)
  
  auto dispatch_mvlgamma_ = [](const at::Tensor & self, int64_t p) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.mvlgamma_(p);
  };
  return wrap(dispatch_mvlgamma_(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// nan_to_num
static PyObject * THPVariable_nan_to_num(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "nan_to_num(double? nan=None, double? posinf=None, double? neginf=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor
  
  auto dispatch_nan_to_num = [](const at::Tensor & self, ::std::optional<double> nan, ::std::optional<double> posinf, ::std::optional<double> neginf) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.nan_to_num(nan, posinf, neginf);
  };
  return wrap(dispatch_nan_to_num(self, _r.toDoubleOptional(0), _r.toDoubleOptional(1), _r.toDoubleOptional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// nan_to_num_
static PyObject * THPVariable_nan_to_num_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "nan_to_num_(double? nan=None, double? posinf=None, double? neginf=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::nan_to_num_(Tensor(a!) self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor(a!)
  
  auto dispatch_nan_to_num_ = [](const at::Tensor & self, ::std::optional<double> nan, ::std::optional<double> posinf, ::std::optional<double> neginf) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.nan_to_num_(nan, posinf, neginf);
  };
  return wrap(dispatch_nan_to_num_(self, _r.toDoubleOptional(0), _r.toDoubleOptional(1), _r.toDoubleOptional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// nanmean
static PyObject * THPVariable_nanmean(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "nanmean(IntArrayRef[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::nanmean(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  
  auto dispatch_nanmean = [](const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.nanmean(dim, keepdim, dtype);
  };
  return wrap(dispatch_nanmean(self, _r.intlistOptional(0), _r.toBool(1), _r.scalartypeOptional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// nanmedian
static PyObject * THPVariable_nanmedian(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_nanmedian_structseq();
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "nanmedian()",
    "nanmedian(int64_t dim, bool keepdim=False)",
    "nanmedian(Dimname dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::nanmedian(Tensor self) -> Tensor
      
      auto dispatch_nanmedian = [](const at::Tensor & self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.nanmedian();
      };
      return wrap(dispatch_nanmedian(self));
    }
    case 1: {
      // aten::nanmedian.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
      
      auto dispatch_nanmedian = [](const at::Tensor & self, int64_t dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.nanmedian(dim, keepdim);
      };
      return wrap(NamedTuple, dispatch_nanmedian(self, _r.toInt64(0), _r.toBool(1)));
    }
    case 2: {
      // aten::nanmedian.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
      
      auto dispatch_nanmedian = [](const at::Tensor & self, at::Dimname dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.nanmedian(dim, keepdim);
      };
      return wrap(NamedTuple, dispatch_nanmedian(self, _r.dimname(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// nanquantile
static PyObject * THPVariable_nanquantile(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "nanquantile(Tensor q, int64_t? dim=None, bool keepdim=False, *, c10::string_view interpolation=\"linear\")",
    "nanquantile(double q, int64_t? dim=None, bool keepdim=False, *, c10::string_view interpolation=\"linear\")",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::nanquantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation='linear') -> Tensor
      
      auto dispatch_nanquantile = [](const at::Tensor & self, const at::Tensor & q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.nanquantile(q, dim, keepdim, interpolation);
      };
      return wrap(dispatch_nanquantile(self, _r.tensor(0), _r.toInt64Optional(1), _r.toBool(2), _r.stringView(3)));
    }
    case 1: {
      // aten::nanquantile.scalar(Tensor self, float q, int? dim=None, bool keepdim=False, *, str interpolation='linear') -> Tensor
      
      auto dispatch_nanquantile = [](const at::Tensor & self, double q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.nanquantile(q, dim, keepdim, interpolation);
      };
      return wrap(dispatch_nanquantile(self, _r.toDouble(0), _r.toInt64Optional(1), _r.toBool(2), _r.stringView(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// nansum
static PyObject * THPVariable_nansum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "nansum(IntArrayRef[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::nansum(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  
  auto dispatch_nansum = [](const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.nansum(dim, keepdim, dtype);
  };
  return wrap(dispatch_nansum(self, _r.intlistOptional(0), _r.toBool(1), _r.scalartypeOptional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// narrow
static PyObject * THPVariable_narrow(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "narrow(int64_t dim, Tensor start, SymInt length)",
    "narrow(int64_t dim, SymInt start, SymInt length)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, SymInt length) -> Tensor(a)
      
      auto dispatch_narrow = [](const at::Tensor & self, int64_t dim, const at::Tensor & start, c10::SymInt length) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.narrow_symint(dim, start, length);
      };
      return wrap(dispatch_narrow(self, _r.toInt64(0), _r.tensor(1), _r.toSymInt(2)));
    }
    case 1: {
      // aten::narrow(Tensor(a) self, int dim, SymInt start, SymInt length) -> Tensor(a)
      
      auto dispatch_narrow = [](const at::Tensor & self, int64_t dim, c10::SymInt start, c10::SymInt length) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.narrow_symint(dim, start, length);
      };
      return wrap(dispatch_narrow(self, _r.toInt64(0), _r.toSymInt(1), _r.toSymInt(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// narrow_copy
static PyObject * THPVariable_narrow_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "narrow_copy(int64_t dim, SymInt start, SymInt length)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::narrow_copy(Tensor self, int dim, SymInt start, SymInt length) -> Tensor
  
  auto dispatch_narrow_copy = [](const at::Tensor & self, int64_t dim, c10::SymInt start, c10::SymInt length) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.narrow_copy_symint(dim, start, length);
  };
  return wrap(dispatch_narrow_copy(self, _r.toInt64(0), _r.toSymInt(1), _r.toSymInt(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// ne
static PyObject * THPVariable_ne(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "ne(Tensor other)",
    "ne(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::ne.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_ne = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.ne(other);
      };
      return wrap(dispatch_ne(self, _r.tensor(0)));
    }
    case 1: {
      // aten::ne.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_ne = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.ne(other);
      };
      return wrap(dispatch_ne(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// ne_
static PyObject * THPVariable_ne_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "ne_(Tensor other)",
    "ne_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::ne_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_ne_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.ne_(other);
      };
      return wrap(dispatch_ne_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::ne_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_ne_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.ne_(other);
      };
      return wrap(dispatch_ne_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// neg
static PyObject * THPVariable_neg(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "neg");
  }
  // aten::neg(Tensor self) -> Tensor
  
  auto dispatch_neg = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.neg();
  };
  return wrap(dispatch_neg(self));
  END_HANDLE_TH_ERRORS
}

// neg_
static PyObject * THPVariable_neg_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "neg_");
  }
  // aten::neg_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_neg_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.neg_();
  };
  return wrap(dispatch_neg_(self));
  END_HANDLE_TH_ERRORS
}

// negative
static PyObject * THPVariable_negative(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "negative");
  }
  // aten::negative(Tensor self) -> Tensor
  
  auto dispatch_negative = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.negative();
  };
  return wrap(dispatch_negative(self));
  END_HANDLE_TH_ERRORS
}

// negative_
static PyObject * THPVariable_negative_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "negative_");
  }
  // aten::negative_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_negative_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.negative_();
  };
  return wrap(dispatch_negative_(self));
  END_HANDLE_TH_ERRORS
}

// new_empty
static PyObject * THPVariable_new_empty(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "new_empty(SymIntArrayRef size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::new_empty(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(1))
      .device(_r.deviceOptional(3))
      .layout(_r.layoutOptional(2))
      .requires_grad(_r.toBool(5))
      .pinned_memory(_r.toBool(4));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch_new_empty = [](const at::Tensor & self, c10::SymIntArrayRef size, at::TensorOptions options) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.new_empty_symint(size, options);
  };
  return wrap(dispatch_new_empty(self, _r.symintlist(0), options).set_requires_grad(_r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// new_empty_strided
static PyObject * THPVariable_new_empty_strided(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "new_empty_strided(SymIntArrayRef size, SymIntArrayRef stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::new_empty_strided(Tensor self, SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(2))
      .device(_r.deviceOptional(4))
      .layout(_r.layoutOptional(3))
      .requires_grad(_r.toBool(6))
      .pinned_memory(_r.toBool(5));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch_new_empty_strided = [](const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, at::TensorOptions options) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.new_empty_strided_symint(size, stride, options);
  };
  return wrap(dispatch_new_empty_strided(self, _r.symintlist(0), _r.symintlist(1), options).set_requires_grad(_r.toBool(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// new_full
static PyObject * THPVariable_new_full(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "new_full(SymIntArrayRef size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::new_full(Tensor self, SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(2))
      .device(_r.deviceOptional(4))
      .layout(_r.layoutOptional(3))
      .requires_grad(_r.toBool(6))
      .pinned_memory(_r.toBool(5));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch_new_full = [](const at::Tensor & self, c10::SymIntArrayRef size, const at::Scalar & fill_value, at::TensorOptions options) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.new_full_symint(size, fill_value, options);
  };
  return wrap(dispatch_new_full(self, _r.symintlist(0), _r.scalar(1), options).set_requires_grad(_r.toBool(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// new_ones
static PyObject * THPVariable_new_ones(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "new_ones(SymIntArrayRef size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::new_ones(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(1))
      .device(_r.deviceOptional(3))
      .layout(_r.layoutOptional(2))
      .requires_grad(_r.toBool(5))
      .pinned_memory(_r.toBool(4));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch_new_ones = [](const at::Tensor & self, c10::SymIntArrayRef size, at::TensorOptions options) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.new_ones_symint(size, options);
  };
  return wrap(dispatch_new_ones(self, _r.symintlist(0), options).set_requires_grad(_r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// new_zeros
static PyObject * THPVariable_new_zeros(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "new_zeros(SymIntArrayRef size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::new_zeros(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(1))
      .device(_r.deviceOptional(3))
      .layout(_r.layoutOptional(2))
      .requires_grad(_r.toBool(5))
      .pinned_memory(_r.toBool(4));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch_new_zeros = [](const at::Tensor & self, c10::SymIntArrayRef size, at::TensorOptions options) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.new_zeros_symint(size, options);
  };
  return wrap(dispatch_new_zeros(self, _r.symintlist(0), options).set_requires_grad(_r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// nextafter
static PyObject * THPVariable_nextafter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "nextafter(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::nextafter(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_nextafter = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.nextafter(other);
  };
  return wrap(dispatch_nextafter(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// nextafter_
static PyObject * THPVariable_nextafter_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "nextafter_(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::nextafter_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  
  auto dispatch_nextafter_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.nextafter_(other);
  };
  return wrap(dispatch_nextafter_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// nonzero_static
static PyObject * THPVariable_nonzero_static(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "nonzero_static(*, SymInt size, int64_t fill_value=-1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::nonzero_static(Tensor self, *, SymInt size, int fill_value=-1) -> Tensor
  
  auto dispatch_nonzero_static = [](const at::Tensor & self, c10::SymInt size, int64_t fill_value) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.nonzero_static_symint(size, fill_value);
  };
  return wrap(dispatch_nonzero_static(self, _r.toSymInt(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// norm
static PyObject * THPVariable_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "norm(Scalar p=2)",
    "norm(Scalar? p, *, ScalarType dtype)",
    "norm(Scalar? p, IntArrayRef[1] dim, bool keepdim, *, ScalarType dtype)",
    "norm(Scalar? p, IntArrayRef[1] dim, bool keepdim=False)",
    "norm(Scalar? p, DimnameList[1] dim, bool keepdim, *, ScalarType dtype)",
    "norm(Scalar? p, DimnameList[1] dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor
      
      auto dispatch_norm = [](const at::Tensor & self, const at::Scalar & p) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.norm(p);
      };
      return wrap(dispatch_norm(self, _r.scalar(0)));
    }
    case 1: {
      // aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor
      
      auto dispatch_norm = [](const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::ScalarType dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.norm(p, dtype);
      };
      return wrap(dispatch_norm(self, _r.scalarOptional(0), _r.scalartype(1)));
    }
    case 2: {
      // aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
      
      auto dispatch_norm = [](const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.norm(p, dim, keepdim, dtype);
      };
      return wrap(dispatch_norm(self, _r.scalarOptional(0), _r.intlist(1), _r.toBool(2), _r.scalartype(3)));
    }
    case 3: {
      // aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor
      
      auto dispatch_norm = [](const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.norm(p, dim, keepdim);
      };
      return wrap(dispatch_norm(self, _r.scalarOptional(0), _r.intlist(1), _r.toBool(2)));
    }
    case 4: {
      // aten::norm.names_ScalarOpt_dim_dtype(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
      
      auto dispatch_norm = [](const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::ScalarType dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.norm(p, dim, keepdim, dtype);
      };
      return wrap(dispatch_norm(self, _r.scalarOptional(0), _r.dimnamelist(1), _r.toBool(2), _r.scalartype(3)));
    }
    case 5: {
      // aten::norm.names_ScalarOpt_dim(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim=False) -> Tensor
      
      auto dispatch_norm = [](const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.norm(p, dim, keepdim);
      };
      return wrap(dispatch_norm(self, _r.scalarOptional(0), _r.dimnamelist(1), _r.toBool(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// normal_
static PyObject * THPVariable_normal_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "normal_(double mean=0, double std=1, *, Generator? generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)
  
  auto dispatch_normal_ = [](const at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.normal_(mean, std, generator);
  };
  return wrap(dispatch_normal_(self, _r.toDouble(0), _r.toDouble(1), _r.generator(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// not_equal
static PyObject * THPVariable_not_equal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "not_equal(Tensor other)",
    "not_equal(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::not_equal.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_not_equal = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.not_equal(other);
      };
      return wrap(dispatch_not_equal(self, _r.tensor(0)));
    }
    case 1: {
      // aten::not_equal.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_not_equal = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.not_equal(other);
      };
      return wrap(dispatch_not_equal(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// not_equal_
static PyObject * THPVariable_not_equal_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "not_equal_(Tensor other)",
    "not_equal_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::not_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_not_equal_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.not_equal_(other);
      };
      return wrap(dispatch_not_equal_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::not_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_not_equal_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.not_equal_(other);
      };
      return wrap(dispatch_not_equal_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// orgqr
static PyObject * THPVariable_orgqr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "orgqr(Tensor input2)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::orgqr(Tensor self, Tensor input2) -> Tensor
  
  auto dispatch_orgqr = [](const at::Tensor & self, const at::Tensor & input2) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.orgqr(input2);
  };
  return wrap(dispatch_orgqr(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ormqr
static PyObject * THPVariable_ormqr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "ormqr(Tensor input2, Tensor input3, bool left=True, bool transpose=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor
  
  auto dispatch_ormqr = [](const at::Tensor & self, const at::Tensor & input2, const at::Tensor & input3, bool left, bool transpose) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.ormqr(input2, input3, left, transpose);
  };
  return wrap(dispatch_ormqr(self, _r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// outer
static PyObject * THPVariable_outer(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "outer(Tensor vec2)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::outer(Tensor self, Tensor vec2) -> Tensor
  
  auto dispatch_outer = [](const at::Tensor & self, const at::Tensor & vec2) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.outer(vec2);
  };
  return wrap(dispatch_outer(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// permute
static PyObject * THPVariable_permute(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "permute(IntArrayRef dims)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)
  
  auto dispatch_permute = [](const at::Tensor & self, at::IntArrayRef dims) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.permute(dims);
  };
  return wrap(dispatch_permute(self, _r.intlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// pin_memory
static PyObject * THPVariable_pin_memory(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "pin_memory(Device? device=None)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::pin_memory(Tensor(a) self, Device? device=None) -> Tensor(a)
  
  auto dispatch_pin_memory = [](const at::Tensor & self, ::std::optional<at::Device> device) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.pin_memory(device);
  };
  return wrap(dispatch_pin_memory(self, _r.deviceOptional(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// pinverse
static PyObject * THPVariable_pinverse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "pinverse(double rcond=1e-15)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::pinverse(Tensor self, float rcond=1e-15) -> Tensor
  
  auto dispatch_pinverse = [](const at::Tensor & self, double rcond) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.pinverse(rcond);
  };
  return wrap(dispatch_pinverse(self, _r.toDouble(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// polygamma
static PyObject * THPVariable_polygamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "polygamma(int64_t n)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::polygamma(int n, Tensor self) -> Tensor
  
  auto dispatch_polygamma = [](int64_t n, const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.polygamma(n);
  };
  return wrap(dispatch_polygamma(_r.toInt64(0), self));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// polygamma_
static PyObject * THPVariable_polygamma_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "polygamma_(int64_t n)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::polygamma_(Tensor(a!) self, int n) -> Tensor(a!)
  
  auto dispatch_polygamma_ = [](const at::Tensor & self, int64_t n) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.polygamma_(n);
  };
  return wrap(dispatch_polygamma_(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// positive
static PyObject * THPVariable_positive(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "positive");
  }
  // aten::positive(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_positive = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.positive();
  };
  return wrap(dispatch_positive(self));
  END_HANDLE_TH_ERRORS
}

\
// pow
static PyObject * THPVariable_pow(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "pow(Tensor exponent)",
    "pow(Scalar exponent)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
      
      auto dispatch_pow = [](const at::Tensor & self, const at::Tensor & exponent) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.pow(exponent);
      };
      return wrap(dispatch_pow(self, _r.tensor(0)));
    }
    case 1: {
      // aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
      
      auto dispatch_pow = [](const at::Tensor & self, const at::Scalar & exponent) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.pow(exponent);
      };
      return wrap(dispatch_pow(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// pow_
static PyObject * THPVariable_pow_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "pow_(Tensor exponent)",
    "pow_(Scalar exponent)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)
      
      auto dispatch_pow_ = [](const at::Tensor & self, const at::Tensor & exponent) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.pow_(exponent);
      };
      return wrap(dispatch_pow_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)
      
      auto dispatch_pow_ = [](const at::Tensor & self, const at::Scalar & exponent) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.pow_(exponent);
      };
      return wrap(dispatch_pow_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// prelu
static PyObject * THPVariable_prelu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "prelu(Tensor weight)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::prelu(Tensor self, Tensor weight) -> Tensor
  
  auto dispatch_prelu = [](const at::Tensor & self, const at::Tensor & weight) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.prelu(weight);
  };
  return wrap(dispatch_prelu(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// prod
static PyObject * THPVariable_prod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "prod(*, ScalarType? dtype=None)",
    "prod(int64_t dim, bool keepdim=False, *, ScalarType? dtype=None)",
    "prod(Dimname dim, bool keepdim=False, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_prod = [](const at::Tensor & self, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.prod(dtype);
      };
      return wrap(dispatch_prod(self, _r.scalartypeOptional(0)));
    }
    case 1: {
      // aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_prod = [](const at::Tensor & self, int64_t dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.prod(dim, keepdim, dtype);
      };
      return wrap(dispatch_prod(self, _r.toInt64(0), _r.toBool(1), _r.scalartypeOptional(2)));
    }
    case 2: {
      // aten::prod.dim_Dimname(Tensor self, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_prod = [](const at::Tensor & self, at::Dimname dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.prod(dim, keepdim, dtype);
      };
      return wrap(dispatch_prod(self, _r.dimname(0), _r.toBool(1), _r.scalartypeOptional(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// put
static PyObject * THPVariable_put(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "put(Tensor index, Tensor source, bool accumulate=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::put(Tensor self, Tensor index, Tensor source, bool accumulate=False) -> Tensor
  
  auto dispatch_put = [](const at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.put(index, source, accumulate);
  };
  return wrap(dispatch_put(self, _r.tensor(0), _r.tensor(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// put_
static PyObject * THPVariable_put_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "put_(Tensor index, Tensor source, bool accumulate=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::put_(Tensor(a!) self, Tensor index, Tensor source, bool accumulate=False) -> Tensor(a!)
  
  auto dispatch_put_ = [](const at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.put_(index, source, accumulate);
  };
  return wrap(dispatch_put_(self, _r.tensor(0), _r.tensor(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// q_per_channel_axis
static PyObject * THPVariable_q_per_channel_axis(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "q_per_channel_axis");
  }
  // aten::q_per_channel_axis(Tensor self) -> int
  
  auto dispatch_q_per_channel_axis = [](const at::Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return self.q_per_channel_axis();
  };
  return wrap(dispatch_q_per_channel_axis(self));
  END_HANDLE_TH_ERRORS
}

// q_per_channel_scales
static PyObject * THPVariable_q_per_channel_scales(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "q_per_channel_scales");
  }
  // aten::q_per_channel_scales(Tensor self) -> Tensor
  
  auto dispatch_q_per_channel_scales = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.q_per_channel_scales();
  };
  return wrap(dispatch_q_per_channel_scales(self));
  END_HANDLE_TH_ERRORS
}

// q_per_channel_zero_points
static PyObject * THPVariable_q_per_channel_zero_points(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "q_per_channel_zero_points");
  }
  // aten::q_per_channel_zero_points(Tensor self) -> Tensor
  
  auto dispatch_q_per_channel_zero_points = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.q_per_channel_zero_points();
  };
  return wrap(dispatch_q_per_channel_zero_points(self));
  END_HANDLE_TH_ERRORS
}

// q_scale
static PyObject * THPVariable_q_scale(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "q_scale");
  }
  // aten::q_scale(Tensor self) -> float
  
  auto dispatch_q_scale = [](const at::Tensor & self) -> double {
    pybind11::gil_scoped_release no_gil;
    return self.q_scale();
  };
  return wrap(dispatch_q_scale(self));
  END_HANDLE_TH_ERRORS
}

// q_zero_point
static PyObject * THPVariable_q_zero_point(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "q_zero_point");
  }
  // aten::q_zero_point(Tensor self) -> int
  
  auto dispatch_q_zero_point = [](const at::Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return self.q_zero_point();
  };
  return wrap(dispatch_q_zero_point(self));
  END_HANDLE_TH_ERRORS
}

// qr
static PyObject * THPVariable_qr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_qr_structseq();
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "qr(bool some=True)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::qr(Tensor self, bool some=True) -> (Tensor Q, Tensor R)
  
  auto dispatch_qr = [](const at::Tensor & self, bool some) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.qr(some);
  };
  return wrap(NamedTuple, dispatch_qr(self, _r.toBool(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// qscheme
static PyObject * THPVariable_qscheme(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "qscheme");
  }
  // aten::qscheme(Tensor self) -> QScheme
  
  auto dispatch_qscheme = [](const at::Tensor & self) -> at::QScheme {
    pybind11::gil_scoped_release no_gil;
    return self.qscheme();
  };
  return wrap(dispatch_qscheme(self));
  END_HANDLE_TH_ERRORS
}

\
// quantile
static PyObject * THPVariable_quantile(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "quantile(Tensor q, int64_t? dim=None, bool keepdim=False, *, c10::string_view interpolation=\"linear\")",
    "quantile(double q, int64_t? dim=None, bool keepdim=False, *, c10::string_view interpolation=\"linear\")",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::quantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation='linear') -> Tensor
      
      auto dispatch_quantile = [](const at::Tensor & self, const at::Tensor & q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.quantile(q, dim, keepdim, interpolation);
      };
      return wrap(dispatch_quantile(self, _r.tensor(0), _r.toInt64Optional(1), _r.toBool(2), _r.stringView(3)));
    }
    case 1: {
      // aten::quantile.scalar(Tensor self, float q, int? dim=None, bool keepdim=False, *, str interpolation='linear') -> Tensor
      
      auto dispatch_quantile = [](const at::Tensor & self, double q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.quantile(q, dim, keepdim, interpolation);
      };
      return wrap(dispatch_quantile(self, _r.toDouble(0), _r.toInt64Optional(1), _r.toBool(2), _r.stringView(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rad2deg
static PyObject * THPVariable_rad2deg(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "rad2deg");
  }
  // aten::rad2deg(Tensor self) -> Tensor
  
  auto dispatch_rad2deg = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.rad2deg();
  };
  return wrap(dispatch_rad2deg(self));
  END_HANDLE_TH_ERRORS
}

// rad2deg_
static PyObject * THPVariable_rad2deg_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "rad2deg_");
  }
  // aten::rad2deg_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_rad2deg_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.rad2deg_();
  };
  return wrap(dispatch_rad2deg_(self));
  END_HANDLE_TH_ERRORS
}

\
// random_
static PyObject * THPVariable_random_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "random_(*, Generator? generator=None)",
    "random_(int64_t from, int64_t? to, *, Generator? generator=None)",
    "random_(int64_t to, *, Generator? generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)
      
      auto dispatch_random_ = [](const at::Tensor & self, ::std::optional<at::Generator> generator) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.random_(generator);
      };
      return wrap(dispatch_random_(self, _r.generator(0)));
    }
    case 1: {
      // aten::random_.from(Tensor(a!) self, int from, int? to, *, Generator? generator=None) -> Tensor(a!)
      
      auto dispatch_random_ = [](const at::Tensor & self, int64_t from, ::std::optional<int64_t> to, ::std::optional<at::Generator> generator) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.random_(from, to, generator);
      };
      return wrap(dispatch_random_(self, _r.toInt64(0), _r.toInt64Optional(1), _r.generator(2)));
    }
    case 2: {
      // aten::random_.to(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)
      
      auto dispatch_random_ = [](const at::Tensor & self, int64_t to, ::std::optional<at::Generator> generator) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.random_(to, generator);
      };
      return wrap(dispatch_random_(self, _r.toInt64(0), _r.generator(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ravel
static PyObject * THPVariable_ravel(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "ravel");
  }
  // aten::ravel(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_ravel = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.ravel();
  };
  return wrap(dispatch_ravel(self));
  END_HANDLE_TH_ERRORS
}

// reciprocal
static PyObject * THPVariable_reciprocal(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "reciprocal");
  }
  // aten::reciprocal(Tensor self) -> Tensor
  
  auto dispatch_reciprocal = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.reciprocal();
  };
  return wrap(dispatch_reciprocal(self));
  END_HANDLE_TH_ERRORS
}

// reciprocal_
static PyObject * THPVariable_reciprocal_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "reciprocal_");
  }
  // aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_reciprocal_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.reciprocal_();
  };
  return wrap(dispatch_reciprocal_(self));
  END_HANDLE_TH_ERRORS
}

// record_stream
static PyObject * THPVariable_record_stream(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "record_stream(Stream s)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::record_stream(Tensor(a!) self, Stream s) -> ()
  
  auto dispatch_record_stream = [](const at::Tensor & self, at::Stream s) -> void {
    pybind11::gil_scoped_release no_gil;
    self.record_stream(s);
  };
  dispatch_record_stream(self, _r.stream(0));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// refine_names
static PyObject * THPVariable_refine_names(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "refine_names(DimnameList names)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::refine_names(Tensor(a) self, Dimname[] names) -> Tensor(a)
  
  auto dispatch_refine_names = [](const at::Tensor & self, at::DimnameList names) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.refine_names(names);
  };
  return wrap(dispatch_refine_names(self, _r.dimnamelist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// relu
static PyObject * THPVariable_relu(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "relu");
  }
  // aten::relu(Tensor self) -> Tensor
  
  auto dispatch_relu = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.relu();
  };
  return wrap(dispatch_relu(self));
  END_HANDLE_TH_ERRORS
}

// relu_
static PyObject * THPVariable_relu_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "relu_");
  }
  // aten::relu_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_relu_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.relu_();
  };
  return wrap(dispatch_relu_(self));
  END_HANDLE_TH_ERRORS
}

\
// remainder
static PyObject * THPVariable_remainder(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "remainder(Tensor other)",
    "remainder(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_remainder = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.remainder(other);
      };
      return wrap(dispatch_remainder(self, _r.tensor(0)));
    }
    case 1: {
      // aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_remainder = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.remainder(other);
      };
      return wrap(dispatch_remainder(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// remainder_
static PyObject * THPVariable_remainder_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "remainder_(Tensor other)",
    "remainder_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::remainder_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_remainder_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.remainder_(other);
      };
      return wrap(dispatch_remainder_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::remainder_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_remainder_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.remainder_(other);
      };
      return wrap(dispatch_remainder_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rename
static PyObject * THPVariable_rename(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "rename(DimnameList? names)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::rename(Tensor(a) self, Dimname[]? names) -> Tensor(a)
  auto __names = _r.toDimnameListOptional(0);
  ::std::optional<DimnameList> names = __names ? ::std::make_optional(DimnameList(__names.value())) : ::std::nullopt;
  auto dispatch_rename = [](const at::Tensor & self, ::std::optional<at::DimnameList> names) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.rename(names);
  };
  return wrap(dispatch_rename(self, names));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rename_
static PyObject * THPVariable_rename_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "rename_(DimnameList? names)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::rename_(Tensor(a!) self, Dimname[]? names) -> Tensor(a!)
  auto __names = _r.toDimnameListOptional(0);
  ::std::optional<DimnameList> names = __names ? ::std::make_optional(DimnameList(__names.value())) : ::std::nullopt;
  auto dispatch_rename_ = [](const at::Tensor & self, ::std::optional<at::DimnameList> names) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.rename_(names);
  };
  return wrap(dispatch_rename_(self, names));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// renorm
static PyObject * THPVariable_renorm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "renorm(Scalar p, int64_t dim, Scalar maxnorm)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor
  
  auto dispatch_renorm = [](const at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.renorm(p, dim, maxnorm);
  };
  return wrap(dispatch_renorm(self, _r.scalar(0), _r.toInt64(1), _r.scalar(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// renorm_
static PyObject * THPVariable_renorm_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "renorm_(Scalar p, int64_t dim, Scalar maxnorm)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::renorm_(Tensor(a!) self, Scalar p, int dim, Scalar maxnorm) -> Tensor(a!)
  
  auto dispatch_renorm_ = [](const at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.renorm_(p, dim, maxnorm);
  };
  return wrap(dispatch_renorm_(self, _r.scalar(0), _r.toInt64(1), _r.scalar(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// repeat
static PyObject * THPVariable_repeat(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "repeat(SymIntArrayRef repeats)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::repeat(Tensor self, SymInt[] repeats) -> Tensor
  
  auto dispatch_repeat = [](const at::Tensor & self, c10::SymIntArrayRef repeats) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.repeat_symint(repeats);
  };
  return wrap(dispatch_repeat(self, _r.symintlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// repeat_interleave
static PyObject * THPVariable_repeat_interleave(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "repeat_interleave(Tensor repeats, int64_t? dim=None, *, SymInt? output_size=None)",
    "repeat_interleave(SymInt repeats, int64_t? dim=None, *, SymInt? output_size=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None, *, SymInt? output_size=None) -> Tensor
      
      auto dispatch_repeat_interleave = [](const at::Tensor & self, const at::Tensor & repeats, ::std::optional<int64_t> dim, ::std::optional<c10::SymInt> output_size) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.repeat_interleave_symint(repeats, dim, output_size);
      };
      return wrap(dispatch_repeat_interleave(self, _r.tensor(0), _r.toInt64Optional(1), _r.toSymIntOptional(2)));
    }
    case 1: {
      // aten::repeat_interleave.self_int(Tensor self, SymInt repeats, int? dim=None, *, SymInt? output_size=None) -> Tensor
      
      auto dispatch_repeat_interleave = [](const at::Tensor & self, c10::SymInt repeats, ::std::optional<int64_t> dim, ::std::optional<c10::SymInt> output_size) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.repeat_interleave_symint(repeats, dim, output_size);
      };
      return wrap(dispatch_repeat_interleave(self, _r.toSymInt(0), _r.toInt64Optional(1), _r.toSymIntOptional(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// reshape
static PyObject * THPVariable_reshape(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "reshape(SymIntArrayRef shape)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)
  
  auto dispatch_reshape = [](const at::Tensor & self, c10::SymIntArrayRef shape) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.reshape_symint(shape);
  };
  return wrap(dispatch_reshape(self, _r.symintlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// reshape_as
static PyObject * THPVariable_reshape_as(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "reshape_as(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::reshape_as(Tensor(a) self, Tensor other) -> Tensor(a)
  
  auto dispatch_reshape_as = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.reshape_as(other);
  };
  return wrap(dispatch_reshape_as(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// resize_
static PyObject * THPVariable_resize_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "resize_(SymIntArrayRef size, *, MemoryFormat? memory_format=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::resize_(Tensor(a!) self, SymInt[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)
  
  auto dispatch_resize_ = [](const at::Tensor & self, c10::SymIntArrayRef size, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.resize__symint(size, memory_format);
  };
  return wrap(dispatch_resize_(self, _r.symintlist(0), _r.memoryformatOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// resize_as_
static PyObject * THPVariable_resize_as_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "resize_as_(Tensor the_template, *, MemoryFormat? memory_format=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::resize_as_(Tensor(a!) self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor(a!)
  
  auto dispatch_resize_as_ = [](const at::Tensor & self, const at::Tensor & the_template, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.resize_as_(the_template, memory_format);
  };
  return wrap(dispatch_resize_as_(self, _r.tensor(0), _r.memoryformatOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// resize_as_sparse_
static PyObject * THPVariable_resize_as_sparse_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "resize_as_sparse_(Tensor the_template)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::resize_as_sparse_(Tensor(a!) self, Tensor the_template) -> Tensor(a!)
  
  auto dispatch_resize_as_sparse_ = [](const at::Tensor & self, const at::Tensor & the_template) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.resize_as_sparse_(the_template);
  };
  return wrap(dispatch_resize_as_sparse_(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// resolve_conj
static PyObject * THPVariable_resolve_conj(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "resolve_conj");
  }
  // aten::resolve_conj(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_resolve_conj = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.resolve_conj();
  };
  return wrap(dispatch_resolve_conj(self));
  END_HANDLE_TH_ERRORS
}

// resolve_neg
static PyObject * THPVariable_resolve_neg(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "resolve_neg");
  }
  // aten::resolve_neg(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_resolve_neg = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.resolve_neg();
  };
  return wrap(dispatch_resolve_neg(self));
  END_HANDLE_TH_ERRORS
}

// retain_grad
static PyObject * THPVariable_retain_grad(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "retain_grad");
  }
  // aten::retain_grad(Tensor(a!) self) -> ()
  
  auto dispatch_retain_grad = [](const at::Tensor & self) -> void {
    pybind11::gil_scoped_release no_gil;
    self.retain_grad();
  };
  dispatch_retain_grad(self);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// roll
static PyObject * THPVariable_roll(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "roll(SymIntArrayRef[1] shifts, IntArrayRef[1] dims=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::roll(Tensor self, SymInt[1] shifts, int[1] dims=[]) -> Tensor
  
  auto dispatch_roll = [](const at::Tensor & self, c10::SymIntArrayRef shifts, at::IntArrayRef dims) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.roll_symint(shifts, dims);
  };
  return wrap(dispatch_roll(self, _r.symintlist(0), _r.intlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rot90
static PyObject * THPVariable_rot90(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "rot90(int64_t k=1, IntArrayRef dims={0,1})",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor
  
  auto dispatch_rot90 = [](const at::Tensor & self, int64_t k, at::IntArrayRef dims) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.rot90(k, dims);
  };
  return wrap(dispatch_rot90(self, _r.toInt64(0), _r.intlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// round
static PyObject * THPVariable_round(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "round()",
    "round(*, int64_t decimals)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::round(Tensor self) -> Tensor
      
      auto dispatch_round = [](const at::Tensor & self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.round();
      };
      return wrap(dispatch_round(self));
    }
    case 1: {
      // aten::round.decimals(Tensor self, *, int decimals) -> Tensor
      
      auto dispatch_round = [](const at::Tensor & self, int64_t decimals) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.round(decimals);
      };
      return wrap(dispatch_round(self, _r.toInt64(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// round_
static PyObject * THPVariable_round_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "round_()",
    "round_(*, int64_t decimals)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::round_(Tensor(a!) self) -> Tensor(a!)
      
      auto dispatch_round_ = [](const at::Tensor & self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.round_();
      };
      return wrap(dispatch_round_(self));
    }
    case 1: {
      // aten::round_.decimals(Tensor(a!) self, *, int decimals) -> Tensor(a!)
      
      auto dispatch_round_ = [](const at::Tensor & self, int64_t decimals) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.round_(decimals);
      };
      return wrap(dispatch_round_(self, _r.toInt64(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// row_indices
static PyObject * THPVariable_row_indices(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "row_indices");
  }
  // aten::row_indices(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_row_indices = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.row_indices();
  };
  return wrap(dispatch_row_indices(self));
  END_HANDLE_TH_ERRORS
}

// rsqrt
static PyObject * THPVariable_rsqrt(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "rsqrt");
  }
  // aten::rsqrt(Tensor self) -> Tensor
  
  auto dispatch_rsqrt = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.rsqrt();
  };
  return wrap(dispatch_rsqrt(self));
  END_HANDLE_TH_ERRORS
}

// rsqrt_
static PyObject * THPVariable_rsqrt_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "rsqrt_");
  }
  // aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_rsqrt_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.rsqrt_();
  };
  return wrap(dispatch_rsqrt_(self));
  END_HANDLE_TH_ERRORS
}

\
// scatter
static PyObject * THPVariable_scatter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "scatter(int64_t dim, Tensor index, Tensor src)",
    "scatter(int64_t dim, Tensor index, Tensor src, *, c10::string_view reduce)",
    "scatter(Dimname dim, Tensor index, Tensor src)",
    "scatter(int64_t dim, Tensor index, Scalar value)",
    "scatter(int64_t dim, Tensor index, Scalar value, *, c10::string_view reduce)",
    "scatter(Dimname dim, Tensor index, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
      
      auto dispatch_scatter = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter(dim, index, src);
      };
      return wrap(dispatch_scatter(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::scatter.reduce(Tensor self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor
      
      auto dispatch_scatter = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter(dim, index, src, reduce);
      };
      return wrap(dispatch_scatter(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2), _r.stringView(3)));
    }
    case 2: {
      // aten::scatter.dimname_src(Tensor self, Dimname dim, Tensor index, Tensor src) -> Tensor
      
      auto dispatch_scatter = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter(dim, index, src);
      };
      return wrap(dispatch_scatter(self, _r.dimname(0), _r.tensor(1), _r.tensor(2)));
    }
    case 3: {
      // aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
      
      auto dispatch_scatter = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter(dim, index, value);
      };
      return wrap(dispatch_scatter(self, _r.toInt64(0), _r.tensor(1), _r.scalar(2)));
    }
    case 4: {
      // aten::scatter.value_reduce(Tensor self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor
      
      auto dispatch_scatter = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter(dim, index, value, reduce);
      };
      return wrap(dispatch_scatter(self, _r.toInt64(0), _r.tensor(1), _r.scalar(2), _r.stringView(3)));
    }
    case 5: {
      // aten::scatter.dimname_value(Tensor self, Dimname dim, Tensor index, Scalar value) -> Tensor
      
      auto dispatch_scatter = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter(dim, index, value);
      };
      return wrap(dispatch_scatter(self, _r.dimname(0), _r.tensor(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// scatter_
static PyObject * THPVariable_scatter_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "scatter_(int64_t dim, Tensor index, Tensor src)",
    "scatter_(int64_t dim, Tensor index, Tensor src, *, c10::string_view reduce)",
    "scatter_(int64_t dim, Tensor index, Scalar value)",
    "scatter_(int64_t dim, Tensor index, Scalar value, *, c10::string_view reduce)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)
      
      auto dispatch_scatter_ = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter_(dim, index, src);
      };
      return wrap(dispatch_scatter_(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::scatter_.reduce(Tensor(a!) self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor(a!)
      
      auto dispatch_scatter_ = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter_(dim, index, src, reduce);
      };
      return wrap(dispatch_scatter_(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2), _r.stringView(3)));
    }
    case 2: {
      // aten::scatter_.value(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)
      
      auto dispatch_scatter_ = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter_(dim, index, value);
      };
      return wrap(dispatch_scatter_(self, _r.toInt64(0), _r.tensor(1), _r.scalar(2)));
    }
    case 3: {
      // aten::scatter_.value_reduce(Tensor(a!) self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor(a!)
      
      auto dispatch_scatter_ = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter_(dim, index, value, reduce);
      };
      return wrap(dispatch_scatter_(self, _r.toInt64(0), _r.tensor(1), _r.scalar(2), _r.stringView(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// scatter_add
static PyObject * THPVariable_scatter_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "scatter_add(int64_t dim, Tensor index, Tensor src)",
    "scatter_add(Dimname dim, Tensor index, Tensor src)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
      
      auto dispatch_scatter_add = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter_add(dim, index, src);
      };
      return wrap(dispatch_scatter_add(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::scatter_add.dimname(Tensor self, Dimname dim, Tensor index, Tensor src) -> Tensor
      
      auto dispatch_scatter_add = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter_add(dim, index, src);
      };
      return wrap(dispatch_scatter_add(self, _r.dimname(0), _r.tensor(1), _r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// scatter_add_
static PyObject * THPVariable_scatter_add_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "scatter_add_(int64_t dim, Tensor index, Tensor src)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)
  
  auto dispatch_scatter_add_ = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.scatter_add_(dim, index, src);
  };
  return wrap(dispatch_scatter_add_(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// scatter_reduce
static PyObject * THPVariable_scatter_reduce(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "scatter_reduce(int64_t dim, Tensor index, Tensor src, c10::string_view reduce, *, bool include_self=True)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor
  
  auto dispatch_scatter_reduce = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, bool include_self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.scatter_reduce(dim, index, src, reduce, include_self);
  };
  return wrap(dispatch_scatter_reduce(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2), _r.stringView(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// scatter_reduce_
static PyObject * THPVariable_scatter_reduce_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "scatter_reduce_(int64_t dim, Tensor index, Tensor src, c10::string_view reduce, *, bool include_self=True)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::scatter_reduce_.two(Tensor(a!) self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor(a!)
  
  auto dispatch_scatter_reduce_ = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, bool include_self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.scatter_reduce_(dim, index, src, reduce, include_self);
  };
  return wrap(dispatch_scatter_reduce_(self, _r.toInt64(0), _r.tensor(1), _r.tensor(2), _r.stringView(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// select
static PyObject * THPVariable_select(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "select(Dimname dim, int64_t index)",
    "select(int64_t dim, SymInt index)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::select.Dimname(Tensor(a) self, Dimname dim, int index) -> Tensor(a)
      
      auto dispatch_select = [](const at::Tensor & self, at::Dimname dim, int64_t index) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.select(dim, index);
      };
      return wrap(dispatch_select(self, _r.dimname(0), _r.toInt64(1)));
    }
    case 1: {
      // aten::select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a)
      
      auto dispatch_select = [](const at::Tensor & self, int64_t dim, c10::SymInt index) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.select_symint(dim, index);
      };
      return wrap(dispatch_select(self, _r.toInt64(0), _r.toSymInt(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// select_scatter
static PyObject * THPVariable_select_scatter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "select_scatter(Tensor src, int64_t dim, SymInt index)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::select_scatter(Tensor self, Tensor src, int dim, SymInt index) -> Tensor
  
  auto dispatch_select_scatter = [](const at::Tensor & self, const at::Tensor & src, int64_t dim, c10::SymInt index) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.select_scatter_symint(src, dim, index);
  };
  return wrap(dispatch_select_scatter(self, _r.tensor(0), _r.toInt64(1), _r.toSymInt(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sgn
static PyObject * THPVariable_sgn(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "sgn");
  }
  // aten::sgn(Tensor self) -> Tensor
  
  auto dispatch_sgn = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sgn();
  };
  return wrap(dispatch_sgn(self));
  END_HANDLE_TH_ERRORS
}

// sgn_
static PyObject * THPVariable_sgn_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "sgn_");
  }
  // aten::sgn_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_sgn_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sgn_();
  };
  return wrap(dispatch_sgn_(self));
  END_HANDLE_TH_ERRORS
}

// sigmoid
static PyObject * THPVariable_sigmoid(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "sigmoid");
  }
  // aten::sigmoid(Tensor self) -> Tensor
  
  auto dispatch_sigmoid = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sigmoid();
  };
  return wrap(dispatch_sigmoid(self));
  END_HANDLE_TH_ERRORS
}

// sigmoid_
static PyObject * THPVariable_sigmoid_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "sigmoid_");
  }
  // aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_sigmoid_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sigmoid_();
  };
  return wrap(dispatch_sigmoid_(self));
  END_HANDLE_TH_ERRORS
}

// sign
static PyObject * THPVariable_sign(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "sign");
  }
  // aten::sign(Tensor self) -> Tensor
  
  auto dispatch_sign = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sign();
  };
  return wrap(dispatch_sign(self));
  END_HANDLE_TH_ERRORS
}

// sign_
static PyObject * THPVariable_sign_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "sign_");
  }
  // aten::sign_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_sign_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sign_();
  };
  return wrap(dispatch_sign_(self));
  END_HANDLE_TH_ERRORS
}

// signbit
static PyObject * THPVariable_signbit(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "signbit");
  }
  // aten::signbit(Tensor self) -> Tensor
  
  auto dispatch_signbit = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.signbit();
  };
  return wrap(dispatch_signbit(self));
  END_HANDLE_TH_ERRORS
}

// sin
static PyObject * THPVariable_sin(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "sin");
  }
  // aten::sin(Tensor self) -> Tensor
  
  auto dispatch_sin = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sin();
  };
  return wrap(dispatch_sin(self));
  END_HANDLE_TH_ERRORS
}

// sin_
static PyObject * THPVariable_sin_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "sin_");
  }
  // aten::sin_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_sin_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sin_();
  };
  return wrap(dispatch_sin_(self));
  END_HANDLE_TH_ERRORS
}

// sinc
static PyObject * THPVariable_sinc(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "sinc");
  }
  // aten::sinc(Tensor self) -> Tensor
  
  auto dispatch_sinc = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sinc();
  };
  return wrap(dispatch_sinc(self));
  END_HANDLE_TH_ERRORS
}

// sinc_
static PyObject * THPVariable_sinc_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "sinc_");
  }
  // aten::sinc_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_sinc_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sinc_();
  };
  return wrap(dispatch_sinc_(self));
  END_HANDLE_TH_ERRORS
}

// sinh
static PyObject * THPVariable_sinh(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "sinh");
  }
  // aten::sinh(Tensor self) -> Tensor
  
  auto dispatch_sinh = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sinh();
  };
  return wrap(dispatch_sinh(self));
  END_HANDLE_TH_ERRORS
}

// sinh_
static PyObject * THPVariable_sinh_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "sinh_");
  }
  // aten::sinh_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_sinh_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sinh_();
  };
  return wrap(dispatch_sinh_(self));
  END_HANDLE_TH_ERRORS
}

// slice_inverse
static PyObject * THPVariable_slice_inverse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "slice_inverse(Tensor src, int64_t dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::slice_inverse(Tensor(a) self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)
  
  auto dispatch_slice_inverse = [](const at::Tensor & self, const at::Tensor & src, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.slice_inverse_symint(src, dim, start, end, step);
  };
  return wrap(dispatch_slice_inverse(self, _r.tensor(0), _r.toInt64(1), _r.toSymIntOptional(2), _r.toSymIntOptional(3), _r.toSymInt(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// slice_scatter
static PyObject * THPVariable_slice_scatter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "slice_scatter(Tensor src, int64_t dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor
  
  auto dispatch_slice_scatter = [](const at::Tensor & self, const at::Tensor & src, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.slice_scatter_symint(src, dim, start, end, step);
  };
  return wrap(dispatch_slice_scatter(self, _r.tensor(0), _r.toInt64(1), _r.toSymIntOptional(2), _r.toSymIntOptional(3), _r.toSymInt(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// slogdet
static PyObject * THPVariable_slogdet(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_slogdet_structseq();
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "slogdet");
  }
  // aten::slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)
  
  auto dispatch_slogdet = [](const at::Tensor & self) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.slogdet();
  };
  return wrap(NamedTuple, dispatch_slogdet(self));
  END_HANDLE_TH_ERRORS
}

// smm
static PyObject * THPVariable_smm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "smm(Tensor mat2)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::smm(Tensor self, Tensor mat2) -> Tensor
  
  auto dispatch_smm = [](const at::Tensor & self, const at::Tensor & mat2) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.smm(mat2);
  };
  return wrap(dispatch_smm(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// softmax
static PyObject * THPVariable_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "softmax(int64_t dim, ScalarType? dtype=None)",
    "softmax(Dimname dim, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_softmax = [](const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.softmax(dim, dtype);
      };
      return wrap(dispatch_softmax(self, _r.toInt64(0), _r.scalartypeOptional(1)));
    }
    case 1: {
      // aten::softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_softmax = [](const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.softmax(dim, dtype);
      };
      return wrap(dispatch_softmax(self, _r.dimname(0), _r.scalartypeOptional(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// sort
static PyObject * THPVariable_sort(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_sort_structseq();
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "sort(*, bool? stable, int64_t dim=-1, bool descending=False)",
    "sort(int64_t dim=-1, bool descending=False)",
    "sort(*, bool? stable, Dimname dim, bool descending=False)",
    "sort(Dimname dim, bool descending=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::sort.stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
      
      auto dispatch_sort = [](const at::Tensor & self, ::std::optional<bool> stable, int64_t dim, bool descending) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.sort(stable, dim, descending);
      };
      return wrap(NamedTuple, dispatch_sort(self, _r.toBoolOptional(0), _r.toInt64(1), _r.toBool(2)));
    }
    case 1: {
      // aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
      
      auto dispatch_sort = [](const at::Tensor & self, int64_t dim, bool descending) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.sort(dim, descending);
      };
      return wrap(NamedTuple, dispatch_sort(self, _r.toInt64(0), _r.toBool(1)));
    }
    case 2: {
      // aten::sort.dimname_stable(Tensor self, *, bool? stable, Dimname dim, bool descending=False) -> (Tensor values, Tensor indices)
      
      auto dispatch_sort = [](const at::Tensor & self, ::std::optional<bool> stable, at::Dimname dim, bool descending) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.sort(stable, dim, descending);
      };
      return wrap(NamedTuple, dispatch_sort(self, _r.toBoolOptional(0), _r.dimname(1), _r.toBool(2)));
    }
    case 3: {
      // aten::sort.dimname(Tensor self, Dimname dim, bool descending=False) -> (Tensor values, Tensor indices)
      
      auto dispatch_sort = [](const at::Tensor & self, at::Dimname dim, bool descending) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.sort(dim, descending);
      };
      return wrap(NamedTuple, dispatch_sort(self, _r.dimname(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sparse_dim
static PyObject * THPVariable_sparse_dim(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "sparse_dim");
  }
  // aten::sparse_dim(Tensor self) -> int
  
  auto dispatch_sparse_dim = [](const at::Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return self.sparse_dim();
  };
  return wrap(dispatch_sparse_dim(self));
  END_HANDLE_TH_ERRORS
}

// sparse_mask
static PyObject * THPVariable_sparse_mask(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "sparse_mask(Tensor mask)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::sparse_mask(Tensor self, Tensor mask) -> Tensor
  
  auto dispatch_sparse_mask = [](const at::Tensor & self, const at::Tensor & mask) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sparse_mask(mask);
  };
  return wrap(dispatch_sparse_mask(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sparse_resize_
static PyObject * THPVariable_sparse_resize_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "sparse_resize_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::sparse_resize_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)
  
  auto dispatch_sparse_resize_ = [](const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sparse_resize_(size, sparse_dim, dense_dim);
  };
  return wrap(dispatch_sparse_resize_(self, _r.intlist(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sparse_resize_and_clear_
static PyObject * THPVariable_sparse_resize_and_clear_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "sparse_resize_and_clear_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::sparse_resize_and_clear_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)
  
  auto dispatch_sparse_resize_and_clear_ = [](const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sparse_resize_and_clear_(size, sparse_dim, dense_dim);
  };
  return wrap(dispatch_sparse_resize_and_clear_(self, _r.intlist(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// split
static PyObject * THPVariable_split(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "split(SymInt split_size, int64_t dim=0)",
    "split(SymIntArrayRef split_size, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::split.Tensor(Tensor(a -> *) self, SymInt split_size, int dim=0) -> Tensor(a)[]
      
      auto dispatch_split = [](const at::Tensor & self, c10::SymInt split_size, int64_t dim) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.split_symint(split_size, dim);
      };
      return wrap(dispatch_split(self, _r.toSymInt(0), _r.toInt64(1)));
    }
    case 1: {
      // aten::split.sizes(Tensor(a -> *) self, SymInt[] split_size, int dim=0) -> Tensor(a)[]
      
      auto dispatch_split = [](const at::Tensor & self, c10::SymIntArrayRef split_size, int64_t dim) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.split_symint(split_size, dim);
      };
      return wrap(dispatch_split(self, _r.symintlist(0), _r.toInt64(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// split_with_sizes
static PyObject * THPVariable_split_with_sizes(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "split_with_sizes(SymIntArrayRef split_sizes, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]
  
  auto dispatch_split_with_sizes = [](const at::Tensor & self, c10::SymIntArrayRef split_sizes, int64_t dim) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.split_with_sizes_symint(split_sizes, dim);
  };
  return wrap(dispatch_split_with_sizes(self, _r.symintlist(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sqrt
static PyObject * THPVariable_sqrt(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "sqrt");
  }
  // aten::sqrt(Tensor self) -> Tensor
  
  auto dispatch_sqrt = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sqrt();
  };
  return wrap(dispatch_sqrt(self));
  END_HANDLE_TH_ERRORS
}

// sqrt_
static PyObject * THPVariable_sqrt_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "sqrt_");
  }
  // aten::sqrt_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_sqrt_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sqrt_();
  };
  return wrap(dispatch_sqrt_(self));
  END_HANDLE_TH_ERRORS
}

// square
static PyObject * THPVariable_square(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "square");
  }
  // aten::square(Tensor self) -> Tensor
  
  auto dispatch_square = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.square();
  };
  return wrap(dispatch_square(self));
  END_HANDLE_TH_ERRORS
}

// square_
static PyObject * THPVariable_square_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "square_");
  }
  // aten::square_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_square_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.square_();
  };
  return wrap(dispatch_square_(self));
  END_HANDLE_TH_ERRORS
}

\
// squeeze
static PyObject * THPVariable_squeeze(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "squeeze()",
    "squeeze(int64_t dim)",
    "squeeze(IntArrayRef dim)",
    "squeeze(Dimname dim)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::squeeze(Tensor(a) self) -> Tensor(a)
      
      auto dispatch_squeeze = [](const at::Tensor & self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze();
      };
      return wrap(dispatch_squeeze(self));
    }
    case 1: {
      // aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
      
      auto dispatch_squeeze = [](const at::Tensor & self, int64_t dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze(dim);
      };
      return wrap(dispatch_squeeze(self, _r.toInt64(0)));
    }
    case 2: {
      // aten::squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)
      
      auto dispatch_squeeze = [](const at::Tensor & self, at::IntArrayRef dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze(dim);
      };
      return wrap(dispatch_squeeze(self, _r.intlist(0)));
    }
    case 3: {
      // aten::squeeze.dimname(Tensor(a) self, Dimname dim) -> Tensor(a)
      
      auto dispatch_squeeze = [](const at::Tensor & self, at::Dimname dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze(dim);
      };
      return wrap(dispatch_squeeze(self, _r.dimname(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// squeeze_
static PyObject * THPVariable_squeeze_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "squeeze_()",
    "squeeze_(int64_t dim)",
    "squeeze_(IntArrayRef dim)",
    "squeeze_(Dimname dim)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::squeeze_(Tensor(a!) self) -> Tensor(a!)
      
      auto dispatch_squeeze_ = [](const at::Tensor & self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze_();
      };
      return wrap(dispatch_squeeze_(self));
    }
    case 1: {
      // aten::squeeze_.dim(Tensor(a!) self, int dim) -> Tensor(a!)
      
      auto dispatch_squeeze_ = [](const at::Tensor & self, int64_t dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze_(dim);
      };
      return wrap(dispatch_squeeze_(self, _r.toInt64(0)));
    }
    case 2: {
      // aten::squeeze_.dims(Tensor(a!) self, int[] dim) -> Tensor(a!)
      
      auto dispatch_squeeze_ = [](const at::Tensor & self, at::IntArrayRef dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze_(dim);
      };
      return wrap(dispatch_squeeze_(self, _r.intlist(0)));
    }
    case 3: {
      // aten::squeeze_.dimname(Tensor(a!) self, Dimname dim) -> Tensor(a!)
      
      auto dispatch_squeeze_ = [](const at::Tensor & self, at::Dimname dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze_(dim);
      };
      return wrap(dispatch_squeeze_(self, _r.dimname(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// sspaddmm
static PyObject * THPVariable_sspaddmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "sspaddmm(Scalar beta, Scalar alpha, Tensor mat1, Tensor mat2)|deprecated",
    "sspaddmm(Scalar beta, Tensor mat1, Tensor mat2)|deprecated",
    "sspaddmm(Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::sspaddmm(Scalar beta, Tensor self, Scalar alpha, Tensor mat1, Tensor mat2) -> Tensor
      
      auto dispatch_sspaddmm = [](const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & mat1, const at::Tensor & mat2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sspaddmm(mat1, mat2, beta, alpha);
      };
      return wrap(dispatch_sspaddmm(_r.scalar(0), self, _r.scalar(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // [deprecated] aten::sspaddmm(Scalar beta, Tensor self, Tensor mat1, Tensor mat2) -> Tensor
      
      auto dispatch_sspaddmm = [](const at::Scalar & beta, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sspaddmm(mat1, mat2, beta, 1);
      };
      return wrap(dispatch_sspaddmm(_r.scalar(0), self, _r.tensor(1), _r.tensor(2)));
    }
    case 2: {
      // aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
      
      auto dispatch_sspaddmm = [](const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sspaddmm(mat1, mat2, beta, alpha);
      };
      return wrap(dispatch_sspaddmm(self, _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// std
static PyObject * THPVariable_std(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "std(IntArrayRef[1]? dim, bool unbiased=True, bool keepdim=False)",
    "std(IntArrayRef[1]? dim=None, *, Scalar? correction=None, bool keepdim=False)",
    "std(bool unbiased=True)",
    "std(DimnameList[1] dim, bool unbiased=True, bool keepdim=False)",
    "std(DimnameList[1] dim, *, Scalar? correction=None, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::std.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor
      
      auto dispatch_std = [](const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.std(dim, unbiased, keepdim);
      };
      return wrap(dispatch_std(self, _r.intlistOptional(0), _r.toBool(1), _r.toBool(2)));
    }
    case 1: {
      // aten::std.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
      
      auto dispatch_std = [](const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.std(dim, correction, keepdim);
      };
      return wrap(dispatch_std(self, _r.intlistOptional(0), _r.scalarOptional(1), _r.toBool(2)));
    }
    case 2: {
      // aten::std(Tensor self, bool unbiased=True) -> Tensor
      
      auto dispatch_std = [](const at::Tensor & self, bool unbiased) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.std(unbiased);
      };
      return wrap(dispatch_std(self, _r.toBool(0)));
    }
    case 3: {
      // aten::std.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
      
      auto dispatch_std = [](const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.std(dim, unbiased, keepdim);
      };
      return wrap(dispatch_std(self, _r.dimnamelist(0), _r.toBool(1), _r.toBool(2)));
    }
    case 4: {
      // aten::std.correction_names(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False) -> Tensor
      
      auto dispatch_std = [](const at::Tensor & self, at::DimnameList dim, const ::std::optional<at::Scalar> & correction, bool keepdim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.std(dim, correction, keepdim);
      };
      return wrap(dispatch_std(self, _r.dimnamelist(0), _r.scalarOptional(1), _r.toBool(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// stft
static PyObject * THPVariable_stft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "stft(int64_t n_fft, int64_t? hop_length=None, int64_t? win_length=None, Tensor? window=None, bool center=True, c10::string_view pad_mode=\"reflect\", bool normalized=False, bool? onesided=None, bool? return_complex=None, bool? align_to_window=None)",
    "stft(int64_t n_fft, int64_t? hop_length=None, int64_t? win_length=None, Tensor? window=None, bool normalized=False, bool? onesided=None, bool? return_complex=None, bool? align_to_window=None)",
  }, /*traceable=*/true);

  ParsedArgs<10> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::stft.center(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool center=True, str pad_mode="reflect", bool normalized=False, bool? onesided=None, bool? return_complex=None, bool? align_to_window=None) -> Tensor
      
      auto dispatch_stft = [](const at::Tensor & self, int64_t n_fft, ::std::optional<int64_t> hop_length, ::std::optional<int64_t> win_length, const ::std::optional<at::Tensor> & window, bool center, c10::string_view pad_mode, bool normalized, ::std::optional<bool> onesided, ::std::optional<bool> return_complex, ::std::optional<bool> align_to_window) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.stft(n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided, return_complex, align_to_window);
      };
      return wrap(dispatch_stft(self, _r.toInt64(0), _r.toInt64Optional(1), _r.toInt64Optional(2), _r.optionalTensor(3), _r.toBool(4), _r.stringView(5), _r.toBool(6), _r.toBoolOptional(7), _r.toBoolOptional(8), _r.toBoolOptional(9)));
    }
    case 1: {
      // aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool? onesided=None, bool? return_complex=None, bool? align_to_window=None) -> Tensor
      
      auto dispatch_stft = [](const at::Tensor & self, int64_t n_fft, ::std::optional<int64_t> hop_length, ::std::optional<int64_t> win_length, const ::std::optional<at::Tensor> & window, bool normalized, ::std::optional<bool> onesided, ::std::optional<bool> return_complex, ::std::optional<bool> align_to_window) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.stft(n_fft, hop_length, win_length, window, normalized, onesided, return_complex, align_to_window);
      };
      return wrap(dispatch_stft(self, _r.toInt64(0), _r.toInt64Optional(1), _r.toInt64Optional(2), _r.optionalTensor(3), _r.toBool(4), _r.toBoolOptional(5), _r.toBoolOptional(6), _r.toBoolOptional(7)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// sub
static PyObject * THPVariable_sub(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "sub(Scalar alpha, Tensor other)|deprecated",
    "sub(Tensor other, *, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::sub(Tensor self, Scalar alpha, Tensor other) -> Tensor
      
      auto dispatch_sub = [](const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sub(other, alpha);
      };
      return wrap(dispatch_sub(self, _r.scalar(0), _r.tensor(1)));
    }
    case 1: {
      // aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
      
      auto dispatch_sub = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sub(other, alpha);
      };
      return wrap(dispatch_sub(self, _r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// sub_
static PyObject * THPVariable_sub_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "sub_(Scalar alpha, Tensor other)|deprecated",
    "sub_(Tensor other, *, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::sub_(Tensor(a!) self, Scalar alpha, Tensor other) -> Tensor(a!)
      
      auto dispatch_sub_ = [](const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sub_(other, alpha);
      };
      return wrap(dispatch_sub_(self, _r.scalar(0), _r.tensor(1)));
    }
    case 1: {
      // aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
      
      auto dispatch_sub_ = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sub_(other, alpha);
      };
      return wrap(dispatch_sub_(self, _r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// subtract
static PyObject * THPVariable_subtract(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "subtract(Tensor other, *, Scalar alpha=1)",
    "subtract(Scalar other, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::subtract.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
      
      auto dispatch_subtract = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.subtract(other, alpha);
      };
      return wrap(dispatch_subtract(self, _r.tensor(0), _r.scalar(1)));
    }
    case 1: {
      // aten::subtract.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
      
      auto dispatch_subtract = [](const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.subtract(other, alpha);
      };
      return wrap(dispatch_subtract(self, _r.scalar(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// subtract_
static PyObject * THPVariable_subtract_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "subtract_(Tensor other, *, Scalar alpha=1)",
    "subtract_(Scalar other, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::subtract_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
      
      auto dispatch_subtract_ = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.subtract_(other, alpha);
      };
      return wrap(dispatch_subtract_(self, _r.tensor(0), _r.scalar(1)));
    }
    case 1: {
      // aten::subtract_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
      
      auto dispatch_subtract_ = [](const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.subtract_(other, alpha);
      };
      return wrap(dispatch_subtract_(self, _r.scalar(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// sum
static PyObject * THPVariable_sum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "sum(*, ScalarType? dtype=None)",
    "sum(IntArrayRef[1]? dim, bool keepdim=False, *, ScalarType? dtype=None)",
    "sum(DimnameList[1] dim, bool keepdim=False, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_sum = [](const at::Tensor & self, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sum(dtype);
      };
      return wrap(dispatch_sum(self, _r.scalartypeOptional(0)));
    }
    case 1: {
      // aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_sum = [](const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sum(dim, keepdim, dtype);
      };
      return wrap(dispatch_sum(self, _r.intlistOptional(0), _r.toBool(1), _r.scalartypeOptional(2)));
    }
    case 2: {
      // aten::sum.dim_DimnameList(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_sum = [](const at::Tensor & self, at::DimnameList dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sum(dim, keepdim, dtype);
      };
      return wrap(dispatch_sum(self, _r.dimnamelist(0), _r.toBool(1), _r.scalartypeOptional(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sum_to_size
static PyObject * THPVariable_sum_to_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "sum_to_size(SymIntArrayRef size)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::sum_to_size(Tensor self, SymInt[] size) -> Tensor
  
  auto dispatch_sum_to_size = [](const at::Tensor & self, c10::SymIntArrayRef size) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sum_to_size_symint(size);
  };
  return wrap(dispatch_sum_to_size(self, _r.symintlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// svd
static PyObject * THPVariable_svd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_svd_structseq();
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "svd(bool some=True, bool compute_uv=True)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)
  
  auto dispatch_svd = [](const at::Tensor & self, bool some, bool compute_uv) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.svd(some, compute_uv);
  };
  return wrap(NamedTuple, dispatch_svd(self, _r.toBool(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// swapaxes
static PyObject * THPVariable_swapaxes(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "swapaxes(int64_t axis0, int64_t axis1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::swapaxes(Tensor(a) self, int axis0, int axis1) -> Tensor(a)
  
  auto dispatch_swapaxes = [](const at::Tensor & self, int64_t axis0, int64_t axis1) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.swapaxes(axis0, axis1);
  };
  return wrap(dispatch_swapaxes(self, _r.toInt64(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// swapaxes_
static PyObject * THPVariable_swapaxes_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "swapaxes_(int64_t axis0, int64_t axis1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::swapaxes_(Tensor(a!) self, int axis0, int axis1) -> Tensor(a!)
  
  auto dispatch_swapaxes_ = [](const at::Tensor & self, int64_t axis0, int64_t axis1) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.swapaxes_(axis0, axis1);
  };
  return wrap(dispatch_swapaxes_(self, _r.toInt64(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// swapdims
static PyObject * THPVariable_swapdims(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "swapdims(int64_t dim0, int64_t dim1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::swapdims(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
  
  auto dispatch_swapdims = [](const at::Tensor & self, int64_t dim0, int64_t dim1) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.swapdims(dim0, dim1);
  };
  return wrap(dispatch_swapdims(self, _r.toInt64(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// swapdims_
static PyObject * THPVariable_swapdims_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "swapdims_(int64_t dim0, int64_t dim1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::swapdims_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
  
  auto dispatch_swapdims_ = [](const at::Tensor & self, int64_t dim0, int64_t dim1) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.swapdims_(dim0, dim1);
  };
  return wrap(dispatch_swapdims_(self, _r.toInt64(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// t
static PyObject * THPVariable_t(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "t");
  }
  // aten::t(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_t = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.t();
  };
  return wrap(dispatch_t(self));
  END_HANDLE_TH_ERRORS
}

// t_
static PyObject * THPVariable_t_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "t_");
  }
  // aten::t_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_t_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.t_();
  };
  return wrap(dispatch_t_(self));
  END_HANDLE_TH_ERRORS
}

// take
static PyObject * THPVariable_take(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "take(Tensor index)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::take(Tensor self, Tensor index) -> Tensor
  
  auto dispatch_take = [](const at::Tensor & self, const at::Tensor & index) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.take(index);
  };
  return wrap(dispatch_take(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// take_along_dim
static PyObject * THPVariable_take_along_dim(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "take_along_dim(Tensor indices, int64_t? dim=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::take_along_dim(Tensor self, Tensor indices, int? dim=None) -> Tensor
  
  auto dispatch_take_along_dim = [](const at::Tensor & self, const at::Tensor & indices, ::std::optional<int64_t> dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.take_along_dim(indices, dim);
  };
  return wrap(dispatch_take_along_dim(self, _r.tensor(0), _r.toInt64Optional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tan
static PyObject * THPVariable_tan(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "tan");
  }
  // aten::tan(Tensor self) -> Tensor
  
  auto dispatch_tan = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.tan();
  };
  return wrap(dispatch_tan(self));
  END_HANDLE_TH_ERRORS
}

// tan_
static PyObject * THPVariable_tan_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "tan_");
  }
  // aten::tan_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_tan_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.tan_();
  };
  return wrap(dispatch_tan_(self));
  END_HANDLE_TH_ERRORS
}

// tanh
static PyObject * THPVariable_tanh(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "tanh");
  }
  // aten::tanh(Tensor self) -> Tensor
  
  auto dispatch_tanh = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.tanh();
  };
  return wrap(dispatch_tanh(self));
  END_HANDLE_TH_ERRORS
}

// tanh_
static PyObject * THPVariable_tanh_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "tanh_");
  }
  // aten::tanh_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_tanh_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.tanh_();
  };
  return wrap(dispatch_tanh_(self));
  END_HANDLE_TH_ERRORS
}

\
// tensor_split
static PyObject * THPVariable_tensor_split(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "tensor_split(SymIntArrayRef indices, int64_t dim=0)",
    "tensor_split(Tensor tensor_indices_or_sections, int64_t dim=0)",
    "tensor_split(SymInt sections, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::tensor_split.indices(Tensor(a -> *) self, SymInt[] indices, int dim=0) -> Tensor(a)[]
      
      auto dispatch_tensor_split = [](const at::Tensor & self, c10::SymIntArrayRef indices, int64_t dim) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.tensor_split_symint(indices, dim);
      };
      return wrap(dispatch_tensor_split(self, _r.symintlist(0), _r.toInt64(1)));
    }
    case 1: {
      // aten::tensor_split.tensor_indices_or_sections(Tensor(a -> *) self, Tensor tensor_indices_or_sections, int dim=0) -> Tensor(a)[]
      
      auto dispatch_tensor_split = [](const at::Tensor & self, const at::Tensor & tensor_indices_or_sections, int64_t dim) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.tensor_split(tensor_indices_or_sections, dim);
      };
      return wrap(dispatch_tensor_split(self, _r.tensor(0), _r.toInt64(1)));
    }
    case 2: {
      // aten::tensor_split.sections(Tensor(a -> *) self, SymInt sections, int dim=0) -> Tensor(a)[]
      
      auto dispatch_tensor_split = [](const at::Tensor & self, c10::SymInt sections, int64_t dim) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.tensor_split_symint(sections, dim);
      };
      return wrap(dispatch_tensor_split(self, _r.toSymInt(0), _r.toInt64(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tile
static PyObject * THPVariable_tile(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "tile(SymIntArrayRef dims)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::tile(Tensor self, SymInt[] dims) -> Tensor
  
  auto dispatch_tile = [](const at::Tensor & self, c10::SymIntArrayRef dims) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.tile_symint(dims);
  };
  return wrap(dispatch_tile(self, _r.symintlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// to_dense
static PyObject * THPVariable_to_dense(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "to_dense(ScalarType? dtype=None, *, bool? masked_grad=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::to_dense(Tensor self, ScalarType? dtype=None, *, bool? masked_grad=None) -> Tensor
  
  auto dispatch_to_dense = [](const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<bool> masked_grad) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.to_dense(dtype, masked_grad);
  };
  return wrap(dispatch_to_dense(self, _r.scalartypeOptional(0), _r.toBoolOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// to_mkldnn
static PyObject * THPVariable_to_mkldnn(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "to_mkldnn(ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::to_mkldnn(Tensor self, ScalarType? dtype=None) -> Tensor
  
  auto dispatch_to_mkldnn = [](const at::Tensor & self, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.to_mkldnn(dtype);
  };
  return wrap(dispatch_to_mkldnn(self, _r.scalartypeOptional(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// to_padded_tensor
static PyObject * THPVariable_to_padded_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "to_padded_tensor(double padding, SymIntArrayRef? output_size=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::to_padded_tensor(Tensor self, float padding, SymInt[]? output_size=None) -> Tensor
  
  auto dispatch_to_padded_tensor = [](const at::Tensor & self, double padding, at::OptionalSymIntArrayRef output_size) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.to_padded_tensor_symint(padding, output_size);
  };
  return wrap(dispatch_to_padded_tensor(self, _r.toDouble(0), _r.symintlistOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// to_sparse
static PyObject * THPVariable_to_sparse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "to_sparse(*, Layout? layout=None, IntArrayRef[2]? blocksize=None, int64_t? dense_dim=None)",
    "to_sparse(int64_t sparse_dim)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::to_sparse(Tensor self, *, Layout? layout=None, int[2]? blocksize=None, int? dense_dim=None) -> Tensor
      
      auto dispatch_to_sparse = [](const at::Tensor & self, ::std::optional<at::Layout> layout, at::OptionalIntArrayRef blocksize, ::std::optional<int64_t> dense_dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.to_sparse(layout, blocksize, dense_dim);
      };
      return wrap(dispatch_to_sparse(self, _r.layoutOptional(0), _r.intlistOptional(1), _r.toInt64Optional(2)));
    }
    case 1: {
      // aten::to_sparse.sparse_dim(Tensor self, int sparse_dim) -> Tensor
      
      auto dispatch_to_sparse = [](const at::Tensor & self, int64_t sparse_dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.to_sparse(sparse_dim);
      };
      return wrap(dispatch_to_sparse(self, _r.toInt64(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// to_sparse_bsc
static PyObject * THPVariable_to_sparse_bsc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "to_sparse_bsc(IntArrayRef[2] blocksize, int64_t? dense_dim=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::to_sparse_bsc(Tensor self, int[2] blocksize, int? dense_dim=None) -> Tensor
  
  auto dispatch_to_sparse_bsc = [](const at::Tensor & self, at::IntArrayRef blocksize, ::std::optional<int64_t> dense_dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.to_sparse_bsc(blocksize, dense_dim);
  };
  return wrap(dispatch_to_sparse_bsc(self, _r.intlist(0), _r.toInt64Optional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// to_sparse_bsr
static PyObject * THPVariable_to_sparse_bsr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "to_sparse_bsr(IntArrayRef[2] blocksize, int64_t? dense_dim=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::to_sparse_bsr(Tensor self, int[2] blocksize, int? dense_dim=None) -> Tensor
  
  auto dispatch_to_sparse_bsr = [](const at::Tensor & self, at::IntArrayRef blocksize, ::std::optional<int64_t> dense_dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.to_sparse_bsr(blocksize, dense_dim);
  };
  return wrap(dispatch_to_sparse_bsr(self, _r.intlist(0), _r.toInt64Optional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// to_sparse_csc
static PyObject * THPVariable_to_sparse_csc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "to_sparse_csc(int64_t? dense_dim=None)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::to_sparse_csc(Tensor self, int? dense_dim=None) -> Tensor
  
  auto dispatch_to_sparse_csc = [](const at::Tensor & self, ::std::optional<int64_t> dense_dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.to_sparse_csc(dense_dim);
  };
  return wrap(dispatch_to_sparse_csc(self, _r.toInt64Optional(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// to_sparse_csr
static PyObject * THPVariable_to_sparse_csr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "to_sparse_csr(int64_t? dense_dim=None)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::to_sparse_csr(Tensor self, int? dense_dim=None) -> Tensor
  
  auto dispatch_to_sparse_csr = [](const at::Tensor & self, ::std::optional<int64_t> dense_dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.to_sparse_csr(dense_dim);
  };
  return wrap(dispatch_to_sparse_csr(self, _r.toInt64Optional(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// topk
static PyObject * THPVariable_topk(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_topk_structseq();
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "topk(SymInt k, int64_t dim=-1, bool largest=True, bool sorted=True)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::topk(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
  
  auto dispatch_topk = [](const at::Tensor & self, c10::SymInt k, int64_t dim, bool largest, bool sorted) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.topk_symint(k, dim, largest, sorted);
  };
  return wrap(NamedTuple, dispatch_topk(self, _r.toSymInt(0), _r.toInt64(1), _r.toBool(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// trace
static PyObject * THPVariable_trace(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "trace");
  }
  // aten::trace(Tensor self) -> Tensor
  
  auto dispatch_trace = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.trace();
  };
  return wrap(dispatch_trace(self));
  END_HANDLE_TH_ERRORS
}

\
// transpose
static PyObject * THPVariable_transpose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "transpose(int64_t dim0, int64_t dim1)",
    "transpose(Dimname dim0, Dimname dim1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
      
      auto dispatch_transpose = [](const at::Tensor & self, int64_t dim0, int64_t dim1) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.transpose(dim0, dim1);
      };
      return wrap(dispatch_transpose(self, _r.toInt64(0), _r.toInt64(1)));
    }
    case 1: {
      // aten::transpose.Dimname(Tensor(a) self, Dimname dim0, Dimname dim1) -> Tensor(a)
      
      auto dispatch_transpose = [](const at::Tensor & self, at::Dimname dim0, at::Dimname dim1) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.transpose(dim0, dim1);
      };
      return wrap(dispatch_transpose(self, _r.dimname(0), _r.dimname(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// transpose_
static PyObject * THPVariable_transpose_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "transpose_(int64_t dim0, int64_t dim1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
  
  auto dispatch_transpose_ = [](const at::Tensor & self, int64_t dim0, int64_t dim1) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.transpose_(dim0, dim1);
  };
  return wrap(dispatch_transpose_(self, _r.toInt64(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// triangular_solve
static PyObject * THPVariable_triangular_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_triangular_solve_structseq();
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "triangular_solve(Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)
  
  auto dispatch_triangular_solve = [](const at::Tensor & self, const at::Tensor & A, bool upper, bool transpose, bool unitriangular) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.triangular_solve(A, upper, transpose, unitriangular);
  };
  return wrap(NamedTuple, dispatch_triangular_solve(self, _r.tensor(0), _r.toBool(1), _r.toBool(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tril
static PyObject * THPVariable_tril(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "tril(int64_t diagonal=0)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::tril(Tensor self, int diagonal=0) -> Tensor
  
  auto dispatch_tril = [](const at::Tensor & self, int64_t diagonal) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.tril(diagonal);
  };
  return wrap(dispatch_tril(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tril_
static PyObject * THPVariable_tril_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "tril_(int64_t diagonal=0)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)
  
  auto dispatch_tril_ = [](const at::Tensor & self, int64_t diagonal) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.tril_(diagonal);
  };
  return wrap(dispatch_tril_(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// triu
static PyObject * THPVariable_triu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "triu(int64_t diagonal=0)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::triu(Tensor self, int diagonal=0) -> Tensor
  
  auto dispatch_triu = [](const at::Tensor & self, int64_t diagonal) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.triu(diagonal);
  };
  return wrap(dispatch_triu(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// triu_
static PyObject * THPVariable_triu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "triu_(int64_t diagonal=0)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)
  
  auto dispatch_triu_ = [](const at::Tensor & self, int64_t diagonal) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.triu_(diagonal);
  };
  return wrap(dispatch_triu_(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// true_divide
static PyObject * THPVariable_true_divide(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "true_divide(Tensor other)",
    "true_divide(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::true_divide.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_true_divide = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.true_divide(other);
      };
      return wrap(dispatch_true_divide(self, _r.tensor(0)));
    }
    case 1: {
      // aten::true_divide.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_true_divide = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.true_divide(other);
      };
      return wrap(dispatch_true_divide(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// true_divide_
static PyObject * THPVariable_true_divide_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "true_divide_(Tensor other)",
    "true_divide_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::true_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_true_divide_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.true_divide_(other);
      };
      return wrap(dispatch_true_divide_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::true_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_true_divide_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.true_divide_(other);
      };
      return wrap(dispatch_true_divide_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// trunc
static PyObject * THPVariable_trunc(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "trunc");
  }
  // aten::trunc(Tensor self) -> Tensor
  
  auto dispatch_trunc = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.trunc();
  };
  return wrap(dispatch_trunc(self));
  END_HANDLE_TH_ERRORS
}

// trunc_
static PyObject * THPVariable_trunc_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "trunc_");
  }
  // aten::trunc_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_trunc_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.trunc_();
  };
  return wrap(dispatch_trunc_(self));
  END_HANDLE_TH_ERRORS
}

// type_as
static PyObject * THPVariable_type_as(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "type_as(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::type_as(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_type_as = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.type_as(other);
  };
  return wrap(dispatch_type_as(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// unbind
static PyObject * THPVariable_unbind(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "unbind(int64_t dim=0)",
    "unbind(Dimname dim)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::unbind.int(Tensor(a -> *) self, int dim=0) -> Tensor(a)[]
      
      auto dispatch_unbind = [](const at::Tensor & self, int64_t dim) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.unbind(dim);
      };
      return wrap(dispatch_unbind(self, _r.toInt64(0)));
    }
    case 1: {
      // aten::unbind.Dimname(Tensor(a -> *) self, Dimname dim) -> Tensor(a)[]
      
      auto dispatch_unbind = [](const at::Tensor & self, at::Dimname dim) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.unbind(dim);
      };
      return wrap(dispatch_unbind(self, _r.dimname(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// unflatten
static PyObject * THPVariable_unflatten(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "unflatten(Dimname dim, SymIntArrayRef sizes, DimnameList names)",
    "unflatten(int64_t dim, SymIntArrayRef sizes)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::unflatten.Dimname(Tensor(a) self, Dimname dim, SymInt[] sizes, Dimname[] names) -> Tensor(a)
      
      auto dispatch_unflatten = [](const at::Tensor & self, at::Dimname dim, c10::SymIntArrayRef sizes, at::DimnameList names) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.unflatten_symint(dim, sizes, names);
      };
      return wrap(dispatch_unflatten(self, _r.dimname(0), _r.symintlist(1), _r.dimnamelist(2)));
    }
    case 1: {
      // aten::unflatten.int(Tensor(a) self, int dim, SymInt[] sizes) -> Tensor(a)
      
      auto dispatch_unflatten = [](const at::Tensor & self, int64_t dim, c10::SymIntArrayRef sizes) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.unflatten_symint(dim, sizes);
      };
      return wrap(dispatch_unflatten(self, _r.toInt64(0), _r.symintlist(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unfold
static PyObject * THPVariable_unfold(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "unfold(int64_t dimension, int64_t size, int64_t step)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)
  
  auto dispatch_unfold = [](const at::Tensor & self, int64_t dimension, int64_t size, int64_t step) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.unfold(dimension, size, step);
  };
  return wrap(dispatch_unfold(self, _r.toInt64(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// uniform_
static PyObject * THPVariable_uniform_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "uniform_(double from=0, double to=1, *, Generator? generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)
  
  auto dispatch_uniform_ = [](const at::Tensor & self, double from, double to, ::std::optional<at::Generator> generator) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.uniform_(from, to, generator);
  };
  return wrap(dispatch_uniform_(self, _r.toDouble(0), _r.toDouble(1), _r.generator(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unsafe_chunk
static PyObject * THPVariable_unsafe_chunk(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "unsafe_chunk(int64_t chunks, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::unsafe_chunk(Tensor self, int chunks, int dim=0) -> Tensor[]
  
  auto dispatch_unsafe_chunk = [](const at::Tensor & self, int64_t chunks, int64_t dim) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.unsafe_chunk(chunks, dim);
  };
  return wrap(dispatch_unsafe_chunk(self, _r.toInt64(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unsafe_split
static PyObject * THPVariable_unsafe_split(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "unsafe_split(SymInt split_size, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::unsafe_split.Tensor(Tensor self, SymInt split_size, int dim=0) -> Tensor[]
  
  auto dispatch_unsafe_split = [](const at::Tensor & self, c10::SymInt split_size, int64_t dim) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.unsafe_split_symint(split_size, dim);
  };
  return wrap(dispatch_unsafe_split(self, _r.toSymInt(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unsafe_split_with_sizes
static PyObject * THPVariable_unsafe_split_with_sizes(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "unsafe_split_with_sizes(SymIntArrayRef split_sizes, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::unsafe_split_with_sizes(Tensor self, SymInt[] split_sizes, int dim=0) -> Tensor[]
  
  auto dispatch_unsafe_split_with_sizes = [](const at::Tensor & self, c10::SymIntArrayRef split_sizes, int64_t dim) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.unsafe_split_with_sizes_symint(split_sizes, dim);
  };
  return wrap(dispatch_unsafe_split_with_sizes(self, _r.symintlist(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unsqueeze
static PyObject * THPVariable_unsqueeze(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "unsqueeze(int64_t dim)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
  
  auto dispatch_unsqueeze = [](const at::Tensor & self, int64_t dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.unsqueeze(dim);
  };
  return wrap(dispatch_unsqueeze(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unsqueeze_
static PyObject * THPVariable_unsqueeze_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "unsqueeze_(int64_t dim)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)
  
  auto dispatch_unsqueeze_ = [](const at::Tensor & self, int64_t dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.unsqueeze_(dim);
  };
  return wrap(dispatch_unsqueeze_(self, _r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// values
static PyObject * THPVariable_values(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "values");
  }
  // aten::values(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_values = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.values();
  };
  return wrap(dispatch_values(self));
  END_HANDLE_TH_ERRORS
}

\
// var
static PyObject * THPVariable_var(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "var(IntArrayRef[1]? dim, bool unbiased=True, bool keepdim=False)",
    "var(IntArrayRef[1]? dim=None, *, Scalar? correction=None, bool keepdim=False)",
    "var(bool unbiased=True)",
    "var(DimnameList[1] dim, bool unbiased=True, bool keepdim=False)",
    "var(DimnameList[1] dim, *, Scalar? correction=None, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::var.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor
      
      auto dispatch_var = [](const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.var(dim, unbiased, keepdim);
      };
      return wrap(dispatch_var(self, _r.intlistOptional(0), _r.toBool(1), _r.toBool(2)));
    }
    case 1: {
      // aten::var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
      
      auto dispatch_var = [](const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.var(dim, correction, keepdim);
      };
      return wrap(dispatch_var(self, _r.intlistOptional(0), _r.scalarOptional(1), _r.toBool(2)));
    }
    case 2: {
      // aten::var(Tensor self, bool unbiased=True) -> Tensor
      
      auto dispatch_var = [](const at::Tensor & self, bool unbiased) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.var(unbiased);
      };
      return wrap(dispatch_var(self, _r.toBool(0)));
    }
    case 3: {
      // aten::var.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
      
      auto dispatch_var = [](const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.var(dim, unbiased, keepdim);
      };
      return wrap(dispatch_var(self, _r.dimnamelist(0), _r.toBool(1), _r.toBool(2)));
    }
    case 4: {
      // aten::var.correction_names(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False) -> Tensor
      
      auto dispatch_var = [](const at::Tensor & self, at::DimnameList dim, const ::std::optional<at::Scalar> & correction, bool keepdim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.var(dim, correction, keepdim);
      };
      return wrap(dispatch_var(self, _r.dimnamelist(0), _r.scalarOptional(1), _r.toBool(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// vdot
static PyObject * THPVariable_vdot(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "vdot(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::vdot(Tensor self, Tensor other) -> Tensor
  
  auto dispatch_vdot = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.vdot(other);
  };
  return wrap(dispatch_vdot(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// view
static PyObject * THPVariable_view(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "view(ScalarType dtype)",
    "view(SymIntArrayRef size)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::view.dtype(Tensor(a) self, ScalarType dtype) -> Tensor(a)
      
      auto dispatch_view = [](const at::Tensor & self, at::ScalarType dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.view(dtype);
      };
      return wrap(dispatch_view(self, _r.scalartype(0)));
    }
    case 1: {
      // aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)
      
      auto dispatch_view = [](const at::Tensor & self, c10::SymIntArrayRef size) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.view_symint(size);
      };
      return wrap(dispatch_view(self, _r.symintlist(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// view_as
static PyObject * THPVariable_view_as(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "view_as(Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  // aten::view_as(Tensor(a) self, Tensor other) -> Tensor(a)
  
  auto dispatch_view_as = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.view_as(other);
  };
  return wrap(dispatch_view_as(self, _r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// vsplit
static PyObject * THPVariable_vsplit(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "vsplit(int64_t sections)",
    "vsplit(IntArrayRef indices)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::vsplit.int(Tensor(a -> *) self, int sections) -> Tensor(a)[]
      
      auto dispatch_vsplit = [](const at::Tensor & self, int64_t sections) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.vsplit(sections);
      };
      return wrap(dispatch_vsplit(self, _r.toInt64(0)));
    }
    case 1: {
      // aten::vsplit.array(Tensor(a -> *) self, int[] indices) -> Tensor(a)[]
      
      auto dispatch_vsplit = [](const at::Tensor & self, at::IntArrayRef indices) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.vsplit(indices);
      };
      return wrap(dispatch_vsplit(self, _r.intlist(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// where
static PyObject * THPVariable_where(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "where(Tensor condition, Tensor other)",
    "where(Tensor condition, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
      
      auto dispatch_where = [](const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.where(condition, other);
      };
      return wrap(dispatch_where(_r.tensor(0), self, _r.tensor(1)));
    }
    case 1: {
      // aten::where.ScalarOther(Tensor condition, Tensor self, Scalar other) -> Tensor
      
      auto dispatch_where = [](const at::Tensor & condition, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.where(condition, other);
      };
      return wrap(dispatch_where(_r.tensor(0), self, _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// xlogy
static PyObject * THPVariable_xlogy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "xlogy(Tensor other)",
    "xlogy(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::xlogy.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch_xlogy = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.xlogy(other);
      };
      return wrap(dispatch_xlogy(self, _r.tensor(0)));
    }
    case 1: {
      // aten::xlogy.Scalar_Other(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_xlogy = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.xlogy(other);
      };
      return wrap(dispatch_xlogy(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// xlogy_
static PyObject * THPVariable_xlogy_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "xlogy_(Tensor other)",
    "xlogy_(Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // aten::xlogy_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_xlogy_ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.xlogy_(other);
      };
      return wrap(dispatch_xlogy_(self, _r.tensor(0)));
    }
    case 1: {
      // aten::xlogy_.Scalar_Other(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_xlogy_ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.xlogy_(other);
      };
      return wrap(dispatch_xlogy_(self, _r.scalar(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// zero_
static PyObject * THPVariable_zero_(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  if(check_has_torch_function(self_)) {
    return handle_torch_function(self_, "zero_");
  }
  // aten::zero_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_zero_ = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.zero_();
  };
  return wrap(dispatch_zero_(self));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_bool_scalar(PyObject* self, PyObject* args) {
  if (check_has_torch_function(self)) {
    HANDLE_TH_ERRORS
    return handle_torch_function(self, "__bool__", args);
    END_HANDLE_TH_ERRORS
  }
  jit::tracer::warn("Converting a tensor to a Python boolean", jit::tracer::WARN_PYTHON_DATAFLOW);
  return THPVariable_is_nonzero(self, args);
}

static PyObject * THPVariable___eq__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
#ifdef USE_NUMPY
  if (torch::utils::is_numpy_available()) {
    static PythonArgParser parser({
      "__eq__(PyObject* other)",
    }, /*traceable=*/true);

    ParsedArgs<1> parsed_args;
    auto _r = parser.parse(self_, args, kwargs, parsed_args);
    if(_r.has_torch_function()) {
      return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
    }
    switch (_r.idx) {
      case 0: {
        auto other = _r.pyobject(0);
        if (PyArray_Check(other)) {
          auto other_tensor = torch::utils::tensor_from_numpy(other);
          auto dispatch_eq = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
            pybind11::gil_scoped_release no_gil;
            return self.eq(other);
          };
          const Tensor& self = THPVariable_Unpack(self_);
          return wrap(dispatch_eq(self, other_tensor));
        }
      }
    }
  }
#endif
  return THPVariable_eq(self_, args, kwargs);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// Wrapper converts a raised TypeError into returning NotImplemented
// Used to implement binary arithmetic operators
template <PyObject* (*Func)(PyObject*, PyObject*, PyObject*)>
static PyObject * TypeError_to_NotImplemented_(PyObject* self, PyObject* args, PyObject* kwargs) {

  PyObject* ret = Func(self, args, kwargs);
  if (!ret && PyErr_ExceptionMatches(PyExc_TypeError)) {
    PyErr_Clear();
    Py_INCREF(Py_NotImplemented);
    ret = Py_NotImplemented;
  }
  return ret;
}

// set_ has to be defined in the template because the c10::Storage object
// does not have a type, and we need to make sure the Python storage object's
// type matches the tensor's type
static PyObject* THPVariable_set_(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser(
      {
          "set_()",
          "set_(Storage source)",
          "set_(Storage source, SymInt storage_offset, SymIntArrayRef size, SymIntArrayRef stride=None)",
          "set_(Tensor source)",
          "set_(Tensor source, SymInt storage_offset, SymIntArrayRef size, SymIntArrayRef stride=None)",
      },
      /*traceable=*/false);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  switch (_r.idx) {
    case 0: {
      // aten::set_(Tensor(a!) self) -> Tensor(a!)
      auto dispatch_set_ = [](const Tensor& self) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.set_();
      };
      return wrap(dispatch_set_(self));
    }
    case 1: {
      // aten::set_.source_Storage(Tensor(a!) self, Storage source) ->
      // Tensor(a!)
      at::ScalarType storage_scalar_type{};
      bool is_typed_storage = true;
      at::Storage storage = _r.storage(0, storage_scalar_type, is_typed_storage);
      TORCH_CHECK(storage_scalar_type == self.dtype() || !is_typed_storage,
        "Expected a Storage of type ", self.dtype(),
        " or an UntypedStorage, but got type ", storage_scalar_type,
        " for argument 1 'storage'");
      auto dispatch_set_ = [](const Tensor& self, Storage source) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.set_(std::move(source));
      };
      return wrap(dispatch_set_(self, storage));
    }
    case 2: {
      // aten::set_.source_Storage_storage_offset(Tensor(a!) self, Storage
      // source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)
      at::ScalarType storage_scalar_type{};
      bool is_typed_storage = true;
      at::Storage storage = _r.storage(0, storage_scalar_type, is_typed_storage);
      TORCH_CHECK(storage_scalar_type == self.dtype() || !is_typed_storage,
        "Expected a Storage of type ", self.dtype(),
        " or an UntypedStorage, but got type ", storage_scalar_type,
        " for argument 1 'storage'");
      auto dispatch_set_ = [](const Tensor& self,
                              Storage source,
                              c10::SymInt storage_offset,
                              c10::SymIntArrayRef size,
                              c10::SymIntArrayRef stride) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.set__symint(std::move(source), std::move(storage_offset), size, stride);
      };
      return wrap(dispatch_set_(
          self, storage, _r.toSymInt(1), _r.symintlist(2), _r.symintlist(3)));
    }
    case 3: {
      // aten::set_.source_Tensor(Tensor(a!) self, Tensor source) -> Tensor(a!)
      auto dispatch_set_ = [](const Tensor& self, const Tensor& source) -> Tensor {
        TORCH_CHECK(source.dtype() == self.dtype(), "Could not set tensor of type ", source.dtype(), " to a tensor of type ", self.dtype());
        pybind11::gil_scoped_release no_gil;
        return self.set_(source);
      };
      return wrap(dispatch_set_(self, _r.tensor(0)));
    }
    case 4: {
      // aten::set_.source_Tensor_storage_offset(Tensor(a!) self, Tensor
      // source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)
      at::Tensor storage = _r.tensor(0);
      auto dispatch_set_ = [](const Tensor& self,
                              const Tensor& source,
                              c10::SymInt storage_offset,
                              c10::SymIntArrayRef size,
                              c10::SymIntArrayRef stride) -> Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.set__symint(source, std::move(storage_offset), size, stride);
      };
      return wrap(dispatch_set_(
          self, storage, _r.toSymInt(1), _r.symintlist(2), _r.symintlist(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// XXX: ops that are bound here are not exposed to the C++ api nor the JIT.
// Any new ops added here should be accompanied with a comment why they are not
// being registered through native_functions.yaml, and be tagged cpp / JIT
PyMethodDef variable_methods[] = {
  // These magic methods are all implemented on python object to wrap NotImplementedError
  {"__add__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_add>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__radd__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_add>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__iadd__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_add_>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__rmul__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_mul>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__mul__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_mul>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__imul__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_mul_>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__sub__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_sub>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__isub__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_sub_>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__div__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_div>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__truediv__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_div>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__floordiv__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_floor_divide>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__idiv__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_div_>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__ifloordiv__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_floor_divide_>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__mod__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_remainder>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__imod__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_remainder_>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__eq__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable___eq__>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__ne__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_ne>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__lt__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_lt>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__le__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_le>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__gt__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_gt>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__ge__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_ge>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__rand__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_bitwise_and>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__ror__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_bitwise_or>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__rxor__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_bitwise_xor>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__bool__", THPVariable_bool_scalar, METH_NOARGS, nullptr},
  {"__float__", THPVariable_float_scalar, METH_NOARGS, nullptr},
  {"__complex__", THPVariable_complex_scalar, METH_NOARGS, nullptr},
  {"__int__", THPVariable_integral_scalar, METH_NOARGS, nullptr},
  {"__long__", THPVariable_integral_scalar, METH_NOARGS, nullptr},
  {"__index__", THPVariable_index_scalar, METH_NOARGS, nullptr},
  {"__nonzero__", THPVariable_bool_scalar, METH_NOARGS, nullptr},
  {"__invert__", THPVariable_invert, METH_NOARGS, nullptr},
  {"__matmul__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_matmul>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_is_view", THPVariable__is_view, METH_NOARGS, nullptr},
  {"apply_", THPVariable_apply_, METH_O, nullptr},
  {"bfloat16", castPyCFunctionWithKeywords(THPVariable_bfloat16), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"byte", castPyCFunctionWithKeywords(THPVariable_byte), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"char", castPyCFunctionWithKeywords(THPVariable_char), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"contiguous", castPyCFunctionWithKeywords(THPVariable_contiguous), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"copy_", castPyCFunctionWithKeywords(THPVariable_copy_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"cpu", castPyCFunctionWithKeywords(THPVariable_cpu), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"cuda", castPyCFunctionWithKeywords(THPVariable_cuda), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"mtia", castPyCFunctionWithKeywords(THPVariable_mtia), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"xpu", castPyCFunctionWithKeywords(THPVariable_xpu), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"ipu", castPyCFunctionWithKeywords(THPVariable_ipu), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"data_ptr", THPVariable_data_ptr, METH_NOARGS, nullptr},
  {"dim", THPVariable_dim, METH_NOARGS, nullptr},
  {"has_names", THPVariable_has_names, METH_NOARGS, nullptr},
  {"double", castPyCFunctionWithKeywords(THPVariable_double), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"cdouble", castPyCFunctionWithKeywords(THPVariable_cdouble), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"element_size", THPVariable_element_size, METH_NOARGS, nullptr},
  {"float", castPyCFunctionWithKeywords(THPVariable_float), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"cfloat", castPyCFunctionWithKeywords(THPVariable_cfloat), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"get_device", THPVariable_get_device, METH_NOARGS, nullptr},
  {"bool", castPyCFunctionWithKeywords(THPVariable_bool), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"half", castPyCFunctionWithKeywords(THPVariable_half), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"int", castPyCFunctionWithKeywords(THPVariable_int), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"is_contiguous", castPyCFunctionWithKeywords(THPVariable_is_contiguous), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"item", THPVariable_item, METH_NOARGS, nullptr},
  {"long", castPyCFunctionWithKeywords(THPVariable_long), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"map_", castPyCFunctionWithKeywords(THPVariable_map_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"map2_", castPyCFunctionWithKeywords(THPVariable_map2_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"ndimension", THPVariable_dim, METH_NOARGS, nullptr},
  {"nelement", THPVariable_numel, METH_NOARGS, nullptr},
  {"new", castPyCFunctionWithKeywords(THPVariable_new), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"new_tensor", castPyCFunctionWithKeywords(THPVariable_new_tensor), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"nonzero", castPyCFunctionWithKeywords(THPVariable_nonzero), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"numel", THPVariable_numel, METH_NOARGS, nullptr},
  {"numpy", castPyCFunctionWithKeywords(THPVariable_numpy), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"requires_grad_", castPyCFunctionWithKeywords(THPVariable_requires_grad_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"set_", castPyCFunctionWithKeywords(THPVariable_set_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"short", castPyCFunctionWithKeywords(THPVariable_short), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"size", castPyCFunctionWithKeywords(THPVariable_size), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"untyped_storage", THPVariable_storage, METH_NOARGS, nullptr},
  {"storage_offset", THPVariable_storage_offset, METH_NOARGS, nullptr},
  {"stride", castPyCFunctionWithKeywords(THPVariable_stride), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"to", castPyCFunctionWithKeywords(THPVariable_to), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"tolist", THPVariable_tolist, METH_NOARGS, nullptr},
  {"type", castPyCFunctionWithKeywords(THPVariable_type), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__and__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable___and__>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__iand__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable___iand__>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__ilshift__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable___ilshift__>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__ior__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable___ior__>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__irshift__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable___irshift__>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__ixor__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable___ixor__>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__lshift__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable___lshift__>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__or__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable___or__>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__rshift__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable___rshift__>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"__xor__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable___xor__>), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_addmm_activation", castPyCFunctionWithKeywords(THPVariable__addmm_activation), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_autocast_to_full_precision", castPyCFunctionWithKeywords(THPVariable__autocast_to_full_precision), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_autocast_to_reduced_precision", castPyCFunctionWithKeywords(THPVariable__autocast_to_reduced_precision), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_coalesced_", castPyCFunctionWithKeywords(THPVariable__coalesced_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_conj", THPVariable__conj, METH_NOARGS, nullptr},
  {"_conj_physical", THPVariable__conj_physical, METH_NOARGS, nullptr},
  {"_dimI", THPVariable__dimI, METH_NOARGS, nullptr},
  {"_dimV", THPVariable__dimV, METH_NOARGS, nullptr},
  {"_indices", THPVariable__indices, METH_NOARGS, nullptr},
  {"_is_all_true", THPVariable__is_all_true, METH_NOARGS, nullptr},
  {"_is_any_true", THPVariable__is_any_true, METH_NOARGS, nullptr},
  {"_is_zerotensor", THPVariable__is_zerotensor, METH_NOARGS, nullptr},
  {"_lazy_clone", THPVariable__lazy_clone, METH_NOARGS, nullptr},
  {"_neg_view", THPVariable__neg_view, METH_NOARGS, nullptr},
  {"_nested_tensor_size", THPVariable__nested_tensor_size, METH_NOARGS, nullptr},
  {"_nested_tensor_storage_offsets", THPVariable__nested_tensor_storage_offsets, METH_NOARGS, nullptr},
  {"_nested_tensor_strides", THPVariable__nested_tensor_strides, METH_NOARGS, nullptr},
  {"_nnz", THPVariable__nnz, METH_NOARGS, nullptr},
  {"_sparse_mask_projection", castPyCFunctionWithKeywords(THPVariable__sparse_mask_projection), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_to_dense", castPyCFunctionWithKeywords(THPVariable__to_dense), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_to_sparse", castPyCFunctionWithKeywords(THPVariable__to_sparse), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_to_sparse_bsc", castPyCFunctionWithKeywords(THPVariable__to_sparse_bsc), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_to_sparse_bsr", castPyCFunctionWithKeywords(THPVariable__to_sparse_bsr), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_to_sparse_csc", castPyCFunctionWithKeywords(THPVariable__to_sparse_csc), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_to_sparse_csr", castPyCFunctionWithKeywords(THPVariable__to_sparse_csr), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_values", THPVariable__values, METH_NOARGS, nullptr},
  {"abs", THPVariable_abs, METH_NOARGS, nullptr},
  {"abs_", THPVariable_abs_, METH_NOARGS, nullptr},
  {"absolute", THPVariable_absolute, METH_NOARGS, nullptr},
  {"absolute_", THPVariable_absolute_, METH_NOARGS, nullptr},
  {"acos", THPVariable_acos, METH_NOARGS, nullptr},
  {"acos_", THPVariable_acos_, METH_NOARGS, nullptr},
  {"acosh", THPVariable_acosh, METH_NOARGS, nullptr},
  {"acosh_", THPVariable_acosh_, METH_NOARGS, nullptr},
  {"add", castPyCFunctionWithKeywords(THPVariable_add), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"add_", castPyCFunctionWithKeywords(THPVariable_add_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"addbmm", castPyCFunctionWithKeywords(THPVariable_addbmm), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"addbmm_", castPyCFunctionWithKeywords(THPVariable_addbmm_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"addcdiv", castPyCFunctionWithKeywords(THPVariable_addcdiv), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"addcdiv_", castPyCFunctionWithKeywords(THPVariable_addcdiv_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"addcmul", castPyCFunctionWithKeywords(THPVariable_addcmul), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"addcmul_", castPyCFunctionWithKeywords(THPVariable_addcmul_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"addmm", castPyCFunctionWithKeywords(THPVariable_addmm), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"addmm_", castPyCFunctionWithKeywords(THPVariable_addmm_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"addmv", castPyCFunctionWithKeywords(THPVariable_addmv), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"addmv_", castPyCFunctionWithKeywords(THPVariable_addmv_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"addr", castPyCFunctionWithKeywords(THPVariable_addr), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"addr_", castPyCFunctionWithKeywords(THPVariable_addr_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"adjoint", THPVariable_adjoint, METH_NOARGS, nullptr},
  {"align_as", castPyCFunctionWithKeywords(THPVariable_align_as), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"align_to", castPyCFunctionWithKeywords(THPVariable_align_to), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"all", castPyCFunctionWithKeywords(THPVariable_all), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"allclose", castPyCFunctionWithKeywords(THPVariable_allclose), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"amax", castPyCFunctionWithKeywords(THPVariable_amax), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"amin", castPyCFunctionWithKeywords(THPVariable_amin), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"aminmax", castPyCFunctionWithKeywords(THPVariable_aminmax), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"angle", THPVariable_angle, METH_NOARGS, nullptr},
  {"any", castPyCFunctionWithKeywords(THPVariable_any), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"arccos", THPVariable_arccos, METH_NOARGS, nullptr},
  {"arccos_", THPVariable_arccos_, METH_NOARGS, nullptr},
  {"arccosh", THPVariable_arccosh, METH_NOARGS, nullptr},
  {"arccosh_", THPVariable_arccosh_, METH_NOARGS, nullptr},
  {"arcsin", THPVariable_arcsin, METH_NOARGS, nullptr},
  {"arcsin_", THPVariable_arcsin_, METH_NOARGS, nullptr},
  {"arcsinh", THPVariable_arcsinh, METH_NOARGS, nullptr},
  {"arcsinh_", THPVariable_arcsinh_, METH_NOARGS, nullptr},
  {"arctan", THPVariable_arctan, METH_NOARGS, nullptr},
  {"arctan2", castPyCFunctionWithKeywords(THPVariable_arctan2), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"arctan2_", castPyCFunctionWithKeywords(THPVariable_arctan2_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"arctan_", THPVariable_arctan_, METH_NOARGS, nullptr},
  {"arctanh", THPVariable_arctanh, METH_NOARGS, nullptr},
  {"arctanh_", THPVariable_arctanh_, METH_NOARGS, nullptr},
  {"argmax", castPyCFunctionWithKeywords(THPVariable_argmax), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"argmin", castPyCFunctionWithKeywords(THPVariable_argmin), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"argsort", castPyCFunctionWithKeywords(THPVariable_argsort), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"argwhere", THPVariable_argwhere, METH_NOARGS, nullptr},
  {"as_strided", castPyCFunctionWithKeywords(THPVariable_as_strided), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"as_strided_", castPyCFunctionWithKeywords(THPVariable_as_strided_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"as_strided_scatter", castPyCFunctionWithKeywords(THPVariable_as_strided_scatter), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"asin", THPVariable_asin, METH_NOARGS, nullptr},
  {"asin_", THPVariable_asin_, METH_NOARGS, nullptr},
  {"asinh", THPVariable_asinh, METH_NOARGS, nullptr},
  {"asinh_", THPVariable_asinh_, METH_NOARGS, nullptr},
  {"atan", THPVariable_atan, METH_NOARGS, nullptr},
  {"atan2", castPyCFunctionWithKeywords(THPVariable_atan2), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"atan2_", castPyCFunctionWithKeywords(THPVariable_atan2_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"atan_", THPVariable_atan_, METH_NOARGS, nullptr},
  {"atanh", THPVariable_atanh, METH_NOARGS, nullptr},
  {"atanh_", THPVariable_atanh_, METH_NOARGS, nullptr},
  {"baddbmm", castPyCFunctionWithKeywords(THPVariable_baddbmm), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"baddbmm_", castPyCFunctionWithKeywords(THPVariable_baddbmm_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"bernoulli", castPyCFunctionWithKeywords(THPVariable_bernoulli), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"bernoulli_", castPyCFunctionWithKeywords(THPVariable_bernoulli_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"bincount", castPyCFunctionWithKeywords(THPVariable_bincount), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"bitwise_and", castPyCFunctionWithKeywords(THPVariable_bitwise_and), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"bitwise_and_", castPyCFunctionWithKeywords(THPVariable_bitwise_and_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"bitwise_left_shift", castPyCFunctionWithKeywords(THPVariable_bitwise_left_shift), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"bitwise_left_shift_", castPyCFunctionWithKeywords(THPVariable_bitwise_left_shift_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"bitwise_not", THPVariable_bitwise_not, METH_NOARGS, nullptr},
  {"bitwise_not_", THPVariable_bitwise_not_, METH_NOARGS, nullptr},
  {"bitwise_or", castPyCFunctionWithKeywords(THPVariable_bitwise_or), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"bitwise_or_", castPyCFunctionWithKeywords(THPVariable_bitwise_or_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"bitwise_right_shift", castPyCFunctionWithKeywords(THPVariable_bitwise_right_shift), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"bitwise_right_shift_", castPyCFunctionWithKeywords(THPVariable_bitwise_right_shift_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"bitwise_xor", castPyCFunctionWithKeywords(THPVariable_bitwise_xor), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"bitwise_xor_", castPyCFunctionWithKeywords(THPVariable_bitwise_xor_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"bmm", castPyCFunctionWithKeywords(THPVariable_bmm), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"broadcast_to", castPyCFunctionWithKeywords(THPVariable_broadcast_to), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"cauchy_", castPyCFunctionWithKeywords(THPVariable_cauchy_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"ccol_indices", THPVariable_ccol_indices, METH_NOARGS, nullptr},
  {"ceil", THPVariable_ceil, METH_NOARGS, nullptr},
  {"ceil_", THPVariable_ceil_, METH_NOARGS, nullptr},
  {"chalf", castPyCFunctionWithKeywords(THPVariable_chalf), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"cholesky", castPyCFunctionWithKeywords(THPVariable_cholesky), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"cholesky_inverse", castPyCFunctionWithKeywords(THPVariable_cholesky_inverse), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"cholesky_solve", castPyCFunctionWithKeywords(THPVariable_cholesky_solve), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"chunk", castPyCFunctionWithKeywords(THPVariable_chunk), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"clamp", castPyCFunctionWithKeywords(THPVariable_clamp), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"clamp_", castPyCFunctionWithKeywords(THPVariable_clamp_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"clamp_max", castPyCFunctionWithKeywords(THPVariable_clamp_max), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"clamp_max_", castPyCFunctionWithKeywords(THPVariable_clamp_max_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"clamp_min", castPyCFunctionWithKeywords(THPVariable_clamp_min), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"clamp_min_", castPyCFunctionWithKeywords(THPVariable_clamp_min_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"clip", castPyCFunctionWithKeywords(THPVariable_clip), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"clip_", castPyCFunctionWithKeywords(THPVariable_clip_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"clone", castPyCFunctionWithKeywords(THPVariable_clone), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"coalesce", THPVariable_coalesce, METH_NOARGS, nullptr},
  {"col_indices", THPVariable_col_indices, METH_NOARGS, nullptr},
  {"conj", THPVariable_conj, METH_NOARGS, nullptr},
  {"conj_physical", THPVariable_conj_physical, METH_NOARGS, nullptr},
  {"conj_physical_", THPVariable_conj_physical_, METH_NOARGS, nullptr},
  {"copysign", castPyCFunctionWithKeywords(THPVariable_copysign), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"copysign_", castPyCFunctionWithKeywords(THPVariable_copysign_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"corrcoef", THPVariable_corrcoef, METH_NOARGS, nullptr},
  {"cos", THPVariable_cos, METH_NOARGS, nullptr},
  {"cos_", THPVariable_cos_, METH_NOARGS, nullptr},
  {"cosh", THPVariable_cosh, METH_NOARGS, nullptr},
  {"cosh_", THPVariable_cosh_, METH_NOARGS, nullptr},
  {"count_nonzero", castPyCFunctionWithKeywords(THPVariable_count_nonzero), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"cov", castPyCFunctionWithKeywords(THPVariable_cov), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"cross", castPyCFunctionWithKeywords(THPVariable_cross), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"crow_indices", THPVariable_crow_indices, METH_NOARGS, nullptr},
  {"cummax", castPyCFunctionWithKeywords(THPVariable_cummax), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"cummin", castPyCFunctionWithKeywords(THPVariable_cummin), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"cumprod", castPyCFunctionWithKeywords(THPVariable_cumprod), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"cumprod_", castPyCFunctionWithKeywords(THPVariable_cumprod_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"cumsum", castPyCFunctionWithKeywords(THPVariable_cumsum), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"cumsum_", castPyCFunctionWithKeywords(THPVariable_cumsum_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"deg2rad", THPVariable_deg2rad, METH_NOARGS, nullptr},
  {"deg2rad_", THPVariable_deg2rad_, METH_NOARGS, nullptr},
  {"dense_dim", THPVariable_dense_dim, METH_NOARGS, nullptr},
  {"dequantize", THPVariable_dequantize, METH_NOARGS, nullptr},
  {"det", THPVariable_det, METH_NOARGS, nullptr},
  {"detach", THPVariable_detach, METH_NOARGS, nullptr},
  {"detach_", THPVariable_detach_, METH_NOARGS, nullptr},
  {"diag", castPyCFunctionWithKeywords(THPVariable_diag), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"diag_embed", castPyCFunctionWithKeywords(THPVariable_diag_embed), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"diagflat", castPyCFunctionWithKeywords(THPVariable_diagflat), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"diagonal", castPyCFunctionWithKeywords(THPVariable_diagonal), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"diagonal_scatter", castPyCFunctionWithKeywords(THPVariable_diagonal_scatter), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"diff", castPyCFunctionWithKeywords(THPVariable_diff), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"digamma", THPVariable_digamma, METH_NOARGS, nullptr},
  {"digamma_", THPVariable_digamma_, METH_NOARGS, nullptr},
  {"dist", castPyCFunctionWithKeywords(THPVariable_dist), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"div", castPyCFunctionWithKeywords(THPVariable_div), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"div_", castPyCFunctionWithKeywords(THPVariable_div_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"divide", castPyCFunctionWithKeywords(THPVariable_divide), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"divide_", castPyCFunctionWithKeywords(THPVariable_divide_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"dot", castPyCFunctionWithKeywords(THPVariable_dot), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"dsplit", castPyCFunctionWithKeywords(THPVariable_dsplit), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"eq", castPyCFunctionWithKeywords(THPVariable_eq), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"eq_", castPyCFunctionWithKeywords(THPVariable_eq_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"equal", castPyCFunctionWithKeywords(THPVariable_equal), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"erf", THPVariable_erf, METH_NOARGS, nullptr},
  {"erf_", THPVariable_erf_, METH_NOARGS, nullptr},
  {"erfc", THPVariable_erfc, METH_NOARGS, nullptr},
  {"erfc_", THPVariable_erfc_, METH_NOARGS, nullptr},
  {"erfinv", THPVariable_erfinv, METH_NOARGS, nullptr},
  {"erfinv_", THPVariable_erfinv_, METH_NOARGS, nullptr},
  {"exp", THPVariable_exp, METH_NOARGS, nullptr},
  {"exp2", THPVariable_exp2, METH_NOARGS, nullptr},
  {"exp2_", THPVariable_exp2_, METH_NOARGS, nullptr},
  {"exp_", THPVariable_exp_, METH_NOARGS, nullptr},
  {"expand", castPyCFunctionWithKeywords(THPVariable_expand), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"expand_as", castPyCFunctionWithKeywords(THPVariable_expand_as), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"expm1", THPVariable_expm1, METH_NOARGS, nullptr},
  {"expm1_", THPVariable_expm1_, METH_NOARGS, nullptr},
  {"exponential_", castPyCFunctionWithKeywords(THPVariable_exponential_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fill_", castPyCFunctionWithKeywords(THPVariable_fill_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fill_diagonal_", castPyCFunctionWithKeywords(THPVariable_fill_diagonal_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fix", THPVariable_fix, METH_NOARGS, nullptr},
  {"fix_", THPVariable_fix_, METH_NOARGS, nullptr},
  {"flatten", castPyCFunctionWithKeywords(THPVariable_flatten), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"flip", castPyCFunctionWithKeywords(THPVariable_flip), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fliplr", THPVariable_fliplr, METH_NOARGS, nullptr},
  {"flipud", THPVariable_flipud, METH_NOARGS, nullptr},
  {"float_power", castPyCFunctionWithKeywords(THPVariable_float_power), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"float_power_", castPyCFunctionWithKeywords(THPVariable_float_power_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"floor", THPVariable_floor, METH_NOARGS, nullptr},
  {"floor_", THPVariable_floor_, METH_NOARGS, nullptr},
  {"floor_divide", castPyCFunctionWithKeywords(THPVariable_floor_divide), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"floor_divide_", castPyCFunctionWithKeywords(THPVariable_floor_divide_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fmax", castPyCFunctionWithKeywords(THPVariable_fmax), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fmin", castPyCFunctionWithKeywords(THPVariable_fmin), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fmod", castPyCFunctionWithKeywords(THPVariable_fmod), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fmod_", castPyCFunctionWithKeywords(THPVariable_fmod_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"frac", THPVariable_frac, METH_NOARGS, nullptr},
  {"frac_", THPVariable_frac_, METH_NOARGS, nullptr},
  {"frexp", THPVariable_frexp, METH_NOARGS, nullptr},
  {"gather", castPyCFunctionWithKeywords(THPVariable_gather), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"gcd", castPyCFunctionWithKeywords(THPVariable_gcd), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"gcd_", castPyCFunctionWithKeywords(THPVariable_gcd_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"ge", castPyCFunctionWithKeywords(THPVariable_ge), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"ge_", castPyCFunctionWithKeywords(THPVariable_ge_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"geometric_", castPyCFunctionWithKeywords(THPVariable_geometric_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"geqrf", THPVariable_geqrf, METH_NOARGS, nullptr},
  {"ger", castPyCFunctionWithKeywords(THPVariable_ger), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"greater", castPyCFunctionWithKeywords(THPVariable_greater), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"greater_", castPyCFunctionWithKeywords(THPVariable_greater_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"greater_equal", castPyCFunctionWithKeywords(THPVariable_greater_equal), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"greater_equal_", castPyCFunctionWithKeywords(THPVariable_greater_equal_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"gt", castPyCFunctionWithKeywords(THPVariable_gt), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"gt_", castPyCFunctionWithKeywords(THPVariable_gt_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"hardshrink", castPyCFunctionWithKeywords(THPVariable_hardshrink), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"heaviside", castPyCFunctionWithKeywords(THPVariable_heaviside), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"heaviside_", castPyCFunctionWithKeywords(THPVariable_heaviside_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"histc", castPyCFunctionWithKeywords(THPVariable_histc), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"histogram", castPyCFunctionWithKeywords(THPVariable_histogram), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"hsplit", castPyCFunctionWithKeywords(THPVariable_hsplit), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"hypot", castPyCFunctionWithKeywords(THPVariable_hypot), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"hypot_", castPyCFunctionWithKeywords(THPVariable_hypot_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"i0", THPVariable_i0, METH_NOARGS, nullptr},
  {"i0_", THPVariable_i0_, METH_NOARGS, nullptr},
  {"igamma", castPyCFunctionWithKeywords(THPVariable_igamma), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"igamma_", castPyCFunctionWithKeywords(THPVariable_igamma_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"igammac", castPyCFunctionWithKeywords(THPVariable_igammac), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"igammac_", castPyCFunctionWithKeywords(THPVariable_igammac_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"index_add", castPyCFunctionWithKeywords(THPVariable_index_add), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"index_add_", castPyCFunctionWithKeywords(THPVariable_index_add_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"index_copy", castPyCFunctionWithKeywords(THPVariable_index_copy), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"index_copy_", castPyCFunctionWithKeywords(THPVariable_index_copy_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"index_fill", castPyCFunctionWithKeywords(THPVariable_index_fill), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"index_fill_", castPyCFunctionWithKeywords(THPVariable_index_fill_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"index_put", castPyCFunctionWithKeywords(THPVariable_index_put), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"index_put_", castPyCFunctionWithKeywords(THPVariable_index_put_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"index_reduce", castPyCFunctionWithKeywords(THPVariable_index_reduce), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"index_reduce_", castPyCFunctionWithKeywords(THPVariable_index_reduce_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"index_select", castPyCFunctionWithKeywords(THPVariable_index_select), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"indices", THPVariable_indices, METH_NOARGS, nullptr},
  {"inner", castPyCFunctionWithKeywords(THPVariable_inner), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"int_repr", THPVariable_int_repr, METH_NOARGS, nullptr},
  {"inverse", THPVariable_inverse, METH_NOARGS, nullptr},
  {"is_coalesced", THPVariable_is_coalesced, METH_NOARGS, nullptr},
  {"is_complex", THPVariable_is_complex, METH_NOARGS, nullptr},
  {"is_conj", THPVariable_is_conj, METH_NOARGS, nullptr},
  {"is_distributed", THPVariable_is_distributed, METH_NOARGS, nullptr},
  {"is_floating_point", THPVariable_is_floating_point, METH_NOARGS, nullptr},
  {"is_inference", THPVariable_is_inference, METH_NOARGS, nullptr},
  {"is_neg", THPVariable_is_neg, METH_NOARGS, nullptr},
  {"is_nonzero", THPVariable_is_nonzero, METH_NOARGS, nullptr},
  {"is_pinned", castPyCFunctionWithKeywords(THPVariable_is_pinned), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"is_same_size", castPyCFunctionWithKeywords(THPVariable_is_same_size), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"is_set_to", castPyCFunctionWithKeywords(THPVariable_is_set_to), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"is_signed", THPVariable_is_signed, METH_NOARGS, nullptr},
  {"isclose", castPyCFunctionWithKeywords(THPVariable_isclose), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"isfinite", THPVariable_isfinite, METH_NOARGS, nullptr},
  {"isinf", THPVariable_isinf, METH_NOARGS, nullptr},
  {"isnan", THPVariable_isnan, METH_NOARGS, nullptr},
  {"isneginf", THPVariable_isneginf, METH_NOARGS, nullptr},
  {"isposinf", THPVariable_isposinf, METH_NOARGS, nullptr},
  {"isreal", THPVariable_isreal, METH_NOARGS, nullptr},
  {"istft", castPyCFunctionWithKeywords(THPVariable_istft), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"kron", castPyCFunctionWithKeywords(THPVariable_kron), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"kthvalue", castPyCFunctionWithKeywords(THPVariable_kthvalue), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"lcm", castPyCFunctionWithKeywords(THPVariable_lcm), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"lcm_", castPyCFunctionWithKeywords(THPVariable_lcm_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"ldexp", castPyCFunctionWithKeywords(THPVariable_ldexp), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"ldexp_", castPyCFunctionWithKeywords(THPVariable_ldexp_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"le", castPyCFunctionWithKeywords(THPVariable_le), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"le_", castPyCFunctionWithKeywords(THPVariable_le_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"lerp", castPyCFunctionWithKeywords(THPVariable_lerp), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"lerp_", castPyCFunctionWithKeywords(THPVariable_lerp_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"less", castPyCFunctionWithKeywords(THPVariable_less), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"less_", castPyCFunctionWithKeywords(THPVariable_less_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"less_equal", castPyCFunctionWithKeywords(THPVariable_less_equal), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"less_equal_", castPyCFunctionWithKeywords(THPVariable_less_equal_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"lgamma", THPVariable_lgamma, METH_NOARGS, nullptr},
  {"lgamma_", THPVariable_lgamma_, METH_NOARGS, nullptr},
  {"log", THPVariable_log, METH_NOARGS, nullptr},
  {"log10", THPVariable_log10, METH_NOARGS, nullptr},
  {"log10_", THPVariable_log10_, METH_NOARGS, nullptr},
  {"log1p", THPVariable_log1p, METH_NOARGS, nullptr},
  {"log1p_", THPVariable_log1p_, METH_NOARGS, nullptr},
  {"log2", THPVariable_log2, METH_NOARGS, nullptr},
  {"log2_", THPVariable_log2_, METH_NOARGS, nullptr},
  {"log_", THPVariable_log_, METH_NOARGS, nullptr},
  {"log_normal_", castPyCFunctionWithKeywords(THPVariable_log_normal_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"log_softmax", castPyCFunctionWithKeywords(THPVariable_log_softmax), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"logaddexp", castPyCFunctionWithKeywords(THPVariable_logaddexp), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"logaddexp2", castPyCFunctionWithKeywords(THPVariable_logaddexp2), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"logcumsumexp", castPyCFunctionWithKeywords(THPVariable_logcumsumexp), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"logdet", THPVariable_logdet, METH_NOARGS, nullptr},
  {"logical_and", castPyCFunctionWithKeywords(THPVariable_logical_and), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"logical_and_", castPyCFunctionWithKeywords(THPVariable_logical_and_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"logical_not", THPVariable_logical_not, METH_NOARGS, nullptr},
  {"logical_not_", THPVariable_logical_not_, METH_NOARGS, nullptr},
  {"logical_or", castPyCFunctionWithKeywords(THPVariable_logical_or), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"logical_or_", castPyCFunctionWithKeywords(THPVariable_logical_or_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"logical_xor", castPyCFunctionWithKeywords(THPVariable_logical_xor), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"logical_xor_", castPyCFunctionWithKeywords(THPVariable_logical_xor_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"logit", castPyCFunctionWithKeywords(THPVariable_logit), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"logit_", castPyCFunctionWithKeywords(THPVariable_logit_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"logsumexp", castPyCFunctionWithKeywords(THPVariable_logsumexp), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"lt", castPyCFunctionWithKeywords(THPVariable_lt), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"lt_", castPyCFunctionWithKeywords(THPVariable_lt_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"lu_solve", castPyCFunctionWithKeywords(THPVariable_lu_solve), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"masked_fill", castPyCFunctionWithKeywords(THPVariable_masked_fill), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"masked_fill_", castPyCFunctionWithKeywords(THPVariable_masked_fill_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"masked_scatter", castPyCFunctionWithKeywords(THPVariable_masked_scatter), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"masked_scatter_", castPyCFunctionWithKeywords(THPVariable_masked_scatter_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"masked_select", castPyCFunctionWithKeywords(THPVariable_masked_select), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"matmul", castPyCFunctionWithKeywords(THPVariable_matmul), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"matrix_exp", THPVariable_matrix_exp, METH_NOARGS, nullptr},
  {"matrix_power", castPyCFunctionWithKeywords(THPVariable_matrix_power), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"max", castPyCFunctionWithKeywords(THPVariable_max), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"maximum", castPyCFunctionWithKeywords(THPVariable_maximum), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"mean", castPyCFunctionWithKeywords(THPVariable_mean), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"median", castPyCFunctionWithKeywords(THPVariable_median), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"min", castPyCFunctionWithKeywords(THPVariable_min), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"minimum", castPyCFunctionWithKeywords(THPVariable_minimum), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"mm", castPyCFunctionWithKeywords(THPVariable_mm), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"mode", castPyCFunctionWithKeywords(THPVariable_mode), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"moveaxis", castPyCFunctionWithKeywords(THPVariable_moveaxis), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"movedim", castPyCFunctionWithKeywords(THPVariable_movedim), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"msort", THPVariable_msort, METH_NOARGS, nullptr},
  {"mul", castPyCFunctionWithKeywords(THPVariable_mul), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"mul_", castPyCFunctionWithKeywords(THPVariable_mul_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"multinomial", castPyCFunctionWithKeywords(THPVariable_multinomial), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"multiply", castPyCFunctionWithKeywords(THPVariable_multiply), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"multiply_", castPyCFunctionWithKeywords(THPVariable_multiply_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"mv", castPyCFunctionWithKeywords(THPVariable_mv), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"mvlgamma", castPyCFunctionWithKeywords(THPVariable_mvlgamma), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"mvlgamma_", castPyCFunctionWithKeywords(THPVariable_mvlgamma_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"nan_to_num", castPyCFunctionWithKeywords(THPVariable_nan_to_num), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"nan_to_num_", castPyCFunctionWithKeywords(THPVariable_nan_to_num_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"nanmean", castPyCFunctionWithKeywords(THPVariable_nanmean), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"nanmedian", castPyCFunctionWithKeywords(THPVariable_nanmedian), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"nanquantile", castPyCFunctionWithKeywords(THPVariable_nanquantile), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"nansum", castPyCFunctionWithKeywords(THPVariable_nansum), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"narrow", castPyCFunctionWithKeywords(THPVariable_narrow), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"narrow_copy", castPyCFunctionWithKeywords(THPVariable_narrow_copy), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"ne", castPyCFunctionWithKeywords(THPVariable_ne), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"ne_", castPyCFunctionWithKeywords(THPVariable_ne_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"neg", THPVariable_neg, METH_NOARGS, nullptr},
  {"neg_", THPVariable_neg_, METH_NOARGS, nullptr},
  {"negative", THPVariable_negative, METH_NOARGS, nullptr},
  {"negative_", THPVariable_negative_, METH_NOARGS, nullptr},
  {"new_empty", castPyCFunctionWithKeywords(THPVariable_new_empty), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"new_empty_strided", castPyCFunctionWithKeywords(THPVariable_new_empty_strided), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"new_full", castPyCFunctionWithKeywords(THPVariable_new_full), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"new_ones", castPyCFunctionWithKeywords(THPVariable_new_ones), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"new_zeros", castPyCFunctionWithKeywords(THPVariable_new_zeros), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"nextafter", castPyCFunctionWithKeywords(THPVariable_nextafter), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"nextafter_", castPyCFunctionWithKeywords(THPVariable_nextafter_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"nonzero_static", castPyCFunctionWithKeywords(THPVariable_nonzero_static), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"norm", castPyCFunctionWithKeywords(THPVariable_norm), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"normal_", castPyCFunctionWithKeywords(THPVariable_normal_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"not_equal", castPyCFunctionWithKeywords(THPVariable_not_equal), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"not_equal_", castPyCFunctionWithKeywords(THPVariable_not_equal_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"orgqr", castPyCFunctionWithKeywords(THPVariable_orgqr), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"ormqr", castPyCFunctionWithKeywords(THPVariable_ormqr), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"outer", castPyCFunctionWithKeywords(THPVariable_outer), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"permute", castPyCFunctionWithKeywords(THPVariable_permute), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"pin_memory", castPyCFunctionWithKeywords(THPVariable_pin_memory), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"pinverse", castPyCFunctionWithKeywords(THPVariable_pinverse), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"polygamma", castPyCFunctionWithKeywords(THPVariable_polygamma), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"polygamma_", castPyCFunctionWithKeywords(THPVariable_polygamma_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"positive", THPVariable_positive, METH_NOARGS, nullptr},
  {"pow", castPyCFunctionWithKeywords(THPVariable_pow), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"pow_", castPyCFunctionWithKeywords(THPVariable_pow_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"prelu", castPyCFunctionWithKeywords(THPVariable_prelu), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"prod", castPyCFunctionWithKeywords(THPVariable_prod), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"put", castPyCFunctionWithKeywords(THPVariable_put), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"put_", castPyCFunctionWithKeywords(THPVariable_put_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"q_per_channel_axis", THPVariable_q_per_channel_axis, METH_NOARGS, nullptr},
  {"q_per_channel_scales", THPVariable_q_per_channel_scales, METH_NOARGS, nullptr},
  {"q_per_channel_zero_points", THPVariable_q_per_channel_zero_points, METH_NOARGS, nullptr},
  {"q_scale", THPVariable_q_scale, METH_NOARGS, nullptr},
  {"q_zero_point", THPVariable_q_zero_point, METH_NOARGS, nullptr},
  {"qr", castPyCFunctionWithKeywords(THPVariable_qr), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"qscheme", THPVariable_qscheme, METH_NOARGS, nullptr},
  {"quantile", castPyCFunctionWithKeywords(THPVariable_quantile), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"rad2deg", THPVariable_rad2deg, METH_NOARGS, nullptr},
  {"rad2deg_", THPVariable_rad2deg_, METH_NOARGS, nullptr},
  {"random_", castPyCFunctionWithKeywords(THPVariable_random_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"ravel", THPVariable_ravel, METH_NOARGS, nullptr},
  {"reciprocal", THPVariable_reciprocal, METH_NOARGS, nullptr},
  {"reciprocal_", THPVariable_reciprocal_, METH_NOARGS, nullptr},
  {"record_stream", castPyCFunctionWithKeywords(THPVariable_record_stream), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"refine_names", castPyCFunctionWithKeywords(THPVariable_refine_names), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"relu", THPVariable_relu, METH_NOARGS, nullptr},
  {"relu_", THPVariable_relu_, METH_NOARGS, nullptr},
  {"remainder", castPyCFunctionWithKeywords(THPVariable_remainder), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"remainder_", castPyCFunctionWithKeywords(THPVariable_remainder_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"rename", castPyCFunctionWithKeywords(THPVariable_rename), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"rename_", castPyCFunctionWithKeywords(THPVariable_rename_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"renorm", castPyCFunctionWithKeywords(THPVariable_renorm), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"renorm_", castPyCFunctionWithKeywords(THPVariable_renorm_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"repeat", castPyCFunctionWithKeywords(THPVariable_repeat), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"repeat_interleave", castPyCFunctionWithKeywords(THPVariable_repeat_interleave), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"reshape", castPyCFunctionWithKeywords(THPVariable_reshape), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"reshape_as", castPyCFunctionWithKeywords(THPVariable_reshape_as), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"resize_", castPyCFunctionWithKeywords(THPVariable_resize_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"resize_as_", castPyCFunctionWithKeywords(THPVariable_resize_as_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"resize_as_sparse_", castPyCFunctionWithKeywords(THPVariable_resize_as_sparse_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"resolve_conj", THPVariable_resolve_conj, METH_NOARGS, nullptr},
  {"resolve_neg", THPVariable_resolve_neg, METH_NOARGS, nullptr},
  {"retain_grad", THPVariable_retain_grad, METH_NOARGS, nullptr},
  {"roll", castPyCFunctionWithKeywords(THPVariable_roll), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"rot90", castPyCFunctionWithKeywords(THPVariable_rot90), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"round", castPyCFunctionWithKeywords(THPVariable_round), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"round_", castPyCFunctionWithKeywords(THPVariable_round_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"row_indices", THPVariable_row_indices, METH_NOARGS, nullptr},
  {"rsqrt", THPVariable_rsqrt, METH_NOARGS, nullptr},
  {"rsqrt_", THPVariable_rsqrt_, METH_NOARGS, nullptr},
  {"scatter", castPyCFunctionWithKeywords(THPVariable_scatter), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"scatter_", castPyCFunctionWithKeywords(THPVariable_scatter_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"scatter_add", castPyCFunctionWithKeywords(THPVariable_scatter_add), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"scatter_add_", castPyCFunctionWithKeywords(THPVariable_scatter_add_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"scatter_reduce", castPyCFunctionWithKeywords(THPVariable_scatter_reduce), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"scatter_reduce_", castPyCFunctionWithKeywords(THPVariable_scatter_reduce_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"select", castPyCFunctionWithKeywords(THPVariable_select), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"select_scatter", castPyCFunctionWithKeywords(THPVariable_select_scatter), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"sgn", THPVariable_sgn, METH_NOARGS, nullptr},
  {"sgn_", THPVariable_sgn_, METH_NOARGS, nullptr},
  {"sigmoid", THPVariable_sigmoid, METH_NOARGS, nullptr},
  {"sigmoid_", THPVariable_sigmoid_, METH_NOARGS, nullptr},
  {"sign", THPVariable_sign, METH_NOARGS, nullptr},
  {"sign_", THPVariable_sign_, METH_NOARGS, nullptr},
  {"signbit", THPVariable_signbit, METH_NOARGS, nullptr},
  {"sin", THPVariable_sin, METH_NOARGS, nullptr},
  {"sin_", THPVariable_sin_, METH_NOARGS, nullptr},
  {"sinc", THPVariable_sinc, METH_NOARGS, nullptr},
  {"sinc_", THPVariable_sinc_, METH_NOARGS, nullptr},
  {"sinh", THPVariable_sinh, METH_NOARGS, nullptr},
  {"sinh_", THPVariable_sinh_, METH_NOARGS, nullptr},
  {"slice_inverse", castPyCFunctionWithKeywords(THPVariable_slice_inverse), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"slice_scatter", castPyCFunctionWithKeywords(THPVariable_slice_scatter), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"slogdet", THPVariable_slogdet, METH_NOARGS, nullptr},
  {"smm", castPyCFunctionWithKeywords(THPVariable_smm), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"softmax", castPyCFunctionWithKeywords(THPVariable_softmax), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"sort", castPyCFunctionWithKeywords(THPVariable_sort), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"sparse_dim", THPVariable_sparse_dim, METH_NOARGS, nullptr},
  {"sparse_mask", castPyCFunctionWithKeywords(THPVariable_sparse_mask), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"sparse_resize_", castPyCFunctionWithKeywords(THPVariable_sparse_resize_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"sparse_resize_and_clear_", castPyCFunctionWithKeywords(THPVariable_sparse_resize_and_clear_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"split", castPyCFunctionWithKeywords(THPVariable_split), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"split_with_sizes", castPyCFunctionWithKeywords(THPVariable_split_with_sizes), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"sqrt", THPVariable_sqrt, METH_NOARGS, nullptr},
  {"sqrt_", THPVariable_sqrt_, METH_NOARGS, nullptr},
  {"square", THPVariable_square, METH_NOARGS, nullptr},
  {"square_", THPVariable_square_, METH_NOARGS, nullptr},
  {"squeeze", castPyCFunctionWithKeywords(THPVariable_squeeze), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"squeeze_", castPyCFunctionWithKeywords(THPVariable_squeeze_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"sspaddmm", castPyCFunctionWithKeywords(THPVariable_sspaddmm), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"std", castPyCFunctionWithKeywords(THPVariable_std), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"stft", castPyCFunctionWithKeywords(THPVariable_stft), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"sub", castPyCFunctionWithKeywords(THPVariable_sub), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"sub_", castPyCFunctionWithKeywords(THPVariable_sub_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"subtract", castPyCFunctionWithKeywords(THPVariable_subtract), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"subtract_", castPyCFunctionWithKeywords(THPVariable_subtract_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"sum", castPyCFunctionWithKeywords(THPVariable_sum), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"sum_to_size", castPyCFunctionWithKeywords(THPVariable_sum_to_size), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"svd", castPyCFunctionWithKeywords(THPVariable_svd), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"swapaxes", castPyCFunctionWithKeywords(THPVariable_swapaxes), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"swapaxes_", castPyCFunctionWithKeywords(THPVariable_swapaxes_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"swapdims", castPyCFunctionWithKeywords(THPVariable_swapdims), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"swapdims_", castPyCFunctionWithKeywords(THPVariable_swapdims_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"t", THPVariable_t, METH_NOARGS, nullptr},
  {"t_", THPVariable_t_, METH_NOARGS, nullptr},
  {"take", castPyCFunctionWithKeywords(THPVariable_take), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"take_along_dim", castPyCFunctionWithKeywords(THPVariable_take_along_dim), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"tan", THPVariable_tan, METH_NOARGS, nullptr},
  {"tan_", THPVariable_tan_, METH_NOARGS, nullptr},
  {"tanh", THPVariable_tanh, METH_NOARGS, nullptr},
  {"tanh_", THPVariable_tanh_, METH_NOARGS, nullptr},
  {"tensor_split", castPyCFunctionWithKeywords(THPVariable_tensor_split), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"tile", castPyCFunctionWithKeywords(THPVariable_tile), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"to_dense", castPyCFunctionWithKeywords(THPVariable_to_dense), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"to_mkldnn", castPyCFunctionWithKeywords(THPVariable_to_mkldnn), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"to_padded_tensor", castPyCFunctionWithKeywords(THPVariable_to_padded_tensor), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"to_sparse", castPyCFunctionWithKeywords(THPVariable_to_sparse), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"to_sparse_bsc", castPyCFunctionWithKeywords(THPVariable_to_sparse_bsc), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"to_sparse_bsr", castPyCFunctionWithKeywords(THPVariable_to_sparse_bsr), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"to_sparse_csc", castPyCFunctionWithKeywords(THPVariable_to_sparse_csc), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"to_sparse_csr", castPyCFunctionWithKeywords(THPVariable_to_sparse_csr), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"topk", castPyCFunctionWithKeywords(THPVariable_topk), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"trace", THPVariable_trace, METH_NOARGS, nullptr},
  {"transpose", castPyCFunctionWithKeywords(THPVariable_transpose), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"transpose_", castPyCFunctionWithKeywords(THPVariable_transpose_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"triangular_solve", castPyCFunctionWithKeywords(THPVariable_triangular_solve), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"tril", castPyCFunctionWithKeywords(THPVariable_tril), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"tril_", castPyCFunctionWithKeywords(THPVariable_tril_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"triu", castPyCFunctionWithKeywords(THPVariable_triu), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"triu_", castPyCFunctionWithKeywords(THPVariable_triu_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"true_divide", castPyCFunctionWithKeywords(THPVariable_true_divide), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"true_divide_", castPyCFunctionWithKeywords(THPVariable_true_divide_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"trunc", THPVariable_trunc, METH_NOARGS, nullptr},
  {"trunc_", THPVariable_trunc_, METH_NOARGS, nullptr},
  {"type_as", castPyCFunctionWithKeywords(THPVariable_type_as), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"unbind", castPyCFunctionWithKeywords(THPVariable_unbind), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"unflatten", castPyCFunctionWithKeywords(THPVariable_unflatten), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"unfold", castPyCFunctionWithKeywords(THPVariable_unfold), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"uniform_", castPyCFunctionWithKeywords(THPVariable_uniform_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"unsafe_chunk", castPyCFunctionWithKeywords(THPVariable_unsafe_chunk), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"unsafe_split", castPyCFunctionWithKeywords(THPVariable_unsafe_split), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"unsafe_split_with_sizes", castPyCFunctionWithKeywords(THPVariable_unsafe_split_with_sizes), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"unsqueeze", castPyCFunctionWithKeywords(THPVariable_unsqueeze), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"unsqueeze_", castPyCFunctionWithKeywords(THPVariable_unsqueeze_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"values", THPVariable_values, METH_NOARGS, nullptr},
  {"var", castPyCFunctionWithKeywords(THPVariable_var), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"vdot", castPyCFunctionWithKeywords(THPVariable_vdot), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"view", castPyCFunctionWithKeywords(THPVariable_view), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"view_as", castPyCFunctionWithKeywords(THPVariable_view_as), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"vsplit", castPyCFunctionWithKeywords(THPVariable_vsplit), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"where", castPyCFunctionWithKeywords(THPVariable_where), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"xlogy", castPyCFunctionWithKeywords(THPVariable_xlogy), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"xlogy_", castPyCFunctionWithKeywords(THPVariable_xlogy_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"zero_", THPVariable_zero_, METH_NOARGS, nullptr},
  {nullptr}
};

} // namespace torch::autograd
