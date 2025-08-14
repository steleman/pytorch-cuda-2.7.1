#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// @generated from ../tools/autograd/templates/python_torch_functions.cpp

// Python bindings for torch.* functions implemented through ATen.
//
// The functions are bound as static methods on a class
// torch._C._VariableFunctions which is also aliased as Variable._torch
// and also copied into 'torch' module.

#include <Python.h>

// Undefine the copysign macro so that at::copysign works as intended with MSVC
// https://github.com/python/cpython/blob/c60394c7fc9cc09b16e9675a3eeb5844b6d8523f/PC/pyconfig.h#L196
#ifdef _MSC_VER
#undef copysign
#endif // _MSC_VER

#include "torch/csrc/autograd/python_torch_functions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/Dtype.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/out_types.h"
#include "torch/csrc/utils/pybind.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/tensor_layouts.h"
#include "torch/csrc/utils/tensor_new.h"
#include "torch/csrc/utils/tensor_numpy.h"
#include "torch/csrc/jit/frontend/tracer.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/utils/device_lazy_init.h"
#include "torch/csrc/autograd/generated/python_return_types.h"

#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_cast_Char.h>
#include <ATen/ops/_cast_Short.h>
#include <ATen/ops/_cast_Half.h>
#include <ATen/ops/_make_dual.h>
#include <ATen/ops/_unpack_dual.h>
#include <ATen/ops/_functional_assert_scalar.h>
#include <ATen/ops/_functional_assert_async.h>
#include <ATen/ops/sym_constrain_range_for_size.h>
#include <ATen/ops/_functional_sym_constrain_range.h>
#include <ATen/ops/_cudnn_ctc_loss.h>
#include <ATen/ops/_cudnn_init_dropout_state.h>
#include <ATen/ops/_debug_has_internal_overlap.h>
#include <ATen/ops/_masked_scale.h>
#include <ATen/ops/_sobol_engine_initialize_state.h>
#include <ATen/ops/feature_alpha_dropout.h>
#include <ATen/ops/feature_alpha_dropout.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/absolute.h>
#include <ATen/ops/view_as_real.h>
#include <ATen/ops/real.h>
#include <ATen/ops/_conj_physical.h>
#include <ATen/ops/resolve_conj.h>
#include <ATen/ops/_neg_view.h>
#include <ATen/ops/acos.h>
#include <ATen/ops/acos.h>
#include <ATen/ops/arccos.h>
#include <ATen/ops/arccos.h>
#include <ATen/ops/adaptive_avg_pool1d.h>
#include <ATen/ops/addmv.h>
#include <ATen/ops/addmv.h>
#include <ATen/ops/_is_all_true.h>
#include <ATen/ops/_test_check_tensor.h>
#include <ATen/ops/argmin.h>
#include <ATen/ops/asinh.h>
#include <ATen/ops/asinh.h>
#include <ATen/ops/arcsinh.h>
#include <ATen/ops/arcsinh.h>
#include <ATen/ops/asin.h>
#include <ATen/ops/asin.h>
#include <ATen/ops/arctan.h>
#include <ATen/ops/arctan.h>
#include <ATen/ops/atleast_2d.h>
#include <ATen/ops/atleast_3d.h>
#include <ATen/ops/quantized_batch_norm.h>
#include <ATen/ops/_batch_norm_impl_index.h>
#include <ATen/ops/bincount.h>
#include <ATen/ops/bitwise_not.h>
#include <ATen/ops/logical_or.h>
#include <ATen/ops/unsafe_chunk.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/clamp_max.h>
#include <ATen/ops/clamp_max.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/conv2d.h>
#include <ATen/ops/conv_tbc.h>
#include <ATen/ops/conv_transpose1d.h>
#include <ATen/ops/conv_transpose3d.h>
#include <ATen/ops/cosine_embedding_loss.h>
#include <ATen/ops/cudnn_affine_grid_generator.h>
#include <ATen/ops/cudnn_batch_norm.h>
#include <ATen/ops/_mps_convolution_transpose.h>
#include <ATen/ops/cudnn_convolution_add_relu.h>
#include <ATen/ops/cudnn_grid_sampler.h>
#include <ATen/ops/_cummax_helper.h>
#include <ATen/ops/cumprod.h>
#include <ATen/ops/gradient.h>
#include <ATen/ops/div.h>
#include <ATen/ops/true_divide.h>
#include <ATen/ops/_embedding_bag_forward_only.h>
#include <ATen/ops/row_stack.h>
#include <ATen/ops/_empty_per_channel_affine_quantized.h>
#include <ATen/ops/_resize_output.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/flatten.h>
#include <ATen/ops/fill.h>
#include <ATen/ops/fill.h>
#include <ATen/ops/grid_sampler_2d.h>
#include <ATen/ops/hann_window.h>
#include <ATen/ops/kaiser_window.h>
#include <ATen/ops/_fft_r2c.h>
#include <ATen/ops/_validate_compressed_sparse_indices.h>
#include <ATen/ops/_cufft_get_plan_cache_size.h>
#include <ATen/ops/_unsafe_masked_index.h>
#include <ATen/ops/index_put.h>
#include <ATen/ops/index_put.h>
#include <ATen/ops/_unsafe_index_put.h>
#include <ATen/ops/is_distributed.h>
#include <ATen/ops/is_complex.h>
#include <ATen/ops/isreal.h>
#include <ATen/ops/kron.h>
#include <ATen/ops/native_layer_norm.h>
#include <ATen/ops/_cslt_compress.h>
#include <ATen/ops/_sparse_semi_structured_tile.h>
#include <ATen/ops/_sparse_semi_structured_linear.h>
#include <ATen/ops/fbgemm_linear_fp16_weight.h>
#include <ATen/ops/ldexp.h>
#include <ATen/ops/ldexp.h>
#include <ATen/ops/log10.h>
#include <ATen/ops/log10.h>
#include <ATen/ops/logaddexp2.h>
#include <ATen/ops/_log_softmax_backward_data.h>
#include <ATen/ops/logcumsumexp.h>
#include <ATen/ops/margin_ranking_loss.h>
#include <ATen/ops/matrix_exp.h>
#include <ATen/ops/amax.h>
#include <ATen/ops/max_pool1d.h>
#include <ATen/ops/mkldnn_max_pool3d.h>
#include <ATen/ops/quantized_max_pool3d.h>
#include <ATen/ops/max_pool3d.h>
#include <ATen/ops/median.h>
#include <ATen/ops/_mps_convolution.h>
#include <ATen/ops/miopen_convolution.h>
#include <ATen/ops/miopen_convolution_transpose.h>
#include <ATen/ops/miopen_convolution_add_relu.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/_int_mm.h>
#include <ATen/ops/_convert_weight_to_int4pack_for_cpu.h>
#include <ATen/ops/multiply.h>
#include <ATen/ops/mvlgamma.h>
#include <ATen/ops/narrow_copy.h>
#include <ATen/ops/_native_batch_norm_legit_no_training.h>
#include <ATen/ops/batch_norm_elemt.h>
#include <ATen/ops/batch_norm_gather_stats_with_counts.h>
#include <ATen/ops/batch_norm_backward_elemt.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/pairwise_distance.h>
#include <ATen/ops/cdist.h>
#include <ATen/ops/cosine_similarity.h>
#include <ATen/ops/movedim.h>
#include <ATen/ops/adjoint.h>
#include <ATen/ops/channel_shuffle.h>
#include <ATen/ops/pinverse.h>
#include <ATen/ops/poisson_nll_loss.h>
#include <ATen/ops/deg2rad.h>
#include <ATen/ops/deg2rad.h>
#include <ATen/ops/randn_like.h>
#include <ATen/ops/repeat_interleave.h>
#include <ATen/ops/_prelu_kernel.h>
#include <ATen/ops/hardshrink.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/celu.h>
#include <ATen/ops/celu.h>
#include <ATen/ops/sinc.h>
#include <ATen/ops/sinc.h>
#include <ATen/ops/sinh.h>
#include <ATen/ops/sinh.h>
#include <ATen/ops/detach.h>
#include <ATen/ops/detach.h>
#include <ATen/ops/_softmax.h>
#include <ATen/ops/_softmax_backward_data.h>
#include <ATen/ops/unsafe_split.h>
#include <ATen/ops/split.h>
#include <ATen/ops/hsplit.h>
#include <ATen/ops/squeeze.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/_stack.h>
#include <ATen/ops/istft.h>
#include <ATen/ops/nansum.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/square.h>
#include <ATen/ops/square.h>
#include <ATen/ops/t.h>
#include <ATen/ops/tanh.h>
#include <ATen/ops/tanh.h>
#include <ATen/ops/tensordot.h>
#include <ATen/ops/tile.h>
#include <ATen/ops/_mkldnn_transpose.h>
#include <ATen/ops/_mkldnn_transpose.h>
#include <ATen/ops/rot90.h>
#include <ATen/ops/_nested_from_padded.h>
#include <ATen/ops/_nested_get_values.h>
#include <ATen/ops/_nested_get_offsets.h>
#include <ATen/ops/_nested_get_jagged_dummy.h>
#include <ATen/ops/triplet_margin_loss.h>
#include <ATen/ops/fix.h>
#include <ATen/ops/fix.h>
#include <ATen/ops/_has_compatible_shallow_copy_type.h>
#include <ATen/ops/_unique.h>
#include <ATen/ops/vander.h>
#include <ATen/ops/var_mean.h>
#include <ATen/ops/_weight_norm.h>
#include <ATen/ops/_standard_gamma_grad.h>
#include <ATen/ops/poisson.h>
#include <ATen/ops/native_norm.h>
#include <ATen/ops/_sparse_softmax_backward_data.h>
#include <ATen/ops/norm.h>
#include <ATen/ops/frexp.h>
#include <ATen/ops/positive.h>
#include <ATen/ops/resize_as.h>
#include <ATen/ops/zero.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/subtract.h>
#include <ATen/ops/rsub.h>
#include <ATen/ops/_validate_sparse_csc_tensor_args.h>
#include <ATen/ops/_coalesce.h>
#include <ATen/ops/unbind.h>
#include <ATen/ops/quantize_per_tensor_dynamic.h>
#include <ATen/ops/quantize_per_tensor.h>
#include <ATen/ops/quantize_per_channel.h>
#include <ATen/ops/q_zero_point.h>
#include <ATen/ops/int_repr.h>
#include <ATen/ops/_make_per_tensor_quantized_tensor.h>
#include <ATen/ops/_make_per_channel_quantized_tensor.h>
#include <ATen/ops/fake_quantize_per_channel_affine.h>
#include <ATen/ops/_fused_moving_avg_obs_fq_helper.h>
#include <ATen/ops/_choose_qparams_per_tensor.h>
#include <ATen/ops/meshgrid.h>
#include <ATen/ops/promote_types.h>
#include <ATen/ops/gru.h>
#include <ATen/ops/rnn_tanh_cell.h>
#include <ATen/ops/quantized_gru_cell.h>
#include <ATen/ops/masked_scatter.h>
#include <ATen/ops/put.h>
#include <ATen/ops/or.h>
#include <ATen/ops/xor.h>
#include <ATen/ops/lshift.h>
#include <ATen/ops/bitwise_right_shift.h>
#include <ATen/ops/tril.h>
#include <ATen/ops/tril_indices.h>
#include <ATen/ops/not_equal.h>
#include <ATen/ops/greater_equal.h>
#include <ATen/ops/less.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/addcmul.h>
#include <ATen/ops/triangular_solve.h>
#include <ATen/ops/swapaxes.h>
#include <ATen/ops/cholesky_inverse.h>
#include <ATen/ops/geqrf.h>
#include <ATen/ops/orgqr.h>
#include <ATen/ops/ormqr.h>
#include <ATen/ops/lu_solve.h>
#include <ATen/ops/lgamma.h>
#include <ATen/ops/digamma.h>
#include <ATen/ops/erfinv.h>
#include <ATen/ops/sign.h>
#include <ATen/ops/dist.h>
#include <ATen/ops/arctan2.h>
#include <ATen/ops/histc.h>
#include <ATen/ops/histogram.h>
#include <ATen/ops/_histogramdd_bin_edges.h>
#include <ATen/ops/histogramdd.h>
#include <ATen/ops/fmax.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/minimum.h>
#include <ATen/ops/nanquantile.h>
#include <ATen/ops/msort.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/float_power.h>
#include <ATen/ops/_foreach_add.h>
#include <ATen/ops/_foreach_add.h>
#include <ATen/ops/_foreach_sub.h>
#include <ATen/ops/_foreach_sub.h>
#include <ATen/ops/_foreach_clamp_max.h>
#include <ATen/ops/_foreach_clamp_max.h>
#include <ATen/ops/_foreach_acos.h>
#include <ATen/ops/_foreach_acos.h>
#include <ATen/ops/_foreach_asin.h>
#include <ATen/ops/_foreach_asin.h>
#include <ATen/ops/_foreach_cos.h>
#include <ATen/ops/_foreach_cos.h>
#include <ATen/ops/_foreach_cosh.h>
#include <ATen/ops/_foreach_cosh.h>
#include <ATen/ops/_foreach_exp.h>
#include <ATen/ops/_foreach_exp.h>
#include <ATen/ops/_foreach_floor.h>
#include <ATen/ops/_foreach_floor.h>
#include <ATen/ops/_foreach_log1p.h>
#include <ATen/ops/_foreach_log1p.h>
#include <ATen/ops/_foreach_log2.h>
#include <ATen/ops/_foreach_log2.h>
#include <ATen/ops/_foreach_neg.h>
#include <ATen/ops/_foreach_neg.h>
#include <ATen/ops/_foreach_norm.h>
#include <ATen/ops/_foreach_sigmoid.h>
#include <ATen/ops/_foreach_sigmoid.h>
#include <ATen/ops/_foreach_sign.h>
#include <ATen/ops/_foreach_sign.h>
#include <ATen/ops/_foreach_sqrt.h>
#include <ATen/ops/_foreach_sqrt.h>
#include <ATen/ops/_foreach_tanh.h>
#include <ATen/ops/_foreach_tanh.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/mkldnn_adaptive_avg_pool2d.h>
#include <ATen/ops/column_stack.h>
#include <ATen/ops/isinf.h>
#include <ATen/ops/isneginf.h>
#include <ATen/ops/_linalg_slogdet.h>
#include <ATen/ops/slogdet.h>
#include <ATen/ops/inverse.h>
#include <ATen/ops/inner.h>
#include <ATen/ops/_linalg_solve_ex.h>
#include <ATen/ops/_test_parallel_materialize.h>
#include <ATen/ops/_test_autograd_multiple_dispatch.h>
#include <ATen/ops/_test_autograd_multiple_dispatch_view_copy.h>
#include <ATen/ops/segment_reduce.h>
#include <ATen/ops/_conj_copy.h>
#include <ATen/ops/_sparse_broadcast_to_copy.h>
#include <ATen/ops/select_copy.h>
#include <ATen/ops/squeeze_copy.h>
#include <ATen/ops/transpose_copy.h>
#include <ATen/ops/values_copy.h>
#include <ATen/ops/unfold_copy.h>
#include <ATen/ops/_fused_sdp_choice.h>
#include <ATen/ops/_scaled_dot_product_efficient_attention.h>
#include <ATen/ops/_scaled_dot_product_cudnn_attention.h>
#include <ATen/ops/_fill_mem_eff_dropout_mask.h>
#include <ATen/ops/_foobar.h>
#include <ATen/ops/_fused_adagrad.h>
#include <ATen/ops/_propagate_xla_data.h>
#endif

#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <utility>

using at::Tensor;
using at::Device;
using at::Layout;
using at::Scalar;
using at::ScalarType;
using at::Backend;
using at::OptionalDeviceGuard;
using at::DeviceGuard;
using at::TensorOptions;
using at::IntArrayRef;
using at::Generator;
using at::TensorList;
using at::Dimname;
using at::DimnameList;
using at::ArrayRef;

using torch::utils::check_out_type_matches;
using namespace torch::autograd::utils;

// NOTE: See [Sharded File] comment in VariableType

namespace torch::autograd {

// generated forward declarations start here

static PyObject * THPVariable__cast_Char(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cast_Short(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cast_Half(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__make_dual(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__unpack_dual(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__functional_assert_scalar(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__functional_assert_async(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sym_constrain_range_for_size(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__functional_sym_constrain_range(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cudnn_ctc_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cudnn_init_dropout_state(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__debug_has_internal_overlap(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__masked_scale(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sobol_engine_initialize_state_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_feature_alpha_dropout(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_feature_alpha_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_abs(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_abs_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_absolute(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_view_as_real(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_real(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__conj_physical(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_resolve_conj(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__neg_view(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_acos(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_acos_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_arccos(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_arccos_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_adaptive_avg_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_addmv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_addmv_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__is_all_true(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__test_check_tensor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_argmin(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_asinh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_asinh_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_arcsinh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_arcsinh_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_asin(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_asin_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_arctan(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_arctan_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_atleast_2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_atleast_3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantized_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__batch_norm_impl_index(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bincount(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bitwise_not(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_logical_or(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_unsafe_chunk(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_clamp(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_clamp_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_clamp_max(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_clamp_max_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_constant_pad_nd(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_convolution(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_conv2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_conv_tbc(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_conv_transpose1d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_conv_transpose3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cosine_embedding_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cudnn_affine_grid_generator(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cudnn_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__mps_convolution_transpose(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cudnn_convolution_add_relu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cudnn_grid_sampler(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cummax_helper(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cumprod(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_gradient(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_div(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_true_divide(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__embedding_bag_forward_only(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_row_stack(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__empty_per_channel_affine_quantized(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__resize_output_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_empty_strided(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_flatten(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fill(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fill_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_grid_sampler_2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_hann_window(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_kaiser_window(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__fft_r2c(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__validate_compressed_sparse_indices(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cufft_get_plan_cache_size(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__unsafe_masked_index(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_index_put_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_index_put(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__unsafe_index_put(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_is_distributed(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_is_complex(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_isreal(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_kron(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_native_layer_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cslt_compress(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_semi_structured_tile(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_semi_structured_linear(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fbgemm_linear_fp16_weight(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ldexp(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ldexp_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_log10(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_log10_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_logaddexp2(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__log_softmax_backward_data(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_logcumsumexp(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_margin_ranking_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_matrix_exp(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_amax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_max_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mkldnn_max_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantized_max_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_max_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_median(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__mps_convolution(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_miopen_convolution(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_miopen_convolution_transpose(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_miopen_convolution_add_relu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__int_mm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__convert_weight_to_int4pack_for_cpu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_multiply(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mvlgamma(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_narrow_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__native_batch_norm_legit_no_training(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_batch_norm_elemt(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_batch_norm_gather_stats_with_counts(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_batch_norm_backward_elemt(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ones_like(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_pairwise_distance(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cdist(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cosine_similarity(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_movedim(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_adjoint(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_channel_shuffle(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_pinverse(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_poisson_nll_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_deg2rad(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_deg2rad_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_randn_like(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_repeat_interleave(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__prelu_kernel(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_hardshrink(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rsqrt(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rsqrt_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_celu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_celu_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sinc(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sinc_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sinh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sinh_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_detach(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_detach_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__softmax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__softmax_backward_data(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_unsafe_split(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_split(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_hsplit(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_squeeze(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_stack(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__stack(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_istft(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_nansum(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sqrt(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sqrt_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_square(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_square_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_t(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_tanh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_tanh_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_tensordot(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_tile(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__mkldnn_transpose(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__mkldnn_transpose_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rot90(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_from_padded(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_get_values(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_get_offsets(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_get_jagged_dummy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_triplet_margin_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fix(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fix_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__has_compatible_shallow_copy_type(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__unique(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_vander(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_var_mean(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__weight_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__standard_gamma_grad(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_poisson(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_native_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_softmax_backward_data(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_frexp(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_positive(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_resize_as_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_zero_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sub(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_subtract(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rsub(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__validate_sparse_csc_tensor_args(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__coalesce(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_unbind(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantize_per_tensor_dynamic(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantize_per_tensor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantize_per_channel(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_q_zero_point(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_int_repr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__make_per_tensor_quantized_tensor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__make_per_channel_quantized_tensor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fake_quantize_per_channel_affine(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__fused_moving_avg_obs_fq_helper(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__choose_qparams_per_tensor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_meshgrid(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_promote_types(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_gru(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rnn_tanh_cell(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantized_gru_cell(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_masked_scatter(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_put(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable___or__(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable___xor__(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable___lshift__(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bitwise_right_shift(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_tril(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_tril_indices(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_not_equal(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_greater_equal(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_less(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_index_select(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_gather(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_addcmul(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_triangular_solve(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_swapaxes(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cholesky_inverse(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_geqrf(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_orgqr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ormqr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_lu_solve(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_lgamma(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_digamma(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_erfinv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sign(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_dist(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_arctan2(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_histc(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_histogram(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__histogramdd_bin_edges(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_histogramdd(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fmax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_maximum(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_minimum(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_nanquantile(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_msort(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_pow(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_float_power(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_add(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_add_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_sub(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_sub_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_clamp_max(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_clamp_max_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_acos(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_acos_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_asin(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_asin_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_cos(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_cos_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_cosh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_cosh_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_exp(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_exp_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_floor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_floor_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_log1p(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_log1p_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_log2(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_log2_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_neg(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_neg_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_sigmoid(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_sigmoid_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_sign(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_sign_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_sqrt(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_sqrt_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_tanh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_tanh_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__convert_indices_from_csr_to_coo(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mkldnn_adaptive_avg_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_column_stack(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_isinf(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_isneginf(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__linalg_slogdet(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_slogdet(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_inverse(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_inner(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__linalg_solve_ex(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__test_parallel_materialize(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__test_autograd_multiple_dispatch(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__test_autograd_multiple_dispatch_view_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_segment_reduce(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__conj_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_broadcast_to_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_select_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_squeeze_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_transpose_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_values_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_unfold_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__fused_sdp_choice(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__scaled_dot_product_efficient_attention(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__scaled_dot_product_cudnn_attention(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__fill_mem_eff_dropout_mask_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foobar(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__fused_adagrad_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__propagate_xla_data(PyObject* self_, PyObject* args, PyObject* kwargs);

static PyMethodDef torch_functions_shard[] = {
  {"_cast_Char", castPyCFunctionWithKeywords(THPVariable__cast_Char), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_cast_Short", castPyCFunctionWithKeywords(THPVariable__cast_Short), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_cast_Half", castPyCFunctionWithKeywords(THPVariable__cast_Half), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_make_dual", castPyCFunctionWithKeywords(THPVariable__make_dual), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_unpack_dual", castPyCFunctionWithKeywords(THPVariable__unpack_dual), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_functional_assert_scalar", castPyCFunctionWithKeywords(THPVariable__functional_assert_scalar), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_functional_assert_async", castPyCFunctionWithKeywords(THPVariable__functional_assert_async), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sym_constrain_range_for_size", castPyCFunctionWithKeywords(THPVariable_sym_constrain_range_for_size), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_functional_sym_constrain_range", castPyCFunctionWithKeywords(THPVariable__functional_sym_constrain_range), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_cudnn_ctc_loss", castPyCFunctionWithKeywords(THPVariable__cudnn_ctc_loss), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_cudnn_init_dropout_state", castPyCFunctionWithKeywords(THPVariable__cudnn_init_dropout_state), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_debug_has_internal_overlap", castPyCFunctionWithKeywords(THPVariable__debug_has_internal_overlap), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_masked_scale", castPyCFunctionWithKeywords(THPVariable__masked_scale), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sobol_engine_initialize_state_", castPyCFunctionWithKeywords(THPVariable__sobol_engine_initialize_state_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"feature_alpha_dropout", castPyCFunctionWithKeywords(THPVariable_feature_alpha_dropout), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"feature_alpha_dropout_", castPyCFunctionWithKeywords(THPVariable_feature_alpha_dropout_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"abs", castPyCFunctionWithKeywords(THPVariable_abs), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"abs_", castPyCFunctionWithKeywords(THPVariable_abs_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"absolute", castPyCFunctionWithKeywords(THPVariable_absolute), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"view_as_real", castPyCFunctionWithKeywords(THPVariable_view_as_real), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"real", castPyCFunctionWithKeywords(THPVariable_real), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_conj_physical", castPyCFunctionWithKeywords(THPVariable__conj_physical), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"resolve_conj", castPyCFunctionWithKeywords(THPVariable_resolve_conj), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_neg_view", castPyCFunctionWithKeywords(THPVariable__neg_view), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"acos", castPyCFunctionWithKeywords(THPVariable_acos), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"acos_", castPyCFunctionWithKeywords(THPVariable_acos_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"arccos", castPyCFunctionWithKeywords(THPVariable_arccos), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"arccos_", castPyCFunctionWithKeywords(THPVariable_arccos_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"adaptive_avg_pool1d", castPyCFunctionWithKeywords(THPVariable_adaptive_avg_pool1d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"addmv", castPyCFunctionWithKeywords(THPVariable_addmv), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"addmv_", castPyCFunctionWithKeywords(THPVariable_addmv_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_is_all_true", castPyCFunctionWithKeywords(THPVariable__is_all_true), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_test_check_tensor", castPyCFunctionWithKeywords(THPVariable__test_check_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"argmin", castPyCFunctionWithKeywords(THPVariable_argmin), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"asinh", castPyCFunctionWithKeywords(THPVariable_asinh), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"asinh_", castPyCFunctionWithKeywords(THPVariable_asinh_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"arcsinh", castPyCFunctionWithKeywords(THPVariable_arcsinh), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"arcsinh_", castPyCFunctionWithKeywords(THPVariable_arcsinh_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"asin", castPyCFunctionWithKeywords(THPVariable_asin), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"asin_", castPyCFunctionWithKeywords(THPVariable_asin_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"arctan", castPyCFunctionWithKeywords(THPVariable_arctan), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"arctan_", castPyCFunctionWithKeywords(THPVariable_arctan_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"atleast_2d", castPyCFunctionWithKeywords(THPVariable_atleast_2d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"atleast_3d", castPyCFunctionWithKeywords(THPVariable_atleast_3d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"quantized_batch_norm", castPyCFunctionWithKeywords(THPVariable_quantized_batch_norm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_batch_norm_impl_index", castPyCFunctionWithKeywords(THPVariable__batch_norm_impl_index), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"bincount", castPyCFunctionWithKeywords(THPVariable_bincount), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"bitwise_not", castPyCFunctionWithKeywords(THPVariable_bitwise_not), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"logical_or", castPyCFunctionWithKeywords(THPVariable_logical_or), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"unsafe_chunk", castPyCFunctionWithKeywords(THPVariable_unsafe_chunk), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"clamp", castPyCFunctionWithKeywords(THPVariable_clamp), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"clamp_", castPyCFunctionWithKeywords(THPVariable_clamp_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"clamp_max", castPyCFunctionWithKeywords(THPVariable_clamp_max), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"clamp_max_", castPyCFunctionWithKeywords(THPVariable_clamp_max_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"constant_pad_nd", castPyCFunctionWithKeywords(THPVariable_constant_pad_nd), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"convolution", castPyCFunctionWithKeywords(THPVariable_convolution), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"conv2d", castPyCFunctionWithKeywords(THPVariable_conv2d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"conv_tbc", castPyCFunctionWithKeywords(THPVariable_conv_tbc), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"conv_transpose1d", castPyCFunctionWithKeywords(THPVariable_conv_transpose1d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"conv_transpose3d", castPyCFunctionWithKeywords(THPVariable_conv_transpose3d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cosine_embedding_loss", castPyCFunctionWithKeywords(THPVariable_cosine_embedding_loss), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cudnn_affine_grid_generator", castPyCFunctionWithKeywords(THPVariable_cudnn_affine_grid_generator), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cudnn_batch_norm", castPyCFunctionWithKeywords(THPVariable_cudnn_batch_norm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_mps_convolution_transpose", castPyCFunctionWithKeywords(THPVariable__mps_convolution_transpose), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cudnn_convolution_add_relu", castPyCFunctionWithKeywords(THPVariable_cudnn_convolution_add_relu), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cudnn_grid_sampler", castPyCFunctionWithKeywords(THPVariable_cudnn_grid_sampler), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_cummax_helper", castPyCFunctionWithKeywords(THPVariable__cummax_helper), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cumprod", castPyCFunctionWithKeywords(THPVariable_cumprod), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"gradient", castPyCFunctionWithKeywords(THPVariable_gradient), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"div", castPyCFunctionWithKeywords(THPVariable_div), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"true_divide", castPyCFunctionWithKeywords(THPVariable_true_divide), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_embedding_bag_forward_only", castPyCFunctionWithKeywords(THPVariable__embedding_bag_forward_only), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"row_stack", castPyCFunctionWithKeywords(THPVariable_row_stack), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_empty_per_channel_affine_quantized", castPyCFunctionWithKeywords(THPVariable__empty_per_channel_affine_quantized), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_resize_output_", castPyCFunctionWithKeywords(THPVariable__resize_output_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"empty_strided", castPyCFunctionWithKeywords(THPVariable_empty_strided), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"flatten", castPyCFunctionWithKeywords(THPVariable_flatten), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"fill", castPyCFunctionWithKeywords(THPVariable_fill), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"fill_", castPyCFunctionWithKeywords(THPVariable_fill_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"grid_sampler_2d", castPyCFunctionWithKeywords(THPVariable_grid_sampler_2d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"hann_window", castPyCFunctionWithKeywords(THPVariable_hann_window), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"kaiser_window", castPyCFunctionWithKeywords(THPVariable_kaiser_window), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_fft_r2c", castPyCFunctionWithKeywords(THPVariable__fft_r2c), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_validate_compressed_sparse_indices", castPyCFunctionWithKeywords(THPVariable__validate_compressed_sparse_indices), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_cufft_get_plan_cache_size", castPyCFunctionWithKeywords(THPVariable__cufft_get_plan_cache_size), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_unsafe_masked_index", castPyCFunctionWithKeywords(THPVariable__unsafe_masked_index), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"index_put_", castPyCFunctionWithKeywords(THPVariable_index_put_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"index_put", castPyCFunctionWithKeywords(THPVariable_index_put), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_unsafe_index_put", castPyCFunctionWithKeywords(THPVariable__unsafe_index_put), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"is_distributed", castPyCFunctionWithKeywords(THPVariable_is_distributed), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"is_complex", castPyCFunctionWithKeywords(THPVariable_is_complex), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"isreal", castPyCFunctionWithKeywords(THPVariable_isreal), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"kron", castPyCFunctionWithKeywords(THPVariable_kron), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"native_layer_norm", castPyCFunctionWithKeywords(THPVariable_native_layer_norm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_cslt_compress", castPyCFunctionWithKeywords(THPVariable__cslt_compress), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_semi_structured_tile", castPyCFunctionWithKeywords(THPVariable__sparse_semi_structured_tile), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_semi_structured_linear", castPyCFunctionWithKeywords(THPVariable__sparse_semi_structured_linear), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"fbgemm_linear_fp16_weight", castPyCFunctionWithKeywords(THPVariable_fbgemm_linear_fp16_weight), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"ldexp", castPyCFunctionWithKeywords(THPVariable_ldexp), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"ldexp_", castPyCFunctionWithKeywords(THPVariable_ldexp_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"log10", castPyCFunctionWithKeywords(THPVariable_log10), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"log10_", castPyCFunctionWithKeywords(THPVariable_log10_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"logaddexp2", castPyCFunctionWithKeywords(THPVariable_logaddexp2), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_log_softmax_backward_data", castPyCFunctionWithKeywords(THPVariable__log_softmax_backward_data), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"logcumsumexp", castPyCFunctionWithKeywords(THPVariable_logcumsumexp), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"margin_ranking_loss", castPyCFunctionWithKeywords(THPVariable_margin_ranking_loss), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"matrix_exp", castPyCFunctionWithKeywords(THPVariable_matrix_exp), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"amax", castPyCFunctionWithKeywords(THPVariable_amax), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"max_pool1d", castPyCFunctionWithKeywords(THPVariable_max_pool1d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"mkldnn_max_pool3d", castPyCFunctionWithKeywords(THPVariable_mkldnn_max_pool3d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"quantized_max_pool3d", castPyCFunctionWithKeywords(THPVariable_quantized_max_pool3d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"max_pool3d", castPyCFunctionWithKeywords(THPVariable_max_pool3d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"median", castPyCFunctionWithKeywords(THPVariable_median), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_mps_convolution", castPyCFunctionWithKeywords(THPVariable__mps_convolution), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"miopen_convolution", castPyCFunctionWithKeywords(THPVariable_miopen_convolution), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"miopen_convolution_transpose", castPyCFunctionWithKeywords(THPVariable_miopen_convolution_transpose), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"miopen_convolution_add_relu", castPyCFunctionWithKeywords(THPVariable_miopen_convolution_add_relu), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"mm", castPyCFunctionWithKeywords(THPVariable_mm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_int_mm", castPyCFunctionWithKeywords(THPVariable__int_mm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_convert_weight_to_int4pack_for_cpu", castPyCFunctionWithKeywords(THPVariable__convert_weight_to_int4pack_for_cpu), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"multiply", castPyCFunctionWithKeywords(THPVariable_multiply), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"mvlgamma", castPyCFunctionWithKeywords(THPVariable_mvlgamma), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"narrow_copy", castPyCFunctionWithKeywords(THPVariable_narrow_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_native_batch_norm_legit_no_training", castPyCFunctionWithKeywords(THPVariable__native_batch_norm_legit_no_training), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"batch_norm_elemt", castPyCFunctionWithKeywords(THPVariable_batch_norm_elemt), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"batch_norm_gather_stats_with_counts", castPyCFunctionWithKeywords(THPVariable_batch_norm_gather_stats_with_counts), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"batch_norm_backward_elemt", castPyCFunctionWithKeywords(THPVariable_batch_norm_backward_elemt), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"ones_like", castPyCFunctionWithKeywords(THPVariable_ones_like), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"pairwise_distance", castPyCFunctionWithKeywords(THPVariable_pairwise_distance), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cdist", castPyCFunctionWithKeywords(THPVariable_cdist), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cosine_similarity", castPyCFunctionWithKeywords(THPVariable_cosine_similarity), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"movedim", castPyCFunctionWithKeywords(THPVariable_movedim), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"adjoint", castPyCFunctionWithKeywords(THPVariable_adjoint), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"channel_shuffle", castPyCFunctionWithKeywords(THPVariable_channel_shuffle), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"pinverse", castPyCFunctionWithKeywords(THPVariable_pinverse), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"poisson_nll_loss", castPyCFunctionWithKeywords(THPVariable_poisson_nll_loss), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"deg2rad", castPyCFunctionWithKeywords(THPVariable_deg2rad), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"deg2rad_", castPyCFunctionWithKeywords(THPVariable_deg2rad_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"randn_like", castPyCFunctionWithKeywords(THPVariable_randn_like), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"repeat_interleave", castPyCFunctionWithKeywords(THPVariable_repeat_interleave), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_prelu_kernel", castPyCFunctionWithKeywords(THPVariable__prelu_kernel), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"hardshrink", castPyCFunctionWithKeywords(THPVariable_hardshrink), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"rsqrt", castPyCFunctionWithKeywords(THPVariable_rsqrt), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"rsqrt_", castPyCFunctionWithKeywords(THPVariable_rsqrt_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"celu", castPyCFunctionWithKeywords(THPVariable_celu), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"celu_", castPyCFunctionWithKeywords(THPVariable_celu_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sinc", castPyCFunctionWithKeywords(THPVariable_sinc), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sinc_", castPyCFunctionWithKeywords(THPVariable_sinc_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sinh", castPyCFunctionWithKeywords(THPVariable_sinh), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sinh_", castPyCFunctionWithKeywords(THPVariable_sinh_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"detach", castPyCFunctionWithKeywords(THPVariable_detach), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"detach_", castPyCFunctionWithKeywords(THPVariable_detach_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_softmax", castPyCFunctionWithKeywords(THPVariable__softmax), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_softmax_backward_data", castPyCFunctionWithKeywords(THPVariable__softmax_backward_data), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"unsafe_split", castPyCFunctionWithKeywords(THPVariable_unsafe_split), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"split", castPyCFunctionWithKeywords(THPVariable_split), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"hsplit", castPyCFunctionWithKeywords(THPVariable_hsplit), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"squeeze", castPyCFunctionWithKeywords(THPVariable_squeeze), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"stack", castPyCFunctionWithKeywords(THPVariable_stack), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_stack", castPyCFunctionWithKeywords(THPVariable__stack), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"istft", castPyCFunctionWithKeywords(THPVariable_istft), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"nansum", castPyCFunctionWithKeywords(THPVariable_nansum), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sqrt", castPyCFunctionWithKeywords(THPVariable_sqrt), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sqrt_", castPyCFunctionWithKeywords(THPVariable_sqrt_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"square", castPyCFunctionWithKeywords(THPVariable_square), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"square_", castPyCFunctionWithKeywords(THPVariable_square_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"t", castPyCFunctionWithKeywords(THPVariable_t), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"tanh", castPyCFunctionWithKeywords(THPVariable_tanh), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"tanh_", castPyCFunctionWithKeywords(THPVariable_tanh_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"tensordot", castPyCFunctionWithKeywords(THPVariable_tensordot), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"tile", castPyCFunctionWithKeywords(THPVariable_tile), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_mkldnn_transpose", castPyCFunctionWithKeywords(THPVariable__mkldnn_transpose), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_mkldnn_transpose_", castPyCFunctionWithKeywords(THPVariable__mkldnn_transpose_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"rot90", castPyCFunctionWithKeywords(THPVariable_rot90), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_from_padded", castPyCFunctionWithKeywords(THPVariable__nested_from_padded), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_get_values", castPyCFunctionWithKeywords(THPVariable__nested_get_values), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_get_offsets", castPyCFunctionWithKeywords(THPVariable__nested_get_offsets), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_get_jagged_dummy", castPyCFunctionWithKeywords(THPVariable__nested_get_jagged_dummy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"triplet_margin_loss", castPyCFunctionWithKeywords(THPVariable_triplet_margin_loss), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"fix", castPyCFunctionWithKeywords(THPVariable_fix), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"fix_", castPyCFunctionWithKeywords(THPVariable_fix_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_has_compatible_shallow_copy_type", castPyCFunctionWithKeywords(THPVariable__has_compatible_shallow_copy_type), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_unique", castPyCFunctionWithKeywords(THPVariable__unique), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"vander", castPyCFunctionWithKeywords(THPVariable_vander), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"var_mean", castPyCFunctionWithKeywords(THPVariable_var_mean), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_weight_norm", castPyCFunctionWithKeywords(THPVariable__weight_norm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_standard_gamma_grad", castPyCFunctionWithKeywords(THPVariable__standard_gamma_grad), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"poisson", castPyCFunctionWithKeywords(THPVariable_poisson), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"native_norm", castPyCFunctionWithKeywords(THPVariable_native_norm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_softmax_backward_data", castPyCFunctionWithKeywords(THPVariable__sparse_softmax_backward_data), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"norm", castPyCFunctionWithKeywords(THPVariable_norm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"frexp", castPyCFunctionWithKeywords(THPVariable_frexp), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"positive", castPyCFunctionWithKeywords(THPVariable_positive), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"resize_as_", castPyCFunctionWithKeywords(THPVariable_resize_as_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"zero_", castPyCFunctionWithKeywords(THPVariable_zero_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sub", castPyCFunctionWithKeywords(THPVariable_sub), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"subtract", castPyCFunctionWithKeywords(THPVariable_subtract), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"rsub", castPyCFunctionWithKeywords(THPVariable_rsub), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_validate_sparse_csc_tensor_args", castPyCFunctionWithKeywords(THPVariable__validate_sparse_csc_tensor_args), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_coalesce", castPyCFunctionWithKeywords(THPVariable__coalesce), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"unbind", castPyCFunctionWithKeywords(THPVariable_unbind), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"quantize_per_tensor_dynamic", castPyCFunctionWithKeywords(THPVariable_quantize_per_tensor_dynamic), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"quantize_per_tensor", castPyCFunctionWithKeywords(THPVariable_quantize_per_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"quantize_per_channel", castPyCFunctionWithKeywords(THPVariable_quantize_per_channel), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"q_zero_point", castPyCFunctionWithKeywords(THPVariable_q_zero_point), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"int_repr", castPyCFunctionWithKeywords(THPVariable_int_repr), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_make_per_tensor_quantized_tensor", castPyCFunctionWithKeywords(THPVariable__make_per_tensor_quantized_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_make_per_channel_quantized_tensor", castPyCFunctionWithKeywords(THPVariable__make_per_channel_quantized_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"fake_quantize_per_channel_affine", castPyCFunctionWithKeywords(THPVariable_fake_quantize_per_channel_affine), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_fused_moving_avg_obs_fq_helper", castPyCFunctionWithKeywords(THPVariable__fused_moving_avg_obs_fq_helper), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_choose_qparams_per_tensor", castPyCFunctionWithKeywords(THPVariable__choose_qparams_per_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"meshgrid", castPyCFunctionWithKeywords(THPVariable_meshgrid), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"promote_types", castPyCFunctionWithKeywords(THPVariable_promote_types), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"gru", castPyCFunctionWithKeywords(THPVariable_gru), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"rnn_tanh_cell", castPyCFunctionWithKeywords(THPVariable_rnn_tanh_cell), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"quantized_gru_cell", castPyCFunctionWithKeywords(THPVariable_quantized_gru_cell), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"masked_scatter", castPyCFunctionWithKeywords(THPVariable_masked_scatter), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"put", castPyCFunctionWithKeywords(THPVariable_put), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"__or__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable___or__>), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"__xor__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable___xor__>), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"__lshift__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable___lshift__>), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"bitwise_right_shift", castPyCFunctionWithKeywords(THPVariable_bitwise_right_shift), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"tril", castPyCFunctionWithKeywords(THPVariable_tril), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"tril_indices", castPyCFunctionWithKeywords(THPVariable_tril_indices), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"not_equal", castPyCFunctionWithKeywords(THPVariable_not_equal), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"greater_equal", castPyCFunctionWithKeywords(THPVariable_greater_equal), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"less", castPyCFunctionWithKeywords(THPVariable_less), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"index_select", castPyCFunctionWithKeywords(THPVariable_index_select), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"gather", castPyCFunctionWithKeywords(THPVariable_gather), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"addcmul", castPyCFunctionWithKeywords(THPVariable_addcmul), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"triangular_solve", castPyCFunctionWithKeywords(THPVariable_triangular_solve), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"swapaxes", castPyCFunctionWithKeywords(THPVariable_swapaxes), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cholesky_inverse", castPyCFunctionWithKeywords(THPVariable_cholesky_inverse), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"geqrf", castPyCFunctionWithKeywords(THPVariable_geqrf), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"orgqr", castPyCFunctionWithKeywords(THPVariable_orgqr), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"ormqr", castPyCFunctionWithKeywords(THPVariable_ormqr), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"lu_solve", castPyCFunctionWithKeywords(THPVariable_lu_solve), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"lgamma", castPyCFunctionWithKeywords(THPVariable_lgamma), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"digamma", castPyCFunctionWithKeywords(THPVariable_digamma), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"erfinv", castPyCFunctionWithKeywords(THPVariable_erfinv), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sign", castPyCFunctionWithKeywords(THPVariable_sign), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"dist", castPyCFunctionWithKeywords(THPVariable_dist), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"arctan2", castPyCFunctionWithKeywords(THPVariable_arctan2), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"histc", castPyCFunctionWithKeywords(THPVariable_histc), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"histogram", castPyCFunctionWithKeywords(THPVariable_histogram), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_histogramdd_bin_edges", castPyCFunctionWithKeywords(THPVariable__histogramdd_bin_edges), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"histogramdd", castPyCFunctionWithKeywords(THPVariable_histogramdd), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"fmax", castPyCFunctionWithKeywords(THPVariable_fmax), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"maximum", castPyCFunctionWithKeywords(THPVariable_maximum), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"minimum", castPyCFunctionWithKeywords(THPVariable_minimum), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"nanquantile", castPyCFunctionWithKeywords(THPVariable_nanquantile), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"msort", castPyCFunctionWithKeywords(THPVariable_msort), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"pow", castPyCFunctionWithKeywords(THPVariable_pow), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"float_power", castPyCFunctionWithKeywords(THPVariable_float_power), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_add", castPyCFunctionWithKeywords(THPVariable__foreach_add), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_add_", castPyCFunctionWithKeywords(THPVariable__foreach_add_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_sub", castPyCFunctionWithKeywords(THPVariable__foreach_sub), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_sub_", castPyCFunctionWithKeywords(THPVariable__foreach_sub_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_clamp_max", castPyCFunctionWithKeywords(THPVariable__foreach_clamp_max), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_clamp_max_", castPyCFunctionWithKeywords(THPVariable__foreach_clamp_max_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_acos", castPyCFunctionWithKeywords(THPVariable__foreach_acos), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_acos_", castPyCFunctionWithKeywords(THPVariable__foreach_acos_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_asin", castPyCFunctionWithKeywords(THPVariable__foreach_asin), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_asin_", castPyCFunctionWithKeywords(THPVariable__foreach_asin_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_cos", castPyCFunctionWithKeywords(THPVariable__foreach_cos), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_cos_", castPyCFunctionWithKeywords(THPVariable__foreach_cos_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_cosh", castPyCFunctionWithKeywords(THPVariable__foreach_cosh), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_cosh_", castPyCFunctionWithKeywords(THPVariable__foreach_cosh_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_exp", castPyCFunctionWithKeywords(THPVariable__foreach_exp), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_exp_", castPyCFunctionWithKeywords(THPVariable__foreach_exp_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_floor", castPyCFunctionWithKeywords(THPVariable__foreach_floor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_floor_", castPyCFunctionWithKeywords(THPVariable__foreach_floor_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_log1p", castPyCFunctionWithKeywords(THPVariable__foreach_log1p), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_log1p_", castPyCFunctionWithKeywords(THPVariable__foreach_log1p_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_log2", castPyCFunctionWithKeywords(THPVariable__foreach_log2), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_log2_", castPyCFunctionWithKeywords(THPVariable__foreach_log2_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_neg", castPyCFunctionWithKeywords(THPVariable__foreach_neg), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_neg_", castPyCFunctionWithKeywords(THPVariable__foreach_neg_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_norm", castPyCFunctionWithKeywords(THPVariable__foreach_norm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_sigmoid", castPyCFunctionWithKeywords(THPVariable__foreach_sigmoid), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_sigmoid_", castPyCFunctionWithKeywords(THPVariable__foreach_sigmoid_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_sign", castPyCFunctionWithKeywords(THPVariable__foreach_sign), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_sign_", castPyCFunctionWithKeywords(THPVariable__foreach_sign_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_sqrt", castPyCFunctionWithKeywords(THPVariable__foreach_sqrt), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_sqrt_", castPyCFunctionWithKeywords(THPVariable__foreach_sqrt_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_tanh", castPyCFunctionWithKeywords(THPVariable__foreach_tanh), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_tanh_", castPyCFunctionWithKeywords(THPVariable__foreach_tanh_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_convert_indices_from_csr_to_coo", castPyCFunctionWithKeywords(THPVariable__convert_indices_from_csr_to_coo), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"mkldnn_adaptive_avg_pool2d", castPyCFunctionWithKeywords(THPVariable_mkldnn_adaptive_avg_pool2d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"column_stack", castPyCFunctionWithKeywords(THPVariable_column_stack), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"isinf", castPyCFunctionWithKeywords(THPVariable_isinf), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"isneginf", castPyCFunctionWithKeywords(THPVariable_isneginf), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_linalg_slogdet", castPyCFunctionWithKeywords(THPVariable__linalg_slogdet), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"slogdet", castPyCFunctionWithKeywords(THPVariable_slogdet), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"inverse", castPyCFunctionWithKeywords(THPVariable_inverse), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"inner", castPyCFunctionWithKeywords(THPVariable_inner), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_linalg_solve_ex", castPyCFunctionWithKeywords(THPVariable__linalg_solve_ex), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_test_parallel_materialize", castPyCFunctionWithKeywords(THPVariable__test_parallel_materialize), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_test_autograd_multiple_dispatch", castPyCFunctionWithKeywords(THPVariable__test_autograd_multiple_dispatch), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_test_autograd_multiple_dispatch_view_copy", castPyCFunctionWithKeywords(THPVariable__test_autograd_multiple_dispatch_view_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"segment_reduce", castPyCFunctionWithKeywords(THPVariable_segment_reduce), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_conj_copy", castPyCFunctionWithKeywords(THPVariable__conj_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_broadcast_to_copy", castPyCFunctionWithKeywords(THPVariable__sparse_broadcast_to_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"select_copy", castPyCFunctionWithKeywords(THPVariable_select_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"squeeze_copy", castPyCFunctionWithKeywords(THPVariable_squeeze_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"transpose_copy", castPyCFunctionWithKeywords(THPVariable_transpose_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"values_copy", castPyCFunctionWithKeywords(THPVariable_values_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"unfold_copy", castPyCFunctionWithKeywords(THPVariable_unfold_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_fused_sdp_choice", castPyCFunctionWithKeywords(THPVariable__fused_sdp_choice), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_scaled_dot_product_efficient_attention", castPyCFunctionWithKeywords(THPVariable__scaled_dot_product_efficient_attention), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_scaled_dot_product_cudnn_attention", castPyCFunctionWithKeywords(THPVariable__scaled_dot_product_cudnn_attention), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_fill_mem_eff_dropout_mask_", castPyCFunctionWithKeywords(THPVariable__fill_mem_eff_dropout_mask_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foobar", castPyCFunctionWithKeywords(THPVariable__foobar), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_fused_adagrad_", castPyCFunctionWithKeywords(THPVariable__fused_adagrad_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_propagate_xla_data", castPyCFunctionWithKeywords(THPVariable__propagate_xla_data), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
};

void gatherTorchFunctions_1(std::vector<PyMethodDef> &torch_functions) {
  constexpr size_t num_functions = sizeof(torch_functions_shard) / sizeof(torch_functions_shard[0]);
  torch_functions.insert(
    torch_functions.end(),
    torch_functions_shard,
    torch_functions_shard + num_functions);
}

// generated methods start here

// _cast_Char
static PyObject * THPVariable__cast_Char(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Char(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cast_Char(Tensor self, bool non_blocking=False) -> Tensor
  
  auto dispatch__cast_Char = [](const at::Tensor & self, bool non_blocking) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cast_Char(self, non_blocking);
  };
  return wrap(dispatch__cast_Char(_r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cast_Short
static PyObject * THPVariable__cast_Short(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Short(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cast_Short(Tensor self, bool non_blocking=False) -> Tensor
  
  auto dispatch__cast_Short = [](const at::Tensor & self, bool non_blocking) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cast_Short(self, non_blocking);
  };
  return wrap(dispatch__cast_Short(_r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cast_Half
static PyObject * THPVariable__cast_Half(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Half(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cast_Half(Tensor self, bool non_blocking=False) -> Tensor
  
  auto dispatch__cast_Half = [](const at::Tensor & self, bool non_blocking) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cast_Half(self, non_blocking);
  };
  return wrap(dispatch__cast_Half(_r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _make_dual
static PyObject * THPVariable__make_dual(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_make_dual(Tensor primal, Tensor tangent, int64_t level)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_make_dual(Tensor(a) primal, Tensor tangent, int level) -> Tensor(a)
  
  auto dispatch__make_dual = [](const at::Tensor & primal, const at::Tensor & tangent, int64_t level) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_make_dual(primal, tangent, level);
  };
  return wrap(dispatch__make_dual(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _unpack_dual
static PyObject * THPVariable__unpack_dual(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get__unpack_dual_structseq();
  static PythonArgParser parser({
    "_unpack_dual(Tensor dual, int64_t level)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_unpack_dual(Tensor(a) dual, int level) -> (Tensor(a) primal, Tensor tangent)
  
  auto dispatch__unpack_dual = [](const at::Tensor & dual, int64_t level) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_unpack_dual(dual, level);
  };
  return wrap(NamedTuple, dispatch__unpack_dual(_r.tensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _functional_assert_scalar
static PyObject * THPVariable__functional_assert_scalar(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_functional_assert_scalar(Scalar self, c10::string_view assert_msg, Tensor dep_token)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_functional_assert_scalar(Scalar self, str assert_msg, Tensor dep_token) -> Tensor
  
  auto dispatch__functional_assert_scalar = [](const at::Scalar & self, c10::string_view assert_msg, const at::Tensor & dep_token) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_functional_assert_scalar(self, assert_msg, dep_token);
  };
  return wrap(dispatch__functional_assert_scalar(_r.scalar(0), _r.stringView(1), _r.tensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _functional_assert_async
static PyObject * THPVariable__functional_assert_async(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_functional_assert_async(Tensor input, c10::string_view assert_msg, Tensor dep_token)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_functional_assert_async.msg(Tensor self, str assert_msg, Tensor dep_token) -> Tensor
  
  auto dispatch__functional_assert_async = [](const at::Tensor & self, c10::string_view assert_msg, const at::Tensor & dep_token) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_functional_assert_async(self, assert_msg, dep_token);
  };
  return wrap(dispatch__functional_assert_async(_r.tensor(0), _r.stringView(1), _r.tensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sym_constrain_range_for_size
static PyObject * THPVariable_sym_constrain_range_for_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sym_constrain_range_for_size(Scalar size, *, int64_t? min=None, int64_t? max=None)",
  }, /*traceable=*/false);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::sym_constrain_range_for_size(Scalar size, *, int? min=None, int? max=None) -> ()
  
  auto dispatch_sym_constrain_range_for_size = [](const at::Scalar & size, ::std::optional<int64_t> min, ::std::optional<int64_t> max) -> void {
    pybind11::gil_scoped_release no_gil;
    at::sym_constrain_range_for_size(size, min, max);
  };
  dispatch_sym_constrain_range_for_size(_r.scalar(0), _r.toInt64Optional(1), _r.toInt64Optional(2));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _functional_sym_constrain_range
static PyObject * THPVariable__functional_sym_constrain_range(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_functional_sym_constrain_range(Scalar size, int64_t? min, int64_t? max, Tensor dep_token)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_functional_sym_constrain_range(Scalar size, int? min, int? max, Tensor dep_token) -> Tensor
  
  auto dispatch__functional_sym_constrain_range = [](const at::Scalar & size, ::std::optional<int64_t> min, ::std::optional<int64_t> max, const at::Tensor & dep_token) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_functional_sym_constrain_range(size, min, max, dep_token);
  };
  return wrap(dispatch__functional_sym_constrain_range(_r.scalar(0), _r.toInt64Optional(1), _r.toInt64Optional(2), _r.tensor(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _cudnn_ctc_loss
static PyObject * THPVariable__cudnn_ctc_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cudnn_ctc_loss(Tensor log_probs, Tensor targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity)",
    "_cudnn_ctc_loss(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int64_t blank, bool deterministic, bool zero_infinity)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank, bool deterministic, bool zero_infinity) -> (Tensor, Tensor)
      
      auto dispatch__cudnn_ctc_loss = [](const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, deterministic, zero_infinity);
      };
      return wrap(dispatch__cudnn_ctc_loss(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.intlist(3), _r.toInt64(4), _r.toBool(5), _r.toBool(6)));
    }
    case 1: {
      // aten::_cudnn_ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank, bool deterministic, bool zero_infinity) -> (Tensor, Tensor)
      
      auto dispatch__cudnn_ctc_loss = [](const at::Tensor & log_probs, const at::Tensor & targets, const at::Tensor & input_lengths, const at::Tensor & target_lengths, int64_t blank, bool deterministic, bool zero_infinity) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, deterministic, zero_infinity);
      };
      return wrap(dispatch__cudnn_ctc_loss(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.toInt64(4), _r.toBool(5), _r.toBool(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cudnn_init_dropout_state
static PyObject * THPVariable__cudnn_init_dropout_state(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cudnn_init_dropout_state(float dropout, bool train, int dropout_seed, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(3))
      .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
      .layout(_r.layoutOptional(4))
      .requires_grad(_r.toBool(7))
      .pinned_memory(_r.toBool(6));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch__cudnn_init_dropout_state = [](double dropout, bool train, int64_t dropout_seed, at::TensorOptions options) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::_cudnn_init_dropout_state(dropout, train, dropout_seed, options);
  };
  return wrap(dispatch__cudnn_init_dropout_state(_r.toDouble(0), _r.toBool(1), _r.toInt64(2), options));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _debug_has_internal_overlap
static PyObject * THPVariable__debug_has_internal_overlap(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_debug_has_internal_overlap(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_debug_has_internal_overlap(Tensor self) -> int
  
  auto dispatch__debug_has_internal_overlap = [](const at::Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return at::_debug_has_internal_overlap(self);
  };
  return wrap(dispatch__debug_has_internal_overlap(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _masked_scale
static PyObject * THPVariable__masked_scale(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_masked_scale(Tensor input, Tensor mask, double scale)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_masked_scale(Tensor self, Tensor mask, float scale) -> Tensor
  
  auto dispatch__masked_scale = [](const at::Tensor & self, const at::Tensor & mask, double scale) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_masked_scale(self, mask, scale);
  };
  return wrap(dispatch__masked_scale(_r.tensor(0), _r.tensor(1), _r.toDouble(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sobol_engine_initialize_state_
static PyObject * THPVariable__sobol_engine_initialize_state_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sobol_engine_initialize_state_(Tensor input, int64_t dimension)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_sobol_engine_initialize_state_(Tensor(a!) self, int dimension) -> Tensor(a!)
  
  auto dispatch__sobol_engine_initialize_state_ = [](at::Tensor self, int64_t dimension) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sobol_engine_initialize_state_(self, dimension);
  };
  return wrap(dispatch__sobol_engine_initialize_state_(_r.tensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// feature_alpha_dropout
static PyObject * THPVariable_feature_alpha_dropout(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "feature_alpha_dropout(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor
  
  auto dispatch_feature_alpha_dropout = [](const at::Tensor & input, double p, bool train) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::feature_alpha_dropout(input, p, train);
  };
  return wrap(dispatch_feature_alpha_dropout(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// feature_alpha_dropout_
static PyObject * THPVariable_feature_alpha_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "feature_alpha_dropout_(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::feature_alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
  
  auto dispatch_feature_alpha_dropout_ = [](at::Tensor self, double p, bool train) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::feature_alpha_dropout_(self, p, train);
  };
  return wrap(dispatch_feature_alpha_dropout_(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// abs
static PyObject * THPVariable_abs(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "abs(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::abs(Tensor self) -> Tensor
    
    auto dispatch_abs = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.abs();
    };
    return wrap(dispatch_abs(_r.tensor(0)));
  } else {
    // aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_abs_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::abs_out(out, self);
    };
    return wrap(dispatch_abs_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// abs_
static PyObject * THPVariable_abs_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "abs_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::abs_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_abs_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.abs_();
  };
  return wrap(dispatch_abs_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// absolute
static PyObject * THPVariable_absolute(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "absolute(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::absolute(Tensor self) -> Tensor
    
    auto dispatch_absolute = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.absolute();
    };
    return wrap(dispatch_absolute(_r.tensor(0)));
  } else {
    // aten::absolute.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_absolute_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::absolute_out(out, self);
    };
    return wrap(dispatch_absolute_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// view_as_real
static PyObject * THPVariable_view_as_real(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "view_as_real(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::view_as_real(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_view_as_real = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::view_as_real(self);
  };
  return wrap(dispatch_view_as_real(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// real
static PyObject * THPVariable_real(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "real(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::real(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_real = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::real(self);
  };
  return wrap(dispatch_real(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _conj_physical
static PyObject * THPVariable__conj_physical(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_conj_physical(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_conj_physical(Tensor self) -> Tensor
  
  auto dispatch__conj_physical = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._conj_physical();
  };
  return wrap(dispatch__conj_physical(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// resolve_conj
static PyObject * THPVariable_resolve_conj(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "resolve_conj(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::resolve_conj(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_resolve_conj = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.resolve_conj();
  };
  return wrap(dispatch_resolve_conj(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _neg_view
static PyObject * THPVariable__neg_view(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_neg_view(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_neg_view(Tensor(a) self) -> Tensor(a)
  
  auto dispatch__neg_view = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._neg_view();
  };
  return wrap(dispatch__neg_view(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// acos
static PyObject * THPVariable_acos(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "acos(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::acos(Tensor self) -> Tensor
    
    auto dispatch_acos = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.acos();
    };
    return wrap(dispatch_acos(_r.tensor(0)));
  } else {
    // aten::acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_acos_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::acos_out(out, self);
    };
    return wrap(dispatch_acos_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// acos_
static PyObject * THPVariable_acos_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "acos_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::acos_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_acos_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.acos_();
  };
  return wrap(dispatch_acos_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// arccos
static PyObject * THPVariable_arccos(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "arccos(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::arccos(Tensor self) -> Tensor
    
    auto dispatch_arccos = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.arccos();
    };
    return wrap(dispatch_arccos(_r.tensor(0)));
  } else {
    // aten::arccos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_arccos_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::arccos_out(out, self);
    };
    return wrap(dispatch_arccos_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// arccos_
static PyObject * THPVariable_arccos_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "arccos_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::arccos_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_arccos_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arccos_();
  };
  return wrap(dispatch_arccos_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// adaptive_avg_pool1d
static PyObject * THPVariable_adaptive_avg_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_avg_pool1d(Tensor input, IntArrayRef[1] output_size)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor
  
  auto dispatch_adaptive_avg_pool1d = [](const at::Tensor & self, at::IntArrayRef output_size) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::adaptive_avg_pool1d(self, output_size);
  };
  return wrap(dispatch_adaptive_avg_pool1d(_r.tensor(0), _r.intlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addmv
static PyObject * THPVariable_addmv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addmv(Scalar beta, Tensor input, Scalar alpha, Tensor mat, Tensor vec, *, Tensor out=None)|deprecated",
    "addmv(Scalar beta, Tensor input, Tensor mat, Tensor vec, *, Tensor out=None)|deprecated",
    "addmv(Tensor input, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(5)) {
        // [deprecated] aten::addmv(Scalar beta, Tensor self, Scalar alpha, Tensor mat, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addmv = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & mat, const at::Tensor & vec) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addmv(mat, vec, beta, alpha);
        };
        return wrap(dispatch_addmv(_r.tensor(5), _r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4)));
      } else {
        // [deprecated] aten::addmv(Scalar beta, Tensor self, Scalar alpha, Tensor mat, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addmv_out = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & mat, const at::Tensor & vec) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addmv_out(out, self, mat, vec, beta, alpha);
        };
        return wrap(dispatch_addmv_out(_r.tensor(5), _r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // [deprecated] aten::addmv(Scalar beta, Tensor self, Tensor mat, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addmv = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addmv(mat, vec, beta, 1);
        };
        return wrap(dispatch_addmv(_r.tensor(4), _r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
      } else {
        // [deprecated] aten::addmv(Scalar beta, Tensor self, Tensor mat, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addmv_out = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addmv_out(out, self, mat, vec, beta, 1);
        };
        return wrap(dispatch_addmv_out(_r.tensor(4), _r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
      }
    }
    case 2: {
      if (_r.isNone(5)) {
        // aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        
        auto dispatch_addmv = [](const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addmv(mat, vec, beta, alpha);
        };
        return wrap(dispatch_addmv(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      } else {
        // aten::addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addmv_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addmv_out(out, self, mat, vec, beta, alpha);
        };
        return wrap(dispatch_addmv_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      }
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
  static PythonArgParser parser({
    "addmv_(Scalar beta, Tensor input, Scalar alpha, Tensor mat, Tensor vec)|deprecated",
    "addmv_(Scalar beta, Tensor input, Tensor mat, Tensor vec)|deprecated",
    "addmv_(Tensor input, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::addmv_(Scalar beta, Tensor(a!) self, Scalar alpha, Tensor mat, Tensor vec) -> Tensor(a!)
      
      auto dispatch_addmv_ = [](const at::Scalar & beta, at::Tensor self, const at::Scalar & alpha, const at::Tensor & mat, const at::Tensor & vec) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmv_(mat, vec, beta, alpha);
      };
      return wrap(dispatch_addmv_(_r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4)));
    }
    case 1: {
      // [deprecated] aten::addmv_(Scalar beta, Tensor(a!) self, Tensor mat, Tensor vec) -> Tensor(a!)
      
      auto dispatch_addmv_ = [](const at::Scalar & beta, at::Tensor self, const at::Tensor & mat, const at::Tensor & vec) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmv_(mat, vec, beta, 1);
      };
      return wrap(dispatch_addmv_(_r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
    }
    case 2: {
      // aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
      
      auto dispatch_addmv_ = [](at::Tensor self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.addmv_(mat, vec, beta, alpha);
      };
      return wrap(dispatch_addmv_(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _is_all_true
static PyObject * THPVariable__is_all_true(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_is_all_true(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_is_all_true(Tensor self) -> Tensor
  
  auto dispatch__is_all_true = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._is_all_true();
  };
  return wrap(dispatch__is_all_true(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _test_check_tensor
static PyObject * THPVariable__test_check_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_test_check_tensor(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_test_check_tensor(Tensor self) -> Tensor
  
  auto dispatch__test_check_tensor = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_test_check_tensor(self);
  };
  return wrap(dispatch__test_check_tensor(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// argmin
static PyObject * THPVariable_argmin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "argmin(Tensor input, int64_t? dim=None, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
    
    auto dispatch_argmin = [](const at::Tensor & self, ::std::optional<int64_t> dim, bool keepdim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.argmin(dim, keepdim);
    };
    return wrap(dispatch_argmin(_r.tensor(0), _r.toInt64Optional(1), _r.toBool(2)));
  } else {
    // aten::argmin.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_argmin_out = [](at::Tensor out, const at::Tensor & self, ::std::optional<int64_t> dim, bool keepdim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::argmin_out(out, self, dim, keepdim);
    };
    return wrap(dispatch_argmin_out(_r.tensor(3), _r.tensor(0), _r.toInt64Optional(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// asinh
static PyObject * THPVariable_asinh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "asinh(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::asinh(Tensor self) -> Tensor
    
    auto dispatch_asinh = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.asinh();
    };
    return wrap(dispatch_asinh(_r.tensor(0)));
  } else {
    // aten::asinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_asinh_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::asinh_out(out, self);
    };
    return wrap(dispatch_asinh_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// asinh_
static PyObject * THPVariable_asinh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "asinh_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::asinh_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_asinh_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.asinh_();
  };
  return wrap(dispatch_asinh_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// arcsinh
static PyObject * THPVariable_arcsinh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "arcsinh(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::arcsinh(Tensor self) -> Tensor
    
    auto dispatch_arcsinh = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.arcsinh();
    };
    return wrap(dispatch_arcsinh(_r.tensor(0)));
  } else {
    // aten::arcsinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_arcsinh_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::arcsinh_out(out, self);
    };
    return wrap(dispatch_arcsinh_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// arcsinh_
static PyObject * THPVariable_arcsinh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "arcsinh_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::arcsinh_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_arcsinh_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arcsinh_();
  };
  return wrap(dispatch_arcsinh_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// asin
static PyObject * THPVariable_asin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "asin(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::asin(Tensor self) -> Tensor
    
    auto dispatch_asin = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.asin();
    };
    return wrap(dispatch_asin(_r.tensor(0)));
  } else {
    // aten::asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_asin_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::asin_out(out, self);
    };
    return wrap(dispatch_asin_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// asin_
static PyObject * THPVariable_asin_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "asin_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::asin_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_asin_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.asin_();
  };
  return wrap(dispatch_asin_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// arctan
static PyObject * THPVariable_arctan(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "arctan(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::arctan(Tensor self) -> Tensor
    
    auto dispatch_arctan = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.arctan();
    };
    return wrap(dispatch_arctan(_r.tensor(0)));
  } else {
    // aten::arctan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_arctan_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::arctan_out(out, self);
    };
    return wrap(dispatch_arctan_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// arctan_
static PyObject * THPVariable_arctan_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "arctan_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::arctan_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_arctan_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arctan_();
  };
  return wrap(dispatch_arctan_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// atleast_2d
static PyObject * THPVariable_atleast_2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atleast_2d(Tensor input)",
    "atleast_2d(TensorList tensors)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::atleast_2d(Tensor self) -> Tensor
      
      auto dispatch_atleast_2d = [](const at::Tensor & self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::atleast_2d(self);
      };
      return wrap(dispatch_atleast_2d(_r.tensor(0)));
    }
    case 1: {
      // aten::atleast_2d.Sequence(Tensor[] tensors) -> Tensor[]
      
      auto dispatch_atleast_2d = [](at::TensorList tensors) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::atleast_2d(tensors);
      };
      return wrap(dispatch_atleast_2d(_r.tensorlist(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// atleast_3d
static PyObject * THPVariable_atleast_3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atleast_3d(Tensor input)",
    "atleast_3d(TensorList tensors)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::atleast_3d(Tensor self) -> Tensor
      
      auto dispatch_atleast_3d = [](const at::Tensor & self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::atleast_3d(self);
      };
      return wrap(dispatch_atleast_3d(_r.tensor(0)));
    }
    case 1: {
      // aten::atleast_3d.Sequence(Tensor[] tensors) -> Tensor[]
      
      auto dispatch_atleast_3d = [](at::TensorList tensors) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::atleast_3d(tensors);
      };
      return wrap(dispatch_atleast_3d(_r.tensorlist(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// quantized_batch_norm
static PyObject * THPVariable_quantized_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantized_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, double eps, double output_scale, int64_t output_zero_point)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::quantized_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor
  
  auto dispatch_quantized_batch_norm = [](const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & var, double eps, double output_scale, int64_t output_zero_point) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::quantized_batch_norm(input, weight, bias, mean, var, eps, output_scale, output_zero_point);
  };
  return wrap(dispatch_quantized_batch_norm(_r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2), _r.tensor(3), _r.tensor(4), _r.toDouble(5), _r.toDouble(6), _r.toInt64(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _batch_norm_impl_index
static PyObject * THPVariable__batch_norm_impl_index(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, double momentum, double eps, bool cudnn_enabled)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor, Tensor, Tensor, Tensor, int)
  
  auto dispatch__batch_norm_impl_index = [](const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t> {
    pybind11::gil_scoped_release no_gil;
    return at::_batch_norm_impl_index(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
  };
  return wrap(dispatch__batch_norm_impl_index(_r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2), _r.optionalTensor(3), _r.optionalTensor(4), _r.toBool(5), _r.toDouble(6), _r.toDouble(7), _r.toBool(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// bincount
static PyObject * THPVariable_bincount(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bincount(Tensor input, Tensor? weights=None, int64_t minlength=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor
  
  auto dispatch_bincount = [](const at::Tensor & self, const ::std::optional<at::Tensor> & weights, int64_t minlength) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.bincount(weights, minlength);
  };
  return wrap(dispatch_bincount(_r.tensor(0), _r.optionalTensor(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// bitwise_not
static PyObject * THPVariable_bitwise_not(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bitwise_not(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::bitwise_not(Tensor self) -> Tensor
    
    auto dispatch_bitwise_not = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.bitwise_not();
    };
    return wrap(dispatch_bitwise_not(_r.tensor(0)));
  } else {
    // aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_bitwise_not_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::bitwise_not_out(out, self);
    };
    return wrap(dispatch_bitwise_not_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logical_or
static PyObject * THPVariable_logical_or(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logical_or(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::logical_or(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_logical_or = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.logical_or(other);
    };
    return wrap(dispatch_logical_or(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::logical_or.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_logical_or_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::logical_or_out(out, self, other);
    };
    return wrap(dispatch_logical_or_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unsafe_chunk
static PyObject * THPVariable_unsafe_chunk(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unsafe_chunk(Tensor input, int64_t chunks, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::unsafe_chunk(Tensor self, int chunks, int dim=0) -> Tensor[]
  
  auto dispatch_unsafe_chunk = [](const at::Tensor & self, int64_t chunks, int64_t dim) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.unsafe_chunk(chunks, dim);
  };
  return wrap(dispatch_unsafe_chunk(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// clamp
static PyObject * THPVariable_clamp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clamp(Tensor input, Tensor? min=None, Tensor? max=None, *, Tensor out=None)",
    "clamp(Tensor input, Scalar? min=None, Scalar? max=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
        
        auto dispatch_clamp = [](const at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.clamp(min, max);
        };
        return wrap(dispatch_clamp(_r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2)));
      } else {
        // aten::clamp.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_clamp_out = [](at::Tensor out, const at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::clamp_out(out, self, min, max);
        };
        return wrap(dispatch_clamp_out(_r.tensor(3), _r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
        
        auto dispatch_clamp = [](const at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.clamp(min, max);
        };
        return wrap(dispatch_clamp(_r.tensor(0), _r.scalarOptional(1), _r.scalarOptional(2)));
      } else {
        // aten::clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_clamp_out = [](at::Tensor out, const at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::clamp_out(out, self, min, max);
        };
        return wrap(dispatch_clamp_out(_r.tensor(3), _r.tensor(0), _r.scalarOptional(1), _r.scalarOptional(2)));
      }
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
  static PythonArgParser parser({
    "clamp_(Tensor input, Tensor? min=None, Tensor? max=None)",
    "clamp_(Tensor input, Scalar? min=None, Scalar? max=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::clamp_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)
      
      auto dispatch_clamp_ = [](at::Tensor self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clamp_(min, max);
      };
      return wrap(dispatch_clamp_(_r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2)));
    }
    case 1: {
      // aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)
      
      auto dispatch_clamp_ = [](at::Tensor self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clamp_(min, max);
      };
      return wrap(dispatch_clamp_(_r.tensor(0), _r.scalarOptional(1), _r.scalarOptional(2)));
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
  static PythonArgParser parser({
    "clamp_max(Tensor input, Tensor max, *, Tensor out=None)",
    "clamp_max(Tensor input, Scalar max, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::clamp_max.Tensor(Tensor self, Tensor max) -> Tensor
        
        auto dispatch_clamp_max = [](const at::Tensor & self, const at::Tensor & max) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.clamp_max(max);
        };
        return wrap(dispatch_clamp_max(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::clamp_max.Tensor_out(Tensor self, Tensor max, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_clamp_max_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & max) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::clamp_max_out(out, self, max);
        };
        return wrap(dispatch_clamp_max_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::clamp_max(Tensor self, Scalar max) -> Tensor
        
        auto dispatch_clamp_max = [](const at::Tensor & self, const at::Scalar & max) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.clamp_max(max);
        };
        return wrap(dispatch_clamp_max(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::clamp_max.out(Tensor self, Scalar max, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_clamp_max_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & max) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::clamp_max_out(out, self, max);
        };
        return wrap(dispatch_clamp_max_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
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
  static PythonArgParser parser({
    "clamp_max_(Tensor input, Tensor max)",
    "clamp_max_(Tensor input, Scalar max)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::clamp_max_.Tensor(Tensor(a!) self, Tensor max) -> Tensor(a!)
      
      auto dispatch_clamp_max_ = [](at::Tensor self, const at::Tensor & max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clamp_max_(max);
      };
      return wrap(dispatch_clamp_max_(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)
      
      auto dispatch_clamp_max_ = [](at::Tensor self, const at::Scalar & max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clamp_max_(max);
      };
      return wrap(dispatch_clamp_max_(_r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// constant_pad_nd
static PyObject * THPVariable_constant_pad_nd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "constant_pad_nd(Tensor input, SymIntArrayRef pad, Scalar value=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0) -> Tensor
  
  auto dispatch_constant_pad_nd = [](const at::Tensor & self, c10::SymIntArrayRef pad, const at::Scalar & value) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::constant_pad_nd_symint(self, pad, value);
  };
  return wrap(dispatch_constant_pad_nd(_r.tensor(0), _r.symintlist(1), _r.scalar(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// convolution
static PyObject * THPVariable_convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "convolution(Tensor input, Tensor weight, Tensor? bias, SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef dilation, bool transposed, SymIntArrayRef output_padding, SymInt groups)",
  }, /*traceable=*/false);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor
  
  auto dispatch_convolution = [](const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::convolution_symint(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
  };
  return wrap(dispatch_convolution(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.symintlist(4), _r.symintlist(5), _r.toBool(6), _r.symintlist(7), _r.toSymInt(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// conv2d
static PyObject * THPVariable_conv2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymIntArrayRef[2] stride=1, SymIntArrayRef[2] padding=0, SymIntArrayRef[2] dilation=1, SymInt groups=1)",
    "conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymIntArrayRef[2] stride=1, c10::string_view padding=\"valid\", SymIntArrayRef[2] dilation=1, SymInt groups=1)",
  }, /*traceable=*/false);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1) -> Tensor
      
      auto dispatch_conv2d = [](const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, c10::SymInt groups) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::conv2d_symint(input, weight, bias, stride, padding, dilation, groups);
      };
      return wrap(dispatch_conv2d(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.symintlist(4), _r.symintlist(5), _r.toSymInt(6)));
    }
    case 1: {
      // aten::conv2d.padding(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, str padding="valid", SymInt[2] dilation=1, SymInt groups=1) -> Tensor
      
      auto dispatch_conv2d = [](const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::string_view padding, c10::SymIntArrayRef dilation, c10::SymInt groups) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::conv2d_symint(input, weight, bias, stride, padding, dilation, groups);
      };
      return wrap(dispatch_conv2d(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.stringView(4), _r.symintlist(5), _r.toSymInt(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// conv_tbc
static PyObject * THPVariable_conv_tbc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv_tbc(Tensor input, Tensor weight, Tensor bias, int64_t pad=0)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad=0) -> Tensor
  
  auto dispatch_conv_tbc = [](const at::Tensor & self, const at::Tensor & weight, const at::Tensor & bias, int64_t pad) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::conv_tbc(self, weight, bias, pad);
  };
  return wrap(dispatch_conv_tbc(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// conv_transpose1d
static PyObject * THPVariable_conv_transpose1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, SymIntArrayRef[1] stride=1, SymIntArrayRef[1] padding=0, SymIntArrayRef[1] output_padding=0, SymInt groups=1, SymIntArrayRef[1] dilation=1)",
  }, /*traceable=*/false);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[1] stride=1, SymInt[1] padding=0, SymInt[1] output_padding=0, SymInt groups=1, SymInt[1] dilation=1) -> Tensor
  
  auto dispatch_conv_transpose1d = [](const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymInt groups, c10::SymIntArrayRef dilation) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::conv_transpose1d_symint(input, weight, bias, stride, padding, output_padding, groups, dilation);
  };
  return wrap(dispatch_conv_transpose1d(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.symintlist(4), _r.symintlist(5), _r.toSymInt(6), _r.symintlist(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// conv_transpose3d
static PyObject * THPVariable_conv_transpose3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv_transpose3d(Tensor input, Tensor weight, Tensor? bias=None, SymIntArrayRef[3] stride=1, SymIntArrayRef[3] padding=0, SymIntArrayRef[3] output_padding=0, SymInt groups=1, SymIntArrayRef[3] dilation=1)",
  }, /*traceable=*/false);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::conv_transpose3d.input(Tensor input, Tensor weight, Tensor? bias=None, SymInt[3] stride=1, SymInt[3] padding=0, SymInt[3] output_padding=0, SymInt groups=1, SymInt[3] dilation=1) -> Tensor
  
  auto dispatch_conv_transpose3d = [](const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymInt groups, c10::SymIntArrayRef dilation) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::conv_transpose3d_symint(input, weight, bias, stride, padding, output_padding, groups, dilation);
  };
  return wrap(dispatch_conv_transpose3d(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.symintlist(4), _r.symintlist(5), _r.toSymInt(6), _r.symintlist(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cosine_embedding_loss
static PyObject * THPVariable_cosine_embedding_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, double margin=0.0, int64_t reduction=at::Reduction::Mean)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor
  
  auto dispatch_cosine_embedding_loss = [](const at::Tensor & input1, const at::Tensor & input2, const at::Tensor & target, double margin, int64_t reduction) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::cosine_embedding_loss(input1, input2, target, margin, reduction);
  };
  return wrap(dispatch_cosine_embedding_loss(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toDouble(3), _r.toInt64(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cudnn_affine_grid_generator
static PyObject * THPVariable_cudnn_affine_grid_generator(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_affine_grid_generator(Tensor theta, int64_t N, int64_t C, int64_t H, int64_t W)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::cudnn_affine_grid_generator(Tensor theta, int N, int C, int H, int W) -> Tensor grid
  
  auto dispatch_cudnn_affine_grid_generator = [](const at::Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::cudnn_affine_grid_generator(theta, N, C, H, W);
  };
  return wrap(dispatch_cudnn_affine_grid_generator(_r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toInt64(3), _r.toInt64(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cudnn_batch_norm
static PyObject * THPVariable_cudnn_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, double exponential_average_factor, double epsilon)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor, Tensor)
  
  auto dispatch_cudnn_batch_norm = [](const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double exponential_average_factor, double epsilon) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::cudnn_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
  };
  return wrap(dispatch_cudnn_batch_norm(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.optionalTensor(3), _r.optionalTensor(4), _r.toBool(5), _r.toDouble(6), _r.toDouble(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _mps_convolution_transpose
static PyObject * THPVariable__mps_convolution_transpose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_mps_convolution_transpose(Tensor input, Tensor weight, SymIntArrayRef padding, SymIntArrayRef output_padding, SymIntArrayRef stride, SymIntArrayRef dilation, SymInt groups)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_mps_convolution_transpose(Tensor self, Tensor weight, SymInt[] padding, SymInt[] output_padding, SymInt[] stride, SymInt[] dilation, SymInt groups) -> Tensor
  
  auto dispatch__mps_convolution_transpose = [](const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_mps_convolution_transpose_symint(self, weight, padding, output_padding, stride, dilation, groups);
  };
  return wrap(dispatch__mps_convolution_transpose(_r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.symintlist(3), _r.symintlist(4), _r.symintlist(5), _r.toSymInt(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cudnn_convolution_add_relu
static PyObject * THPVariable_cudnn_convolution_add_relu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_convolution_add_relu(Tensor input, Tensor weight, Tensor z, Scalar? alpha, Tensor? bias, SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef dilation, SymInt groups)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::cudnn_convolution_add_relu(Tensor self, Tensor weight, Tensor z, Scalar? alpha, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, SymInt groups) -> Tensor
  
  auto dispatch_cudnn_convolution_add_relu = [](const at::Tensor & self, const at::Tensor & weight, const at::Tensor & z, const ::std::optional<at::Scalar> & alpha, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, c10::SymInt groups) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::cudnn_convolution_add_relu_symint(self, weight, z, alpha, bias, stride, padding, dilation, groups);
  };
  return wrap(dispatch_cudnn_convolution_add_relu(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalarOptional(3), _r.optionalTensor(4), _r.symintlist(5), _r.symintlist(6), _r.symintlist(7), _r.toSymInt(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cudnn_grid_sampler
static PyObject * THPVariable_cudnn_grid_sampler(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_grid_sampler(Tensor input, Tensor grid)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::cudnn_grid_sampler(Tensor self, Tensor grid) -> Tensor output
  
  auto dispatch_cudnn_grid_sampler = [](const at::Tensor & self, const at::Tensor & grid) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::cudnn_grid_sampler(self, grid);
  };
  return wrap(dispatch_cudnn_grid_sampler(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cummax_helper
static PyObject * THPVariable__cummax_helper(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cummax_helper(Tensor input, Tensor values, Tensor indices, int64_t dim)",
  }, /*traceable=*/false);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
  
  auto dispatch__cummax_helper = [](const at::Tensor & self, at::Tensor values, at::Tensor indices, int64_t dim) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_cummax_helper(self, values, indices, dim);
  };
  dispatch__cummax_helper(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// cumprod
static PyObject * THPVariable_cumprod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cumprod(Tensor input, int64_t dim, *, ScalarType? dtype=None, Tensor out=None)",
    "cumprod(Tensor input, Dimname dim, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_cumprod = [](const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.cumprod(dim, dtype);
        };
        return wrap(dispatch_cumprod(_r.tensor(0), _r.toInt64(1), _r.scalartypeOptional(2)));
      } else {
        // aten::cumprod.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_cumprod_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::cumprod_out(out, self, dim, dtype);
        };
        return wrap(dispatch_cumprod_out(_r.tensor(3), _r.tensor(0), _r.toInt64(1), _r.scalartypeOptional(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::cumprod.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_cumprod = [](const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.cumprod(dim, dtype);
        };
        return wrap(dispatch_cumprod(_r.tensor(0), _r.dimname(1), _r.scalartypeOptional(2)));
      } else {
        // aten::cumprod.dimname_out(Tensor self, Dimname dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_cumprod_out = [](at::Tensor out, const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::cumprod_out(out, self, dim, dtype);
        };
        return wrap(dispatch_cumprod_out(_r.tensor(3), _r.tensor(0), _r.dimname(1), _r.scalartypeOptional(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// gradient
static PyObject * THPVariable_gradient(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gradient(Tensor input, *, IntArrayRef dim, int64_t edge_order=1)",
    "gradient(Tensor input, *, Scalar spacing, IntArrayRef dim, int64_t edge_order=1)",
    "gradient(Tensor input, *, Scalar? spacing=None, int64_t? dim=None, int64_t edge_order=1)",
    "gradient(Tensor input, *, ScalarList spacing, int64_t? dim=None, int64_t edge_order=1)",
    "gradient(Tensor input, *, ScalarList spacing, IntArrayRef dim, int64_t edge_order=1)",
    "gradient(Tensor input, *, TensorList spacing, int64_t? dim=None, int64_t edge_order=1)",
    "gradient(Tensor input, *, TensorList spacing, IntArrayRef dim, int64_t edge_order=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::gradient.array(Tensor self, *, int[] dim, int edge_order=1) -> Tensor[]
      
      auto dispatch_gradient = [](const at::Tensor & self, at::IntArrayRef dim, int64_t edge_order) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::gradient(self, dim, edge_order);
      };
      return wrap(dispatch_gradient(_r.tensor(0), _r.intlist(1), _r.toInt64(2)));
    }
    case 1: {
      // aten::gradient.scalararray(Tensor self, *, Scalar spacing, int[] dim, int edge_order=1) -> Tensor[]
      
      auto dispatch_gradient = [](const at::Tensor & self, const at::Scalar & spacing, at::IntArrayRef dim, int64_t edge_order) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::gradient(self, spacing, dim, edge_order);
      };
      return wrap(dispatch_gradient(_r.tensor(0), _r.scalar(1), _r.intlist(2), _r.toInt64(3)));
    }
    case 2: {
      // aten::gradient.scalarint(Tensor self, *, Scalar? spacing=None, int? dim=None, int edge_order=1) -> Tensor[]
      
      auto dispatch_gradient = [](const at::Tensor & self, const ::std::optional<at::Scalar> & spacing, ::std::optional<int64_t> dim, int64_t edge_order) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::gradient(self, spacing, dim, edge_order);
      };
      return wrap(dispatch_gradient(_r.tensor(0), _r.scalarOptional(1), _r.toInt64Optional(2), _r.toInt64(3)));
    }
    case 3: {
      // aten::gradient.scalarrayint(Tensor self, *, Scalar[] spacing, int? dim=None, int edge_order=1) -> Tensor[]
      
      auto dispatch_gradient = [](const at::Tensor & self, at::ArrayRef<at::Scalar> spacing, ::std::optional<int64_t> dim, int64_t edge_order) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::gradient(self, spacing, dim, edge_order);
      };
      return wrap(dispatch_gradient(_r.tensor(0), _r.scalarlist(1), _r.toInt64Optional(2), _r.toInt64(3)));
    }
    case 4: {
      // aten::gradient.scalarrayarray(Tensor self, *, Scalar[] spacing, int[] dim, int edge_order=1) -> Tensor[]
      
      auto dispatch_gradient = [](const at::Tensor & self, at::ArrayRef<at::Scalar> spacing, at::IntArrayRef dim, int64_t edge_order) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::gradient(self, spacing, dim, edge_order);
      };
      return wrap(dispatch_gradient(_r.tensor(0), _r.scalarlist(1), _r.intlist(2), _r.toInt64(3)));
    }
    case 5: {
      // aten::gradient.tensorarrayint(Tensor self, *, Tensor[] spacing, int? dim=None, int edge_order=1) -> Tensor[]
      
      auto dispatch_gradient = [](const at::Tensor & self, at::TensorList spacing, ::std::optional<int64_t> dim, int64_t edge_order) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::gradient(self, spacing, dim, edge_order);
      };
      return wrap(dispatch_gradient(_r.tensor(0), _r.tensorlist(1), _r.toInt64Optional(2), _r.toInt64(3)));
    }
    case 6: {
      // aten::gradient.tensorarray(Tensor self, *, Tensor[] spacing, int[] dim, int edge_order=1) -> Tensor[]
      
      auto dispatch_gradient = [](const at::Tensor & self, at::TensorList spacing, at::IntArrayRef dim, int64_t edge_order) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::gradient(self, spacing, dim, edge_order);
      };
      return wrap(dispatch_gradient(_r.tensor(0), _r.tensorlist(1), _r.intlist(2), _r.toInt64(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// div
static PyObject * THPVariable_div(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "div(Tensor input, Tensor other, *, Tensor out=None)",
    "div(Tensor input, Tensor other, *, c10::string_view? rounding_mode, Tensor out=None)",
    "div(Tensor input, Scalar other, *, c10::string_view? rounding_mode)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::div.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_div = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.div(other);
        };
        return wrap(dispatch_div(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_div_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::div_out(out, self, other);
        };
        return wrap(dispatch_div_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
        
        auto dispatch_div = [](const at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.div(other, rounding_mode);
        };
        return wrap(dispatch_div(_r.tensor(0), _r.tensor(1), _r.stringViewOptional(2)));
      } else {
        // aten::div.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_div_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::div_out(out, self, other, rounding_mode);
        };
        return wrap(dispatch_div_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.stringViewOptional(2)));
      }
    }
    case 2: {
      // aten::div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor
      
      auto dispatch_div = [](const at::Tensor & self, const at::Scalar & other, ::std::optional<c10::string_view> rounding_mode) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.div(other, rounding_mode);
      };
      return wrap(dispatch_div(_r.tensor(0), _r.scalar(1), _r.stringViewOptional(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// true_divide
static PyObject * THPVariable_true_divide(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "true_divide(Tensor input, Tensor other, *, Tensor out=None)",
    "true_divide(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::true_divide.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_true_divide = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.true_divide(other);
        };
        return wrap(dispatch_true_divide(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::true_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_true_divide_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::true_divide_out(out, self, other);
        };
        return wrap(dispatch_true_divide_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      // aten::true_divide.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_true_divide = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.true_divide(other);
      };
      return wrap(dispatch_true_divide(_r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _embedding_bag_forward_only
static PyObject * THPVariable__embedding_bag_forward_only(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_embedding_bag_forward_only(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int64_t mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int64_t padding_idx=-1)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_embedding_bag_forward_only(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
  
  auto dispatch__embedding_bag_forward_only = [](const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const ::std::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_embedding_bag_forward_only(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
  };
  return wrap(dispatch__embedding_bag_forward_only(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toBool(3), _r.toInt64(4), _r.toBool(5), _r.optionalTensor(6), _r.toBool(7), _r.toInt64(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// row_stack
static PyObject * THPVariable_row_stack(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "row_stack(TensorList tensors, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::row_stack(Tensor[] tensors) -> Tensor
    
    auto dispatch_row_stack = [](at::TensorList tensors) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::row_stack(tensors);
    };
    return wrap(dispatch_row_stack(_r.tensorlist(0)));
  } else {
    // aten::row_stack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_row_stack_out = [](at::Tensor out, at::TensorList tensors) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::row_stack_out(out, tensors);
    };
    return wrap(dispatch_row_stack_out(_r.tensor(1), _r.tensorlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _empty_per_channel_affine_quantized
static PyObject * THPVariable__empty_per_channel_affine_quantized(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_empty_per_channel_affine_quantized(SymIntArrayRef size, *, Tensor scales, Tensor zero_points, int64_t axis, MemoryFormat? memory_format=c10::MemoryFormat::Contiguous, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<10> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_empty_per_channel_affine_quantized(SymInt[] size, *, Tensor scales, Tensor zero_points, int axis, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=contiguous_format) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(5))
      .device(_r.deviceWithDefault(7, torch::tensors::get_default_device()))
      .layout(_r.layoutOptional(6))
      .requires_grad(_r.toBool(9))
      .pinned_memory(_r.toBool(8));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch__empty_per_channel_affine_quantized = [](c10::SymIntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, at::TensorOptions options, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::_empty_per_channel_affine_quantized_symint(size, scales, zero_points, axis, options, memory_format);
  };
  return wrap(dispatch__empty_per_channel_affine_quantized(_r.symintlist(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3), options, _r.memoryformat(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _resize_output_
static PyObject * THPVariable__resize_output_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_resize_output_(Tensor input, SymIntArrayRef size, Device device)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_resize_output_(Tensor(a!) self, SymInt[] size, Device device) -> Tensor(a!)
  
  auto dispatch__resize_output_ = [](const at::Tensor & self, c10::SymIntArrayRef size, at::Device device) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_resize_output__symint(self, size, device);
  };
  return wrap(dispatch__resize_output_(_r.tensor(0), _r.symintlist(1), _r.device(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// empty_strided
static PyObject * THPVariable_empty_strided(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "empty_strided(SymIntArrayRef size, SymIntArrayRef stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(2))
      .device(_r.deviceWithDefault(4, torch::tensors::get_default_device()))
      .layout(_r.layoutOptional(3))
      .requires_grad(_r.toBool(6))
      .pinned_memory(_r.toBool(5));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch_empty_strided = [](c10::SymIntArrayRef size, c10::SymIntArrayRef stride, at::TensorOptions options) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::empty_strided_symint(size, stride, options);
  };
  return wrap(dispatch_empty_strided(_r.symintlist(0), _r.symintlist(1), options));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// flatten
static PyObject * THPVariable_flatten(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "flatten(Tensor input, int64_t start_dim, int64_t end_dim, Dimname out_dim)",
    "flatten(Tensor input, int64_t start_dim=0, int64_t end_dim=-1)",
    "flatten(Tensor input, Dimname start_dim, Dimname end_dim, Dimname out_dim)",
    "flatten(Tensor input, DimnameList dims, Dimname out_dim)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::flatten.named_out_dim(Tensor(a) self, int start_dim, int end_dim, Dimname out_dim) -> Tensor(a)
      
      auto dispatch_flatten = [](const at::Tensor & self, int64_t start_dim, int64_t end_dim, at::Dimname out_dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.flatten(start_dim, end_dim, out_dim);
      };
      return wrap(dispatch_flatten(_r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.dimname(3)));
    }
    case 1: {
      // aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)
      
      auto dispatch_flatten = [](const at::Tensor & self, int64_t start_dim, int64_t end_dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.flatten(start_dim, end_dim);
      };
      return wrap(dispatch_flatten(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
    }
    case 2: {
      // aten::flatten.using_names(Tensor(a) self, Dimname start_dim, Dimname end_dim, Dimname out_dim) -> Tensor(a)
      
      auto dispatch_flatten = [](const at::Tensor & self, at::Dimname start_dim, at::Dimname end_dim, at::Dimname out_dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.flatten(start_dim, end_dim, out_dim);
      };
      return wrap(dispatch_flatten(_r.tensor(0), _r.dimname(1), _r.dimname(2), _r.dimname(3)));
    }
    case 3: {
      // aten::flatten.DimnameList(Tensor(a) self, Dimname[] dims, Dimname out_dim) -> Tensor(a)
      
      auto dispatch_flatten = [](const at::Tensor & self, at::DimnameList dims, at::Dimname out_dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.flatten(dims, out_dim);
      };
      return wrap(dispatch_flatten(_r.tensor(0), _r.dimnamelist(1), _r.dimname(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// fill
static PyObject * THPVariable_fill(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fill(Tensor input, Tensor value)",
    "fill(Tensor input, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::fill.Tensor(Tensor self, Tensor value) -> Tensor
      
      auto dispatch_fill = [](const at::Tensor & self, const at::Tensor & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::fill(self, value);
      };
      return wrap(dispatch_fill(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::fill.Scalar(Tensor self, Scalar value) -> Tensor
      
      auto dispatch_fill = [](const at::Tensor & self, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::fill(self, value);
      };
      return wrap(dispatch_fill(_r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// fill_
static PyObject * THPVariable_fill_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fill_(Tensor input, Tensor value)",
    "fill_(Tensor input, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)
      
      auto dispatch_fill_ = [](at::Tensor self, const at::Tensor & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.fill_(value);
      };
      return wrap(dispatch_fill_(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
      
      auto dispatch_fill_ = [](at::Tensor self, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.fill_(value);
      };
      return wrap(dispatch_fill_(_r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// grid_sampler_2d
static PyObject * THPVariable_grid_sampler_2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "grid_sampler_2d(Tensor input, Tensor grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
  
  auto dispatch_grid_sampler_2d = [](const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners);
  };
  return wrap(dispatch_grid_sampler_2d(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toInt64(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// hann_window
static PyObject * THPVariable_hann_window(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hann_window(int64_t window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "hann_window(int64_t window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::hann_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(1))
          .device(_r.deviceWithDefault(3, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(2))
          .requires_grad(_r.toBool(5))
          .pinned_memory(_r.toBool(4));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_hann_window = [](int64_t window_length, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::hann_window(window_length, options);
      };
      return wrap(dispatch_hann_window(_r.toInt64(0), options));
    }
    case 1: {
      // aten::hann_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(2))
          .device(_r.deviceWithDefault(4, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(3))
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_hann_window = [](int64_t window_length, bool periodic, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::hann_window(window_length, periodic, options);
      };
      return wrap(dispatch_hann_window(_r.toInt64(0), _r.toBool(1), options));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// kaiser_window
static PyObject * THPVariable_kaiser_window(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "kaiser_window(int64_t window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "kaiser_window(int64_t window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "kaiser_window(int64_t window_length, bool periodic, double beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::kaiser_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(1))
          .device(_r.deviceWithDefault(3, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(2))
          .requires_grad(_r.toBool(5))
          .pinned_memory(_r.toBool(4));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_kaiser_window = [](int64_t window_length, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::kaiser_window(window_length, options);
      };
      return wrap(dispatch_kaiser_window(_r.toInt64(0), options));
    }
    case 1: {
      // aten::kaiser_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(2))
          .device(_r.deviceWithDefault(4, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(3))
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_kaiser_window = [](int64_t window_length, bool periodic, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::kaiser_window(window_length, periodic, options);
      };
      return wrap(dispatch_kaiser_window(_r.toInt64(0), _r.toBool(1), options));
    }
    case 2: {
      // aten::kaiser_window.beta(int window_length, bool periodic, float beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(3))
          .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(4))
          .requires_grad(_r.toBool(7))
          .pinned_memory(_r.toBool(6));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_kaiser_window = [](int64_t window_length, bool periodic, double beta, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::kaiser_window(window_length, periodic, beta, options);
      };
      return wrap(dispatch_kaiser_window(_r.toInt64(0), _r.toBool(1), _r.toDouble(2), options));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _fft_r2c
static PyObject * THPVariable__fft_r2c(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_fft_r2c(Tensor input, IntArrayRef dim, int64_t normalization, bool onesided, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::_fft_r2c(Tensor self, int[] dim, int normalization, bool onesided) -> Tensor
    
    auto dispatch__fft_r2c = [](const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool onesided) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_fft_r2c(self, dim, normalization, onesided);
    };
    return wrap(dispatch__fft_r2c(_r.tensor(0), _r.intlist(1), _r.toInt64(2), _r.toBool(3)));
  } else {
    // aten::_fft_r2c.out(Tensor self, int[] dim, int normalization, bool onesided, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__fft_r2c_out = [](at::Tensor out, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool onesided) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_fft_r2c_out(out, self, dim, normalization, onesided);
    };
    return wrap(dispatch__fft_r2c_out(_r.tensor(4), _r.tensor(0), _r.intlist(1), _r.toInt64(2), _r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _validate_compressed_sparse_indices
static PyObject * THPVariable__validate_compressed_sparse_indices(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_validate_compressed_sparse_indices(bool is_crow, Tensor compressed_idx, Tensor plain_idx, int64_t cdim, int64_t dim, int64_t nnz)",
  }, /*traceable=*/false);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_validate_compressed_sparse_indices(bool is_crow, Tensor compressed_idx, Tensor plain_idx, int cdim, int dim, int nnz) -> ()
  
  auto dispatch__validate_compressed_sparse_indices = [](bool is_crow, const at::Tensor & compressed_idx, const at::Tensor & plain_idx, int64_t cdim, int64_t dim, int64_t nnz) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_validate_compressed_sparse_indices(is_crow, compressed_idx, plain_idx, cdim, dim, nnz);
  };
  dispatch__validate_compressed_sparse_indices(_r.toBool(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3), _r.toInt64(4), _r.toInt64(5));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cufft_get_plan_cache_size
static PyObject * THPVariable__cufft_get_plan_cache_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cufft_get_plan_cache_size(DeviceIndex device_index)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cufft_get_plan_cache_size(DeviceIndex device_index) -> int
  
  auto dispatch__cufft_get_plan_cache_size = [](at::DeviceIndex device_index) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return at::_cufft_get_plan_cache_size(device_index);
  };
  return wrap(dispatch__cufft_get_plan_cache_size(_r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _unsafe_masked_index
static PyObject * THPVariable__unsafe_masked_index(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_unsafe_masked_index(Tensor input, Tensor mask, c10::List<::std::optional<Tensor>> indices, Scalar fill)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_unsafe_masked_index(Tensor self, Tensor mask, Tensor?[] indices, Scalar fill) -> Tensor
  
  auto dispatch__unsafe_masked_index = [](const at::Tensor & self, const at::Tensor & mask, const c10::List<::std::optional<at::Tensor>> & indices, const at::Scalar & fill) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_unsafe_masked_index(self, mask, indices, fill);
  };
  return wrap(dispatch__unsafe_masked_index(_r.tensor(0), _r.tensor(1), _r.list_of_optional_tensors(2), _r.scalar(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// index_put_
static PyObject * THPVariable_index_put_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_put_(Tensor input, c10::List<::std::optional<Tensor>> indices, Tensor values, bool accumulate=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)
  
  auto dispatch_index_put_ = [](at::Tensor self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.index_put_(indices, values, accumulate);
  };
  return wrap(dispatch_index_put_(_r.tensor(0), _r.list_of_optional_tensors(1), _r.tensor(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// index_put
static PyObject * THPVariable_index_put(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_put(Tensor input, c10::List<::std::optional<Tensor>> indices, Tensor values, bool accumulate=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
  
  auto dispatch_index_put = [](const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.index_put(indices, values, accumulate);
  };
  return wrap(dispatch_index_put(_r.tensor(0), _r.list_of_optional_tensors(1), _r.tensor(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _unsafe_index_put
static PyObject * THPVariable__unsafe_index_put(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_unsafe_index_put(Tensor input, c10::List<::std::optional<Tensor>> indices, Tensor values, bool accumulate=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_unsafe_index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
  
  auto dispatch__unsafe_index_put = [](const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_unsafe_index_put(self, indices, values, accumulate);
  };
  return wrap(dispatch__unsafe_index_put(_r.tensor(0), _r.list_of_optional_tensors(1), _r.tensor(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_distributed
static PyObject * THPVariable_is_distributed(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_distributed(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::is_distributed(Tensor self) -> bool
  
  auto dispatch_is_distributed = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_distributed();
  };
  return wrap(dispatch_is_distributed(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_complex
static PyObject * THPVariable_is_complex(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_complex(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::is_complex(Tensor self) -> bool
  
  auto dispatch_is_complex = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_complex();
  };
  return wrap(dispatch_is_complex(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// isreal
static PyObject * THPVariable_isreal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "isreal(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::isreal(Tensor self) -> Tensor
  
  auto dispatch_isreal = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.isreal();
  };
  return wrap(dispatch_isreal(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// kron
static PyObject * THPVariable_kron(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "kron(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::kron(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_kron = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.kron(other);
    };
    return wrap(dispatch_kron(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::kron.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_kron_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::kron_out(out, self, other);
    };
    return wrap(dispatch_kron_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// native_layer_norm
static PyObject * THPVariable_native_layer_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "native_layer_norm(Tensor input, SymIntArrayRef normalized_shape, Tensor? weight, Tensor? bias, double eps)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)
  
  auto dispatch_native_layer_norm = [](const at::Tensor & input, c10::SymIntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::native_layer_norm_symint(input, normalized_shape, weight, bias, eps);
  };
  return wrap(dispatch_native_layer_norm(_r.tensor(0), _r.symintlist(1), _r.optionalTensor(2), _r.optionalTensor(3), _r.toDouble(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cslt_compress
static PyObject * THPVariable__cslt_compress(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cslt_compress(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cslt_compress(Tensor input) -> Tensor
  
  auto dispatch__cslt_compress = [](const at::Tensor & input) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cslt_compress(input);
  };
  return wrap(dispatch__cslt_compress(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sparse_semi_structured_tile
static PyObject * THPVariable__sparse_semi_structured_tile(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_semi_structured_tile(Tensor input, c10::string_view algorithm=\"\", bool use_cutlass=True)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_sparse_semi_structured_tile(Tensor input, str algorithm="", bool use_cutlass=True) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
  
  auto dispatch__sparse_semi_structured_tile = [](const at::Tensor & input, c10::string_view algorithm, bool use_cutlass) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_sparse_semi_structured_tile(input, algorithm, use_cutlass);
  };
  return wrap(dispatch__sparse_semi_structured_tile(_r.tensor(0), _r.stringView(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sparse_semi_structured_linear
static PyObject * THPVariable__sparse_semi_structured_linear(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_semi_structured_linear(Tensor input, Tensor weight, Tensor meta, *, Tensor? bias=None, c10::string_view? activation=None, ScalarType? out_dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_sparse_semi_structured_linear(Tensor input, Tensor weight, Tensor meta, *, Tensor? bias=None, str? activation=None, ScalarType? out_dtype=None) -> Tensor
  
  auto dispatch__sparse_semi_structured_linear = [](const at::Tensor & input, const at::Tensor & weight, const at::Tensor & meta, const ::std::optional<at::Tensor> & bias, ::std::optional<c10::string_view> activation, ::std::optional<at::ScalarType> out_dtype) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sparse_semi_structured_linear(input, weight, meta, bias, activation, out_dtype);
  };
  return wrap(dispatch__sparse_semi_structured_linear(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.optionalTensor(3), _r.stringViewOptional(4), _r.scalartypeOptional(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fbgemm_linear_fp16_weight
static PyObject * THPVariable_fbgemm_linear_fp16_weight(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fbgemm_linear_fp16_weight(Tensor input, Tensor packed_weight, Tensor bias)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::fbgemm_linear_fp16_weight(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor
  
  auto dispatch_fbgemm_linear_fp16_weight = [](const at::Tensor & input, const at::Tensor & packed_weight, const at::Tensor & bias) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::fbgemm_linear_fp16_weight(input, packed_weight, bias);
  };
  return wrap(dispatch_fbgemm_linear_fp16_weight(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ldexp
static PyObject * THPVariable_ldexp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ldexp(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::ldexp.Tensor(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_ldexp = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.ldexp(other);
    };
    return wrap(dispatch_ldexp(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::ldexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_ldexp_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::ldexp_out(out, self, other);
    };
    return wrap(dispatch_ldexp_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ldexp_
static PyObject * THPVariable_ldexp_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ldexp_(Tensor input, Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::ldexp_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  
  auto dispatch_ldexp_ = [](at::Tensor self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.ldexp_(other);
  };
  return wrap(dispatch_ldexp_(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// log10
static PyObject * THPVariable_log10(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log10(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::log10(Tensor self) -> Tensor
    
    auto dispatch_log10 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.log10();
    };
    return wrap(dispatch_log10(_r.tensor(0)));
  } else {
    // aten::log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_log10_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::log10_out(out, self);
    };
    return wrap(dispatch_log10_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// log10_
static PyObject * THPVariable_log10_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log10_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::log10_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_log10_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log10_();
  };
  return wrap(dispatch_log10_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logaddexp2
static PyObject * THPVariable_logaddexp2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logaddexp2(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::logaddexp2(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_logaddexp2 = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.logaddexp2(other);
    };
    return wrap(dispatch_logaddexp2(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::logaddexp2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_logaddexp2_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::logaddexp2_out(out, self, other);
    };
    return wrap(dispatch_logaddexp2_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _log_softmax_backward_data
static PyObject * THPVariable__log_softmax_backward_data(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_log_softmax_backward_data(Tensor grad_output, Tensor output, int64_t dim, ScalarType input_dtype, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor
    
    auto dispatch__log_softmax_backward_data = [](const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_log_softmax_backward_data(grad_output, output, dim, input_dtype);
    };
    return wrap(dispatch__log_softmax_backward_data(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.scalartype(3)));
  } else {
    // aten::_log_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__log_softmax_backward_data_out = [](at::Tensor out, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_log_softmax_backward_data_out(out, grad_output, output, dim, input_dtype);
    };
    return wrap(dispatch__log_softmax_backward_data_out(_r.tensor(4), _r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.scalartype(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// logcumsumexp
static PyObject * THPVariable_logcumsumexp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logcumsumexp(Tensor input, int64_t dim, *, Tensor out=None)",
    "logcumsumexp(Tensor input, Dimname dim, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::logcumsumexp(Tensor self, int dim) -> Tensor
        
        auto dispatch_logcumsumexp = [](const at::Tensor & self, int64_t dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.logcumsumexp(dim);
        };
        return wrap(dispatch_logcumsumexp(_r.tensor(0), _r.toInt64(1)));
      } else {
        // aten::logcumsumexp.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_logcumsumexp_out = [](at::Tensor out, const at::Tensor & self, int64_t dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::logcumsumexp_out(out, self, dim);
        };
        return wrap(dispatch_logcumsumexp_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::logcumsumexp.dimname(Tensor self, Dimname dim) -> Tensor
        
        auto dispatch_logcumsumexp = [](const at::Tensor & self, at::Dimname dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.logcumsumexp(dim);
        };
        return wrap(dispatch_logcumsumexp(_r.tensor(0), _r.dimname(1)));
      } else {
        // aten::logcumsumexp.dimname_out(Tensor self, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_logcumsumexp_out = [](at::Tensor out, const at::Tensor & self, at::Dimname dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::logcumsumexp_out(out, self, dim);
        };
        return wrap(dispatch_logcumsumexp_out(_r.tensor(2), _r.tensor(0), _r.dimname(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// margin_ranking_loss
static PyObject * THPVariable_margin_ranking_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, double margin=0.0, int64_t reduction=at::Reduction::Mean)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor
  
  auto dispatch_margin_ranking_loss = [](const at::Tensor & input1, const at::Tensor & input2, const at::Tensor & target, double margin, int64_t reduction) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::margin_ranking_loss(input1, input2, target, margin, reduction);
  };
  return wrap(dispatch_margin_ranking_loss(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toDouble(3), _r.toInt64(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// matrix_exp
static PyObject * THPVariable_matrix_exp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "matrix_exp(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::matrix_exp(Tensor self) -> Tensor
  
  auto dispatch_matrix_exp = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.matrix_exp();
  };
  return wrap(dispatch_matrix_exp(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// amax
static PyObject * THPVariable_amax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "amax(Tensor input, IntArrayRef[1] dim=None, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
    
    auto dispatch_amax = [](const at::Tensor & self, at::IntArrayRef dim, bool keepdim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.amax(dim, keepdim);
    };
    return wrap(dispatch_amax(_r.tensor(0), _r.intlist(1), _r.toBool(2)));
  } else {
    // aten::amax.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_amax_out = [](at::Tensor out, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::amax_out(out, self, dim, keepdim);
    };
    return wrap(dispatch_amax_out(_r.tensor(3), _r.tensor(0), _r.intlist(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// max_pool1d
static PyObject * THPVariable_max_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_pool1d(Tensor input, IntArrayRef[1] kernel_size, IntArrayRef[1] stride=None, IntArrayRef[1] padding=0, IntArrayRef[1] dilation=1, bool ceil_mode=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor
  
  auto dispatch_max_pool1d = [](const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);
  };
  return wrap(dispatch_max_pool1d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mkldnn_max_pool3d
static PyObject * THPVariable_mkldnn_max_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mkldnn_max_pool3d(Tensor input, IntArrayRef[3] kernel_size, IntArrayRef[3] stride=None, IntArrayRef[3] padding=0, IntArrayRef[3] dilation=1, bool ceil_mode=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::mkldnn_max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
  
  auto dispatch_mkldnn_max_pool3d = [](const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::mkldnn_max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode);
  };
  return wrap(dispatch_mkldnn_max_pool3d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// quantized_max_pool3d
static PyObject * THPVariable_quantized_max_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantized_max_pool3d(Tensor input, IntArrayRef[3] kernel_size, IntArrayRef[3] stride=None, IntArrayRef[3] padding=0, IntArrayRef[3] dilation=1, bool ceil_mode=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::quantized_max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
  
  auto dispatch_quantized_max_pool3d = [](const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::quantized_max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode);
  };
  return wrap(dispatch_quantized_max_pool3d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// max_pool3d
static PyObject * THPVariable_max_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_pool3d(Tensor input, IntArrayRef[3] kernel_size, IntArrayRef[3] stride=None, IntArrayRef[3] padding=0, IntArrayRef[3] dilation=1, bool ceil_mode=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
  
  auto dispatch_max_pool3d = [](const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode);
  };
  return wrap(dispatch_max_pool3d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// median
static PyObject * THPVariable_median(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_median_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_median_out_structseq();
  static PythonArgParser parser({
    "median(Tensor input)",
    "median(Tensor input, int64_t dim, bool keepdim=False, *, TensorList[2] out=None)",
    "median(Tensor input, Dimname dim, bool keepdim=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::median(Tensor self) -> Tensor
      
      auto dispatch_median = [](const at::Tensor & self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.median();
      };
      return wrap(dispatch_median(_r.tensor(0)));
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::median.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        
        auto dispatch_median = [](const at::Tensor & self, int64_t dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.median(dim, keepdim);
        };
        return wrap(NamedTuple, dispatch_median(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      } else {
        // aten::median.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_median_out = [](at::Tensor & values, at::Tensor & indices, const at::Tensor & self, int64_t dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::median_out(values, indices, self, dim, keepdim);
        };
        return wrap(NamedTuple1, dispatch_median_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      }
    }
    case 2: {
      if (_r.isNone(3)) {
        // aten::median.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        
        auto dispatch_median = [](const at::Tensor & self, at::Dimname dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.median(dim, keepdim);
        };
        return wrap(NamedTuple, dispatch_median(_r.tensor(0), _r.dimname(1), _r.toBool(2)));
      } else {
        // aten::median.names_dim_values(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_median_out = [](at::Tensor & values, at::Tensor & indices, const at::Tensor & self, at::Dimname dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::median_out(values, indices, self, dim, keepdim);
        };
        return wrap(NamedTuple1, dispatch_median_out(out[0], out[1], _r.tensor(0), _r.dimname(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _mps_convolution
static PyObject * THPVariable__mps_convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_mps_convolution(Tensor input, Tensor weight, Tensor? bias, SymIntArrayRef padding, SymIntArrayRef stride, SymIntArrayRef dilation, SymInt groups)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_mps_convolution(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups) -> Tensor
  
  auto dispatch__mps_convolution = [](const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_mps_convolution_symint(self, weight, bias, padding, stride, dilation, groups);
  };
  return wrap(dispatch__mps_convolution(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.symintlist(4), _r.symintlist(5), _r.toSymInt(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// miopen_convolution
static PyObject * THPVariable_miopen_convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "miopen_convolution(Tensor input, Tensor weight, Tensor? bias, SymIntArrayRef padding, SymIntArrayRef stride, SymIntArrayRef dilation, SymInt groups, bool benchmark, bool deterministic)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::miopen_convolution(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, bool deterministic) -> Tensor
  
  auto dispatch_miopen_convolution = [](const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, bool benchmark, bool deterministic) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::miopen_convolution_symint(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
  };
  return wrap(dispatch_miopen_convolution(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.symintlist(4), _r.symintlist(5), _r.toSymInt(6), _r.toBool(7), _r.toBool(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// miopen_convolution_transpose
static PyObject * THPVariable_miopen_convolution_transpose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "miopen_convolution_transpose(Tensor input, Tensor weight, Tensor? bias, SymIntArrayRef padding, SymIntArrayRef output_padding, SymIntArrayRef stride, SymIntArrayRef dilation, SymInt groups, bool benchmark, bool deterministic)",
  }, /*traceable=*/true);

  ParsedArgs<10> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::miopen_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, SymInt[] output_padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, bool deterministic) -> Tensor
  
  auto dispatch_miopen_convolution_transpose = [](const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, bool benchmark, bool deterministic) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::miopen_convolution_transpose_symint(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  };
  return wrap(dispatch_miopen_convolution_transpose(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.symintlist(4), _r.symintlist(5), _r.symintlist(6), _r.toSymInt(7), _r.toBool(8), _r.toBool(9)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// miopen_convolution_add_relu
static PyObject * THPVariable_miopen_convolution_add_relu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "miopen_convolution_add_relu(Tensor input, Tensor weight, Tensor z, Scalar? alpha, Tensor? bias, SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef dilation, SymInt groups)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::miopen_convolution_add_relu(Tensor self, Tensor weight, Tensor z, Scalar? alpha, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, SymInt groups) -> Tensor
  
  auto dispatch_miopen_convolution_add_relu = [](const at::Tensor & self, const at::Tensor & weight, const at::Tensor & z, const ::std::optional<at::Scalar> & alpha, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, c10::SymInt groups) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::miopen_convolution_add_relu_symint(self, weight, z, alpha, bias, stride, padding, dilation, groups);
  };
  return wrap(dispatch_miopen_convolution_add_relu(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalarOptional(3), _r.optionalTensor(4), _r.symintlist(5), _r.symintlist(6), _r.symintlist(7), _r.toSymInt(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mm
static PyObject * THPVariable_mm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mm(Tensor input, Tensor mat2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::mm(Tensor self, Tensor mat2) -> Tensor
    
    auto dispatch_mm = [](const at::Tensor & self, const at::Tensor & mat2) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.mm(mat2);
    };
    return wrap(dispatch_mm(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_mm_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & mat2) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::mm_out(out, self, mat2);
    };
    return wrap(dispatch_mm_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _int_mm
static PyObject * THPVariable__int_mm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_int_mm(Tensor input, Tensor mat2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::_int_mm(Tensor self, Tensor mat2) -> Tensor
    
    auto dispatch__int_mm = [](const at::Tensor & self, const at::Tensor & mat2) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_int_mm(self, mat2);
    };
    return wrap(dispatch__int_mm(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::_int_mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__int_mm_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & mat2) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_int_mm_out(out, self, mat2);
    };
    return wrap(dispatch__int_mm_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _convert_weight_to_int4pack_for_cpu
static PyObject * THPVariable__convert_weight_to_int4pack_for_cpu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_convert_weight_to_int4pack_for_cpu(Tensor input, int64_t innerKTiles)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_convert_weight_to_int4pack_for_cpu(Tensor self, int innerKTiles) -> Tensor
  
  auto dispatch__convert_weight_to_int4pack_for_cpu = [](const at::Tensor & self, int64_t innerKTiles) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_convert_weight_to_int4pack_for_cpu(self, innerKTiles);
  };
  return wrap(dispatch__convert_weight_to_int4pack_for_cpu(_r.tensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// multiply
static PyObject * THPVariable_multiply(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "multiply(Tensor input, Tensor other, *, Tensor out=None)",
    "multiply(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::multiply.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_multiply = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.multiply(other);
        };
        return wrap(dispatch_multiply(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::multiply.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_multiply_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::multiply_out(out, self, other);
        };
        return wrap(dispatch_multiply_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      // aten::multiply.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_multiply = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.multiply(other);
      };
      return wrap(dispatch_multiply(_r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mvlgamma
static PyObject * THPVariable_mvlgamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mvlgamma(Tensor input, int64_t p, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::mvlgamma(Tensor self, int p) -> Tensor
    
    auto dispatch_mvlgamma = [](const at::Tensor & self, int64_t p) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.mvlgamma(p);
    };
    return wrap(dispatch_mvlgamma(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::mvlgamma.out(Tensor self, int p, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_mvlgamma_out = [](at::Tensor out, const at::Tensor & self, int64_t p) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::mvlgamma_out(out, self, p);
    };
    return wrap(dispatch_mvlgamma_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// narrow_copy
static PyObject * THPVariable_narrow_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "narrow_copy(Tensor input, int64_t dim, SymInt start, SymInt length, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::narrow_copy(Tensor self, int dim, SymInt start, SymInt length) -> Tensor
    
    auto dispatch_narrow_copy = [](const at::Tensor & self, int64_t dim, c10::SymInt start, c10::SymInt length) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.narrow_copy_symint(dim, start, length);
    };
    return wrap(dispatch_narrow_copy(_r.tensor(0), _r.toInt64(1), _r.toSymInt(2), _r.toSymInt(3)));
  } else {
    // aten::narrow_copy.out(Tensor self, int dim, SymInt start, SymInt length, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_narrow_copy_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, c10::SymInt start, c10::SymInt length) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::narrow_copy_symint_out(out, self, dim, start, length);
    };
    return wrap(dispatch_narrow_copy_out(_r.tensor(4), _r.tensor(0), _r.toInt64(1), _r.toSymInt(2), _r.toSymInt(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _native_batch_norm_legit_no_training
static PyObject * THPVariable__native_batch_norm_legit_no_training(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_native_batch_norm_legit_no_training(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, double momentum, double eps)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_native_batch_norm_legit_no_training(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor)
  
  auto dispatch__native_batch_norm_legit_no_training = [](const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const at::Tensor & running_mean, const at::Tensor & running_var, double momentum, double eps) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_native_batch_norm_legit_no_training(input, weight, bias, running_mean, running_var, momentum, eps);
  };
  return wrap(dispatch__native_batch_norm_legit_no_training(_r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2), _r.tensor(3), _r.tensor(4), _r.toDouble(5), _r.toDouble(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// batch_norm_elemt
static PyObject * THPVariable_batch_norm_elemt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "batch_norm_elemt(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, double eps, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(6)) {
    // aten::batch_norm_elemt(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps) -> Tensor
    
    auto dispatch_batch_norm_elemt = [](const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::batch_norm_elemt(input, weight, bias, mean, invstd, eps);
    };
    return wrap(dispatch_batch_norm_elemt(_r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2), _r.tensor(3), _r.tensor(4), _r.toDouble(5)));
  } else {
    // aten::batch_norm_elemt.out(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_batch_norm_elemt_out = [](at::Tensor out, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::batch_norm_elemt_out(out, input, weight, bias, mean, invstd, eps);
    };
    return wrap(dispatch_batch_norm_elemt_out(_r.tensor(6), _r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2), _r.tensor(3), _r.tensor(4), _r.toDouble(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// batch_norm_gather_stats_with_counts
static PyObject * THPVariable_batch_norm_gather_stats_with_counts(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "batch_norm_gather_stats_with_counts(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, double momentum, double eps, Tensor counts)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::batch_norm_gather_stats_with_counts(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, Tensor counts) -> (Tensor, Tensor)
  
  auto dispatch_batch_norm_gather_stats_with_counts = [](const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, double momentum, double eps, const at::Tensor & counts) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::batch_norm_gather_stats_with_counts(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
  };
  return wrap(dispatch_batch_norm_gather_stats_with_counts(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.optionalTensor(3), _r.optionalTensor(4), _r.toDouble(5), _r.toDouble(6), _r.tensor(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// batch_norm_backward_elemt
static PyObject * THPVariable_batch_norm_backward_elemt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "batch_norm_backward_elemt(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, Tensor sum_dy, Tensor sum_dy_xmu, Tensor count)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::batch_norm_backward_elemt(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, Tensor sum_dy, Tensor sum_dy_xmu, Tensor count) -> Tensor
  
  auto dispatch_batch_norm_backward_elemt = [](const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const ::std::optional<at::Tensor> & weight, const at::Tensor & sum_dy, const at::Tensor & sum_dy_xmu, const at::Tensor & count) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::batch_norm_backward_elemt(grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
  };
  return wrap(dispatch_batch_norm_backward_elemt(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.optionalTensor(4), _r.tensor(5), _r.tensor(6), _r.tensor(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ones_like
static PyObject * THPVariable_ones_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ones_like(Tensor input, *, MemoryFormat? memory_format=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::ones_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
  auto self = _r.tensor(0);
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(2))
      .device(_r.deviceOptional(4))
      .layout(_r.layoutOptional(3))
      .requires_grad(_r.toBool(6))
      .pinned_memory(_r.toBool(5));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch_ones_like = [](const at::Tensor & self, at::TensorOptions options, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::ones_like(self, options, memory_format);
  };
  return wrap(dispatch_ones_like(self, options, _r.memoryformatOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// pairwise_distance
static PyObject * THPVariable_pairwise_distance(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pairwise_distance(Tensor x1, Tensor x2, double p=2, double eps=1e-06, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::pairwise_distance(Tensor x1, Tensor x2, float p=2, float eps=1e-06, bool keepdim=False) -> Tensor
  
  auto dispatch_pairwise_distance = [](const at::Tensor & x1, const at::Tensor & x2, double p, double eps, bool keepdim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::pairwise_distance(x1, x2, p, eps, keepdim);
  };
  return wrap(dispatch_pairwise_distance(_r.tensor(0), _r.tensor(1), _r.toDouble(2), _r.toDouble(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cdist
static PyObject * THPVariable_cdist(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cdist(Tensor x1, Tensor x2, double p=2, int64_t? compute_mode=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::cdist(Tensor x1, Tensor x2, float p=2, int? compute_mode=None) -> Tensor
  
  auto dispatch_cdist = [](const at::Tensor & x1, const at::Tensor & x2, double p, ::std::optional<int64_t> compute_mode) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::cdist(x1, x2, p, compute_mode);
  };
  return wrap(dispatch_cdist(_r.tensor(0), _r.tensor(1), _r.toDouble(2), _r.toInt64Optional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cosine_similarity
static PyObject * THPVariable_cosine_similarity(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cosine_similarity(Tensor x1, Tensor x2, int64_t dim=1, double eps=1e-08)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::cosine_similarity(Tensor x1, Tensor x2, int dim=1, float eps=1e-08) -> Tensor
  
  auto dispatch_cosine_similarity = [](const at::Tensor & x1, const at::Tensor & x2, int64_t dim, double eps) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::cosine_similarity(x1, x2, dim, eps);
  };
  return wrap(dispatch_cosine_similarity(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toDouble(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// movedim
static PyObject * THPVariable_movedim(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "movedim(Tensor input, int64_t source, int64_t destination)",
    "movedim(Tensor input, IntArrayRef source, IntArrayRef destination)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::movedim.int(Tensor(a) self, int source, int destination) -> Tensor(a)
      
      auto dispatch_movedim = [](const at::Tensor & self, int64_t source, int64_t destination) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.movedim(source, destination);
      };
      return wrap(dispatch_movedim(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
    }
    case 1: {
      // aten::movedim.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)
      
      auto dispatch_movedim = [](const at::Tensor & self, at::IntArrayRef source, at::IntArrayRef destination) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.movedim(source, destination);
      };
      return wrap(dispatch_movedim(_r.tensor(0), _r.intlist(1), _r.intlist(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// adjoint
static PyObject * THPVariable_adjoint(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adjoint(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::adjoint(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_adjoint = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.adjoint();
  };
  return wrap(dispatch_adjoint(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// channel_shuffle
static PyObject * THPVariable_channel_shuffle(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "channel_shuffle(Tensor input, SymInt groups)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::channel_shuffle(Tensor self, SymInt groups) -> Tensor
  
  auto dispatch_channel_shuffle = [](const at::Tensor & self, c10::SymInt groups) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::channel_shuffle_symint(self, groups);
  };
  return wrap(dispatch_channel_shuffle(_r.tensor(0), _r.toSymInt(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// pinverse
static PyObject * THPVariable_pinverse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pinverse(Tensor input, double rcond=1e-15)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::pinverse(Tensor self, float rcond=1e-15) -> Tensor
  
  auto dispatch_pinverse = [](const at::Tensor & self, double rcond) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.pinverse(rcond);
  };
  return wrap(dispatch_pinverse(_r.tensor(0), _r.toDouble(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// poisson_nll_loss
static PyObject * THPVariable_poisson_nll_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "poisson_nll_loss(Tensor input, Tensor target, bool log_input, bool full, double eps, int64_t reduction)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::poisson_nll_loss(Tensor input, Tensor target, bool log_input, bool full, float eps, int reduction) -> Tensor
  
  auto dispatch_poisson_nll_loss = [](const at::Tensor & input, const at::Tensor & target, bool log_input, bool full, double eps, int64_t reduction) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::poisson_nll_loss(input, target, log_input, full, eps, reduction);
  };
  return wrap(dispatch_poisson_nll_loss(_r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3), _r.toDouble(4), _r.toInt64(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// deg2rad
static PyObject * THPVariable_deg2rad(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "deg2rad(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::deg2rad(Tensor self) -> Tensor
    
    auto dispatch_deg2rad = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.deg2rad();
    };
    return wrap(dispatch_deg2rad(_r.tensor(0)));
  } else {
    // aten::deg2rad.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_deg2rad_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::deg2rad_out(out, self);
    };
    return wrap(dispatch_deg2rad_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// deg2rad_
static PyObject * THPVariable_deg2rad_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "deg2rad_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::deg2rad_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_deg2rad_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.deg2rad_();
  };
  return wrap(dispatch_deg2rad_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// randn_like
static PyObject * THPVariable_randn_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randn_like(Tensor input, *, MemoryFormat? memory_format=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::randn_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
  auto self = _r.tensor(0);
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(2))
      .device(_r.deviceOptional(4))
      .layout(_r.layoutOptional(3))
      .requires_grad(_r.toBool(6))
      .pinned_memory(_r.toBool(5));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch_randn_like = [](const at::Tensor & self, at::TensorOptions options, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::randn_like(self, options, memory_format);
  };
  return wrap(dispatch_randn_like(self, options, _r.memoryformatOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// repeat_interleave
static PyObject * THPVariable_repeat_interleave(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "repeat_interleave(Tensor input, Tensor repeats, int64_t? dim=None, *, SymInt? output_size=None)",
    "repeat_interleave(Tensor repeats, *, SymInt? output_size=None)",
    "repeat_interleave(Tensor input, SymInt repeats, int64_t? dim=None, *, SymInt? output_size=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None, *, SymInt? output_size=None) -> Tensor
      
      auto dispatch_repeat_interleave = [](const at::Tensor & self, const at::Tensor & repeats, ::std::optional<int64_t> dim, ::std::optional<c10::SymInt> output_size) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.repeat_interleave_symint(repeats, dim, output_size);
      };
      return wrap(dispatch_repeat_interleave(_r.tensor(0), _r.tensor(1), _r.toInt64Optional(2), _r.toSymIntOptional(3)));
    }
    case 1: {
      // aten::repeat_interleave.Tensor(Tensor repeats, *, SymInt? output_size=None) -> Tensor
      
      auto dispatch_repeat_interleave = [](const at::Tensor & repeats, ::std::optional<c10::SymInt> output_size) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::repeat_interleave_symint(repeats, output_size);
      };
      return wrap(dispatch_repeat_interleave(_r.tensor(0), _r.toSymIntOptional(1)));
    }
    case 2: {
      // aten::repeat_interleave.self_int(Tensor self, SymInt repeats, int? dim=None, *, SymInt? output_size=None) -> Tensor
      
      auto dispatch_repeat_interleave = [](const at::Tensor & self, c10::SymInt repeats, ::std::optional<int64_t> dim, ::std::optional<c10::SymInt> output_size) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.repeat_interleave_symint(repeats, dim, output_size);
      };
      return wrap(dispatch_repeat_interleave(_r.tensor(0), _r.toSymInt(1), _r.toInt64Optional(2), _r.toSymIntOptional(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _prelu_kernel
static PyObject * THPVariable__prelu_kernel(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_prelu_kernel(Tensor input, Tensor weight)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_prelu_kernel(Tensor self, Tensor weight) -> Tensor
  
  auto dispatch__prelu_kernel = [](const at::Tensor & self, const at::Tensor & weight) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_prelu_kernel(self, weight);
  };
  return wrap(dispatch__prelu_kernel(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hardshrink
static PyObject * THPVariable_hardshrink(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hardshrink(Tensor input, Scalar lambd=0.5, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor
    
    auto dispatch_hardshrink = [](const at::Tensor & self, const at::Scalar & lambd) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.hardshrink(lambd);
    };
    return wrap(dispatch_hardshrink(_r.tensor(0), _r.scalar(1)));
  } else {
    // aten::hardshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_hardshrink_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & lambd) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::hardshrink_out(out, self, lambd);
    };
    return wrap(dispatch_hardshrink_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rsqrt
static PyObject * THPVariable_rsqrt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rsqrt(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::rsqrt(Tensor self) -> Tensor
    
    auto dispatch_rsqrt = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.rsqrt();
    };
    return wrap(dispatch_rsqrt(_r.tensor(0)));
  } else {
    // aten::rsqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_rsqrt_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::rsqrt_out(out, self);
    };
    return wrap(dispatch_rsqrt_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rsqrt_
static PyObject * THPVariable_rsqrt_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rsqrt_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_rsqrt_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.rsqrt_();
  };
  return wrap(dispatch_rsqrt_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// celu
static PyObject * THPVariable_celu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "celu(Tensor input, Scalar alpha=1.0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::celu(Tensor self, Scalar alpha=1.0) -> Tensor
  
  auto dispatch_celu = [](const at::Tensor & self, const at::Scalar & alpha) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::celu(self, alpha);
  };
  return wrap(dispatch_celu(_r.tensor(0), _r.scalar(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// celu_
static PyObject * THPVariable_celu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "celu_(Tensor input, Scalar alpha=1.0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::celu_(Tensor(a!) self, Scalar alpha=1.0) -> Tensor(a!)
  
  auto dispatch_celu_ = [](at::Tensor self, const at::Scalar & alpha) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::celu_(self, alpha);
  };
  return wrap(dispatch_celu_(_r.tensor(0), _r.scalar(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sinc
static PyObject * THPVariable_sinc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sinc(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::sinc(Tensor self) -> Tensor
    
    auto dispatch_sinc = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.sinc();
    };
    return wrap(dispatch_sinc(_r.tensor(0)));
  } else {
    // aten::sinc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_sinc_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::sinc_out(out, self);
    };
    return wrap(dispatch_sinc_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sinc_
static PyObject * THPVariable_sinc_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sinc_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::sinc_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_sinc_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sinc_();
  };
  return wrap(dispatch_sinc_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sinh
static PyObject * THPVariable_sinh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sinh(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::sinh(Tensor self) -> Tensor
    
    auto dispatch_sinh = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.sinh();
    };
    return wrap(dispatch_sinh(_r.tensor(0)));
  } else {
    // aten::sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_sinh_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::sinh_out(out, self);
    };
    return wrap(dispatch_sinh_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sinh_
static PyObject * THPVariable_sinh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sinh_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::sinh_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_sinh_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sinh_();
  };
  return wrap(dispatch_sinh_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// detach
static PyObject * THPVariable_detach(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "detach(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::detach(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_detach = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.detach();
  };
  return wrap(dispatch_detach(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// detach_
static PyObject * THPVariable_detach_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "detach_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::detach_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_detach_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.detach_();
  };
  return wrap(dispatch_detach_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _softmax
static PyObject * THPVariable__softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_softmax(Tensor input, int64_t dim, bool half_to_float, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
    
    auto dispatch__softmax = [](const at::Tensor & self, int64_t dim, bool half_to_float) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_softmax(self, dim, half_to_float);
    };
    return wrap(dispatch__softmax(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
  } else {
    // aten::_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__softmax_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, bool half_to_float) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_softmax_out(out, self, dim, half_to_float);
    };
    return wrap(dispatch__softmax_out(_r.tensor(3), _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _softmax_backward_data
static PyObject * THPVariable__softmax_backward_data(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_softmax_backward_data(Tensor grad_output, Tensor output, int64_t dim, ScalarType input_dtype, *, Tensor grad_input=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor
    
    auto dispatch__softmax_backward_data = [](const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_softmax_backward_data(grad_output, output, dim, input_dtype);
    };
    return wrap(dispatch__softmax_backward_data(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.scalartype(3)));
  } else {
    // aten::_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) grad_input) -> Tensor(a!)
    
    auto dispatch__softmax_backward_data_out = [](at::Tensor grad_input, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_softmax_backward_data_out(grad_input, grad_output, output, dim, input_dtype);
    };
    return wrap(dispatch__softmax_backward_data_out(_r.tensor(4), _r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.scalartype(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unsafe_split
static PyObject * THPVariable_unsafe_split(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unsafe_split(Tensor input, SymInt split_size, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::unsafe_split.Tensor(Tensor self, SymInt split_size, int dim=0) -> Tensor[]
  
  auto dispatch_unsafe_split = [](const at::Tensor & self, c10::SymInt split_size, int64_t dim) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.unsafe_split_symint(split_size, dim);
  };
  return wrap(dispatch_unsafe_split(_r.tensor(0), _r.toSymInt(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// split
static PyObject * THPVariable_split(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "split(Tensor input, SymInt split_size, int64_t dim=0)",
    "split(Tensor input, SymIntArrayRef split_size, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::split.Tensor(Tensor(a -> *) self, SymInt split_size, int dim=0) -> Tensor(a)[]
      
      auto dispatch_split = [](const at::Tensor & self, c10::SymInt split_size, int64_t dim) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.split_symint(split_size, dim);
      };
      return wrap(dispatch_split(_r.tensor(0), _r.toSymInt(1), _r.toInt64(2)));
    }
    case 1: {
      // aten::split.sizes(Tensor(a -> *) self, SymInt[] split_size, int dim=0) -> Tensor(a)[]
      
      auto dispatch_split = [](const at::Tensor & self, c10::SymIntArrayRef split_size, int64_t dim) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.split_symint(split_size, dim);
      };
      return wrap(dispatch_split(_r.tensor(0), _r.symintlist(1), _r.toInt64(2)));
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
  static PythonArgParser parser({
    "hsplit(Tensor input, int64_t sections)",
    "hsplit(Tensor input, IntArrayRef indices)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::hsplit.int(Tensor(a -> *) self, int sections) -> Tensor(a)[]
      
      auto dispatch_hsplit = [](const at::Tensor & self, int64_t sections) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.hsplit(sections);
      };
      return wrap(dispatch_hsplit(_r.tensor(0), _r.toInt64(1)));
    }
    case 1: {
      // aten::hsplit.array(Tensor(a -> *) self, int[] indices) -> Tensor(a)[]
      
      auto dispatch_hsplit = [](const at::Tensor & self, at::IntArrayRef indices) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.hsplit(indices);
      };
      return wrap(dispatch_hsplit(_r.tensor(0), _r.intlist(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// squeeze
static PyObject * THPVariable_squeeze(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "squeeze(Tensor input)",
    "squeeze(Tensor input, int64_t dim)",
    "squeeze(Tensor input, IntArrayRef dim)",
    "squeeze(Tensor input, Dimname dim)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::squeeze(Tensor(a) self) -> Tensor(a)
      
      auto dispatch_squeeze = [](const at::Tensor & self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze();
      };
      return wrap(dispatch_squeeze(_r.tensor(0)));
    }
    case 1: {
      // aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
      
      auto dispatch_squeeze = [](const at::Tensor & self, int64_t dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze(dim);
      };
      return wrap(dispatch_squeeze(_r.tensor(0), _r.toInt64(1)));
    }
    case 2: {
      // aten::squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)
      
      auto dispatch_squeeze = [](const at::Tensor & self, at::IntArrayRef dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze(dim);
      };
      return wrap(dispatch_squeeze(_r.tensor(0), _r.intlist(1)));
    }
    case 3: {
      // aten::squeeze.dimname(Tensor(a) self, Dimname dim) -> Tensor(a)
      
      auto dispatch_squeeze = [](const at::Tensor & self, at::Dimname dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.squeeze(dim);
      };
      return wrap(dispatch_squeeze(_r.tensor(0), _r.dimname(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// stack
static PyObject * THPVariable_stack(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "stack(TensorList tensors, int64_t dim=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::stack(Tensor[] tensors, int dim=0) -> Tensor
    
    auto dispatch_stack = [](at::TensorList tensors, int64_t dim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::stack(tensors, dim);
    };
    return wrap(dispatch_stack(_r.tensorlist(0), _r.toInt64(1)));
  } else {
    // aten::stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_stack_out = [](at::Tensor out, at::TensorList tensors, int64_t dim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::stack_out(out, tensors, dim);
    };
    return wrap(dispatch_stack_out(_r.tensor(2), _r.tensorlist(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _stack
static PyObject * THPVariable__stack(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_stack(TensorList tensors, int64_t dim=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::_stack(Tensor[] tensors, int dim=0) -> Tensor
    
    auto dispatch__stack = [](at::TensorList tensors, int64_t dim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_stack(tensors, dim);
    };
    return wrap(dispatch__stack(_r.tensorlist(0), _r.toInt64(1)));
  } else {
    // aten::_stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__stack_out = [](at::Tensor out, at::TensorList tensors, int64_t dim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_stack_out(out, tensors, dim);
    };
    return wrap(dispatch__stack_out(_r.tensor(2), _r.tensorlist(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// istft
static PyObject * THPVariable_istft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "istft(Tensor input, int64_t n_fft, int64_t? hop_length=None, int64_t? win_length=None, Tensor? window=None, bool center=True, bool normalized=False, bool? onesided=None, int64_t? length=None, bool return_complex=False)",
  }, /*traceable=*/true);

  ParsedArgs<10> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::istft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool center=True, bool normalized=False, bool? onesided=None, int? length=None, bool return_complex=False) -> Tensor
  
  auto dispatch_istft = [](const at::Tensor & self, int64_t n_fft, ::std::optional<int64_t> hop_length, ::std::optional<int64_t> win_length, const ::std::optional<at::Tensor> & window, bool center, bool normalized, ::std::optional<bool> onesided, ::std::optional<int64_t> length, bool return_complex) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.istft(n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex);
  };
  return wrap(dispatch_istft(_r.tensor(0), _r.toInt64(1), _r.toInt64Optional(2), _r.toInt64Optional(3), _r.optionalTensor(4), _r.toBool(5), _r.toBool(6), _r.toBoolOptional(7), _r.toInt64Optional(8), _r.toBool(9)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// nansum
static PyObject * THPVariable_nansum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nansum(Tensor input, IntArrayRef[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::nansum(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    
    auto dispatch_nansum = [](const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.nansum(dim, keepdim, dtype);
    };
    return wrap(dispatch_nansum(_r.tensor(0), _r.intlistOptional(1), _r.toBool(2), _r.scalartypeOptional(3)));
  } else {
    // aten::nansum.out(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_nansum_out = [](at::Tensor out, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::nansum_out(out, self, dim, keepdim, dtype);
    };
    return wrap(dispatch_nansum_out(_r.tensor(4), _r.tensor(0), _r.intlistOptional(1), _r.toBool(2), _r.scalartypeOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sqrt
static PyObject * THPVariable_sqrt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sqrt(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::sqrt(Tensor self) -> Tensor
    
    auto dispatch_sqrt = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.sqrt();
    };
    return wrap(dispatch_sqrt(_r.tensor(0)));
  } else {
    // aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_sqrt_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::sqrt_out(out, self);
    };
    return wrap(dispatch_sqrt_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sqrt_
static PyObject * THPVariable_sqrt_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sqrt_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::sqrt_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_sqrt_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sqrt_();
  };
  return wrap(dispatch_sqrt_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// square
static PyObject * THPVariable_square(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "square(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::square(Tensor self) -> Tensor
    
    auto dispatch_square = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.square();
    };
    return wrap(dispatch_square(_r.tensor(0)));
  } else {
    // aten::square.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_square_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::square_out(out, self);
    };
    return wrap(dispatch_square_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// square_
static PyObject * THPVariable_square_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "square_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::square_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_square_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.square_();
  };
  return wrap(dispatch_square_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// t
static PyObject * THPVariable_t(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "t(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::t(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_t = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.t();
  };
  return wrap(dispatch_t(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tanh
static PyObject * THPVariable_tanh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tanh(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::tanh(Tensor self) -> Tensor
    
    auto dispatch_tanh = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.tanh();
    };
    return wrap(dispatch_tanh(_r.tensor(0)));
  } else {
    // aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_tanh_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::tanh_out(out, self);
    };
    return wrap(dispatch_tanh_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tanh_
static PyObject * THPVariable_tanh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tanh_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::tanh_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_tanh_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.tanh_();
  };
  return wrap(dispatch_tanh_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tensordot
static PyObject * THPVariable_tensordot(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tensordot(Tensor input, Tensor other, IntArrayRef dims_self, IntArrayRef dims_other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::tensordot(Tensor self, Tensor other, int[] dims_self, int[] dims_other) -> Tensor
    
    auto dispatch_tensordot = [](const at::Tensor & self, const at::Tensor & other, at::IntArrayRef dims_self, at::IntArrayRef dims_other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::tensordot(self, other, dims_self, dims_other);
    };
    return wrap(dispatch_tensordot(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.intlist(3)));
  } else {
    // aten::tensordot.out(Tensor self, Tensor other, int[] dims_self, int[] dims_other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_tensordot_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other, at::IntArrayRef dims_self, at::IntArrayRef dims_other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::tensordot_out(out, self, other, dims_self, dims_other);
    };
    return wrap(dispatch_tensordot_out(_r.tensor(4), _r.tensor(0), _r.tensor(1), _r.intlist(2), _r.intlist(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tile
static PyObject * THPVariable_tile(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tile(Tensor input, SymIntArrayRef dims)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::tile(Tensor self, SymInt[] dims) -> Tensor
  
  auto dispatch_tile = [](const at::Tensor & self, c10::SymIntArrayRef dims) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.tile_symint(dims);
  };
  return wrap(dispatch_tile(_r.tensor(0), _r.symintlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _mkldnn_transpose
static PyObject * THPVariable__mkldnn_transpose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_mkldnn_transpose(Tensor input, int64_t dim0, int64_t dim1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_mkldnn_transpose(Tensor self, int dim0, int dim1) -> Tensor
  
  auto dispatch__mkldnn_transpose = [](const at::Tensor & self, int64_t dim0, int64_t dim1) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_mkldnn_transpose(self, dim0, dim1);
  };
  return wrap(dispatch__mkldnn_transpose(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _mkldnn_transpose_
static PyObject * THPVariable__mkldnn_transpose_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_mkldnn_transpose_(Tensor input, int64_t dim0, int64_t dim1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_mkldnn_transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
  
  auto dispatch__mkldnn_transpose_ = [](at::Tensor self, int64_t dim0, int64_t dim1) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_mkldnn_transpose_(self, dim0, dim1);
  };
  return wrap(dispatch__mkldnn_transpose_(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rot90
static PyObject * THPVariable_rot90(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rot90(Tensor input, int64_t k=1, IntArrayRef dims={0,1})",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor
  
  auto dispatch_rot90 = [](const at::Tensor & self, int64_t k, at::IntArrayRef dims) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.rot90(k, dims);
  };
  return wrap(dispatch_rot90(_r.tensor(0), _r.toInt64(1), _r.intlist(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_from_padded
static PyObject * THPVariable__nested_from_padded(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_from_padded(Tensor padded, Tensor cpu_nested_shape_example, bool fuse_transform_0213=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_nested_from_padded(Tensor padded, Tensor cpu_nested_shape_example, bool fuse_transform_0213=False) -> Tensor
  
  auto dispatch__nested_from_padded = [](const at::Tensor & padded, const at::Tensor & cpu_nested_shape_example, bool fuse_transform_0213) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_nested_from_padded(padded, cpu_nested_shape_example, fuse_transform_0213);
  };
  return wrap(dispatch__nested_from_padded(_r.tensor(0), _r.tensor(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_get_values
static PyObject * THPVariable__nested_get_values(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_get_values(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_nested_get_values(Tensor(a) self) -> Tensor(a)
  
  auto dispatch__nested_get_values = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_nested_get_values(self);
  };
  return wrap(dispatch__nested_get_values(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_get_offsets
static PyObject * THPVariable__nested_get_offsets(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_get_offsets(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_nested_get_offsets(Tensor self) -> Tensor
  
  auto dispatch__nested_get_offsets = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_nested_get_offsets(self);
  };
  return wrap(dispatch__nested_get_offsets(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_get_jagged_dummy
static PyObject * THPVariable__nested_get_jagged_dummy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_get_jagged_dummy(Tensor any)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_nested_get_jagged_dummy(Tensor any) -> Tensor
  
  auto dispatch__nested_get_jagged_dummy = [](const at::Tensor & any) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_nested_get_jagged_dummy(any);
  };
  return wrap(dispatch__nested_get_jagged_dummy(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// triplet_margin_loss
static PyObject * THPVariable_triplet_margin_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, double margin=1.0, double p=2, double eps=1e-06, bool swap=False, int64_t reduction=at::Reduction::Mean)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, float margin=1.0, float p=2, float eps=1e-06, bool swap=False, int reduction=Mean) -> Tensor
  
  auto dispatch_triplet_margin_loss = [](const at::Tensor & anchor, const at::Tensor & positive, const at::Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap, reduction);
  };
  return wrap(dispatch_triplet_margin_loss(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toDouble(3), _r.toDouble(4), _r.toDouble(5), _r.toBool(6), _r.toInt64(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fix
static PyObject * THPVariable_fix(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fix(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::fix(Tensor self) -> Tensor
    
    auto dispatch_fix = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.fix();
    };
    return wrap(dispatch_fix(_r.tensor(0)));
  } else {
    // aten::fix.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fix_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fix_out(out, self);
    };
    return wrap(dispatch_fix_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fix_
static PyObject * THPVariable_fix_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fix_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::fix_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_fix_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.fix_();
  };
  return wrap(dispatch_fix_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _has_compatible_shallow_copy_type
static PyObject * THPVariable__has_compatible_shallow_copy_type(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_has_compatible_shallow_copy_type(Tensor input, Tensor from)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_has_compatible_shallow_copy_type(Tensor self, Tensor from) -> bool
  
  auto dispatch__has_compatible_shallow_copy_type = [](const at::Tensor & self, const at::Tensor & from) -> bool {
    pybind11::gil_scoped_release no_gil;
    return at::_has_compatible_shallow_copy_type(self, from);
  };
  return wrap(dispatch__has_compatible_shallow_copy_type(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _unique
static PyObject * THPVariable__unique(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_unique(Tensor input, bool sorted=True, bool return_inverse=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_unique(Tensor self, bool sorted=True, bool return_inverse=False) -> (Tensor, Tensor)
  
  auto dispatch__unique = [](const at::Tensor & self, bool sorted, bool return_inverse) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_unique(self, sorted, return_inverse);
  };
  return wrap(dispatch__unique(_r.tensor(0), _r.toBool(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// vander
static PyObject * THPVariable_vander(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "vander(Tensor x, int64_t? N=None, bool increasing=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::vander(Tensor x, int? N=None, bool increasing=False) -> Tensor
  
  auto dispatch_vander = [](const at::Tensor & x, ::std::optional<int64_t> N, bool increasing) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::vander(x, N, increasing);
  };
  return wrap(dispatch_vander(_r.tensor(0), _r.toInt64Optional(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// var_mean
static PyObject * THPVariable_var_mean(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "var_mean(Tensor input, IntArrayRef[1]? dim, bool unbiased=True, bool keepdim=False)",
    "var_mean(Tensor input, IntArrayRef[1]? dim=None, *, Scalar? correction=None, bool keepdim=False)",
    "var_mean(Tensor input, bool unbiased=True)",
    "var_mean(Tensor input, DimnameList[1] dim, bool unbiased=True, bool keepdim=False)",
    "var_mean(Tensor input, DimnameList[1] dim, *, Scalar? correction=None, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::var_mean.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
      
      auto dispatch_var_mean = [](const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::var_mean(self, dim, unbiased, keepdim);
      };
      return wrap(dispatch_var_mean(_r.tensor(0), _r.intlistOptional(1), _r.toBool(2), _r.toBool(3)));
    }
    case 1: {
      // aten::var_mean.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)
      
      auto dispatch_var_mean = [](const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::var_mean(self, dim, correction, keepdim);
      };
      return wrap(dispatch_var_mean(_r.tensor(0), _r.intlistOptional(1), _r.scalarOptional(2), _r.toBool(3)));
    }
    case 2: {
      // aten::var_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
      
      auto dispatch_var_mean = [](const at::Tensor & self, bool unbiased) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::var_mean(self, unbiased);
      };
      return wrap(dispatch_var_mean(_r.tensor(0), _r.toBool(1)));
    }
    case 3: {
      // aten::var_mean.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
      
      auto dispatch_var_mean = [](const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::var_mean(self, dim, unbiased, keepdim);
      };
      return wrap(dispatch_var_mean(_r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.toBool(3)));
    }
    case 4: {
      // aten::var_mean.correction_names(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)
      
      auto dispatch_var_mean = [](const at::Tensor & self, at::DimnameList dim, const ::std::optional<at::Scalar> & correction, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::var_mean(self, dim, correction, keepdim);
      };
      return wrap(dispatch_var_mean(_r.tensor(0), _r.dimnamelist(1), _r.scalarOptional(2), _r.toBool(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _weight_norm
static PyObject * THPVariable__weight_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_weight_norm(Tensor v, Tensor g, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_weight_norm(Tensor v, Tensor g, int dim=0) -> Tensor
  
  auto dispatch__weight_norm = [](const at::Tensor & v, const at::Tensor & g, int64_t dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_weight_norm(v, g, dim);
  };
  return wrap(dispatch__weight_norm(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _standard_gamma_grad
static PyObject * THPVariable__standard_gamma_grad(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_standard_gamma_grad(Tensor input, Tensor output)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_standard_gamma_grad(Tensor self, Tensor output) -> Tensor
  
  auto dispatch__standard_gamma_grad = [](const at::Tensor & self, const at::Tensor & output) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_standard_gamma_grad(self, output);
  };
  return wrap(dispatch__standard_gamma_grad(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// poisson
static PyObject * THPVariable_poisson(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "poisson(Tensor input, Generator? generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::poisson(Tensor self, Generator? generator=None) -> Tensor
  
  auto dispatch_poisson = [](const at::Tensor & self, ::std::optional<at::Generator> generator) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::poisson(self, generator);
  };
  return wrap(dispatch_poisson(_r.tensor(0), _r.generator(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// native_norm
static PyObject * THPVariable_native_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "native_norm(Tensor input, Scalar p=2)",
    "native_norm(Tensor input, Scalar? p, IntArrayRef[1] dim, bool keepdim, ScalarType? dtype)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::native_norm(Tensor self, Scalar p=2) -> Tensor
      
      auto dispatch_native_norm = [](const at::Tensor & self, const at::Scalar & p) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::native_norm(self, p);
      };
      return wrap(dispatch_native_norm(_r.tensor(0), _r.scalar(1)));
    }
    case 1: {
      // aten::native_norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, ScalarType? dtype) -> Tensor
      
      auto dispatch_native_norm = [](const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::native_norm(self, p, dim, keepdim, dtype);
      };
      return wrap(dispatch_native_norm(_r.tensor(0), _r.scalarOptional(1), _r.intlist(2), _r.toBool(3), _r.scalartypeOptional(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sparse_softmax_backward_data
static PyObject * THPVariable__sparse_softmax_backward_data(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_softmax_backward_data(Tensor grad_output, Tensor output, int64_t dim, Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_sparse_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
  
  auto dispatch__sparse_softmax_backward_data = [](const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sparse_softmax_backward_data(grad_output, output, dim, self);
  };
  return wrap(dispatch__sparse_softmax_backward_data(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.tensor(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// norm
static PyObject * THPVariable_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "norm(Tensor input, Scalar p=2)",
    "norm(Tensor input, Scalar? p, *, ScalarType dtype)",
    "norm(Tensor input, Scalar? p, IntArrayRef[1] dim, bool keepdim, *, ScalarType dtype, Tensor out=None)",
    "norm(Tensor input, Scalar? p, IntArrayRef[1] dim, bool keepdim=False, *, Tensor out=None)",
    "norm(Tensor input, Scalar? p, DimnameList[1] dim, bool keepdim, *, ScalarType dtype, Tensor out=None)",
    "norm(Tensor input, Scalar? p, DimnameList[1] dim, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor
      
      auto dispatch_norm = [](const at::Tensor & self, const at::Scalar & p) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.norm(p);
      };
      return wrap(dispatch_norm(_r.tensor(0), _r.scalar(1)));
    }
    case 1: {
      // aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor
      
      auto dispatch_norm = [](const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::ScalarType dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.norm(p, dtype);
      };
      return wrap(dispatch_norm(_r.tensor(0), _r.scalarOptional(1), _r.scalartype(2)));
    }
    case 2: {
      if (_r.isNone(5)) {
        // aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
        
        auto dispatch_norm = [](const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.norm(p, dim, keepdim, dtype);
        };
        return wrap(dispatch_norm(_r.tensor(0), _r.scalarOptional(1), _r.intlist(2), _r.toBool(3), _r.scalartype(4)));
      } else {
        // aten::norm.dtype_out(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_norm_out = [](at::Tensor out, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::norm_out(out, self, p, dim, keepdim, dtype);
        };
        return wrap(dispatch_norm_out(_r.tensor(5), _r.tensor(0), _r.scalarOptional(1), _r.intlist(2), _r.toBool(3), _r.scalartype(4)));
      }
    }
    case 3: {
      if (_r.isNone(4)) {
        // aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor
        
        auto dispatch_norm = [](const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.norm(p, dim, keepdim);
        };
        return wrap(dispatch_norm(_r.tensor(0), _r.scalarOptional(1), _r.intlist(2), _r.toBool(3)));
      } else {
        // aten::norm.out(Tensor self, Scalar? p, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_norm_out = [](at::Tensor out, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::norm_out(out, self, p, dim, keepdim);
        };
        return wrap(dispatch_norm_out(_r.tensor(4), _r.tensor(0), _r.scalarOptional(1), _r.intlist(2), _r.toBool(3)));
      }
    }
    case 4: {
      if (_r.isNone(5)) {
        // aten::norm.names_ScalarOpt_dim_dtype(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
        
        auto dispatch_norm = [](const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::ScalarType dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.norm(p, dim, keepdim, dtype);
        };
        return wrap(dispatch_norm(_r.tensor(0), _r.scalarOptional(1), _r.dimnamelist(2), _r.toBool(3), _r.scalartype(4)));
      } else {
        // aten::norm.names_dtype_out(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_norm_out = [](at::Tensor out, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::ScalarType dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::norm_out(out, self, p, dim, keepdim, dtype);
        };
        return wrap(dispatch_norm_out(_r.tensor(5), _r.tensor(0), _r.scalarOptional(1), _r.dimnamelist(2), _r.toBool(3), _r.scalartype(4)));
      }
    }
    case 5: {
      if (_r.isNone(4)) {
        // aten::norm.names_ScalarOpt_dim(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim=False) -> Tensor
        
        auto dispatch_norm = [](const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.norm(p, dim, keepdim);
        };
        return wrap(dispatch_norm(_r.tensor(0), _r.scalarOptional(1), _r.dimnamelist(2), _r.toBool(3)));
      } else {
        // aten::norm.names_out(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_norm_out = [](at::Tensor out, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::norm_out(out, self, p, dim, keepdim);
        };
        return wrap(dispatch_norm_out(_r.tensor(4), _r.tensor(0), _r.scalarOptional(1), _r.dimnamelist(2), _r.toBool(3)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// frexp
static PyObject * THPVariable_frexp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_frexp_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_frexp_out_structseq();
  static PythonArgParser parser({
    "frexp(Tensor input, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::frexp.Tensor(Tensor self) -> (Tensor mantissa, Tensor exponent)
    
    auto dispatch_frexp = [](const at::Tensor & self) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return self.frexp();
    };
    return wrap(NamedTuple, dispatch_frexp(_r.tensor(0)));
  } else {
    // aten::frexp.Tensor_out(Tensor self, *, Tensor(a!) mantissa, Tensor(b!) exponent) -> (Tensor(a!) mantissa, Tensor(b!) exponent)
    auto out = _r.tensorlist_n<2>(1);
    auto dispatch_frexp_out = [](at::Tensor & mantissa, at::Tensor & exponent, const at::Tensor & self) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::frexp_out(mantissa, exponent, self);
    };
    return wrap(NamedTuple1, dispatch_frexp_out(out[0], out[1], _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// positive
static PyObject * THPVariable_positive(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "positive(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::positive(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_positive = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.positive();
  };
  return wrap(dispatch_positive(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// resize_as_
static PyObject * THPVariable_resize_as_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "resize_as_(Tensor input, Tensor the_template, *, MemoryFormat? memory_format=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::resize_as_(Tensor(a!) self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor(a!)
  
  auto dispatch_resize_as_ = [](const at::Tensor & self, const at::Tensor & the_template, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.resize_as_(the_template, memory_format);
  };
  return wrap(dispatch_resize_as_(_r.tensor(0), _r.tensor(1), _r.memoryformatOptional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// zero_
static PyObject * THPVariable_zero_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "zero_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::zero_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_zero_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.zero_();
  };
  return wrap(dispatch_zero_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// sub
static PyObject * THPVariable_sub(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sub(Tensor input, Scalar alpha, Tensor other, *, Tensor out=None)|deprecated",
    "sub(Tensor input, Tensor other, *, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // [deprecated] aten::sub(Tensor self, Scalar alpha, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_sub = [](at::Tensor out, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.sub(other, alpha);
        };
        return wrap(dispatch_sub(_r.tensor(3), _r.tensor(0), _r.scalar(1), _r.tensor(2)));
      } else {
        // [deprecated] aten::sub(Tensor self, Scalar alpha, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_sub_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::sub_out(out, self, other, alpha);
        };
        return wrap(dispatch_sub_out(_r.tensor(3), _r.tensor(0), _r.scalar(1), _r.tensor(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
        
        auto dispatch_sub = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.sub(other, alpha);
        };
        return wrap(dispatch_sub(_r.tensor(0), _r.tensor(1), _r.scalar(2)));
      } else {
        // aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_sub_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::sub_out(out, self, other, alpha);
        };
        return wrap(dispatch_sub_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.scalar(2)));
      }
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
  static PythonArgParser parser({
    "subtract(Tensor input, Tensor other, *, Scalar alpha=1, Tensor out=None)",
    "subtract(Tensor input, Scalar other, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::subtract.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
        
        auto dispatch_subtract = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.subtract(other, alpha);
        };
        return wrap(dispatch_subtract(_r.tensor(0), _r.tensor(1), _r.scalar(2)));
      } else {
        // aten::subtract.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_subtract_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::subtract_out(out, self, other, alpha);
        };
        return wrap(dispatch_subtract_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.scalar(2)));
      }
    }
    case 1: {
      // aten::subtract.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
      
      auto dispatch_subtract = [](const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.subtract(other, alpha);
      };
      return wrap(dispatch_subtract(_r.tensor(0), _r.scalar(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// rsub
static PyObject * THPVariable_rsub(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rsub(Tensor input, Tensor other, *, Scalar alpha=1)",
    "rsub(Tensor input, Scalar other, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
      
      auto dispatch_rsub = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::rsub(self, other, alpha);
      };
      return wrap(dispatch_rsub(_r.tensor(0), _r.tensor(1), _r.scalar(2)));
    }
    case 1: {
      // aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
      
      auto dispatch_rsub = [](const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::rsub(self, other, alpha);
      };
      return wrap(dispatch_rsub(_r.tensor(0), _r.scalar(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _validate_sparse_csc_tensor_args
static PyObject * THPVariable__validate_sparse_csc_tensor_args(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_validate_sparse_csc_tensor_args(Tensor ccol_indices, Tensor row_indices, Tensor values, IntArrayRef size)",
  }, /*traceable=*/false);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_validate_sparse_csc_tensor_args(Tensor ccol_indices, Tensor row_indices, Tensor values, int[] size) -> ()
  
  auto dispatch__validate_sparse_csc_tensor_args = [](const at::Tensor & ccol_indices, const at::Tensor & row_indices, const at::Tensor & values, at::IntArrayRef size) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_validate_sparse_csc_tensor_args(ccol_indices, row_indices, values, size);
  };
  dispatch__validate_sparse_csc_tensor_args(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _coalesce
static PyObject * THPVariable__coalesce(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_coalesce(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_coalesce(Tensor self) -> Tensor
  
  auto dispatch__coalesce = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_coalesce(self);
  };
  return wrap(dispatch__coalesce(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// unbind
static PyObject * THPVariable_unbind(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unbind(Tensor input, int64_t dim=0)",
    "unbind(Tensor input, Dimname dim)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::unbind.int(Tensor(a -> *) self, int dim=0) -> Tensor(a)[]
      
      auto dispatch_unbind = [](const at::Tensor & self, int64_t dim) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.unbind(dim);
      };
      return wrap(dispatch_unbind(_r.tensor(0), _r.toInt64(1)));
    }
    case 1: {
      // aten::unbind.Dimname(Tensor(a -> *) self, Dimname dim) -> Tensor(a)[]
      
      auto dispatch_unbind = [](const at::Tensor & self, at::Dimname dim) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.unbind(dim);
      };
      return wrap(dispatch_unbind(_r.tensor(0), _r.dimname(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// quantize_per_tensor_dynamic
static PyObject * THPVariable_quantize_per_tensor_dynamic(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantize_per_tensor_dynamic(Tensor input, ScalarType dtype, bool reduce_range)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::quantize_per_tensor_dynamic(Tensor self, ScalarType dtype, bool reduce_range) -> Tensor
  
  auto dispatch_quantize_per_tensor_dynamic = [](const at::Tensor & self, at::ScalarType dtype, bool reduce_range) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::quantize_per_tensor_dynamic(self, dtype, reduce_range);
  };
  return wrap(dispatch_quantize_per_tensor_dynamic(_r.tensor(0), _r.scalartype(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// quantize_per_tensor
static PyObject * THPVariable_quantize_per_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantize_per_tensor(Tensor input, Tensor scale, Tensor zero_point, ScalarType dtype)",
    "quantize_per_tensor(Tensor input, double scale, int64_t zero_point, ScalarType dtype)",
    "quantize_per_tensor(TensorList tensors, Tensor scales, Tensor zero_points, ScalarType dtype)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::quantize_per_tensor.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, ScalarType dtype) -> Tensor
      
      auto dispatch_quantize_per_tensor = [](const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, at::ScalarType dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::quantize_per_tensor(self, scale, zero_point, dtype);
      };
      return wrap(dispatch_quantize_per_tensor(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalartype(3)));
    }
    case 1: {
      // aten::quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor
      
      auto dispatch_quantize_per_tensor = [](const at::Tensor & self, double scale, int64_t zero_point, at::ScalarType dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::quantize_per_tensor(self, scale, zero_point, dtype);
      };
      return wrap(dispatch_quantize_per_tensor(_r.tensor(0), _r.toDouble(1), _r.toInt64(2), _r.scalartype(3)));
    }
    case 2: {
      // aten::quantize_per_tensor.tensors(Tensor[] tensors, Tensor scales, Tensor zero_points, ScalarType dtype) -> Tensor[]
      
      auto dispatch_quantize_per_tensor = [](at::TensorList tensors, const at::Tensor & scales, const at::Tensor & zero_points, at::ScalarType dtype) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::quantize_per_tensor(tensors, scales, zero_points, dtype);
      };
      return wrap(dispatch_quantize_per_tensor(_r.tensorlist(0), _r.tensor(1), _r.tensor(2), _r.scalartype(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// quantize_per_channel
static PyObject * THPVariable_quantize_per_channel(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, int64_t axis, ScalarType dtype)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::quantize_per_channel(Tensor self, Tensor scales, Tensor zero_points, int axis, ScalarType dtype) -> Tensor
  
  auto dispatch_quantize_per_channel = [](const at::Tensor & self, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, at::ScalarType dtype) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::quantize_per_channel(self, scales, zero_points, axis, dtype);
  };
  return wrap(dispatch_quantize_per_channel(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3), _r.scalartype(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// q_zero_point
static PyObject * THPVariable_q_zero_point(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "q_zero_point(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::q_zero_point(Tensor self) -> int
  
  auto dispatch_q_zero_point = [](const at::Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return self.q_zero_point();
  };
  return wrap(dispatch_q_zero_point(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// int_repr
static PyObject * THPVariable_int_repr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "int_repr(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::int_repr(Tensor self) -> Tensor
  
  auto dispatch_int_repr = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.int_repr();
  };
  return wrap(dispatch_int_repr(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _make_per_tensor_quantized_tensor
static PyObject * THPVariable__make_per_tensor_quantized_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_make_per_tensor_quantized_tensor(Tensor input, double scale, int64_t zero_point)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_make_per_tensor_quantized_tensor(Tensor self, float scale, int zero_point) -> Tensor
  
  auto dispatch__make_per_tensor_quantized_tensor = [](const at::Tensor & self, double scale, int64_t zero_point) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_make_per_tensor_quantized_tensor(self, scale, zero_point);
  };
  return wrap(dispatch__make_per_tensor_quantized_tensor(_r.tensor(0), _r.toDouble(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _make_per_channel_quantized_tensor
static PyObject * THPVariable__make_per_channel_quantized_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_make_per_channel_quantized_tensor(Tensor input, Tensor scale, Tensor zero_point, int64_t axis)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_make_per_channel_quantized_tensor(Tensor self, Tensor scale, Tensor zero_point, int axis) -> Tensor
  
  auto dispatch__make_per_channel_quantized_tensor = [](const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_make_per_channel_quantized_tensor(self, scale, zero_point, axis);
  };
  return wrap(dispatch__make_per_channel_quantized_tensor(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fake_quantize_per_channel_affine
static PyObject * THPVariable_fake_quantize_per_channel_affine(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fake_quantize_per_channel_affine(Tensor input, Tensor scale, Tensor zero_point, int64_t axis, int64_t quant_min, int64_t quant_max)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::fake_quantize_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> Tensor
  
  auto dispatch_fake_quantize_per_channel_affine = [](const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::fake_quantize_per_channel_affine(self, scale, zero_point, axis, quant_min, quant_max);
  };
  return wrap(dispatch_fake_quantize_per_channel_affine(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3), _r.toInt64(4), _r.toInt64(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _fused_moving_avg_obs_fq_helper
static PyObject * THPVariable__fused_moving_avg_obs_fq_helper(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get__fused_moving_avg_obs_fq_helper_structseq();
  static PythonArgParser parser({
    "_fused_moving_avg_obs_fq_helper(Tensor input, Tensor observer_on, Tensor fake_quant_on, Tensor running_min, Tensor running_max, Tensor scale, Tensor zero_point, double averaging_const, int64_t quant_min, int64_t quant_max, int64_t ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False)",
  }, /*traceable=*/true);

  ParsedArgs<13> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_fused_moving_avg_obs_fq_helper(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor output, Tensor mask)
  
  auto dispatch__fused_moving_avg_obs_fq_helper = [](const at::Tensor & self, const at::Tensor & observer_on, const at::Tensor & fake_quant_on, at::Tensor running_min, at::Tensor running_max, at::Tensor scale, at::Tensor zero_point, double averaging_const, int64_t quant_min, int64_t quant_max, int64_t ch_axis, bool per_row_fake_quant, bool symmetric_quant) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_fused_moving_avg_obs_fq_helper(self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant);
  };
  return wrap(NamedTuple, dispatch__fused_moving_avg_obs_fq_helper(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.tensor(5), _r.tensor(6), _r.toDouble(7), _r.toInt64(8), _r.toInt64(9), _r.toInt64(10), _r.toBool(11), _r.toBool(12)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _choose_qparams_per_tensor
static PyObject * THPVariable__choose_qparams_per_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_choose_qparams_per_tensor(Tensor input, bool reduce_range=False)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_choose_qparams_per_tensor(Tensor self, bool reduce_range=False) -> (float, int)
  
  auto dispatch__choose_qparams_per_tensor = [](const at::Tensor & self, bool reduce_range) -> ::std::tuple<double,int64_t> {
    pybind11::gil_scoped_release no_gil;
    return at::_choose_qparams_per_tensor(self, reduce_range);
  };
  return wrap(dispatch__choose_qparams_per_tensor(_r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// meshgrid
static PyObject * THPVariable_meshgrid(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "meshgrid(TensorList tensors)",
    "meshgrid(TensorList tensors, *, c10::string_view indexing)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::meshgrid(Tensor[] tensors) -> Tensor[]
      
      auto dispatch_meshgrid = [](at::TensorList tensors) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::meshgrid(tensors);
      };
      return wrap(dispatch_meshgrid(_r.tensorlist(0)));
    }
    case 1: {
      // aten::meshgrid.indexing(Tensor[] tensors, *, str indexing) -> Tensor[]
      
      auto dispatch_meshgrid = [](at::TensorList tensors, c10::string_view indexing) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::meshgrid(tensors, indexing);
      };
      return wrap(dispatch_meshgrid(_r.tensorlist(0), _r.stringView(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// promote_types
static PyObject * THPVariable_promote_types(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "promote_types(ScalarType type1, ScalarType type2)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::promote_types(ScalarType type1, ScalarType type2) -> ScalarType
  
  auto dispatch_promote_types = [](at::ScalarType type1, at::ScalarType type2) -> at::ScalarType {
    pybind11::gil_scoped_release no_gil;
    return at::promote_types(type1, type2);
  };
  return wrap(dispatch_promote_types(_r.scalartype(0), _r.scalartype(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// gru
static PyObject * THPVariable_gru(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gru(Tensor data, Tensor batch_sizes, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)",
    "gru(Tensor input, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::gru.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
      
      auto dispatch_gru = [](const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::gru(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
      };
      return wrap(dispatch_gru(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensorlist(3), _r.toBool(4), _r.toInt64(5), _r.toDouble(6), _r.toBool(7), _r.toBool(8)));
    }
    case 1: {
      // aten::gru.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
      
      auto dispatch_gru = [](const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::gru(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
      };
      return wrap(dispatch_gru(_r.tensor(0), _r.tensor(1), _r.tensorlist(2), _r.toBool(3), _r.toInt64(4), _r.toDouble(5), _r.toBool(6), _r.toBool(7), _r.toBool(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rnn_tanh_cell
static PyObject * THPVariable_rnn_tanh_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None)",
  }, /*traceable=*/false);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
  
  auto dispatch_rnn_tanh_cell = [](const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const ::std::optional<at::Tensor> & b_ih, const ::std::optional<at::Tensor> & b_hh) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  };
  return wrap(dispatch_rnn_tanh_cell(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.optionalTensor(4), _r.optionalTensor(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// quantized_gru_cell
static PyObject * THPVariable_quantized_gru_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantized_gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh)",
  }, /*traceable=*/true);

  ParsedArgs<14> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::quantized_gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
  
  auto dispatch_quantized_gru_cell = [](const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::quantized_gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  };
  return wrap(dispatch_quantized_gru_cell(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.tensor(5), _r.tensor(6), _r.tensor(7), _r.tensor(8), _r.tensor(9), _r.scalar(10), _r.scalar(11), _r.scalar(12), _r.scalar(13)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// masked_scatter
static PyObject * THPVariable_masked_scatter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "masked_scatter(Tensor input, Tensor mask, Tensor source)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor
  
  auto dispatch_masked_scatter = [](const at::Tensor & self, const at::Tensor & mask, const at::Tensor & source) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.masked_scatter(mask, source);
  };
  return wrap(dispatch_masked_scatter(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// put
static PyObject * THPVariable_put(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "put(Tensor input, Tensor index, Tensor source, bool accumulate=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::put(Tensor self, Tensor index, Tensor source, bool accumulate=False) -> Tensor
  
  auto dispatch_put = [](const at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.put(index, source, accumulate);
  };
  return wrap(dispatch_put(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __or__
static PyObject * THPVariable___or__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__or__(Tensor input, Tensor other)",
    "__or__(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::__or__.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch___or__ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__or__(other);
      };
      return wrap(dispatch___or__(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::__or__.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch___or__ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__or__(other);
      };
      return wrap(dispatch___or__(_r.tensor(0), _r.scalar(1)));
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
  static PythonArgParser parser({
    "__xor__(Tensor input, Tensor other)",
    "__xor__(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::__xor__.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch___xor__ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__xor__(other);
      };
      return wrap(dispatch___xor__(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::__xor__.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch___xor__ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__xor__(other);
      };
      return wrap(dispatch___xor__(_r.tensor(0), _r.scalar(1)));
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
  static PythonArgParser parser({
    "__lshift__(Tensor input, Tensor other)",
    "__lshift__(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::__lshift__.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch___lshift__ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__lshift__(other);
      };
      return wrap(dispatch___lshift__(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch___lshift__ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__lshift__(other);
      };
      return wrap(dispatch___lshift__(_r.tensor(0), _r.scalar(1)));
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
  static PythonArgParser parser({
    "bitwise_right_shift(Tensor input, Tensor other, *, Tensor out=None)",
    "bitwise_right_shift(Scalar self, Tensor other)",
    "bitwise_right_shift(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_bitwise_right_shift = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.bitwise_right_shift(other);
        };
        return wrap(dispatch_bitwise_right_shift(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::bitwise_right_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_bitwise_right_shift_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bitwise_right_shift_out(out, self, other);
        };
        return wrap(dispatch_bitwise_right_shift_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      // aten::bitwise_right_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
      
      auto dispatch_bitwise_right_shift = [](const at::Scalar & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::bitwise_right_shift(self, other);
      };
      return wrap(dispatch_bitwise_right_shift(_r.scalar(0), _r.tensor(1)));
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::bitwise_right_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_bitwise_right_shift = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.bitwise_right_shift(other);
        };
        return wrap(dispatch_bitwise_right_shift(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::bitwise_right_shift.Tensor_Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_bitwise_right_shift_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bitwise_right_shift_out(out, self, other);
        };
        return wrap(dispatch_bitwise_right_shift_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tril
static PyObject * THPVariable_tril(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tril(Tensor input, int64_t diagonal=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::tril(Tensor self, int diagonal=0) -> Tensor
    
    auto dispatch_tril = [](const at::Tensor & self, int64_t diagonal) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.tril(diagonal);
    };
    return wrap(dispatch_tril(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::tril.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_tril_out = [](at::Tensor out, const at::Tensor & self, int64_t diagonal) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::tril_out(out, self, diagonal);
    };
    return wrap(dispatch_tril_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tril_indices
static PyObject * THPVariable_tril_indices(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tril_indices(int64_t row, int64_t col, int64_t offset=0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeWithDefault(3, at::kLong))
      .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
      .layout(_r.layoutOptional(4))
      .requires_grad(_r.toBool(7))
      .pinned_memory(_r.toBool(6));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch_tril_indices = [](int64_t row, int64_t col, int64_t offset, at::TensorOptions options) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::tril_indices(row, col, offset, options);
  };
  return wrap(dispatch_tril_indices(_r.toInt64(0), _r.toInt64(1), _r.toInt64(2), options));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// not_equal
static PyObject * THPVariable_not_equal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "not_equal(Tensor input, Tensor other, *, Tensor out=None)",
    "not_equal(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::not_equal.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_not_equal = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.not_equal(other);
        };
        return wrap(dispatch_not_equal(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::not_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_not_equal_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::not_equal_out(out, self, other);
        };
        return wrap(dispatch_not_equal_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::not_equal.Scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_not_equal = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.not_equal(other);
        };
        return wrap(dispatch_not_equal(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::not_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_not_equal_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::not_equal_out(out, self, other);
        };
        return wrap(dispatch_not_equal_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
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
  static PythonArgParser parser({
    "greater_equal(Tensor input, Tensor other, *, Tensor out=None)",
    "greater_equal(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::greater_equal.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_greater_equal = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.greater_equal(other);
        };
        return wrap(dispatch_greater_equal(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::greater_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_greater_equal_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::greater_equal_out(out, self, other);
        };
        return wrap(dispatch_greater_equal_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::greater_equal.Scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_greater_equal = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.greater_equal(other);
        };
        return wrap(dispatch_greater_equal(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::greater_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_greater_equal_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::greater_equal_out(out, self, other);
        };
        return wrap(dispatch_greater_equal_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
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
  static PythonArgParser parser({
    "less(Tensor input, Tensor other, *, Tensor out=None)",
    "less(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::less.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_less = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.less(other);
        };
        return wrap(dispatch_less(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::less.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_less_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::less_out(out, self, other);
        };
        return wrap(dispatch_less_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::less.Scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_less = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.less(other);
        };
        return wrap(dispatch_less(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::less.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_less_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::less_out(out, self, other);
        };
        return wrap(dispatch_less_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// index_select
static PyObject * THPVariable_index_select(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_select(Tensor input, int64_t dim, Tensor index, *, Tensor out=None)",
    "index_select(Tensor input, Dimname dim, Tensor index, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::index_select(Tensor self, int dim, Tensor index) -> Tensor
        
        auto dispatch_index_select = [](const at::Tensor & self, int64_t dim, const at::Tensor & index) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.index_select(dim, index);
        };
        return wrap(dispatch_index_select(_r.tensor(0), _r.toInt64(1), _r.tensor(2)));
      } else {
        // aten::index_select.out(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_index_select_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, const at::Tensor & index) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::index_select_out(out, self, dim, index);
        };
        return wrap(dispatch_index_select_out(_r.tensor(3), _r.tensor(0), _r.toInt64(1), _r.tensor(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::index_select.dimname(Tensor self, Dimname dim, Tensor index) -> Tensor
        
        auto dispatch_index_select = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.index_select(dim, index);
        };
        return wrap(dispatch_index_select(_r.tensor(0), _r.dimname(1), _r.tensor(2)));
      } else {
        // aten::index_select.dimname_out(Tensor self, Dimname dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_index_select_out = [](at::Tensor out, const at::Tensor & self, at::Dimname dim, const at::Tensor & index) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::index_select_out(out, self, dim, index);
        };
        return wrap(dispatch_index_select_out(_r.tensor(3), _r.tensor(0), _r.dimname(1), _r.tensor(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// gather
static PyObject * THPVariable_gather(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gather(Tensor input, int64_t dim, Tensor index, *, bool sparse_grad=False, Tensor out=None)",
    "gather(Tensor input, Dimname dim, Tensor index, *, bool sparse_grad=False, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
        
        auto dispatch_gather = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.gather(dim, index, sparse_grad);
        };
        return wrap(dispatch_gather(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.toBool(3)));
      } else {
        // aten::gather.out(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_gather_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::gather_out(out, self, dim, index, sparse_grad);
        };
        return wrap(dispatch_gather_out(_r.tensor(4), _r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.toBool(3)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::gather.dimname(Tensor self, Dimname dim, Tensor index, *, bool sparse_grad=False) -> Tensor
        
        auto dispatch_gather = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index, bool sparse_grad) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.gather(dim, index, sparse_grad);
        };
        return wrap(dispatch_gather(_r.tensor(0), _r.dimname(1), _r.tensor(2), _r.toBool(3)));
      } else {
        // aten::gather.dimname_out(Tensor self, Dimname dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_gather_out = [](at::Tensor out, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, bool sparse_grad) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::gather_out(out, self, dim, index, sparse_grad);
        };
        return wrap(dispatch_gather_out(_r.tensor(4), _r.tensor(0), _r.dimname(1), _r.tensor(2), _r.toBool(3)));
      }
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
  static PythonArgParser parser({
    "addcmul(Tensor input, Scalar value, Tensor tensor1, Tensor tensor2, *, Tensor out=None)|deprecated",
    "addcmul(Tensor input, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // [deprecated] aten::addcmul(Tensor self, Scalar value, Tensor tensor1, Tensor tensor2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addcmul = [](at::Tensor out, const at::Tensor & self, const at::Scalar & value, const at::Tensor & tensor1, const at::Tensor & tensor2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addcmul(tensor1, tensor2, value);
        };
        return wrap(dispatch_addcmul(_r.tensor(4), _r.tensor(0), _r.scalar(1), _r.tensor(2), _r.tensor(3)));
      } else {
        // [deprecated] aten::addcmul(Tensor self, Scalar value, Tensor tensor1, Tensor tensor2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addcmul_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & value, const at::Tensor & tensor1, const at::Tensor & tensor2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addcmul_out(out, self, tensor1, tensor2, value);
        };
        return wrap(dispatch_addcmul_out(_r.tensor(4), _r.tensor(0), _r.scalar(1), _r.tensor(2), _r.tensor(3)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
        
        auto dispatch_addcmul = [](const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addcmul(tensor1, tensor2, value);
        };
        return wrap(dispatch_addcmul(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3)));
      } else {
        // aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addcmul_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addcmul_out(out, self, tensor1, tensor2, value);
        };
        return wrap(dispatch_addcmul_out(_r.tensor(4), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// triangular_solve
static PyObject * THPVariable_triangular_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_triangular_solve_out_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_triangular_solve_structseq();
  static PythonArgParser parser({
    "triangular_solve(Tensor input, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(5)) {
    // aten::triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)
    
    auto dispatch_triangular_solve = [](const at::Tensor & self, const at::Tensor & A, bool upper, bool transpose, bool unitriangular) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return self.triangular_solve(A, upper, transpose, unitriangular);
    };
    return wrap(NamedTuple1, dispatch_triangular_solve(_r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3), _r.toBool(4)));
  } else {
    // aten::triangular_solve.X(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False, *, Tensor(a!) X, Tensor(b!) M) -> (Tensor(a!) solution, Tensor(b!) cloned_coefficient)
    auto out = _r.tensorlist_n<2>(5);
    auto dispatch_triangular_solve_out = [](at::Tensor & X, at::Tensor & M, const at::Tensor & self, const at::Tensor & A, bool upper, bool transpose, bool unitriangular) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::triangular_solve_out(X, M, self, A, upper, transpose, unitriangular);
    };
    return wrap(NamedTuple, dispatch_triangular_solve_out(out[0], out[1], _r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3), _r.toBool(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// swapaxes
static PyObject * THPVariable_swapaxes(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "swapaxes(Tensor input, int64_t axis0, int64_t axis1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::swapaxes(Tensor(a) self, int axis0, int axis1) -> Tensor(a)
  
  auto dispatch_swapaxes = [](const at::Tensor & self, int64_t axis0, int64_t axis1) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.swapaxes(axis0, axis1);
  };
  return wrap(dispatch_swapaxes(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cholesky_inverse
static PyObject * THPVariable_cholesky_inverse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cholesky_inverse(Tensor input, bool upper=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::cholesky_inverse(Tensor self, bool upper=False) -> Tensor
    
    auto dispatch_cholesky_inverse = [](const at::Tensor & self, bool upper) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.cholesky_inverse(upper);
    };
    return wrap(dispatch_cholesky_inverse(_r.tensor(0), _r.toBool(1)));
  } else {
    // aten::cholesky_inverse.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_cholesky_inverse_out = [](at::Tensor out, const at::Tensor & self, bool upper) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::cholesky_inverse_out(out, self, upper);
    };
    return wrap(dispatch_cholesky_inverse_out(_r.tensor(2), _r.tensor(0), _r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// geqrf
static PyObject * THPVariable_geqrf(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_geqrf_out_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_geqrf_structseq();
  static PythonArgParser parser({
    "geqrf(Tensor input, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::geqrf(Tensor self) -> (Tensor a, Tensor tau)
    
    auto dispatch_geqrf = [](const at::Tensor & self) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return self.geqrf();
    };
    return wrap(NamedTuple1, dispatch_geqrf(_r.tensor(0)));
  } else {
    // aten::geqrf.a(Tensor self, *, Tensor(a!) a, Tensor(b!) tau) -> (Tensor(a!) a, Tensor(b!) tau)
    auto out = _r.tensorlist_n<2>(1);
    auto dispatch_geqrf_out = [](at::Tensor & a, at::Tensor & tau, const at::Tensor & self) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::geqrf_out(a, tau, self);
    };
    return wrap(NamedTuple, dispatch_geqrf_out(out[0], out[1], _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// orgqr
static PyObject * THPVariable_orgqr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "orgqr(Tensor input, Tensor input2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::orgqr(Tensor self, Tensor input2) -> Tensor
    
    auto dispatch_orgqr = [](const at::Tensor & self, const at::Tensor & input2) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.orgqr(input2);
    };
    return wrap(dispatch_orgqr(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::orgqr.out(Tensor self, Tensor input2, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_orgqr_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & input2) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::orgqr_out(out, self, input2);
    };
    return wrap(dispatch_orgqr_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ormqr
static PyObject * THPVariable_ormqr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ormqr(Tensor input, Tensor input2, Tensor input3, bool left=True, bool transpose=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(5)) {
    // aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor
    
    auto dispatch_ormqr = [](const at::Tensor & self, const at::Tensor & input2, const at::Tensor & input3, bool left, bool transpose) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.ormqr(input2, input3, left, transpose);
    };
    return wrap(dispatch_ormqr(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toBool(3), _r.toBool(4)));
  } else {
    // aten::ormqr.out(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_ormqr_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & input2, const at::Tensor & input3, bool left, bool transpose) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::ormqr_out(out, self, input2, input3, left, transpose);
    };
    return wrap(dispatch_ormqr_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toBool(3), _r.toBool(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// lu_solve
static PyObject * THPVariable_lu_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lu_solve(Tensor input, Tensor LU_data, Tensor LU_pivots, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor
    
    auto dispatch_lu_solve = [](const at::Tensor & self, const at::Tensor & LU_data, const at::Tensor & LU_pivots) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.lu_solve(LU_data, LU_pivots);
    };
    return wrap(dispatch_lu_solve(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
  } else {
    // aten::lu_solve.out(Tensor self, Tensor LU_data, Tensor LU_pivots, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_lu_solve_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & LU_data, const at::Tensor & LU_pivots) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::lu_solve_out(out, self, LU_data, LU_pivots);
    };
    return wrap(dispatch_lu_solve_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.tensor(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// lgamma
static PyObject * THPVariable_lgamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lgamma(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::lgamma(Tensor self) -> Tensor
    
    auto dispatch_lgamma = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.lgamma();
    };
    return wrap(dispatch_lgamma(_r.tensor(0)));
  } else {
    // aten::lgamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_lgamma_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::lgamma_out(out, self);
    };
    return wrap(dispatch_lgamma_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// digamma
static PyObject * THPVariable_digamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "digamma(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::digamma(Tensor self) -> Tensor
    
    auto dispatch_digamma = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.digamma();
    };
    return wrap(dispatch_digamma(_r.tensor(0)));
  } else {
    // aten::digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_digamma_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::digamma_out(out, self);
    };
    return wrap(dispatch_digamma_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// erfinv
static PyObject * THPVariable_erfinv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "erfinv(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::erfinv(Tensor self) -> Tensor
    
    auto dispatch_erfinv = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.erfinv();
    };
    return wrap(dispatch_erfinv(_r.tensor(0)));
  } else {
    // aten::erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_erfinv_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::erfinv_out(out, self);
    };
    return wrap(dispatch_erfinv_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sign
static PyObject * THPVariable_sign(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sign(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::sign(Tensor self) -> Tensor
    
    auto dispatch_sign = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.sign();
    };
    return wrap(dispatch_sign(_r.tensor(0)));
  } else {
    // aten::sign.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_sign_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::sign_out(out, self);
    };
    return wrap(dispatch_sign_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// dist
static PyObject * THPVariable_dist(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dist(Tensor input, Tensor other, Scalar p=2)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::dist(Tensor self, Tensor other, Scalar p=2) -> Tensor
  
  auto dispatch_dist = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & p) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.dist(other, p);
  };
  return wrap(dispatch_dist(_r.tensor(0), _r.tensor(1), _r.scalar(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// arctan2
static PyObject * THPVariable_arctan2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "arctan2(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::arctan2(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_arctan2 = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.arctan2(other);
    };
    return wrap(dispatch_arctan2(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::arctan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_arctan2_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::arctan2_out(out, self, other);
    };
    return wrap(dispatch_arctan2_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// histc
static PyObject * THPVariable_histc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "histc(Tensor input, int64_t bins=100, Scalar min=0, Scalar max=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor
    
    auto dispatch_histc = [](const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.histc(bins, min, max);
    };
    return wrap(dispatch_histc(_r.tensor(0), _r.toInt64(1), _r.scalar(2), _r.scalar(3)));
  } else {
    // aten::histc.out(Tensor self, int bins=100, Scalar min=0, Scalar max=0, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_histc_out = [](at::Tensor out, const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::histc_out(out, self, bins, min, max);
    };
    return wrap(dispatch_histc_out(_r.tensor(4), _r.tensor(0), _r.toInt64(1), _r.scalar(2), _r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// histogram
static PyObject * THPVariable_histogram(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_histogram_out_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_histogram_structseq();
  static PythonArgParser parser({
    "histogram(Tensor input, Tensor bins, *, Tensor? weight=None, bool density=False, TensorList[2] out=None)",
    "histogram(Tensor input, int64_t bins=100, *, ArrayRef<double>? range=None, Tensor? weight=None, bool density=False, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // aten::histogram.bins_tensor(Tensor self, Tensor bins, *, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor bin_edges)
        
        auto dispatch_histogram = [](const at::Tensor & self, const at::Tensor & bins, const ::std::optional<at::Tensor> & weight, bool density) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.histogram(bins, weight, density);
        };
        return wrap(NamedTuple1, dispatch_histogram(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.toBool(3)));
      } else {
        // aten::histogram.bins_tensor_out(Tensor self, Tensor bins, *, Tensor? weight=None, bool density=False, Tensor(a!) hist, Tensor(b!) bin_edges) -> (Tensor(a!) hist, Tensor(b!) bin_edges)
        auto out = _r.tensorlist_n<2>(4);
        auto dispatch_histogram_out = [](at::Tensor & hist, at::Tensor & bin_edges, const at::Tensor & self, const at::Tensor & bins, const ::std::optional<at::Tensor> & weight, bool density) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::histogram_out(hist, bin_edges, self, bins, weight, density);
        };
        return wrap(NamedTuple, dispatch_histogram_out(out[0], out[1], _r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.toBool(3)));
      }
    }
    case 1: {
      if (_r.isNone(5)) {
        // aten::histogram.bin_ct(Tensor self, int bins=100, *, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor bin_edges)
        
        auto dispatch_histogram = [](const at::Tensor & self, int64_t bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.histogram(bins, range, weight, density);
        };
        return wrap(NamedTuple1, dispatch_histogram(_r.tensor(0), _r.toInt64(1), _r.doublelistOptional(2), _r.optionalTensor(3), _r.toBool(4)));
      } else {
        // aten::histogram.bin_ct_out(Tensor self, int bins=100, *, float[]? range=None, Tensor? weight=None, bool density=False, Tensor(a!) hist, Tensor(b!) bin_edges) -> (Tensor(a!) hist, Tensor(b!) bin_edges)
        auto out = _r.tensorlist_n<2>(5);
        auto dispatch_histogram_out = [](at::Tensor & hist, at::Tensor & bin_edges, const at::Tensor & self, int64_t bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::histogram_out(hist, bin_edges, self, bins, range, weight, density);
        };
        return wrap(NamedTuple, dispatch_histogram_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.doublelistOptional(2), _r.optionalTensor(3), _r.toBool(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _histogramdd_bin_edges
static PyObject * THPVariable__histogramdd_bin_edges(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_histogramdd_bin_edges(Tensor input, IntArrayRef bins, *, ArrayRef<double>? range=None, Tensor? weight=None, bool density=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_histogramdd_bin_edges(Tensor self, int[] bins, *, float[]? range=None, Tensor? weight=None, bool density=False) -> Tensor[]
  
  auto dispatch__histogramdd_bin_edges = [](const at::Tensor & self, at::IntArrayRef bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_histogramdd_bin_edges(self, bins, range, weight, density);
  };
  return wrap(dispatch__histogramdd_bin_edges(_r.tensor(0), _r.intlist(1), _r.doublelistOptional(2), _r.optionalTensor(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// histogramdd
static PyObject * THPVariable_histogramdd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_histogramdd_structseq();
  static PythonArgParser parser({
    "histogramdd(Tensor input, int64_t bins, ArrayRef<double>? range=None, Tensor? weight=None, bool density=False)",
    "histogramdd(Tensor input, IntArrayRef bins, ArrayRef<double>? range=None, Tensor? weight=None, bool density=False)",
    "histogramdd(Tensor input, TensorList bins, ArrayRef<double>? range=None, Tensor? weight=None, bool density=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::histogramdd.int_bins(Tensor self, int bins, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor[] bin_edges)
      
      auto dispatch_histogramdd = [](const at::Tensor & self, int64_t bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density) -> ::std::tuple<at::Tensor,::std::vector<at::Tensor>> {
        pybind11::gil_scoped_release no_gil;
        return at::histogramdd(self, bins, range, weight, density);
      };
      return wrap(NamedTuple, dispatch_histogramdd(_r.tensor(0), _r.toInt64(1), _r.doublelistOptional(2), _r.optionalTensor(3), _r.toBool(4)));
    }
    case 1: {
      // aten::histogramdd(Tensor self, int[] bins, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor[] bin_edges)
      
      auto dispatch_histogramdd = [](const at::Tensor & self, at::IntArrayRef bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density) -> ::std::tuple<at::Tensor,::std::vector<at::Tensor>> {
        pybind11::gil_scoped_release no_gil;
        return at::histogramdd(self, bins, range, weight, density);
      };
      return wrap(NamedTuple, dispatch_histogramdd(_r.tensor(0), _r.intlist(1), _r.doublelistOptional(2), _r.optionalTensor(3), _r.toBool(4)));
    }
    case 2: {
      // aten::histogramdd.TensorList_bins(Tensor self, Tensor[] bins, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor[] bin_edges)
      
      auto dispatch_histogramdd = [](const at::Tensor & self, at::TensorList bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density) -> ::std::tuple<at::Tensor,::std::vector<at::Tensor>> {
        pybind11::gil_scoped_release no_gil;
        return at::histogramdd(self, bins, range, weight, density);
      };
      return wrap(NamedTuple, dispatch_histogramdd(_r.tensor(0), _r.tensorlist(1), _r.doublelistOptional(2), _r.optionalTensor(3), _r.toBool(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fmax
static PyObject * THPVariable_fmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fmax(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::fmax(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_fmax = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.fmax(other);
    };
    return wrap(dispatch_fmax(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::fmax.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fmax_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fmax_out(out, self, other);
    };
    return wrap(dispatch_fmax_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// maximum
static PyObject * THPVariable_maximum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "maximum(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::maximum(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_maximum = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.maximum(other);
    };
    return wrap(dispatch_maximum(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::maximum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_maximum_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::maximum_out(out, self, other);
    };
    return wrap(dispatch_maximum_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// minimum
static PyObject * THPVariable_minimum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "minimum(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::minimum(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_minimum = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.minimum(other);
    };
    return wrap(dispatch_minimum(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::minimum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_minimum_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::minimum_out(out, self, other);
    };
    return wrap(dispatch_minimum_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// nanquantile
static PyObject * THPVariable_nanquantile(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nanquantile(Tensor input, Tensor q, int64_t? dim=None, bool keepdim=False, *, c10::string_view interpolation=\"linear\", Tensor out=None)",
    "nanquantile(Tensor input, double q, int64_t? dim=None, bool keepdim=False, *, c10::string_view interpolation=\"linear\", Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(5)) {
        // aten::nanquantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation='linear') -> Tensor
        
        auto dispatch_nanquantile = [](const at::Tensor & self, const at::Tensor & q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.nanquantile(q, dim, keepdim, interpolation);
        };
        return wrap(dispatch_nanquantile(_r.tensor(0), _r.tensor(1), _r.toInt64Optional(2), _r.toBool(3), _r.stringView(4)));
      } else {
        // aten::nanquantile.out(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation='linear', Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_nanquantile_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::nanquantile_out(out, self, q, dim, keepdim, interpolation);
        };
        return wrap(dispatch_nanquantile_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.toInt64Optional(2), _r.toBool(3), _r.stringView(4)));
      }
    }
    case 1: {
      if (_r.isNone(5)) {
        // aten::nanquantile.scalar(Tensor self, float q, int? dim=None, bool keepdim=False, *, str interpolation='linear') -> Tensor
        
        auto dispatch_nanquantile = [](const at::Tensor & self, double q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.nanquantile(q, dim, keepdim, interpolation);
        };
        return wrap(dispatch_nanquantile(_r.tensor(0), _r.toDouble(1), _r.toInt64Optional(2), _r.toBool(3), _r.stringView(4)));
      } else {
        // aten::nanquantile.scalar_out(Tensor self, float q, int? dim=None, bool keepdim=False, *, str interpolation='linear', Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_nanquantile_out = [](at::Tensor out, const at::Tensor & self, double q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::nanquantile_out(out, self, q, dim, keepdim, interpolation);
        };
        return wrap(dispatch_nanquantile_out(_r.tensor(5), _r.tensor(0), _r.toDouble(1), _r.toInt64Optional(2), _r.toBool(3), _r.stringView(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// msort
static PyObject * THPVariable_msort(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "msort(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::msort(Tensor self) -> Tensor
    
    auto dispatch_msort = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.msort();
    };
    return wrap(dispatch_msort(_r.tensor(0)));
  } else {
    // aten::msort.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_msort_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::msort_out(out, self);
    };
    return wrap(dispatch_msort_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// pow
static PyObject * THPVariable_pow(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pow(Tensor input, Tensor exponent, *, Tensor out=None)",
    "pow(Scalar self, Tensor exponent, *, Tensor out=None)",
    "pow(Tensor input, Scalar exponent, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
        
        auto dispatch_pow = [](const at::Tensor & self, const at::Tensor & exponent) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.pow(exponent);
        };
        return wrap(dispatch_pow(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::pow.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_pow_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & exponent) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::pow_out(out, self, exponent);
        };
        return wrap(dispatch_pow_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor
        
        auto dispatch_pow = [](const at::Scalar & self, const at::Tensor & exponent) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::pow(self, exponent);
        };
        return wrap(dispatch_pow(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_pow_out = [](at::Tensor out, const at::Scalar & self, const at::Tensor & exponent) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::pow_out(out, self, exponent);
        };
        return wrap(dispatch_pow_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
        
        auto dispatch_pow = [](const at::Tensor & self, const at::Scalar & exponent) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.pow(exponent);
        };
        return wrap(dispatch_pow(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_pow_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & exponent) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::pow_out(out, self, exponent);
        };
        return wrap(dispatch_pow_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// float_power
static PyObject * THPVariable_float_power(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "float_power(Tensor input, Tensor exponent, *, Tensor out=None)",
    "float_power(Scalar self, Tensor exponent, *, Tensor out=None)",
    "float_power(Tensor input, Scalar exponent, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::float_power.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
        
        auto dispatch_float_power = [](const at::Tensor & self, const at::Tensor & exponent) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.float_power(exponent);
        };
        return wrap(dispatch_float_power(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::float_power.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_float_power_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & exponent) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::float_power_out(out, self, exponent);
        };
        return wrap(dispatch_float_power_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::float_power.Scalar(Scalar self, Tensor exponent) -> Tensor
        
        auto dispatch_float_power = [](const at::Scalar & self, const at::Tensor & exponent) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::float_power(self, exponent);
        };
        return wrap(dispatch_float_power(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::float_power.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_float_power_out = [](at::Tensor out, const at::Scalar & self, const at::Tensor & exponent) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::float_power_out(out, self, exponent);
        };
        return wrap(dispatch_float_power_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::float_power.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
        
        auto dispatch_float_power = [](const at::Tensor & self, const at::Scalar & exponent) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.float_power(exponent);
        };
        return wrap(dispatch_float_power(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::float_power.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_float_power_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & exponent) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::float_power_out(out, self, exponent);
        };
        return wrap(dispatch_float_power_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_add
static PyObject * THPVariable__foreach_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_add(TensorList self, Scalar scalar)",
    "_foreach_add(TensorList self, ScalarList scalars)",
    "_foreach_add(TensorList self, Tensor other, *, Scalar alpha=1)",
    "_foreach_add(TensorList self, TensorList other, *, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_add.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
      
      auto dispatch__foreach_add = [](at::TensorList self, const at::Scalar & scalar) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_add(self, scalar);
      };
      return wrap(dispatch__foreach_add(_r.tensorlist(0), _r.scalar(1)));
    }
    case 1: {
      // aten::_foreach_add.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
      
      auto dispatch__foreach_add = [](at::TensorList self, at::ArrayRef<at::Scalar> scalars) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_add(self, scalars);
      };
      return wrap(dispatch__foreach_add(_r.tensorlist(0), _r.scalarlist(1)));
    }
    case 2: {
      // aten::_foreach_add.Tensor(Tensor[] self, Tensor other, *, Scalar alpha=1) -> Tensor[]
      
      auto dispatch__foreach_add = [](at::TensorList self, const at::Tensor & other, const at::Scalar & alpha) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_add(self, other, alpha);
      };
      return wrap(dispatch__foreach_add(_r.tensorlist(0), _r.tensor(1), _r.scalar(2)));
    }
    case 3: {
      // aten::_foreach_add.List(Tensor[] self, Tensor[] other, *, Scalar alpha=1) -> Tensor[]
      
      auto dispatch__foreach_add = [](at::TensorList self, at::TensorList other, const at::Scalar & alpha) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_add(self, other, alpha);
      };
      return wrap(dispatch__foreach_add(_r.tensorlist(0), _r.tensorlist(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_add_
static PyObject * THPVariable__foreach_add_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_add_(TensorList self, Scalar scalar)",
    "_foreach_add_(TensorList self, ScalarList scalars)",
    "_foreach_add_(TensorList self, Tensor other, *, Scalar alpha=1)",
    "_foreach_add_(TensorList self, TensorList other, *, Scalar alpha=1)",
  }, /*traceable=*/false);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_add_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
      
      auto dispatch__foreach_add_ = [](at::TensorList self, const at::Scalar & scalar) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_add_(self, scalar);
      };
      dispatch__foreach_add_(_r.tensorlist(0), _r.scalar(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 1: {
      // aten::_foreach_add_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
      
      auto dispatch__foreach_add_ = [](at::TensorList self, at::ArrayRef<at::Scalar> scalars) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_add_(self, scalars);
      };
      dispatch__foreach_add_(_r.tensorlist(0), _r.scalarlist(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 2: {
      // aten::_foreach_add_.Tensor(Tensor(a!)[] self, Tensor other, *, Scalar alpha=1) -> ()
      
      auto dispatch__foreach_add_ = [](at::TensorList self, const at::Tensor & other, const at::Scalar & alpha) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_add_(self, other, alpha);
      };
      dispatch__foreach_add_(_r.tensorlist(0), _r.tensor(1), _r.scalar(2));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 3: {
      // aten::_foreach_add_.List(Tensor(a!)[] self, Tensor[] other, *, Scalar alpha=1) -> ()
      
      auto dispatch__foreach_add_ = [](at::TensorList self, at::TensorList other, const at::Scalar & alpha) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_add_(self, other, alpha);
      };
      dispatch__foreach_add_(_r.tensorlist(0), _r.tensorlist(1), _r.scalar(2));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_sub
static PyObject * THPVariable__foreach_sub(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_sub(TensorList self, Scalar scalar)",
    "_foreach_sub(TensorList self, ScalarList scalars)",
    "_foreach_sub(TensorList self, TensorList other, *, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_sub.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
      
      auto dispatch__foreach_sub = [](at::TensorList self, const at::Scalar & scalar) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_sub(self, scalar);
      };
      return wrap(dispatch__foreach_sub(_r.tensorlist(0), _r.scalar(1)));
    }
    case 1: {
      // aten::_foreach_sub.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
      
      auto dispatch__foreach_sub = [](at::TensorList self, at::ArrayRef<at::Scalar> scalars) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_sub(self, scalars);
      };
      return wrap(dispatch__foreach_sub(_r.tensorlist(0), _r.scalarlist(1)));
    }
    case 2: {
      // aten::_foreach_sub.List(Tensor[] self, Tensor[] other, *, Scalar alpha=1) -> Tensor[]
      
      auto dispatch__foreach_sub = [](at::TensorList self, at::TensorList other, const at::Scalar & alpha) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_sub(self, other, alpha);
      };
      return wrap(dispatch__foreach_sub(_r.tensorlist(0), _r.tensorlist(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_sub_
static PyObject * THPVariable__foreach_sub_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_sub_(TensorList self, Scalar scalar)",
    "_foreach_sub_(TensorList self, ScalarList scalars)",
    "_foreach_sub_(TensorList self, TensorList other, *, Scalar alpha=1)",
  }, /*traceable=*/false);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_sub_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
      
      auto dispatch__foreach_sub_ = [](at::TensorList self, const at::Scalar & scalar) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_sub_(self, scalar);
      };
      dispatch__foreach_sub_(_r.tensorlist(0), _r.scalar(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 1: {
      // aten::_foreach_sub_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
      
      auto dispatch__foreach_sub_ = [](at::TensorList self, at::ArrayRef<at::Scalar> scalars) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_sub_(self, scalars);
      };
      dispatch__foreach_sub_(_r.tensorlist(0), _r.scalarlist(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 2: {
      // aten::_foreach_sub_.List(Tensor(a!)[] self, Tensor[] other, *, Scalar alpha=1) -> ()
      
      auto dispatch__foreach_sub_ = [](at::TensorList self, at::TensorList other, const at::Scalar & alpha) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_sub_(self, other, alpha);
      };
      dispatch__foreach_sub_(_r.tensorlist(0), _r.tensorlist(1), _r.scalar(2));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_clamp_max
static PyObject * THPVariable__foreach_clamp_max(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_clamp_max(TensorList self, Scalar scalar)",
    "_foreach_clamp_max(TensorList self, ScalarList scalars)",
    "_foreach_clamp_max(TensorList self, TensorList other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_clamp_max.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
      
      auto dispatch__foreach_clamp_max = [](at::TensorList self, const at::Scalar & scalar) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_clamp_max(self, scalar);
      };
      return wrap(dispatch__foreach_clamp_max(_r.tensorlist(0), _r.scalar(1)));
    }
    case 1: {
      // aten::_foreach_clamp_max.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
      
      auto dispatch__foreach_clamp_max = [](at::TensorList self, at::ArrayRef<at::Scalar> scalars) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_clamp_max(self, scalars);
      };
      return wrap(dispatch__foreach_clamp_max(_r.tensorlist(0), _r.scalarlist(1)));
    }
    case 2: {
      // aten::_foreach_clamp_max.List(Tensor[] self, Tensor[] other) -> Tensor[]
      
      auto dispatch__foreach_clamp_max = [](at::TensorList self, at::TensorList other) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_clamp_max(self, other);
      };
      return wrap(dispatch__foreach_clamp_max(_r.tensorlist(0), _r.tensorlist(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_clamp_max_
static PyObject * THPVariable__foreach_clamp_max_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_clamp_max_(TensorList self, Scalar scalar)",
    "_foreach_clamp_max_(TensorList self, ScalarList scalars)",
    "_foreach_clamp_max_(TensorList self, TensorList other)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_clamp_max_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
      
      auto dispatch__foreach_clamp_max_ = [](at::TensorList self, const at::Scalar & scalar) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_clamp_max_(self, scalar);
      };
      dispatch__foreach_clamp_max_(_r.tensorlist(0), _r.scalar(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 1: {
      // aten::_foreach_clamp_max_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
      
      auto dispatch__foreach_clamp_max_ = [](at::TensorList self, at::ArrayRef<at::Scalar> scalars) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_clamp_max_(self, scalars);
      };
      dispatch__foreach_clamp_max_(_r.tensorlist(0), _r.scalarlist(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 2: {
      // aten::_foreach_clamp_max_.List(Tensor(a!)[] self, Tensor[] other) -> ()
      
      auto dispatch__foreach_clamp_max_ = [](at::TensorList self, at::TensorList other) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_clamp_max_(self, other);
      };
      dispatch__foreach_clamp_max_(_r.tensorlist(0), _r.tensorlist(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_acos
static PyObject * THPVariable__foreach_acos(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_acos(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_acos(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_acos = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_acos(self);
  };
  return wrap(dispatch__foreach_acos(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_acos_
static PyObject * THPVariable__foreach_acos_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_acos_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_acos_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_acos_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_acos_(self);
  };
  dispatch__foreach_acos_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_asin
static PyObject * THPVariable__foreach_asin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_asin(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_asin(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_asin = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_asin(self);
  };
  return wrap(dispatch__foreach_asin(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_asin_
static PyObject * THPVariable__foreach_asin_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_asin_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_asin_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_asin_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_asin_(self);
  };
  dispatch__foreach_asin_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_cos
static PyObject * THPVariable__foreach_cos(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_cos(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_cos(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_cos = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_cos(self);
  };
  return wrap(dispatch__foreach_cos(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_cos_
static PyObject * THPVariable__foreach_cos_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_cos_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_cos_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_cos_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_cos_(self);
  };
  dispatch__foreach_cos_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_cosh
static PyObject * THPVariable__foreach_cosh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_cosh(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_cosh(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_cosh = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_cosh(self);
  };
  return wrap(dispatch__foreach_cosh(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_cosh_
static PyObject * THPVariable__foreach_cosh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_cosh_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_cosh_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_cosh_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_cosh_(self);
  };
  dispatch__foreach_cosh_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_exp
static PyObject * THPVariable__foreach_exp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_exp(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_exp(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_exp = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_exp(self);
  };
  return wrap(dispatch__foreach_exp(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_exp_
static PyObject * THPVariable__foreach_exp_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_exp_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_exp_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_exp_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_exp_(self);
  };
  dispatch__foreach_exp_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_floor
static PyObject * THPVariable__foreach_floor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_floor(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_floor(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_floor = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_floor(self);
  };
  return wrap(dispatch__foreach_floor(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_floor_
static PyObject * THPVariable__foreach_floor_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_floor_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_floor_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_floor_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_floor_(self);
  };
  dispatch__foreach_floor_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_log1p
static PyObject * THPVariable__foreach_log1p(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_log1p(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_log1p(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_log1p = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_log1p(self);
  };
  return wrap(dispatch__foreach_log1p(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_log1p_
static PyObject * THPVariable__foreach_log1p_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_log1p_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_log1p_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_log1p_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_log1p_(self);
  };
  dispatch__foreach_log1p_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_log2
static PyObject * THPVariable__foreach_log2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_log2(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_log2(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_log2 = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_log2(self);
  };
  return wrap(dispatch__foreach_log2(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_log2_
static PyObject * THPVariable__foreach_log2_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_log2_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_log2_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_log2_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_log2_(self);
  };
  dispatch__foreach_log2_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_neg
static PyObject * THPVariable__foreach_neg(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_neg(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_neg(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_neg = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_neg(self);
  };
  return wrap(dispatch__foreach_neg(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_neg_
static PyObject * THPVariable__foreach_neg_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_neg_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_neg_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_neg_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_neg_(self);
  };
  dispatch__foreach_neg_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_norm
static PyObject * THPVariable__foreach_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_norm(TensorList self, Scalar ord=2, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_norm.Scalar(Tensor[] self, Scalar ord=2, ScalarType? dtype=None) -> Tensor[]
  
  auto dispatch__foreach_norm = [](at::TensorList self, const at::Scalar & ord, ::std::optional<at::ScalarType> dtype) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_norm(self, ord, dtype);
  };
  return wrap(dispatch__foreach_norm(_r.tensorlist(0), _r.scalar(1), _r.scalartypeOptional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_sigmoid
static PyObject * THPVariable__foreach_sigmoid(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_sigmoid(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_sigmoid(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_sigmoid = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_sigmoid(self);
  };
  return wrap(dispatch__foreach_sigmoid(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_sigmoid_
static PyObject * THPVariable__foreach_sigmoid_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_sigmoid_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_sigmoid_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_sigmoid_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_sigmoid_(self);
  };
  dispatch__foreach_sigmoid_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_sign
static PyObject * THPVariable__foreach_sign(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_sign(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_sign(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_sign = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_sign(self);
  };
  return wrap(dispatch__foreach_sign(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_sign_
static PyObject * THPVariable__foreach_sign_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_sign_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_sign_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_sign_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_sign_(self);
  };
  dispatch__foreach_sign_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_sqrt
static PyObject * THPVariable__foreach_sqrt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_sqrt(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_sqrt(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_sqrt = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_sqrt(self);
  };
  return wrap(dispatch__foreach_sqrt(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_sqrt_
static PyObject * THPVariable__foreach_sqrt_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_sqrt_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_sqrt_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_sqrt_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_sqrt_(self);
  };
  dispatch__foreach_sqrt_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_tanh
static PyObject * THPVariable__foreach_tanh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_tanh(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_tanh(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_tanh = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_tanh(self);
  };
  return wrap(dispatch__foreach_tanh(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_tanh_
static PyObject * THPVariable__foreach_tanh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_tanh_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_tanh_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_tanh_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_tanh_(self);
  };
  dispatch__foreach_tanh_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _convert_indices_from_csr_to_coo
static PyObject * THPVariable__convert_indices_from_csr_to_coo(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_convert_indices_from_csr_to_coo(Tensor crow_indices, Tensor col_indices, *, bool out_int32=False, bool transpose=False, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::_convert_indices_from_csr_to_coo(Tensor crow_indices, Tensor col_indices, *, bool out_int32=False, bool transpose=False) -> Tensor
    
    auto dispatch__convert_indices_from_csr_to_coo = [](const at::Tensor & crow_indices, const at::Tensor & col_indices, bool out_int32, bool transpose) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_convert_indices_from_csr_to_coo(crow_indices, col_indices, out_int32, transpose);
    };
    return wrap(dispatch__convert_indices_from_csr_to_coo(_r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3)));
  } else {
    // aten::_convert_indices_from_csr_to_coo.out(Tensor crow_indices, Tensor col_indices, *, bool out_int32=False, bool transpose=False, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__convert_indices_from_csr_to_coo_out = [](at::Tensor out, const at::Tensor & crow_indices, const at::Tensor & col_indices, bool out_int32, bool transpose) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_convert_indices_from_csr_to_coo_out(out, crow_indices, col_indices, out_int32, transpose);
    };
    return wrap(dispatch__convert_indices_from_csr_to_coo_out(_r.tensor(4), _r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mkldnn_adaptive_avg_pool2d
static PyObject * THPVariable_mkldnn_adaptive_avg_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mkldnn_adaptive_avg_pool2d(Tensor input, IntArrayRef[2] output_size, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::mkldnn_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
    
    auto dispatch_mkldnn_adaptive_avg_pool2d = [](const at::Tensor & self, at::IntArrayRef output_size) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::mkldnn_adaptive_avg_pool2d(self, output_size);
    };
    return wrap(dispatch_mkldnn_adaptive_avg_pool2d(_r.tensor(0), _r.intlist(1)));
  } else {
    // aten::mkldnn_adaptive_avg_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_mkldnn_adaptive_avg_pool2d_out = [](at::Tensor out, const at::Tensor & self, at::IntArrayRef output_size) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::mkldnn_adaptive_avg_pool2d_out(out, self, output_size);
    };
    return wrap(dispatch_mkldnn_adaptive_avg_pool2d_out(_r.tensor(2), _r.tensor(0), _r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// column_stack
static PyObject * THPVariable_column_stack(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "column_stack(TensorList tensors, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::column_stack(Tensor[] tensors) -> Tensor
    
    auto dispatch_column_stack = [](at::TensorList tensors) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::column_stack(tensors);
    };
    return wrap(dispatch_column_stack(_r.tensorlist(0)));
  } else {
    // aten::column_stack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_column_stack_out = [](at::Tensor out, at::TensorList tensors) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::column_stack_out(out, tensors);
    };
    return wrap(dispatch_column_stack_out(_r.tensor(1), _r.tensorlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// isinf
static PyObject * THPVariable_isinf(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "isinf(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::isinf(Tensor self) -> Tensor
  
  auto dispatch_isinf = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.isinf();
  };
  return wrap(dispatch_isinf(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// isneginf
static PyObject * THPVariable_isneginf(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "isneginf(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::isneginf(Tensor self) -> Tensor
    
    auto dispatch_isneginf = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.isneginf();
    };
    return wrap(dispatch_isneginf(_r.tensor(0)));
  } else {
    // aten::isneginf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_isneginf_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::isneginf_out(out, self);
    };
    return wrap(dispatch_isneginf_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _linalg_slogdet
static PyObject * THPVariable__linalg_slogdet(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get__linalg_slogdet_structseq();
  static PyTypeObject* NamedTuple1 = generated::get__linalg_slogdet_out_structseq();
  static PythonArgParser parser({
    "_linalg_slogdet(Tensor A, *, TensorList[4] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::_linalg_slogdet(Tensor A) -> (Tensor sign, Tensor logabsdet, Tensor LU, Tensor pivots)
    
    auto dispatch__linalg_slogdet = [](const at::Tensor & A) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::_linalg_slogdet(A);
    };
    return wrap(NamedTuple, dispatch__linalg_slogdet(_r.tensor(0)));
  } else {
    // aten::_linalg_slogdet.sign(Tensor A, *, Tensor(a!) sign, Tensor(b!) logabsdet, Tensor(c!) LU, Tensor(d!) pivots) -> (Tensor(a!) sign, Tensor(b!) logabsdet, Tensor(c!) LU, Tensor(d!) pivots)
    auto out = _r.tensorlist_n<4>(1);
    auto dispatch__linalg_slogdet_out = [](at::Tensor & sign, at::Tensor & logabsdet, at::Tensor & LU, at::Tensor & pivots, const at::Tensor & A) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::_linalg_slogdet_out(sign, logabsdet, LU, pivots, A);
    };
    return wrap(NamedTuple1, dispatch__linalg_slogdet_out(out[0], out[1], out[2], out[3], _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// slogdet
static PyObject * THPVariable_slogdet(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_slogdet_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_slogdet_out_structseq();
  static PythonArgParser parser({
    "slogdet(Tensor input, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)
    
    auto dispatch_slogdet = [](const at::Tensor & self) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return self.slogdet();
    };
    return wrap(NamedTuple, dispatch_slogdet(_r.tensor(0)));
  } else {
    // aten::slogdet.out(Tensor self, *, Tensor(a!) sign, Tensor(b!) logabsdet) -> (Tensor(a!) sign, Tensor(b!) logabsdet)
    auto out = _r.tensorlist_n<2>(1);
    auto dispatch_slogdet_out = [](at::Tensor & sign, at::Tensor & logabsdet, const at::Tensor & self) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::slogdet_out(sign, logabsdet, self);
    };
    return wrap(NamedTuple1, dispatch_slogdet_out(out[0], out[1], _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// inverse
static PyObject * THPVariable_inverse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "inverse(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::inverse(Tensor self) -> Tensor
    
    auto dispatch_inverse = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.inverse();
    };
    return wrap(dispatch_inverse(_r.tensor(0)));
  } else {
    // aten::inverse.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_inverse_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::inverse_out(out, self);
    };
    return wrap(dispatch_inverse_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// inner
static PyObject * THPVariable_inner(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "inner(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::inner(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_inner = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.inner(other);
    };
    return wrap(dispatch_inner(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::inner.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_inner_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::inner_out(out, self, other);
    };
    return wrap(dispatch_inner_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _linalg_solve_ex
static PyObject * THPVariable__linalg_solve_ex(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get__linalg_solve_ex_structseq();
  static PyTypeObject* NamedTuple1 = generated::get__linalg_solve_ex_out_structseq();
  static PythonArgParser parser({
    "_linalg_solve_ex(Tensor A, Tensor B, *, bool left=True, bool check_errors=False, TensorList[4] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::_linalg_solve_ex(Tensor A, Tensor B, *, bool left=True, bool check_errors=False) -> (Tensor result, Tensor LU, Tensor pivots, Tensor info)
    
    auto dispatch__linalg_solve_ex = [](const at::Tensor & A, const at::Tensor & B, bool left, bool check_errors) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::_linalg_solve_ex(A, B, left, check_errors);
    };
    return wrap(NamedTuple, dispatch__linalg_solve_ex(_r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3)));
  } else {
    // aten::_linalg_solve_ex.result(Tensor A, Tensor B, *, bool left=True, bool check_errors=False, Tensor(a!) result, Tensor(b!) LU, Tensor(c!) pivots, Tensor(d!) info) -> (Tensor(a!) result, Tensor(b!) LU, Tensor(c!) pivots, Tensor(d!) info)
    auto out = _r.tensorlist_n<4>(4);
    auto dispatch__linalg_solve_ex_out = [](at::Tensor & result, at::Tensor & LU, at::Tensor & pivots, at::Tensor & info, const at::Tensor & A, const at::Tensor & B, bool left, bool check_errors) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::_linalg_solve_ex_out(result, LU, pivots, info, A, B, left, check_errors);
    };
    return wrap(NamedTuple1, dispatch__linalg_solve_ex_out(out[0], out[1], out[2], out[3], _r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _test_parallel_materialize
static PyObject * THPVariable__test_parallel_materialize(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_test_parallel_materialize(Tensor input, int64_t num_parallel, bool skip_first=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_test_parallel_materialize(Tensor self, int num_parallel, bool skip_first=False) -> Tensor
  
  auto dispatch__test_parallel_materialize = [](const at::Tensor & self, int64_t num_parallel, bool skip_first) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_test_parallel_materialize(self, num_parallel, skip_first);
  };
  return wrap(dispatch__test_parallel_materialize(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _test_autograd_multiple_dispatch
static PyObject * THPVariable__test_autograd_multiple_dispatch(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_test_autograd_multiple_dispatch(Tensor input)",
    "_test_autograd_multiple_dispatch(Tensor input, bool b)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_test_autograd_multiple_dispatch.fullcoverage(Tensor self) -> Tensor
      
      auto dispatch__test_autograd_multiple_dispatch = [](const at::Tensor & self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_test_autograd_multiple_dispatch(self);
      };
      return wrap(dispatch__test_autograd_multiple_dispatch(_r.tensor(0)));
    }
    case 1: {
      // aten::_test_autograd_multiple_dispatch.ntonly(Tensor self, bool b) -> Tensor
      
      auto dispatch__test_autograd_multiple_dispatch = [](const at::Tensor & self, bool b) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_test_autograd_multiple_dispatch(self, b);
      };
      return wrap(dispatch__test_autograd_multiple_dispatch(_r.tensor(0), _r.toBool(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _test_autograd_multiple_dispatch_view_copy
static PyObject * THPVariable__test_autograd_multiple_dispatch_view_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_test_autograd_multiple_dispatch_view_copy(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::_test_autograd_multiple_dispatch_view_copy(Tensor self) -> Tensor
    
    auto dispatch__test_autograd_multiple_dispatch_view_copy = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_test_autograd_multiple_dispatch_view_copy(self);
    };
    return wrap(dispatch__test_autograd_multiple_dispatch_view_copy(_r.tensor(0)));
  } else {
    // aten::_test_autograd_multiple_dispatch_view_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__test_autograd_multiple_dispatch_view_copy_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_test_autograd_multiple_dispatch_view_copy_out(out, self);
    };
    return wrap(dispatch__test_autograd_multiple_dispatch_view_copy_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// segment_reduce
static PyObject * THPVariable_segment_reduce(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "segment_reduce(Tensor data, c10::string_view reduce, *, Tensor? lengths=None, Tensor? indices=None, Tensor? offsets=None, int64_t axis=0, bool unsafe=False, Scalar? initial=None)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::segment_reduce(Tensor data, str reduce, *, Tensor? lengths=None, Tensor? indices=None, Tensor? offsets=None, int axis=0, bool unsafe=False, Scalar? initial=None) -> Tensor
  
  auto dispatch_segment_reduce = [](const at::Tensor & data, c10::string_view reduce, const ::std::optional<at::Tensor> & lengths, const ::std::optional<at::Tensor> & indices, const ::std::optional<at::Tensor> & offsets, int64_t axis, bool unsafe, const ::std::optional<at::Scalar> & initial) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::segment_reduce(data, reduce, lengths, indices, offsets, axis, unsafe, initial);
  };
  return wrap(dispatch_segment_reduce(_r.tensor(0), _r.stringView(1), _r.optionalTensor(2), _r.optionalTensor(3), _r.optionalTensor(4), _r.toInt64(5), _r.toBool(6), _r.scalarOptional(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _conj_copy
static PyObject * THPVariable__conj_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_conj_copy(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::_conj_copy(Tensor self) -> Tensor
    
    auto dispatch__conj_copy = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_conj_copy(self);
    };
    return wrap(dispatch__conj_copy(_r.tensor(0)));
  } else {
    // aten::_conj_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__conj_copy_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_conj_copy_out(out, self);
    };
    return wrap(dispatch__conj_copy_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sparse_broadcast_to_copy
static PyObject * THPVariable__sparse_broadcast_to_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_broadcast_to_copy(Tensor input, IntArrayRef size, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::_sparse_broadcast_to_copy(Tensor self, int[] size) -> Tensor
    
    auto dispatch__sparse_broadcast_to_copy = [](const at::Tensor & self, at::IntArrayRef size) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_sparse_broadcast_to_copy(self, size);
    };
    return wrap(dispatch__sparse_broadcast_to_copy(_r.tensor(0), _r.intlist(1)));
  } else {
    // aten::_sparse_broadcast_to_copy.out(Tensor self, int[] size, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__sparse_broadcast_to_copy_out = [](at::Tensor out, const at::Tensor & self, at::IntArrayRef size) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_sparse_broadcast_to_copy_out(out, self, size);
    };
    return wrap(dispatch__sparse_broadcast_to_copy_out(_r.tensor(2), _r.tensor(0), _r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// select_copy
static PyObject * THPVariable_select_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "select_copy(Tensor input, int64_t dim, SymInt index, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::select_copy.int(Tensor self, int dim, SymInt index) -> Tensor
    
    auto dispatch_select_copy = [](const at::Tensor & self, int64_t dim, c10::SymInt index) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::select_copy_symint(self, dim, index);
    };
    return wrap(dispatch_select_copy(_r.tensor(0), _r.toInt64(1), _r.toSymInt(2)));
  } else {
    // aten::select_copy.int_out(Tensor self, int dim, SymInt index, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_select_copy_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, c10::SymInt index) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::select_copy_symint_out(out, self, dim, index);
    };
    return wrap(dispatch_select_copy_out(_r.tensor(3), _r.tensor(0), _r.toInt64(1), _r.toSymInt(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// squeeze_copy
static PyObject * THPVariable_squeeze_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "squeeze_copy(Tensor input, *, Tensor out=None)",
    "squeeze_copy(Tensor input, int64_t dim, *, Tensor out=None)",
    "squeeze_copy(Tensor input, IntArrayRef dim, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(1)) {
        // aten::squeeze_copy(Tensor self) -> Tensor
        
        auto dispatch_squeeze_copy = [](const at::Tensor & self) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::squeeze_copy(self);
        };
        return wrap(dispatch_squeeze_copy(_r.tensor(0)));
      } else {
        // aten::squeeze_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_squeeze_copy_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::squeeze_copy_out(out, self);
        };
        return wrap(dispatch_squeeze_copy_out(_r.tensor(1), _r.tensor(0)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::squeeze_copy.dim(Tensor self, int dim) -> Tensor
        
        auto dispatch_squeeze_copy = [](const at::Tensor & self, int64_t dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::squeeze_copy(self, dim);
        };
        return wrap(dispatch_squeeze_copy(_r.tensor(0), _r.toInt64(1)));
      } else {
        // aten::squeeze_copy.dim_out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_squeeze_copy_out = [](at::Tensor out, const at::Tensor & self, int64_t dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::squeeze_copy_out(out, self, dim);
        };
        return wrap(dispatch_squeeze_copy_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::squeeze_copy.dims(Tensor self, int[] dim) -> Tensor
        
        auto dispatch_squeeze_copy = [](const at::Tensor & self, at::IntArrayRef dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::squeeze_copy(self, dim);
        };
        return wrap(dispatch_squeeze_copy(_r.tensor(0), _r.intlist(1)));
      } else {
        // aten::squeeze_copy.dims_out(Tensor self, int[] dim, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_squeeze_copy_out = [](at::Tensor out, const at::Tensor & self, at::IntArrayRef dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::squeeze_copy_out(out, self, dim);
        };
        return wrap(dispatch_squeeze_copy_out(_r.tensor(2), _r.tensor(0), _r.intlist(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// transpose_copy
static PyObject * THPVariable_transpose_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "transpose_copy(Tensor input, int64_t dim0, int64_t dim1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::transpose_copy.int(Tensor self, int dim0, int dim1) -> Tensor
    
    auto dispatch_transpose_copy = [](const at::Tensor & self, int64_t dim0, int64_t dim1) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::transpose_copy(self, dim0, dim1);
    };
    return wrap(dispatch_transpose_copy(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
  } else {
    // aten::transpose_copy.int_out(Tensor self, int dim0, int dim1, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_transpose_copy_out = [](at::Tensor out, const at::Tensor & self, int64_t dim0, int64_t dim1) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::transpose_copy_out(out, self, dim0, dim1);
    };
    return wrap(dispatch_transpose_copy_out(_r.tensor(3), _r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// values_copy
static PyObject * THPVariable_values_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "values_copy(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::values_copy(Tensor self) -> Tensor
    
    auto dispatch_values_copy = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::values_copy(self);
    };
    return wrap(dispatch_values_copy(_r.tensor(0)));
  } else {
    // aten::values_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_values_copy_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::values_copy_out(out, self);
    };
    return wrap(dispatch_values_copy_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unfold_copy
static PyObject * THPVariable_unfold_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unfold_copy(Tensor input, int64_t dimension, int64_t size, int64_t step, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::unfold_copy(Tensor self, int dimension, int size, int step) -> Tensor
    
    auto dispatch_unfold_copy = [](const at::Tensor & self, int64_t dimension, int64_t size, int64_t step) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::unfold_copy(self, dimension, size, step);
    };
    return wrap(dispatch_unfold_copy(_r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toInt64(3)));
  } else {
    // aten::unfold_copy.out(Tensor self, int dimension, int size, int step, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_unfold_copy_out = [](at::Tensor out, const at::Tensor & self, int64_t dimension, int64_t size, int64_t step) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::unfold_copy_out(out, self, dimension, size, step);
    };
    return wrap(dispatch_unfold_copy_out(_r.tensor(4), _r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toInt64(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _fused_sdp_choice
static PyObject * THPVariable__fused_sdp_choice(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_fused_sdp_choice(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, double dropout_p=0.0, bool is_causal=False, *, double? scale=None, bool enable_gqa=False)",
  }, /*traceable=*/false);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_fused_sdp_choice(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, *, float? scale=None, bool enable_gqa=False) -> int
  
  auto dispatch__fused_sdp_choice = [](const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & attn_mask, double dropout_p, bool is_causal, ::std::optional<double> scale, bool enable_gqa) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return at::_fused_sdp_choice(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
  };
  return wrap(dispatch__fused_sdp_choice(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.optionalTensor(3), _r.toDouble(4), _r.toBool(5), _r.toDoubleOptional(6), _r.toBool(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _scaled_dot_product_efficient_attention
static PyObject * THPVariable__scaled_dot_product_efficient_attention(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get__scaled_dot_product_efficient_attention_structseq();
  static PythonArgParser parser({
    "_scaled_dot_product_efficient_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_bias, bool compute_log_sumexp, double dropout_p=0.0, bool is_causal=False, *, double? scale=None)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_scaled_dot_product_efficient_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_bias, bool compute_log_sumexp, float dropout_p=0.0, bool is_causal=False, *, float? scale=None) -> (Tensor output, Tensor log_sumexp, Tensor philox_seed, Tensor philox_offset)
  
  auto dispatch__scaled_dot_product_efficient_attention = [](const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & attn_bias, bool compute_log_sumexp, double dropout_p, bool is_causal, ::std::optional<double> scale) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_scaled_dot_product_efficient_attention(query, key, value, attn_bias, compute_log_sumexp, dropout_p, is_causal, scale);
  };
  return wrap(NamedTuple, dispatch__scaled_dot_product_efficient_attention(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.optionalTensor(3), _r.toBool(4), _r.toDouble(5), _r.toBool(6), _r.toDoubleOptional(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _scaled_dot_product_cudnn_attention
static PyObject * THPVariable__scaled_dot_product_cudnn_attention(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get__scaled_dot_product_cudnn_attention_structseq();
  static PythonArgParser parser({
    "_scaled_dot_product_cudnn_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_bias, bool compute_log_sumexp, double dropout_p=0.0, bool is_causal=False, bool return_debug_mask=False, *, double? scale=None)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_scaled_dot_product_cudnn_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_bias, bool compute_log_sumexp, float dropout_p=0.0, bool is_causal=False, bool return_debug_mask=False, *, float? scale=None) -> (Tensor output, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, Tensor philox_seed, Tensor philox_offset, Tensor debug_attn_mask)
  
  auto dispatch__scaled_dot_product_cudnn_attention = [](const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & attn_bias, bool compute_log_sumexp, double dropout_p, bool is_causal, bool return_debug_mask, ::std::optional<double> scale) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,c10::SymInt,c10::SymInt,at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_scaled_dot_product_cudnn_attention(query, key, value, attn_bias, compute_log_sumexp, dropout_p, is_causal, return_debug_mask, scale);
  };
  return wrap(NamedTuple, dispatch__scaled_dot_product_cudnn_attention(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.optionalTensor(3), _r.toBool(4), _r.toDouble(5), _r.toBool(6), _r.toBool(7), _r.toDoubleOptional(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _fill_mem_eff_dropout_mask_
static PyObject * THPVariable__fill_mem_eff_dropout_mask_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_fill_mem_eff_dropout_mask_(Tensor input, double dropout_p, int64_t seed, int64_t offset)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_fill_mem_eff_dropout_mask_(Tensor(a!) self, float dropout_p, int seed, int offset) -> Tensor(a!)
  
  auto dispatch__fill_mem_eff_dropout_mask_ = [](at::Tensor self, double dropout_p, int64_t seed, int64_t offset) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_fill_mem_eff_dropout_mask_(self, dropout_p, seed, offset);
  };
  return wrap(dispatch__fill_mem_eff_dropout_mask_(_r.tensor(0), _r.toDouble(1), _r.toInt64(2), _r.toInt64(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foobar
static PyObject * THPVariable__foobar(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foobar(Tensor input, bool arg1=True, bool arg2=True, *, bool arg3=True)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foobar(Tensor self, bool arg1=True, bool arg2=True, *, bool arg3=True) -> Tensor
  
  auto dispatch__foobar = [](const at::Tensor & self, bool arg1, bool arg2, bool arg3) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_foobar(self, arg1, arg2, arg3);
  };
  return wrap(dispatch__foobar(_r.tensor(0), _r.toBool(1), _r.toBool(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _fused_adagrad_
static PyObject * THPVariable__fused_adagrad_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_fused_adagrad_(TensorList self, TensorList grads, TensorList state_sums, TensorList state_steps, *, double lr, double lr_decay, double weight_decay, double eps, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None)",
  }, /*traceable=*/false);

  ParsedArgs<11> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_fused_adagrad_(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] state_sums, Tensor(d!)[] state_steps, *, float lr, float lr_decay, float weight_decay, float eps, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
  
  auto dispatch__fused_adagrad_ = [](at::TensorList self, at::TensorList grads, at::TensorList state_sums, at::TensorList state_steps, double lr, double lr_decay, double weight_decay, double eps, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_fused_adagrad_(self, grads, state_sums, state_steps, lr, lr_decay, weight_decay, eps, maximize, grad_scale, found_inf);
  };
  dispatch__fused_adagrad_(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.tensorlist(3), _r.toDouble(4), _r.toDouble(5), _r.toDouble(6), _r.toDouble(7), _r.toBool(8), _r.optionalTensor(9), _r.optionalTensor(10));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _propagate_xla_data
static PyObject * THPVariable__propagate_xla_data(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_propagate_xla_data(Tensor input, Tensor output)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_propagate_xla_data(Tensor input, Tensor output) -> ()
  
  auto dispatch__propagate_xla_data = [](const at::Tensor & input, const at::Tensor & output) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_propagate_xla_data(input, output);
  };
  dispatch__propagate_xla_data(_r.tensor(0), _r.tensor(1));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

} // namespace torch::autograd
