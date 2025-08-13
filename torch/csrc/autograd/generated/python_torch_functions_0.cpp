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
#include <ATen/ops/_cast_Byte.h>
#include <ATen/ops/_cast_Float.h>
#include <ATen/ops/_cast_Long.h>
#include <ATen/ops/_assert_async.h>
#include <ATen/ops/_assert_scalar.h>
#include <ATen/ops/_assert_tensor_metadata.h>
#include <ATen/ops/sym_constrain_range.h>
#include <ATen/ops/_functional_sym_constrain_range_for_size.h>
#include <ATen/ops/_use_cudnn_ctc_loss.h>
#include <ATen/ops/_cudnn_rnn_flatten_weight.h>
#include <ATen/ops/native_dropout.h>
#include <ATen/ops/_sobol_engine_draw.h>
#include <ATen/ops/_reshape_from_tensor.h>
#include <ATen/ops/feature_dropout.h>
#include <ATen/ops/feature_dropout.h>
#include <ATen/ops/alpha_dropout.h>
#include <ATen/ops/alpha_dropout.h>
#include <ATen/ops/angle.h>
#include <ATen/ops/view_as_complex.h>
#include <ATen/ops/sgn.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/conj.h>
#include <ATen/ops/resolve_neg.h>
#include <ATen/ops/adaptive_max_pool1d.h>
#include <ATen/ops/affine_grid_generator.h>
#include <ATen/ops/_is_any_true.h>
#include <ATen/ops/any.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/_dim_arange.h>
#include <ATen/ops/atanh.h>
#include <ATen/ops/atanh.h>
#include <ATen/ops/as_strided.h>
#include <ATen/ops/as_strided.h>
#include <ATen/ops/atleast_1d.h>
#include <ATen/ops/bernoulli.h>
#include <ATen/ops/bilinear.h>
#include <ATen/ops/binary_cross_entropy_with_logits.h>
#include <ATen/ops/logical_and.h>
#include <ATen/ops/blackman_window.h>
#include <ATen/ops/concat.h>
#include <ATen/ops/concatenate.h>
#include <ATen/ops/block_diag.h>
#include <ATen/ops/ceil.h>
#include <ATen/ops/ceil.h>
#include <ATen/ops/cudnn_is_acceptable.h>
#include <ATen/ops/conv_transpose2d.h>
#include <ATen/ops/_copy_from.h>
#include <ATen/ops/_copy_from_and_resize.h>
#include <ATen/ops/cosh.h>
#include <ATen/ops/cosh.h>
#include <ATen/ops/cov.h>
#include <ATen/ops/cudnn_convolution.h>
#include <ATen/ops/cummax.h>
#include <ATen/ops/cummin.h>
#include <ATen/ops/_cummin_helper.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/cumulative_trapezoid.h>
#include <ATen/ops/_ctc_loss.h>
#include <ATen/ops/diag_embed.h>
#include <ATen/ops/diagflat.h>
#include <ATen/ops/diagonal.h>
#include <ATen/ops/diff.h>
#include <ATen/ops/divide.h>
#include <ATen/ops/dot.h>
#include <ATen/ops/vdot.h>
#include <ATen/ops/einsum.h>
#include <ATen/ops/embedding.h>
#include <ATen/ops/embedding_renorm.h>
#include <ATen/ops/_rowwise_prune.h>
#include <ATen/ops/empty_permuted.h>
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/empty_quantized.h>
#include <ATen/ops/exp2.h>
#include <ATen/ops/exp2.h>
#include <ATen/ops/expm1.h>
#include <ATen/ops/expm1.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/full.h>
#include <ATen/ops/from_file.h>
#include <ATen/ops/gcd.h>
#include <ATen/ops/gcd.h>
#include <ATen/ops/lcm.h>
#include <ATen/ops/lcm.h>
#include <ATen/ops/hamming_window.h>
#include <ATen/ops/hinge_embedding_loss.h>
#include <ATen/ops/group_norm.h>
#include <ATen/ops/native_group_norm.h>
#include <ATen/ops/_fft_c2c.h>
#include <ATen/ops/_cufft_set_plan_cache_max_size.h>
#include <ATen/ops/_unsafe_index.h>
#include <ATen/ops/index_copy.h>
#include <ATen/ops/instance_norm.h>
#include <ATen/ops/isclose.h>
#include <ATen/ops/is_floating_point.h>
#include <ATen/ops/_is_zerotensor.h>
#include <ATen/ops/is_neg.h>
#include <ATen/ops/is_nonzero.h>
#include <ATen/ops/is_same_size.h>
#include <ATen/ops/is_signed.h>
#include <ATen/ops/kl_div.h>
#include <ATen/ops/kthvalue.h>
#include <ATen/ops/rms_norm.h>
#include <ATen/ops/_cslt_sparse_mm_search.h>
#include <ATen/ops/_sparse_semi_structured_apply.h>
#include <ATen/ops/_sparse_semi_structured_apply_dense.h>
#include <ATen/ops/fbgemm_linear_int8_weight_fp32_activation.h>
#include <ATen/ops/fbgemm_linear_int8_weight.h>
#include <ATen/ops/fbgemm_linear_quantize_weight.h>
#include <ATen/ops/_wrapped_linear_prepack.h>
#include <ATen/ops/fbgemm_pack_quantized_matrix.h>
#include <ATen/ops/log2.h>
#include <ATen/ops/log2.h>
#include <ATen/ops/logspace.h>
#include <ATen/ops/log_softmax.h>
#include <ATen/ops/_log_softmax.h>
#include <ATen/ops/_logcumsumexp.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/_compute_linear_combination.h>
#include <ATen/ops/max_pool1d_with_indices.h>
#include <ATen/ops/max_pool2d.h>
#include <ATen/ops/mkldnn_max_pool2d.h>
#include <ATen/ops/mkldnn_rnn_layer.h>
#include <ATen/ops/miopen_convolution_relu.h>
#include <ATen/ops/miopen_rnn.h>
#include <ATen/ops/_convert_weight_to_int4pack.h>
#include <ATen/ops/_weight_int4pack_mm.h>
#include <ATen/ops/_weight_int4pack_mm_for_cpu.h>
#include <ATen/ops/_dyn_quant_pack_4bit_weight.h>
#include <ATen/ops/_dyn_quant_matmul_4bit.h>
#include <ATen/ops/_sparse_sparse_matmul.h>
#include <ATen/ops/mode.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/native_batch_norm.h>
#include <ATen/ops/_native_batch_norm_legit.h>
#include <ATen/ops/batch_norm_gather_stats.h>
#include <ATen/ops/batch_norm_backward_reduce.h>
#include <ATen/ops/is_vulkan_available.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/pdist.h>
#include <ATen/ops/moveaxis.h>
#include <ATen/ops/pixel_unshuffle.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/ravel.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/round.h>
#include <ATen/ops/round.h>
#include <ATen/ops/rrelu.h>
#include <ATen/ops/rrelu.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/slice_scatter.h>
#include <ATen/ops/select_scatter.h>
#include <ATen/ops/diagonal_scatter.h>
#include <ATen/ops/as_strided_scatter.h>
#include <ATen/ops/smm.h>
#include <ATen/ops/dsplit.h>
#include <ATen/ops/sspaddmm.h>
#include <ATen/ops/stft.h>
#include <ATen/ops/std.h>
#include <ATen/ops/std_mean.h>
#include <ATen/ops/prod.h>
#include <ATen/ops/threshold.h>
#include <ATen/ops/threshold.h>
#include <ATen/ops/transpose.h>
#include <ATen/ops/flip.h>
#include <ATen/ops/fliplr.h>
#include <ATen/ops/flipud.h>
#include <ATen/ops/roll.h>
#include <ATen/ops/trapezoid.h>
#include <ATen/ops/_transform_bias_rescale_qkv.h>
#include <ATen/ops/_nested_tensor_from_mask.h>
#include <ATen/ops/_nested_view_from_buffer_copy.h>
#include <ATen/ops/_nested_view_from_jagged.h>
#include <ATen/ops/_nested_view_from_jagged_copy.h>
#include <ATen/ops/_nested_get_ragged_idx.h>
#include <ATen/ops/_nested_get_min_seqlen.h>
#include <ATen/ops/_nested_get_max_seqlen.h>
#include <ATen/ops/_nested_compute_contiguous_strides_offsets.h>
#include <ATen/ops/_trilinear.h>
#include <ATen/ops/_unique2.h>
#include <ATen/ops/unsqueeze.h>
#include <ATen/ops/var.h>
#include <ATen/ops/norm_except_dim.h>
#include <ATen/ops/_weight_norm_interface.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/_efficientzerotensor.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/_sample_dirichlet.h>
#include <ATen/ops/_sparse_sum.h>
#include <ATen/ops/_sparse_csr_prod.h>
#include <ATen/ops/frobenius_norm.h>
#include <ATen/ops/resize_as_sparse.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/_addmm_activation.h>
#include <ATen/ops/_scaled_grouped_mm.h>
#include <ATen/ops/_validate_sparse_coo_tensor_args.h>
#include <ATen/ops/_validate_sparse_bsr_tensor_args.h>
#include <ATen/ops/_validate_sparse_bsc_tensor_args.h>
#include <ATen/ops/dequantize.h>
#include <ATen/ops/q_scale.h>
#include <ATen/ops/fake_quantize_per_tensor_affine.h>
#include <ATen/ops/_fake_quantize_learnable_per_channel_affine.h>
#include <ATen/ops/cartesian_prod.h>
#include <ATen/ops/rnn_tanh.h>
#include <ATen/ops/rnn_relu.h>
#include <ATen/ops/gru_cell.h>
#include <ATen/ops/rnn_relu_cell.h>
#include <ATen/ops/quantized_lstm_cell.h>
#include <ATen/ops/_pack_padded_sequence.h>
#include <ATen/ops/_pad_packed_sequence.h>
#include <ATen/ops/masked_fill.h>
#include <ATen/ops/_masked_softmax.h>
#include <ATen/ops/index_fill.h>
#include <ATen/ops/scatter.h>
#include <ATen/ops/and.h>
#include <ATen/ops/bitwise_or.h>
#include <ATen/ops/triu.h>
#include <ATen/ops/ge.h>
#include <ATen/ops/le.h>
#include <ATen/ops/less_equal.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/take_along_dim.h>
#include <ATen/ops/masked_select.h>
#include <ATen/ops/nonzero_static.h>
#include <ATen/ops/argwhere.h>
#include <ATen/ops/addcdiv.h>
#include <ATen/ops/swapdims.h>
#include <ATen/ops/cholesky_solve.h>
#include <ATen/ops/qr.h>
#include <ATen/ops/polygamma.h>
#include <ATen/ops/signbit.h>
#include <ATen/ops/lerp.h>
#include <ATen/ops/_histogramdd_from_bin_cts.h>
#include <ATen/ops/_histogramdd_from_bin_tensors.h>
#include <ATen/ops/fmod.h>
#include <ATen/ops/igammac.h>
#include <ATen/ops/remainder.h>
#include <ATen/ops/quantile.h>
#include <ATen/ops/argsort.h>
#include <ATen/ops/topk.h>
#include <ATen/ops/equal.h>
#include <ATen/ops/normal.h>
#include <ATen/ops/_foreach_div.h>
#include <ATen/ops/_foreach_div.h>
#include <ATen/ops/_foreach_clamp_min.h>
#include <ATen/ops/_foreach_clamp_min.h>
#include <ATen/ops/_foreach_addcmul.h>
#include <ATen/ops/_foreach_addcmul.h>
#include <ATen/ops/_foreach_atan.h>
#include <ATen/ops/_foreach_atan.h>
#include <ATen/ops/_foreach_ceil.h>
#include <ATen/ops/_foreach_ceil.h>
#include <ATen/ops/_foreach_erfc.h>
#include <ATen/ops/_foreach_erfc.h>
#include <ATen/ops/_foreach_expm1.h>
#include <ATen/ops/_foreach_expm1.h>
#include <ATen/ops/_foreach_lerp.h>
#include <ATen/ops/_foreach_lerp.h>
#include <ATen/ops/_foreach_log.h>
#include <ATen/ops/_foreach_log.h>
#include <ATen/ops/_foreach_log10.h>
#include <ATen/ops/_foreach_log10.h>
#include <ATen/ops/_foreach_max.h>
#include <ATen/ops/_foreach_rsqrt.h>
#include <ATen/ops/_foreach_rsqrt.h>
#include <ATen/ops/_foreach_sin.h>
#include <ATen/ops/_foreach_sin.h>
#include <ATen/ops/_foreach_trunc.h>
#include <ATen/ops/_foreach_trunc.h>
#include <ATen/ops/_foreach_zero.h>
#include <ATen/ops/_adaptive_avg_pool2d.h>
#include <ATen/ops/_adaptive_avg_pool3d.h>
#include <ATen/ops/isposinf.h>
#include <ATen/ops/_add_batch_dim.h>
#include <ATen/ops/_linalg_det.h>
#include <ATen/ops/det.h>
#include <ATen/ops/logdet.h>
#include <ATen/ops/ger.h>
#include <ATen/ops/_linalg_svd.h>
#include <ATen/ops/_nested_tensor_from_tensor_list.h>
#include <ATen/ops/_fw_primal_copy.h>
#include <ATen/ops/_make_dual_copy.h>
#include <ATen/ops/view_as_complex_copy.h>
#include <ATen/ops/_neg_view_copy.h>
#include <ATen/ops/as_strided_copy.h>
#include <ATen/ops/diagonal_copy.h>
#include <ATen/ops/expand_copy.h>
#include <ATen/ops/_reshape_alias_copy.h>
#include <ATen/ops/slice_copy.h>
#include <ATen/ops/split_with_sizes_copy.h>
#include <ATen/ops/t_copy.h>
#include <ATen/ops/indices_copy.h>
#include <ATen/ops/crow_indices_copy.h>
#include <ATen/ops/col_indices_copy.h>
#include <ATen/ops/ccol_indices_copy.h>
#include <ATen/ops/row_indices_copy.h>
#include <ATen/ops/view_copy.h>
#include <ATen/ops/_nested_from_padded_tensor.h>
#include <ATen/ops/_transformer_encoder_layer_fwd.h>
#include <ATen/ops/_scaled_dot_product_attention_math.h>
#include <ATen/ops/_triton_multi_head_attention.h>
#include <ATen/ops/_fused_adam.h>
#include <ATen/ops/_fused_sgd.h>
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

static PyObject * THPVariable__cast_Byte(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cast_Float(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cast_Long(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__assert_async(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__assert_scalar(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__assert_tensor_metadata(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sym_constrain_range(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__functional_sym_constrain_range_for_size(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__use_cudnn_ctc_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cudnn_rnn_flatten_weight(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_native_dropout(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sobol_engine_draw(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__reshape_from_tensor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_feature_dropout(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_feature_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_alpha_dropout(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_alpha_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_angle(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_view_as_complex(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sgn(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_imag(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_conj(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_resolve_neg(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_adaptive_max_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_affine_grid_generator(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__is_any_true(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_any(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_arange(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__dim_arange(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_atanh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_atanh_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_as_strided(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_as_strided_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_atleast_1d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bernoulli(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bilinear(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_binary_cross_entropy_with_logits(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_logical_and(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_blackman_window(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_concat(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_concatenate(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_block_diag(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ceil(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ceil_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cudnn_is_acceptable(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_conv_transpose2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__copy_from(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__copy_from_and_resize(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cosh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cosh_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cov(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cudnn_convolution(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cummax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cummin(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cummin_helper(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cumsum(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cumulative_trapezoid(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__ctc_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_diag_embed(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_diagflat(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_diagonal(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_diff(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_divide(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_dot(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_vdot(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_einsum(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_embedding(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_embedding_renorm_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__rowwise_prune(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_empty_permuted(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__empty_affine_quantized(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_empty_quantized(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_exp2(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_exp2_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_expm1(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_expm1_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_eye(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_full(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_from_file(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_gcd(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_gcd_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_lcm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_lcm_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_hamming_window(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_hinge_embedding_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_group_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_native_group_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__fft_c2c(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cufft_set_plan_cache_max_size(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__unsafe_index(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_index_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_instance_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_isclose(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_is_floating_point(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__is_zerotensor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_is_neg(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_is_nonzero(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_is_same_size(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_is_signed(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_kl_div(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_kthvalue(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rms_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cslt_sparse_mm_search(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_semi_structured_apply(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_semi_structured_apply_dense(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fbgemm_linear_int8_weight_fp32_activation(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fbgemm_linear_int8_weight(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fbgemm_linear_quantize_weight(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__wrapped_linear_prepack(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fbgemm_pack_quantized_matrix(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_log2(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_log2_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_logspace(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_log_softmax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__log_softmax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__logcumsumexp(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_matmul(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_aminmax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__compute_linear_combination(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_max_pool1d_with_indices(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_max_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mkldnn_max_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mkldnn_rnn_layer(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_miopen_convolution_relu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_miopen_rnn(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__convert_weight_to_int4pack(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__weight_int4pack_mm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__weight_int4pack_mm_for_cpu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__dyn_quant_pack_4bit_weight(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__dyn_quant_matmul_4bit(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_sparse_matmul(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mode(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mul(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_native_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__native_batch_norm_legit(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_batch_norm_gather_stats(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_batch_norm_backward_reduce(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_is_vulkan_available(PyObject* self_, PyObject* args);
static PyObject * THPVariable_ones(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_pdist(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_moveaxis(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_pixel_unshuffle(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_scalar_tensor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rand(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ravel(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_neg(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_neg_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_round(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_round_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rrelu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rrelu_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sigmoid(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sigmoid_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_slice_scatter(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_select_scatter(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_diagonal_scatter(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_as_strided_scatter(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_smm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_dsplit(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sspaddmm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_stft(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_std(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_std_mean(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_prod(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_threshold(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_threshold_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_transpose(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_flip(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fliplr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_flipud(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_roll(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_trapezoid(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__transform_bias_rescale_qkv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_tensor_from_mask(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_view_from_buffer_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_view_from_jagged(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_view_from_jagged_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_get_ragged_idx(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_get_min_seqlen(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_get_max_seqlen(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_compute_contiguous_strides_offsets(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__trilinear(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__unique2(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_unsqueeze(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_var(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_norm_except_dim(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__weight_norm_interface(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_zeros(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__efficientzerotensor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_zeros_like(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sample_dirichlet(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_sum(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_csr_prod(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_frobenius_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_resize_as_sparse_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_addmm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__addmm_activation(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__scaled_grouped_mm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__validate_sparse_coo_tensor_args(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__validate_sparse_bsr_tensor_args(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__validate_sparse_bsc_tensor_args(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_dequantize(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_q_scale(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fake_quantize_per_tensor_affine(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__fake_quantize_learnable_per_channel_affine(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cartesian_prod(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rnn_tanh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rnn_relu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_gru_cell(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rnn_relu_cell(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantized_lstm_cell(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__pack_padded_sequence(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__pad_packed_sequence(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_masked_fill(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__masked_softmax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_index_fill(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_scatter(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable___and__(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bitwise_or(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_triu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ge(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_le(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_less_equal(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_gt(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_lt(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_take_along_dim(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_masked_select(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_nonzero_static(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_argwhere(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_addcdiv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_swapdims(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cholesky_solve(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_qr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_polygamma(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_signbit(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_lerp(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__histogramdd_from_bin_cts(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__histogramdd_from_bin_tensors(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fmod(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_igammac(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_remainder(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantile(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_argsort(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_topk(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_equal(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_normal(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_div(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_div_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_clamp_min(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_clamp_min_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_addcmul(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_addcmul_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_atan(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_atan_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_ceil(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_ceil_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_erfc(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_erfc_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_expm1(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_expm1_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_lerp(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_lerp_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_log(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_log_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_log10(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_log10_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_max(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_rsqrt(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_rsqrt_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_sin(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_sin_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_trunc(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_trunc_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_zero_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__adaptive_avg_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__adaptive_avg_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_isposinf(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__add_batch_dim(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__linalg_det(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_det(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_logdet(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ger(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__linalg_svd(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_tensor_from_tensor_list(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__fw_primal_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__make_dual_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_view_as_complex_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__neg_view_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_as_strided_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_diagonal_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_expand_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__reshape_alias_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_slice_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_split_with_sizes_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_t_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_indices_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_crow_indices_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_col_indices_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ccol_indices_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_row_indices_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_view_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_from_padded_tensor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__transformer_encoder_layer_fwd(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__scaled_dot_product_attention_math(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__triton_multi_head_attention(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__fused_adam_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__fused_sgd_(PyObject* self_, PyObject* args, PyObject* kwargs);

static PyMethodDef torch_functions_shard[] = {
  {"_cast_Byte", castPyCFunctionWithKeywords(THPVariable__cast_Byte), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_cast_Float", castPyCFunctionWithKeywords(THPVariable__cast_Float), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_cast_Long", castPyCFunctionWithKeywords(THPVariable__cast_Long), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_assert_async", castPyCFunctionWithKeywords(THPVariable__assert_async), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_assert_scalar", castPyCFunctionWithKeywords(THPVariable__assert_scalar), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_assert_tensor_metadata", castPyCFunctionWithKeywords(THPVariable__assert_tensor_metadata), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sym_constrain_range", castPyCFunctionWithKeywords(THPVariable_sym_constrain_range), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_functional_sym_constrain_range_for_size", castPyCFunctionWithKeywords(THPVariable__functional_sym_constrain_range_for_size), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_use_cudnn_ctc_loss", castPyCFunctionWithKeywords(THPVariable__use_cudnn_ctc_loss), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_cudnn_rnn_flatten_weight", castPyCFunctionWithKeywords(THPVariable__cudnn_rnn_flatten_weight), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"native_dropout", castPyCFunctionWithKeywords(THPVariable_native_dropout), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sobol_engine_draw", castPyCFunctionWithKeywords(THPVariable__sobol_engine_draw), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_reshape_from_tensor", castPyCFunctionWithKeywords(THPVariable__reshape_from_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"feature_dropout", castPyCFunctionWithKeywords(THPVariable_feature_dropout), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"feature_dropout_", castPyCFunctionWithKeywords(THPVariable_feature_dropout_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"alpha_dropout", castPyCFunctionWithKeywords(THPVariable_alpha_dropout), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"alpha_dropout_", castPyCFunctionWithKeywords(THPVariable_alpha_dropout_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"angle", castPyCFunctionWithKeywords(THPVariable_angle), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"view_as_complex", castPyCFunctionWithKeywords(THPVariable_view_as_complex), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sgn", castPyCFunctionWithKeywords(THPVariable_sgn), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"imag", castPyCFunctionWithKeywords(THPVariable_imag), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"conj", castPyCFunctionWithKeywords(THPVariable_conj), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"resolve_neg", castPyCFunctionWithKeywords(THPVariable_resolve_neg), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"adaptive_max_pool1d", castPyCFunctionWithKeywords(THPVariable_adaptive_max_pool1d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"affine_grid_generator", castPyCFunctionWithKeywords(THPVariable_affine_grid_generator), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_is_any_true", castPyCFunctionWithKeywords(THPVariable__is_any_true), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"any", castPyCFunctionWithKeywords(THPVariable_any), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"arange", castPyCFunctionWithKeywords(THPVariable_arange), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_dim_arange", castPyCFunctionWithKeywords(THPVariable__dim_arange), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"atanh", castPyCFunctionWithKeywords(THPVariable_atanh), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"atanh_", castPyCFunctionWithKeywords(THPVariable_atanh_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"as_strided", castPyCFunctionWithKeywords(THPVariable_as_strided), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"as_strided_", castPyCFunctionWithKeywords(THPVariable_as_strided_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"atleast_1d", castPyCFunctionWithKeywords(THPVariable_atleast_1d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"bernoulli", castPyCFunctionWithKeywords(THPVariable_bernoulli), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"bilinear", castPyCFunctionWithKeywords(THPVariable_bilinear), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"binary_cross_entropy_with_logits", castPyCFunctionWithKeywords(THPVariable_binary_cross_entropy_with_logits), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"logical_and", castPyCFunctionWithKeywords(THPVariable_logical_and), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"blackman_window", castPyCFunctionWithKeywords(THPVariable_blackman_window), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"concat", castPyCFunctionWithKeywords(THPVariable_concat), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"concatenate", castPyCFunctionWithKeywords(THPVariable_concatenate), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"block_diag", castPyCFunctionWithKeywords(THPVariable_block_diag), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"ceil", castPyCFunctionWithKeywords(THPVariable_ceil), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"ceil_", castPyCFunctionWithKeywords(THPVariable_ceil_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cudnn_is_acceptable", castPyCFunctionWithKeywords(THPVariable_cudnn_is_acceptable), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"conv_transpose2d", castPyCFunctionWithKeywords(THPVariable_conv_transpose2d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_copy_from", castPyCFunctionWithKeywords(THPVariable__copy_from), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_copy_from_and_resize", castPyCFunctionWithKeywords(THPVariable__copy_from_and_resize), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cosh", castPyCFunctionWithKeywords(THPVariable_cosh), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cosh_", castPyCFunctionWithKeywords(THPVariable_cosh_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cov", castPyCFunctionWithKeywords(THPVariable_cov), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cudnn_convolution", castPyCFunctionWithKeywords(THPVariable_cudnn_convolution), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cummax", castPyCFunctionWithKeywords(THPVariable_cummax), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cummin", castPyCFunctionWithKeywords(THPVariable_cummin), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_cummin_helper", castPyCFunctionWithKeywords(THPVariable__cummin_helper), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cumsum", castPyCFunctionWithKeywords(THPVariable_cumsum), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cumulative_trapezoid", castPyCFunctionWithKeywords(THPVariable_cumulative_trapezoid), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_ctc_loss", castPyCFunctionWithKeywords(THPVariable__ctc_loss), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"diag_embed", castPyCFunctionWithKeywords(THPVariable_diag_embed), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"diagflat", castPyCFunctionWithKeywords(THPVariable_diagflat), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"diagonal", castPyCFunctionWithKeywords(THPVariable_diagonal), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"diff", castPyCFunctionWithKeywords(THPVariable_diff), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"divide", castPyCFunctionWithKeywords(THPVariable_divide), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"dot", castPyCFunctionWithKeywords(THPVariable_dot), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"vdot", castPyCFunctionWithKeywords(THPVariable_vdot), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"einsum", castPyCFunctionWithKeywords(THPVariable_einsum), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"embedding", castPyCFunctionWithKeywords(THPVariable_embedding), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"embedding_renorm_", castPyCFunctionWithKeywords(THPVariable_embedding_renorm_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_rowwise_prune", castPyCFunctionWithKeywords(THPVariable__rowwise_prune), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"empty_permuted", castPyCFunctionWithKeywords(THPVariable_empty_permuted), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_empty_affine_quantized", castPyCFunctionWithKeywords(THPVariable__empty_affine_quantized), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"empty_quantized", castPyCFunctionWithKeywords(THPVariable_empty_quantized), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"exp2", castPyCFunctionWithKeywords(THPVariable_exp2), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"exp2_", castPyCFunctionWithKeywords(THPVariable_exp2_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"expm1", castPyCFunctionWithKeywords(THPVariable_expm1), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"expm1_", castPyCFunctionWithKeywords(THPVariable_expm1_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"eye", castPyCFunctionWithKeywords(THPVariable_eye), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"full", castPyCFunctionWithKeywords(THPVariable_full), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"from_file", castPyCFunctionWithKeywords(THPVariable_from_file), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"gcd", castPyCFunctionWithKeywords(THPVariable_gcd), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"gcd_", castPyCFunctionWithKeywords(THPVariable_gcd_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"lcm", castPyCFunctionWithKeywords(THPVariable_lcm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"lcm_", castPyCFunctionWithKeywords(THPVariable_lcm_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"hamming_window", castPyCFunctionWithKeywords(THPVariable_hamming_window), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"hinge_embedding_loss", castPyCFunctionWithKeywords(THPVariable_hinge_embedding_loss), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"group_norm", castPyCFunctionWithKeywords(THPVariable_group_norm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"native_group_norm", castPyCFunctionWithKeywords(THPVariable_native_group_norm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_fft_c2c", castPyCFunctionWithKeywords(THPVariable__fft_c2c), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_cufft_set_plan_cache_max_size", castPyCFunctionWithKeywords(THPVariable__cufft_set_plan_cache_max_size), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_unsafe_index", castPyCFunctionWithKeywords(THPVariable__unsafe_index), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"index_copy", castPyCFunctionWithKeywords(THPVariable_index_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"instance_norm", castPyCFunctionWithKeywords(THPVariable_instance_norm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"isclose", castPyCFunctionWithKeywords(THPVariable_isclose), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"is_floating_point", castPyCFunctionWithKeywords(THPVariable_is_floating_point), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_is_zerotensor", castPyCFunctionWithKeywords(THPVariable__is_zerotensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"is_neg", castPyCFunctionWithKeywords(THPVariable_is_neg), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"is_nonzero", castPyCFunctionWithKeywords(THPVariable_is_nonzero), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"is_same_size", castPyCFunctionWithKeywords(THPVariable_is_same_size), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"is_signed", castPyCFunctionWithKeywords(THPVariable_is_signed), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"kl_div", castPyCFunctionWithKeywords(THPVariable_kl_div), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"kthvalue", castPyCFunctionWithKeywords(THPVariable_kthvalue), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"rms_norm", castPyCFunctionWithKeywords(THPVariable_rms_norm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_cslt_sparse_mm_search", castPyCFunctionWithKeywords(THPVariable__cslt_sparse_mm_search), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_semi_structured_apply", castPyCFunctionWithKeywords(THPVariable__sparse_semi_structured_apply), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_semi_structured_apply_dense", castPyCFunctionWithKeywords(THPVariable__sparse_semi_structured_apply_dense), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"fbgemm_linear_int8_weight_fp32_activation", castPyCFunctionWithKeywords(THPVariable_fbgemm_linear_int8_weight_fp32_activation), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"fbgemm_linear_int8_weight", castPyCFunctionWithKeywords(THPVariable_fbgemm_linear_int8_weight), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"fbgemm_linear_quantize_weight", castPyCFunctionWithKeywords(THPVariable_fbgemm_linear_quantize_weight), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_wrapped_linear_prepack", castPyCFunctionWithKeywords(THPVariable__wrapped_linear_prepack), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"fbgemm_pack_quantized_matrix", castPyCFunctionWithKeywords(THPVariable_fbgemm_pack_quantized_matrix), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"log2", castPyCFunctionWithKeywords(THPVariable_log2), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"log2_", castPyCFunctionWithKeywords(THPVariable_log2_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"logspace", castPyCFunctionWithKeywords(THPVariable_logspace), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"log_softmax", castPyCFunctionWithKeywords(THPVariable_log_softmax), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_log_softmax", castPyCFunctionWithKeywords(THPVariable__log_softmax), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_logcumsumexp", castPyCFunctionWithKeywords(THPVariable__logcumsumexp), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"matmul", castPyCFunctionWithKeywords(THPVariable_matmul), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"aminmax", castPyCFunctionWithKeywords(THPVariable_aminmax), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_compute_linear_combination", castPyCFunctionWithKeywords(THPVariable__compute_linear_combination), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"max_pool1d_with_indices", castPyCFunctionWithKeywords(THPVariable_max_pool1d_with_indices), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"max_pool2d", castPyCFunctionWithKeywords(THPVariable_max_pool2d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"mkldnn_max_pool2d", castPyCFunctionWithKeywords(THPVariable_mkldnn_max_pool2d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"mkldnn_rnn_layer", castPyCFunctionWithKeywords(THPVariable_mkldnn_rnn_layer), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"miopen_convolution_relu", castPyCFunctionWithKeywords(THPVariable_miopen_convolution_relu), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"miopen_rnn", castPyCFunctionWithKeywords(THPVariable_miopen_rnn), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_convert_weight_to_int4pack", castPyCFunctionWithKeywords(THPVariable__convert_weight_to_int4pack), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_weight_int4pack_mm", castPyCFunctionWithKeywords(THPVariable__weight_int4pack_mm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_weight_int4pack_mm_for_cpu", castPyCFunctionWithKeywords(THPVariable__weight_int4pack_mm_for_cpu), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_dyn_quant_pack_4bit_weight", castPyCFunctionWithKeywords(THPVariable__dyn_quant_pack_4bit_weight), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_dyn_quant_matmul_4bit", castPyCFunctionWithKeywords(THPVariable__dyn_quant_matmul_4bit), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_sparse_matmul", castPyCFunctionWithKeywords(THPVariable__sparse_sparse_matmul), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"mode", castPyCFunctionWithKeywords(THPVariable_mode), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"mul", castPyCFunctionWithKeywords(THPVariable_mul), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"native_batch_norm", castPyCFunctionWithKeywords(THPVariable_native_batch_norm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_native_batch_norm_legit", castPyCFunctionWithKeywords(THPVariable__native_batch_norm_legit), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"batch_norm_gather_stats", castPyCFunctionWithKeywords(THPVariable_batch_norm_gather_stats), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"batch_norm_backward_reduce", castPyCFunctionWithKeywords(THPVariable_batch_norm_backward_reduce), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"is_vulkan_available", THPVariable_is_vulkan_available, METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"ones", castPyCFunctionWithKeywords(THPVariable_ones), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"pdist", castPyCFunctionWithKeywords(THPVariable_pdist), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"moveaxis", castPyCFunctionWithKeywords(THPVariable_moveaxis), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"pixel_unshuffle", castPyCFunctionWithKeywords(THPVariable_pixel_unshuffle), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"scalar_tensor", castPyCFunctionWithKeywords(THPVariable_scalar_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"rand", castPyCFunctionWithKeywords(THPVariable_rand), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"ravel", castPyCFunctionWithKeywords(THPVariable_ravel), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"neg", castPyCFunctionWithKeywords(THPVariable_neg), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"neg_", castPyCFunctionWithKeywords(THPVariable_neg_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"round", castPyCFunctionWithKeywords(THPVariable_round), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"round_", castPyCFunctionWithKeywords(THPVariable_round_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"rrelu", castPyCFunctionWithKeywords(THPVariable_rrelu), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"rrelu_", castPyCFunctionWithKeywords(THPVariable_rrelu_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sigmoid", castPyCFunctionWithKeywords(THPVariable_sigmoid), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sigmoid_", castPyCFunctionWithKeywords(THPVariable_sigmoid_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"slice_scatter", castPyCFunctionWithKeywords(THPVariable_slice_scatter), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"select_scatter", castPyCFunctionWithKeywords(THPVariable_select_scatter), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"diagonal_scatter", castPyCFunctionWithKeywords(THPVariable_diagonal_scatter), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"as_strided_scatter", castPyCFunctionWithKeywords(THPVariable_as_strided_scatter), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"smm", castPyCFunctionWithKeywords(THPVariable_smm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"dsplit", castPyCFunctionWithKeywords(THPVariable_dsplit), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sspaddmm", castPyCFunctionWithKeywords(THPVariable_sspaddmm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"stft", castPyCFunctionWithKeywords(THPVariable_stft), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"std", castPyCFunctionWithKeywords(THPVariable_std), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"std_mean", castPyCFunctionWithKeywords(THPVariable_std_mean), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"prod", castPyCFunctionWithKeywords(THPVariable_prod), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"threshold", castPyCFunctionWithKeywords(THPVariable_threshold), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"threshold_", castPyCFunctionWithKeywords(THPVariable_threshold_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"transpose", castPyCFunctionWithKeywords(THPVariable_transpose), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"flip", castPyCFunctionWithKeywords(THPVariable_flip), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"fliplr", castPyCFunctionWithKeywords(THPVariable_fliplr), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"flipud", castPyCFunctionWithKeywords(THPVariable_flipud), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"roll", castPyCFunctionWithKeywords(THPVariable_roll), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"trapezoid", castPyCFunctionWithKeywords(THPVariable_trapezoid), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_transform_bias_rescale_qkv", castPyCFunctionWithKeywords(THPVariable__transform_bias_rescale_qkv), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_tensor_from_mask", castPyCFunctionWithKeywords(THPVariable__nested_tensor_from_mask), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_view_from_buffer_copy", castPyCFunctionWithKeywords(THPVariable__nested_view_from_buffer_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_view_from_jagged", castPyCFunctionWithKeywords(THPVariable__nested_view_from_jagged), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_view_from_jagged_copy", castPyCFunctionWithKeywords(THPVariable__nested_view_from_jagged_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_get_ragged_idx", castPyCFunctionWithKeywords(THPVariable__nested_get_ragged_idx), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_get_min_seqlen", castPyCFunctionWithKeywords(THPVariable__nested_get_min_seqlen), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_get_max_seqlen", castPyCFunctionWithKeywords(THPVariable__nested_get_max_seqlen), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_compute_contiguous_strides_offsets", castPyCFunctionWithKeywords(THPVariable__nested_compute_contiguous_strides_offsets), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_trilinear", castPyCFunctionWithKeywords(THPVariable__trilinear), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_unique2", castPyCFunctionWithKeywords(THPVariable__unique2), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"unsqueeze", castPyCFunctionWithKeywords(THPVariable_unsqueeze), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"var", castPyCFunctionWithKeywords(THPVariable_var), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"norm_except_dim", castPyCFunctionWithKeywords(THPVariable_norm_except_dim), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_weight_norm_interface", castPyCFunctionWithKeywords(THPVariable__weight_norm_interface), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"zeros", castPyCFunctionWithKeywords(THPVariable_zeros), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_efficientzerotensor", castPyCFunctionWithKeywords(THPVariable__efficientzerotensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"zeros_like", castPyCFunctionWithKeywords(THPVariable_zeros_like), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sample_dirichlet", castPyCFunctionWithKeywords(THPVariable__sample_dirichlet), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_sum", castPyCFunctionWithKeywords(THPVariable__sparse_sum), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_csr_prod", castPyCFunctionWithKeywords(THPVariable__sparse_csr_prod), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"frobenius_norm", castPyCFunctionWithKeywords(THPVariable_frobenius_norm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"resize_as_sparse_", castPyCFunctionWithKeywords(THPVariable_resize_as_sparse_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"addmm", castPyCFunctionWithKeywords(THPVariable_addmm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_addmm_activation", castPyCFunctionWithKeywords(THPVariable__addmm_activation), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_scaled_grouped_mm", castPyCFunctionWithKeywords(THPVariable__scaled_grouped_mm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_validate_sparse_coo_tensor_args", castPyCFunctionWithKeywords(THPVariable__validate_sparse_coo_tensor_args), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_validate_sparse_bsr_tensor_args", castPyCFunctionWithKeywords(THPVariable__validate_sparse_bsr_tensor_args), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_validate_sparse_bsc_tensor_args", castPyCFunctionWithKeywords(THPVariable__validate_sparse_bsc_tensor_args), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"dequantize", castPyCFunctionWithKeywords(THPVariable_dequantize), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"q_scale", castPyCFunctionWithKeywords(THPVariable_q_scale), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"fake_quantize_per_tensor_affine", castPyCFunctionWithKeywords(THPVariable_fake_quantize_per_tensor_affine), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_fake_quantize_learnable_per_channel_affine", castPyCFunctionWithKeywords(THPVariable__fake_quantize_learnable_per_channel_affine), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cartesian_prod", castPyCFunctionWithKeywords(THPVariable_cartesian_prod), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"rnn_tanh", castPyCFunctionWithKeywords(THPVariable_rnn_tanh), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"rnn_relu", castPyCFunctionWithKeywords(THPVariable_rnn_relu), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"gru_cell", castPyCFunctionWithKeywords(THPVariable_gru_cell), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"rnn_relu_cell", castPyCFunctionWithKeywords(THPVariable_rnn_relu_cell), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"quantized_lstm_cell", castPyCFunctionWithKeywords(THPVariable_quantized_lstm_cell), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_pack_padded_sequence", castPyCFunctionWithKeywords(THPVariable__pack_padded_sequence), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_pad_packed_sequence", castPyCFunctionWithKeywords(THPVariable__pad_packed_sequence), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"masked_fill", castPyCFunctionWithKeywords(THPVariable_masked_fill), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_masked_softmax", castPyCFunctionWithKeywords(THPVariable__masked_softmax), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"index_fill", castPyCFunctionWithKeywords(THPVariable_index_fill), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"scatter", castPyCFunctionWithKeywords(THPVariable_scatter), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"__and__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable___and__>), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"bitwise_or", castPyCFunctionWithKeywords(THPVariable_bitwise_or), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"triu", castPyCFunctionWithKeywords(THPVariable_triu), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"ge", castPyCFunctionWithKeywords(THPVariable_ge), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"le", castPyCFunctionWithKeywords(THPVariable_le), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"less_equal", castPyCFunctionWithKeywords(THPVariable_less_equal), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"gt", castPyCFunctionWithKeywords(THPVariable_gt), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"lt", castPyCFunctionWithKeywords(THPVariable_lt), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"take_along_dim", castPyCFunctionWithKeywords(THPVariable_take_along_dim), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"masked_select", castPyCFunctionWithKeywords(THPVariable_masked_select), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"nonzero_static", castPyCFunctionWithKeywords(THPVariable_nonzero_static), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"argwhere", castPyCFunctionWithKeywords(THPVariable_argwhere), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"addcdiv", castPyCFunctionWithKeywords(THPVariable_addcdiv), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"swapdims", castPyCFunctionWithKeywords(THPVariable_swapdims), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cholesky_solve", castPyCFunctionWithKeywords(THPVariable_cholesky_solve), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"qr", castPyCFunctionWithKeywords(THPVariable_qr), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"polygamma", castPyCFunctionWithKeywords(THPVariable_polygamma), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"signbit", castPyCFunctionWithKeywords(THPVariable_signbit), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"lerp", castPyCFunctionWithKeywords(THPVariable_lerp), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_histogramdd_from_bin_cts", castPyCFunctionWithKeywords(THPVariable__histogramdd_from_bin_cts), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_histogramdd_from_bin_tensors", castPyCFunctionWithKeywords(THPVariable__histogramdd_from_bin_tensors), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"fmod", castPyCFunctionWithKeywords(THPVariable_fmod), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"igammac", castPyCFunctionWithKeywords(THPVariable_igammac), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"remainder", castPyCFunctionWithKeywords(THPVariable_remainder), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"quantile", castPyCFunctionWithKeywords(THPVariable_quantile), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"argsort", castPyCFunctionWithKeywords(THPVariable_argsort), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"topk", castPyCFunctionWithKeywords(THPVariable_topk), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"equal", castPyCFunctionWithKeywords(THPVariable_equal), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"normal", castPyCFunctionWithKeywords(THPVariable_normal), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_div", castPyCFunctionWithKeywords(THPVariable__foreach_div), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_div_", castPyCFunctionWithKeywords(THPVariable__foreach_div_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_clamp_min", castPyCFunctionWithKeywords(THPVariable__foreach_clamp_min), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_clamp_min_", castPyCFunctionWithKeywords(THPVariable__foreach_clamp_min_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_addcmul", castPyCFunctionWithKeywords(THPVariable__foreach_addcmul), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_addcmul_", castPyCFunctionWithKeywords(THPVariable__foreach_addcmul_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_atan", castPyCFunctionWithKeywords(THPVariable__foreach_atan), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_atan_", castPyCFunctionWithKeywords(THPVariable__foreach_atan_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_ceil", castPyCFunctionWithKeywords(THPVariable__foreach_ceil), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_ceil_", castPyCFunctionWithKeywords(THPVariable__foreach_ceil_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_erfc", castPyCFunctionWithKeywords(THPVariable__foreach_erfc), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_erfc_", castPyCFunctionWithKeywords(THPVariable__foreach_erfc_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_expm1", castPyCFunctionWithKeywords(THPVariable__foreach_expm1), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_expm1_", castPyCFunctionWithKeywords(THPVariable__foreach_expm1_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_lerp", castPyCFunctionWithKeywords(THPVariable__foreach_lerp), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_lerp_", castPyCFunctionWithKeywords(THPVariable__foreach_lerp_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_log", castPyCFunctionWithKeywords(THPVariable__foreach_log), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_log_", castPyCFunctionWithKeywords(THPVariable__foreach_log_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_log10", castPyCFunctionWithKeywords(THPVariable__foreach_log10), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_log10_", castPyCFunctionWithKeywords(THPVariable__foreach_log10_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_max", castPyCFunctionWithKeywords(THPVariable__foreach_max), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_rsqrt", castPyCFunctionWithKeywords(THPVariable__foreach_rsqrt), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_rsqrt_", castPyCFunctionWithKeywords(THPVariable__foreach_rsqrt_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_sin", castPyCFunctionWithKeywords(THPVariable__foreach_sin), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_sin_", castPyCFunctionWithKeywords(THPVariable__foreach_sin_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_trunc", castPyCFunctionWithKeywords(THPVariable__foreach_trunc), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_trunc_", castPyCFunctionWithKeywords(THPVariable__foreach_trunc_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_zero_", castPyCFunctionWithKeywords(THPVariable__foreach_zero_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_adaptive_avg_pool2d", castPyCFunctionWithKeywords(THPVariable__adaptive_avg_pool2d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_adaptive_avg_pool3d", castPyCFunctionWithKeywords(THPVariable__adaptive_avg_pool3d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"isposinf", castPyCFunctionWithKeywords(THPVariable_isposinf), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_add_batch_dim", castPyCFunctionWithKeywords(THPVariable__add_batch_dim), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_linalg_det", castPyCFunctionWithKeywords(THPVariable__linalg_det), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"det", castPyCFunctionWithKeywords(THPVariable_det), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"logdet", castPyCFunctionWithKeywords(THPVariable_logdet), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"ger", castPyCFunctionWithKeywords(THPVariable_ger), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_linalg_svd", castPyCFunctionWithKeywords(THPVariable__linalg_svd), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_tensor_from_tensor_list", castPyCFunctionWithKeywords(THPVariable__nested_tensor_from_tensor_list), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_fw_primal_copy", castPyCFunctionWithKeywords(THPVariable__fw_primal_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_make_dual_copy", castPyCFunctionWithKeywords(THPVariable__make_dual_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"view_as_complex_copy", castPyCFunctionWithKeywords(THPVariable_view_as_complex_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_neg_view_copy", castPyCFunctionWithKeywords(THPVariable__neg_view_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"as_strided_copy", castPyCFunctionWithKeywords(THPVariable_as_strided_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"diagonal_copy", castPyCFunctionWithKeywords(THPVariable_diagonal_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"expand_copy", castPyCFunctionWithKeywords(THPVariable_expand_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_reshape_alias_copy", castPyCFunctionWithKeywords(THPVariable__reshape_alias_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"slice_copy", castPyCFunctionWithKeywords(THPVariable_slice_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"split_with_sizes_copy", castPyCFunctionWithKeywords(THPVariable_split_with_sizes_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"t_copy", castPyCFunctionWithKeywords(THPVariable_t_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"indices_copy", castPyCFunctionWithKeywords(THPVariable_indices_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"crow_indices_copy", castPyCFunctionWithKeywords(THPVariable_crow_indices_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"col_indices_copy", castPyCFunctionWithKeywords(THPVariable_col_indices_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"ccol_indices_copy", castPyCFunctionWithKeywords(THPVariable_ccol_indices_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"row_indices_copy", castPyCFunctionWithKeywords(THPVariable_row_indices_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"view_copy", castPyCFunctionWithKeywords(THPVariable_view_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_from_padded_tensor", castPyCFunctionWithKeywords(THPVariable__nested_from_padded_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_transformer_encoder_layer_fwd", castPyCFunctionWithKeywords(THPVariable__transformer_encoder_layer_fwd), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_scaled_dot_product_attention_math", castPyCFunctionWithKeywords(THPVariable__scaled_dot_product_attention_math), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_triton_multi_head_attention", castPyCFunctionWithKeywords(THPVariable__triton_multi_head_attention), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_fused_adam_", castPyCFunctionWithKeywords(THPVariable__fused_adam_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_fused_sgd_", castPyCFunctionWithKeywords(THPVariable__fused_sgd_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
};

void gatherTorchFunctions_0(std::vector<PyMethodDef> &torch_functions) {
  constexpr size_t num_functions = sizeof(torch_functions_shard) / sizeof(torch_functions_shard[0]);
  torch_functions.insert(
    torch_functions.end(),
    torch_functions_shard,
    torch_functions_shard + num_functions);
}

// generated methods start here

// _cast_Byte
static PyObject * THPVariable__cast_Byte(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Byte(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cast_Byte(Tensor self, bool non_blocking=False) -> Tensor
  
  auto dispatch__cast_Byte = [](const at::Tensor & self, bool non_blocking) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cast_Byte(self, non_blocking);
  };
  return wrap(dispatch__cast_Byte(_r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cast_Float
static PyObject * THPVariable__cast_Float(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Float(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cast_Float(Tensor self, bool non_blocking=False) -> Tensor
  
  auto dispatch__cast_Float = [](const at::Tensor & self, bool non_blocking) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cast_Float(self, non_blocking);
  };
  return wrap(dispatch__cast_Float(_r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cast_Long
static PyObject * THPVariable__cast_Long(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Long(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cast_Long(Tensor self, bool non_blocking=False) -> Tensor
  
  auto dispatch__cast_Long = [](const at::Tensor & self, bool non_blocking) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cast_Long(self, non_blocking);
  };
  return wrap(dispatch__cast_Long(_r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _assert_async
static PyObject * THPVariable__assert_async(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_assert_async(Tensor input)",
    "_assert_async(Tensor input, c10::string_view assert_msg)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_assert_async(Tensor self) -> ()
      
      auto dispatch__assert_async = [](const at::Tensor & self) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_assert_async(self);
      };
      dispatch__assert_async(_r.tensor(0));
      Py_RETURN_NONE;
    }
    case 1: {
      // aten::_assert_async.msg(Tensor self, str assert_msg) -> ()
      
      auto dispatch__assert_async = [](const at::Tensor & self, c10::string_view assert_msg) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_assert_async(self, assert_msg);
      };
      dispatch__assert_async(_r.tensor(0), _r.stringView(1));
      Py_RETURN_NONE;
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _assert_scalar
static PyObject * THPVariable__assert_scalar(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_assert_scalar(Scalar self, c10::string_view assert_msg)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_assert_scalar(Scalar self, str assert_msg) -> ()
  
  auto dispatch__assert_scalar = [](const at::Scalar & self, c10::string_view assert_msg) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_assert_scalar(self, assert_msg);
  };
  dispatch__assert_scalar(_r.scalar(0), _r.stringView(1));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _assert_tensor_metadata
static PyObject * THPVariable__assert_tensor_metadata(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_assert_tensor_metadata(Tensor a, SymIntArrayRef? size=None, SymIntArrayRef? stride=None, ScalarType? dtype=None, *, Device? device=None, Layout? layout=None)",
  }, /*traceable=*/false);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_assert_tensor_metadata(Tensor a, SymInt[]? size=None, SymInt[]? stride=None, ScalarType? dtype=None, *, Device? device=None, Layout? layout=None) -> ()
  
  auto dispatch__assert_tensor_metadata = [](const at::Tensor & a, at::OptionalSymIntArrayRef size, at::OptionalSymIntArrayRef stride, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Device> device, ::std::optional<at::Layout> layout) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_assert_tensor_metadata_symint(a, size, stride, dtype, device, layout);
  };
  dispatch__assert_tensor_metadata(_r.tensor(0), _r.symintlistOptional(1), _r.symintlistOptional(2), _r.scalartypeOptional(3), _r.deviceOptional(4), _r.layoutOptional(5));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sym_constrain_range
static PyObject * THPVariable_sym_constrain_range(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sym_constrain_range(Scalar size, *, int64_t? min=None, int64_t? max=None)",
  }, /*traceable=*/false);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::sym_constrain_range(Scalar size, *, int? min=None, int? max=None) -> ()
  
  auto dispatch_sym_constrain_range = [](const at::Scalar & size, ::std::optional<int64_t> min, ::std::optional<int64_t> max) -> void {
    pybind11::gil_scoped_release no_gil;
    at::sym_constrain_range(size, min, max);
  };
  dispatch_sym_constrain_range(_r.scalar(0), _r.toInt64Optional(1), _r.toInt64Optional(2));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _functional_sym_constrain_range_for_size
static PyObject * THPVariable__functional_sym_constrain_range_for_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_functional_sym_constrain_range_for_size(Scalar size, int64_t? min, int64_t? max, Tensor dep_token)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_functional_sym_constrain_range_for_size(Scalar size, int? min, int? max, Tensor dep_token) -> Tensor
  
  auto dispatch__functional_sym_constrain_range_for_size = [](const at::Scalar & size, ::std::optional<int64_t> min, ::std::optional<int64_t> max, const at::Tensor & dep_token) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_functional_sym_constrain_range_for_size(size, min, max, dep_token);
  };
  return wrap(dispatch__functional_sym_constrain_range_for_size(_r.scalar(0), _r.toInt64Optional(1), _r.toInt64Optional(2), _r.tensor(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _use_cudnn_ctc_loss
static PyObject * THPVariable__use_cudnn_ctc_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_use_cudnn_ctc_loss(Tensor log_probs, Tensor targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank)",
    "_use_cudnn_ctc_loss(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int64_t blank)",
  }, /*traceable=*/false);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_use_cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank) -> bool
      
      auto dispatch__use_cudnn_ctc_loss = [](const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank) -> bool {
        pybind11::gil_scoped_release no_gil;
        return at::_use_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank);
      };
      return wrap(dispatch__use_cudnn_ctc_loss(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.intlist(3), _r.toInt64(4)));
    }
    case 1: {
      // aten::_use_cudnn_ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank) -> bool
      
      auto dispatch__use_cudnn_ctc_loss = [](const at::Tensor & log_probs, const at::Tensor & targets, const at::Tensor & input_lengths, const at::Tensor & target_lengths, int64_t blank) -> bool {
        pybind11::gil_scoped_release no_gil;
        return at::_use_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank);
      };
      return wrap(dispatch__use_cudnn_ctc_loss(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.toInt64(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cudnn_rnn_flatten_weight
static PyObject * THPVariable__cudnn_rnn_flatten_weight(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cudnn_rnn_flatten_weight(TensorList weight_arr, int64_t weight_stride0, SymInt input_size, int64_t mode, SymInt hidden_size, SymInt proj_size, int64_t num_layers, bool batch_first, bool bidirectional)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cudnn_rnn_flatten_weight(Tensor[] weight_arr, int weight_stride0, SymInt input_size, int mode, SymInt hidden_size, SymInt proj_size, int num_layers, bool batch_first, bool bidirectional) -> Tensor
  
  auto dispatch__cudnn_rnn_flatten_weight = [](at::TensorList weight_arr, int64_t weight_stride0, c10::SymInt input_size, int64_t mode, c10::SymInt hidden_size, c10::SymInt proj_size, int64_t num_layers, bool batch_first, bool bidirectional) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cudnn_rnn_flatten_weight_symint(weight_arr, weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional);
  };
  return wrap(dispatch__cudnn_rnn_flatten_weight(_r.tensorlist(0), _r.toInt64(1), _r.toSymInt(2), _r.toInt64(3), _r.toSymInt(4), _r.toSymInt(5), _r.toInt64(6), _r.toBool(7), _r.toBool(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// native_dropout
static PyObject * THPVariable_native_dropout(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "native_dropout(Tensor input, double p, bool? train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)
  
  auto dispatch_native_dropout = [](const at::Tensor & input, double p, ::std::optional<bool> train) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::native_dropout(input, p, train);
  };
  return wrap(dispatch_native_dropout(_r.tensor(0), _r.toDouble(1), _r.toBoolOptional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sobol_engine_draw
static PyObject * THPVariable__sobol_engine_draw(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sobol_engine_draw(Tensor quasi, int64_t n, Tensor sobolstate, int64_t dimension, int64_t num_generated, ScalarType? dtype)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_sobol_engine_draw(Tensor quasi, int n, Tensor sobolstate, int dimension, int num_generated, ScalarType? dtype) -> (Tensor, Tensor)
  
  auto dispatch__sobol_engine_draw = [](const at::Tensor & quasi, int64_t n, const at::Tensor & sobolstate, int64_t dimension, int64_t num_generated, ::std::optional<at::ScalarType> dtype) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_sobol_engine_draw(quasi, n, sobolstate, dimension, num_generated, dtype);
  };
  return wrap(dispatch__sobol_engine_draw(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.toInt64(3), _r.toInt64(4), _r.scalartypeOptional(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _reshape_from_tensor
static PyObject * THPVariable__reshape_from_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_reshape_from_tensor(Tensor input, Tensor shape)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_reshape_from_tensor(Tensor self, Tensor shape) -> Tensor
  
  auto dispatch__reshape_from_tensor = [](const at::Tensor & self, const at::Tensor & shape) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_reshape_from_tensor(self, shape);
  };
  return wrap(dispatch__reshape_from_tensor(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// feature_dropout
static PyObject * THPVariable_feature_dropout(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "feature_dropout(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::feature_dropout(Tensor input, float p, bool train) -> Tensor
  
  auto dispatch_feature_dropout = [](const at::Tensor & input, double p, bool train) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::feature_dropout(input, p, train);
  };
  return wrap(dispatch_feature_dropout(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// feature_dropout_
static PyObject * THPVariable_feature_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "feature_dropout_(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::feature_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
  
  auto dispatch_feature_dropout_ = [](at::Tensor self, double p, bool train) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::feature_dropout_(self, p, train);
  };
  return wrap(dispatch_feature_dropout_(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// alpha_dropout
static PyObject * THPVariable_alpha_dropout(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "alpha_dropout(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::alpha_dropout(Tensor input, float p, bool train) -> Tensor
  
  auto dispatch_alpha_dropout = [](const at::Tensor & input, double p, bool train) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::alpha_dropout(input, p, train);
  };
  return wrap(dispatch_alpha_dropout(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// alpha_dropout_
static PyObject * THPVariable_alpha_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "alpha_dropout_(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
  
  auto dispatch_alpha_dropout_ = [](at::Tensor self, double p, bool train) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::alpha_dropout_(self, p, train);
  };
  return wrap(dispatch_alpha_dropout_(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// angle
static PyObject * THPVariable_angle(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "angle(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::angle(Tensor self) -> Tensor
    
    auto dispatch_angle = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.angle();
    };
    return wrap(dispatch_angle(_r.tensor(0)));
  } else {
    // aten::angle.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_angle_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::angle_out(out, self);
    };
    return wrap(dispatch_angle_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// view_as_complex
static PyObject * THPVariable_view_as_complex(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "view_as_complex(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::view_as_complex(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_view_as_complex = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::view_as_complex(self);
  };
  return wrap(dispatch_view_as_complex(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sgn
static PyObject * THPVariable_sgn(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sgn(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::sgn(Tensor self) -> Tensor
    
    auto dispatch_sgn = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.sgn();
    };
    return wrap(dispatch_sgn(_r.tensor(0)));
  } else {
    // aten::sgn.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_sgn_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::sgn_out(out, self);
    };
    return wrap(dispatch_sgn_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// imag
static PyObject * THPVariable_imag(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "imag(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::imag(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_imag = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::imag(self);
  };
  return wrap(dispatch_imag(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// conj
static PyObject * THPVariable_conj(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conj(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::conj(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_conj = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.conj();
  };
  return wrap(dispatch_conj(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// resolve_neg
static PyObject * THPVariable_resolve_neg(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "resolve_neg(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::resolve_neg(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_resolve_neg = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.resolve_neg();
  };
  return wrap(dispatch_resolve_neg(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// adaptive_max_pool1d
static PyObject * THPVariable_adaptive_max_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_max_pool1d(Tensor input, IntArrayRef[1] output_size)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::adaptive_max_pool1d(Tensor self, int[1] output_size) -> (Tensor, Tensor)
  
  auto dispatch_adaptive_max_pool1d = [](const at::Tensor & self, at::IntArrayRef output_size) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::adaptive_max_pool1d(self, output_size);
  };
  return wrap(dispatch_adaptive_max_pool1d(_r.tensor(0), _r.intlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// affine_grid_generator
static PyObject * THPVariable_affine_grid_generator(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "affine_grid_generator(Tensor theta, SymIntArrayRef size, bool align_corners)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::affine_grid_generator(Tensor theta, SymInt[] size, bool align_corners) -> Tensor
  
  auto dispatch_affine_grid_generator = [](const at::Tensor & theta, c10::SymIntArrayRef size, bool align_corners) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::affine_grid_generator_symint(theta, size, align_corners);
  };
  return wrap(dispatch_affine_grid_generator(_r.tensor(0), _r.symintlist(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _is_any_true
static PyObject * THPVariable__is_any_true(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_is_any_true(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_is_any_true(Tensor self) -> Tensor
  
  auto dispatch__is_any_true = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._is_any_true();
  };
  return wrap(dispatch__is_any_true(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// any
static PyObject * THPVariable_any(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "any(Tensor input, *, Tensor out=None)",
    "any(Tensor input, IntArrayRef? dim=None, bool keepdim=False, *, Tensor out=None)",
    "any(Tensor input, int64_t dim, bool keepdim=False, *, Tensor out=None)",
    "any(Tensor input, Dimname dim, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(1)) {
        // aten::any(Tensor self) -> Tensor
        
        auto dispatch_any = [](const at::Tensor & self) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.any();
        };
        return wrap(dispatch_any(_r.tensor(0)));
      } else {
        // aten::any.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_any_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::any_out(out, self);
        };
        return wrap(dispatch_any_out(_r.tensor(1), _r.tensor(0)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::any.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor
        
        auto dispatch_any = [](const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.any(dim, keepdim);
        };
        return wrap(dispatch_any(_r.tensor(0), _r.intlistOptional(1), _r.toBool(2)));
      } else {
        // aten::any.dims_out(Tensor self, int[]? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_any_out = [](at::Tensor out, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::any_out(out, self, dim, keepdim);
        };
        return wrap(dispatch_any_out(_r.tensor(3), _r.tensor(0), _r.intlistOptional(1), _r.toBool(2)));
      }
    }
    case 2: {
      if (_r.isNone(3)) {
        // aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
        
        auto dispatch_any = [](const at::Tensor & self, int64_t dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.any(dim, keepdim);
        };
        return wrap(dispatch_any(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      } else {
        // aten::any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_any_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::any_out(out, self, dim, keepdim);
        };
        return wrap(dispatch_any_out(_r.tensor(3), _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      }
    }
    case 3: {
      if (_r.isNone(3)) {
        // aten::any.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor
        
        auto dispatch_any = [](const at::Tensor & self, at::Dimname dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.any(dim, keepdim);
        };
        return wrap(dispatch_any(_r.tensor(0), _r.dimname(1), _r.toBool(2)));
      } else {
        // aten::any.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_any_out = [](at::Tensor out, const at::Tensor & self, at::Dimname dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::any_out(out, self, dim, keepdim);
        };
        return wrap(dispatch_any_out(_r.tensor(3), _r.tensor(0), _r.dimname(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// arange
static PyObject * THPVariable_arange(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "arange(Scalar end, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "arange(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "arange(Scalar start, Scalar end, Scalar step=1, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(1)) {
        // aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(2))
            .device(_r.deviceWithDefault(4, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(3))
            .requires_grad(_r.toBool(6))
            .pinned_memory(_r.toBool(5));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_arange = [](const at::Scalar & end, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::arange(end, options);
        };
        return wrap(dispatch_arange(_r.scalar(0), options));
      } else {
        // aten::arange.out(Scalar end, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(1), _r.scalartypeOptional(2),
                               _r.isNone(2), _r.layoutOptional(3),
                               _r.deviceWithDefault(4, torch::tensors::get_default_device()), _r.isNone(4));
        
        auto dispatch_arange_out = [](at::Tensor out, const at::Scalar & end) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::arange_out(out, end);
        };
        return wrap(dispatch_arange_out(_r.tensor(1), _r.scalar(0)).set_requires_grad(_r.toBool(6)));
      }
    }
    case 1: {
      // aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(2))
          .device(_r.deviceWithDefault(4, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(3))
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_arange = [](const at::Scalar & start, const at::Scalar & end, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::arange(start, end, options);
      };
      return wrap(dispatch_arange(_r.scalar(0), _r.scalar(1), options));
    }
    case 2: {
      if (_r.isNone(3)) {
        // aten::arange.start_step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(4))
            .device(_r.deviceWithDefault(6, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(5))
            .requires_grad(_r.toBool(8))
            .pinned_memory(_r.toBool(7));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_arange = [](const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::arange(start, end, step, options);
        };
        return wrap(dispatch_arange(_r.scalar(0), _r.scalar(1), _r.scalar(2), options));
      } else {
        // aten::arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(3), _r.scalartypeOptional(4),
                               _r.isNone(4), _r.layoutOptional(5),
                               _r.deviceWithDefault(6, torch::tensors::get_default_device()), _r.isNone(6));
        
        auto dispatch_arange_out = [](at::Tensor out, const at::Scalar & start, const at::Scalar & end, const at::Scalar & step) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::arange_out(out, start, end, step);
        };
        return wrap(dispatch_arange_out(_r.tensor(3), _r.scalar(0), _r.scalar(1), _r.scalar(2)).set_requires_grad(_r.toBool(8)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _dim_arange
static PyObject * THPVariable__dim_arange(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_dim_arange(Tensor like, int64_t dim)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_dim_arange(Tensor like, int dim) -> Tensor
  
  auto dispatch__dim_arange = [](const at::Tensor & like, int64_t dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_dim_arange(like, dim);
  };
  return wrap(dispatch__dim_arange(_r.tensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// atanh
static PyObject * THPVariable_atanh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atanh(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::atanh(Tensor self) -> Tensor
    
    auto dispatch_atanh = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.atanh();
    };
    return wrap(dispatch_atanh(_r.tensor(0)));
  } else {
    // aten::atanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_atanh_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::atanh_out(out, self);
    };
    return wrap(dispatch_atanh_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// atanh_
static PyObject * THPVariable_atanh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atanh_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::atanh_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_atanh_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.atanh_();
  };
  return wrap(dispatch_atanh_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// as_strided
static PyObject * THPVariable_as_strided(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "as_strided(Tensor input, SymIntArrayRef size, SymIntArrayRef stride, SymInt? storage_offset=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)
  
  auto dispatch_as_strided = [](const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<c10::SymInt> storage_offset) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.as_strided_symint(size, stride, storage_offset);
  };
  return wrap(dispatch_as_strided(_r.tensor(0), _r.symintlist(1), _r.symintlist(2), _r.toSymIntOptional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// as_strided_
static PyObject * THPVariable_as_strided_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "as_strided_(Tensor input, SymIntArrayRef size, SymIntArrayRef stride, SymInt? storage_offset=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::as_strided_(Tensor(a!) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a!)
  
  auto dispatch_as_strided_ = [](const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<c10::SymInt> storage_offset) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.as_strided__symint(size, stride, storage_offset);
  };
  return wrap(dispatch_as_strided_(_r.tensor(0), _r.symintlist(1), _r.symintlist(2), _r.toSymIntOptional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// atleast_1d
static PyObject * THPVariable_atleast_1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atleast_1d(Tensor input)",
    "atleast_1d(TensorList tensors)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::atleast_1d(Tensor self) -> Tensor
      
      auto dispatch_atleast_1d = [](const at::Tensor & self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::atleast_1d(self);
      };
      return wrap(dispatch_atleast_1d(_r.tensor(0)));
    }
    case 1: {
      // aten::atleast_1d.Sequence(Tensor[] tensors) -> Tensor[]
      
      auto dispatch_atleast_1d = [](at::TensorList tensors) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::atleast_1d(tensors);
      };
      return wrap(dispatch_atleast_1d(_r.tensorlist(0)));
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
  static PythonArgParser parser({
    "bernoulli(Tensor input, *, Generator? generator=None, Tensor out=None)",
    "bernoulli(Tensor input, double p, *, Generator? generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor
        
        auto dispatch_bernoulli = [](const at::Tensor & self, ::std::optional<at::Generator> generator) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.bernoulli(generator);
        };
        return wrap(dispatch_bernoulli(_r.tensor(0), _r.generator(1)));
      } else {
        // aten::bernoulli.out(Tensor self, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_bernoulli_out = [](at::Tensor out, const at::Tensor & self, ::std::optional<at::Generator> generator) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bernoulli_out(out, self, generator);
        };
        return wrap(dispatch_bernoulli_out(_r.tensor(2), _r.tensor(0), _r.generator(1)));
      }
    }
    case 1: {
      // aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> Tensor
      
      auto dispatch_bernoulli = [](const at::Tensor & self, double p, ::std::optional<at::Generator> generator) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.bernoulli(p, generator);
      };
      return wrap(dispatch_bernoulli(_r.tensor(0), _r.toDouble(1), _r.generator(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// bilinear
static PyObject * THPVariable_bilinear(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias=None) -> Tensor
  
  auto dispatch_bilinear = [](const at::Tensor & input1, const at::Tensor & input2, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::bilinear(input1, input2, weight, bias);
  };
  return wrap(dispatch_bilinear(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.optionalTensor(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// binary_cross_entropy_with_logits
static PyObject * THPVariable_binary_cross_entropy_with_logits(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "binary_cross_entropy_with_logits(Tensor input, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int64_t reduction=at::Reduction::Mean)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::binary_cross_entropy_with_logits(Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor
  
  auto dispatch_binary_cross_entropy_with_logits = [](const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & pos_weight, int64_t reduction) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::binary_cross_entropy_with_logits(self, target, weight, pos_weight, reduction);
  };
  return wrap(dispatch_binary_cross_entropy_with_logits(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.optionalTensor(3), _r.toInt64(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logical_and
static PyObject * THPVariable_logical_and(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logical_and(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::logical_and(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_logical_and = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.logical_and(other);
    };
    return wrap(dispatch_logical_and(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::logical_and.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_logical_and_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::logical_and_out(out, self, other);
    };
    return wrap(dispatch_logical_and_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// blackman_window
static PyObject * THPVariable_blackman_window(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "blackman_window(int64_t window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "blackman_window(int64_t window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::blackman_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(1))
          .device(_r.deviceWithDefault(3, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(2))
          .requires_grad(_r.toBool(5))
          .pinned_memory(_r.toBool(4));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_blackman_window = [](int64_t window_length, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::blackman_window(window_length, options);
      };
      return wrap(dispatch_blackman_window(_r.toInt64(0), options));
    }
    case 1: {
      // aten::blackman_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(2))
          .device(_r.deviceWithDefault(4, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(3))
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_blackman_window = [](int64_t window_length, bool periodic, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::blackman_window(window_length, periodic, options);
      };
      return wrap(dispatch_blackman_window(_r.toInt64(0), _r.toBool(1), options));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// concat
static PyObject * THPVariable_concat(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "concat(TensorList tensors, int64_t dim=0, *, Tensor out=None)",
    "concat(TensorList tensors, Dimname dim, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::concat(Tensor[] tensors, int dim=0) -> Tensor
        
        auto dispatch_concat = [](at::TensorList tensors, int64_t dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::concat(tensors, dim);
        };
        return wrap(dispatch_concat(_r.tensorlist(0), _r.toInt64(1)));
      } else {
        // aten::concat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_concat_out = [](at::Tensor out, at::TensorList tensors, int64_t dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::concat_out(out, tensors, dim);
        };
        return wrap(dispatch_concat_out(_r.tensor(2), _r.tensorlist(0), _r.toInt64(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::concat.names(Tensor[] tensors, Dimname dim) -> Tensor
        
        auto dispatch_concat = [](at::TensorList tensors, at::Dimname dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::concat(tensors, dim);
        };
        return wrap(dispatch_concat(_r.tensorlist(0), _r.dimname(1)));
      } else {
        // aten::concat.names_out(Tensor[] tensors, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_concat_out = [](at::Tensor out, at::TensorList tensors, at::Dimname dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::concat_out(out, tensors, dim);
        };
        return wrap(dispatch_concat_out(_r.tensor(2), _r.tensorlist(0), _r.dimname(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// concatenate
static PyObject * THPVariable_concatenate(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "concatenate(TensorList tensors, int64_t dim=0, *, Tensor out=None)",
    "concatenate(TensorList tensors, Dimname dim, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::concatenate(Tensor[] tensors, int dim=0) -> Tensor
        
        auto dispatch_concatenate = [](at::TensorList tensors, int64_t dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::concatenate(tensors, dim);
        };
        return wrap(dispatch_concatenate(_r.tensorlist(0), _r.toInt64(1)));
      } else {
        // aten::concatenate.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_concatenate_out = [](at::Tensor out, at::TensorList tensors, int64_t dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::concatenate_out(out, tensors, dim);
        };
        return wrap(dispatch_concatenate_out(_r.tensor(2), _r.tensorlist(0), _r.toInt64(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::concatenate.names(Tensor[] tensors, Dimname dim) -> Tensor
        
        auto dispatch_concatenate = [](at::TensorList tensors, at::Dimname dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::concatenate(tensors, dim);
        };
        return wrap(dispatch_concatenate(_r.tensorlist(0), _r.dimname(1)));
      } else {
        // aten::concatenate.names_out(Tensor[] tensors, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_concatenate_out = [](at::Tensor out, at::TensorList tensors, at::Dimname dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::concatenate_out(out, tensors, dim);
        };
        return wrap(dispatch_concatenate_out(_r.tensor(2), _r.tensorlist(0), _r.dimname(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// block_diag
static PyObject * THPVariable_block_diag(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "block_diag(TensorList tensors)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::block_diag(Tensor[] tensors) -> Tensor
  
  auto dispatch_block_diag = [](at::TensorList tensors) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::block_diag(tensors);
  };
  return wrap(dispatch_block_diag(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ceil
static PyObject * THPVariable_ceil(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ceil(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::ceil(Tensor self) -> Tensor
    
    auto dispatch_ceil = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.ceil();
    };
    return wrap(dispatch_ceil(_r.tensor(0)));
  } else {
    // aten::ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_ceil_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::ceil_out(out, self);
    };
    return wrap(dispatch_ceil_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ceil_
static PyObject * THPVariable_ceil_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ceil_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::ceil_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_ceil_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.ceil_();
  };
  return wrap(dispatch_ceil_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cudnn_is_acceptable
static PyObject * THPVariable_cudnn_is_acceptable(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_is_acceptable(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::cudnn_is_acceptable(Tensor self) -> bool
  
  auto dispatch_cudnn_is_acceptable = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return at::cudnn_is_acceptable(self);
  };
  return wrap(dispatch_cudnn_is_acceptable(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// conv_transpose2d
static PyObject * THPVariable_conv_transpose2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv_transpose2d(Tensor input, Tensor weight, Tensor? bias=None, SymIntArrayRef[2] stride=1, SymIntArrayRef[2] padding=0, SymIntArrayRef[2] output_padding=0, SymInt groups=1, SymIntArrayRef[2] dilation=1)",
  }, /*traceable=*/false);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] output_padding=0, SymInt groups=1, SymInt[2] dilation=1) -> Tensor
  
  auto dispatch_conv_transpose2d = [](const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymInt groups, c10::SymIntArrayRef dilation) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::conv_transpose2d_symint(input, weight, bias, stride, padding, output_padding, groups, dilation);
  };
  return wrap(dispatch_conv_transpose2d(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.symintlist(4), _r.symintlist(5), _r.toSymInt(6), _r.symintlist(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _copy_from
static PyObject * THPVariable__copy_from(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_copy_from(Tensor input, Tensor dst, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> Tensor
  
  auto dispatch__copy_from = [](const at::Tensor & self, const at::Tensor & dst, bool non_blocking) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_copy_from(self, dst, non_blocking);
  };
  return wrap(dispatch__copy_from(_r.tensor(0), _r.tensor(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _copy_from_and_resize
static PyObject * THPVariable__copy_from_and_resize(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_copy_from_and_resize(Tensor input, Tensor dst)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_copy_from_and_resize(Tensor self, Tensor dst) -> Tensor
  
  auto dispatch__copy_from_and_resize = [](const at::Tensor & self, const at::Tensor & dst) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_copy_from_and_resize(self, dst);
  };
  return wrap(dispatch__copy_from_and_resize(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cosh
static PyObject * THPVariable_cosh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cosh(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::cosh(Tensor self) -> Tensor
    
    auto dispatch_cosh = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.cosh();
    };
    return wrap(dispatch_cosh(_r.tensor(0)));
  } else {
    // aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_cosh_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::cosh_out(out, self);
    };
    return wrap(dispatch_cosh_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cosh_
static PyObject * THPVariable_cosh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cosh_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::cosh_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_cosh_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cosh_();
  };
  return wrap(dispatch_cosh_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cov
static PyObject * THPVariable_cov(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cov(Tensor input, *, int64_t correction=1, Tensor? fweights=None, Tensor? aweights=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::cov(Tensor self, *, int correction=1, Tensor? fweights=None, Tensor? aweights=None) -> Tensor
  
  auto dispatch_cov = [](const at::Tensor & self, int64_t correction, const ::std::optional<at::Tensor> & fweights, const ::std::optional<at::Tensor> & aweights) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cov(correction, fweights, aweights);
  };
  return wrap(dispatch_cov(_r.tensor(0), _r.toInt64(1), _r.optionalTensor(2), _r.optionalTensor(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cudnn_convolution
static PyObject * THPVariable_cudnn_convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_convolution(Tensor input, Tensor weight, SymIntArrayRef padding, SymIntArrayRef stride, SymIntArrayRef dilation, SymInt groups, bool benchmark, bool deterministic, bool allow_tf32, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<10> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(9)) {
    // aten::cudnn_convolution(Tensor self, Tensor weight, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
    
    auto dispatch_cudnn_convolution = [](const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, bool benchmark, bool deterministic, bool allow_tf32) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::cudnn_convolution_symint(self, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
    };
    return wrap(dispatch_cudnn_convolution(_r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.symintlist(3), _r.symintlist(4), _r.toSymInt(5), _r.toBool(6), _r.toBool(7), _r.toBool(8)));
  } else {
    // aten::cudnn_convolution.out(Tensor self, Tensor weight, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, bool deterministic, bool allow_tf32, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_cudnn_convolution_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, bool benchmark, bool deterministic, bool allow_tf32) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::cudnn_convolution_symint_out(out, self, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
    };
    return wrap(dispatch_cudnn_convolution_out(_r.tensor(9), _r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.symintlist(3), _r.symintlist(4), _r.toSymInt(5), _r.toBool(6), _r.toBool(7), _r.toBool(8)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// cummax
static PyObject * THPVariable_cummax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_cummax_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_cummax_out_structseq();
  static PythonArgParser parser({
    "cummax(Tensor input, int64_t dim, *, TensorList[2] out=None)",
    "cummax(Tensor input, Dimname dim, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)
        
        auto dispatch_cummax = [](const at::Tensor & self, int64_t dim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.cummax(dim);
        };
        return wrap(NamedTuple, dispatch_cummax(_r.tensor(0), _r.toInt64(1)));
      } else {
        // aten::cummax.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(2);
        auto dispatch_cummax_out = [](at::Tensor & values, at::Tensor & indices, const at::Tensor & self, int64_t dim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::cummax_out(values, indices, self, dim);
        };
        return wrap(NamedTuple1, dispatch_cummax_out(out[0], out[1], _r.tensor(0), _r.toInt64(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::cummax.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)
        
        auto dispatch_cummax = [](const at::Tensor & self, at::Dimname dim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.cummax(dim);
        };
        return wrap(NamedTuple, dispatch_cummax(_r.tensor(0), _r.dimname(1)));
      } else {
        // aten::cummax.dimname_out(Tensor self, Dimname dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(2);
        auto dispatch_cummax_out = [](at::Tensor & values, at::Tensor & indices, const at::Tensor & self, at::Dimname dim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::cummax_out(values, indices, self, dim);
        };
        return wrap(NamedTuple1, dispatch_cummax_out(out[0], out[1], _r.tensor(0), _r.dimname(1)));
      }
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
  static PyTypeObject* NamedTuple1 = generated::get_cummin_out_structseq();
  static PythonArgParser parser({
    "cummin(Tensor input, int64_t dim, *, TensorList[2] out=None)",
    "cummin(Tensor input, Dimname dim, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::cummin(Tensor self, int dim) -> (Tensor values, Tensor indices)
        
        auto dispatch_cummin = [](const at::Tensor & self, int64_t dim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.cummin(dim);
        };
        return wrap(NamedTuple, dispatch_cummin(_r.tensor(0), _r.toInt64(1)));
      } else {
        // aten::cummin.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(2);
        auto dispatch_cummin_out = [](at::Tensor & values, at::Tensor & indices, const at::Tensor & self, int64_t dim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::cummin_out(values, indices, self, dim);
        };
        return wrap(NamedTuple1, dispatch_cummin_out(out[0], out[1], _r.tensor(0), _r.toInt64(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::cummin.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)
        
        auto dispatch_cummin = [](const at::Tensor & self, at::Dimname dim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.cummin(dim);
        };
        return wrap(NamedTuple, dispatch_cummin(_r.tensor(0), _r.dimname(1)));
      } else {
        // aten::cummin.dimname_out(Tensor self, Dimname dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(2);
        auto dispatch_cummin_out = [](at::Tensor & values, at::Tensor & indices, const at::Tensor & self, at::Dimname dim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::cummin_out(values, indices, self, dim);
        };
        return wrap(NamedTuple1, dispatch_cummin_out(out[0], out[1], _r.tensor(0), _r.dimname(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cummin_helper
static PyObject * THPVariable__cummin_helper(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cummin_helper(Tensor input, Tensor values, Tensor indices, int64_t dim)",
  }, /*traceable=*/false);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cummin_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
  
  auto dispatch__cummin_helper = [](const at::Tensor & self, at::Tensor values, at::Tensor indices, int64_t dim) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_cummin_helper(self, values, indices, dim);
  };
  dispatch__cummin_helper(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// cumsum
static PyObject * THPVariable_cumsum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cumsum(Tensor input, int64_t dim, *, ScalarType? dtype=None, Tensor out=None)",
    "cumsum(Tensor input, Dimname dim, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_cumsum = [](const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.cumsum(dim, dtype);
        };
        return wrap(dispatch_cumsum(_r.tensor(0), _r.toInt64(1), _r.scalartypeOptional(2)));
      } else {
        // aten::cumsum.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_cumsum_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::cumsum_out(out, self, dim, dtype);
        };
        return wrap(dispatch_cumsum_out(_r.tensor(3), _r.tensor(0), _r.toInt64(1), _r.scalartypeOptional(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::cumsum.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_cumsum = [](const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.cumsum(dim, dtype);
        };
        return wrap(dispatch_cumsum(_r.tensor(0), _r.dimname(1), _r.scalartypeOptional(2)));
      } else {
        // aten::cumsum.dimname_out(Tensor self, Dimname dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_cumsum_out = [](at::Tensor out, const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::cumsum_out(out, self, dim, dtype);
        };
        return wrap(dispatch_cumsum_out(_r.tensor(3), _r.tensor(0), _r.dimname(1), _r.scalartypeOptional(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// cumulative_trapezoid
static PyObject * THPVariable_cumulative_trapezoid(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cumulative_trapezoid(Tensor y, Tensor x, *, int64_t dim=-1)",
    "cumulative_trapezoid(Tensor y, *, Scalar dx=1, int64_t dim=-1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::cumulative_trapezoid.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor
      
      auto dispatch_cumulative_trapezoid = [](const at::Tensor & y, const at::Tensor & x, int64_t dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::cumulative_trapezoid(y, x, dim);
      };
      return wrap(dispatch_cumulative_trapezoid(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
    }
    case 1: {
      // aten::cumulative_trapezoid.dx(Tensor y, *, Scalar dx=1, int dim=-1) -> Tensor
      
      auto dispatch_cumulative_trapezoid = [](const at::Tensor & y, const at::Scalar & dx, int64_t dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::cumulative_trapezoid(y, dx, dim);
      };
      return wrap(dispatch_cumulative_trapezoid(_r.tensor(0), _r.scalar(1), _r.toInt64(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _ctc_loss
static PyObject * THPVariable__ctc_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_ctc_loss(Tensor log_probs, Tensor targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank=0, bool zero_infinity=False)",
    "_ctc_loss(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int64_t blank=0, bool zero_infinity=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)
      
      auto dispatch__ctc_loss = [](const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, bool zero_infinity) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
      };
      return wrap(dispatch__ctc_loss(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.intlist(3), _r.toInt64(4), _r.toBool(5)));
    }
    case 1: {
      // aten::_ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)
      
      auto dispatch__ctc_loss = [](const at::Tensor & log_probs, const at::Tensor & targets, const at::Tensor & input_lengths, const at::Tensor & target_lengths, int64_t blank, bool zero_infinity) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
      };
      return wrap(dispatch__ctc_loss(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.toInt64(4), _r.toBool(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// diag_embed
static PyObject * THPVariable_diag_embed(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diag_embed(Tensor input, int64_t offset=0, int64_t dim1=-2, int64_t dim2=-1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor
  
  auto dispatch_diag_embed = [](const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.diag_embed(offset, dim1, dim2);
  };
  return wrap(dispatch_diag_embed(_r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toInt64(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// diagflat
static PyObject * THPVariable_diagflat(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diagflat(Tensor input, int64_t offset=0)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::diagflat(Tensor self, int offset=0) -> Tensor
  
  auto dispatch_diagflat = [](const at::Tensor & self, int64_t offset) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.diagflat(offset);
  };
  return wrap(dispatch_diagflat(_r.tensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// diagonal
static PyObject * THPVariable_diagonal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diagonal(Tensor input, *, Dimname outdim, Dimname dim1, Dimname dim2, int64_t offset=0)",
    "diagonal(Tensor input, int64_t offset=0, int64_t dim1=0, int64_t dim2=1)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::diagonal.Dimname(Tensor(a) self, *, Dimname outdim, Dimname dim1, Dimname dim2, int offset=0) -> Tensor(a)
      
      auto dispatch_diagonal = [](const at::Tensor & self, at::Dimname outdim, at::Dimname dim1, at::Dimname dim2, int64_t offset) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.diagonal(outdim, dim1, dim2, offset);
      };
      return wrap(dispatch_diagonal(_r.tensor(0), _r.dimname(1), _r.dimname(2), _r.dimname(3), _r.toInt64(4)));
    }
    case 1: {
      // aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)
      
      auto dispatch_diagonal = [](const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.diagonal(offset, dim1, dim2);
      };
      return wrap(dispatch_diagonal(_r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toInt64(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// diff
static PyObject * THPVariable_diff(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diff(Tensor input, int64_t n=1, int64_t dim=-1, Tensor? prepend=None, Tensor? append=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(5)) {
    // aten::diff(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None) -> Tensor
    
    auto dispatch_diff = [](const at::Tensor & self, int64_t n, int64_t dim, const ::std::optional<at::Tensor> & prepend, const ::std::optional<at::Tensor> & append) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.diff(n, dim, prepend, append);
    };
    return wrap(dispatch_diff(_r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.optionalTensor(3), _r.optionalTensor(4)));
  } else {
    // aten::diff.out(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_diff_out = [](at::Tensor out, const at::Tensor & self, int64_t n, int64_t dim, const ::std::optional<at::Tensor> & prepend, const ::std::optional<at::Tensor> & append) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::diff_out(out, self, n, dim, prepend, append);
    };
    return wrap(dispatch_diff_out(_r.tensor(5), _r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.optionalTensor(3), _r.optionalTensor(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// divide
static PyObject * THPVariable_divide(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "divide(Tensor input, Tensor other, *, Tensor out=None)",
    "divide(Tensor input, Tensor other, *, c10::string_view? rounding_mode, Tensor out=None)",
    "divide(Tensor input, Scalar other)",
    "divide(Tensor input, Scalar other, *, c10::string_view? rounding_mode)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::divide.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_divide = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.divide(other);
        };
        return wrap(dispatch_divide(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_divide_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::divide_out(out, self, other);
        };
        return wrap(dispatch_divide_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::divide.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
        
        auto dispatch_divide = [](const at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.divide(other, rounding_mode);
        };
        return wrap(dispatch_divide(_r.tensor(0), _r.tensor(1), _r.stringViewOptional(2)));
      } else {
        // aten::divide.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_divide_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::divide_out(out, self, other, rounding_mode);
        };
        return wrap(dispatch_divide_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.stringViewOptional(2)));
      }
    }
    case 2: {
      // aten::divide.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_divide = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.divide(other);
      };
      return wrap(dispatch_divide(_r.tensor(0), _r.scalar(1)));
    }
    case 3: {
      // aten::divide.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor
      
      auto dispatch_divide = [](const at::Tensor & self, const at::Scalar & other, ::std::optional<c10::string_view> rounding_mode) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.divide(other, rounding_mode);
      };
      return wrap(dispatch_divide(_r.tensor(0), _r.scalar(1), _r.stringViewOptional(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// dot
static PyObject * THPVariable_dot(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dot(Tensor input, Tensor tensor, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::dot(Tensor self, Tensor tensor) -> Tensor
    
    auto dispatch_dot = [](const at::Tensor & self, const at::Tensor & tensor) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.dot(tensor);
    };
    return wrap(dispatch_dot(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::dot.out(Tensor self, Tensor tensor, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_dot_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & tensor) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::dot_out(out, self, tensor);
    };
    return wrap(dispatch_dot_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// vdot
static PyObject * THPVariable_vdot(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "vdot(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::vdot(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_vdot = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.vdot(other);
    };
    return wrap(dispatch_vdot(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::vdot.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_vdot_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::vdot_out(out, self, other);
    };
    return wrap(dispatch_vdot_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// einsum
static PyObject * THPVariable_einsum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "einsum(c10::string_view equation, TensorList tensors, *, IntArrayRef? path=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::einsum(str equation, Tensor[] tensors, *, int[]? path=None) -> Tensor
  
  auto dispatch_einsum = [](c10::string_view equation, at::TensorList tensors, at::OptionalIntArrayRef path) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::einsum(equation, tensors, path);
  };
  return wrap(dispatch_einsum(_r.stringView(0), _r.tensorlist(1), _r.intlistOptional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// embedding
static PyObject * THPVariable_embedding(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
  
  auto dispatch_embedding = [](const at::Tensor & weight, const at::Tensor & indices, c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::embedding_symint(weight, indices, padding_idx, scale_grad_by_freq, sparse);
  };
  return wrap(dispatch_embedding(_r.tensor(0), _r.tensor(1), _r.toSymInt(2), _r.toBool(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// embedding_renorm_
static PyObject * THPVariable_embedding_renorm_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "embedding_renorm_(Tensor input, Tensor indices, double max_norm, double norm_type)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::embedding_renorm_(Tensor(a!) self, Tensor indices, float max_norm, float norm_type) -> Tensor(a!)
  
  auto dispatch_embedding_renorm_ = [](at::Tensor self, const at::Tensor & indices, double max_norm, double norm_type) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::embedding_renorm_(self, indices, max_norm, norm_type);
  };
  return wrap(dispatch_embedding_renorm_(_r.tensor(0), _r.tensor(1), _r.toDouble(2), _r.toDouble(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _rowwise_prune
static PyObject * THPVariable__rowwise_prune(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_rowwise_prune(Tensor weight, Tensor mask, ScalarType compressed_indices_dtype)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_rowwise_prune(Tensor weight, Tensor mask, ScalarType compressed_indices_dtype) -> (Tensor, Tensor)
  
  auto dispatch__rowwise_prune = [](const at::Tensor & weight, const at::Tensor & mask, at::ScalarType compressed_indices_dtype) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_rowwise_prune(weight, mask, compressed_indices_dtype);
  };
  return wrap(dispatch__rowwise_prune(_r.tensor(0), _r.tensor(1), _r.scalartype(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// empty_permuted
static PyObject * THPVariable_empty_permuted(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "empty_permuted(SymIntArrayRef size, IntArrayRef physical_layout, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::empty_permuted(SymInt[] size, int[] physical_layout, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(2))
      .device(_r.deviceWithDefault(4, torch::tensors::get_default_device()))
      .layout(_r.layoutOptional(3))
      .requires_grad(_r.toBool(6))
      .pinned_memory(_r.toBool(5));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch_empty_permuted = [](c10::SymIntArrayRef size, at::IntArrayRef physical_layout, at::TensorOptions options) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::empty_permuted_symint(size, physical_layout, options);
  };
  return wrap(dispatch_empty_permuted(_r.symintlist(0), _r.intlist(1), options));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _empty_affine_quantized
static PyObject * THPVariable__empty_affine_quantized(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_empty_affine_quantized(SymIntArrayRef size, *, double scale=1, int64_t zero_point=0, MemoryFormat? memory_format=c10::MemoryFormat::Contiguous, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_empty_affine_quantized(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, float scale=1, int zero_point=0, MemoryFormat? memory_format=contiguous_format) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(4))
      .device(_r.deviceWithDefault(6, torch::tensors::get_default_device()))
      .layout(_r.layoutOptional(5))
      .requires_grad(_r.toBool(8))
      .pinned_memory(_r.toBool(7));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch__empty_affine_quantized = [](c10::SymIntArrayRef size, at::TensorOptions options, double scale, int64_t zero_point, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::_empty_affine_quantized_symint(size, options, scale, zero_point, memory_format);
  };
  return wrap(dispatch__empty_affine_quantized(_r.symintlist(0), options, _r.toDouble(1), _r.toInt64(2), _r.memoryformat(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// empty_quantized
static PyObject * THPVariable_empty_quantized(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "empty_quantized(IntArrayRef size, Tensor qtensor, *, MemoryFormat? memory_format=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::empty_quantized(int[] size, Tensor qtensor, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(3))
      .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
      .layout(_r.layoutOptional(4))
      .requires_grad(_r.toBool(7))
      .pinned_memory(_r.toBool(6));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch_empty_quantized = [](at::IntArrayRef size, const at::Tensor & qtensor, at::TensorOptions options, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::empty_quantized(size, qtensor, options, memory_format);
  };
  return wrap(dispatch_empty_quantized(_r.intlist(0), _r.tensor(1), options, _r.memoryformatOptional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// exp2
static PyObject * THPVariable_exp2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "exp2(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::exp2(Tensor self) -> Tensor
    
    auto dispatch_exp2 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.exp2();
    };
    return wrap(dispatch_exp2(_r.tensor(0)));
  } else {
    // aten::exp2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_exp2_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::exp2_out(out, self);
    };
    return wrap(dispatch_exp2_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// exp2_
static PyObject * THPVariable_exp2_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "exp2_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::exp2_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_exp2_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.exp2_();
  };
  return wrap(dispatch_exp2_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// expm1
static PyObject * THPVariable_expm1(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "expm1(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::expm1(Tensor self) -> Tensor
    
    auto dispatch_expm1 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.expm1();
    };
    return wrap(dispatch_expm1(_r.tensor(0)));
  } else {
    // aten::expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_expm1_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::expm1_out(out, self);
    };
    return wrap(dispatch_expm1_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// expm1_
static PyObject * THPVariable_expm1_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "expm1_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::expm1_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_expm1_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.expm1_();
  };
  return wrap(dispatch_expm1_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// eye
static PyObject * THPVariable_eye(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "eye(SymInt n, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "eye(SymInt n, SymInt m, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(1)) {
        // aten::eye(SymInt n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(2))
            .device(_r.deviceWithDefault(4, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(3))
            .requires_grad(_r.toBool(6))
            .pinned_memory(_r.toBool(5));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_eye = [](c10::SymInt n, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::eye_symint(n, options);
        };
        return wrap(dispatch_eye(_r.toSymInt(0), options));
      } else {
        // aten::eye.out(SymInt n, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(1), _r.scalartypeOptional(2),
                               _r.isNone(2), _r.layoutOptional(3),
                               _r.deviceWithDefault(4, torch::tensors::get_default_device()), _r.isNone(4));
        
        auto dispatch_eye_out = [](at::Tensor out, c10::SymInt n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::eye_symint_out(out, n);
        };
        return wrap(dispatch_eye_out(_r.tensor(1), _r.toSymInt(0)).set_requires_grad(_r.toBool(6)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::eye.m(SymInt n, SymInt m, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(3))
            .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(4))
            .requires_grad(_r.toBool(7))
            .pinned_memory(_r.toBool(6));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_eye = [](c10::SymInt n, c10::SymInt m, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::eye_symint(n, m, options);
        };
        return wrap(dispatch_eye(_r.toSymInt(0), _r.toSymInt(1), options));
      } else {
        // aten::eye.m_out(SymInt n, SymInt m, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(2), _r.scalartypeOptional(3),
                               _r.isNone(3), _r.layoutOptional(4),
                               _r.deviceWithDefault(5, torch::tensors::get_default_device()), _r.isNone(5));
        
        auto dispatch_eye_out = [](at::Tensor out, c10::SymInt n, c10::SymInt m) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::eye_symint_out(out, n, m);
        };
        return wrap(dispatch_eye_out(_r.tensor(2), _r.toSymInt(0), _r.toSymInt(1)).set_requires_grad(_r.toBool(7)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// full
static PyObject * THPVariable_full(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "full(IntArrayRef size, Scalar fill_value, *, DimnameList? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "full(SymIntArrayRef size, Scalar fill_value, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::full.names(int[] size, Scalar fill_value, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      auto __names = _r.toDimnameListOptional(2);
      ::std::optional<DimnameList> names = __names ? ::std::make_optional(DimnameList(__names.value())) : ::std::nullopt;
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(3))
          .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(4))
          .requires_grad(_r.toBool(7))
          .pinned_memory(_r.toBool(6));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_full = [](at::IntArrayRef size, const at::Scalar & fill_value, ::std::optional<at::DimnameList> names, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::full(size, fill_value, names, options);
      };
      return wrap(dispatch_full(_r.intlist(0), _r.scalar(1), names, options));
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(3))
            .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(4))
            .requires_grad(_r.toBool(7))
            .pinned_memory(_r.toBool(6));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_full = [](c10::SymIntArrayRef size, const at::Scalar & fill_value, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::full_symint(size, fill_value, options);
        };
        return wrap(dispatch_full(_r.symintlist(0), _r.scalar(1), options));
      } else {
        // aten::full.out(SymInt[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(2), _r.scalartypeOptional(3),
                               _r.isNone(3), _r.layoutOptional(4),
                               _r.deviceWithDefault(5, torch::tensors::get_default_device()), _r.isNone(5));
        
        auto dispatch_full_out = [](at::Tensor out, c10::SymIntArrayRef size, const at::Scalar & fill_value) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::full_symint_out(out, size, fill_value);
        };
        return wrap(dispatch_full_out(_r.tensor(2), _r.symintlist(0), _r.scalar(1)).set_requires_grad(_r.toBool(7)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// from_file
static PyObject * THPVariable_from_file(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "from_file(c10::string_view filename, bool? shared=None, int64_t? size=0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::from_file(str filename, bool? shared=None, int? size=0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(3))
      .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
      .layout(_r.layoutOptional(4))
      .requires_grad(_r.toBool(7))
      .pinned_memory(_r.toBool(6));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch_from_file = [](c10::string_view filename, ::std::optional<bool> shared, ::std::optional<int64_t> size, at::TensorOptions options) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::from_file(filename, shared, size, options);
  };
  return wrap(dispatch_from_file(_r.stringView(0), _r.toBoolOptional(1), _r.toInt64(2), options));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// gcd
static PyObject * THPVariable_gcd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gcd(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::gcd(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_gcd = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.gcd(other);
    };
    return wrap(dispatch_gcd(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::gcd.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_gcd_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::gcd_out(out, self, other);
    };
    return wrap(dispatch_gcd_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// gcd_
static PyObject * THPVariable_gcd_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gcd_(Tensor input, Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::gcd_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  
  auto dispatch_gcd_ = [](at::Tensor self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.gcd_(other);
  };
  return wrap(dispatch_gcd_(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// lcm
static PyObject * THPVariable_lcm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lcm(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::lcm(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_lcm = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.lcm(other);
    };
    return wrap(dispatch_lcm(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::lcm.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_lcm_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::lcm_out(out, self, other);
    };
    return wrap(dispatch_lcm_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// lcm_
static PyObject * THPVariable_lcm_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lcm_(Tensor input, Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::lcm_(Tensor(a!) self, Tensor other) -> Tensor(a!)
  
  auto dispatch_lcm_ = [](at::Tensor self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.lcm_(other);
  };
  return wrap(dispatch_lcm_(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// hamming_window
static PyObject * THPVariable_hamming_window(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hamming_window(int64_t window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "hamming_window(int64_t window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "hamming_window(int64_t window_length, bool periodic, double alpha, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "hamming_window(int64_t window_length, bool periodic, double alpha, double beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::hamming_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(1))
          .device(_r.deviceWithDefault(3, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(2))
          .requires_grad(_r.toBool(5))
          .pinned_memory(_r.toBool(4));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_hamming_window = [](int64_t window_length, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::hamming_window(window_length, options);
      };
      return wrap(dispatch_hamming_window(_r.toInt64(0), options));
    }
    case 1: {
      // aten::hamming_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(2))
          .device(_r.deviceWithDefault(4, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(3))
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_hamming_window = [](int64_t window_length, bool periodic, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::hamming_window(window_length, periodic, options);
      };
      return wrap(dispatch_hamming_window(_r.toInt64(0), _r.toBool(1), options));
    }
    case 2: {
      // aten::hamming_window.periodic_alpha(int window_length, bool periodic, float alpha, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(3))
          .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(4))
          .requires_grad(_r.toBool(7))
          .pinned_memory(_r.toBool(6));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_hamming_window = [](int64_t window_length, bool periodic, double alpha, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::hamming_window(window_length, periodic, alpha, options);
      };
      return wrap(dispatch_hamming_window(_r.toInt64(0), _r.toBool(1), _r.toDouble(2), options));
    }
    case 3: {
      // aten::hamming_window.periodic_alpha_beta(int window_length, bool periodic, float alpha, float beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(4))
          .device(_r.deviceWithDefault(6, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(5))
          .requires_grad(_r.toBool(8))
          .pinned_memory(_r.toBool(7));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_hamming_window = [](int64_t window_length, bool periodic, double alpha, double beta, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::hamming_window(window_length, periodic, alpha, beta, options);
      };
      return wrap(dispatch_hamming_window(_r.toInt64(0), _r.toBool(1), _r.toDouble(2), _r.toDouble(3), options));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hinge_embedding_loss
static PyObject * THPVariable_hinge_embedding_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hinge_embedding_loss(Tensor input, Tensor target, double margin=1.0, int64_t reduction=at::Reduction::Mean)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::hinge_embedding_loss(Tensor self, Tensor target, float margin=1.0, int reduction=Mean) -> Tensor
  
  auto dispatch_hinge_embedding_loss = [](const at::Tensor & self, const at::Tensor & target, double margin, int64_t reduction) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::hinge_embedding_loss(self, target, margin, reduction);
  };
  return wrap(dispatch_hinge_embedding_loss(_r.tensor(0), _r.tensor(1), _r.toDouble(2), _r.toInt64(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// group_norm
static PyObject * THPVariable_group_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "group_norm(Tensor input, int64_t num_groups, Tensor? weight=None, Tensor? bias=None, double eps=1e-05, bool cudnn_enabled=True)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor
  
  auto dispatch_group_norm = [](const at::Tensor & input, int64_t num_groups, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps, bool cudnn_enabled) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::group_norm(input, num_groups, weight, bias, eps, cudnn_enabled);
  };
  return wrap(dispatch_group_norm(_r.tensor(0), _r.toInt64(1), _r.optionalTensor(2), _r.optionalTensor(3), _r.toDouble(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// native_group_norm
static PyObject * THPVariable_native_group_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int64_t group, double eps)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)
  
  auto dispatch_native_group_norm = [](const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, c10::SymInt N, c10::SymInt C, c10::SymInt HxW, int64_t group, double eps) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::native_group_norm_symint(input, weight, bias, N, C, HxW, group, eps);
  };
  return wrap(dispatch_native_group_norm(_r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2), _r.toSymInt(3), _r.toSymInt(4), _r.toSymInt(5), _r.toInt64(6), _r.toDouble(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _fft_c2c
static PyObject * THPVariable__fft_c2c(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_fft_c2c(Tensor input, SymIntArrayRef dim, int64_t normalization, bool forward, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::_fft_c2c(Tensor self, SymInt[] dim, int normalization, bool forward) -> Tensor
    
    auto dispatch__fft_c2c = [](const at::Tensor & self, c10::SymIntArrayRef dim, int64_t normalization, bool forward) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_fft_c2c_symint(self, dim, normalization, forward);
    };
    return wrap(dispatch__fft_c2c(_r.tensor(0), _r.symintlist(1), _r.toInt64(2), _r.toBool(3)));
  } else {
    // aten::_fft_c2c.out(Tensor self, SymInt[] dim, int normalization, bool forward, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__fft_c2c_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef dim, int64_t normalization, bool forward) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_fft_c2c_symint_out(out, self, dim, normalization, forward);
    };
    return wrap(dispatch__fft_c2c_out(_r.tensor(4), _r.tensor(0), _r.symintlist(1), _r.toInt64(2), _r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cufft_set_plan_cache_max_size
static PyObject * THPVariable__cufft_set_plan_cache_max_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cufft_set_plan_cache_max_size(DeviceIndex device_index, int64_t max_size)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cufft_set_plan_cache_max_size(DeviceIndex device_index, int max_size) -> ()
  
  auto dispatch__cufft_set_plan_cache_max_size = [](at::DeviceIndex device_index, int64_t max_size) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_cufft_set_plan_cache_max_size(device_index, max_size);
  };
  dispatch__cufft_set_plan_cache_max_size(_r.toInt64(0), _r.toInt64(1));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _unsafe_index
static PyObject * THPVariable__unsafe_index(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_unsafe_index(Tensor input, c10::List<::std::optional<Tensor>> indices)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_unsafe_index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
  
  auto dispatch__unsafe_index = [](const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_unsafe_index(self, indices);
  };
  return wrap(dispatch__unsafe_index(_r.tensor(0), _r.list_of_optional_tensors(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// index_copy
static PyObject * THPVariable_index_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_copy(Tensor input, int64_t dim, Tensor index, Tensor source, *, Tensor out=None)",
    "index_copy(Tensor input, Dimname dim, Tensor index, Tensor source)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
        
        auto dispatch_index_copy = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.index_copy(dim, index, source);
        };
        return wrap(dispatch_index_copy(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3)));
      } else {
        // aten::index_copy.out(Tensor self, int dim, Tensor index, Tensor source, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_index_copy_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::index_copy_out(out, self, dim, index, source);
        };
        return wrap(dispatch_index_copy_out(_r.tensor(4), _r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3)));
      }
    }
    case 1: {
      // aten::index_copy.dimname(Tensor self, Dimname dim, Tensor index, Tensor source) -> Tensor
      
      auto dispatch_index_copy = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_copy(dim, index, source);
      };
      return wrap(dispatch_index_copy(_r.tensor(0), _r.dimname(1), _r.tensor(2), _r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// instance_norm
static PyObject * THPVariable_instance_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor
  
  auto dispatch_instance_norm = [](const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::instance_norm(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
  };
  return wrap(dispatch_instance_norm(_r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2), _r.optionalTensor(3), _r.optionalTensor(4), _r.toBool(5), _r.toDouble(6), _r.toDouble(7), _r.toBool(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// isclose
static PyObject * THPVariable_isclose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "isclose(Tensor input, Tensor other, double rtol=1e-05, double atol=1e-08, bool equal_nan=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor
  
  auto dispatch_isclose = [](const at::Tensor & self, const at::Tensor & other, double rtol, double atol, bool equal_nan) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.isclose(other, rtol, atol, equal_nan);
  };
  return wrap(dispatch_isclose(_r.tensor(0), _r.tensor(1), _r.toDouble(2), _r.toDouble(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_floating_point
static PyObject * THPVariable_is_floating_point(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_floating_point(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::is_floating_point(Tensor self) -> bool
  
  auto dispatch_is_floating_point = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_floating_point();
  };
  return wrap(dispatch_is_floating_point(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _is_zerotensor
static PyObject * THPVariable__is_zerotensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_is_zerotensor(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_is_zerotensor(Tensor self) -> bool
  
  auto dispatch__is_zerotensor = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self._is_zerotensor();
  };
  return wrap(dispatch__is_zerotensor(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_neg
static PyObject * THPVariable_is_neg(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_neg(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::is_neg(Tensor self) -> bool
  
  auto dispatch_is_neg = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_neg();
  };
  return wrap(dispatch_is_neg(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_nonzero
static PyObject * THPVariable_is_nonzero(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_nonzero(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::is_nonzero(Tensor self) -> bool
  
  auto dispatch_is_nonzero = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_nonzero();
  };
  return wrap(dispatch_is_nonzero(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_same_size
static PyObject * THPVariable_is_same_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_same_size(Tensor input, Tensor other)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::is_same_size(Tensor self, Tensor other) -> bool
  
  auto dispatch_is_same_size = [](const at::Tensor & self, const at::Tensor & other) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_same_size(other);
  };
  return wrap(dispatch_is_same_size(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_signed
static PyObject * THPVariable_is_signed(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_signed(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::is_signed(Tensor self) -> bool
  
  auto dispatch_is_signed = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_signed();
  };
  return wrap(dispatch_is_signed(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// kl_div
static PyObject * THPVariable_kl_div(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "kl_div(Tensor input, Tensor target, int64_t reduction=at::Reduction::Mean, *, bool log_target=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::kl_div(Tensor self, Tensor target, int reduction=Mean, *, bool log_target=False) -> Tensor
  
  auto dispatch_kl_div = [](const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::kl_div(self, target, reduction, log_target);
  };
  return wrap(dispatch_kl_div(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// kthvalue
static PyObject * THPVariable_kthvalue(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_kthvalue_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_kthvalue_out_structseq();
  static PythonArgParser parser({
    "kthvalue(Tensor input, int64_t k, int64_t dim=-1, bool keepdim=False, *, TensorList[2] out=None)",
    "kthvalue(Tensor input, int64_t k, Dimname dim, bool keepdim=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // aten::kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
        
        auto dispatch_kthvalue = [](const at::Tensor & self, int64_t k, int64_t dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.kthvalue(k, dim, keepdim);
        };
        return wrap(NamedTuple, dispatch_kthvalue(_r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toBool(3)));
      } else {
        // aten::kthvalue.values(Tensor self, int k, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(4);
        auto dispatch_kthvalue_out = [](at::Tensor & values, at::Tensor & indices, const at::Tensor & self, int64_t k, int64_t dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::kthvalue_out(values, indices, self, k, dim, keepdim);
        };
        return wrap(NamedTuple1, dispatch_kthvalue_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toBool(3)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::kthvalue.dimname(Tensor self, int k, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        
        auto dispatch_kthvalue = [](const at::Tensor & self, int64_t k, at::Dimname dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.kthvalue(k, dim, keepdim);
        };
        return wrap(NamedTuple, dispatch_kthvalue(_r.tensor(0), _r.toInt64(1), _r.dimname(2), _r.toBool(3)));
      } else {
        // aten::kthvalue.dimname_out(Tensor self, int k, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(4);
        auto dispatch_kthvalue_out = [](at::Tensor & values, at::Tensor & indices, const at::Tensor & self, int64_t k, at::Dimname dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::kthvalue_out(values, indices, self, k, dim, keepdim);
        };
        return wrap(NamedTuple1, dispatch_kthvalue_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.dimname(2), _r.toBool(3)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rms_norm
static PyObject * THPVariable_rms_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rms_norm(Tensor input, SymIntArrayRef normalized_shape, Tensor? weight=None, double? eps=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::rms_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, float? eps=None) -> Tensor
  
  auto dispatch_rms_norm = [](const at::Tensor & input, c10::SymIntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, ::std::optional<double> eps) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::rms_norm_symint(input, normalized_shape, weight, eps);
  };
  return wrap(dispatch_rms_norm(_r.tensor(0), _r.symintlist(1), _r.optionalTensor(2), _r.toDoubleOptional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cslt_sparse_mm_search
static PyObject * THPVariable__cslt_sparse_mm_search(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cslt_sparse_mm_search(Tensor compressed_A, Tensor dense_B, Tensor? bias=None, Tensor? alpha=None, ScalarType? out_dtype=None, bool transpose_result=False)",
  }, /*traceable=*/false);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cslt_sparse_mm_search(Tensor compressed_A, Tensor dense_B, Tensor? bias=None, Tensor? alpha=None, ScalarType? out_dtype=None, bool transpose_result=False) -> int
  
  auto dispatch__cslt_sparse_mm_search = [](const at::Tensor & compressed_A, const at::Tensor & dense_B, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & alpha, ::std::optional<at::ScalarType> out_dtype, bool transpose_result) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return at::_cslt_sparse_mm_search(compressed_A, dense_B, bias, alpha, out_dtype, transpose_result);
  };
  return wrap(dispatch__cslt_sparse_mm_search(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.optionalTensor(3), _r.scalartypeOptional(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sparse_semi_structured_apply
static PyObject * THPVariable__sparse_semi_structured_apply(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_semi_structured_apply(Tensor input, Tensor thread_masks)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_sparse_semi_structured_apply(Tensor input, Tensor thread_masks) -> (Tensor, Tensor)
  
  auto dispatch__sparse_semi_structured_apply = [](const at::Tensor & input, const at::Tensor & thread_masks) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_sparse_semi_structured_apply(input, thread_masks);
  };
  return wrap(dispatch__sparse_semi_structured_apply(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sparse_semi_structured_apply_dense
static PyObject * THPVariable__sparse_semi_structured_apply_dense(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_semi_structured_apply_dense(Tensor input, Tensor thread_masks)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_sparse_semi_structured_apply_dense(Tensor input, Tensor thread_masks) -> Tensor
  
  auto dispatch__sparse_semi_structured_apply_dense = [](const at::Tensor & input, const at::Tensor & thread_masks) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sparse_semi_structured_apply_dense(input, thread_masks);
  };
  return wrap(dispatch__sparse_semi_structured_apply_dense(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fbgemm_linear_int8_weight_fp32_activation
static PyObject * THPVariable_fbgemm_linear_int8_weight_fp32_activation(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fbgemm_linear_int8_weight_fp32_activation(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::fbgemm_linear_int8_weight_fp32_activation(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor
  
  auto dispatch_fbgemm_linear_int8_weight_fp32_activation = [](const at::Tensor & input, const at::Tensor & weight, const at::Tensor & packed, const at::Tensor & col_offsets, const at::Scalar & weight_scale, const at::Scalar & weight_zero_point, const at::Tensor & bias) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::fbgemm_linear_int8_weight_fp32_activation(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
  };
  return wrap(dispatch_fbgemm_linear_int8_weight_fp32_activation(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.scalar(4), _r.scalar(5), _r.tensor(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fbgemm_linear_int8_weight
static PyObject * THPVariable_fbgemm_linear_int8_weight(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fbgemm_linear_int8_weight(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::fbgemm_linear_int8_weight(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor
  
  auto dispatch_fbgemm_linear_int8_weight = [](const at::Tensor & input, const at::Tensor & weight, const at::Tensor & packed, const at::Tensor & col_offsets, const at::Scalar & weight_scale, const at::Scalar & weight_zero_point, const at::Tensor & bias) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::fbgemm_linear_int8_weight(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
  };
  return wrap(dispatch_fbgemm_linear_int8_weight(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.scalar(4), _r.scalar(5), _r.tensor(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fbgemm_linear_quantize_weight
static PyObject * THPVariable_fbgemm_linear_quantize_weight(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fbgemm_linear_quantize_weight(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::fbgemm_linear_quantize_weight(Tensor input) -> (Tensor, Tensor, float, int)
  
  auto dispatch_fbgemm_linear_quantize_weight = [](const at::Tensor & input) -> ::std::tuple<at::Tensor,at::Tensor,double,int64_t> {
    pybind11::gil_scoped_release no_gil;
    return at::fbgemm_linear_quantize_weight(input);
  };
  return wrap(dispatch_fbgemm_linear_quantize_weight(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _wrapped_linear_prepack
static PyObject * THPVariable__wrapped_linear_prepack(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_wrapped_linear_prepack(Tensor weight, Tensor weight_scale, Tensor weight_zero_point, Tensor bias)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_wrapped_linear_prepack(Tensor weight, Tensor weight_scale, Tensor weight_zero_point, Tensor bias) -> Tensor
  
  auto dispatch__wrapped_linear_prepack = [](const at::Tensor & weight, const at::Tensor & weight_scale, const at::Tensor & weight_zero_point, const at::Tensor & bias) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_wrapped_linear_prepack(weight, weight_scale, weight_zero_point, bias);
  };
  return wrap(dispatch__wrapped_linear_prepack(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// fbgemm_pack_quantized_matrix
static PyObject * THPVariable_fbgemm_pack_quantized_matrix(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fbgemm_pack_quantized_matrix(Tensor input)",
    "fbgemm_pack_quantized_matrix(Tensor input, int64_t K, int64_t N)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::fbgemm_pack_quantized_matrix(Tensor input) -> Tensor
      
      auto dispatch_fbgemm_pack_quantized_matrix = [](const at::Tensor & input) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::fbgemm_pack_quantized_matrix(input);
      };
      return wrap(dispatch_fbgemm_pack_quantized_matrix(_r.tensor(0)));
    }
    case 1: {
      // aten::fbgemm_pack_quantized_matrix.KN(Tensor input, int K, int N) -> Tensor
      
      auto dispatch_fbgemm_pack_quantized_matrix = [](const at::Tensor & input, int64_t K, int64_t N) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::fbgemm_pack_quantized_matrix(input, K, N);
      };
      return wrap(dispatch_fbgemm_pack_quantized_matrix(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// log2
static PyObject * THPVariable_log2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log2(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::log2(Tensor self) -> Tensor
    
    auto dispatch_log2 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.log2();
    };
    return wrap(dispatch_log2(_r.tensor(0)));
  } else {
    // aten::log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_log2_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::log2_out(out, self);
    };
    return wrap(dispatch_log2_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// log2_
static PyObject * THPVariable_log2_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log2_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::log2_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_log2_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log2_();
  };
  return wrap(dispatch_log2_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// logspace
static PyObject * THPVariable_logspace(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logspace(Tensor start, Tensor end, int64_t steps, double base=10.0, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "logspace(Scalar start, Tensor end, int64_t steps, double base=10.0, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "logspace(Tensor start, Scalar end, int64_t steps, double base=10.0, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "logspace(Scalar start, Scalar end, int64_t steps, double base=10.0, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<10> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // aten::logspace.Tensor_Tensor(Tensor start, Tensor end, int steps, float base=10.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(5))
            .device(_r.deviceWithDefault(7, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(6))
            .requires_grad(_r.toBool(9))
            .pinned_memory(_r.toBool(8));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_logspace = [](const at::Tensor & start, const at::Tensor & end, int64_t steps, double base, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::logspace(start, end, steps, base, options);
        };
        return wrap(dispatch_logspace(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toDouble(3), options));
      } else {
        // aten::logspace.Tensor_Tensor_out(Tensor start, Tensor end, int steps, float base=10.0, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(4), _r.scalartypeOptional(5),
                               _r.isNone(5), _r.layoutOptional(6),
                               _r.deviceWithDefault(7, torch::tensors::get_default_device()), _r.isNone(7));
        
        auto dispatch_logspace_out = [](at::Tensor out, const at::Tensor & start, const at::Tensor & end, int64_t steps, double base) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::logspace_out(out, start, end, steps, base);
        };
        return wrap(dispatch_logspace_out(_r.tensor(4), _r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toDouble(3)).set_requires_grad(_r.toBool(9)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::logspace.Scalar_Tensor(Scalar start, Tensor end, int steps, float base=10.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(5))
            .device(_r.deviceWithDefault(7, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(6))
            .requires_grad(_r.toBool(9))
            .pinned_memory(_r.toBool(8));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_logspace = [](const at::Scalar & start, const at::Tensor & end, int64_t steps, double base, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::logspace(start, end, steps, base, options);
        };
        return wrap(dispatch_logspace(_r.scalar(0), _r.tensor(1), _r.toInt64(2), _r.toDouble(3), options));
      } else {
        // aten::logspace.Scalar_Tensor_out(Scalar start, Tensor end, int steps, float base=10.0, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(4), _r.scalartypeOptional(5),
                               _r.isNone(5), _r.layoutOptional(6),
                               _r.deviceWithDefault(7, torch::tensors::get_default_device()), _r.isNone(7));
        
        auto dispatch_logspace_out = [](at::Tensor out, const at::Scalar & start, const at::Tensor & end, int64_t steps, double base) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::logspace_out(out, start, end, steps, base);
        };
        return wrap(dispatch_logspace_out(_r.tensor(4), _r.scalar(0), _r.tensor(1), _r.toInt64(2), _r.toDouble(3)).set_requires_grad(_r.toBool(9)));
      }
    }
    case 2: {
      if (_r.isNone(4)) {
        // aten::logspace.Tensor_Scalar(Tensor start, Scalar end, int steps, float base=10.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(5))
            .device(_r.deviceWithDefault(7, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(6))
            .requires_grad(_r.toBool(9))
            .pinned_memory(_r.toBool(8));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_logspace = [](const at::Tensor & start, const at::Scalar & end, int64_t steps, double base, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::logspace(start, end, steps, base, options);
        };
        return wrap(dispatch_logspace(_r.tensor(0), _r.scalar(1), _r.toInt64(2), _r.toDouble(3), options));
      } else {
        // aten::logspace.Tensor_Scalar_out(Tensor start, Scalar end, int steps, float base=10.0, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(4), _r.scalartypeOptional(5),
                               _r.isNone(5), _r.layoutOptional(6),
                               _r.deviceWithDefault(7, torch::tensors::get_default_device()), _r.isNone(7));
        
        auto dispatch_logspace_out = [](at::Tensor out, const at::Tensor & start, const at::Scalar & end, int64_t steps, double base) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::logspace_out(out, start, end, steps, base);
        };
        return wrap(dispatch_logspace_out(_r.tensor(4), _r.tensor(0), _r.scalar(1), _r.toInt64(2), _r.toDouble(3)).set_requires_grad(_r.toBool(9)));
      }
    }
    case 3: {
      if (_r.isNone(4)) {
        // aten::logspace(Scalar start, Scalar end, int steps, float base=10.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(5))
            .device(_r.deviceWithDefault(7, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(6))
            .requires_grad(_r.toBool(9))
            .pinned_memory(_r.toBool(8));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_logspace = [](const at::Scalar & start, const at::Scalar & end, int64_t steps, double base, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::logspace(start, end, steps, base, options);
        };
        return wrap(dispatch_logspace(_r.scalar(0), _r.scalar(1), _r.toInt64(2), _r.toDouble(3), options));
      } else {
        // aten::logspace.out(Scalar start, Scalar end, int steps, float base=10.0, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(4), _r.scalartypeOptional(5),
                               _r.isNone(5), _r.layoutOptional(6),
                               _r.deviceWithDefault(7, torch::tensors::get_default_device()), _r.isNone(7));
        
        auto dispatch_logspace_out = [](at::Tensor out, const at::Scalar & start, const at::Scalar & end, int64_t steps, double base) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::logspace_out(out, start, end, steps, base);
        };
        return wrap(dispatch_logspace_out(_r.tensor(4), _r.scalar(0), _r.scalar(1), _r.toInt64(2), _r.toDouble(3)).set_requires_grad(_r.toBool(9)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// log_softmax
static PyObject * THPVariable_log_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log_softmax(Tensor input, int64_t dim, ScalarType? dtype=None, *, Tensor out=None)",
    "log_softmax(Tensor input, Dimname dim, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_log_softmax = [](const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.log_softmax(dim, dtype);
        };
        return wrap(dispatch_log_softmax(_r.tensor(0), _r.toInt64(1), _r.scalartypeOptional(2)));
      } else {
        // aten::log_softmax.int_out(Tensor self, int dim, ScalarType? dtype=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_log_softmax_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::log_softmax_out(out, self, dim, dtype);
        };
        return wrap(dispatch_log_softmax_out(_r.tensor(3), _r.tensor(0), _r.toInt64(1), _r.scalartypeOptional(2)));
      }
    }
    case 1: {
      // aten::log_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_log_softmax = [](const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.log_softmax(dim, dtype);
      };
      return wrap(dispatch_log_softmax(_r.tensor(0), _r.dimname(1), _r.scalartypeOptional(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _log_softmax
static PyObject * THPVariable__log_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_log_softmax(Tensor input, int64_t dim, bool half_to_float, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
    
    auto dispatch__log_softmax = [](const at::Tensor & self, int64_t dim, bool half_to_float) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_log_softmax(self, dim, half_to_float);
    };
    return wrap(dispatch__log_softmax(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
  } else {
    // aten::_log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__log_softmax_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, bool half_to_float) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_log_softmax_out(out, self, dim, half_to_float);
    };
    return wrap(dispatch__log_softmax_out(_r.tensor(3), _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _logcumsumexp
static PyObject * THPVariable__logcumsumexp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_logcumsumexp(Tensor input, int64_t dim, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::_logcumsumexp(Tensor self, int dim) -> Tensor
    
    auto dispatch__logcumsumexp = [](const at::Tensor & self, int64_t dim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_logcumsumexp(self, dim);
    };
    return wrap(dispatch__logcumsumexp(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::_logcumsumexp.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__logcumsumexp_out = [](at::Tensor out, const at::Tensor & self, int64_t dim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_logcumsumexp_out(out, self, dim);
    };
    return wrap(dispatch__logcumsumexp_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// matmul
static PyObject * THPVariable_matmul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "matmul(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::matmul(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_matmul = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.matmul(other);
    };
    return wrap(dispatch_matmul(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_matmul_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::matmul_out(out, self, other);
    };
    return wrap(dispatch_matmul_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// aminmax
static PyObject * THPVariable_aminmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_aminmax_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_aminmax_out_structseq();
  static PythonArgParser parser({
    "aminmax(Tensor input, *, int64_t? dim=None, bool keepdim=False, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::aminmax(Tensor self, *, int? dim=None, bool keepdim=False) -> (Tensor min, Tensor max)
    
    auto dispatch_aminmax = [](const at::Tensor & self, ::std::optional<int64_t> dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return self.aminmax(dim, keepdim);
    };
    return wrap(NamedTuple, dispatch_aminmax(_r.tensor(0), _r.toInt64Optional(1), _r.toBool(2)));
  } else {
    // aten::aminmax.out(Tensor self, *, int? dim=None, bool keepdim=False, Tensor(a!) min, Tensor(b!) max) -> (Tensor(a!) min, Tensor(b!) max)
    auto out = _r.tensorlist_n<2>(3);
    auto dispatch_aminmax_out = [](at::Tensor & min, at::Tensor & max, const at::Tensor & self, ::std::optional<int64_t> dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::aminmax_out(min, max, self, dim, keepdim);
    };
    return wrap(NamedTuple1, dispatch_aminmax_out(out[0], out[1], _r.tensor(0), _r.toInt64Optional(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _compute_linear_combination
static PyObject * THPVariable__compute_linear_combination(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_compute_linear_combination(Tensor input, Tensor coefficients, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::_compute_linear_combination(Tensor input, Tensor coefficients) -> Tensor
    
    auto dispatch__compute_linear_combination = [](const at::Tensor & input, const at::Tensor & coefficients) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_compute_linear_combination(input, coefficients);
    };
    return wrap(dispatch__compute_linear_combination(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::_compute_linear_combination.out(Tensor input, Tensor coefficients, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__compute_linear_combination_out = [](at::Tensor out, const at::Tensor & input, const at::Tensor & coefficients) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_compute_linear_combination_out(out, input, coefficients);
    };
    return wrap(dispatch__compute_linear_combination_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// max_pool1d_with_indices
static PyObject * THPVariable_max_pool1d_with_indices(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_pool1d_with_indices(Tensor input, IntArrayRef[1] kernel_size, IntArrayRef[1] stride=None, IntArrayRef[1] padding=0, IntArrayRef[1] dilation=1, bool ceil_mode=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::max_pool1d_with_indices(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
  
  auto dispatch_max_pool1d_with_indices = [](const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::max_pool1d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
  };
  return wrap(dispatch_max_pool1d_with_indices(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// max_pool2d
static PyObject * THPVariable_max_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_pool2d(Tensor input, IntArrayRef[2] kernel_size, IntArrayRef[2] stride=None, IntArrayRef[2] padding=0, IntArrayRef[2] dilation=1, bool ceil_mode=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
  
  auto dispatch_max_pool2d = [](const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
  };
  return wrap(dispatch_max_pool2d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mkldnn_max_pool2d
static PyObject * THPVariable_mkldnn_max_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mkldnn_max_pool2d(Tensor input, IntArrayRef[2] kernel_size, IntArrayRef[2] stride=None, IntArrayRef[2] padding=0, IntArrayRef[2] dilation=1, bool ceil_mode=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::mkldnn_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
  
  auto dispatch_mkldnn_max_pool2d = [](const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::mkldnn_max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
  };
  return wrap(dispatch_mkldnn_max_pool2d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mkldnn_rnn_layer
static PyObject * THPVariable_mkldnn_rnn_layer(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mkldnn_rnn_layer(Tensor input, Tensor weight0, Tensor weight1, Tensor weight2, Tensor weight3, Tensor hx_, Tensor cx_, bool reverse, IntArrayRef batch_sizes, int64_t mode, int64_t hidden_size, int64_t num_layers, bool has_biases, bool bidirectional, bool batch_first, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<16> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::mkldnn_rnn_layer(Tensor input, Tensor weight0, Tensor weight1, Tensor weight2, Tensor weight3, Tensor hx_, Tensor cx_, bool reverse, int[] batch_sizes, int mode, int hidden_size, int num_layers, bool has_biases, bool bidirectional, bool batch_first, bool train) -> (Tensor, Tensor, Tensor, Tensor)
  
  auto dispatch_mkldnn_rnn_layer = [](const at::Tensor & input, const at::Tensor & weight0, const at::Tensor & weight1, const at::Tensor & weight2, const at::Tensor & weight3, const at::Tensor & hx_, const at::Tensor & cx_, bool reverse, at::IntArrayRef batch_sizes, int64_t mode, int64_t hidden_size, int64_t num_layers, bool has_biases, bool bidirectional, bool batch_first, bool train) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::mkldnn_rnn_layer(input, weight0, weight1, weight2, weight3, hx_, cx_, reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train);
  };
  return wrap(dispatch_mkldnn_rnn_layer(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.tensor(5), _r.tensor(6), _r.toBool(7), _r.intlist(8), _r.toInt64(9), _r.toInt64(10), _r.toInt64(11), _r.toBool(12), _r.toBool(13), _r.toBool(14), _r.toBool(15)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// miopen_convolution_relu
static PyObject * THPVariable_miopen_convolution_relu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "miopen_convolution_relu(Tensor input, Tensor weight, Tensor? bias, SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef dilation, SymInt groups)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::miopen_convolution_relu(Tensor self, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, SymInt groups) -> Tensor
  
  auto dispatch_miopen_convolution_relu = [](const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, c10::SymInt groups) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::miopen_convolution_relu_symint(self, weight, bias, stride, padding, dilation, groups);
  };
  return wrap(dispatch_miopen_convolution_relu(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.symintlist(4), _r.symintlist(5), _r.toSymInt(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// miopen_rnn
static PyObject * THPVariable_miopen_rnn(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "miopen_rnn(Tensor input, TensorList weight, int64_t weight_stride0, Tensor hx, Tensor? cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, Tensor? dropout_state)",
  }, /*traceable=*/true);

  ParsedArgs<14> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::miopen_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor hx, Tensor? cx, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
  
  auto dispatch_miopen_rnn = [](const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & hx, const ::std::optional<at::Tensor> & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const ::std::optional<at::Tensor> & dropout_state) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::miopen_rnn(input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
  };
  return wrap(dispatch_miopen_rnn(_r.tensor(0), _r.tensorlist(1), _r.toInt64(2), _r.tensor(3), _r.optionalTensor(4), _r.toInt64(5), _r.toInt64(6), _r.toInt64(7), _r.toBool(8), _r.toDouble(9), _r.toBool(10), _r.toBool(11), _r.intlist(12), _r.optionalTensor(13)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _convert_weight_to_int4pack
static PyObject * THPVariable__convert_weight_to_int4pack(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_convert_weight_to_int4pack(Tensor input, int64_t innerKTiles)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_convert_weight_to_int4pack(Tensor self, int innerKTiles) -> Tensor
  
  auto dispatch__convert_weight_to_int4pack = [](const at::Tensor & self, int64_t innerKTiles) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_convert_weight_to_int4pack(self, innerKTiles);
  };
  return wrap(dispatch__convert_weight_to_int4pack(_r.tensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _weight_int4pack_mm
static PyObject * THPVariable__weight_int4pack_mm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_weight_int4pack_mm(Tensor input, Tensor mat2, int64_t qGroupSize, Tensor qScaleAndZeros)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_weight_int4pack_mm(Tensor self, Tensor mat2, int qGroupSize, Tensor qScaleAndZeros) -> Tensor
  
  auto dispatch__weight_int4pack_mm = [](const at::Tensor & self, const at::Tensor & mat2, int64_t qGroupSize, const at::Tensor & qScaleAndZeros) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_weight_int4pack_mm(self, mat2, qGroupSize, qScaleAndZeros);
  };
  return wrap(dispatch__weight_int4pack_mm(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.tensor(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _weight_int4pack_mm_for_cpu
static PyObject * THPVariable__weight_int4pack_mm_for_cpu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_weight_int4pack_mm_for_cpu(Tensor input, Tensor mat2, int64_t qGroupSize, Tensor qScaleAndZeros)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_weight_int4pack_mm_for_cpu(Tensor self, Tensor mat2, int qGroupSize, Tensor qScaleAndZeros) -> Tensor
  
  auto dispatch__weight_int4pack_mm_for_cpu = [](const at::Tensor & self, const at::Tensor & mat2, int64_t qGroupSize, const at::Tensor & qScaleAndZeros) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_weight_int4pack_mm_for_cpu(self, mat2, qGroupSize, qScaleAndZeros);
  };
  return wrap(dispatch__weight_int4pack_mm_for_cpu(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.tensor(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _dyn_quant_pack_4bit_weight
static PyObject * THPVariable__dyn_quant_pack_4bit_weight(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_dyn_quant_pack_4bit_weight(Tensor weights, Tensor scales_zeros, Tensor? bias, int64_t block_size, int64_t in_features, int64_t out_features)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_dyn_quant_pack_4bit_weight(Tensor weights, Tensor scales_zeros, Tensor? bias, int block_size, int in_features, int out_features) -> Tensor
  
  auto dispatch__dyn_quant_pack_4bit_weight = [](const at::Tensor & weights, const at::Tensor & scales_zeros, const ::std::optional<at::Tensor> & bias, int64_t block_size, int64_t in_features, int64_t out_features) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_dyn_quant_pack_4bit_weight(weights, scales_zeros, bias, block_size, in_features, out_features);
  };
  return wrap(dispatch__dyn_quant_pack_4bit_weight(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.toInt64(3), _r.toInt64(4), _r.toInt64(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _dyn_quant_matmul_4bit
static PyObject * THPVariable__dyn_quant_matmul_4bit(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_dyn_quant_matmul_4bit(Tensor inp, Tensor packed_weights, int64_t block_size, int64_t in_features, int64_t out_features)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_dyn_quant_matmul_4bit(Tensor inp, Tensor packed_weights, int block_size, int in_features, int out_features) -> Tensor
  
  auto dispatch__dyn_quant_matmul_4bit = [](const at::Tensor & inp, const at::Tensor & packed_weights, int64_t block_size, int64_t in_features, int64_t out_features) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_dyn_quant_matmul_4bit(inp, packed_weights, block_size, in_features, out_features);
  };
  return wrap(dispatch__dyn_quant_matmul_4bit(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toInt64(3), _r.toInt64(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sparse_sparse_matmul
static PyObject * THPVariable__sparse_sparse_matmul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_sparse_matmul(Tensor input, Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_sparse_sparse_matmul(Tensor self, Tensor other) -> Tensor
  
  auto dispatch__sparse_sparse_matmul = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sparse_sparse_matmul(self, other);
  };
  return wrap(dispatch__sparse_sparse_matmul(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// mode
static PyObject * THPVariable_mode(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_mode_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_mode_out_structseq();
  static PythonArgParser parser({
    "mode(Tensor input, int64_t dim=-1, bool keepdim=False, *, TensorList[2] out=None)",
    "mode(Tensor input, Dimname dim, bool keepdim=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
        
        auto dispatch_mode = [](const at::Tensor & self, int64_t dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.mode(dim, keepdim);
        };
        return wrap(NamedTuple, dispatch_mode(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      } else {
        // aten::mode.values(Tensor self, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_mode_out = [](at::Tensor & values, at::Tensor & indices, const at::Tensor & self, int64_t dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::mode_out(values, indices, self, dim, keepdim);
        };
        return wrap(NamedTuple1, dispatch_mode_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::mode.dimname(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        
        auto dispatch_mode = [](const at::Tensor & self, at::Dimname dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.mode(dim, keepdim);
        };
        return wrap(NamedTuple, dispatch_mode(_r.tensor(0), _r.dimname(1), _r.toBool(2)));
      } else {
        // aten::mode.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_mode_out = [](at::Tensor & values, at::Tensor & indices, const at::Tensor & self, at::Dimname dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::mode_out(values, indices, self, dim, keepdim);
        };
        return wrap(NamedTuple1, dispatch_mode_out(out[0], out[1], _r.tensor(0), _r.dimname(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mul
static PyObject * THPVariable_mul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mul(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::mul.Tensor(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_mul = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.mul(other);
    };
    return wrap(dispatch_mul(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_mul_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::mul_out(out, self, other);
    };
    return wrap(dispatch_mul_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// native_batch_norm
static PyObject * THPVariable_native_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, double momentum, double eps, *, TensorList[3] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(8)) {
    // aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
    
    auto dispatch_native_batch_norm = [](const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double momentum, double eps) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
    };
    return wrap(dispatch_native_batch_norm(_r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2), _r.optionalTensor(3), _r.optionalTensor(4), _r.toBool(5), _r.toDouble(6), _r.toDouble(7)));
  } else {
    // aten::native_batch_norm.out(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, *, Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd) -> (Tensor(a!), Tensor(b!), Tensor(c!))
    auto out = _r.tensorlist_n<3>(8);
    auto dispatch_native_batch_norm_out = [](at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double momentum, double eps) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::native_batch_norm_out(out, save_mean, save_invstd, input, weight, bias, running_mean, running_var, training, momentum, eps);
    };
    return wrap(dispatch_native_batch_norm_out(out[0], out[1], out[2], _r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2), _r.optionalTensor(3), _r.optionalTensor(4), _r.toBool(5), _r.toDouble(6), _r.toDouble(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _native_batch_norm_legit
static PyObject * THPVariable__native_batch_norm_legit(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, bool training, double momentum, double eps, *, TensorList[3] out=None)",
    "_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, bool training, double momentum, double eps, *, TensorList[3] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(8)) {
        // aten::_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
        
        auto dispatch__native_batch_norm_legit = [](const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, at::Tensor running_mean, at::Tensor running_var, bool training, double momentum, double eps) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::_native_batch_norm_legit(input, weight, bias, running_mean, running_var, training, momentum, eps);
        };
        return wrap(dispatch__native_batch_norm_legit(_r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2), _r.tensor(3), _r.tensor(4), _r.toBool(5), _r.toDouble(6), _r.toDouble(7)));
      } else {
        // aten::_native_batch_norm_legit.out(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps, *, Tensor(d!) out, Tensor(e!) save_mean, Tensor(f!) save_invstd) -> (Tensor(d!), Tensor(e!), Tensor(f!))
        auto out = _r.tensorlist_n<3>(8);
        auto dispatch__native_batch_norm_legit_out = [](at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, at::Tensor running_mean, at::Tensor running_var, bool training, double momentum, double eps) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::_native_batch_norm_legit_out(out, save_mean, save_invstd, input, weight, bias, running_mean, running_var, training, momentum, eps);
        };
        return wrap(dispatch__native_batch_norm_legit_out(out[0], out[1], out[2], _r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2), _r.tensor(3), _r.tensor(4), _r.toBool(5), _r.toDouble(6), _r.toDouble(7)));
      }
    }
    case 1: {
      if (_r.isNone(6)) {
        // aten::_native_batch_norm_legit.no_stats(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
        
        auto dispatch__native_batch_norm_legit = [](const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, bool training, double momentum, double eps) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::_native_batch_norm_legit(input, weight, bias, training, momentum, eps);
        };
        return wrap(dispatch__native_batch_norm_legit(_r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2), _r.toBool(3), _r.toDouble(4), _r.toDouble(5)));
      } else {
        // aten::_native_batch_norm_legit.no_stats_out(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps, *, Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd) -> (Tensor(a!), Tensor(b!), Tensor(c!))
        auto out = _r.tensorlist_n<3>(6);
        auto dispatch__native_batch_norm_legit_out = [](at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, bool training, double momentum, double eps) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::_native_batch_norm_legit_out(out, save_mean, save_invstd, input, weight, bias, training, momentum, eps);
        };
        return wrap(dispatch__native_batch_norm_legit_out(out[0], out[1], out[2], _r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2), _r.toBool(3), _r.toDouble(4), _r.toDouble(5)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// batch_norm_gather_stats
static PyObject * THPVariable_batch_norm_gather_stats(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "batch_norm_gather_stats(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, double momentum, double eps, int64_t count)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::batch_norm_gather_stats(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, int count) -> (Tensor, Tensor)
  
  auto dispatch_batch_norm_gather_stats = [](const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, double momentum, double eps, int64_t count) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::batch_norm_gather_stats(input, mean, invstd, running_mean, running_var, momentum, eps, count);
  };
  return wrap(dispatch_batch_norm_gather_stats(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.optionalTensor(3), _r.optionalTensor(4), _r.toDouble(5), _r.toDouble(6), _r.toInt64(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// batch_norm_backward_reduce
static PyObject * THPVariable_batch_norm_backward_reduce(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "batch_norm_backward_reduce(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, bool input_g, bool weight_g, bool bias_g)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::batch_norm_backward_reduce(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, bool input_g, bool weight_g, bool bias_g) -> (Tensor, Tensor, Tensor, Tensor)
  
  auto dispatch_batch_norm_backward_reduce = [](const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const ::std::optional<at::Tensor> & weight, bool input_g, bool weight_g, bool bias_g) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::batch_norm_backward_reduce(grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
  };
  return wrap(dispatch_batch_norm_backward_reduce(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.optionalTensor(4), _r.toBool(5), _r.toBool(6), _r.toBool(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_vulkan_available
static PyObject * THPVariable_is_vulkan_available(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  // aten::is_vulkan_available() -> bool
  
  auto dispatch_is_vulkan_available = []() -> bool {
    pybind11::gil_scoped_release no_gil;
    return at::is_vulkan_available();
  };
  return wrap(dispatch_is_vulkan_available());
  END_HANDLE_TH_ERRORS
}

\
// ones
static PyObject * THPVariable_ones(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ones(IntArrayRef size, *, DimnameList? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "ones(SymIntArrayRef size, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::ones.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      auto __names = _r.toDimnameListOptional(1);
      ::std::optional<DimnameList> names = __names ? ::std::make_optional(DimnameList(__names.value())) : ::std::nullopt;
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(2))
          .device(_r.deviceWithDefault(4, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(3))
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_ones = [](at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::ones(size, names, options);
      };
      return wrap(dispatch_ones(_r.intlist(0), names, options));
    }
    case 1: {
      if (_r.isNone(1)) {
        // aten::ones(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(2))
            .device(_r.deviceWithDefault(4, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(3))
            .requires_grad(_r.toBool(6))
            .pinned_memory(_r.toBool(5));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_ones = [](c10::SymIntArrayRef size, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::ones_symint(size, options);
        };
        return wrap(dispatch_ones(_r.symintlist(0), options));
      } else {
        // aten::ones.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(1), _r.scalartypeOptional(2),
                               _r.isNone(2), _r.layoutOptional(3),
                               _r.deviceWithDefault(4, torch::tensors::get_default_device()), _r.isNone(4));
        
        auto dispatch_ones_out = [](at::Tensor out, c10::SymIntArrayRef size) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::ones_symint_out(out, size);
        };
        return wrap(dispatch_ones_out(_r.tensor(1), _r.symintlist(0)).set_requires_grad(_r.toBool(6)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// pdist
static PyObject * THPVariable_pdist(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pdist(Tensor input, double p=2)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::pdist(Tensor self, float p=2) -> Tensor
  
  auto dispatch_pdist = [](const at::Tensor & self, double p) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::pdist(self, p);
  };
  return wrap(dispatch_pdist(_r.tensor(0), _r.toDouble(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// moveaxis
static PyObject * THPVariable_moveaxis(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "moveaxis(Tensor input, int64_t source, int64_t destination)",
    "moveaxis(Tensor input, IntArrayRef source, IntArrayRef destination)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::moveaxis.int(Tensor(a) self, int source, int destination) -> Tensor(a)
      
      auto dispatch_moveaxis = [](const at::Tensor & self, int64_t source, int64_t destination) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.moveaxis(source, destination);
      };
      return wrap(dispatch_moveaxis(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
    }
    case 1: {
      // aten::moveaxis.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)
      
      auto dispatch_moveaxis = [](const at::Tensor & self, at::IntArrayRef source, at::IntArrayRef destination) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.moveaxis(source, destination);
      };
      return wrap(dispatch_moveaxis(_r.tensor(0), _r.intlist(1), _r.intlist(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// pixel_unshuffle
static PyObject * THPVariable_pixel_unshuffle(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pixel_unshuffle(Tensor input, int64_t downscale_factor)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::pixel_unshuffle(Tensor self, int downscale_factor) -> Tensor
  
  auto dispatch_pixel_unshuffle = [](const at::Tensor & self, int64_t downscale_factor) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::pixel_unshuffle(self, downscale_factor);
  };
  return wrap(dispatch_pixel_unshuffle(_r.tensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// scalar_tensor
static PyObject * THPVariable_scalar_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(1))
      .device(_r.deviceWithDefault(3, torch::tensors::get_default_device()))
      .layout(_r.layoutOptional(2))
      .requires_grad(_r.toBool(5))
      .pinned_memory(_r.toBool(4));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch_scalar_tensor = [](const at::Scalar & s, at::TensorOptions options) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::scalar_tensor(s, options);
  };
  return wrap(dispatch_scalar_tensor(_r.scalar(0), options));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// rand
static PyObject * THPVariable_rand(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rand(SymIntArrayRef size, *, Generator? generator, DimnameList? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "rand(SymIntArrayRef size, *, Generator? generator, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "rand(SymIntArrayRef size, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "rand(SymIntArrayRef size, *, DimnameList? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::rand.generator_with_names(SymInt[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      auto __names = _r.toDimnameListOptional(2);
      ::std::optional<DimnameList> names = __names ? ::std::make_optional(DimnameList(__names.value())) : ::std::nullopt;
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(3))
          .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(4))
          .requires_grad(_r.toBool(7))
          .pinned_memory(_r.toBool(6));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_rand = [](c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::rand_symint(size, generator, names, options);
      };
      return wrap(dispatch_rand(_r.symintlist(0), _r.generator(1), names, options));
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::rand.generator(SymInt[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(3))
            .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(4))
            .requires_grad(_r.toBool(7))
            .pinned_memory(_r.toBool(6));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_rand = [](c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::rand_symint(size, generator, options);
        };
        return wrap(dispatch_rand(_r.symintlist(0), _r.generator(1), options));
      } else {
        // aten::rand.generator_out(SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(2), _r.scalartypeOptional(3),
                               _r.isNone(3), _r.layoutOptional(4),
                               _r.deviceWithDefault(5, torch::tensors::get_default_device()), _r.isNone(5));
        
        auto dispatch_rand_out = [](at::Tensor out, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::rand_symint_out(out, size, generator);
        };
        return wrap(dispatch_rand_out(_r.tensor(2), _r.symintlist(0), _r.generator(1)).set_requires_grad(_r.toBool(7)));
      }
    }
    case 2: {
      if (_r.isNone(1)) {
        // aten::rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(2))
            .device(_r.deviceWithDefault(4, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(3))
            .requires_grad(_r.toBool(6))
            .pinned_memory(_r.toBool(5));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_rand = [](c10::SymIntArrayRef size, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::rand_symint(size, options);
        };
        return wrap(dispatch_rand(_r.symintlist(0), options));
      } else {
        // aten::rand.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(1), _r.scalartypeOptional(2),
                               _r.isNone(2), _r.layoutOptional(3),
                               _r.deviceWithDefault(4, torch::tensors::get_default_device()), _r.isNone(4));
        
        auto dispatch_rand_out = [](at::Tensor out, c10::SymIntArrayRef size) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::rand_symint_out(out, size);
        };
        return wrap(dispatch_rand_out(_r.tensor(1), _r.symintlist(0)).set_requires_grad(_r.toBool(6)));
      }
    }
    case 3: {
      // aten::rand.names(SymInt[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      auto __names = _r.toDimnameListOptional(1);
      ::std::optional<DimnameList> names = __names ? ::std::make_optional(DimnameList(__names.value())) : ::std::nullopt;
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(2))
          .device(_r.deviceWithDefault(4, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(3))
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_rand = [](c10::SymIntArrayRef size, ::std::optional<at::DimnameList> names, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::rand_symint(size, names, options);
      };
      return wrap(dispatch_rand(_r.symintlist(0), names, options));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ravel
static PyObject * THPVariable_ravel(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ravel(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::ravel(Tensor(a) self) -> Tensor(a)
  
  auto dispatch_ravel = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.ravel();
  };
  return wrap(dispatch_ravel(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// neg
static PyObject * THPVariable_neg(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "neg(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::neg(Tensor self) -> Tensor
    
    auto dispatch_neg = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.neg();
    };
    return wrap(dispatch_neg(_r.tensor(0)));
  } else {
    // aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_neg_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::neg_out(out, self);
    };
    return wrap(dispatch_neg_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// neg_
static PyObject * THPVariable_neg_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "neg_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::neg_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_neg_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.neg_();
  };
  return wrap(dispatch_neg_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// round
static PyObject * THPVariable_round(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "round(Tensor input, *, Tensor out=None)",
    "round(Tensor input, *, int64_t decimals, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(1)) {
        // aten::round(Tensor self) -> Tensor
        
        auto dispatch_round = [](const at::Tensor & self) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.round();
        };
        return wrap(dispatch_round(_r.tensor(0)));
      } else {
        // aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_round_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::round_out(out, self);
        };
        return wrap(dispatch_round_out(_r.tensor(1), _r.tensor(0)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::round.decimals(Tensor self, *, int decimals) -> Tensor
        
        auto dispatch_round = [](const at::Tensor & self, int64_t decimals) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.round(decimals);
        };
        return wrap(dispatch_round(_r.tensor(0), _r.toInt64(1)));
      } else {
        // aten::round.decimals_out(Tensor self, *, int decimals, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_round_out = [](at::Tensor out, const at::Tensor & self, int64_t decimals) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::round_out(out, self, decimals);
        };
        return wrap(dispatch_round_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
      }
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
  static PythonArgParser parser({
    "round_(Tensor input)",
    "round_(Tensor input, *, int64_t decimals)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::round_(Tensor(a!) self) -> Tensor(a!)
      
      auto dispatch_round_ = [](at::Tensor self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.round_();
      };
      return wrap(dispatch_round_(_r.tensor(0)));
    }
    case 1: {
      // aten::round_.decimals(Tensor(a!) self, *, int decimals) -> Tensor(a!)
      
      auto dispatch_round_ = [](at::Tensor self, int64_t decimals) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.round_(decimals);
      };
      return wrap(dispatch_round_(_r.tensor(0), _r.toInt64(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rrelu
static PyObject * THPVariable_rrelu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rrelu(Tensor input, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::rrelu(Tensor self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor
  
  auto dispatch_rrelu = [](const at::Tensor & self, const at::Scalar & lower, const at::Scalar & upper, bool training, ::std::optional<at::Generator> generator) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::rrelu(self, lower, upper, training, generator);
  };
  return wrap(dispatch_rrelu(_r.tensor(0), _r.scalar(1), _r.scalar(2), _r.toBool(3), _r.generator(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rrelu_
static PyObject * THPVariable_rrelu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rrelu_(Tensor input, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::rrelu_(Tensor(a!) self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)
  
  auto dispatch_rrelu_ = [](at::Tensor self, const at::Scalar & lower, const at::Scalar & upper, bool training, ::std::optional<at::Generator> generator) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::rrelu_(self, lower, upper, training, generator);
  };
  return wrap(dispatch_rrelu_(_r.tensor(0), _r.scalar(1), _r.scalar(2), _r.toBool(3), _r.generator(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sigmoid
static PyObject * THPVariable_sigmoid(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sigmoid(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::sigmoid(Tensor self) -> Tensor
    
    auto dispatch_sigmoid = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.sigmoid();
    };
    return wrap(dispatch_sigmoid(_r.tensor(0)));
  } else {
    // aten::sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_sigmoid_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::sigmoid_out(out, self);
    };
    return wrap(dispatch_sigmoid_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sigmoid_
static PyObject * THPVariable_sigmoid_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sigmoid_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_sigmoid_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sigmoid_();
  };
  return wrap(dispatch_sigmoid_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// slice_scatter
static PyObject * THPVariable_slice_scatter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "slice_scatter(Tensor input, Tensor src, int64_t dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(6)) {
    // aten::slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor
    
    auto dispatch_slice_scatter = [](const at::Tensor & self, const at::Tensor & src, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.slice_scatter_symint(src, dim, start, end, step);
    };
    return wrap(dispatch_slice_scatter(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toSymIntOptional(3), _r.toSymIntOptional(4), _r.toSymInt(5)));
  } else {
    // aten::slice_scatter.out(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_slice_scatter_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & src, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::slice_scatter_symint_out(out, self, src, dim, start, end, step);
    };
    return wrap(dispatch_slice_scatter_out(_r.tensor(6), _r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toSymIntOptional(3), _r.toSymIntOptional(4), _r.toSymInt(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// select_scatter
static PyObject * THPVariable_select_scatter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "select_scatter(Tensor input, Tensor src, int64_t dim, SymInt index)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::select_scatter(Tensor self, Tensor src, int dim, SymInt index) -> Tensor
  
  auto dispatch_select_scatter = [](const at::Tensor & self, const at::Tensor & src, int64_t dim, c10::SymInt index) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.select_scatter_symint(src, dim, index);
  };
  return wrap(dispatch_select_scatter(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toSymInt(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// diagonal_scatter
static PyObject * THPVariable_diagonal_scatter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diagonal_scatter(Tensor input, Tensor src, int64_t offset=0, int64_t dim1=0, int64_t dim2=1)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::diagonal_scatter(Tensor self, Tensor src, int offset=0, int dim1=0, int dim2=1) -> Tensor
  
  auto dispatch_diagonal_scatter = [](const at::Tensor & self, const at::Tensor & src, int64_t offset, int64_t dim1, int64_t dim2) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.diagonal_scatter(src, offset, dim1, dim2);
  };
  return wrap(dispatch_diagonal_scatter(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toInt64(3), _r.toInt64(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// as_strided_scatter
static PyObject * THPVariable_as_strided_scatter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "as_strided_scatter(Tensor input, Tensor src, SymIntArrayRef size, SymIntArrayRef stride, SymInt? storage_offset=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::as_strided_scatter(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor
  
  auto dispatch_as_strided_scatter = [](const at::Tensor & self, const at::Tensor & src, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<c10::SymInt> storage_offset) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.as_strided_scatter_symint(src, size, stride, storage_offset);
  };
  return wrap(dispatch_as_strided_scatter(_r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.symintlist(3), _r.toSymIntOptional(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// smm
static PyObject * THPVariable_smm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "smm(Tensor input, Tensor mat2)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::smm(Tensor self, Tensor mat2) -> Tensor
  
  auto dispatch_smm = [](const at::Tensor & self, const at::Tensor & mat2) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.smm(mat2);
  };
  return wrap(dispatch_smm(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// dsplit
static PyObject * THPVariable_dsplit(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dsplit(Tensor input, int64_t sections)",
    "dsplit(Tensor input, IntArrayRef indices)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::dsplit.int(Tensor(a -> *) self, int sections) -> Tensor(a)[]
      
      auto dispatch_dsplit = [](const at::Tensor & self, int64_t sections) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.dsplit(sections);
      };
      return wrap(dispatch_dsplit(_r.tensor(0), _r.toInt64(1)));
    }
    case 1: {
      // aten::dsplit.array(Tensor(a -> *) self, int[] indices) -> Tensor(a)[]
      
      auto dispatch_dsplit = [](const at::Tensor & self, at::IntArrayRef indices) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.dsplit(indices);
      };
      return wrap(dispatch_dsplit(_r.tensor(0), _r.intlist(1)));
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
  static PythonArgParser parser({
    "sspaddmm(Scalar beta, Tensor input, Scalar alpha, Tensor mat1, Tensor mat2)|deprecated",
    "sspaddmm(Scalar beta, Tensor input, Tensor mat1, Tensor mat2)|deprecated",
    "sspaddmm(Tensor input, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::sspaddmm(Scalar beta, Tensor self, Scalar alpha, Tensor mat1, Tensor mat2) -> Tensor
      
      auto dispatch_sspaddmm = [](const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & mat1, const at::Tensor & mat2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sspaddmm(mat1, mat2, beta, alpha);
      };
      return wrap(dispatch_sspaddmm(_r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4)));
    }
    case 1: {
      // [deprecated] aten::sspaddmm(Scalar beta, Tensor self, Tensor mat1, Tensor mat2) -> Tensor
      
      auto dispatch_sspaddmm = [](const at::Scalar & beta, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sspaddmm(mat1, mat2, beta, 1);
      };
      return wrap(dispatch_sspaddmm(_r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
    }
    case 2: {
      if (_r.isNone(5)) {
        // aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        
        auto dispatch_sspaddmm = [](const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.sspaddmm(mat1, mat2, beta, alpha);
        };
        return wrap(dispatch_sspaddmm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      } else {
        // aten::sspaddmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_sspaddmm_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::sspaddmm_out(out, self, mat1, mat2, beta, alpha);
        };
        return wrap(dispatch_sspaddmm_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      }
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
  static PythonArgParser parser({
    "stft(Tensor input, int64_t n_fft, int64_t? hop_length=None, int64_t? win_length=None, Tensor? window=None, bool center=True, c10::string_view pad_mode=\"reflect\", bool normalized=False, bool? onesided=None, bool? return_complex=None, bool? align_to_window=None)",
    "stft(Tensor input, int64_t n_fft, int64_t? hop_length=None, int64_t? win_length=None, Tensor? window=None, bool normalized=False, bool? onesided=None, bool? return_complex=None, bool? align_to_window=None)",
  }, /*traceable=*/true);

  ParsedArgs<11> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::stft.center(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool center=True, str pad_mode="reflect", bool normalized=False, bool? onesided=None, bool? return_complex=None, bool? align_to_window=None) -> Tensor
      
      auto dispatch_stft = [](const at::Tensor & self, int64_t n_fft, ::std::optional<int64_t> hop_length, ::std::optional<int64_t> win_length, const ::std::optional<at::Tensor> & window, bool center, c10::string_view pad_mode, bool normalized, ::std::optional<bool> onesided, ::std::optional<bool> return_complex, ::std::optional<bool> align_to_window) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.stft(n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided, return_complex, align_to_window);
      };
      return wrap(dispatch_stft(_r.tensor(0), _r.toInt64(1), _r.toInt64Optional(2), _r.toInt64Optional(3), _r.optionalTensor(4), _r.toBool(5), _r.stringView(6), _r.toBool(7), _r.toBoolOptional(8), _r.toBoolOptional(9), _r.toBoolOptional(10)));
    }
    case 1: {
      // aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool? onesided=None, bool? return_complex=None, bool? align_to_window=None) -> Tensor
      
      auto dispatch_stft = [](const at::Tensor & self, int64_t n_fft, ::std::optional<int64_t> hop_length, ::std::optional<int64_t> win_length, const ::std::optional<at::Tensor> & window, bool normalized, ::std::optional<bool> onesided, ::std::optional<bool> return_complex, ::std::optional<bool> align_to_window) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.stft(n_fft, hop_length, win_length, window, normalized, onesided, return_complex, align_to_window);
      };
      return wrap(dispatch_stft(_r.tensor(0), _r.toInt64(1), _r.toInt64Optional(2), _r.toInt64Optional(3), _r.optionalTensor(4), _r.toBool(5), _r.toBoolOptional(6), _r.toBoolOptional(7), _r.toBoolOptional(8)));
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
  static PythonArgParser parser({
    "std(Tensor input, IntArrayRef[1]? dim, bool unbiased=True, bool keepdim=False, *, Tensor out=None)",
    "std(Tensor input, IntArrayRef[1]? dim=None, *, Scalar? correction=None, bool keepdim=False, Tensor out=None)",
    "std(Tensor input, bool unbiased=True)",
    "std(Tensor input, DimnameList[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor out=None)",
    "std(Tensor input, DimnameList[1] dim, *, Scalar? correction=None, bool keepdim=False, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // aten::std.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor
        
        auto dispatch_std = [](const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.std(dim, unbiased, keepdim);
        };
        return wrap(dispatch_std(_r.tensor(0), _r.intlistOptional(1), _r.toBool(2), _r.toBool(3)));
      } else {
        // aten::std.out(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_std_out = [](at::Tensor out, const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::std_out(out, self, dim, unbiased, keepdim);
        };
        return wrap(dispatch_std_out(_r.tensor(4), _r.tensor(0), _r.intlistOptional(1), _r.toBool(2), _r.toBool(3)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::std.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
        
        auto dispatch_std = [](const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.std(dim, correction, keepdim);
        };
        return wrap(dispatch_std(_r.tensor(0), _r.intlistOptional(1), _r.scalarOptional(2), _r.toBool(3)));
      } else {
        // aten::std.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_std_out = [](at::Tensor out, const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::std_out(out, self, dim, correction, keepdim);
        };
        return wrap(dispatch_std_out(_r.tensor(4), _r.tensor(0), _r.intlistOptional(1), _r.scalarOptional(2), _r.toBool(3)));
      }
    }
    case 2: {
      // aten::std(Tensor self, bool unbiased=True) -> Tensor
      
      auto dispatch_std = [](const at::Tensor & self, bool unbiased) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.std(unbiased);
      };
      return wrap(dispatch_std(_r.tensor(0), _r.toBool(1)));
    }
    case 3: {
      if (_r.isNone(4)) {
        // aten::std.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
        
        auto dispatch_std = [](const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.std(dim, unbiased, keepdim);
        };
        return wrap(dispatch_std(_r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.toBool(3)));
      } else {
        // aten::std.names_out(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_std_out = [](at::Tensor out, const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::std_out(out, self, dim, unbiased, keepdim);
        };
        return wrap(dispatch_std_out(_r.tensor(4), _r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.toBool(3)));
      }
    }
    case 4: {
      if (_r.isNone(4)) {
        // aten::std.correction_names(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False) -> Tensor
        
        auto dispatch_std = [](const at::Tensor & self, at::DimnameList dim, const ::std::optional<at::Scalar> & correction, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.std(dim, correction, keepdim);
        };
        return wrap(dispatch_std(_r.tensor(0), _r.dimnamelist(1), _r.scalarOptional(2), _r.toBool(3)));
      } else {
        // aten::std.correction_names_out(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_std_out = [](at::Tensor out, const at::Tensor & self, at::DimnameList dim, const ::std::optional<at::Scalar> & correction, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::std_out(out, self, dim, correction, keepdim);
        };
        return wrap(dispatch_std_out(_r.tensor(4), _r.tensor(0), _r.dimnamelist(1), _r.scalarOptional(2), _r.toBool(3)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// std_mean
static PyObject * THPVariable_std_mean(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "std_mean(Tensor input, IntArrayRef[1]? dim, bool unbiased=True, bool keepdim=False)",
    "std_mean(Tensor input, IntArrayRef[1]? dim=None, *, Scalar? correction=None, bool keepdim=False)",
    "std_mean(Tensor input, bool unbiased=True)",
    "std_mean(Tensor input, DimnameList[1] dim, bool unbiased=True, bool keepdim=False)",
    "std_mean(Tensor input, DimnameList[1] dim, *, Scalar? correction=None, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::std_mean.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
      
      auto dispatch_std_mean = [](const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::std_mean(self, dim, unbiased, keepdim);
      };
      return wrap(dispatch_std_mean(_r.tensor(0), _r.intlistOptional(1), _r.toBool(2), _r.toBool(3)));
    }
    case 1: {
      // aten::std_mean.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)
      
      auto dispatch_std_mean = [](const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::std_mean(self, dim, correction, keepdim);
      };
      return wrap(dispatch_std_mean(_r.tensor(0), _r.intlistOptional(1), _r.scalarOptional(2), _r.toBool(3)));
    }
    case 2: {
      // aten::std_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
      
      auto dispatch_std_mean = [](const at::Tensor & self, bool unbiased) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::std_mean(self, unbiased);
      };
      return wrap(dispatch_std_mean(_r.tensor(0), _r.toBool(1)));
    }
    case 3: {
      // aten::std_mean.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
      
      auto dispatch_std_mean = [](const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::std_mean(self, dim, unbiased, keepdim);
      };
      return wrap(dispatch_std_mean(_r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.toBool(3)));
    }
    case 4: {
      // aten::std_mean.correction_names(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)
      
      auto dispatch_std_mean = [](const at::Tensor & self, at::DimnameList dim, const ::std::optional<at::Scalar> & correction, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::std_mean(self, dim, correction, keepdim);
      };
      return wrap(dispatch_std_mean(_r.tensor(0), _r.dimnamelist(1), _r.scalarOptional(2), _r.toBool(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// prod
static PyObject * THPVariable_prod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "prod(Tensor input, *, ScalarType? dtype=None)",
    "prod(Tensor input, int64_t dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
    "prod(Tensor input, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_prod = [](const at::Tensor & self, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.prod(dtype);
      };
      return wrap(dispatch_prod(_r.tensor(0), _r.scalartypeOptional(1)));
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_prod = [](const at::Tensor & self, int64_t dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.prod(dim, keepdim, dtype);
        };
        return wrap(dispatch_prod(_r.tensor(0), _r.toInt64(1), _r.toBool(2), _r.scalartypeOptional(3)));
      } else {
        // aten::prod.int_out(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_prod_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::prod_out(out, self, dim, keepdim, dtype);
        };
        return wrap(dispatch_prod_out(_r.tensor(4), _r.tensor(0), _r.toInt64(1), _r.toBool(2), _r.scalartypeOptional(3)));
      }
    }
    case 2: {
      if (_r.isNone(4)) {
        // aten::prod.dim_Dimname(Tensor self, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_prod = [](const at::Tensor & self, at::Dimname dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.prod(dim, keepdim, dtype);
        };
        return wrap(dispatch_prod(_r.tensor(0), _r.dimname(1), _r.toBool(2), _r.scalartypeOptional(3)));
      } else {
        // aten::prod.Dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_prod_out = [](at::Tensor out, const at::Tensor & self, at::Dimname dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::prod_out(out, self, dim, keepdim, dtype);
        };
        return wrap(dispatch_prod_out(_r.tensor(4), _r.tensor(0), _r.dimname(1), _r.toBool(2), _r.scalartypeOptional(3)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// threshold
static PyObject * THPVariable_threshold(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "threshold(Tensor input, Scalar threshold, Scalar value, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor
    
    auto dispatch_threshold = [](const at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::threshold(self, threshold, value);
    };
    return wrap(dispatch_threshold(_r.tensor(0), _r.scalar(1), _r.scalar(2)));
  } else {
    // aten::threshold.out(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_threshold_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::threshold_out(out, self, threshold, value);
    };
    return wrap(dispatch_threshold_out(_r.tensor(3), _r.tensor(0), _r.scalar(1), _r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// threshold_
static PyObject * THPVariable_threshold_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "threshold_(Tensor input, Scalar threshold, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::threshold_(Tensor(a!) self, Scalar threshold, Scalar value) -> Tensor(a!)
  
  auto dispatch_threshold_ = [](at::Tensor self, const at::Scalar & threshold, const at::Scalar & value) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::threshold_(self, threshold, value);
  };
  return wrap(dispatch_threshold_(_r.tensor(0), _r.scalar(1), _r.scalar(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// transpose
static PyObject * THPVariable_transpose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "transpose(Tensor input, int64_t dim0, int64_t dim1)",
    "transpose(Tensor input, Dimname dim0, Dimname dim1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
      
      auto dispatch_transpose = [](const at::Tensor & self, int64_t dim0, int64_t dim1) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.transpose(dim0, dim1);
      };
      return wrap(dispatch_transpose(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
    }
    case 1: {
      // aten::transpose.Dimname(Tensor(a) self, Dimname dim0, Dimname dim1) -> Tensor(a)
      
      auto dispatch_transpose = [](const at::Tensor & self, at::Dimname dim0, at::Dimname dim1) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.transpose(dim0, dim1);
      };
      return wrap(dispatch_transpose(_r.tensor(0), _r.dimname(1), _r.dimname(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// flip
static PyObject * THPVariable_flip(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "flip(Tensor input, IntArrayRef dims)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::flip(Tensor self, int[] dims) -> Tensor
  
  auto dispatch_flip = [](const at::Tensor & self, at::IntArrayRef dims) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.flip(dims);
  };
  return wrap(dispatch_flip(_r.tensor(0), _r.intlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fliplr
static PyObject * THPVariable_fliplr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fliplr(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::fliplr(Tensor self) -> Tensor
  
  auto dispatch_fliplr = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.fliplr();
  };
  return wrap(dispatch_fliplr(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// flipud
static PyObject * THPVariable_flipud(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "flipud(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::flipud(Tensor self) -> Tensor
  
  auto dispatch_flipud = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.flipud();
  };
  return wrap(dispatch_flipud(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// roll
static PyObject * THPVariable_roll(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "roll(Tensor input, SymIntArrayRef[1] shifts, IntArrayRef[1] dims=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::roll(Tensor self, SymInt[1] shifts, int[1] dims=[]) -> Tensor
  
  auto dispatch_roll = [](const at::Tensor & self, c10::SymIntArrayRef shifts, at::IntArrayRef dims) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.roll_symint(shifts, dims);
  };
  return wrap(dispatch_roll(_r.tensor(0), _r.symintlist(1), _r.intlist(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// trapezoid
static PyObject * THPVariable_trapezoid(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "trapezoid(Tensor y, Tensor x, *, int64_t dim=-1)",
    "trapezoid(Tensor y, *, Scalar dx=1, int64_t dim=-1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::trapezoid.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor
      
      auto dispatch_trapezoid = [](const at::Tensor & y, const at::Tensor & x, int64_t dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::trapezoid(y, x, dim);
      };
      return wrap(dispatch_trapezoid(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
    }
    case 1: {
      // aten::trapezoid.dx(Tensor y, *, Scalar dx=1, int dim=-1) -> Tensor
      
      auto dispatch_trapezoid = [](const at::Tensor & y, const at::Scalar & dx, int64_t dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::trapezoid(y, dx, dim);
      };
      return wrap(dispatch_trapezoid(_r.tensor(0), _r.scalar(1), _r.toInt64(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _transform_bias_rescale_qkv
static PyObject * THPVariable__transform_bias_rescale_qkv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_transform_bias_rescale_qkv(Tensor qkv, Tensor qkv_bias, int64_t num_heads)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_transform_bias_rescale_qkv(Tensor qkv, Tensor qkv_bias, int num_heads) -> (Tensor, Tensor, Tensor)
  
  auto dispatch__transform_bias_rescale_qkv = [](const at::Tensor & qkv, const at::Tensor & qkv_bias, int64_t num_heads) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_transform_bias_rescale_qkv(qkv, qkv_bias, num_heads);
  };
  return wrap(dispatch__transform_bias_rescale_qkv(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_tensor_from_mask
static PyObject * THPVariable__nested_tensor_from_mask(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_tensor_from_mask(Tensor t, Tensor mask, bool mask_check=True)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_nested_tensor_from_mask(Tensor t, Tensor mask, bool mask_check=True) -> Tensor
  
  auto dispatch__nested_tensor_from_mask = [](const at::Tensor & t, const at::Tensor & mask, bool mask_check) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_nested_tensor_from_mask(t, mask, mask_check);
  };
  return wrap(dispatch__nested_tensor_from_mask(_r.tensor(0), _r.tensor(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_view_from_buffer_copy
static PyObject * THPVariable__nested_view_from_buffer_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_view_from_buffer_copy(Tensor input, Tensor nested_size, Tensor nested_strides, Tensor offsets, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::_nested_view_from_buffer_copy(Tensor self, Tensor nested_size, Tensor nested_strides, Tensor offsets) -> Tensor
    
    auto dispatch__nested_view_from_buffer_copy = [](const at::Tensor & self, const at::Tensor & nested_size, const at::Tensor & nested_strides, const at::Tensor & offsets) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_nested_view_from_buffer_copy(self, nested_size, nested_strides, offsets);
    };
    return wrap(dispatch__nested_view_from_buffer_copy(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
  } else {
    // aten::_nested_view_from_buffer_copy.out(Tensor self, Tensor nested_size, Tensor nested_strides, Tensor offsets, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__nested_view_from_buffer_copy_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & nested_size, const at::Tensor & nested_strides, const at::Tensor & offsets) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_nested_view_from_buffer_copy_out(out, self, nested_size, nested_strides, offsets);
    };
    return wrap(dispatch__nested_view_from_buffer_copy_out(_r.tensor(4), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_view_from_jagged
static PyObject * THPVariable__nested_view_from_jagged(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_view_from_jagged(Tensor input, Tensor offsets, Tensor dummy, Tensor? lengths=None, int64_t ragged_idx=1, Tensor? min_seqlen=None, Tensor? max_seqlen=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_nested_view_from_jagged(Tensor(a) self, Tensor offsets, Tensor dummy, Tensor? lengths=None, int ragged_idx=1, Tensor? min_seqlen=None, Tensor? max_seqlen=None) -> Tensor(a)
  
  auto dispatch__nested_view_from_jagged = [](const at::Tensor & self, const at::Tensor & offsets, const at::Tensor & dummy, const ::std::optional<at::Tensor> & lengths, int64_t ragged_idx, const ::std::optional<at::Tensor> & min_seqlen, const ::std::optional<at::Tensor> & max_seqlen) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_nested_view_from_jagged(self, offsets, dummy, lengths, ragged_idx, min_seqlen, max_seqlen);
  };
  return wrap(dispatch__nested_view_from_jagged(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.optionalTensor(3), _r.toInt64(4), _r.optionalTensor(5), _r.optionalTensor(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_view_from_jagged_copy
static PyObject * THPVariable__nested_view_from_jagged_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_view_from_jagged_copy(Tensor input, Tensor offsets, Tensor dummy, Tensor? lengths=None, int64_t ragged_idx=1, Tensor? min_seqlen=None, Tensor? max_seqlen=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(7)) {
    // aten::_nested_view_from_jagged_copy(Tensor self, Tensor offsets, Tensor dummy, Tensor? lengths=None, int ragged_idx=1, Tensor? min_seqlen=None, Tensor? max_seqlen=None) -> Tensor
    
    auto dispatch__nested_view_from_jagged_copy = [](const at::Tensor & self, const at::Tensor & offsets, const at::Tensor & dummy, const ::std::optional<at::Tensor> & lengths, int64_t ragged_idx, const ::std::optional<at::Tensor> & min_seqlen, const ::std::optional<at::Tensor> & max_seqlen) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_nested_view_from_jagged_copy(self, offsets, dummy, lengths, ragged_idx, min_seqlen, max_seqlen);
    };
    return wrap(dispatch__nested_view_from_jagged_copy(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.optionalTensor(3), _r.toInt64(4), _r.optionalTensor(5), _r.optionalTensor(6)));
  } else {
    // aten::_nested_view_from_jagged_copy.out(Tensor self, Tensor offsets, Tensor dummy, Tensor? lengths=None, int ragged_idx=1, Tensor? min_seqlen=None, Tensor? max_seqlen=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__nested_view_from_jagged_copy_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & offsets, const at::Tensor & dummy, const ::std::optional<at::Tensor> & lengths, int64_t ragged_idx, const ::std::optional<at::Tensor> & min_seqlen, const ::std::optional<at::Tensor> & max_seqlen) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_nested_view_from_jagged_copy_out(out, self, offsets, dummy, lengths, ragged_idx, min_seqlen, max_seqlen);
    };
    return wrap(dispatch__nested_view_from_jagged_copy_out(_r.tensor(7), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.optionalTensor(3), _r.toInt64(4), _r.optionalTensor(5), _r.optionalTensor(6)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_get_ragged_idx
static PyObject * THPVariable__nested_get_ragged_idx(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_get_ragged_idx(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_nested_get_ragged_idx(Tensor self) -> int
  
  auto dispatch__nested_get_ragged_idx = [](const at::Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return at::_nested_get_ragged_idx(self);
  };
  return wrap(dispatch__nested_get_ragged_idx(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_get_min_seqlen
static PyObject * THPVariable__nested_get_min_seqlen(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_get_min_seqlen(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_nested_get_min_seqlen(Tensor self) -> Tensor
  
  auto dispatch__nested_get_min_seqlen = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_nested_get_min_seqlen(self);
  };
  return wrap(dispatch__nested_get_min_seqlen(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_get_max_seqlen
static PyObject * THPVariable__nested_get_max_seqlen(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_get_max_seqlen(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_nested_get_max_seqlen(Tensor self) -> Tensor
  
  auto dispatch__nested_get_max_seqlen = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_nested_get_max_seqlen(self);
  };
  return wrap(dispatch__nested_get_max_seqlen(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_compute_contiguous_strides_offsets
static PyObject * THPVariable__nested_compute_contiguous_strides_offsets(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_compute_contiguous_strides_offsets(Tensor nested_size)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_nested_compute_contiguous_strides_offsets(Tensor nested_size) -> (Tensor, Tensor)
  
  auto dispatch__nested_compute_contiguous_strides_offsets = [](const at::Tensor & nested_size) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_nested_compute_contiguous_strides_offsets(nested_size);
  };
  return wrap(dispatch__nested_compute_contiguous_strides_offsets(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _trilinear
static PyObject * THPVariable__trilinear(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_trilinear(Tensor i1, Tensor i2, Tensor i3, IntArrayRef expand1, IntArrayRef expand2, IntArrayRef expand3, IntArrayRef sumdim, int64_t unroll_dim=1)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_trilinear(Tensor i1, Tensor i2, Tensor i3, int[] expand1, int[] expand2, int[] expand3, int[] sumdim, int unroll_dim=1) -> Tensor
  
  auto dispatch__trilinear = [](const at::Tensor & i1, const at::Tensor & i2, const at::Tensor & i3, at::IntArrayRef expand1, at::IntArrayRef expand2, at::IntArrayRef expand3, at::IntArrayRef sumdim, int64_t unroll_dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
  };
  return wrap(dispatch__trilinear(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.intlist(4), _r.intlist(5), _r.intlist(6), _r.toInt64(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _unique2
static PyObject * THPVariable__unique2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_unique2(Tensor input, bool sorted=True, bool return_inverse=False, bool return_counts=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_unique2(Tensor self, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
  
  auto dispatch__unique2 = [](const at::Tensor & self, bool sorted, bool return_inverse, bool return_counts) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_unique2(self, sorted, return_inverse, return_counts);
  };
  return wrap(dispatch__unique2(_r.tensor(0), _r.toBool(1), _r.toBool(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unsqueeze
static PyObject * THPVariable_unsqueeze(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unsqueeze(Tensor input, int64_t dim)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
  
  auto dispatch_unsqueeze = [](const at::Tensor & self, int64_t dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.unsqueeze(dim);
  };
  return wrap(dispatch_unsqueeze(_r.tensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// var
static PyObject * THPVariable_var(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "var(Tensor input, IntArrayRef[1]? dim, bool unbiased=True, bool keepdim=False, *, Tensor out=None)",
    "var(Tensor input, IntArrayRef[1]? dim=None, *, Scalar? correction=None, bool keepdim=False, Tensor out=None)",
    "var(Tensor input, bool unbiased=True)",
    "var(Tensor input, DimnameList[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor out=None)",
    "var(Tensor input, DimnameList[1] dim, *, Scalar? correction=None, bool keepdim=False, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // aten::var.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor
        
        auto dispatch_var = [](const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.var(dim, unbiased, keepdim);
        };
        return wrap(dispatch_var(_r.tensor(0), _r.intlistOptional(1), _r.toBool(2), _r.toBool(3)));
      } else {
        // aten::var.out(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_var_out = [](at::Tensor out, const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::var_out(out, self, dim, unbiased, keepdim);
        };
        return wrap(dispatch_var_out(_r.tensor(4), _r.tensor(0), _r.intlistOptional(1), _r.toBool(2), _r.toBool(3)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
        
        auto dispatch_var = [](const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.var(dim, correction, keepdim);
        };
        return wrap(dispatch_var(_r.tensor(0), _r.intlistOptional(1), _r.scalarOptional(2), _r.toBool(3)));
      } else {
        // aten::var.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_var_out = [](at::Tensor out, const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::var_out(out, self, dim, correction, keepdim);
        };
        return wrap(dispatch_var_out(_r.tensor(4), _r.tensor(0), _r.intlistOptional(1), _r.scalarOptional(2), _r.toBool(3)));
      }
    }
    case 2: {
      // aten::var(Tensor self, bool unbiased=True) -> Tensor
      
      auto dispatch_var = [](const at::Tensor & self, bool unbiased) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.var(unbiased);
      };
      return wrap(dispatch_var(_r.tensor(0), _r.toBool(1)));
    }
    case 3: {
      if (_r.isNone(4)) {
        // aten::var.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
        
        auto dispatch_var = [](const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.var(dim, unbiased, keepdim);
        };
        return wrap(dispatch_var(_r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.toBool(3)));
      } else {
        // aten::var.names_out(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_var_out = [](at::Tensor out, const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::var_out(out, self, dim, unbiased, keepdim);
        };
        return wrap(dispatch_var_out(_r.tensor(4), _r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.toBool(3)));
      }
    }
    case 4: {
      if (_r.isNone(4)) {
        // aten::var.correction_names(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False) -> Tensor
        
        auto dispatch_var = [](const at::Tensor & self, at::DimnameList dim, const ::std::optional<at::Scalar> & correction, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.var(dim, correction, keepdim);
        };
        return wrap(dispatch_var(_r.tensor(0), _r.dimnamelist(1), _r.scalarOptional(2), _r.toBool(3)));
      } else {
        // aten::var.correction_names_out(Tensor self, Dimname[1] dim, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_var_out = [](at::Tensor out, const at::Tensor & self, at::DimnameList dim, const ::std::optional<at::Scalar> & correction, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::var_out(out, self, dim, correction, keepdim);
        };
        return wrap(dispatch_var_out(_r.tensor(4), _r.tensor(0), _r.dimnamelist(1), _r.scalarOptional(2), _r.toBool(3)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// norm_except_dim
static PyObject * THPVariable_norm_except_dim(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "norm_except_dim(Tensor v, int64_t pow=2, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::norm_except_dim(Tensor v, int pow=2, int dim=0) -> Tensor
  
  auto dispatch_norm_except_dim = [](const at::Tensor & v, int64_t pow, int64_t dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::norm_except_dim(v, pow, dim);
  };
  return wrap(dispatch_norm_except_dim(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _weight_norm_interface
static PyObject * THPVariable__weight_norm_interface(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_weight_norm_interface(Tensor v, Tensor g, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_weight_norm_interface(Tensor v, Tensor g, int dim=0) -> (Tensor, Tensor)
  
  auto dispatch__weight_norm_interface = [](const at::Tensor & v, const at::Tensor & g, int64_t dim) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_weight_norm_interface(v, g, dim);
  };
  return wrap(dispatch__weight_norm_interface(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// zeros
static PyObject * THPVariable_zeros(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "zeros(IntArrayRef size, *, DimnameList? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "zeros(SymIntArrayRef size, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::zeros.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      auto __names = _r.toDimnameListOptional(1);
      ::std::optional<DimnameList> names = __names ? ::std::make_optional(DimnameList(__names.value())) : ::std::nullopt;
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(2))
          .device(_r.deviceWithDefault(4, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(3))
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_zeros = [](at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::zeros(size, names, options);
      };
      return wrap(dispatch_zeros(_r.intlist(0), names, options));
    }
    case 1: {
      if (_r.isNone(1)) {
        // aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(2))
            .device(_r.deviceWithDefault(4, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(3))
            .requires_grad(_r.toBool(6))
            .pinned_memory(_r.toBool(5));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_zeros = [](c10::SymIntArrayRef size, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::zeros_symint(size, options);
        };
        return wrap(dispatch_zeros(_r.symintlist(0), options));
      } else {
        // aten::zeros.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(1), _r.scalartypeOptional(2),
                               _r.isNone(2), _r.layoutOptional(3),
                               _r.deviceWithDefault(4, torch::tensors::get_default_device()), _r.isNone(4));
        
        auto dispatch_zeros_out = [](at::Tensor out, c10::SymIntArrayRef size) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::zeros_symint_out(out, size);
        };
        return wrap(dispatch_zeros_out(_r.tensor(1), _r.symintlist(0)).set_requires_grad(_r.toBool(6)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _efficientzerotensor
static PyObject * THPVariable__efficientzerotensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_efficientzerotensor(SymIntArrayRef size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_efficientzerotensor(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(1))
      .device(_r.deviceWithDefault(3, torch::tensors::get_default_device()))
      .layout(_r.layoutOptional(2))
      .requires_grad(_r.toBool(5))
      .pinned_memory(_r.toBool(4));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch__efficientzerotensor = [](c10::SymIntArrayRef size, at::TensorOptions options) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::_efficientzerotensor_symint(size, options);
  };
  return wrap(dispatch__efficientzerotensor(_r.symintlist(0), options));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// zeros_like
static PyObject * THPVariable_zeros_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "zeros_like(Tensor input, *, MemoryFormat? memory_format=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
  auto self = _r.tensor(0);
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(2))
      .device(_r.deviceOptional(4))
      .layout(_r.layoutOptional(3))
      .requires_grad(_r.toBool(6))
      .pinned_memory(_r.toBool(5));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch_zeros_like = [](const at::Tensor & self, at::TensorOptions options, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::zeros_like(self, options, memory_format);
  };
  return wrap(dispatch_zeros_like(self, options, _r.memoryformatOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sample_dirichlet
static PyObject * THPVariable__sample_dirichlet(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sample_dirichlet(Tensor input, Generator? generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_sample_dirichlet(Tensor self, Generator? generator=None) -> Tensor
  
  auto dispatch__sample_dirichlet = [](const at::Tensor & self, ::std::optional<at::Generator> generator) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sample_dirichlet(self, generator);
  };
  return wrap(dispatch__sample_dirichlet(_r.tensor(0), _r.generator(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _sparse_sum
static PyObject * THPVariable__sparse_sum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_sum(Tensor input)",
    "_sparse_sum(Tensor input, *, ScalarType dtype)",
    "_sparse_sum(Tensor input, IntArrayRef[1] dim)",
    "_sparse_sum(Tensor input, IntArrayRef[1] dim, *, ScalarType dtype)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_sparse_sum(Tensor self) -> Tensor
      
      auto dispatch__sparse_sum = [](const at::Tensor & self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_sparse_sum(self);
      };
      return wrap(dispatch__sparse_sum(_r.tensor(0)));
    }
    case 1: {
      // aten::_sparse_sum.dtype(Tensor self, *, ScalarType dtype) -> Tensor
      
      auto dispatch__sparse_sum = [](const at::Tensor & self, at::ScalarType dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_sparse_sum(self, dtype);
      };
      return wrap(dispatch__sparse_sum(_r.tensor(0), _r.scalartype(1)));
    }
    case 2: {
      // aten::_sparse_sum.dim(Tensor self, int[1] dim) -> Tensor
      
      auto dispatch__sparse_sum = [](const at::Tensor & self, at::IntArrayRef dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_sparse_sum(self, dim);
      };
      return wrap(dispatch__sparse_sum(_r.tensor(0), _r.intlist(1)));
    }
    case 3: {
      // aten::_sparse_sum.dim_dtype(Tensor self, int[1] dim, *, ScalarType dtype) -> Tensor
      
      auto dispatch__sparse_sum = [](const at::Tensor & self, at::IntArrayRef dim, at::ScalarType dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_sparse_sum(self, dim, dtype);
      };
      return wrap(dispatch__sparse_sum(_r.tensor(0), _r.intlist(1), _r.scalartype(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sparse_csr_prod
static PyObject * THPVariable__sparse_csr_prod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_csr_prod(Tensor input, IntArrayRef[1] dim, bool keepdim=False, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_sparse_csr_prod.dim_dtype(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  
  auto dispatch__sparse_csr_prod = [](const at::Tensor & self, at::IntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sparse_csr_prod(self, dim, keepdim, dtype);
  };
  return wrap(dispatch__sparse_csr_prod(_r.tensor(0), _r.intlist(1), _r.toBool(2), _r.scalartypeOptional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// frobenius_norm
static PyObject * THPVariable_frobenius_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "frobenius_norm(Tensor input, IntArrayRef[1] dim, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::frobenius_norm.dim(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
    
    auto dispatch_frobenius_norm = [](const at::Tensor & self, at::IntArrayRef dim, bool keepdim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::frobenius_norm(self, dim, keepdim);
    };
    return wrap(dispatch_frobenius_norm(_r.tensor(0), _r.intlist(1), _r.toBool(2)));
  } else {
    // aten::frobenius_norm.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_frobenius_norm_out = [](at::Tensor out, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::frobenius_norm_out(out, self, dim, keepdim);
    };
    return wrap(dispatch_frobenius_norm_out(_r.tensor(3), _r.tensor(0), _r.intlist(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// resize_as_sparse_
static PyObject * THPVariable_resize_as_sparse_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "resize_as_sparse_(Tensor input, Tensor the_template)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::resize_as_sparse_(Tensor(a!) self, Tensor the_template) -> Tensor(a!)
  
  auto dispatch_resize_as_sparse_ = [](const at::Tensor & self, const at::Tensor & the_template) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.resize_as_sparse_(the_template);
  };
  return wrap(dispatch_resize_as_sparse_(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addmm
static PyObject * THPVariable_addmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addmm(Scalar beta, Tensor input, Scalar alpha, Tensor mat1, Tensor mat2, *, Tensor out=None)|deprecated",
    "addmm(Scalar beta, Tensor input, Tensor mat1, Tensor mat2, *, Tensor out=None)|deprecated",
    "addmm(Tensor input, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(5)) {
        // [deprecated] aten::addmm(Scalar beta, Tensor self, Scalar alpha, Tensor mat1, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addmm = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & mat1, const at::Tensor & mat2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addmm(mat1, mat2, beta, alpha);
        };
        return wrap(dispatch_addmm(_r.tensor(5), _r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4)));
      } else {
        // [deprecated] aten::addmm(Scalar beta, Tensor self, Scalar alpha, Tensor mat1, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addmm_out = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & mat1, const at::Tensor & mat2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addmm_out(out, self, mat1, mat2, beta, alpha);
        };
        return wrap(dispatch_addmm_out(_r.tensor(5), _r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // [deprecated] aten::addmm(Scalar beta, Tensor self, Tensor mat1, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addmm = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addmm(mat1, mat2, beta, 1);
        };
        return wrap(dispatch_addmm(_r.tensor(4), _r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
      } else {
        // [deprecated] aten::addmm(Scalar beta, Tensor self, Tensor mat1, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addmm_out = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addmm_out(out, self, mat1, mat2, beta, 1);
        };
        return wrap(dispatch_addmm_out(_r.tensor(4), _r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
      }
    }
    case 2: {
      if (_r.isNone(5)) {
        // aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        
        auto dispatch_addmm = [](const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addmm(mat1, mat2, beta, alpha);
        };
        return wrap(dispatch_addmm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      } else {
        // aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addmm_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addmm_out(out, self, mat1, mat2, beta, alpha);
        };
        return wrap(dispatch_addmm_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _addmm_activation
static PyObject * THPVariable__addmm_activation(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_addmm_activation(Tensor input, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(6)) {
    // aten::_addmm_activation(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False) -> Tensor
    
    auto dispatch__addmm_activation = [](const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, bool use_gelu) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self._addmm_activation(mat1, mat2, beta, alpha, use_gelu);
    };
    return wrap(dispatch__addmm_activation(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4), _r.toBool(5)));
  } else {
    // aten::_addmm_activation.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__addmm_activation_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, bool use_gelu) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_addmm_activation_out(out, self, mat1, mat2, beta, alpha, use_gelu);
    };
    return wrap(dispatch__addmm_activation_out(_r.tensor(6), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4), _r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _scaled_grouped_mm
static PyObject * THPVariable__scaled_grouped_mm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_scaled_grouped_mm(Tensor input, Tensor mat2, Tensor scale_a, Tensor scale_b, Tensor? offs=None, Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, bool use_fast_accum=False)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_scaled_grouped_mm(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b, Tensor? offs=None, Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, bool use_fast_accum=False) -> Tensor
  
  auto dispatch__scaled_grouped_mm = [](const at::Tensor & self, const at::Tensor & mat2, const at::Tensor & scale_a, const at::Tensor & scale_b, const ::std::optional<at::Tensor> & offs, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & scale_result, ::std::optional<at::ScalarType> out_dtype, bool use_fast_accum) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_scaled_grouped_mm(self, mat2, scale_a, scale_b, offs, bias, scale_result, out_dtype, use_fast_accum);
  };
  return wrap(dispatch__scaled_grouped_mm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.optionalTensor(4), _r.optionalTensor(5), _r.optionalTensor(6), _r.scalartypeOptional(7), _r.toBool(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _validate_sparse_coo_tensor_args
static PyObject * THPVariable__validate_sparse_coo_tensor_args(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_validate_sparse_coo_tensor_args(Tensor indices, Tensor values, IntArrayRef size, bool? is_coalesced=None)",
  }, /*traceable=*/false);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_validate_sparse_coo_tensor_args(Tensor indices, Tensor values, int[] size, bool? is_coalesced=None) -> ()
  
  auto dispatch__validate_sparse_coo_tensor_args = [](const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, ::std::optional<bool> is_coalesced) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_validate_sparse_coo_tensor_args(indices, values, size, is_coalesced);
  };
  dispatch__validate_sparse_coo_tensor_args(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.toBoolOptional(3));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _validate_sparse_bsr_tensor_args
static PyObject * THPVariable__validate_sparse_bsr_tensor_args(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_validate_sparse_bsr_tensor_args(Tensor crow_indices, Tensor col_indices, Tensor values, IntArrayRef size)",
  }, /*traceable=*/false);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_validate_sparse_bsr_tensor_args(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size) -> ()
  
  auto dispatch__validate_sparse_bsr_tensor_args = [](const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::IntArrayRef size) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_validate_sparse_bsr_tensor_args(crow_indices, col_indices, values, size);
  };
  dispatch__validate_sparse_bsr_tensor_args(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _validate_sparse_bsc_tensor_args
static PyObject * THPVariable__validate_sparse_bsc_tensor_args(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_validate_sparse_bsc_tensor_args(Tensor ccol_indices, Tensor row_indices, Tensor values, IntArrayRef size)",
  }, /*traceable=*/false);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_validate_sparse_bsc_tensor_args(Tensor ccol_indices, Tensor row_indices, Tensor values, int[] size) -> ()
  
  auto dispatch__validate_sparse_bsc_tensor_args = [](const at::Tensor & ccol_indices, const at::Tensor & row_indices, const at::Tensor & values, at::IntArrayRef size) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_validate_sparse_bsc_tensor_args(ccol_indices, row_indices, values, size);
  };
  dispatch__validate_sparse_bsc_tensor_args(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// dequantize
static PyObject * THPVariable_dequantize(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dequantize(Tensor input)",
    "dequantize(TensorList tensors)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::dequantize.self(Tensor self) -> Tensor
      
      auto dispatch_dequantize = [](const at::Tensor & self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.dequantize();
      };
      return wrap(dispatch_dequantize(_r.tensor(0)));
    }
    case 1: {
      // aten::dequantize.tensors(Tensor[] tensors) -> Tensor[]
      
      auto dispatch_dequantize = [](at::TensorList tensors) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::dequantize(tensors);
      };
      return wrap(dispatch_dequantize(_r.tensorlist(0)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// q_scale
static PyObject * THPVariable_q_scale(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "q_scale(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::q_scale(Tensor self) -> float
  
  auto dispatch_q_scale = [](const at::Tensor & self) -> double {
    pybind11::gil_scoped_release no_gil;
    return self.q_scale();
  };
  return wrap(dispatch_q_scale(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// fake_quantize_per_tensor_affine
static PyObject * THPVariable_fake_quantize_per_tensor_affine(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fake_quantize_per_tensor_affine(Tensor input, Tensor scale, Tensor zero_point, int64_t quant_min, int64_t quant_max)",
    "fake_quantize_per_tensor_affine(Tensor input, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::fake_quantize_per_tensor_affine.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max) -> Tensor
      
      auto dispatch_fake_quantize_per_tensor_affine = [](const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t quant_min, int64_t quant_max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::fake_quantize_per_tensor_affine(self, scale, zero_point, quant_min, quant_max);
      };
      return wrap(dispatch_fake_quantize_per_tensor_affine(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3), _r.toInt64(4)));
    }
    case 1: {
      // aten::fake_quantize_per_tensor_affine(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> Tensor
      
      auto dispatch_fake_quantize_per_tensor_affine = [](const at::Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::fake_quantize_per_tensor_affine(self, scale, zero_point, quant_min, quant_max);
      };
      return wrap(dispatch_fake_quantize_per_tensor_affine(_r.tensor(0), _r.toDouble(1), _r.toInt64(2), _r.toInt64(3), _r.toInt64(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _fake_quantize_learnable_per_channel_affine
static PyObject * THPVariable__fake_quantize_learnable_per_channel_affine(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_fake_quantize_learnable_per_channel_affine(Tensor input, Tensor scale, Tensor zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor=1.0)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_fake_quantize_learnable_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, float grad_factor=1.0) -> Tensor
  
  auto dispatch__fake_quantize_learnable_per_channel_affine = [](const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_fake_quantize_learnable_per_channel_affine(self, scale, zero_point, axis, quant_min, quant_max, grad_factor);
  };
  return wrap(dispatch__fake_quantize_learnable_per_channel_affine(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3), _r.toInt64(4), _r.toInt64(5), _r.toDouble(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cartesian_prod
static PyObject * THPVariable_cartesian_prod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cartesian_prod(TensorList tensors)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::cartesian_prod(Tensor[] tensors) -> Tensor
  
  auto dispatch_cartesian_prod = [](at::TensorList tensors) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::cartesian_prod(tensors);
  };
  return wrap(dispatch_cartesian_prod(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// rnn_tanh
static PyObject * THPVariable_rnn_tanh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rnn_tanh(Tensor data, Tensor batch_sizes, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)",
    "rnn_tanh(Tensor input, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::rnn_tanh.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
      
      auto dispatch_rnn_tanh = [](const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::rnn_tanh(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
      };
      return wrap(dispatch_rnn_tanh(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensorlist(3), _r.toBool(4), _r.toInt64(5), _r.toDouble(6), _r.toBool(7), _r.toBool(8)));
    }
    case 1: {
      // aten::rnn_tanh.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
      
      auto dispatch_rnn_tanh = [](const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::rnn_tanh(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
      };
      return wrap(dispatch_rnn_tanh(_r.tensor(0), _r.tensor(1), _r.tensorlist(2), _r.toBool(3), _r.toInt64(4), _r.toDouble(5), _r.toBool(6), _r.toBool(7), _r.toBool(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// rnn_relu
static PyObject * THPVariable_rnn_relu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rnn_relu(Tensor data, Tensor batch_sizes, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)",
    "rnn_relu(Tensor input, Tensor hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::rnn_relu.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
      
      auto dispatch_rnn_relu = [](const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::rnn_relu(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
      };
      return wrap(dispatch_rnn_relu(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensorlist(3), _r.toBool(4), _r.toInt64(5), _r.toDouble(6), _r.toBool(7), _r.toBool(8)));
    }
    case 1: {
      // aten::rnn_relu.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
      
      auto dispatch_rnn_relu = [](const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::rnn_relu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
      };
      return wrap(dispatch_rnn_relu(_r.tensor(0), _r.tensor(1), _r.tensorlist(2), _r.toBool(3), _r.toInt64(4), _r.toDouble(5), _r.toBool(6), _r.toBool(7), _r.toBool(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// gru_cell
static PyObject * THPVariable_gru_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None)",
  }, /*traceable=*/false);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
  
  auto dispatch_gru_cell = [](const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const ::std::optional<at::Tensor> & b_ih, const ::std::optional<at::Tensor> & b_hh) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  };
  return wrap(dispatch_gru_cell(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.optionalTensor(4), _r.optionalTensor(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rnn_relu_cell
static PyObject * THPVariable_rnn_relu_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None)",
  }, /*traceable=*/false);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
  
  auto dispatch_rnn_relu_cell = [](const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const ::std::optional<at::Tensor> & b_ih, const ::std::optional<at::Tensor> & b_hh) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  };
  return wrap(dispatch_rnn_relu_cell(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.optionalTensor(4), _r.optionalTensor(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// quantized_lstm_cell
static PyObject * THPVariable_quantized_lstm_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantized_lstm_cell(Tensor input, TensorList hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh)",
  }, /*traceable=*/true);

  ParsedArgs<14> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::quantized_lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> (Tensor, Tensor)
  
  auto dispatch_quantized_lstm_cell = [](const at::Tensor & input, at::TensorList hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::quantized_lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  };
  return wrap(dispatch_quantized_lstm_cell(_r.tensor(0), _r.tensorlist(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.tensor(5), _r.tensor(6), _r.tensor(7), _r.tensor(8), _r.tensor(9), _r.scalar(10), _r.scalar(11), _r.scalar(12), _r.scalar(13)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _pack_padded_sequence
static PyObject * THPVariable__pack_padded_sequence(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first) -> (Tensor, Tensor)
  
  auto dispatch__pack_padded_sequence = [](const at::Tensor & input, const at::Tensor & lengths, bool batch_first) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_pack_padded_sequence(input, lengths, batch_first);
  };
  return wrap(dispatch__pack_padded_sequence(_r.tensor(0), _r.tensor(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _pad_packed_sequence
static PyObject * THPVariable__pad_packed_sequence(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_pad_packed_sequence(Tensor data, Tensor batch_sizes, bool batch_first, Scalar padding_value, int64_t total_length)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_pad_packed_sequence(Tensor data, Tensor batch_sizes, bool batch_first, Scalar padding_value, int total_length) -> (Tensor, Tensor)
  
  auto dispatch__pad_packed_sequence = [](const at::Tensor & data, const at::Tensor & batch_sizes, bool batch_first, const at::Scalar & padding_value, int64_t total_length) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_pad_packed_sequence(data, batch_sizes, batch_first, padding_value, total_length);
  };
  return wrap(dispatch__pad_packed_sequence(_r.tensor(0), _r.tensor(1), _r.toBool(2), _r.scalar(3), _r.toInt64(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// masked_fill
static PyObject * THPVariable_masked_fill(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "masked_fill(Tensor input, Tensor mask, Tensor value)",
    "masked_fill(Tensor input, Tensor mask, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor
      
      auto dispatch_masked_fill = [](const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.masked_fill(mask, value);
      };
      return wrap(dispatch_masked_fill(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
    }
    case 1: {
      // aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
      
      auto dispatch_masked_fill = [](const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.masked_fill(mask, value);
      };
      return wrap(dispatch_masked_fill(_r.tensor(0), _r.tensor(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _masked_softmax
static PyObject * THPVariable__masked_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_masked_softmax(Tensor input, Tensor mask, int64_t? dim=None, int64_t? mask_type=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_masked_softmax(Tensor self, Tensor mask, int? dim=None, int? mask_type=None) -> Tensor
  
  auto dispatch__masked_softmax = [](const at::Tensor & self, const at::Tensor & mask, ::std::optional<int64_t> dim, ::std::optional<int64_t> mask_type) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_masked_softmax(self, mask, dim, mask_type);
  };
  return wrap(dispatch__masked_softmax(_r.tensor(0), _r.tensor(1), _r.toInt64Optional(2), _r.toInt64Optional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// index_fill
static PyObject * THPVariable_index_fill(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_fill(Tensor input, int64_t dim, Tensor index, Tensor value)",
    "index_fill(Tensor input, Dimname dim, Tensor index, Tensor value)",
    "index_fill(Tensor input, int64_t dim, Tensor index, Scalar value)",
    "index_fill(Tensor input, Dimname dim, Tensor index, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::index_fill.int_Tensor(Tensor self, int dim, Tensor index, Tensor value) -> Tensor
      
      auto dispatch_index_fill = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill(dim, index, value);
      };
      return wrap(dispatch_index_fill(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3)));
    }
    case 1: {
      // aten::index_fill.Dimname_Tensor(Tensor self, Dimname dim, Tensor index, Tensor value) -> Tensor
      
      auto dispatch_index_fill = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill(dim, index, value);
      };
      return wrap(dispatch_index_fill(_r.tensor(0), _r.dimname(1), _r.tensor(2), _r.tensor(3)));
    }
    case 2: {
      // aten::index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
      
      auto dispatch_index_fill = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill(dim, index, value);
      };
      return wrap(dispatch_index_fill(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.scalar(3)));
    }
    case 3: {
      // aten::index_fill.Dimname_Scalar(Tensor self, Dimname dim, Tensor index, Scalar value) -> Tensor
      
      auto dispatch_index_fill = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_fill(dim, index, value);
      };
      return wrap(dispatch_index_fill(_r.tensor(0), _r.dimname(1), _r.tensor(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// scatter
static PyObject * THPVariable_scatter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "scatter(Tensor input, int64_t dim, Tensor index, Tensor src, *, Tensor out=None)",
    "scatter(Tensor input, int64_t dim, Tensor index, Tensor src, *, c10::string_view reduce, Tensor out=None)",
    "scatter(Tensor input, Dimname dim, Tensor index, Tensor src)",
    "scatter(Tensor input, int64_t dim, Tensor index, Scalar value, *, Tensor out=None)",
    "scatter(Tensor input, int64_t dim, Tensor index, Scalar value, *, c10::string_view reduce, Tensor out=None)",
    "scatter(Tensor input, Dimname dim, Tensor index, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
        
        auto dispatch_scatter = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.scatter(dim, index, src);
        };
        return wrap(dispatch_scatter(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3)));
      } else {
        // aten::scatter.src_out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_scatter_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::scatter_out(out, self, dim, index, src);
        };
        return wrap(dispatch_scatter_out(_r.tensor(4), _r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3)));
      }
    }
    case 1: {
      if (_r.isNone(5)) {
        // aten::scatter.reduce(Tensor self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor
        
        auto dispatch_scatter = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.scatter(dim, index, src, reduce);
        };
        return wrap(dispatch_scatter(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3), _r.stringView(4)));
      } else {
        // aten::scatter.reduce_out(Tensor self, int dim, Tensor index, Tensor src, *, str reduce, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_scatter_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::scatter_out(out, self, dim, index, src, reduce);
        };
        return wrap(dispatch_scatter_out(_r.tensor(5), _r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3), _r.stringView(4)));
      }
    }
    case 2: {
      // aten::scatter.dimname_src(Tensor self, Dimname dim, Tensor index, Tensor src) -> Tensor
      
      auto dispatch_scatter = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter(dim, index, src);
      };
      return wrap(dispatch_scatter(_r.tensor(0), _r.dimname(1), _r.tensor(2), _r.tensor(3)));
    }
    case 3: {
      if (_r.isNone(4)) {
        // aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
        
        auto dispatch_scatter = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.scatter(dim, index, value);
        };
        return wrap(dispatch_scatter(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.scalar(3)));
      } else {
        // aten::scatter.value_out(Tensor self, int dim, Tensor index, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_scatter_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::scatter_out(out, self, dim, index, value);
        };
        return wrap(dispatch_scatter_out(_r.tensor(4), _r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.scalar(3)));
      }
    }
    case 4: {
      if (_r.isNone(5)) {
        // aten::scatter.value_reduce(Tensor self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor
        
        auto dispatch_scatter = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.scatter(dim, index, value, reduce);
        };
        return wrap(dispatch_scatter(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.scalar(3), _r.stringView(4)));
      } else {
        // aten::scatter.value_reduce_out(Tensor self, int dim, Tensor index, Scalar value, *, str reduce, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_scatter_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::scatter_out(out, self, dim, index, value, reduce);
        };
        return wrap(dispatch_scatter_out(_r.tensor(5), _r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.scalar(3), _r.stringView(4)));
      }
    }
    case 5: {
      // aten::scatter.dimname_value(Tensor self, Dimname dim, Tensor index, Scalar value) -> Tensor
      
      auto dispatch_scatter = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter(dim, index, value);
      };
      return wrap(dispatch_scatter(_r.tensor(0), _r.dimname(1), _r.tensor(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// __and__
static PyObject * THPVariable___and__(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "__and__(Tensor input, Tensor other)",
    "__and__(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch___and__ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__and__(other);
      };
      return wrap(dispatch___and__(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch___and__ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__and__(other);
      };
      return wrap(dispatch___and__(_r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bitwise_or
static PyObject * THPVariable_bitwise_or(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bitwise_or(Tensor input, Tensor other, *, Tensor out=None)",
    "bitwise_or(Scalar self, Tensor other)",
    "bitwise_or(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_bitwise_or = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.bitwise_or(other);
        };
        return wrap(dispatch_bitwise_or(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::bitwise_or.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_bitwise_or_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bitwise_or_out(out, self, other);
        };
        return wrap(dispatch_bitwise_or_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      // aten::bitwise_or.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
      
      auto dispatch_bitwise_or = [](const at::Scalar & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::bitwise_or(self, other);
      };
      return wrap(dispatch_bitwise_or(_r.scalar(0), _r.tensor(1)));
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_bitwise_or = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.bitwise_or(other);
        };
        return wrap(dispatch_bitwise_or(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::bitwise_or.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_bitwise_or_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bitwise_or_out(out, self, other);
        };
        return wrap(dispatch_bitwise_or_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// triu
static PyObject * THPVariable_triu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "triu(Tensor input, int64_t diagonal=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::triu(Tensor self, int diagonal=0) -> Tensor
    
    auto dispatch_triu = [](const at::Tensor & self, int64_t diagonal) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.triu(diagonal);
    };
    return wrap(dispatch_triu(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::triu.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_triu_out = [](at::Tensor out, const at::Tensor & self, int64_t diagonal) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::triu_out(out, self, diagonal);
    };
    return wrap(dispatch_triu_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// ge
static PyObject * THPVariable_ge(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ge(Tensor input, Tensor other, *, Tensor out=None)",
    "ge(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::ge.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_ge = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.ge(other);
        };
        return wrap(dispatch_ge(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_ge_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::ge_out(out, self, other);
        };
        return wrap(dispatch_ge_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::ge.Scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_ge = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.ge(other);
        };
        return wrap(dispatch_ge(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_ge_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::ge_out(out, self, other);
        };
        return wrap(dispatch_ge_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// le
static PyObject * THPVariable_le(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "le(Tensor input, Tensor other, *, Tensor out=None)",
    "le(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::le.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_le = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.le(other);
        };
        return wrap(dispatch_le(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_le_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::le_out(out, self, other);
        };
        return wrap(dispatch_le_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::le.Scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_le = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.le(other);
        };
        return wrap(dispatch_le(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_le_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::le_out(out, self, other);
        };
        return wrap(dispatch_le_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
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
  static PythonArgParser parser({
    "less_equal(Tensor input, Tensor other, *, Tensor out=None)",
    "less_equal(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::less_equal.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_less_equal = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.less_equal(other);
        };
        return wrap(dispatch_less_equal(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::less_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_less_equal_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::less_equal_out(out, self, other);
        };
        return wrap(dispatch_less_equal_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::less_equal.Scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_less_equal = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.less_equal(other);
        };
        return wrap(dispatch_less_equal(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::less_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_less_equal_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::less_equal_out(out, self, other);
        };
        return wrap(dispatch_less_equal_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
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
  static PythonArgParser parser({
    "gt(Tensor input, Tensor other, *, Tensor out=None)",
    "gt(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::gt.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_gt = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.gt(other);
        };
        return wrap(dispatch_gt(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_gt_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::gt_out(out, self, other);
        };
        return wrap(dispatch_gt_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::gt.Scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_gt = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.gt(other);
        };
        return wrap(dispatch_gt(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_gt_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::gt_out(out, self, other);
        };
        return wrap(dispatch_gt_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
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
  static PythonArgParser parser({
    "lt(Tensor input, Tensor other, *, Tensor out=None)",
    "lt(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::lt.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_lt = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.lt(other);
        };
        return wrap(dispatch_lt(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_lt_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::lt_out(out, self, other);
        };
        return wrap(dispatch_lt_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::lt.Scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_lt = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.lt(other);
        };
        return wrap(dispatch_lt(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_lt_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::lt_out(out, self, other);
        };
        return wrap(dispatch_lt_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// take_along_dim
static PyObject * THPVariable_take_along_dim(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "take_along_dim(Tensor input, Tensor indices, int64_t? dim=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::take_along_dim(Tensor self, Tensor indices, int? dim=None) -> Tensor
    
    auto dispatch_take_along_dim = [](const at::Tensor & self, const at::Tensor & indices, ::std::optional<int64_t> dim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.take_along_dim(indices, dim);
    };
    return wrap(dispatch_take_along_dim(_r.tensor(0), _r.tensor(1), _r.toInt64Optional(2)));
  } else {
    // aten::take_along_dim.out(Tensor self, Tensor indices, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_take_along_dim_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & indices, ::std::optional<int64_t> dim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::take_along_dim_out(out, self, indices, dim);
    };
    return wrap(dispatch_take_along_dim_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toInt64Optional(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// masked_select
static PyObject * THPVariable_masked_select(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "masked_select(Tensor input, Tensor mask, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::masked_select(Tensor self, Tensor mask) -> Tensor
    
    auto dispatch_masked_select = [](const at::Tensor & self, const at::Tensor & mask) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.masked_select(mask);
    };
    return wrap(dispatch_masked_select(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::masked_select.out(Tensor self, Tensor mask, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_masked_select_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & mask) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::masked_select_out(out, self, mask);
    };
    return wrap(dispatch_masked_select_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// nonzero_static
static PyObject * THPVariable_nonzero_static(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nonzero_static(Tensor input, *, SymInt size, int64_t fill_value=-1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::nonzero_static(Tensor self, *, SymInt size, int fill_value=-1) -> Tensor
    
    auto dispatch_nonzero_static = [](const at::Tensor & self, c10::SymInt size, int64_t fill_value) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.nonzero_static_symint(size, fill_value);
    };
    return wrap(dispatch_nonzero_static(_r.tensor(0), _r.toSymInt(1), _r.toInt64(2)));
  } else {
    // aten::nonzero_static.out(Tensor self, *, SymInt size, int fill_value=-1, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_nonzero_static_out = [](at::Tensor out, const at::Tensor & self, c10::SymInt size, int64_t fill_value) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::nonzero_static_symint_out(out, self, size, fill_value);
    };
    return wrap(dispatch_nonzero_static_out(_r.tensor(3), _r.tensor(0), _r.toSymInt(1), _r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// argwhere
static PyObject * THPVariable_argwhere(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "argwhere(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::argwhere(Tensor self) -> Tensor
  
  auto dispatch_argwhere = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.argwhere();
  };
  return wrap(dispatch_argwhere(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// addcdiv
static PyObject * THPVariable_addcdiv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "addcdiv(Tensor input, Scalar value, Tensor tensor1, Tensor tensor2, *, Tensor out=None)|deprecated",
    "addcdiv(Tensor input, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // [deprecated] aten::addcdiv(Tensor self, Scalar value, Tensor tensor1, Tensor tensor2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addcdiv = [](at::Tensor out, const at::Tensor & self, const at::Scalar & value, const at::Tensor & tensor1, const at::Tensor & tensor2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addcdiv(tensor1, tensor2, value);
        };
        return wrap(dispatch_addcdiv(_r.tensor(4), _r.tensor(0), _r.scalar(1), _r.tensor(2), _r.tensor(3)));
      } else {
        // [deprecated] aten::addcdiv(Tensor self, Scalar value, Tensor tensor1, Tensor tensor2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addcdiv_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & value, const at::Tensor & tensor1, const at::Tensor & tensor2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addcdiv_out(out, self, tensor1, tensor2, value);
        };
        return wrap(dispatch_addcdiv_out(_r.tensor(4), _r.tensor(0), _r.scalar(1), _r.tensor(2), _r.tensor(3)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
        
        auto dispatch_addcdiv = [](const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addcdiv(tensor1, tensor2, value);
        };
        return wrap(dispatch_addcdiv(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3)));
      } else {
        // aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addcdiv_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addcdiv_out(out, self, tensor1, tensor2, value);
        };
        return wrap(dispatch_addcdiv_out(_r.tensor(4), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// swapdims
static PyObject * THPVariable_swapdims(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "swapdims(Tensor input, int64_t dim0, int64_t dim1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::swapdims(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
  
  auto dispatch_swapdims = [](const at::Tensor & self, int64_t dim0, int64_t dim1) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.swapdims(dim0, dim1);
  };
  return wrap(dispatch_swapdims(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cholesky_solve
static PyObject * THPVariable_cholesky_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cholesky_solve(Tensor input, Tensor input2, bool upper=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor
    
    auto dispatch_cholesky_solve = [](const at::Tensor & self, const at::Tensor & input2, bool upper) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.cholesky_solve(input2, upper);
    };
    return wrap(dispatch_cholesky_solve(_r.tensor(0), _r.tensor(1), _r.toBool(2)));
  } else {
    // aten::cholesky_solve.out(Tensor self, Tensor input2, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_cholesky_solve_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & input2, bool upper) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::cholesky_solve_out(out, self, input2, upper);
    };
    return wrap(dispatch_cholesky_solve_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// qr
static PyObject * THPVariable_qr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_qr_out_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_qr_structseq();
  static PythonArgParser parser({
    "qr(Tensor input, bool some=True, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::qr(Tensor self, bool some=True) -> (Tensor Q, Tensor R)
    
    auto dispatch_qr = [](const at::Tensor & self, bool some) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return self.qr(some);
    };
    return wrap(NamedTuple1, dispatch_qr(_r.tensor(0), _r.toBool(1)));
  } else {
    // aten::qr.Q(Tensor self, bool some=True, *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_qr_out = [](at::Tensor & Q, at::Tensor & R, const at::Tensor & self, bool some) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::qr_out(Q, R, self, some);
    };
    return wrap(NamedTuple, dispatch_qr_out(out[0], out[1], _r.tensor(0), _r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// polygamma
static PyObject * THPVariable_polygamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "polygamma(int64_t n, Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::polygamma(int n, Tensor self) -> Tensor
    
    auto dispatch_polygamma = [](int64_t n, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.polygamma(n);
    };
    return wrap(dispatch_polygamma(_r.toInt64(0), _r.tensor(1)));
  } else {
    // aten::polygamma.out(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_polygamma_out = [](at::Tensor out, int64_t n, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::polygamma_out(out, n, self);
    };
    return wrap(dispatch_polygamma_out(_r.tensor(2), _r.toInt64(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// signbit
static PyObject * THPVariable_signbit(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "signbit(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::signbit(Tensor self) -> Tensor
    
    auto dispatch_signbit = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.signbit();
    };
    return wrap(dispatch_signbit(_r.tensor(0)));
  } else {
    // aten::signbit.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_signbit_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::signbit_out(out, self);
    };
    return wrap(dispatch_signbit_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// lerp
static PyObject * THPVariable_lerp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lerp(Tensor input, Tensor end, Tensor weight, *, Tensor out=None)",
    "lerp(Tensor input, Tensor end, Scalar weight, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor
        
        auto dispatch_lerp = [](const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.lerp(end, weight);
        };
        return wrap(dispatch_lerp(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
      } else {
        // aten::lerp.Tensor_out(Tensor self, Tensor end, Tensor weight, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_lerp_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::lerp_out(out, self, end, weight);
        };
        return wrap(dispatch_lerp_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.tensor(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor
        
        auto dispatch_lerp = [](const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.lerp(end, weight);
        };
        return wrap(dispatch_lerp(_r.tensor(0), _r.tensor(1), _r.scalar(2)));
      } else {
        // aten::lerp.Scalar_out(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_lerp_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::lerp_out(out, self, end, weight);
        };
        return wrap(dispatch_lerp_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.scalar(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _histogramdd_from_bin_cts
static PyObject * THPVariable__histogramdd_from_bin_cts(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_histogramdd_from_bin_cts(Tensor input, IntArrayRef bins, *, ArrayRef<double>? range=None, Tensor? weight=None, bool density=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_histogramdd_from_bin_cts(Tensor self, int[] bins, *, float[]? range=None, Tensor? weight=None, bool density=False) -> Tensor
  
  auto dispatch__histogramdd_from_bin_cts = [](const at::Tensor & self, at::IntArrayRef bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_histogramdd_from_bin_cts(self, bins, range, weight, density);
  };
  return wrap(dispatch__histogramdd_from_bin_cts(_r.tensor(0), _r.intlist(1), _r.doublelistOptional(2), _r.optionalTensor(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _histogramdd_from_bin_tensors
static PyObject * THPVariable__histogramdd_from_bin_tensors(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_histogramdd_from_bin_tensors(Tensor input, TensorList bins, *, Tensor? weight=None, bool density=False)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_histogramdd_from_bin_tensors(Tensor self, Tensor[] bins, *, Tensor? weight=None, bool density=False) -> Tensor
  
  auto dispatch__histogramdd_from_bin_tensors = [](const at::Tensor & self, at::TensorList bins, const ::std::optional<at::Tensor> & weight, bool density) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_histogramdd_from_bin_tensors(self, bins, weight, density);
  };
  return wrap(dispatch__histogramdd_from_bin_tensors(_r.tensor(0), _r.tensorlist(1), _r.optionalTensor(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// fmod
static PyObject * THPVariable_fmod(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fmod(Tensor input, Tensor other, *, Tensor out=None)",
    "fmod(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_fmod = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.fmod(other);
        };
        return wrap(dispatch_fmod(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::fmod.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_fmod_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::fmod_out(out, self, other);
        };
        return wrap(dispatch_fmod_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_fmod = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.fmod(other);
        };
        return wrap(dispatch_fmod(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::fmod.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_fmod_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::fmod_out(out, self, other);
        };
        return wrap(dispatch_fmod_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// igammac
static PyObject * THPVariable_igammac(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "igammac(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::igammac(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_igammac = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.igammac(other);
    };
    return wrap(dispatch_igammac(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::igammac.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_igammac_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::igammac_out(out, self, other);
    };
    return wrap(dispatch_igammac_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// remainder
static PyObject * THPVariable_remainder(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "remainder(Tensor input, Tensor other, *, Tensor out=None)",
    "remainder(Scalar self, Tensor other)",
    "remainder(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_remainder = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.remainder(other);
        };
        return wrap(dispatch_remainder(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::remainder.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_remainder_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::remainder_out(out, self, other);
        };
        return wrap(dispatch_remainder_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      // aten::remainder.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
      
      auto dispatch_remainder = [](const at::Scalar & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::remainder(self, other);
      };
      return wrap(dispatch_remainder(_r.scalar(0), _r.tensor(1)));
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_remainder = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.remainder(other);
        };
        return wrap(dispatch_remainder(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::remainder.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_remainder_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::remainder_out(out, self, other);
        };
        return wrap(dispatch_remainder_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// quantile
static PyObject * THPVariable_quantile(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantile(Tensor input, Tensor q, int64_t? dim=None, bool keepdim=False, *, c10::string_view interpolation=\"linear\", Tensor out=None)",
    "quantile(Tensor input, double q, int64_t? dim=None, bool keepdim=False, *, c10::string_view interpolation=\"linear\", Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(5)) {
        // aten::quantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation='linear') -> Tensor
        
        auto dispatch_quantile = [](const at::Tensor & self, const at::Tensor & q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.quantile(q, dim, keepdim, interpolation);
        };
        return wrap(dispatch_quantile(_r.tensor(0), _r.tensor(1), _r.toInt64Optional(2), _r.toBool(3), _r.stringView(4)));
      } else {
        // aten::quantile.out(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation='linear', Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_quantile_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::quantile_out(out, self, q, dim, keepdim, interpolation);
        };
        return wrap(dispatch_quantile_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.toInt64Optional(2), _r.toBool(3), _r.stringView(4)));
      }
    }
    case 1: {
      if (_r.isNone(5)) {
        // aten::quantile.scalar(Tensor self, float q, int? dim=None, bool keepdim=False, *, str interpolation='linear') -> Tensor
        
        auto dispatch_quantile = [](const at::Tensor & self, double q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.quantile(q, dim, keepdim, interpolation);
        };
        return wrap(dispatch_quantile(_r.tensor(0), _r.toDouble(1), _r.toInt64Optional(2), _r.toBool(3), _r.stringView(4)));
      } else {
        // aten::quantile.scalar_out(Tensor self, float q, int? dim=None, bool keepdim=False, *, str interpolation='linear', Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_quantile_out = [](at::Tensor out, const at::Tensor & self, double q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::quantile_out(out, self, q, dim, keepdim, interpolation);
        };
        return wrap(dispatch_quantile_out(_r.tensor(5), _r.tensor(0), _r.toDouble(1), _r.toInt64Optional(2), _r.toBool(3), _r.stringView(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// argsort
static PyObject * THPVariable_argsort(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "argsort(Tensor input, *, bool stable, int64_t dim=-1, bool descending=False, Tensor out=None)",
    "argsort(Tensor input, int64_t dim=-1, bool descending=False)",
    "argsort(Tensor input, Dimname dim, bool descending=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // aten::argsort.stable(Tensor self, *, bool stable, int dim=-1, bool descending=False) -> Tensor
        
        auto dispatch_argsort = [](const at::Tensor & self, bool stable, int64_t dim, bool descending) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.argsort(stable, dim, descending);
        };
        return wrap(dispatch_argsort(_r.tensor(0), _r.toBool(1), _r.toInt64(2), _r.toBool(3)));
      } else {
        // aten::argsort.stable_out(Tensor self, *, bool stable, int dim=-1, bool descending=False, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_argsort_out = [](at::Tensor out, const at::Tensor & self, bool stable, int64_t dim, bool descending) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::argsort_out(out, self, stable, dim, descending);
        };
        return wrap(dispatch_argsort_out(_r.tensor(4), _r.tensor(0), _r.toBool(1), _r.toInt64(2), _r.toBool(3)));
      }
    }
    case 1: {
      // aten::argsort(Tensor self, int dim=-1, bool descending=False) -> Tensor
      
      auto dispatch_argsort = [](const at::Tensor & self, int64_t dim, bool descending) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.argsort(dim, descending);
      };
      return wrap(dispatch_argsort(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
    }
    case 2: {
      // aten::argsort.dimname(Tensor self, Dimname dim, bool descending=False) -> Tensor
      
      auto dispatch_argsort = [](const at::Tensor & self, at::Dimname dim, bool descending) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.argsort(dim, descending);
      };
      return wrap(dispatch_argsort(_r.tensor(0), _r.dimname(1), _r.toBool(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// topk
static PyObject * THPVariable_topk(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_topk_out_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_topk_structseq();
  static PythonArgParser parser({
    "topk(Tensor input, SymInt k, int64_t dim=-1, bool largest=True, bool sorted=True, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(5)) {
    // aten::topk(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
    
    auto dispatch_topk = [](const at::Tensor & self, c10::SymInt k, int64_t dim, bool largest, bool sorted) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return self.topk_symint(k, dim, largest, sorted);
    };
    return wrap(NamedTuple1, dispatch_topk(_r.tensor(0), _r.toSymInt(1), _r.toInt64(2), _r.toBool(3), _r.toBool(4)));
  } else {
    // aten::topk.values(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
    auto out = _r.tensorlist_n<2>(5);
    auto dispatch_topk_out = [](at::Tensor & values, at::Tensor & indices, const at::Tensor & self, c10::SymInt k, int64_t dim, bool largest, bool sorted) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::topk_symint_out(values, indices, self, k, dim, largest, sorted);
    };
    return wrap(NamedTuple, dispatch_topk_out(out[0], out[1], _r.tensor(0), _r.toSymInt(1), _r.toInt64(2), _r.toBool(3), _r.toBool(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// equal
static PyObject * THPVariable_equal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "equal(Tensor input, Tensor other)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::equal(Tensor self, Tensor other) -> bool
  
  auto dispatch_equal = [](const at::Tensor & self, const at::Tensor & other) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.equal(other);
  };
  return wrap(dispatch_equal(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// normal
static PyObject * THPVariable_normal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "normal(Tensor mean, Tensor std, *, Generator? generator=None, Tensor out=None)",
    "normal(Tensor mean, double std=1, *, Generator? generator=None, Tensor out=None)",
    "normal(double mean, Tensor std, *, Generator? generator=None, Tensor out=None)",
    "normal(double mean, double std, SymIntArrayRef size, *, Generator? generator=None, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<10> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor
        
        auto dispatch_normal = [](const at::Tensor & mean, const at::Tensor & std, ::std::optional<at::Generator> generator) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::normal(mean, std, generator);
        };
        return wrap(dispatch_normal(_r.tensor(0), _r.tensor(1), _r.generator(2)));
      } else {
        // aten::normal.Tensor_Tensor_out(Tensor mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_normal_out = [](at::Tensor out, const at::Tensor & mean, const at::Tensor & std, ::std::optional<at::Generator> generator) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::normal_out(out, mean, std, generator);
        };
        return wrap(dispatch_normal_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.generator(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::normal.Tensor_float(Tensor mean, float std=1, *, Generator? generator=None) -> Tensor
        
        auto dispatch_normal = [](const at::Tensor & mean, double std, ::std::optional<at::Generator> generator) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::normal(mean, std, generator);
        };
        return wrap(dispatch_normal(_r.tensor(0), _r.toDouble(1), _r.generator(2)));
      } else {
        // aten::normal.Tensor_float_out(Tensor mean, float std=1, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_normal_out = [](at::Tensor out, const at::Tensor & mean, double std, ::std::optional<at::Generator> generator) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::normal_out(out, mean, std, generator);
        };
        return wrap(dispatch_normal_out(_r.tensor(3), _r.tensor(0), _r.toDouble(1), _r.generator(2)));
      }
    }
    case 2: {
      if (_r.isNone(3)) {
        // aten::normal.float_Tensor(float mean, Tensor std, *, Generator? generator=None) -> Tensor
        
        auto dispatch_normal = [](double mean, const at::Tensor & std, ::std::optional<at::Generator> generator) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::normal(mean, std, generator);
        };
        return wrap(dispatch_normal(_r.toDouble(0), _r.tensor(1), _r.generator(2)));
      } else {
        // aten::normal.float_Tensor_out(float mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_normal_out = [](at::Tensor out, double mean, const at::Tensor & std, ::std::optional<at::Generator> generator) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::normal_out(out, mean, std, generator);
        };
        return wrap(dispatch_normal_out(_r.tensor(3), _r.toDouble(0), _r.tensor(1), _r.generator(2)));
      }
    }
    case 3: {
      if (_r.isNone(4)) {
        // aten::normal.float_float(float mean, float std, SymInt[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(5))
            .device(_r.deviceWithDefault(7, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(6))
            .requires_grad(_r.toBool(9))
            .pinned_memory(_r.toBool(8));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_normal = [](double mean, double std, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::normal_symint(mean, std, size, generator, options);
        };
        return wrap(dispatch_normal(_r.toDouble(0), _r.toDouble(1), _r.symintlist(2), _r.generator(3), options));
      } else {
        // aten::normal.float_float_out(float mean, float std, SymInt[] size, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(4), _r.scalartypeOptional(5),
                               _r.isNone(5), _r.layoutOptional(6),
                               _r.deviceWithDefault(7, torch::tensors::get_default_device()), _r.isNone(7));
        
        auto dispatch_normal_out = [](at::Tensor out, double mean, double std, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::normal_symint_out(out, mean, std, size, generator);
        };
        return wrap(dispatch_normal_out(_r.tensor(4), _r.toDouble(0), _r.toDouble(1), _r.symintlist(2), _r.generator(3)).set_requires_grad(_r.toBool(9)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_div
static PyObject * THPVariable__foreach_div(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_div(TensorList self, ScalarList scalars)",
    "_foreach_div(TensorList self, Tensor other)",
    "_foreach_div(TensorList self, TensorList other)",
    "_foreach_div(TensorList self, Scalar scalar)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_div.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
      
      auto dispatch__foreach_div = [](at::TensorList self, at::ArrayRef<at::Scalar> scalars) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_div(self, scalars);
      };
      return wrap(dispatch__foreach_div(_r.tensorlist(0), _r.scalarlist(1)));
    }
    case 1: {
      // aten::_foreach_div.Tensor(Tensor[] self, Tensor other) -> Tensor[]
      
      auto dispatch__foreach_div = [](at::TensorList self, const at::Tensor & other) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_div(self, other);
      };
      return wrap(dispatch__foreach_div(_r.tensorlist(0), _r.tensor(1)));
    }
    case 2: {
      // aten::_foreach_div.List(Tensor[] self, Tensor[] other) -> Tensor[]
      
      auto dispatch__foreach_div = [](at::TensorList self, at::TensorList other) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_div(self, other);
      };
      return wrap(dispatch__foreach_div(_r.tensorlist(0), _r.tensorlist(1)));
    }
    case 3: {
      // aten::_foreach_div.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
      
      auto dispatch__foreach_div = [](at::TensorList self, const at::Scalar & scalar) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_div(self, scalar);
      };
      return wrap(dispatch__foreach_div(_r.tensorlist(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_div_
static PyObject * THPVariable__foreach_div_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_div_(TensorList self, ScalarList scalars)",
    "_foreach_div_(TensorList self, Tensor other)",
    "_foreach_div_(TensorList self, TensorList other)",
    "_foreach_div_(TensorList self, Scalar scalar)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_div_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
      
      auto dispatch__foreach_div_ = [](at::TensorList self, at::ArrayRef<at::Scalar> scalars) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_div_(self, scalars);
      };
      dispatch__foreach_div_(_r.tensorlist(0), _r.scalarlist(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 1: {
      // aten::_foreach_div_.Tensor(Tensor(a!)[] self, Tensor other) -> ()
      
      auto dispatch__foreach_div_ = [](at::TensorList self, const at::Tensor & other) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_div_(self, other);
      };
      dispatch__foreach_div_(_r.tensorlist(0), _r.tensor(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 2: {
      // aten::_foreach_div_.List(Tensor(a!)[] self, Tensor[] other) -> ()
      
      auto dispatch__foreach_div_ = [](at::TensorList self, at::TensorList other) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_div_(self, other);
      };
      dispatch__foreach_div_(_r.tensorlist(0), _r.tensorlist(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 3: {
      // aten::_foreach_div_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
      
      auto dispatch__foreach_div_ = [](at::TensorList self, const at::Scalar & scalar) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_div_(self, scalar);
      };
      dispatch__foreach_div_(_r.tensorlist(0), _r.scalar(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_clamp_min
static PyObject * THPVariable__foreach_clamp_min(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_clamp_min(TensorList self, Scalar scalar)",
    "_foreach_clamp_min(TensorList self, ScalarList scalars)",
    "_foreach_clamp_min(TensorList self, TensorList other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_clamp_min.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
      
      auto dispatch__foreach_clamp_min = [](at::TensorList self, const at::Scalar & scalar) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_clamp_min(self, scalar);
      };
      return wrap(dispatch__foreach_clamp_min(_r.tensorlist(0), _r.scalar(1)));
    }
    case 1: {
      // aten::_foreach_clamp_min.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
      
      auto dispatch__foreach_clamp_min = [](at::TensorList self, at::ArrayRef<at::Scalar> scalars) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_clamp_min(self, scalars);
      };
      return wrap(dispatch__foreach_clamp_min(_r.tensorlist(0), _r.scalarlist(1)));
    }
    case 2: {
      // aten::_foreach_clamp_min.List(Tensor[] self, Tensor[] other) -> Tensor[]
      
      auto dispatch__foreach_clamp_min = [](at::TensorList self, at::TensorList other) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_clamp_min(self, other);
      };
      return wrap(dispatch__foreach_clamp_min(_r.tensorlist(0), _r.tensorlist(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_clamp_min_
static PyObject * THPVariable__foreach_clamp_min_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_clamp_min_(TensorList self, Scalar scalar)",
    "_foreach_clamp_min_(TensorList self, ScalarList scalars)",
    "_foreach_clamp_min_(TensorList self, TensorList other)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_clamp_min_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
      
      auto dispatch__foreach_clamp_min_ = [](at::TensorList self, const at::Scalar & scalar) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_clamp_min_(self, scalar);
      };
      dispatch__foreach_clamp_min_(_r.tensorlist(0), _r.scalar(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 1: {
      // aten::_foreach_clamp_min_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
      
      auto dispatch__foreach_clamp_min_ = [](at::TensorList self, at::ArrayRef<at::Scalar> scalars) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_clamp_min_(self, scalars);
      };
      dispatch__foreach_clamp_min_(_r.tensorlist(0), _r.scalarlist(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 2: {
      // aten::_foreach_clamp_min_.List(Tensor(a!)[] self, Tensor[] other) -> ()
      
      auto dispatch__foreach_clamp_min_ = [](at::TensorList self, at::TensorList other) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_clamp_min_(self, other);
      };
      dispatch__foreach_clamp_min_(_r.tensorlist(0), _r.tensorlist(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_addcmul
static PyObject * THPVariable__foreach_addcmul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_addcmul(TensorList self, TensorList tensor1, TensorList tensor2, ScalarList scalars)",
    "_foreach_addcmul(TensorList self, TensorList tensor1, TensorList tensor2, Tensor scalars)",
    "_foreach_addcmul(TensorList self, TensorList tensor1, TensorList tensor2, Scalar value=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_addcmul.ScalarList(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> Tensor[]
      
      auto dispatch__foreach_addcmul = [](at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_addcmul(self, tensor1, tensor2, scalars);
      };
      return wrap(dispatch__foreach_addcmul(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.scalarlist(3)));
    }
    case 1: {
      // aten::_foreach_addcmul.Tensor(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars) -> Tensor[]
      
      auto dispatch__foreach_addcmul = [](at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_addcmul(self, tensor1, tensor2, scalars);
      };
      return wrap(dispatch__foreach_addcmul(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.tensor(3)));
    }
    case 2: {
      // aten::_foreach_addcmul.Scalar(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> Tensor[]
      
      auto dispatch__foreach_addcmul = [](at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_addcmul(self, tensor1, tensor2, value);
      };
      return wrap(dispatch__foreach_addcmul(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_addcmul_
static PyObject * THPVariable__foreach_addcmul_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_addcmul_(TensorList self, TensorList tensor1, TensorList tensor2, ScalarList scalars)",
    "_foreach_addcmul_(TensorList self, TensorList tensor1, TensorList tensor2, Tensor scalars)",
    "_foreach_addcmul_(TensorList self, TensorList tensor1, TensorList tensor2, Scalar value=1)",
  }, /*traceable=*/false);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_addcmul_.ScalarList(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> ()
      
      auto dispatch__foreach_addcmul_ = [](at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_addcmul_(self, tensor1, tensor2, scalars);
      };
      dispatch__foreach_addcmul_(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.scalarlist(3));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 1: {
      // aten::_foreach_addcmul_.Tensor(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars) -> ()
      
      auto dispatch__foreach_addcmul_ = [](at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_addcmul_(self, tensor1, tensor2, scalars);
      };
      dispatch__foreach_addcmul_(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.tensor(3));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 2: {
      // aten::_foreach_addcmul_.Scalar(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> ()
      
      auto dispatch__foreach_addcmul_ = [](at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_addcmul_(self, tensor1, tensor2, value);
      };
      dispatch__foreach_addcmul_(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.scalar(3));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_atan
static PyObject * THPVariable__foreach_atan(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_atan(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_atan(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_atan = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_atan(self);
  };
  return wrap(dispatch__foreach_atan(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_atan_
static PyObject * THPVariable__foreach_atan_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_atan_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_atan_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_atan_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_atan_(self);
  };
  dispatch__foreach_atan_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_ceil
static PyObject * THPVariable__foreach_ceil(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_ceil(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_ceil(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_ceil = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_ceil(self);
  };
  return wrap(dispatch__foreach_ceil(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_ceil_
static PyObject * THPVariable__foreach_ceil_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_ceil_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_ceil_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_ceil_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_ceil_(self);
  };
  dispatch__foreach_ceil_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_erfc
static PyObject * THPVariable__foreach_erfc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_erfc(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_erfc(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_erfc = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_erfc(self);
  };
  return wrap(dispatch__foreach_erfc(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_erfc_
static PyObject * THPVariable__foreach_erfc_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_erfc_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_erfc_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_erfc_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_erfc_(self);
  };
  dispatch__foreach_erfc_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_expm1
static PyObject * THPVariable__foreach_expm1(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_expm1(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_expm1(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_expm1 = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_expm1(self);
  };
  return wrap(dispatch__foreach_expm1(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_expm1_
static PyObject * THPVariable__foreach_expm1_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_expm1_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_expm1_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_expm1_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_expm1_(self);
  };
  dispatch__foreach_expm1_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_lerp
static PyObject * THPVariable__foreach_lerp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_lerp(TensorList self, TensorList tensors1, Scalar weight)",
    "_foreach_lerp(TensorList self, TensorList tensors1, ScalarList weight)",
    "_foreach_lerp(TensorList self, TensorList tensors1, TensorList weights)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_lerp.Scalar(Tensor[] self, Tensor[] tensors1, Scalar weight) -> Tensor[]
      
      auto dispatch__foreach_lerp = [](at::TensorList self, at::TensorList tensors1, const at::Scalar & weight) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_lerp(self, tensors1, weight);
      };
      return wrap(dispatch__foreach_lerp(_r.tensorlist(0), _r.tensorlist(1), _r.scalar(2)));
    }
    case 1: {
      // aten::_foreach_lerp.ScalarList(Tensor[] self, Tensor[] tensors1, Scalar[] weight) -> Tensor[]
      
      auto dispatch__foreach_lerp = [](at::TensorList self, at::TensorList tensors1, at::ArrayRef<at::Scalar> weight) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_lerp(self, tensors1, weight);
      };
      return wrap(dispatch__foreach_lerp(_r.tensorlist(0), _r.tensorlist(1), _r.scalarlist(2)));
    }
    case 2: {
      // aten::_foreach_lerp.List(Tensor[] self, Tensor[] tensors1, Tensor[] weights) -> Tensor[]
      
      auto dispatch__foreach_lerp = [](at::TensorList self, at::TensorList tensors1, at::TensorList weights) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_lerp(self, tensors1, weights);
      };
      return wrap(dispatch__foreach_lerp(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_lerp_
static PyObject * THPVariable__foreach_lerp_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_lerp_(TensorList self, TensorList tensors1, Scalar weight)",
    "_foreach_lerp_(TensorList self, TensorList tensors1, ScalarList weight)",
    "_foreach_lerp_(TensorList self, TensorList tensors1, TensorList weights)",
  }, /*traceable=*/false);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_lerp_.Scalar(Tensor(a!)[] self, Tensor[] tensors1, Scalar weight) -> ()
      
      auto dispatch__foreach_lerp_ = [](at::TensorList self, at::TensorList tensors1, const at::Scalar & weight) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_lerp_(self, tensors1, weight);
      };
      dispatch__foreach_lerp_(_r.tensorlist(0), _r.tensorlist(1), _r.scalar(2));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 1: {
      // aten::_foreach_lerp_.ScalarList(Tensor(a!)[] self, Tensor[] tensors1, Scalar[] weight) -> ()
      
      auto dispatch__foreach_lerp_ = [](at::TensorList self, at::TensorList tensors1, at::ArrayRef<at::Scalar> weight) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_lerp_(self, tensors1, weight);
      };
      dispatch__foreach_lerp_(_r.tensorlist(0), _r.tensorlist(1), _r.scalarlist(2));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 2: {
      // aten::_foreach_lerp_.List(Tensor(a!)[] self, Tensor[] tensors1, Tensor[] weights) -> ()
      
      auto dispatch__foreach_lerp_ = [](at::TensorList self, at::TensorList tensors1, at::TensorList weights) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_lerp_(self, tensors1, weights);
      };
      dispatch__foreach_lerp_(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_log
static PyObject * THPVariable__foreach_log(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_log(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_log(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_log = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_log(self);
  };
  return wrap(dispatch__foreach_log(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_log_
static PyObject * THPVariable__foreach_log_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_log_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_log_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_log_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_log_(self);
  };
  dispatch__foreach_log_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_log10
static PyObject * THPVariable__foreach_log10(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_log10(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_log10(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_log10 = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_log10(self);
  };
  return wrap(dispatch__foreach_log10(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_log10_
static PyObject * THPVariable__foreach_log10_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_log10_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_log10_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_log10_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_log10_(self);
  };
  dispatch__foreach_log10_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_max
static PyObject * THPVariable__foreach_max(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_max(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_max(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_max = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_max(self);
  };
  return wrap(dispatch__foreach_max(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_rsqrt
static PyObject * THPVariable__foreach_rsqrt(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_rsqrt(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_rsqrt(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_rsqrt = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_rsqrt(self);
  };
  return wrap(dispatch__foreach_rsqrt(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_rsqrt_
static PyObject * THPVariable__foreach_rsqrt_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_rsqrt_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_rsqrt_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_rsqrt_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_rsqrt_(self);
  };
  dispatch__foreach_rsqrt_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_sin
static PyObject * THPVariable__foreach_sin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_sin(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_sin(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_sin = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_sin(self);
  };
  return wrap(dispatch__foreach_sin(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_sin_
static PyObject * THPVariable__foreach_sin_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_sin_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_sin_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_sin_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_sin_(self);
  };
  dispatch__foreach_sin_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_trunc
static PyObject * THPVariable__foreach_trunc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_trunc(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_trunc(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_trunc = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_trunc(self);
  };
  return wrap(dispatch__foreach_trunc(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_trunc_
static PyObject * THPVariable__foreach_trunc_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_trunc_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_trunc_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_trunc_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_trunc_(self);
  };
  dispatch__foreach_trunc_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_zero_
static PyObject * THPVariable__foreach_zero_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_zero_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_zero_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_zero_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_zero_(self);
  };
  dispatch__foreach_zero_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _adaptive_avg_pool2d
static PyObject * THPVariable__adaptive_avg_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_adaptive_avg_pool2d(Tensor input, SymIntArrayRef[2] output_size)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor
  
  auto dispatch__adaptive_avg_pool2d = [](const at::Tensor & self, c10::SymIntArrayRef output_size) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_adaptive_avg_pool2d_symint(self, output_size);
  };
  return wrap(dispatch__adaptive_avg_pool2d(_r.tensor(0), _r.symintlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _adaptive_avg_pool3d
static PyObject * THPVariable__adaptive_avg_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_adaptive_avg_pool3d(Tensor input, SymIntArrayRef[3] output_size)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_adaptive_avg_pool3d(Tensor self, SymInt[3] output_size) -> Tensor
  
  auto dispatch__adaptive_avg_pool3d = [](const at::Tensor & self, c10::SymIntArrayRef output_size) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_adaptive_avg_pool3d_symint(self, output_size);
  };
  return wrap(dispatch__adaptive_avg_pool3d(_r.tensor(0), _r.symintlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// isposinf
static PyObject * THPVariable_isposinf(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "isposinf(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::isposinf(Tensor self) -> Tensor
    
    auto dispatch_isposinf = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.isposinf();
    };
    return wrap(dispatch_isposinf(_r.tensor(0)));
  } else {
    // aten::isposinf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_isposinf_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::isposinf_out(out, self);
    };
    return wrap(dispatch_isposinf_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _add_batch_dim
static PyObject * THPVariable__add_batch_dim(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_add_batch_dim(Tensor input, int64_t batch_dim, int64_t level)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_add_batch_dim(Tensor self, int batch_dim, int level) -> Tensor
  
  auto dispatch__add_batch_dim = [](const at::Tensor & self, int64_t batch_dim, int64_t level) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_add_batch_dim(self, batch_dim, level);
  };
  return wrap(dispatch__add_batch_dim(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _linalg_det
static PyObject * THPVariable__linalg_det(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get__linalg_det_structseq();
  static PyTypeObject* NamedTuple1 = generated::get__linalg_det_out_structseq();
  static PythonArgParser parser({
    "_linalg_det(Tensor A, *, TensorList[3] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::_linalg_det(Tensor A) -> (Tensor result, Tensor LU, Tensor pivots)
    
    auto dispatch__linalg_det = [](const at::Tensor & A) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::_linalg_det(A);
    };
    return wrap(NamedTuple, dispatch__linalg_det(_r.tensor(0)));
  } else {
    // aten::_linalg_det.result(Tensor A, *, Tensor(a!) result, Tensor(b!) LU, Tensor(c!) pivots) -> (Tensor(a!) result, Tensor(b!) LU, Tensor(c!) pivots)
    auto out = _r.tensorlist_n<3>(1);
    auto dispatch__linalg_det_out = [](at::Tensor & result, at::Tensor & LU, at::Tensor & pivots, const at::Tensor & A) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::_linalg_det_out(result, LU, pivots, A);
    };
    return wrap(NamedTuple1, dispatch__linalg_det_out(out[0], out[1], out[2], _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// det
static PyObject * THPVariable_det(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "det(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::det(Tensor self) -> Tensor
  
  auto dispatch_det = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.det();
  };
  return wrap(dispatch_det(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logdet
static PyObject * THPVariable_logdet(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logdet(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::logdet(Tensor self) -> Tensor
  
  auto dispatch_logdet = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logdet();
  };
  return wrap(dispatch_logdet(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ger
static PyObject * THPVariable_ger(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ger(Tensor input, Tensor vec2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::ger(Tensor self, Tensor vec2) -> Tensor
    
    auto dispatch_ger = [](const at::Tensor & self, const at::Tensor & vec2) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.ger(vec2);
    };
    return wrap(dispatch_ger(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::ger.out(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_ger_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & vec2) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::ger_out(out, self, vec2);
    };
    return wrap(dispatch_ger_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _linalg_svd
static PyObject * THPVariable__linalg_svd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get__linalg_svd_structseq();
  static PyTypeObject* NamedTuple1 = generated::get__linalg_svd_out_structseq();
  static PythonArgParser parser({
    "_linalg_svd(Tensor A, bool full_matrices=False, bool compute_uv=True, *, c10::string_view? driver=None, TensorList[3] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::_linalg_svd(Tensor A, bool full_matrices=False, bool compute_uv=True, *, str? driver=None) -> (Tensor U, Tensor S, Tensor Vh)
    
    auto dispatch__linalg_svd = [](const at::Tensor & A, bool full_matrices, bool compute_uv, ::std::optional<c10::string_view> driver) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::_linalg_svd(A, full_matrices, compute_uv, driver);
    };
    return wrap(NamedTuple, dispatch__linalg_svd(_r.tensor(0), _r.toBool(1), _r.toBool(2), _r.stringViewOptional(3)));
  } else {
    // aten::_linalg_svd.U(Tensor A, bool full_matrices=False, bool compute_uv=True, *, str? driver=None, Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh)
    auto out = _r.tensorlist_n<3>(4);
    auto dispatch__linalg_svd_out = [](at::Tensor & U, at::Tensor & S, at::Tensor & Vh, const at::Tensor & A, bool full_matrices, bool compute_uv, ::std::optional<c10::string_view> driver) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::_linalg_svd_out(U, S, Vh, A, full_matrices, compute_uv, driver);
    };
    return wrap(NamedTuple1, dispatch__linalg_svd_out(out[0], out[1], out[2], _r.tensor(0), _r.toBool(1), _r.toBool(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_tensor_from_tensor_list
static PyObject * THPVariable__nested_tensor_from_tensor_list(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_tensor_from_tensor_list(TensorList list, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_nested_tensor_from_tensor_list(Tensor[] list, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  
  auto dispatch__nested_tensor_from_tensor_list = [](at::TensorList list, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_nested_tensor_from_tensor_list(list, dtype, layout, device, pin_memory);
  };
  return wrap(dispatch__nested_tensor_from_tensor_list(_r.tensorlist(0), _r.scalartypeOptional(1), _r.layoutOptional(2), _r.deviceOptional(3), _r.toBoolOptional(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _fw_primal_copy
static PyObject * THPVariable__fw_primal_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_fw_primal_copy(Tensor input, int64_t level, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::_fw_primal_copy(Tensor self, int level) -> Tensor
    
    auto dispatch__fw_primal_copy = [](const at::Tensor & self, int64_t level) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_fw_primal_copy(self, level);
    };
    return wrap(dispatch__fw_primal_copy(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::_fw_primal_copy.out(Tensor self, int level, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__fw_primal_copy_out = [](at::Tensor out, const at::Tensor & self, int64_t level) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_fw_primal_copy_out(out, self, level);
    };
    return wrap(dispatch__fw_primal_copy_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _make_dual_copy
static PyObject * THPVariable__make_dual_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_make_dual_copy(Tensor primal, Tensor tangent, int64_t level, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::_make_dual_copy(Tensor primal, Tensor tangent, int level) -> Tensor
    
    auto dispatch__make_dual_copy = [](const at::Tensor & primal, const at::Tensor & tangent, int64_t level) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_make_dual_copy(primal, tangent, level);
    };
    return wrap(dispatch__make_dual_copy(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  } else {
    // aten::_make_dual_copy.out(Tensor primal, Tensor tangent, int level, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__make_dual_copy_out = [](at::Tensor out, const at::Tensor & primal, const at::Tensor & tangent, int64_t level) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_make_dual_copy_out(out, primal, tangent, level);
    };
    return wrap(dispatch__make_dual_copy_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// view_as_complex_copy
static PyObject * THPVariable_view_as_complex_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "view_as_complex_copy(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::view_as_complex_copy(Tensor self) -> Tensor
    
    auto dispatch_view_as_complex_copy = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::view_as_complex_copy(self);
    };
    return wrap(dispatch_view_as_complex_copy(_r.tensor(0)));
  } else {
    // aten::view_as_complex_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_view_as_complex_copy_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::view_as_complex_copy_out(out, self);
    };
    return wrap(dispatch_view_as_complex_copy_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _neg_view_copy
static PyObject * THPVariable__neg_view_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_neg_view_copy(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::_neg_view_copy(Tensor self) -> Tensor
    
    auto dispatch__neg_view_copy = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_neg_view_copy(self);
    };
    return wrap(dispatch__neg_view_copy(_r.tensor(0)));
  } else {
    // aten::_neg_view_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__neg_view_copy_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_neg_view_copy_out(out, self);
    };
    return wrap(dispatch__neg_view_copy_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// as_strided_copy
static PyObject * THPVariable_as_strided_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "as_strided_copy(Tensor input, SymIntArrayRef size, SymIntArrayRef stride, SymInt? storage_offset=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::as_strided_copy(Tensor self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor
    
    auto dispatch_as_strided_copy = [](const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<c10::SymInt> storage_offset) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::as_strided_copy_symint(self, size, stride, storage_offset);
    };
    return wrap(dispatch_as_strided_copy(_r.tensor(0), _r.symintlist(1), _r.symintlist(2), _r.toSymIntOptional(3)));
  } else {
    // aten::as_strided_copy.out(Tensor self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_as_strided_copy_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<c10::SymInt> storage_offset) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::as_strided_copy_symint_out(out, self, size, stride, storage_offset);
    };
    return wrap(dispatch_as_strided_copy_out(_r.tensor(4), _r.tensor(0), _r.symintlist(1), _r.symintlist(2), _r.toSymIntOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// diagonal_copy
static PyObject * THPVariable_diagonal_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diagonal_copy(Tensor input, int64_t offset=0, int64_t dim1=0, int64_t dim2=1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::diagonal_copy(Tensor self, int offset=0, int dim1=0, int dim2=1) -> Tensor
    
    auto dispatch_diagonal_copy = [](const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::diagonal_copy(self, offset, dim1, dim2);
    };
    return wrap(dispatch_diagonal_copy(_r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toInt64(3)));
  } else {
    // aten::diagonal_copy.out(Tensor self, int offset=0, int dim1=0, int dim2=1, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_diagonal_copy_out = [](at::Tensor out, const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::diagonal_copy_out(out, self, offset, dim1, dim2);
    };
    return wrap(dispatch_diagonal_copy_out(_r.tensor(4), _r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toInt64(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// expand_copy
static PyObject * THPVariable_expand_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "expand_copy(Tensor input, SymIntArrayRef size, *, bool implicit=False, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::expand_copy(Tensor self, SymInt[] size, *, bool implicit=False) -> Tensor
    
    auto dispatch_expand_copy = [](const at::Tensor & self, c10::SymIntArrayRef size, bool implicit) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::expand_copy_symint(self, size, implicit);
    };
    return wrap(dispatch_expand_copy(_r.tensor(0), _r.symintlist(1), _r.toBool(2)));
  } else {
    // aten::expand_copy.out(Tensor self, SymInt[] size, *, bool implicit=False, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_expand_copy_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef size, bool implicit) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::expand_copy_symint_out(out, self, size, implicit);
    };
    return wrap(dispatch_expand_copy_out(_r.tensor(3), _r.tensor(0), _r.symintlist(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _reshape_alias_copy
static PyObject * THPVariable__reshape_alias_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_reshape_alias_copy(Tensor input, SymIntArrayRef size, SymIntArrayRef stride, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::_reshape_alias_copy(Tensor self, SymInt[] size, SymInt[] stride) -> Tensor
    
    auto dispatch__reshape_alias_copy = [](const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_reshape_alias_copy_symint(self, size, stride);
    };
    return wrap(dispatch__reshape_alias_copy(_r.tensor(0), _r.symintlist(1), _r.symintlist(2)));
  } else {
    // aten::_reshape_alias_copy.out(Tensor self, SymInt[] size, SymInt[] stride, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__reshape_alias_copy_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_reshape_alias_copy_symint_out(out, self, size, stride);
    };
    return wrap(dispatch__reshape_alias_copy_out(_r.tensor(3), _r.tensor(0), _r.symintlist(1), _r.symintlist(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// slice_copy
static PyObject * THPVariable_slice_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "slice_copy(Tensor input, int64_t dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(5)) {
    // aten::slice_copy.Tensor(Tensor self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor
    
    auto dispatch_slice_copy = [](const at::Tensor & self, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::slice_copy_symint(self, dim, start, end, step);
    };
    return wrap(dispatch_slice_copy(_r.tensor(0), _r.toInt64(1), _r.toSymIntOptional(2), _r.toSymIntOptional(3), _r.toSymInt(4)));
  } else {
    // aten::slice_copy.Tensor_out(Tensor self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_slice_copy_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::slice_copy_symint_out(out, self, dim, start, end, step);
    };
    return wrap(dispatch_slice_copy_out(_r.tensor(5), _r.tensor(0), _r.toInt64(1), _r.toSymIntOptional(2), _r.toSymIntOptional(3), _r.toSymInt(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// split_with_sizes_copy
static PyObject * THPVariable_split_with_sizes_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "split_with_sizes_copy(Tensor input, SymIntArrayRef split_sizes, int64_t dim=0, *, TensorList out=None)",
  }, /*traceable=*/false);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::split_with_sizes_copy(Tensor self, SymInt[] split_sizes, int dim=0) -> Tensor[]
    
    auto dispatch_split_with_sizes_copy = [](const at::Tensor & self, c10::SymIntArrayRef split_sizes, int64_t dim) -> ::std::vector<at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::split_with_sizes_copy_symint(self, split_sizes, dim);
    };
    return wrap(dispatch_split_with_sizes_copy(_r.tensor(0), _r.symintlist(1), _r.toInt64(2)));
  } else {
    // aten::split_with_sizes_copy.out(Tensor self, SymInt[] split_sizes, int dim=0, *, Tensor(a!)[] out) -> ()
    
    auto dispatch_split_with_sizes_copy_out = [](at::TensorList out, const at::Tensor & self, c10::SymIntArrayRef split_sizes, int64_t dim) -> void {
      pybind11::gil_scoped_release no_gil;
      at::split_with_sizes_copy_symint_out(out, self, split_sizes, dim);
    };
    dispatch_split_with_sizes_copy_out(_r.tensorlist(3), _r.tensor(0), _r.symintlist(1), _r.toInt64(2));
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// t_copy
static PyObject * THPVariable_t_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "t_copy(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::t_copy(Tensor self) -> Tensor
    
    auto dispatch_t_copy = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::t_copy(self);
    };
    return wrap(dispatch_t_copy(_r.tensor(0)));
  } else {
    // aten::t_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_t_copy_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::t_copy_out(out, self);
    };
    return wrap(dispatch_t_copy_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// indices_copy
static PyObject * THPVariable_indices_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "indices_copy(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::indices_copy(Tensor self) -> Tensor
    
    auto dispatch_indices_copy = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::indices_copy(self);
    };
    return wrap(dispatch_indices_copy(_r.tensor(0)));
  } else {
    // aten::indices_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_indices_copy_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::indices_copy_out(out, self);
    };
    return wrap(dispatch_indices_copy_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// crow_indices_copy
static PyObject * THPVariable_crow_indices_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "crow_indices_copy(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::crow_indices_copy(Tensor self) -> Tensor
    
    auto dispatch_crow_indices_copy = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::crow_indices_copy(self);
    };
    return wrap(dispatch_crow_indices_copy(_r.tensor(0)));
  } else {
    // aten::crow_indices_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_crow_indices_copy_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::crow_indices_copy_out(out, self);
    };
    return wrap(dispatch_crow_indices_copy_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// col_indices_copy
static PyObject * THPVariable_col_indices_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "col_indices_copy(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::col_indices_copy(Tensor self) -> Tensor
    
    auto dispatch_col_indices_copy = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::col_indices_copy(self);
    };
    return wrap(dispatch_col_indices_copy(_r.tensor(0)));
  } else {
    // aten::col_indices_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_col_indices_copy_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::col_indices_copy_out(out, self);
    };
    return wrap(dispatch_col_indices_copy_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// ccol_indices_copy
static PyObject * THPVariable_ccol_indices_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ccol_indices_copy(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::ccol_indices_copy(Tensor self) -> Tensor
    
    auto dispatch_ccol_indices_copy = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::ccol_indices_copy(self);
    };
    return wrap(dispatch_ccol_indices_copy(_r.tensor(0)));
  } else {
    // aten::ccol_indices_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_ccol_indices_copy_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::ccol_indices_copy_out(out, self);
    };
    return wrap(dispatch_ccol_indices_copy_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// row_indices_copy
static PyObject * THPVariable_row_indices_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "row_indices_copy(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::row_indices_copy(Tensor self) -> Tensor
    
    auto dispatch_row_indices_copy = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::row_indices_copy(self);
    };
    return wrap(dispatch_row_indices_copy(_r.tensor(0)));
  } else {
    // aten::row_indices_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_row_indices_copy_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::row_indices_copy_out(out, self);
    };
    return wrap(dispatch_row_indices_copy_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// view_copy
static PyObject * THPVariable_view_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "view_copy(Tensor input, ScalarType dtype, *, Tensor out=None)",
    "view_copy(Tensor input, SymIntArrayRef size, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::view_copy.dtype(Tensor self, ScalarType dtype) -> Tensor
        
        auto dispatch_view_copy = [](const at::Tensor & self, at::ScalarType dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::view_copy(self, dtype);
        };
        return wrap(dispatch_view_copy(_r.tensor(0), _r.scalartype(1)));
      } else {
        // aten::view_copy.dtype_out(Tensor self, ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_view_copy_out = [](at::Tensor out, const at::Tensor & self, at::ScalarType dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::view_copy_out(out, self, dtype);
        };
        return wrap(dispatch_view_copy_out(_r.tensor(2), _r.tensor(0), _r.scalartype(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::view_copy(Tensor self, SymInt[] size) -> Tensor
        
        auto dispatch_view_copy = [](const at::Tensor & self, c10::SymIntArrayRef size) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::view_copy_symint(self, size);
        };
        return wrap(dispatch_view_copy(_r.tensor(0), _r.symintlist(1)));
      } else {
        // aten::view_copy.out(Tensor self, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_view_copy_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef size) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::view_copy_symint_out(out, self, size);
        };
        return wrap(dispatch_view_copy_out(_r.tensor(2), _r.tensor(0), _r.symintlist(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_from_padded_tensor
static PyObject * THPVariable__nested_from_padded_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_from_padded_tensor(Tensor padded, Tensor offsets, Tensor dummy, int64_t ragged_idx=1, Tensor? min_seqlen=None, Tensor? max_seqlen=None, SymInt? sum_S=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_nested_from_padded_tensor(Tensor padded, Tensor offsets, Tensor dummy, int ragged_idx=1, Tensor? min_seqlen=None, Tensor? max_seqlen=None, SymInt? sum_S=None) -> Tensor
  
  auto dispatch__nested_from_padded_tensor = [](const at::Tensor & padded, const at::Tensor & offsets, const at::Tensor & dummy, int64_t ragged_idx, const ::std::optional<at::Tensor> & min_seqlen, const ::std::optional<at::Tensor> & max_seqlen, ::std::optional<c10::SymInt> sum_S) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_nested_from_padded_tensor_symint(padded, offsets, dummy, ragged_idx, min_seqlen, max_seqlen, sum_S);
  };
  return wrap(dispatch__nested_from_padded_tensor(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3), _r.optionalTensor(4), _r.optionalTensor(5), _r.toSymIntOptional(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _transformer_encoder_layer_fwd
static PyObject * THPVariable__transformer_encoder_layer_fwd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_transformer_encoder_layer_fwd(Tensor src, int64_t embed_dim, int64_t num_heads, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, Tensor proj_bias, bool use_gelu, bool norm_first, double eps, Tensor norm_weight_1, Tensor norm_bias_1, Tensor norm_weight_2, Tensor norm_bias_2, Tensor ffn_weight_1, Tensor ffn_bias_1, Tensor ffn_weight_2, Tensor ffn_bias_2, Tensor? mask=None, int64_t? mask_type=None)",
  }, /*traceable=*/true);

  ParsedArgs<20> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_transformer_encoder_layer_fwd(Tensor src, int embed_dim, int num_heads, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, Tensor proj_bias, bool use_gelu, bool norm_first, float eps, Tensor norm_weight_1, Tensor norm_bias_1, Tensor norm_weight_2, Tensor norm_bias_2, Tensor ffn_weight_1, Tensor ffn_bias_1, Tensor ffn_weight_2, Tensor ffn_bias_2, Tensor? mask=None, int? mask_type=None) -> Tensor
  
  auto dispatch__transformer_encoder_layer_fwd = [](const at::Tensor & src, int64_t embed_dim, int64_t num_heads, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, bool use_gelu, bool norm_first, double eps, const at::Tensor & norm_weight_1, const at::Tensor & norm_bias_1, const at::Tensor & norm_weight_2, const at::Tensor & norm_bias_2, const at::Tensor & ffn_weight_1, const at::Tensor & ffn_bias_1, const at::Tensor & ffn_weight_2, const at::Tensor & ffn_bias_2, const ::std::optional<at::Tensor> & mask, ::std::optional<int64_t> mask_type) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_transformer_encoder_layer_fwd(src, embed_dim, num_heads, qkv_weight, qkv_bias, proj_weight, proj_bias, use_gelu, norm_first, eps, norm_weight_1, norm_bias_1, norm_weight_2, norm_bias_2, ffn_weight_1, ffn_bias_1, ffn_weight_2, ffn_bias_2, mask, mask_type);
  };
  return wrap(dispatch__transformer_encoder_layer_fwd(_r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.tensor(3), _r.tensor(4), _r.tensor(5), _r.tensor(6), _r.toBool(7), _r.toBool(8), _r.toDouble(9), _r.tensor(10), _r.tensor(11), _r.tensor(12), _r.tensor(13), _r.tensor(14), _r.tensor(15), _r.tensor(16), _r.tensor(17), _r.optionalTensor(18), _r.toInt64Optional(19)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _scaled_dot_product_attention_math
static PyObject * THPVariable__scaled_dot_product_attention_math(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_scaled_dot_product_attention_math(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, double dropout_p=0.0, bool is_causal=False, Tensor? dropout_mask=None, *, double? scale=None, bool enable_gqa=False)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_scaled_dot_product_attention_math(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, Tensor? dropout_mask=None, *, float? scale=None, bool enable_gqa=False) -> (Tensor, Tensor)
  
  auto dispatch__scaled_dot_product_attention_math = [](const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & attn_mask, double dropout_p, bool is_causal, const ::std::optional<at::Tensor> & dropout_mask, ::std::optional<double> scale, bool enable_gqa) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_scaled_dot_product_attention_math(query, key, value, attn_mask, dropout_p, is_causal, dropout_mask, scale, enable_gqa);
  };
  return wrap(dispatch__scaled_dot_product_attention_math(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.optionalTensor(3), _r.toDouble(4), _r.toBool(5), _r.optionalTensor(6), _r.toDoubleOptional(7), _r.toBool(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _triton_multi_head_attention
static PyObject * THPVariable__triton_multi_head_attention(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_triton_multi_head_attention(Tensor query, Tensor key, Tensor value, int64_t embed_dim, int64_t num_head, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, Tensor proj_bias, Tensor? mask=None)",
  }, /*traceable=*/true);

  ParsedArgs<10> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_triton_multi_head_attention(Tensor query, Tensor key, Tensor value, int embed_dim, int num_head, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, Tensor proj_bias, Tensor? mask=None) -> Tensor
  
  auto dispatch__triton_multi_head_attention = [](const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t embed_dim, int64_t num_head, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, const ::std::optional<at::Tensor> & mask) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_triton_multi_head_attention(query, key, value, embed_dim, num_head, qkv_weight, qkv_bias, proj_weight, proj_bias, mask);
  };
  return wrap(dispatch__triton_multi_head_attention(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3), _r.toInt64(4), _r.tensor(5), _r.tensor(6), _r.tensor(7), _r.tensor(8), _r.optionalTensor(9)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _fused_adam_
static PyObject * THPVariable__fused_adam_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_fused_adam_(TensorList self, TensorList grads, TensorList exp_avgs, TensorList exp_avg_sqs, TensorList max_exp_avg_sqs, TensorList state_steps, *, Tensor lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None)",
    "_fused_adam_(TensorList self, TensorList grads, TensorList exp_avgs, TensorList exp_avg_sqs, TensorList max_exp_avg_sqs, TensorList state_steps, *, double lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None)",
  }, /*traceable=*/false);

  ParsedArgs<15> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_fused_adam_.tensor_lr(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, Tensor lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
      
      auto dispatch__fused_adam_ = [](at::TensorList self, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps, const at::Tensor & lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_fused_adam_(self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
      };
      dispatch__fused_adam_(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.tensorlist(3), _r.tensorlist(4), _r.tensorlist(5), _r.tensor(6), _r.toDouble(7), _r.toDouble(8), _r.toDouble(9), _r.toDouble(10), _r.toBool(11), _r.toBool(12), _r.optionalTensor(13), _r.optionalTensor(14));
      Py_RETURN_NONE;
    }
    case 1: {
      // aten::_fused_adam_(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
      
      auto dispatch__fused_adam_ = [](at::TensorList self, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps, double lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_fused_adam_(self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
      };
      dispatch__fused_adam_(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.tensorlist(3), _r.tensorlist(4), _r.tensorlist(5), _r.toDouble(6), _r.toDouble(7), _r.toDouble(8), _r.toDouble(9), _r.toDouble(10), _r.toBool(11), _r.toBool(12), _r.optionalTensor(13), _r.optionalTensor(14));
      Py_RETURN_NONE;
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _fused_sgd_
static PyObject * THPVariable__fused_sgd_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_fused_sgd_(TensorList self, TensorList grads, TensorList momentum_buffer_list, *, double weight_decay, double momentum, Tensor lr, double dampening, bool nesterov, bool maximize, bool is_first_step, Tensor? grad_scale=None, Tensor? found_inf=None)",
    "_fused_sgd_(TensorList self, TensorList grads, TensorList momentum_buffer_list, *, double weight_decay, double momentum, double lr, double dampening, bool nesterov, bool maximize, bool is_first_step, Tensor? grad_scale=None, Tensor? found_inf=None)",
  }, /*traceable=*/false);

  ParsedArgs<12> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_fused_sgd_.tensor_lr(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] momentum_buffer_list, *, float weight_decay, float momentum, Tensor lr, float dampening, bool nesterov, bool maximize, bool is_first_step, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
      
      auto dispatch__fused_sgd_ = [](at::TensorList self, at::TensorList grads, at::TensorList momentum_buffer_list, double weight_decay, double momentum, const at::Tensor & lr, double dampening, bool nesterov, bool maximize, bool is_first_step, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_fused_sgd_(self, grads, momentum_buffer_list, weight_decay, momentum, lr, dampening, nesterov, maximize, is_first_step, grad_scale, found_inf);
      };
      dispatch__fused_sgd_(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.toDouble(3), _r.toDouble(4), _r.tensor(5), _r.toDouble(6), _r.toBool(7), _r.toBool(8), _r.toBool(9), _r.optionalTensor(10), _r.optionalTensor(11));
      Py_RETURN_NONE;
    }
    case 1: {
      // aten::_fused_sgd_(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] momentum_buffer_list, *, float weight_decay, float momentum, float lr, float dampening, bool nesterov, bool maximize, bool is_first_step, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
      
      auto dispatch__fused_sgd_ = [](at::TensorList self, at::TensorList grads, at::TensorList momentum_buffer_list, double weight_decay, double momentum, double lr, double dampening, bool nesterov, bool maximize, bool is_first_step, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_fused_sgd_(self, grads, momentum_buffer_list, weight_decay, momentum, lr, dampening, nesterov, maximize, is_first_step, grad_scale, found_inf);
      };
      dispatch__fused_sgd_(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.toDouble(3), _r.toDouble(4), _r.toDouble(5), _r.toDouble(6), _r.toBool(7), _r.toBool(8), _r.toBool(9), _r.optionalTensor(10), _r.optionalTensor(11));
      Py_RETURN_NONE;
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

} // namespace torch::autograd
