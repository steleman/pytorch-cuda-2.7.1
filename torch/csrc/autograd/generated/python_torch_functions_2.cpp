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
#include <ATen/ops/_cast_Double.h>
#include <ATen/ops/_cast_Int.h>
#include <ATen/ops/align_tensors.h>
#include <ATen/ops/_print.h>
#include <ATen/ops/_make_dep_token.h>
#include <ATen/ops/_use_cudnn_rnn_flatten_weight.h>
#include <ATen/ops/_cudnn_rnn.h>
#include <ATen/ops/_fused_dropout.h>
#include <ATen/ops/_sobol_engine_ff.h>
#include <ATen/ops/_sobol_engine_scramble.h>
#include <ATen/ops/_shape_as_tensor.h>
#include <ATen/ops/dropout.h>
#include <ATen/ops/dropout.h>
#include <ATen/ops/_conj.h>
#include <ATen/ops/conj_physical.h>
#include <ATen/ops/conj_physical.h>
#include <ATen/ops/avg_pool1d.h>
#include <ATen/ops/add.h>
#include <ATen/ops/_add_relu.h>
#include <ATen/ops/_add_relu.h>
#include <ATen/ops/addr.h>
#include <ATen/ops/_test_functorch_fallback.h>
#include <ATen/ops/all.h>
#include <ATen/ops/allclose.h>
#include <ATen/ops/argmax.h>
#include <ATen/ops/acosh.h>
#include <ATen/ops/acosh.h>
#include <ATen/ops/arccosh.h>
#include <ATen/ops/arccosh.h>
#include <ATen/ops/arctanh.h>
#include <ATen/ops/arctanh.h>
#include <ATen/ops/arcsin.h>
#include <ATen/ops/arcsin.h>
#include <ATen/ops/atan.h>
#include <ATen/ops/atan.h>
#include <ATen/ops/baddbmm.h>
#include <ATen/ops/bartlett_window.h>
#include <ATen/ops/batch_norm.h>
#include <ATen/ops/copysign.h>
#include <ATen/ops/_lazy_clone.h>
#include <ATen/ops/logical_not.h>
#include <ATen/ops/logical_xor.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/broadcast_tensors.h>
#include <ATen/ops/broadcast_to.h>
#include <ATen/ops/_sparse_broadcast_to.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/chain_matmul.h>
#include <ATen/ops/chunk.h>
#include <ATen/ops/tensor_split.h>
#include <ATen/ops/clamp_min.h>
#include <ATen/ops/clamp_min.h>
#include <ATen/ops/clip.h>
#include <ATen/ops/clip.h>
#include <ATen/ops/complex.h>
#include <ATen/ops/polar.h>
#include <ATen/ops/_convolution.h>
#include <ATen/ops/_convolution_mode.h>
#include <ATen/ops/conv1d.h>
#include <ATen/ops/conv3d.h>
#include <ATen/ops/cos.h>
#include <ATen/ops/cos.h>
#include <ATen/ops/count_nonzero.h>
#include <ATen/ops/corrcoef.h>
#include <ATen/ops/cudnn_convolution_transpose.h>
#include <ATen/ops/cudnn_convolution_relu.h>
#include <ATen/ops/ctc_loss.h>
#include <ATen/ops/embedding_bag.h>
#include <ATen/ops/_embedding_bag.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/erf.h>
#include <ATen/ops/erf.h>
#include <ATen/ops/erfc.h>
#include <ATen/ops/erfc.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/unflatten.h>
#include <ATen/ops/floor.h>
#include <ATen/ops/floor.h>
#include <ATen/ops/floor_divide.h>
#include <ATen/ops/frac.h>
#include <ATen/ops/frac.h>
#include <ATen/ops/full_like.h>
#include <ATen/ops/grid_sampler.h>
#include <ATen/ops/_grid_sampler_2d_cpu_fallback.h>
#include <ATen/ops/grid_sampler_3d.h>
#include <ATen/ops/_fft_c2r.h>
#include <ATen/ops/_cufft_get_plan_cache_max_size.h>
#include <ATen/ops/_cufft_clear_plan_cache.h>
#include <ATen/ops/_unsafe_masked_index_put_accumulate.h>
#include <ATen/ops/_index_put_impl.h>
#include <ATen/ops/isin.h>
#include <ATen/ops/isnan.h>
#include <ATen/ops/is_conj.h>
#include <ATen/ops/is_inference.h>
#include <ATen/ops/layer_norm.h>
#include <ATen/ops/nan_to_num.h>
#include <ATen/ops/nan_to_num.h>
#include <ATen/ops/mkldnn_linear_backward_weights.h>
#include <ATen/ops/_cslt_sparse_mm.h>
#include <ATen/ops/_sparse_semi_structured_mm.h>
#include <ATen/ops/_sparse_semi_structured_addmm.h>
#include <ATen/ops/_mixed_dtypes_linear.h>
#include <ATen/ops/fbgemm_pack_gemm_matrix_fp16.h>
#include <ATen/ops/_wrapped_quantized_linear_prepacked.h>
#include <ATen/ops/fbgemm_linear_fp16_weight_fp32_activation.h>
#include <ATen/ops/linspace.h>
#include <ATen/ops/log.h>
#include <ATen/ops/log.h>
#include <ATen/ops/log1p.h>
#include <ATen/ops/log1p.h>
#include <ATen/ops/logaddexp.h>
#include <ATen/ops/xlogy.h>
#include <ATen/ops/xlogy.h>
#include <ATen/ops/logsumexp.h>
#include <ATen/ops/matrix_power.h>
#include <ATen/ops/_aminmax.h>
#include <ATen/ops/max.h>
#include <ATen/ops/quantized_max_pool1d.h>
#include <ATen/ops/quantized_max_pool2d.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/nanmean.h>
#include <ATen/ops/nanmedian.h>
#include <ATen/ops/min.h>
#include <ATen/ops/amin.h>
#include <ATen/ops/mkldnn_convolution.h>
#include <ATen/ops/miopen_batch_norm.h>
#include <ATen/ops/miopen_depthwise_convolution.h>
#include <ATen/ops/_weight_int8pack_mm.h>
#include <ATen/ops/mv.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/batch_norm_stats.h>
#include <ATen/ops/batch_norm_update_stats.h>
#include <ATen/ops/_nnpack_available.h>
#include <ATen/ops/_nnpack_spatial_convolution.h>
#include <ATen/ops/_euclidean_dist.h>
#include <ATen/ops/permute.h>
#include <ATen/ops/pixel_shuffle.h>
#include <ATen/ops/native_channel_shuffle.h>
#include <ATen/ops/_pin_memory.h>
#include <ATen/ops/rad2deg.h>
#include <ATen/ops/rad2deg.h>
#include <ATen/ops/rand_like.h>
#include <ATen/ops/randint.h>
#include <ATen/ops/randint_like.h>
#include <ATen/ops/randn.h>
#include <ATen/ops/randperm.h>
#include <ATen/ops/reciprocal.h>
#include <ATen/ops/reciprocal.h>
#include <ATen/ops/negative.h>
#include <ATen/ops/negative.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/_mkldnn_reshape.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/prelu.h>
#include <ATen/ops/select.h>
#include <ATen/ops/selu.h>
#include <ATen/ops/selu.h>
#include <ATen/ops/logit.h>
#include <ATen/ops/logit.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/slice_inverse.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/unsafe_split_with_sizes.h>
#include <ATen/ops/split_with_sizes.h>
#include <ATen/ops/vsplit.h>
#include <ATen/ops/_chunk_cat.h>
#include <ATen/ops/hstack.h>
#include <ATen/ops/vstack.h>
#include <ATen/ops/dstack.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/tan.h>
#include <ATen/ops/tan.h>
#include <ATen/ops/trapz.h>
#include <ATen/ops/_nested_tensor_from_mask_left_aligned.h>
#include <ATen/ops/_nested_from_padded_and_nested_example.h>
#include <ATen/ops/_nested_view_from_buffer.h>
#include <ATen/ops/_nested_get_values_copy.h>
#include <ATen/ops/_nested_get_lengths.h>
#include <ATen/ops/trunc.h>
#include <ATen/ops/trunc.h>
#include <ATen/ops/unique_dim.h>
#include <ATen/ops/unique_consecutive.h>
#include <ATen/ops/where.h>
#include <ATen/ops/_standard_gamma.h>
#include <ATen/ops/_dirichlet_grad.h>
#include <ATen/ops/binomial.h>
#include <ATen/ops/_sparse_csr_sum.h>
#include <ATen/ops/_sparse_log_softmax_backward_data.h>
#include <ATen/ops/nuclear_norm.h>
#include <ATen/ops/clone.h>
#include <ATen/ops/heaviside.h>
#include <ATen/ops/_scaled_mm.h>
#include <ATen/ops/_validate_sparse_compressed_tensor_args.h>
#include <ATen/ops/_validate_sparse_csr_tensor_args.h>
#include <ATen/ops/_to_cpu.h>
#include <ATen/ops/hspmm.h>
#include <ATen/ops/_to_sparse_semi_structured.h>
#include <ATen/ops/q_per_channel_scales.h>
#include <ATen/ops/q_per_channel_zero_points.h>
#include <ATen/ops/q_per_channel_axis.h>
#include <ATen/ops/_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.h>
#include <ATen/ops/_fake_quantize_learnable_per_tensor_affine.h>
#include <ATen/ops/fused_moving_avg_obs_fake_quant.h>
#include <ATen/ops/_saturate_weight_to_fp16.h>
#include <ATen/ops/choose_qparams_optimized.h>
#include <ATen/ops/combinations.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/can_cast.h>
#include <ATen/ops/_lstm_mps.h>
#include <ATen/ops/lstm.h>
#include <ATen/ops/lstm_cell.h>
#include <ATen/ops/quantized_rnn_relu_cell.h>
#include <ATen/ops/quantized_rnn_tanh_cell.h>
#include <ATen/ops/index_add.h>
#include <ATen/ops/index_reduce.h>
#include <ATen/ops/scatter_add.h>
#include <ATen/ops/scatter_reduce.h>
#include <ATen/ops/bitwise_and.h>
#include <ATen/ops/bitwise_xor.h>
#include <ATen/ops/bitwise_left_shift.h>
#include <ATen/ops/rshift.h>
#include <ATen/ops/addbmm.h>
#include <ATen/ops/diag.h>
#include <ATen/ops/cross.h>
#include <ATen/ops/triu_indices.h>
#include <ATen/ops/trace.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/eq.h>
#include <ATen/ops/greater.h>
#include <ATen/ops/take.h>
#include <ATen/ops/_linalg_check_errors.h>
#include <ATen/ops/svd.h>
#include <ATen/ops/cholesky.h>
#include <ATen/ops/_lu_with_info.h>
#include <ATen/ops/lu_unpack.h>
#include <ATen/ops/multinomial.h>
#include <ATen/ops/i0.h>
#include <ATen/ops/i0.h>
#include <ATen/ops/atan2.h>
#include <ATen/ops/hypot.h>
#include <ATen/ops/igamma.h>
#include <ATen/ops/nextafter.h>
#include <ATen/ops/fmin.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/renorm.h>
#include <ATen/ops/_amp_foreach_non_finite_check_and_unscale.h>
#include <ATen/ops/_amp_update_scale.h>
#include <ATen/ops/_foreach_mul.h>
#include <ATen/ops/_foreach_mul.h>
#include <ATen/ops/_foreach_maximum.h>
#include <ATen/ops/_foreach_maximum.h>
#include <ATen/ops/_foreach_minimum.h>
#include <ATen/ops/_foreach_minimum.h>
#include <ATen/ops/_foreach_addcdiv.h>
#include <ATen/ops/_foreach_addcdiv.h>
#include <ATen/ops/_foreach_abs.h>
#include <ATen/ops/_foreach_abs.h>
#include <ATen/ops/_foreach_erf.h>
#include <ATen/ops/_foreach_erf.h>
#include <ATen/ops/_foreach_frac.h>
#include <ATen/ops/_foreach_frac.h>
#include <ATen/ops/_foreach_lgamma.h>
#include <ATen/ops/_foreach_lgamma.h>
#include <ATen/ops/_foreach_pow.h>
#include <ATen/ops/_foreach_pow.h>
#include <ATen/ops/_foreach_reciprocal.h>
#include <ATen/ops/_foreach_reciprocal.h>
#include <ATen/ops/_foreach_round.h>
#include <ATen/ops/_foreach_round.h>
#include <ATen/ops/_foreach_sinh.h>
#include <ATen/ops/_foreach_sinh.h>
#include <ATen/ops/_foreach_tan.h>
#include <ATen/ops/_foreach_tan.h>
#include <ATen/ops/_foreach_copy.h>
#include <ATen/ops/bucketize.h>
#include <ATen/ops/searchsorted.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr.h>
#include <ATen/ops/isfinite.h>
#include <ATen/ops/_remove_batch_dim.h>
#include <ATen/ops/_linalg_eigh.h>
#include <ATen/ops/outer.h>
#include <ATen/ops/_test_serialization_subcmul.h>
#include <ATen/ops/_test_autograd_multiple_dispatch_view.h>
#include <ATen/ops/view_as_real_copy.h>
#include <ATen/ops/permute_copy.h>
#include <ATen/ops/detach_copy.h>
#include <ATen/ops/split_copy.h>
#include <ATen/ops/unsqueeze_copy.h>
#include <ATen/ops/_indices_copy.h>
#include <ATen/ops/_values_copy.h>
#include <ATen/ops/unbind_copy.h>
#include <ATen/ops/alias_copy.h>
#include <ATen/ops/_nested_tensor_softmax_with_shape.h>
#include <ATen/ops/_safe_softmax.h>
#include <ATen/ops/_native_multi_head_attention.h>
#include <ATen/ops/_scaled_dot_product_attention_math_for_mps.h>
#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_for_cpu.h>
#include <ATen/ops/_triton_scaled_dot_attention.h>
#include <ATen/ops/_fused_adamw.h>
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

static PyObject * THPVariable__cast_Double(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cast_Int(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_align_tensors(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__print(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__make_dep_token(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__use_cudnn_rnn_flatten_weight(PyObject* self_, PyObject* args);
static PyObject * THPVariable__cudnn_rnn(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__fused_dropout(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sobol_engine_ff_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sobol_engine_scramble_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__shape_as_tensor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_dropout(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__conj(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_conj_physical(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_conj_physical_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_avg_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_add(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__add_relu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__add_relu_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_addr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__test_functorch_fallback(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_all(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_allclose(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_argmax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_acosh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_acosh_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_arccosh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_arccosh_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_arctanh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_arctanh_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_arcsin(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_arcsin_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_atan(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_atan_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_baddbmm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bartlett_window(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_copysign(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__lazy_clone(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_logical_not(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_logical_xor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bmm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_broadcast_tensors(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_broadcast_to(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_broadcast_to(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cat(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_chain_matmul(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_chunk(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_tensor_split(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_clamp_min(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_clamp_min_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_clip(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_clip_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_complex(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_polar(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__convolution(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__convolution_mode(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_conv1d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_conv3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cos(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cos_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_count_nonzero(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_corrcoef(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cudnn_convolution_transpose(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cudnn_convolution_relu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ctc_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_embedding_bag(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__embedding_bag(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_empty(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_empty_like(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_erf(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_erf_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_erfc(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_erfc_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_exp(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_exp_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_unflatten(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_floor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_floor_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_floor_divide(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_frac(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_frac_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_full_like(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_grid_sampler(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__grid_sampler_2d_cpu_fallback(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_grid_sampler_3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__fft_c2r(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cufft_get_plan_cache_max_size(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cufft_clear_plan_cache(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__unsafe_masked_index_put_accumulate(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__index_put_impl_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_isin(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_isnan(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_is_conj(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_is_inference(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_layer_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_nan_to_num(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_nan_to_num_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mkldnn_linear_backward_weights(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__cslt_sparse_mm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_semi_structured_mm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_semi_structured_addmm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__mixed_dtypes_linear(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fbgemm_pack_gemm_matrix_fp16(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__wrapped_quantized_linear_prepacked(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fbgemm_linear_fp16_weight_fp32_activation(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linspace(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_log(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_log_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_log1p(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_log1p_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_logaddexp(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_xlogy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_xlogy_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_logsumexp(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_matrix_power(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__aminmax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_max(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantized_max_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantized_max_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mean(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_nanmean(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_nanmedian(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_min(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_amin(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mkldnn_convolution(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_miopen_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_miopen_depthwise_convolution(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__weight_int8pack_mm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_narrow(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_batch_norm_stats(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_batch_norm_update_stats(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nnpack_available(PyObject* self_, PyObject* args);
static PyObject * THPVariable__nnpack_spatial_convolution(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__euclidean_dist(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_permute(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_pixel_shuffle(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_native_channel_shuffle(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__pin_memory(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rad2deg(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rad2deg_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rand_like(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_randint(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_randint_like(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_randn(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_randperm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_reciprocal(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_reciprocal_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_negative(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_negative_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_reshape(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__mkldnn_reshape(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_relu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_relu_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_prelu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_select(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_selu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_selu_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_logit(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_logit_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sin(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sin_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_slice_inverse(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_softmax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_unsafe_split_with_sizes(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_split_with_sizes(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_vsplit(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__chunk_cat(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_hstack(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_vstack(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_dstack(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sum(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_tan(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_tan_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_trapz(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_tensor_from_mask_left_aligned(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_from_padded_and_nested_example(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_view_from_buffer(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_get_values_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_get_lengths(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_trunc(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_trunc_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_unique_dim(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_unique_consecutive(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_where(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__standard_gamma(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__dirichlet_grad(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_binomial(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_csr_sum(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_log_softmax_backward_data(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_nuclear_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_clone(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_heaviside(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__scaled_mm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__validate_sparse_compressed_tensor_args(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__validate_sparse_csr_tensor_args(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__to_cpu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_hspmm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__to_sparse_semi_structured(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_q_per_channel_scales(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_q_per_channel_zero_points(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_q_per_channel_axis(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__fake_quantize_per_tensor_affine_cachemask_tensor_qparams(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__fake_quantize_learnable_per_tensor_affine(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fused_moving_avg_obs_fake_quant(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__saturate_weight_to_fp16(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_choose_qparams_optimized(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_combinations(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_result_type(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_can_cast(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__lstm_mps(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_lstm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_lstm_cell(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantized_rnn_relu_cell(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_quantized_rnn_tanh_cell(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_index_add(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_index_reduce(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_scatter_add(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_scatter_reduce(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bitwise_and(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bitwise_xor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bitwise_left_shift(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable___rshift__(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_addbmm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_diag(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cross(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_triu_indices(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_trace(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_ne(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_eq(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_greater(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_take(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__linalg_check_errors(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_svd(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cholesky(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__lu_with_info(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_lu_unpack(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_multinomial(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_i0(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_i0_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_atan2(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_hypot(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_igamma(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_nextafter(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fmin(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sort(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_renorm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__amp_foreach_non_finite_check_and_unscale_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__amp_update_scale_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_mul(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_mul_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_maximum(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_maximum_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_minimum(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_minimum_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_addcdiv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_addcdiv_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_abs(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_abs_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_erf(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_erf_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_frac(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_frac_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_lgamma(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_lgamma_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_pow(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_pow_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_reciprocal(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_reciprocal_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_round(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_round_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_sinh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_sinh_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_tan(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_tan_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__foreach_copy_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_bucketize(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_searchsorted(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__convert_indices_from_coo_to_csr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_isfinite(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__remove_batch_dim(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__linalg_eigh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_outer(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__test_serialization_subcmul(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__test_autograd_multiple_dispatch_view(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_view_as_real_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_permute_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_detach_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_split_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_unsqueeze_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__indices_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__values_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_unbind_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_alias_copy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__nested_tensor_softmax_with_shape(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__safe_softmax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__native_multi_head_attention(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__scaled_dot_product_attention_math_for_mps(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__scaled_dot_product_flash_attention(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__scaled_dot_product_flash_attention_for_cpu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__triton_scaled_dot_attention(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__fused_adamw_(PyObject* self_, PyObject* args, PyObject* kwargs);

static PyMethodDef torch_functions_shard[] = {
  {"_cast_Double", castPyCFunctionWithKeywords(THPVariable__cast_Double), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_cast_Int", castPyCFunctionWithKeywords(THPVariable__cast_Int), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"align_tensors", castPyCFunctionWithKeywords(THPVariable_align_tensors), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_print", castPyCFunctionWithKeywords(THPVariable__print), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_make_dep_token", castPyCFunctionWithKeywords(THPVariable__make_dep_token), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_use_cudnn_rnn_flatten_weight", THPVariable__use_cudnn_rnn_flatten_weight, METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_cudnn_rnn", castPyCFunctionWithKeywords(THPVariable__cudnn_rnn), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_fused_dropout", castPyCFunctionWithKeywords(THPVariable__fused_dropout), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sobol_engine_ff_", castPyCFunctionWithKeywords(THPVariable__sobol_engine_ff_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sobol_engine_scramble_", castPyCFunctionWithKeywords(THPVariable__sobol_engine_scramble_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_shape_as_tensor", castPyCFunctionWithKeywords(THPVariable__shape_as_tensor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"dropout", castPyCFunctionWithKeywords(THPVariable_dropout), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"dropout_", castPyCFunctionWithKeywords(THPVariable_dropout_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_conj", castPyCFunctionWithKeywords(THPVariable__conj), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"conj_physical", castPyCFunctionWithKeywords(THPVariable_conj_physical), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"conj_physical_", castPyCFunctionWithKeywords(THPVariable_conj_physical_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"avg_pool1d", castPyCFunctionWithKeywords(THPVariable_avg_pool1d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"add", castPyCFunctionWithKeywords(THPVariable_add), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_add_relu", castPyCFunctionWithKeywords(THPVariable__add_relu), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_add_relu_", castPyCFunctionWithKeywords(THPVariable__add_relu_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"addr", castPyCFunctionWithKeywords(THPVariable_addr), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_test_functorch_fallback", castPyCFunctionWithKeywords(THPVariable__test_functorch_fallback), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"all", castPyCFunctionWithKeywords(THPVariable_all), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"allclose", castPyCFunctionWithKeywords(THPVariable_allclose), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"argmax", castPyCFunctionWithKeywords(THPVariable_argmax), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"acosh", castPyCFunctionWithKeywords(THPVariable_acosh), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"acosh_", castPyCFunctionWithKeywords(THPVariable_acosh_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"arccosh", castPyCFunctionWithKeywords(THPVariable_arccosh), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"arccosh_", castPyCFunctionWithKeywords(THPVariable_arccosh_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"arctanh", castPyCFunctionWithKeywords(THPVariable_arctanh), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"arctanh_", castPyCFunctionWithKeywords(THPVariable_arctanh_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"arcsin", castPyCFunctionWithKeywords(THPVariable_arcsin), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"arcsin_", castPyCFunctionWithKeywords(THPVariable_arcsin_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"atan", castPyCFunctionWithKeywords(THPVariable_atan), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"atan_", castPyCFunctionWithKeywords(THPVariable_atan_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"baddbmm", castPyCFunctionWithKeywords(THPVariable_baddbmm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"bartlett_window", castPyCFunctionWithKeywords(THPVariable_bartlett_window), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"batch_norm", castPyCFunctionWithKeywords(THPVariable_batch_norm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"copysign", castPyCFunctionWithKeywords(THPVariable_copysign), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_lazy_clone", castPyCFunctionWithKeywords(THPVariable__lazy_clone), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"logical_not", castPyCFunctionWithKeywords(THPVariable_logical_not), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"logical_xor", castPyCFunctionWithKeywords(THPVariable_logical_xor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"bmm", castPyCFunctionWithKeywords(THPVariable_bmm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"broadcast_tensors", castPyCFunctionWithKeywords(THPVariable_broadcast_tensors), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"broadcast_to", castPyCFunctionWithKeywords(THPVariable_broadcast_to), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_broadcast_to", castPyCFunctionWithKeywords(THPVariable__sparse_broadcast_to), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cat", castPyCFunctionWithKeywords(THPVariable_cat), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"chain_matmul", castPyCFunctionWithKeywords(THPVariable_chain_matmul), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"chunk", castPyCFunctionWithKeywords(THPVariable_chunk), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"tensor_split", castPyCFunctionWithKeywords(THPVariable_tensor_split), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"clamp_min", castPyCFunctionWithKeywords(THPVariable_clamp_min), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"clamp_min_", castPyCFunctionWithKeywords(THPVariable_clamp_min_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"clip", castPyCFunctionWithKeywords(THPVariable_clip), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"clip_", castPyCFunctionWithKeywords(THPVariable_clip_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"complex", castPyCFunctionWithKeywords(THPVariable_complex), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"polar", castPyCFunctionWithKeywords(THPVariable_polar), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_convolution", castPyCFunctionWithKeywords(THPVariable__convolution), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_convolution_mode", castPyCFunctionWithKeywords(THPVariable__convolution_mode), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"conv1d", castPyCFunctionWithKeywords(THPVariable_conv1d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"conv3d", castPyCFunctionWithKeywords(THPVariable_conv3d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cos", castPyCFunctionWithKeywords(THPVariable_cos), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cos_", castPyCFunctionWithKeywords(THPVariable_cos_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"count_nonzero", castPyCFunctionWithKeywords(THPVariable_count_nonzero), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"corrcoef", castPyCFunctionWithKeywords(THPVariable_corrcoef), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cudnn_convolution_transpose", castPyCFunctionWithKeywords(THPVariable_cudnn_convolution_transpose), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cudnn_convolution_relu", castPyCFunctionWithKeywords(THPVariable_cudnn_convolution_relu), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"ctc_loss", castPyCFunctionWithKeywords(THPVariable_ctc_loss), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"embedding_bag", castPyCFunctionWithKeywords(THPVariable_embedding_bag), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_embedding_bag", castPyCFunctionWithKeywords(THPVariable__embedding_bag), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"empty", castPyCFunctionWithKeywords(THPVariable_empty), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"empty_like", castPyCFunctionWithKeywords(THPVariable_empty_like), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"erf", castPyCFunctionWithKeywords(THPVariable_erf), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"erf_", castPyCFunctionWithKeywords(THPVariable_erf_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"erfc", castPyCFunctionWithKeywords(THPVariable_erfc), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"erfc_", castPyCFunctionWithKeywords(THPVariable_erfc_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"exp", castPyCFunctionWithKeywords(THPVariable_exp), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"exp_", castPyCFunctionWithKeywords(THPVariable_exp_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"unflatten", castPyCFunctionWithKeywords(THPVariable_unflatten), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"floor", castPyCFunctionWithKeywords(THPVariable_floor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"floor_", castPyCFunctionWithKeywords(THPVariable_floor_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"floor_divide", castPyCFunctionWithKeywords(THPVariable_floor_divide), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"frac", castPyCFunctionWithKeywords(THPVariable_frac), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"frac_", castPyCFunctionWithKeywords(THPVariable_frac_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"full_like", castPyCFunctionWithKeywords(THPVariable_full_like), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"grid_sampler", castPyCFunctionWithKeywords(THPVariable_grid_sampler), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_grid_sampler_2d_cpu_fallback", castPyCFunctionWithKeywords(THPVariable__grid_sampler_2d_cpu_fallback), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"grid_sampler_3d", castPyCFunctionWithKeywords(THPVariable_grid_sampler_3d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_fft_c2r", castPyCFunctionWithKeywords(THPVariable__fft_c2r), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_cufft_get_plan_cache_max_size", castPyCFunctionWithKeywords(THPVariable__cufft_get_plan_cache_max_size), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_cufft_clear_plan_cache", castPyCFunctionWithKeywords(THPVariable__cufft_clear_plan_cache), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_unsafe_masked_index_put_accumulate", castPyCFunctionWithKeywords(THPVariable__unsafe_masked_index_put_accumulate), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_index_put_impl_", castPyCFunctionWithKeywords(THPVariable__index_put_impl_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"isin", castPyCFunctionWithKeywords(THPVariable_isin), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"isnan", castPyCFunctionWithKeywords(THPVariable_isnan), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"is_conj", castPyCFunctionWithKeywords(THPVariable_is_conj), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"is_inference", castPyCFunctionWithKeywords(THPVariable_is_inference), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"layer_norm", castPyCFunctionWithKeywords(THPVariable_layer_norm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"nan_to_num", castPyCFunctionWithKeywords(THPVariable_nan_to_num), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"nan_to_num_", castPyCFunctionWithKeywords(THPVariable_nan_to_num_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"mkldnn_linear_backward_weights", castPyCFunctionWithKeywords(THPVariable_mkldnn_linear_backward_weights), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_cslt_sparse_mm", castPyCFunctionWithKeywords(THPVariable__cslt_sparse_mm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_semi_structured_mm", castPyCFunctionWithKeywords(THPVariable__sparse_semi_structured_mm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_semi_structured_addmm", castPyCFunctionWithKeywords(THPVariable__sparse_semi_structured_addmm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_mixed_dtypes_linear", castPyCFunctionWithKeywords(THPVariable__mixed_dtypes_linear), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"fbgemm_pack_gemm_matrix_fp16", castPyCFunctionWithKeywords(THPVariable_fbgemm_pack_gemm_matrix_fp16), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_wrapped_quantized_linear_prepacked", castPyCFunctionWithKeywords(THPVariable__wrapped_quantized_linear_prepacked), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"fbgemm_linear_fp16_weight_fp32_activation", castPyCFunctionWithKeywords(THPVariable_fbgemm_linear_fp16_weight_fp32_activation), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"linspace", castPyCFunctionWithKeywords(THPVariable_linspace), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"log", castPyCFunctionWithKeywords(THPVariable_log), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"log_", castPyCFunctionWithKeywords(THPVariable_log_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"log1p", castPyCFunctionWithKeywords(THPVariable_log1p), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"log1p_", castPyCFunctionWithKeywords(THPVariable_log1p_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"logaddexp", castPyCFunctionWithKeywords(THPVariable_logaddexp), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"xlogy", castPyCFunctionWithKeywords(THPVariable_xlogy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"xlogy_", castPyCFunctionWithKeywords(THPVariable_xlogy_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"logsumexp", castPyCFunctionWithKeywords(THPVariable_logsumexp), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"matrix_power", castPyCFunctionWithKeywords(THPVariable_matrix_power), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_aminmax", castPyCFunctionWithKeywords(THPVariable__aminmax), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"max", castPyCFunctionWithKeywords(THPVariable_max), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"quantized_max_pool1d", castPyCFunctionWithKeywords(THPVariable_quantized_max_pool1d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"quantized_max_pool2d", castPyCFunctionWithKeywords(THPVariable_quantized_max_pool2d), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"mean", castPyCFunctionWithKeywords(THPVariable_mean), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"nanmean", castPyCFunctionWithKeywords(THPVariable_nanmean), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"nanmedian", castPyCFunctionWithKeywords(THPVariable_nanmedian), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"min", castPyCFunctionWithKeywords(THPVariable_min), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"amin", castPyCFunctionWithKeywords(THPVariable_amin), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"mkldnn_convolution", castPyCFunctionWithKeywords(THPVariable_mkldnn_convolution), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"miopen_batch_norm", castPyCFunctionWithKeywords(THPVariable_miopen_batch_norm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"miopen_depthwise_convolution", castPyCFunctionWithKeywords(THPVariable_miopen_depthwise_convolution), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_weight_int8pack_mm", castPyCFunctionWithKeywords(THPVariable__weight_int8pack_mm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"mv", castPyCFunctionWithKeywords(THPVariable_mv), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"narrow", castPyCFunctionWithKeywords(THPVariable_narrow), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"batch_norm_stats", castPyCFunctionWithKeywords(THPVariable_batch_norm_stats), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"batch_norm_update_stats", castPyCFunctionWithKeywords(THPVariable_batch_norm_update_stats), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nnpack_available", THPVariable__nnpack_available, METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nnpack_spatial_convolution", castPyCFunctionWithKeywords(THPVariable__nnpack_spatial_convolution), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_euclidean_dist", castPyCFunctionWithKeywords(THPVariable__euclidean_dist), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"permute", castPyCFunctionWithKeywords(THPVariable_permute), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"pixel_shuffle", castPyCFunctionWithKeywords(THPVariable_pixel_shuffle), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"native_channel_shuffle", castPyCFunctionWithKeywords(THPVariable_native_channel_shuffle), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_pin_memory", castPyCFunctionWithKeywords(THPVariable__pin_memory), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"rad2deg", castPyCFunctionWithKeywords(THPVariable_rad2deg), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"rad2deg_", castPyCFunctionWithKeywords(THPVariable_rad2deg_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"rand_like", castPyCFunctionWithKeywords(THPVariable_rand_like), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"randint", castPyCFunctionWithKeywords(THPVariable_randint), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"randint_like", castPyCFunctionWithKeywords(THPVariable_randint_like), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"randn", castPyCFunctionWithKeywords(THPVariable_randn), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"randperm", castPyCFunctionWithKeywords(THPVariable_randperm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"reciprocal", castPyCFunctionWithKeywords(THPVariable_reciprocal), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"reciprocal_", castPyCFunctionWithKeywords(THPVariable_reciprocal_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"negative", castPyCFunctionWithKeywords(THPVariable_negative), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"negative_", castPyCFunctionWithKeywords(THPVariable_negative_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"reshape", castPyCFunctionWithKeywords(THPVariable_reshape), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_mkldnn_reshape", castPyCFunctionWithKeywords(THPVariable__mkldnn_reshape), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"relu", castPyCFunctionWithKeywords(THPVariable_relu), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"relu_", castPyCFunctionWithKeywords(THPVariable_relu_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"prelu", castPyCFunctionWithKeywords(THPVariable_prelu), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"select", castPyCFunctionWithKeywords(THPVariable_select), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"selu", castPyCFunctionWithKeywords(THPVariable_selu), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"selu_", castPyCFunctionWithKeywords(THPVariable_selu_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"logit", castPyCFunctionWithKeywords(THPVariable_logit), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"logit_", castPyCFunctionWithKeywords(THPVariable_logit_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sin", castPyCFunctionWithKeywords(THPVariable_sin), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sin_", castPyCFunctionWithKeywords(THPVariable_sin_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"slice_inverse", castPyCFunctionWithKeywords(THPVariable_slice_inverse), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"softmax", castPyCFunctionWithKeywords(THPVariable_softmax), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"unsafe_split_with_sizes", castPyCFunctionWithKeywords(THPVariable_unsafe_split_with_sizes), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"split_with_sizes", castPyCFunctionWithKeywords(THPVariable_split_with_sizes), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"vsplit", castPyCFunctionWithKeywords(THPVariable_vsplit), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_chunk_cat", castPyCFunctionWithKeywords(THPVariable__chunk_cat), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"hstack", castPyCFunctionWithKeywords(THPVariable_hstack), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"vstack", castPyCFunctionWithKeywords(THPVariable_vstack), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"dstack", castPyCFunctionWithKeywords(THPVariable_dstack), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sum", castPyCFunctionWithKeywords(THPVariable_sum), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"tan", castPyCFunctionWithKeywords(THPVariable_tan), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"tan_", castPyCFunctionWithKeywords(THPVariable_tan_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"trapz", castPyCFunctionWithKeywords(THPVariable_trapz), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_tensor_from_mask_left_aligned", castPyCFunctionWithKeywords(THPVariable__nested_tensor_from_mask_left_aligned), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_from_padded_and_nested_example", castPyCFunctionWithKeywords(THPVariable__nested_from_padded_and_nested_example), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_view_from_buffer", castPyCFunctionWithKeywords(THPVariable__nested_view_from_buffer), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_get_values_copy", castPyCFunctionWithKeywords(THPVariable__nested_get_values_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_get_lengths", castPyCFunctionWithKeywords(THPVariable__nested_get_lengths), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"trunc", castPyCFunctionWithKeywords(THPVariable_trunc), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"trunc_", castPyCFunctionWithKeywords(THPVariable_trunc_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"unique_dim", castPyCFunctionWithKeywords(THPVariable_unique_dim), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"unique_consecutive", castPyCFunctionWithKeywords(THPVariable_unique_consecutive), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"where", castPyCFunctionWithKeywords(THPVariable_where), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_standard_gamma", castPyCFunctionWithKeywords(THPVariable__standard_gamma), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_dirichlet_grad", castPyCFunctionWithKeywords(THPVariable__dirichlet_grad), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"binomial", castPyCFunctionWithKeywords(THPVariable_binomial), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_csr_sum", castPyCFunctionWithKeywords(THPVariable__sparse_csr_sum), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_sparse_log_softmax_backward_data", castPyCFunctionWithKeywords(THPVariable__sparse_log_softmax_backward_data), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"nuclear_norm", castPyCFunctionWithKeywords(THPVariable_nuclear_norm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"clone", castPyCFunctionWithKeywords(THPVariable_clone), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"heaviside", castPyCFunctionWithKeywords(THPVariable_heaviside), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_scaled_mm", castPyCFunctionWithKeywords(THPVariable__scaled_mm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_validate_sparse_compressed_tensor_args", castPyCFunctionWithKeywords(THPVariable__validate_sparse_compressed_tensor_args), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_validate_sparse_csr_tensor_args", castPyCFunctionWithKeywords(THPVariable__validate_sparse_csr_tensor_args), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_to_cpu", castPyCFunctionWithKeywords(THPVariable__to_cpu), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"hspmm", castPyCFunctionWithKeywords(THPVariable_hspmm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_to_sparse_semi_structured", castPyCFunctionWithKeywords(THPVariable__to_sparse_semi_structured), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"q_per_channel_scales", castPyCFunctionWithKeywords(THPVariable_q_per_channel_scales), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"q_per_channel_zero_points", castPyCFunctionWithKeywords(THPVariable_q_per_channel_zero_points), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"q_per_channel_axis", castPyCFunctionWithKeywords(THPVariable_q_per_channel_axis), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_fake_quantize_per_tensor_affine_cachemask_tensor_qparams", castPyCFunctionWithKeywords(THPVariable__fake_quantize_per_tensor_affine_cachemask_tensor_qparams), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_fake_quantize_learnable_per_tensor_affine", castPyCFunctionWithKeywords(THPVariable__fake_quantize_learnable_per_tensor_affine), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"fused_moving_avg_obs_fake_quant", castPyCFunctionWithKeywords(THPVariable_fused_moving_avg_obs_fake_quant), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_saturate_weight_to_fp16", castPyCFunctionWithKeywords(THPVariable__saturate_weight_to_fp16), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"choose_qparams_optimized", castPyCFunctionWithKeywords(THPVariable_choose_qparams_optimized), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"combinations", castPyCFunctionWithKeywords(THPVariable_combinations), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"result_type", castPyCFunctionWithKeywords(THPVariable_result_type), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"can_cast", castPyCFunctionWithKeywords(THPVariable_can_cast), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_lstm_mps", castPyCFunctionWithKeywords(THPVariable__lstm_mps), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"lstm", castPyCFunctionWithKeywords(THPVariable_lstm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"lstm_cell", castPyCFunctionWithKeywords(THPVariable_lstm_cell), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"quantized_rnn_relu_cell", castPyCFunctionWithKeywords(THPVariable_quantized_rnn_relu_cell), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"quantized_rnn_tanh_cell", castPyCFunctionWithKeywords(THPVariable_quantized_rnn_tanh_cell), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"index_add", castPyCFunctionWithKeywords(THPVariable_index_add), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"index_reduce", castPyCFunctionWithKeywords(THPVariable_index_reduce), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"scatter_add", castPyCFunctionWithKeywords(THPVariable_scatter_add), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"scatter_reduce", castPyCFunctionWithKeywords(THPVariable_scatter_reduce), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"bitwise_and", castPyCFunctionWithKeywords(THPVariable_bitwise_and), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"bitwise_xor", castPyCFunctionWithKeywords(THPVariable_bitwise_xor), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"bitwise_left_shift", castPyCFunctionWithKeywords(THPVariable_bitwise_left_shift), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"__rshift__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable___rshift__>), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"addbmm", castPyCFunctionWithKeywords(THPVariable_addbmm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"diag", castPyCFunctionWithKeywords(THPVariable_diag), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cross", castPyCFunctionWithKeywords(THPVariable_cross), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"triu_indices", castPyCFunctionWithKeywords(THPVariable_triu_indices), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"trace", castPyCFunctionWithKeywords(THPVariable_trace), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"ne", castPyCFunctionWithKeywords(THPVariable_ne), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"eq", castPyCFunctionWithKeywords(THPVariable_eq), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"greater", castPyCFunctionWithKeywords(THPVariable_greater), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"take", castPyCFunctionWithKeywords(THPVariable_take), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_linalg_check_errors", castPyCFunctionWithKeywords(THPVariable__linalg_check_errors), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"svd", castPyCFunctionWithKeywords(THPVariable_svd), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"cholesky", castPyCFunctionWithKeywords(THPVariable_cholesky), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_lu_with_info", castPyCFunctionWithKeywords(THPVariable__lu_with_info), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"lu_unpack", castPyCFunctionWithKeywords(THPVariable_lu_unpack), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"multinomial", castPyCFunctionWithKeywords(THPVariable_multinomial), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"i0", castPyCFunctionWithKeywords(THPVariable_i0), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"i0_", castPyCFunctionWithKeywords(THPVariable_i0_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"atan2", castPyCFunctionWithKeywords(THPVariable_atan2), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"hypot", castPyCFunctionWithKeywords(THPVariable_hypot), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"igamma", castPyCFunctionWithKeywords(THPVariable_igamma), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"nextafter", castPyCFunctionWithKeywords(THPVariable_nextafter), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"fmin", castPyCFunctionWithKeywords(THPVariable_fmin), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"sort", castPyCFunctionWithKeywords(THPVariable_sort), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"renorm", castPyCFunctionWithKeywords(THPVariable_renorm), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_amp_foreach_non_finite_check_and_unscale_", castPyCFunctionWithKeywords(THPVariable__amp_foreach_non_finite_check_and_unscale_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_amp_update_scale_", castPyCFunctionWithKeywords(THPVariable__amp_update_scale_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_mul", castPyCFunctionWithKeywords(THPVariable__foreach_mul), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_mul_", castPyCFunctionWithKeywords(THPVariable__foreach_mul_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_maximum", castPyCFunctionWithKeywords(THPVariable__foreach_maximum), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_maximum_", castPyCFunctionWithKeywords(THPVariable__foreach_maximum_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_minimum", castPyCFunctionWithKeywords(THPVariable__foreach_minimum), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_minimum_", castPyCFunctionWithKeywords(THPVariable__foreach_minimum_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_addcdiv", castPyCFunctionWithKeywords(THPVariable__foreach_addcdiv), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_addcdiv_", castPyCFunctionWithKeywords(THPVariable__foreach_addcdiv_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_abs", castPyCFunctionWithKeywords(THPVariable__foreach_abs), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_abs_", castPyCFunctionWithKeywords(THPVariable__foreach_abs_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_erf", castPyCFunctionWithKeywords(THPVariable__foreach_erf), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_erf_", castPyCFunctionWithKeywords(THPVariable__foreach_erf_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_frac", castPyCFunctionWithKeywords(THPVariable__foreach_frac), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_frac_", castPyCFunctionWithKeywords(THPVariable__foreach_frac_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_lgamma", castPyCFunctionWithKeywords(THPVariable__foreach_lgamma), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_lgamma_", castPyCFunctionWithKeywords(THPVariable__foreach_lgamma_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_pow", castPyCFunctionWithKeywords(THPVariable__foreach_pow), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_pow_", castPyCFunctionWithKeywords(THPVariable__foreach_pow_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_reciprocal", castPyCFunctionWithKeywords(THPVariable__foreach_reciprocal), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_reciprocal_", castPyCFunctionWithKeywords(THPVariable__foreach_reciprocal_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_round", castPyCFunctionWithKeywords(THPVariable__foreach_round), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_round_", castPyCFunctionWithKeywords(THPVariable__foreach_round_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_sinh", castPyCFunctionWithKeywords(THPVariable__foreach_sinh), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_sinh_", castPyCFunctionWithKeywords(THPVariable__foreach_sinh_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_tan", castPyCFunctionWithKeywords(THPVariable__foreach_tan), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_tan_", castPyCFunctionWithKeywords(THPVariable__foreach_tan_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_foreach_copy_", castPyCFunctionWithKeywords(THPVariable__foreach_copy_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"bucketize", castPyCFunctionWithKeywords(THPVariable_bucketize), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"searchsorted", castPyCFunctionWithKeywords(THPVariable_searchsorted), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_convert_indices_from_coo_to_csr", castPyCFunctionWithKeywords(THPVariable__convert_indices_from_coo_to_csr), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"isfinite", castPyCFunctionWithKeywords(THPVariable_isfinite), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_remove_batch_dim", castPyCFunctionWithKeywords(THPVariable__remove_batch_dim), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_linalg_eigh", castPyCFunctionWithKeywords(THPVariable__linalg_eigh), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"outer", castPyCFunctionWithKeywords(THPVariable_outer), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_test_serialization_subcmul", castPyCFunctionWithKeywords(THPVariable__test_serialization_subcmul), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_test_autograd_multiple_dispatch_view", castPyCFunctionWithKeywords(THPVariable__test_autograd_multiple_dispatch_view), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"view_as_real_copy", castPyCFunctionWithKeywords(THPVariable_view_as_real_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"permute_copy", castPyCFunctionWithKeywords(THPVariable_permute_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"detach_copy", castPyCFunctionWithKeywords(THPVariable_detach_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"split_copy", castPyCFunctionWithKeywords(THPVariable_split_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"unsqueeze_copy", castPyCFunctionWithKeywords(THPVariable_unsqueeze_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_indices_copy", castPyCFunctionWithKeywords(THPVariable__indices_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_values_copy", castPyCFunctionWithKeywords(THPVariable__values_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"unbind_copy", castPyCFunctionWithKeywords(THPVariable_unbind_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"alias_copy", castPyCFunctionWithKeywords(THPVariable_alias_copy), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_nested_tensor_softmax_with_shape", castPyCFunctionWithKeywords(THPVariable__nested_tensor_softmax_with_shape), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_safe_softmax", castPyCFunctionWithKeywords(THPVariable__safe_softmax), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_native_multi_head_attention", castPyCFunctionWithKeywords(THPVariable__native_multi_head_attention), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_scaled_dot_product_attention_math_for_mps", castPyCFunctionWithKeywords(THPVariable__scaled_dot_product_attention_math_for_mps), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_scaled_dot_product_flash_attention", castPyCFunctionWithKeywords(THPVariable__scaled_dot_product_flash_attention), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_scaled_dot_product_flash_attention_for_cpu", castPyCFunctionWithKeywords(THPVariable__scaled_dot_product_flash_attention_for_cpu), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_triton_scaled_dot_attention", castPyCFunctionWithKeywords(THPVariable__triton_scaled_dot_attention), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
  {"_fused_adamw_", castPyCFunctionWithKeywords(THPVariable__fused_adamw_), METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
};

void gatherTorchFunctions_2(std::vector<PyMethodDef> &torch_functions) {
  constexpr size_t num_functions = sizeof(torch_functions_shard) / sizeof(torch_functions_shard[0]);
  torch_functions.insert(
    torch_functions.end(),
    torch_functions_shard,
    torch_functions_shard + num_functions);
}

// generated methods start here

// _cast_Double
static PyObject * THPVariable__cast_Double(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Double(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cast_Double(Tensor self, bool non_blocking=False) -> Tensor
  
  auto dispatch__cast_Double = [](const at::Tensor & self, bool non_blocking) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cast_Double(self, non_blocking);
  };
  return wrap(dispatch__cast_Double(_r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cast_Int
static PyObject * THPVariable__cast_Int(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cast_Int(Tensor input, bool non_blocking=False)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cast_Int(Tensor self, bool non_blocking=False) -> Tensor
  
  auto dispatch__cast_Int = [](const at::Tensor & self, bool non_blocking) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cast_Int(self, non_blocking);
  };
  return wrap(dispatch__cast_Int(_r.tensor(0), _r.toBool(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// align_tensors
static PyObject * THPVariable_align_tensors(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "align_tensors(TensorList tensors)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::align_tensors(Tensor[] tensors) -> Tensor[]
  
  auto dispatch_align_tensors = [](at::TensorList tensors) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::align_tensors(tensors);
  };
  return wrap(dispatch_align_tensors(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _print
static PyObject * THPVariable__print(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_print(c10::string_view s)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_print(str s) -> ()
  
  auto dispatch__print = [](c10::string_view s) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_print(s);
  };
  dispatch__print(_r.stringView(0));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _make_dep_token
static PyObject * THPVariable__make_dep_token(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_make_dep_token(*, MemoryFormat? memory_format=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_make_dep_token(*, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(1))
      .device(_r.deviceWithDefault(3, torch::tensors::get_default_device()))
      .layout(_r.layoutOptional(2))
      .requires_grad(_r.toBool(5))
      .pinned_memory(_r.toBool(4));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch__make_dep_token = [](at::TensorOptions options, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::_make_dep_token(options, memory_format);
  };
  return wrap(dispatch__make_dep_token(options, _r.memoryformatOptional(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _use_cudnn_rnn_flatten_weight
static PyObject * THPVariable__use_cudnn_rnn_flatten_weight(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  // aten::_use_cudnn_rnn_flatten_weight() -> bool
  
  auto dispatch__use_cudnn_rnn_flatten_weight = []() -> bool {
    pybind11::gil_scoped_release no_gil;
    return at::_use_cudnn_rnn_flatten_weight();
  };
  return wrap(dispatch__use_cudnn_rnn_flatten_weight());
  END_HANDLE_TH_ERRORS
}

// _cudnn_rnn
static PyObject * THPVariable__cudnn_rnn(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cudnn_rnn(Tensor input, TensorList weight, int64_t weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int64_t mode, SymInt hidden_size, SymInt proj_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, SymIntArrayRef batch_sizes, Tensor? dropout_state)",
  }, /*traceable=*/true);

  ParsedArgs<16> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cudnn_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int mode, SymInt hidden_size, SymInt proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, SymInt[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
  
  auto dispatch__cudnn_rnn = [](const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const ::std::optional<at::Tensor> & weight_buf, const at::Tensor & hx, const ::std::optional<at::Tensor> & cx, int64_t mode, c10::SymInt hidden_size, c10::SymInt proj_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, c10::SymIntArrayRef batch_sizes, const ::std::optional<at::Tensor> & dropout_state) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_cudnn_rnn_symint(input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
  };
  return wrap(dispatch__cudnn_rnn(_r.tensor(0), _r.tensorlist(1), _r.toInt64(2), _r.optionalTensor(3), _r.tensor(4), _r.optionalTensor(5), _r.toInt64(6), _r.toSymInt(7), _r.toSymInt(8), _r.toInt64(9), _r.toBool(10), _r.toDouble(11), _r.toBool(12), _r.toBool(13), _r.symintlist(14), _r.optionalTensor(15)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _fused_dropout
static PyObject * THPVariable__fused_dropout(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_fused_dropout(Tensor input, double p, Generator? generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)
  
  auto dispatch__fused_dropout = [](const at::Tensor & self, double p, ::std::optional<at::Generator> generator) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_fused_dropout(self, p, generator);
  };
  return wrap(dispatch__fused_dropout(_r.tensor(0), _r.toDouble(1), _r.generator(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sobol_engine_ff_
static PyObject * THPVariable__sobol_engine_ff_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sobol_engine_ff_(Tensor input, int64_t n, Tensor sobolstate, int64_t dimension, int64_t num_generated)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_sobol_engine_ff_(Tensor(a!) self, int n, Tensor sobolstate, int dimension, int num_generated) -> Tensor(a!)
  
  auto dispatch__sobol_engine_ff_ = [](at::Tensor self, int64_t n, const at::Tensor & sobolstate, int64_t dimension, int64_t num_generated) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sobol_engine_ff_(self, n, sobolstate, dimension, num_generated);
  };
  return wrap(dispatch__sobol_engine_ff_(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.toInt64(3), _r.toInt64(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sobol_engine_scramble_
static PyObject * THPVariable__sobol_engine_scramble_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sobol_engine_scramble_(Tensor input, Tensor ltm, int64_t dimension)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_sobol_engine_scramble_(Tensor(a!) self, Tensor ltm, int dimension) -> Tensor(a!)
  
  auto dispatch__sobol_engine_scramble_ = [](at::Tensor self, const at::Tensor & ltm, int64_t dimension) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sobol_engine_scramble_(self, ltm, dimension);
  };
  return wrap(dispatch__sobol_engine_scramble_(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _shape_as_tensor
static PyObject * THPVariable__shape_as_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_shape_as_tensor(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_shape_as_tensor(Tensor self) -> Tensor
  
  auto dispatch__shape_as_tensor = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_shape_as_tensor(self);
  };
  return wrap(dispatch__shape_as_tensor(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// dropout
static PyObject * THPVariable_dropout(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dropout(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::dropout(Tensor input, float p, bool train) -> Tensor
  
  auto dispatch_dropout = [](const at::Tensor & input, double p, bool train) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::dropout(input, p, train);
  };
  return wrap(dispatch_dropout(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// dropout_
static PyObject * THPVariable_dropout_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dropout_(Tensor input, double p, bool train)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
  
  auto dispatch_dropout_ = [](at::Tensor self, double p, bool train) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::dropout_(self, p, train);
  };
  return wrap(dispatch_dropout_(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _conj
static PyObject * THPVariable__conj(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_conj(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_conj(Tensor(a) self) -> Tensor(a)
  
  auto dispatch__conj = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._conj();
  };
  return wrap(dispatch__conj(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// conj_physical
static PyObject * THPVariable_conj_physical(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conj_physical(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::conj_physical(Tensor self) -> Tensor
    
    auto dispatch_conj_physical = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.conj_physical();
    };
    return wrap(dispatch_conj_physical(_r.tensor(0)));
  } else {
    // aten::conj_physical.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_conj_physical_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::conj_physical_out(out, self);
    };
    return wrap(dispatch_conj_physical_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// conj_physical_
static PyObject * THPVariable_conj_physical_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conj_physical_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::conj_physical_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_conj_physical_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.conj_physical_();
  };
  return wrap(dispatch_conj_physical_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// avg_pool1d
static PyObject * THPVariable_avg_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "avg_pool1d(Tensor input, IntArrayRef[1] kernel_size, IntArrayRef[1] stride=None, IntArrayRef[1] padding=0, bool ceil_mode=False, bool count_include_pad=True)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor
  
  auto dispatch_avg_pool1d = [](const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::avg_pool1d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
  };
  return wrap(dispatch_avg_pool1d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.toBool(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// add
static PyObject * THPVariable_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "add(Tensor input, Scalar alpha, Tensor other, *, Tensor out=None)|deprecated",
    "add(Tensor input, Tensor other, *, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // [deprecated] aten::add(Tensor self, Scalar alpha, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_add = [](at::Tensor out, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.add(other, alpha);
        };
        return wrap(dispatch_add(_r.tensor(3), _r.tensor(0), _r.scalar(1), _r.tensor(2)));
      } else {
        // [deprecated] aten::add(Tensor self, Scalar alpha, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_add_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::add_out(out, self, other, alpha);
        };
        return wrap(dispatch_add_out(_r.tensor(3), _r.tensor(0), _r.scalar(1), _r.tensor(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
        
        auto dispatch_add = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.add(other, alpha);
        };
        return wrap(dispatch_add(_r.tensor(0), _r.tensor(1), _r.scalar(2)));
      } else {
        // aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_add_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::add_out(out, self, other, alpha);
        };
        return wrap(dispatch_add_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.scalar(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _add_relu
static PyObject * THPVariable__add_relu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_add_relu(Tensor input, Tensor other, *, Scalar alpha=1, Tensor out=None)",
    "_add_relu(Tensor input, Scalar other, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::_add_relu.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
        
        auto dispatch__add_relu = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::_add_relu(self, other, alpha);
        };
        return wrap(dispatch__add_relu(_r.tensor(0), _r.tensor(1), _r.scalar(2)));
      } else {
        // aten::_add_relu.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch__add_relu_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::_add_relu_out(out, self, other, alpha);
        };
        return wrap(dispatch__add_relu_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.scalar(2)));
      }
    }
    case 1: {
      // aten::_add_relu.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
      
      auto dispatch__add_relu = [](const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_add_relu(self, other, alpha);
      };
      return wrap(dispatch__add_relu(_r.tensor(0), _r.scalar(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _add_relu_
static PyObject * THPVariable__add_relu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_add_relu_(Tensor input, Tensor other, *, Scalar alpha=1)",
    "_add_relu_(Tensor input, Scalar other, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_add_relu_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
      
      auto dispatch__add_relu_ = [](at::Tensor self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_add_relu_(self, other, alpha);
      };
      return wrap(dispatch__add_relu_(_r.tensor(0), _r.tensor(1), _r.scalar(2)));
    }
    case 1: {
      // aten::_add_relu_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
      
      auto dispatch__add_relu_ = [](at::Tensor self, const at::Scalar & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_add_relu_(self, other, alpha);
      };
      return wrap(dispatch__add_relu_(_r.tensor(0), _r.scalar(1), _r.scalar(2)));
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
  static PythonArgParser parser({
    "addr(Scalar beta, Tensor input, Scalar alpha, Tensor vec1, Tensor vec2, *, Tensor out=None)|deprecated",
    "addr(Scalar beta, Tensor input, Tensor vec1, Tensor vec2, *, Tensor out=None)|deprecated",
    "addr(Tensor input, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(5)) {
        // [deprecated] aten::addr(Scalar beta, Tensor self, Scalar alpha, Tensor vec1, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addr = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & vec1, const at::Tensor & vec2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addr(vec1, vec2, beta, alpha);
        };
        return wrap(dispatch_addr(_r.tensor(5), _r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4)));
      } else {
        // [deprecated] aten::addr(Scalar beta, Tensor self, Scalar alpha, Tensor vec1, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addr_out = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & vec1, const at::Tensor & vec2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addr_out(out, self, vec1, vec2, beta, alpha);
        };
        return wrap(dispatch_addr_out(_r.tensor(5), _r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // [deprecated] aten::addr(Scalar beta, Tensor self, Tensor vec1, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addr = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addr(vec1, vec2, beta, 1);
        };
        return wrap(dispatch_addr(_r.tensor(4), _r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
      } else {
        // [deprecated] aten::addr(Scalar beta, Tensor self, Tensor vec1, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addr_out = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addr_out(out, self, vec1, vec2, beta, 1);
        };
        return wrap(dispatch_addr_out(_r.tensor(4), _r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
      }
    }
    case 2: {
      if (_r.isNone(5)) {
        // aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        
        auto dispatch_addr = [](const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addr(vec1, vec2, beta, alpha);
        };
        return wrap(dispatch_addr(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      } else {
        // aten::addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addr_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addr_out(out, self, vec1, vec2, beta, alpha);
        };
        return wrap(dispatch_addr_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _test_functorch_fallback
static PyObject * THPVariable__test_functorch_fallback(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_test_functorch_fallback(Tensor input, Tensor other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_test_functorch_fallback(Tensor self, Tensor other) -> Tensor
  
  auto dispatch__test_functorch_fallback = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_test_functorch_fallback(self, other);
  };
  return wrap(dispatch__test_functorch_fallback(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// all
static PyObject * THPVariable_all(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "all(Tensor input, *, Tensor out=None)",
    "all(Tensor input, IntArrayRef? dim=None, bool keepdim=False, *, Tensor out=None)",
    "all(Tensor input, int64_t dim, bool keepdim=False, *, Tensor out=None)",
    "all(Tensor input, Dimname dim, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(1)) {
        // aten::all(Tensor self) -> Tensor
        
        auto dispatch_all = [](const at::Tensor & self) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.all();
        };
        return wrap(dispatch_all(_r.tensor(0)));
      } else {
        // aten::all.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_all_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::all_out(out, self);
        };
        return wrap(dispatch_all_out(_r.tensor(1), _r.tensor(0)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::all.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor
        
        auto dispatch_all = [](const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.all(dim, keepdim);
        };
        return wrap(dispatch_all(_r.tensor(0), _r.intlistOptional(1), _r.toBool(2)));
      } else {
        // aten::all.dims_out(Tensor self, int[]? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_all_out = [](at::Tensor out, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::all_out(out, self, dim, keepdim);
        };
        return wrap(dispatch_all_out(_r.tensor(3), _r.tensor(0), _r.intlistOptional(1), _r.toBool(2)));
      }
    }
    case 2: {
      if (_r.isNone(3)) {
        // aten::all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
        
        auto dispatch_all = [](const at::Tensor & self, int64_t dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.all(dim, keepdim);
        };
        return wrap(dispatch_all(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      } else {
        // aten::all.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_all_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::all_out(out, self, dim, keepdim);
        };
        return wrap(dispatch_all_out(_r.tensor(3), _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      }
    }
    case 3: {
      if (_r.isNone(3)) {
        // aten::all.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor
        
        auto dispatch_all = [](const at::Tensor & self, at::Dimname dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.all(dim, keepdim);
        };
        return wrap(dispatch_all(_r.tensor(0), _r.dimname(1), _r.toBool(2)));
      } else {
        // aten::all.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_all_out = [](at::Tensor out, const at::Tensor & self, at::Dimname dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::all_out(out, self, dim, keepdim);
        };
        return wrap(dispatch_all_out(_r.tensor(3), _r.tensor(0), _r.dimname(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// allclose
static PyObject * THPVariable_allclose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "allclose(Tensor input, Tensor other, double rtol=1e-05, double atol=1e-08, bool equal_nan=False)",
  }, /*traceable=*/false);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool
  
  auto dispatch_allclose = [](const at::Tensor & self, const at::Tensor & other, double rtol, double atol, bool equal_nan) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.allclose(other, rtol, atol, equal_nan);
  };
  return wrap(dispatch_allclose(_r.tensor(0), _r.tensor(1), _r.toDouble(2), _r.toDouble(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// argmax
static PyObject * THPVariable_argmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "argmax(Tensor input, int64_t? dim=None, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
    
    auto dispatch_argmax = [](const at::Tensor & self, ::std::optional<int64_t> dim, bool keepdim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.argmax(dim, keepdim);
    };
    return wrap(dispatch_argmax(_r.tensor(0), _r.toInt64Optional(1), _r.toBool(2)));
  } else {
    // aten::argmax.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_argmax_out = [](at::Tensor out, const at::Tensor & self, ::std::optional<int64_t> dim, bool keepdim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::argmax_out(out, self, dim, keepdim);
    };
    return wrap(dispatch_argmax_out(_r.tensor(3), _r.tensor(0), _r.toInt64Optional(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// acosh
static PyObject * THPVariable_acosh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "acosh(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::acosh(Tensor self) -> Tensor
    
    auto dispatch_acosh = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.acosh();
    };
    return wrap(dispatch_acosh(_r.tensor(0)));
  } else {
    // aten::acosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_acosh_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::acosh_out(out, self);
    };
    return wrap(dispatch_acosh_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// acosh_
static PyObject * THPVariable_acosh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "acosh_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::acosh_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_acosh_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.acosh_();
  };
  return wrap(dispatch_acosh_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// arccosh
static PyObject * THPVariable_arccosh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "arccosh(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::arccosh(Tensor self) -> Tensor
    
    auto dispatch_arccosh = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.arccosh();
    };
    return wrap(dispatch_arccosh(_r.tensor(0)));
  } else {
    // aten::arccosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_arccosh_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::arccosh_out(out, self);
    };
    return wrap(dispatch_arccosh_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// arccosh_
static PyObject * THPVariable_arccosh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "arccosh_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::arccosh_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_arccosh_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arccosh_();
  };
  return wrap(dispatch_arccosh_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// arctanh
static PyObject * THPVariable_arctanh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "arctanh(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::arctanh(Tensor self) -> Tensor
    
    auto dispatch_arctanh = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.arctanh();
    };
    return wrap(dispatch_arctanh(_r.tensor(0)));
  } else {
    // aten::arctanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_arctanh_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::arctanh_out(out, self);
    };
    return wrap(dispatch_arctanh_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// arctanh_
static PyObject * THPVariable_arctanh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "arctanh_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::arctanh_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_arctanh_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arctanh_();
  };
  return wrap(dispatch_arctanh_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// arcsin
static PyObject * THPVariable_arcsin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "arcsin(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::arcsin(Tensor self) -> Tensor
    
    auto dispatch_arcsin = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.arcsin();
    };
    return wrap(dispatch_arcsin(_r.tensor(0)));
  } else {
    // aten::arcsin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_arcsin_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::arcsin_out(out, self);
    };
    return wrap(dispatch_arcsin_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// arcsin_
static PyObject * THPVariable_arcsin_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "arcsin_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::arcsin_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_arcsin_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.arcsin_();
  };
  return wrap(dispatch_arcsin_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// atan
static PyObject * THPVariable_atan(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atan(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::atan(Tensor self) -> Tensor
    
    auto dispatch_atan = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.atan();
    };
    return wrap(dispatch_atan(_r.tensor(0)));
  } else {
    // aten::atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_atan_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::atan_out(out, self);
    };
    return wrap(dispatch_atan_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// atan_
static PyObject * THPVariable_atan_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atan_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::atan_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_atan_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.atan_();
  };
  return wrap(dispatch_atan_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// baddbmm
static PyObject * THPVariable_baddbmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "baddbmm(Scalar beta, Tensor input, Scalar alpha, Tensor batch1, Tensor batch2, *, Tensor out=None)|deprecated",
    "baddbmm(Scalar beta, Tensor input, Tensor batch1, Tensor batch2, *, Tensor out=None)|deprecated",
    "baddbmm(Tensor input, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(5)) {
        // [deprecated] aten::baddbmm(Scalar beta, Tensor self, Scalar alpha, Tensor batch1, Tensor batch2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_baddbmm = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & batch1, const at::Tensor & batch2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.baddbmm(batch1, batch2, beta, alpha);
        };
        return wrap(dispatch_baddbmm(_r.tensor(5), _r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4)));
      } else {
        // [deprecated] aten::baddbmm(Scalar beta, Tensor self, Scalar alpha, Tensor batch1, Tensor batch2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_baddbmm_out = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & batch1, const at::Tensor & batch2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::baddbmm_out(out, self, batch1, batch2, beta, alpha);
        };
        return wrap(dispatch_baddbmm_out(_r.tensor(5), _r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // [deprecated] aten::baddbmm(Scalar beta, Tensor self, Tensor batch1, Tensor batch2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_baddbmm = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.baddbmm(batch1, batch2, beta, 1);
        };
        return wrap(dispatch_baddbmm(_r.tensor(4), _r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
      } else {
        // [deprecated] aten::baddbmm(Scalar beta, Tensor self, Tensor batch1, Tensor batch2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_baddbmm_out = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::baddbmm_out(out, self, batch1, batch2, beta, 1);
        };
        return wrap(dispatch_baddbmm_out(_r.tensor(4), _r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
      }
    }
    case 2: {
      if (_r.isNone(5)) {
        // aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        
        auto dispatch_baddbmm = [](const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.baddbmm(batch1, batch2, beta, alpha);
        };
        return wrap(dispatch_baddbmm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      } else {
        // aten::baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_baddbmm_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::baddbmm_out(out, self, batch1, batch2, beta, alpha);
        };
        return wrap(dispatch_baddbmm_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bartlett_window
static PyObject * THPVariable_bartlett_window(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bartlett_window(int64_t window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "bartlett_window(int64_t window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::bartlett_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(1))
          .device(_r.deviceWithDefault(3, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(2))
          .requires_grad(_r.toBool(5))
          .pinned_memory(_r.toBool(4));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_bartlett_window = [](int64_t window_length, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::bartlett_window(window_length, options);
      };
      return wrap(dispatch_bartlett_window(_r.toInt64(0), options));
    }
    case 1: {
      // aten::bartlett_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(2))
          .device(_r.deviceWithDefault(4, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(3))
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_bartlett_window = [](int64_t window_length, bool periodic, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::bartlett_window(window_length, periodic, options);
      };
      return wrap(dispatch_bartlett_window(_r.toInt64(0), _r.toBool(1), options));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// batch_norm
static PyObject * THPVariable_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, double momentum, double eps, bool cudnn_enabled)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor
  
  auto dispatch_batch_norm = [](const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
  };
  return wrap(dispatch_batch_norm(_r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2), _r.optionalTensor(3), _r.optionalTensor(4), _r.toBool(5), _r.toDouble(6), _r.toDouble(7), _r.toBool(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// copysign
static PyObject * THPVariable_copysign(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "copysign(Tensor input, Tensor other, *, Tensor out=None)",
    "copysign(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::copysign.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_copysign = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.copysign(other);
        };
        return wrap(dispatch_copysign(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::copysign.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_copysign_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::copysign_out(out, self, other);
        };
        return wrap(dispatch_copysign_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::copysign.Scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_copysign = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.copysign(other);
        };
        return wrap(dispatch_copysign(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::copysign.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_copysign_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::copysign_out(out, self, other);
        };
        return wrap(dispatch_copysign_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _lazy_clone
static PyObject * THPVariable__lazy_clone(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_lazy_clone(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_lazy_clone(Tensor self) -> Tensor
  
  auto dispatch__lazy_clone = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self._lazy_clone();
  };
  return wrap(dispatch__lazy_clone(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logical_not
static PyObject * THPVariable_logical_not(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logical_not(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::logical_not(Tensor self) -> Tensor
    
    auto dispatch_logical_not = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.logical_not();
    };
    return wrap(dispatch_logical_not(_r.tensor(0)));
  } else {
    // aten::logical_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_logical_not_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::logical_not_out(out, self);
    };
    return wrap(dispatch_logical_not_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logical_xor
static PyObject * THPVariable_logical_xor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logical_xor(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::logical_xor(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_logical_xor = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.logical_xor(other);
    };
    return wrap(dispatch_logical_xor(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::logical_xor.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_logical_xor_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::logical_xor_out(out, self, other);
    };
    return wrap(dispatch_logical_xor_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// bmm
static PyObject * THPVariable_bmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bmm(Tensor input, Tensor mat2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::bmm(Tensor self, Tensor mat2) -> Tensor
    
    auto dispatch_bmm = [](const at::Tensor & self, const at::Tensor & mat2) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.bmm(mat2);
    };
    return wrap(dispatch_bmm(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_bmm_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & mat2) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::bmm_out(out, self, mat2);
    };
    return wrap(dispatch_bmm_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// broadcast_tensors
static PyObject * THPVariable_broadcast_tensors(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "broadcast_tensors(TensorList tensors)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::broadcast_tensors(Tensor[] tensors) -> Tensor[]
  
  auto dispatch_broadcast_tensors = [](at::TensorList tensors) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::broadcast_tensors(tensors);
  };
  return wrap(dispatch_broadcast_tensors(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// broadcast_to
static PyObject * THPVariable_broadcast_to(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "broadcast_to(Tensor input, SymIntArrayRef size)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::broadcast_to(Tensor(a) self, SymInt[] size) -> Tensor(a)
  
  auto dispatch_broadcast_to = [](const at::Tensor & self, c10::SymIntArrayRef size) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.broadcast_to_symint(size);
  };
  return wrap(dispatch_broadcast_to(_r.tensor(0), _r.symintlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sparse_broadcast_to
static PyObject * THPVariable__sparse_broadcast_to(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_broadcast_to(Tensor input, IntArrayRef size)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_sparse_broadcast_to(Tensor(a) self, int[] size) -> Tensor(a)
  
  auto dispatch__sparse_broadcast_to = [](const at::Tensor & self, at::IntArrayRef size) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sparse_broadcast_to(self, size);
  };
  return wrap(dispatch__sparse_broadcast_to(_r.tensor(0), _r.intlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// cat
static PyObject * THPVariable_cat(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cat(TensorList tensors, int64_t dim=0, *, Tensor out=None)",
    "cat(TensorList tensors, Dimname dim, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::cat(Tensor[] tensors, int dim=0) -> Tensor
        
        auto dispatch_cat = [](const at::ITensorListRef & tensors, int64_t dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::cat(tensors, dim);
        };
        return wrap(dispatch_cat(_r.tensorlist(0), _r.toInt64(1)));
      } else {
        // aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_cat_out = [](at::Tensor out, const at::ITensorListRef & tensors, int64_t dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::cat_out(out, tensors, dim);
        };
        return wrap(dispatch_cat_out(_r.tensor(2), _r.tensorlist(0), _r.toInt64(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::cat.names(Tensor[] tensors, Dimname dim) -> Tensor
        
        auto dispatch_cat = [](at::TensorList tensors, at::Dimname dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::cat(tensors, dim);
        };
        return wrap(dispatch_cat(_r.tensorlist(0), _r.dimname(1)));
      } else {
        // aten::cat.names_out(Tensor[] tensors, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_cat_out = [](at::Tensor out, at::TensorList tensors, at::Dimname dim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::cat_out(out, tensors, dim);
        };
        return wrap(dispatch_cat_out(_r.tensor(2), _r.tensorlist(0), _r.dimname(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// chain_matmul
static PyObject * THPVariable_chain_matmul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "chain_matmul(TensorList matrices, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::chain_matmul(Tensor[] matrices) -> Tensor
    
    auto dispatch_chain_matmul = [](at::TensorList matrices) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::chain_matmul(matrices);
    };
    return wrap(dispatch_chain_matmul(_r.tensorlist(0)));
  } else {
    // aten::chain_matmul.out(Tensor[] matrices, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_chain_matmul_out = [](at::Tensor out, at::TensorList matrices) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::chain_matmul_out(out, matrices);
    };
    return wrap(dispatch_chain_matmul_out(_r.tensor(1), _r.tensorlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// chunk
static PyObject * THPVariable_chunk(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "chunk(Tensor input, int64_t chunks, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::chunk(Tensor(a -> *) self, int chunks, int dim=0) -> Tensor(a)[]
  
  auto dispatch_chunk = [](const at::Tensor & self, int64_t chunks, int64_t dim) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.chunk(chunks, dim);
  };
  return wrap(dispatch_chunk(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// tensor_split
static PyObject * THPVariable_tensor_split(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tensor_split(Tensor input, SymIntArrayRef indices, int64_t dim=0)",
    "tensor_split(Tensor input, Tensor tensor_indices_or_sections, int64_t dim=0)",
    "tensor_split(Tensor input, SymInt sections, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::tensor_split.indices(Tensor(a -> *) self, SymInt[] indices, int dim=0) -> Tensor(a)[]
      
      auto dispatch_tensor_split = [](const at::Tensor & self, c10::SymIntArrayRef indices, int64_t dim) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.tensor_split_symint(indices, dim);
      };
      return wrap(dispatch_tensor_split(_r.tensor(0), _r.symintlist(1), _r.toInt64(2)));
    }
    case 1: {
      // aten::tensor_split.tensor_indices_or_sections(Tensor(a -> *) self, Tensor tensor_indices_or_sections, int dim=0) -> Tensor(a)[]
      
      auto dispatch_tensor_split = [](const at::Tensor & self, const at::Tensor & tensor_indices_or_sections, int64_t dim) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.tensor_split(tensor_indices_or_sections, dim);
      };
      return wrap(dispatch_tensor_split(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
    }
    case 2: {
      // aten::tensor_split.sections(Tensor(a -> *) self, SymInt sections, int dim=0) -> Tensor(a)[]
      
      auto dispatch_tensor_split = [](const at::Tensor & self, c10::SymInt sections, int64_t dim) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.tensor_split_symint(sections, dim);
      };
      return wrap(dispatch_tensor_split(_r.tensor(0), _r.toSymInt(1), _r.toInt64(2)));
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
  static PythonArgParser parser({
    "clamp_min(Tensor input, Tensor min, *, Tensor out=None)",
    "clamp_min(Tensor input, Scalar min, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::clamp_min.Tensor(Tensor self, Tensor min) -> Tensor
        
        auto dispatch_clamp_min = [](const at::Tensor & self, const at::Tensor & min) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.clamp_min(min);
        };
        return wrap(dispatch_clamp_min(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::clamp_min.Tensor_out(Tensor self, Tensor min, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_clamp_min_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & min) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::clamp_min_out(out, self, min);
        };
        return wrap(dispatch_clamp_min_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::clamp_min(Tensor self, Scalar min) -> Tensor
        
        auto dispatch_clamp_min = [](const at::Tensor & self, const at::Scalar & min) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.clamp_min(min);
        };
        return wrap(dispatch_clamp_min(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::clamp_min.out(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_clamp_min_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & min) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::clamp_min_out(out, self, min);
        };
        return wrap(dispatch_clamp_min_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
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
  static PythonArgParser parser({
    "clamp_min_(Tensor input, Tensor min)",
    "clamp_min_(Tensor input, Scalar min)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::clamp_min_.Tensor(Tensor(a!) self, Tensor min) -> Tensor(a!)
      
      auto dispatch_clamp_min_ = [](at::Tensor self, const at::Tensor & min) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clamp_min_(min);
      };
      return wrap(dispatch_clamp_min_(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)
      
      auto dispatch_clamp_min_ = [](at::Tensor self, const at::Scalar & min) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clamp_min_(min);
      };
      return wrap(dispatch_clamp_min_(_r.tensor(0), _r.scalar(1)));
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
  static PythonArgParser parser({
    "clip(Tensor input, Tensor? min=None, Tensor? max=None, *, Tensor out=None)",
    "clip(Tensor input, Scalar? min=None, Scalar? max=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::clip.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
        
        auto dispatch_clip = [](const at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.clip(min, max);
        };
        return wrap(dispatch_clip(_r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2)));
      } else {
        // aten::clip.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_clip_out = [](at::Tensor out, const at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::clip_out(out, self, min, max);
        };
        return wrap(dispatch_clip_out(_r.tensor(3), _r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::clip(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
        
        auto dispatch_clip = [](const at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.clip(min, max);
        };
        return wrap(dispatch_clip(_r.tensor(0), _r.scalarOptional(1), _r.scalarOptional(2)));
      } else {
        // aten::clip.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_clip_out = [](at::Tensor out, const at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::clip_out(out, self, min, max);
        };
        return wrap(dispatch_clip_out(_r.tensor(3), _r.tensor(0), _r.scalarOptional(1), _r.scalarOptional(2)));
      }
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
  static PythonArgParser parser({
    "clip_(Tensor input, Tensor? min=None, Tensor? max=None)",
    "clip_(Tensor input, Scalar? min=None, Scalar? max=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::clip_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)
      
      auto dispatch_clip_ = [](at::Tensor self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clip_(min, max);
      };
      return wrap(dispatch_clip_(_r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2)));
    }
    case 1: {
      // aten::clip_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)
      
      auto dispatch_clip_ = [](at::Tensor self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.clip_(min, max);
      };
      return wrap(dispatch_clip_(_r.tensor(0), _r.scalarOptional(1), _r.scalarOptional(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// complex
static PyObject * THPVariable_complex(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "complex(Tensor real, Tensor imag, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::complex(Tensor real, Tensor imag) -> Tensor
    
    auto dispatch_complex = [](const at::Tensor & real, const at::Tensor & imag) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::complex(real, imag);
    };
    return wrap(dispatch_complex(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::complex.out(Tensor real, Tensor imag, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_complex_out = [](at::Tensor out, const at::Tensor & real, const at::Tensor & imag) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::complex_out(out, real, imag);
    };
    return wrap(dispatch_complex_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// polar
static PyObject * THPVariable_polar(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "polar(Tensor abs, Tensor angle, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::polar(Tensor abs, Tensor angle) -> Tensor
    
    auto dispatch_polar = [](const at::Tensor & abs, const at::Tensor & angle) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::polar(abs, angle);
    };
    return wrap(dispatch_polar(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::polar.out(Tensor abs, Tensor angle, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_polar_out = [](at::Tensor out, const at::Tensor & abs, const at::Tensor & angle) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::polar_out(out, abs, angle);
    };
    return wrap(dispatch_polar_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _convolution
static PyObject * THPVariable__convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_convolution(Tensor input, Tensor weight, Tensor? bias, SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef dilation, bool transposed, IntArrayRef output_padding, SymInt groups, bool benchmark, bool deterministic, bool cudnn_enabled)",
    "_convolution(Tensor input, Tensor weight, Tensor? bias, SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef dilation, bool transposed, SymIntArrayRef output_padding, SymInt groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32)",
  }, /*traceable=*/true);

  ParsedArgs<13> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_convolution.deprecated(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, int[] output_padding, SymInt groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor
      
      auto dispatch__convolution = [](const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, c10::SymInt groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_convolution_symint(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
      };
      return wrap(dispatch__convolution(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.symintlist(4), _r.symintlist(5), _r.toBool(6), _r.intlist(7), _r.toSymInt(8), _r.toBool(9), _r.toBool(10), _r.toBool(11)));
    }
    case 1: {
      // aten::_convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor
      
      auto dispatch__convolution = [](const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_convolution_symint(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
      };
      return wrap(dispatch__convolution(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.symintlist(4), _r.symintlist(5), _r.toBool(6), _r.symintlist(7), _r.toSymInt(8), _r.toBool(9), _r.toBool(10), _r.toBool(11), _r.toBool(12)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _convolution_mode
static PyObject * THPVariable__convolution_mode(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_convolution_mode(Tensor input, Tensor weight, Tensor? bias, SymIntArrayRef stride, c10::string_view padding, SymIntArrayRef dilation, SymInt groups)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_convolution_mode(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, str padding, SymInt[] dilation, SymInt groups) -> Tensor
  
  auto dispatch__convolution_mode = [](const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::string_view padding, c10::SymIntArrayRef dilation, c10::SymInt groups) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_convolution_mode_symint(input, weight, bias, stride, padding, dilation, groups);
  };
  return wrap(dispatch__convolution_mode(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.stringView(4), _r.symintlist(5), _r.toSymInt(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// conv1d
static PyObject * THPVariable_conv1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv1d(Tensor input, Tensor weight, Tensor? bias=None, SymIntArrayRef[1] stride=1, SymIntArrayRef[1] padding=0, SymIntArrayRef[1] dilation=1, SymInt groups=1)",
    "conv1d(Tensor input, Tensor weight, Tensor? bias=None, SymIntArrayRef[1] stride=1, c10::string_view padding=\"valid\", SymIntArrayRef[1] dilation=1, SymInt groups=1)",
  }, /*traceable=*/false);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[1] stride=1, SymInt[1] padding=0, SymInt[1] dilation=1, SymInt groups=1) -> Tensor
      
      auto dispatch_conv1d = [](const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, c10::SymInt groups) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::conv1d_symint(input, weight, bias, stride, padding, dilation, groups);
      };
      return wrap(dispatch_conv1d(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.symintlist(4), _r.symintlist(5), _r.toSymInt(6)));
    }
    case 1: {
      // aten::conv1d.padding(Tensor input, Tensor weight, Tensor? bias=None, SymInt[1] stride=1, str padding="valid", SymInt[1] dilation=1, SymInt groups=1) -> Tensor
      
      auto dispatch_conv1d = [](const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::string_view padding, c10::SymIntArrayRef dilation, c10::SymInt groups) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::conv1d_symint(input, weight, bias, stride, padding, dilation, groups);
      };
      return wrap(dispatch_conv1d(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.stringView(4), _r.symintlist(5), _r.toSymInt(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// conv3d
static PyObject * THPVariable_conv3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv3d(Tensor input, Tensor weight, Tensor? bias=None, SymIntArrayRef[3] stride=1, SymIntArrayRef[3] padding=0, SymIntArrayRef[3] dilation=1, SymInt groups=1)",
    "conv3d(Tensor input, Tensor weight, Tensor? bias=None, SymIntArrayRef[3] stride=1, c10::string_view padding=\"valid\", SymIntArrayRef[3] dilation=1, SymInt groups=1)",
  }, /*traceable=*/false);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[3] stride=1, SymInt[3] padding=0, SymInt[3] dilation=1, SymInt groups=1) -> Tensor
      
      auto dispatch_conv3d = [](const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, c10::SymInt groups) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::conv3d_symint(input, weight, bias, stride, padding, dilation, groups);
      };
      return wrap(dispatch_conv3d(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.symintlist(4), _r.symintlist(5), _r.toSymInt(6)));
    }
    case 1: {
      // aten::conv3d.padding(Tensor input, Tensor weight, Tensor? bias=None, SymInt[3] stride=1, str padding="valid", SymInt[3] dilation=1, SymInt groups=1) -> Tensor
      
      auto dispatch_conv3d = [](const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::string_view padding, c10::SymIntArrayRef dilation, c10::SymInt groups) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::conv3d_symint(input, weight, bias, stride, padding, dilation, groups);
      };
      return wrap(dispatch_conv3d(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.stringView(4), _r.symintlist(5), _r.toSymInt(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cos
static PyObject * THPVariable_cos(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cos(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::cos(Tensor self) -> Tensor
    
    auto dispatch_cos = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.cos();
    };
    return wrap(dispatch_cos(_r.tensor(0)));
  } else {
    // aten::cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_cos_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::cos_out(out, self);
    };
    return wrap(dispatch_cos_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cos_
static PyObject * THPVariable_cos_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cos_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::cos_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_cos_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.cos_();
  };
  return wrap(dispatch_cos_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// count_nonzero
static PyObject * THPVariable_count_nonzero(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "count_nonzero(Tensor input, int64_t? dim=None)",
    "count_nonzero(Tensor input, IntArrayRef dim)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::count_nonzero(Tensor self, int? dim=None) -> Tensor
      
      auto dispatch_count_nonzero = [](const at::Tensor & self, ::std::optional<int64_t> dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.count_nonzero(dim);
      };
      return wrap(dispatch_count_nonzero(_r.tensor(0), _r.toInt64Optional(1)));
    }
    case 1: {
      // aten::count_nonzero.dim_IntList(Tensor self, int[] dim) -> Tensor
      
      auto dispatch_count_nonzero = [](const at::Tensor & self, at::IntArrayRef dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.count_nonzero(dim);
      };
      return wrap(dispatch_count_nonzero(_r.tensor(0), _r.intlist(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// corrcoef
static PyObject * THPVariable_corrcoef(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "corrcoef(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::corrcoef(Tensor self) -> Tensor
  
  auto dispatch_corrcoef = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.corrcoef();
  };
  return wrap(dispatch_corrcoef(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cudnn_convolution_transpose
static PyObject * THPVariable_cudnn_convolution_transpose(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_convolution_transpose(Tensor input, Tensor weight, SymIntArrayRef padding, SymIntArrayRef output_padding, SymIntArrayRef stride, SymIntArrayRef dilation, SymInt groups, bool benchmark, bool deterministic, bool allow_tf32)",
  }, /*traceable=*/true);

  ParsedArgs<10> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::cudnn_convolution_transpose(Tensor self, Tensor weight, SymInt[] padding, SymInt[] output_padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
  
  auto dispatch_cudnn_convolution_transpose = [](const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, bool benchmark, bool deterministic, bool allow_tf32) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::cudnn_convolution_transpose_symint(self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  };
  return wrap(dispatch_cudnn_convolution_transpose(_r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.symintlist(3), _r.symintlist(4), _r.symintlist(5), _r.toSymInt(6), _r.toBool(7), _r.toBool(8), _r.toBool(9)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cudnn_convolution_relu
static PyObject * THPVariable_cudnn_convolution_relu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cudnn_convolution_relu(Tensor input, Tensor weight, Tensor? bias, SymIntArrayRef stride, SymIntArrayRef padding, SymIntArrayRef dilation, SymInt groups)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::cudnn_convolution_relu(Tensor self, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, SymInt groups) -> Tensor
  
  auto dispatch_cudnn_convolution_relu = [](const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, c10::SymInt groups) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::cudnn_convolution_relu_symint(self, weight, bias, stride, padding, dilation, groups);
  };
  return wrap(dispatch_cudnn_convolution_relu(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.symintlist(4), _r.symintlist(5), _r.toSymInt(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// ctc_loss
static PyObject * THPVariable_ctc_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ctc_loss(Tensor log_probs, Tensor targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank=0, int64_t reduction=at::Reduction::Mean, bool zero_infinity=False)",
    "ctc_loss(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int64_t blank=0, int64_t reduction=at::Reduction::Mean, bool zero_infinity=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::ctc_loss.IntList(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor
      
      auto dispatch_ctc_loss = [](const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
      };
      return wrap(dispatch_ctc_loss(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.intlist(3), _r.toInt64(4), _r.toInt64(5), _r.toBool(6)));
    }
    case 1: {
      // aten::ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor
      
      auto dispatch_ctc_loss = [](const at::Tensor & log_probs, const at::Tensor & targets, const at::Tensor & input_lengths, const at::Tensor & target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
      };
      return wrap(dispatch_ctc_loss(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.toInt64(4), _r.toInt64(5), _r.toBool(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// embedding_bag
static PyObject * THPVariable_embedding_bag(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, Tensor? per_sample_weights, bool include_last_offset, int64_t? padding_idx)",
    "embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int64_t mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::embedding_bag.padding_idx(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, bool include_last_offset, int? padding_idx) -> (Tensor, Tensor, Tensor, Tensor)
      
      auto dispatch_embedding_bag = [](const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const ::std::optional<at::Tensor> & per_sample_weights, bool include_last_offset, ::std::optional<int64_t> padding_idx) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
      };
      return wrap(dispatch_embedding_bag(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toBool(3), _r.toInt64(4), _r.toBool(5), _r.optionalTensor(6), _r.toBool(7), _r.toInt64Optional(8)));
    }
    case 1: {
      // aten::embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)
      
      auto dispatch_embedding_bag = [](const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const ::std::optional<at::Tensor> & per_sample_weights, bool include_last_offset) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
      };
      return wrap(dispatch_embedding_bag(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toBool(3), _r.toInt64(4), _r.toBool(5), _r.optionalTensor(6), _r.toBool(7)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _embedding_bag
static PyObject * THPVariable__embedding_bag(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int64_t mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int64_t padding_idx=-1)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
  
  auto dispatch__embedding_bag = [](const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const ::std::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
  };
  return wrap(dispatch__embedding_bag(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toBool(3), _r.toInt64(4), _r.toBool(5), _r.optionalTensor(6), _r.toBool(7), _r.toInt64(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// empty
static PyObject * THPVariable_empty(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "empty(IntArrayRef size, *, DimnameList? names, MemoryFormat? memory_format=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "empty(SymIntArrayRef size, *, MemoryFormat? memory_format=None, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::empty.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
      auto __names = _r.toDimnameListOptional(1);
      ::std::optional<DimnameList> names = __names ? ::std::make_optional(DimnameList(__names.value())) : ::std::nullopt;
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(3))
          .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(4))
          .requires_grad(_r.toBool(7))
          .pinned_memory(_r.toBool(6));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_empty = [](at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::TensorOptions options, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::empty(size, names, options, memory_format);
      };
      return wrap(dispatch_empty(_r.intlist(0), names, options, _r.memoryformatOptional(2)));
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(3))
            .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(4))
            .requires_grad(_r.toBool(7))
            .pinned_memory(_r.toBool(6));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_empty = [](c10::SymIntArrayRef size, at::TensorOptions options, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::empty_symint(size, options, memory_format);
        };
        return wrap(dispatch_empty(_r.symintlist(0), options, _r.memoryformatOptional(1)));
      } else {
        // aten::empty.out(SymInt[] size, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(2), _r.scalartypeOptional(3),
                               _r.isNone(3), _r.layoutOptional(4),
                               _r.deviceWithDefault(5, torch::tensors::get_default_device()), _r.isNone(5));
        
        auto dispatch_empty_out = [](at::Tensor out, c10::SymIntArrayRef size, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::empty_symint_out(out, size, memory_format);
        };
        return wrap(dispatch_empty_out(_r.tensor(2), _r.symintlist(0), _r.memoryformatOptional(1)).set_requires_grad(_r.toBool(7)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// empty_like
static PyObject * THPVariable_empty_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "empty_like(Tensor input, *, MemoryFormat? memory_format=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
  auto self = _r.tensor(0);
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(2))
      .device(_r.deviceOptional(4))
      .layout(_r.layoutOptional(3))
      .requires_grad(_r.toBool(6))
      .pinned_memory(_r.toBool(5));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch_empty_like = [](const at::Tensor & self, at::TensorOptions options, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::empty_like(self, options, memory_format);
  };
  return wrap(dispatch_empty_like(self, options, _r.memoryformatOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// erf
static PyObject * THPVariable_erf(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "erf(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::erf(Tensor self) -> Tensor
    
    auto dispatch_erf = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.erf();
    };
    return wrap(dispatch_erf(_r.tensor(0)));
  } else {
    // aten::erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_erf_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::erf_out(out, self);
    };
    return wrap(dispatch_erf_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// erf_
static PyObject * THPVariable_erf_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "erf_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::erf_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_erf_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.erf_();
  };
  return wrap(dispatch_erf_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// erfc
static PyObject * THPVariable_erfc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "erfc(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::erfc(Tensor self) -> Tensor
    
    auto dispatch_erfc = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.erfc();
    };
    return wrap(dispatch_erfc(_r.tensor(0)));
  } else {
    // aten::erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_erfc_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::erfc_out(out, self);
    };
    return wrap(dispatch_erfc_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// erfc_
static PyObject * THPVariable_erfc_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "erfc_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::erfc_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_erfc_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.erfc_();
  };
  return wrap(dispatch_erfc_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// exp
static PyObject * THPVariable_exp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "exp(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::exp(Tensor self) -> Tensor
    
    auto dispatch_exp = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.exp();
    };
    return wrap(dispatch_exp(_r.tensor(0)));
  } else {
    // aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_exp_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::exp_out(out, self);
    };
    return wrap(dispatch_exp_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// exp_
static PyObject * THPVariable_exp_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "exp_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::exp_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_exp_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.exp_();
  };
  return wrap(dispatch_exp_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// unflatten
static PyObject * THPVariable_unflatten(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unflatten(Tensor input, Dimname dim, SymIntArrayRef sizes, DimnameList names)",
    "unflatten(Tensor input, int64_t dim, SymIntArrayRef sizes)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::unflatten.Dimname(Tensor(a) self, Dimname dim, SymInt[] sizes, Dimname[] names) -> Tensor(a)
      
      auto dispatch_unflatten = [](const at::Tensor & self, at::Dimname dim, c10::SymIntArrayRef sizes, at::DimnameList names) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.unflatten_symint(dim, sizes, names);
      };
      return wrap(dispatch_unflatten(_r.tensor(0), _r.dimname(1), _r.symintlist(2), _r.dimnamelist(3)));
    }
    case 1: {
      // aten::unflatten.int(Tensor(a) self, int dim, SymInt[] sizes) -> Tensor(a)
      
      auto dispatch_unflatten = [](const at::Tensor & self, int64_t dim, c10::SymIntArrayRef sizes) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.unflatten_symint(dim, sizes);
      };
      return wrap(dispatch_unflatten(_r.tensor(0), _r.toInt64(1), _r.symintlist(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// floor
static PyObject * THPVariable_floor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "floor(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::floor(Tensor self) -> Tensor
    
    auto dispatch_floor = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.floor();
    };
    return wrap(dispatch_floor(_r.tensor(0)));
  } else {
    // aten::floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_floor_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::floor_out(out, self);
    };
    return wrap(dispatch_floor_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// floor_
static PyObject * THPVariable_floor_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "floor_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::floor_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_floor_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.floor_();
  };
  return wrap(dispatch_floor_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// floor_divide
static PyObject * THPVariable_floor_divide(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "floor_divide(Tensor input, Tensor other, *, Tensor out=None)",
    "floor_divide(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::floor_divide(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_floor_divide = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.floor_divide(other);
        };
        return wrap(dispatch_floor_divide(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::floor_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_floor_divide_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::floor_divide_out(out, self, other);
        };
        return wrap(dispatch_floor_divide_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      // aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch_floor_divide = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.floor_divide(other);
      };
      return wrap(dispatch_floor_divide(_r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// frac
static PyObject * THPVariable_frac(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "frac(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::frac(Tensor self) -> Tensor
    
    auto dispatch_frac = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.frac();
    };
    return wrap(dispatch_frac(_r.tensor(0)));
  } else {
    // aten::frac.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_frac_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::frac_out(out, self);
    };
    return wrap(dispatch_frac_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// frac_
static PyObject * THPVariable_frac_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "frac_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::frac_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_frac_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.frac_();
  };
  return wrap(dispatch_frac_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// full_like
static PyObject * THPVariable_full_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "full_like(Tensor input, Scalar fill_value, *, MemoryFormat? memory_format=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
  auto self = _r.tensor(0);
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(3))
      .device(_r.deviceOptional(5))
      .layout(_r.layoutOptional(4))
      .requires_grad(_r.toBool(7))
      .pinned_memory(_r.toBool(6));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch_full_like = [](const at::Tensor & self, const at::Scalar & fill_value, at::TensorOptions options, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::full_like(self, fill_value, options, memory_format);
  };
  return wrap(dispatch_full_like(self, _r.scalar(1), options, _r.memoryformatOptional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// grid_sampler
static PyObject * THPVariable_grid_sampler(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "grid_sampler(Tensor input, Tensor grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::grid_sampler(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
  
  auto dispatch_grid_sampler = [](const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::grid_sampler(input, grid, interpolation_mode, padding_mode, align_corners);
  };
  return wrap(dispatch_grid_sampler(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toInt64(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _grid_sampler_2d_cpu_fallback
static PyObject * THPVariable__grid_sampler_2d_cpu_fallback(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_grid_sampler_2d_cpu_fallback(Tensor input, Tensor grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_grid_sampler_2d_cpu_fallback(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
  
  auto dispatch__grid_sampler_2d_cpu_fallback = [](const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_grid_sampler_2d_cpu_fallback(input, grid, interpolation_mode, padding_mode, align_corners);
  };
  return wrap(dispatch__grid_sampler_2d_cpu_fallback(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toInt64(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// grid_sampler_3d
static PyObject * THPVariable_grid_sampler_3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "grid_sampler_3d(Tensor input, Tensor grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::grid_sampler_3d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
  
  auto dispatch_grid_sampler_3d = [](const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners);
  };
  return wrap(dispatch_grid_sampler_3d(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toInt64(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _fft_c2r
static PyObject * THPVariable__fft_c2r(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_fft_c2r(Tensor input, IntArrayRef dim, int64_t normalization, SymInt last_dim_size, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::_fft_c2r(Tensor self, int[] dim, int normalization, SymInt last_dim_size) -> Tensor
    
    auto dispatch__fft_c2r = [](const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, c10::SymInt last_dim_size) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_fft_c2r_symint(self, dim, normalization, last_dim_size);
    };
    return wrap(dispatch__fft_c2r(_r.tensor(0), _r.intlist(1), _r.toInt64(2), _r.toSymInt(3)));
  } else {
    // aten::_fft_c2r.out(Tensor self, int[] dim, int normalization, SymInt last_dim_size, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__fft_c2r_out = [](at::Tensor out, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, c10::SymInt last_dim_size) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_fft_c2r_symint_out(out, self, dim, normalization, last_dim_size);
    };
    return wrap(dispatch__fft_c2r_out(_r.tensor(4), _r.tensor(0), _r.intlist(1), _r.toInt64(2), _r.toSymInt(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cufft_get_plan_cache_max_size
static PyObject * THPVariable__cufft_get_plan_cache_max_size(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cufft_get_plan_cache_max_size(DeviceIndex device_index)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cufft_get_plan_cache_max_size(DeviceIndex device_index) -> int
  
  auto dispatch__cufft_get_plan_cache_max_size = [](at::DeviceIndex device_index) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return at::_cufft_get_plan_cache_max_size(device_index);
  };
  return wrap(dispatch__cufft_get_plan_cache_max_size(_r.toInt64(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cufft_clear_plan_cache
static PyObject * THPVariable__cufft_clear_plan_cache(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cufft_clear_plan_cache(DeviceIndex device_index)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cufft_clear_plan_cache(DeviceIndex device_index) -> ()
  
  auto dispatch__cufft_clear_plan_cache = [](at::DeviceIndex device_index) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_cufft_clear_plan_cache(device_index);
  };
  dispatch__cufft_clear_plan_cache(_r.toInt64(0));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _unsafe_masked_index_put_accumulate
static PyObject * THPVariable__unsafe_masked_index_put_accumulate(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_unsafe_masked_index_put_accumulate(Tensor input, Tensor mask, c10::List<::std::optional<Tensor>> indices, Tensor values)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_unsafe_masked_index_put_accumulate(Tensor self, Tensor mask, Tensor?[] indices, Tensor values) -> Tensor
  
  auto dispatch__unsafe_masked_index_put_accumulate = [](const at::Tensor & self, const at::Tensor & mask, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_unsafe_masked_index_put_accumulate(self, mask, indices, values);
  };
  return wrap(dispatch__unsafe_masked_index_put_accumulate(_r.tensor(0), _r.tensor(1), _r.list_of_optional_tensors(2), _r.tensor(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _index_put_impl_
static PyObject * THPVariable__index_put_impl_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_index_put_impl_(Tensor input, c10::List<::std::optional<Tensor>> indices, Tensor values, bool accumulate=False, bool unsafe=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_index_put_impl_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)
  
  auto dispatch__index_put_impl_ = [](at::Tensor self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_index_put_impl_(self, indices, values, accumulate, unsafe);
  };
  return wrap(dispatch__index_put_impl_(_r.tensor(0), _r.list_of_optional_tensors(1), _r.tensor(2), _r.toBool(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// isin
static PyObject * THPVariable_isin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "isin(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False, Tensor out=None)",
    "isin(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False, Tensor out=None)",
    "isin(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // aten::isin.Tensor_Tensor(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor
        
        auto dispatch_isin = [](const at::Tensor & elements, const at::Tensor & test_elements, bool assume_unique, bool invert) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::isin(elements, test_elements, assume_unique, invert);
        };
        return wrap(dispatch_isin(_r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3)));
      } else {
        // aten::isin.Tensor_Tensor_out(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_isin_out = [](at::Tensor out, const at::Tensor & elements, const at::Tensor & test_elements, bool assume_unique, bool invert) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::isin_out(out, elements, test_elements, assume_unique, invert);
        };
        return wrap(dispatch_isin_out(_r.tensor(4), _r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::isin.Scalar_Tensor(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor
        
        auto dispatch_isin = [](const at::Scalar & element, const at::Tensor & test_elements, bool assume_unique, bool invert) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::isin(element, test_elements, assume_unique, invert);
        };
        return wrap(dispatch_isin(_r.scalar(0), _r.tensor(1), _r.toBool(2), _r.toBool(3)));
      } else {
        // aten::isin.Scalar_Tensor_out(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_isin_out = [](at::Tensor out, const at::Scalar & element, const at::Tensor & test_elements, bool assume_unique, bool invert) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::isin_out(out, element, test_elements, assume_unique, invert);
        };
        return wrap(dispatch_isin_out(_r.tensor(4), _r.scalar(0), _r.tensor(1), _r.toBool(2), _r.toBool(3)));
      }
    }
    case 2: {
      if (_r.isNone(4)) {
        // aten::isin.Tensor_Scalar(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False) -> Tensor
        
        auto dispatch_isin = [](const at::Tensor & elements, const at::Scalar & test_element, bool assume_unique, bool invert) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::isin(elements, test_element, assume_unique, invert);
        };
        return wrap(dispatch_isin(_r.tensor(0), _r.scalar(1), _r.toBool(2), _r.toBool(3)));
      } else {
        // aten::isin.Tensor_Scalar_out(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_isin_out = [](at::Tensor out, const at::Tensor & elements, const at::Scalar & test_element, bool assume_unique, bool invert) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::isin_out(out, elements, test_element, assume_unique, invert);
        };
        return wrap(dispatch_isin_out(_r.tensor(4), _r.tensor(0), _r.scalar(1), _r.toBool(2), _r.toBool(3)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// isnan
static PyObject * THPVariable_isnan(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "isnan(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::isnan(Tensor self) -> Tensor
  
  auto dispatch_isnan = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.isnan();
  };
  return wrap(dispatch_isnan(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_conj
static PyObject * THPVariable_is_conj(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_conj(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::is_conj(Tensor self) -> bool
  
  auto dispatch_is_conj = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_conj();
  };
  return wrap(dispatch_is_conj(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// is_inference
static PyObject * THPVariable_is_inference(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "is_inference(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::is_inference(Tensor self) -> bool
  
  auto dispatch_is_inference = [](const at::Tensor & self) -> bool {
    pybind11::gil_scoped_release no_gil;
    return self.is_inference();
  };
  return wrap(dispatch_is_inference(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// layer_norm
static PyObject * THPVariable_layer_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "layer_norm(Tensor input, SymIntArrayRef normalized_shape, Tensor? weight=None, Tensor? bias=None, double eps=1e-05, bool cudnn_enable=True)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor
  
  auto dispatch_layer_norm = [](const at::Tensor & input, c10::SymIntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps, bool cudnn_enable) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::layer_norm_symint(input, normalized_shape, weight, bias, eps, cudnn_enable);
  };
  return wrap(dispatch_layer_norm(_r.tensor(0), _r.symintlist(1), _r.optionalTensor(2), _r.optionalTensor(3), _r.toDouble(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// nan_to_num
static PyObject * THPVariable_nan_to_num(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nan_to_num(Tensor input, double? nan=None, double? posinf=None, double? neginf=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor
    
    auto dispatch_nan_to_num = [](const at::Tensor & self, ::std::optional<double> nan, ::std::optional<double> posinf, ::std::optional<double> neginf) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.nan_to_num(nan, posinf, neginf);
    };
    return wrap(dispatch_nan_to_num(_r.tensor(0), _r.toDoubleOptional(1), _r.toDoubleOptional(2), _r.toDoubleOptional(3)));
  } else {
    // aten::nan_to_num.out(Tensor self, float? nan=None, float? posinf=None, float? neginf=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_nan_to_num_out = [](at::Tensor out, const at::Tensor & self, ::std::optional<double> nan, ::std::optional<double> posinf, ::std::optional<double> neginf) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::nan_to_num_out(out, self, nan, posinf, neginf);
    };
    return wrap(dispatch_nan_to_num_out(_r.tensor(4), _r.tensor(0), _r.toDoubleOptional(1), _r.toDoubleOptional(2), _r.toDoubleOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// nan_to_num_
static PyObject * THPVariable_nan_to_num_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nan_to_num_(Tensor input, double? nan=None, double? posinf=None, double? neginf=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::nan_to_num_(Tensor(a!) self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor(a!)
  
  auto dispatch_nan_to_num_ = [](at::Tensor self, ::std::optional<double> nan, ::std::optional<double> posinf, ::std::optional<double> neginf) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.nan_to_num_(nan, posinf, neginf);
  };
  return wrap(dispatch_nan_to_num_(_r.tensor(0), _r.toDoubleOptional(1), _r.toDoubleOptional(2), _r.toDoubleOptional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mkldnn_linear_backward_weights
static PyObject * THPVariable_mkldnn_linear_backward_weights(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mkldnn_linear_backward_weights(Tensor grad_output, Tensor input, Tensor weight, bool bias_defined)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::mkldnn_linear_backward_weights(Tensor grad_output, Tensor input, Tensor weight, bool bias_defined) -> (Tensor, Tensor)
  
  auto dispatch_mkldnn_linear_backward_weights = [](const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, bool bias_defined) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::mkldnn_linear_backward_weights(grad_output, input, weight, bias_defined);
  };
  return wrap(dispatch_mkldnn_linear_backward_weights(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toBool(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _cslt_sparse_mm
static PyObject * THPVariable__cslt_sparse_mm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_cslt_sparse_mm(Tensor compressed_A, Tensor dense_B, Tensor? bias=None, Tensor? alpha=None, ScalarType? out_dtype=None, bool transpose_result=False, int64_t alg_id=0, int64_t split_k=1, bool split_k_one_kernel=True)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_cslt_sparse_mm(Tensor compressed_A, Tensor dense_B, Tensor? bias=None, Tensor? alpha=None, ScalarType? out_dtype=None, bool transpose_result=False, int alg_id=0, int split_k=1, bool split_k_one_kernel=True) -> Tensor
  
  auto dispatch__cslt_sparse_mm = [](const at::Tensor & compressed_A, const at::Tensor & dense_B, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & alpha, ::std::optional<at::ScalarType> out_dtype, bool transpose_result, int64_t alg_id, int64_t split_k, bool split_k_one_kernel) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_cslt_sparse_mm(compressed_A, dense_B, bias, alpha, out_dtype, transpose_result, alg_id, split_k, split_k_one_kernel);
  };
  return wrap(dispatch__cslt_sparse_mm(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.optionalTensor(3), _r.scalartypeOptional(4), _r.toBool(5), _r.toInt64(6), _r.toInt64(7), _r.toBool(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sparse_semi_structured_mm
static PyObject * THPVariable__sparse_semi_structured_mm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_semi_structured_mm(Tensor mat1, Tensor mat1_meta, Tensor mat2, *, ScalarType? out_dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_sparse_semi_structured_mm(Tensor mat1, Tensor mat1_meta, Tensor mat2, *, ScalarType? out_dtype=None) -> Tensor
  
  auto dispatch__sparse_semi_structured_mm = [](const at::Tensor & mat1, const at::Tensor & mat1_meta, const at::Tensor & mat2, ::std::optional<at::ScalarType> out_dtype) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sparse_semi_structured_mm(mat1, mat1_meta, mat2, out_dtype);
  };
  return wrap(dispatch__sparse_semi_structured_mm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalartypeOptional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sparse_semi_structured_addmm
static PyObject * THPVariable__sparse_semi_structured_addmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_semi_structured_addmm(Tensor input, Tensor mat1, Tensor mat1_meta, Tensor mat2, *, Scalar alpha=1, Scalar beta=1, ScalarType? out_dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_sparse_semi_structured_addmm(Tensor input, Tensor mat1, Tensor mat1_meta, Tensor mat2, *, Scalar alpha=1, Scalar beta=1, ScalarType? out_dtype=None) -> Tensor
  
  auto dispatch__sparse_semi_structured_addmm = [](const at::Tensor & input, const at::Tensor & mat1, const at::Tensor & mat1_meta, const at::Tensor & mat2, const at::Scalar & alpha, const at::Scalar & beta, ::std::optional<at::ScalarType> out_dtype) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sparse_semi_structured_addmm(input, mat1, mat1_meta, mat2, alpha, beta, out_dtype);
  };
  return wrap(dispatch__sparse_semi_structured_addmm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.scalar(4), _r.scalar(5), _r.scalartypeOptional(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _mixed_dtypes_linear
static PyObject * THPVariable__mixed_dtypes_linear(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_mixed_dtypes_linear(Tensor input, Tensor weight, Tensor scale, *, Tensor? bias=None, c10::string_view? activation=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_mixed_dtypes_linear(Tensor input, Tensor weight, Tensor scale, *, Tensor? bias=None, str? activation=None) -> Tensor
  
  auto dispatch__mixed_dtypes_linear = [](const at::Tensor & input, const at::Tensor & weight, const at::Tensor & scale, const ::std::optional<at::Tensor> & bias, ::std::optional<c10::string_view> activation) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_mixed_dtypes_linear(input, weight, scale, bias, activation);
  };
  return wrap(dispatch__mixed_dtypes_linear(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.optionalTensor(3), _r.stringViewOptional(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fbgemm_pack_gemm_matrix_fp16
static PyObject * THPVariable_fbgemm_pack_gemm_matrix_fp16(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fbgemm_pack_gemm_matrix_fp16(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::fbgemm_pack_gemm_matrix_fp16(Tensor input) -> Tensor
  
  auto dispatch_fbgemm_pack_gemm_matrix_fp16 = [](const at::Tensor & input) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::fbgemm_pack_gemm_matrix_fp16(input);
  };
  return wrap(dispatch_fbgemm_pack_gemm_matrix_fp16(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _wrapped_quantized_linear_prepacked
static PyObject * THPVariable__wrapped_quantized_linear_prepacked(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_wrapped_quantized_linear_prepacked(Tensor input, Tensor input_scale, Tensor input_zero_point, Tensor packed_weight, Tensor output_scale, Tensor output_zero_point, int64_t out_channel)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_wrapped_quantized_linear_prepacked(Tensor input, Tensor input_scale, Tensor input_zero_point, Tensor packed_weight, Tensor output_scale, Tensor output_zero_point, int out_channel) -> Tensor
  
  auto dispatch__wrapped_quantized_linear_prepacked = [](const at::Tensor & input, const at::Tensor & input_scale, const at::Tensor & input_zero_point, const at::Tensor & packed_weight, const at::Tensor & output_scale, const at::Tensor & output_zero_point, int64_t out_channel) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_wrapped_quantized_linear_prepacked(input, input_scale, input_zero_point, packed_weight, output_scale, output_zero_point, out_channel);
  };
  return wrap(dispatch__wrapped_quantized_linear_prepacked(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.tensor(5), _r.toInt64(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fbgemm_linear_fp16_weight_fp32_activation
static PyObject * THPVariable_fbgemm_linear_fp16_weight_fp32_activation(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fbgemm_linear_fp16_weight_fp32_activation(Tensor input, Tensor packed_weight, Tensor bias)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::fbgemm_linear_fp16_weight_fp32_activation(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor
  
  auto dispatch_fbgemm_linear_fp16_weight_fp32_activation = [](const at::Tensor & input, const at::Tensor & packed_weight, const at::Tensor & bias) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::fbgemm_linear_fp16_weight_fp32_activation(input, packed_weight, bias);
  };
  return wrap(dispatch_fbgemm_linear_fp16_weight_fp32_activation(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// linspace
static PyObject * THPVariable_linspace(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linspace(Tensor start, Tensor end, int64_t steps, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "linspace(Scalar start, Tensor end, int64_t steps, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "linspace(Tensor start, Scalar end, int64_t steps, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "linspace(Scalar start, Scalar end, int64_t steps, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::linspace.Tensor_Tensor(Tensor start, Tensor end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(4))
            .device(_r.deviceWithDefault(6, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(5))
            .requires_grad(_r.toBool(8))
            .pinned_memory(_r.toBool(7));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_linspace = [](const at::Tensor & start, const at::Tensor & end, int64_t steps, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::linspace(start, end, steps, options);
        };
        return wrap(dispatch_linspace(_r.tensor(0), _r.tensor(1), _r.toInt64(2), options));
      } else {
        // aten::linspace.Tensor_Tensor_out(Tensor start, Tensor end, int steps, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(3), _r.scalartypeOptional(4),
                               _r.isNone(4), _r.layoutOptional(5),
                               _r.deviceWithDefault(6, torch::tensors::get_default_device()), _r.isNone(6));
        
        auto dispatch_linspace_out = [](at::Tensor out, const at::Tensor & start, const at::Tensor & end, int64_t steps) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linspace_out(out, start, end, steps);
        };
        return wrap(dispatch_linspace_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toInt64(2)).set_requires_grad(_r.toBool(8)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::linspace.Scalar_Tensor(Scalar start, Tensor end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(4))
            .device(_r.deviceWithDefault(6, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(5))
            .requires_grad(_r.toBool(8))
            .pinned_memory(_r.toBool(7));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_linspace = [](const at::Scalar & start, const at::Tensor & end, int64_t steps, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::linspace(start, end, steps, options);
        };
        return wrap(dispatch_linspace(_r.scalar(0), _r.tensor(1), _r.toInt64(2), options));
      } else {
        // aten::linspace.Scalar_Tensor_out(Scalar start, Tensor end, int steps, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(3), _r.scalartypeOptional(4),
                               _r.isNone(4), _r.layoutOptional(5),
                               _r.deviceWithDefault(6, torch::tensors::get_default_device()), _r.isNone(6));
        
        auto dispatch_linspace_out = [](at::Tensor out, const at::Scalar & start, const at::Tensor & end, int64_t steps) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linspace_out(out, start, end, steps);
        };
        return wrap(dispatch_linspace_out(_r.tensor(3), _r.scalar(0), _r.tensor(1), _r.toInt64(2)).set_requires_grad(_r.toBool(8)));
      }
    }
    case 2: {
      if (_r.isNone(3)) {
        // aten::linspace.Tensor_Scalar(Tensor start, Scalar end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(4))
            .device(_r.deviceWithDefault(6, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(5))
            .requires_grad(_r.toBool(8))
            .pinned_memory(_r.toBool(7));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_linspace = [](const at::Tensor & start, const at::Scalar & end, int64_t steps, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::linspace(start, end, steps, options);
        };
        return wrap(dispatch_linspace(_r.tensor(0), _r.scalar(1), _r.toInt64(2), options));
      } else {
        // aten::linspace.Tensor_Scalar_out(Tensor start, Scalar end, int steps, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(3), _r.scalartypeOptional(4),
                               _r.isNone(4), _r.layoutOptional(5),
                               _r.deviceWithDefault(6, torch::tensors::get_default_device()), _r.isNone(6));
        
        auto dispatch_linspace_out = [](at::Tensor out, const at::Tensor & start, const at::Scalar & end, int64_t steps) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linspace_out(out, start, end, steps);
        };
        return wrap(dispatch_linspace_out(_r.tensor(3), _r.tensor(0), _r.scalar(1), _r.toInt64(2)).set_requires_grad(_r.toBool(8)));
      }
    }
    case 3: {
      if (_r.isNone(3)) {
        // aten::linspace(Scalar start, Scalar end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(4))
            .device(_r.deviceWithDefault(6, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(5))
            .requires_grad(_r.toBool(8))
            .pinned_memory(_r.toBool(7));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_linspace = [](const at::Scalar & start, const at::Scalar & end, int64_t steps, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::linspace(start, end, steps, options);
        };
        return wrap(dispatch_linspace(_r.scalar(0), _r.scalar(1), _r.toInt64(2), options));
      } else {
        // aten::linspace.out(Scalar start, Scalar end, int steps, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(3), _r.scalartypeOptional(4),
                               _r.isNone(4), _r.layoutOptional(5),
                               _r.deviceWithDefault(6, torch::tensors::get_default_device()), _r.isNone(6));
        
        auto dispatch_linspace_out = [](at::Tensor out, const at::Scalar & start, const at::Scalar & end, int64_t steps) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linspace_out(out, start, end, steps);
        };
        return wrap(dispatch_linspace_out(_r.tensor(3), _r.scalar(0), _r.scalar(1), _r.toInt64(2)).set_requires_grad(_r.toBool(8)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// log
static PyObject * THPVariable_log(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::log(Tensor self) -> Tensor
    
    auto dispatch_log = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.log();
    };
    return wrap(dispatch_log(_r.tensor(0)));
  } else {
    // aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_log_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::log_out(out, self);
    };
    return wrap(dispatch_log_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// log_
static PyObject * THPVariable_log_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::log_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_log_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log_();
  };
  return wrap(dispatch_log_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// log1p
static PyObject * THPVariable_log1p(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log1p(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::log1p(Tensor self) -> Tensor
    
    auto dispatch_log1p = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.log1p();
    };
    return wrap(dispatch_log1p(_r.tensor(0)));
  } else {
    // aten::log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_log1p_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::log1p_out(out, self);
    };
    return wrap(dispatch_log1p_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// log1p_
static PyObject * THPVariable_log1p_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log1p_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::log1p_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_log1p_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.log1p_();
  };
  return wrap(dispatch_log1p_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logaddexp
static PyObject * THPVariable_logaddexp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logaddexp(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::logaddexp(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_logaddexp = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.logaddexp(other);
    };
    return wrap(dispatch_logaddexp(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::logaddexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_logaddexp_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::logaddexp_out(out, self, other);
    };
    return wrap(dispatch_logaddexp_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// xlogy
static PyObject * THPVariable_xlogy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "xlogy(Tensor input, Tensor other, *, Tensor out=None)",
    "xlogy(Scalar self, Tensor other, *, Tensor out=None)",
    "xlogy(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::xlogy.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_xlogy = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.xlogy(other);
        };
        return wrap(dispatch_xlogy(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::xlogy.OutTensor(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_xlogy_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::xlogy_out(out, self, other);
        };
        return wrap(dispatch_xlogy_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::xlogy.Scalar_Self(Scalar self, Tensor other) -> Tensor
        
        auto dispatch_xlogy = [](const at::Scalar & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::xlogy(self, other);
        };
        return wrap(dispatch_xlogy(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::xlogy.OutScalar_Self(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_xlogy_out = [](at::Tensor out, const at::Scalar & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::xlogy_out(out, self, other);
        };
        return wrap(dispatch_xlogy_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::xlogy.Scalar_Other(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_xlogy = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.xlogy(other);
        };
        return wrap(dispatch_xlogy(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::xlogy.OutScalar_Other(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_xlogy_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::xlogy_out(out, self, other);
        };
        return wrap(dispatch_xlogy_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
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
  static PythonArgParser parser({
    "xlogy_(Tensor input, Tensor other)",
    "xlogy_(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::xlogy_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
      
      auto dispatch_xlogy_ = [](at::Tensor self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.xlogy_(other);
      };
      return wrap(dispatch_xlogy_(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::xlogy_.Scalar_Other(Tensor(a!) self, Scalar other) -> Tensor(a!)
      
      auto dispatch_xlogy_ = [](at::Tensor self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.xlogy_(other);
      };
      return wrap(dispatch_xlogy_(_r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// logsumexp
static PyObject * THPVariable_logsumexp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logsumexp(Tensor input, IntArrayRef[1] dim, bool keepdim=False, *, Tensor out=None)",
    "logsumexp(Tensor input, DimnameList[1] dim, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
        
        auto dispatch_logsumexp = [](const at::Tensor & self, at::IntArrayRef dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.logsumexp(dim, keepdim);
        };
        return wrap(dispatch_logsumexp(_r.tensor(0), _r.intlist(1), _r.toBool(2)));
      } else {
        // aten::logsumexp.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_logsumexp_out = [](at::Tensor out, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::logsumexp_out(out, self, dim, keepdim);
        };
        return wrap(dispatch_logsumexp_out(_r.tensor(3), _r.tensor(0), _r.intlist(1), _r.toBool(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::logsumexp.names(Tensor self, Dimname[1] dim, bool keepdim=False) -> Tensor
        
        auto dispatch_logsumexp = [](const at::Tensor & self, at::DimnameList dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.logsumexp(dim, keepdim);
        };
        return wrap(dispatch_logsumexp(_r.tensor(0), _r.dimnamelist(1), _r.toBool(2)));
      } else {
        // aten::logsumexp.names_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_logsumexp_out = [](at::Tensor out, const at::Tensor & self, at::DimnameList dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::logsumexp_out(out, self, dim, keepdim);
        };
        return wrap(dispatch_logsumexp_out(_r.tensor(3), _r.tensor(0), _r.dimnamelist(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// matrix_power
static PyObject * THPVariable_matrix_power(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "matrix_power(Tensor input, int64_t n, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::matrix_power(Tensor self, int n) -> Tensor
    
    auto dispatch_matrix_power = [](const at::Tensor & self, int64_t n) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.matrix_power(n);
    };
    return wrap(dispatch_matrix_power(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::matrix_power.out(Tensor self, int n, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_matrix_power_out = [](at::Tensor out, const at::Tensor & self, int64_t n) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::matrix_power_out(out, self, n);
    };
    return wrap(dispatch_matrix_power_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _aminmax
static PyObject * THPVariable__aminmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_aminmax(Tensor input)",
    "_aminmax(Tensor input, int64_t dim, bool keepdim=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_aminmax(Tensor self) -> (Tensor, Tensor)
      
      auto dispatch__aminmax = [](const at::Tensor & self) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_aminmax(self);
      };
      return wrap(dispatch__aminmax(_r.tensor(0)));
    }
    case 1: {
      // aten::_aminmax.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)
      
      auto dispatch__aminmax = [](const at::Tensor & self, int64_t dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_aminmax(self, dim, keepdim);
      };
      return wrap(dispatch__aminmax(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// max
static PyObject * THPVariable_max(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_max_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_max_out_structseq();
  static PythonArgParser parser({
    "max(Tensor input, *, Tensor out=None)",
    "max(Tensor input, Tensor other, *, Tensor out=None)",
    "max(Tensor input, int64_t dim, bool keepdim=False, *, TensorList[2] out=None)",
    "max(Tensor input, Dimname dim, bool keepdim=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(1)) {
        // aten::max(Tensor self) -> Tensor
        
        auto dispatch_max = [](const at::Tensor & self) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.max();
        };
        return wrap(dispatch_max(_r.tensor(0)));
      } else {
        // aten::max.unary_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_max_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::max_out(out, self);
        };
        return wrap(dispatch_max_out(_r.tensor(1), _r.tensor(0)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::max.other(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_max = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.max(other);
        };
        return wrap(dispatch_max(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::max.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_max_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::max_out(out, self, other);
        };
        return wrap(dispatch_max_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(3)) {
        // aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        
        auto dispatch_max = [](const at::Tensor & self, int64_t dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.max(dim, keepdim);
        };
        return wrap(NamedTuple, dispatch_max(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      } else {
        // aten::max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_max_out = [](at::Tensor & max, at::Tensor & max_values, const at::Tensor & self, int64_t dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::max_out(max, max_values, self, dim, keepdim);
        };
        return wrap(NamedTuple1, dispatch_max_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      }
    }
    case 3: {
      if (_r.isNone(3)) {
        // aten::max.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        
        auto dispatch_max = [](const at::Tensor & self, at::Dimname dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.max(dim, keepdim);
        };
        return wrap(NamedTuple, dispatch_max(_r.tensor(0), _r.dimname(1), _r.toBool(2)));
      } else {
        // aten::max.names_dim_max(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_max_out = [](at::Tensor & max, at::Tensor & max_values, const at::Tensor & self, at::Dimname dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::max_out(max, max_values, self, dim, keepdim);
        };
        return wrap(NamedTuple1, dispatch_max_out(out[0], out[1], _r.tensor(0), _r.dimname(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// quantized_max_pool1d
static PyObject * THPVariable_quantized_max_pool1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantized_max_pool1d(Tensor input, IntArrayRef[1] kernel_size, IntArrayRef[1] stride=None, IntArrayRef[1] padding=0, IntArrayRef[1] dilation=1, bool ceil_mode=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::quantized_max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor
  
  auto dispatch_quantized_max_pool1d = [](const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::quantized_max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);
  };
  return wrap(dispatch_quantized_max_pool1d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// quantized_max_pool2d
static PyObject * THPVariable_quantized_max_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantized_max_pool2d(Tensor input, IntArrayRef[2] kernel_size, IntArrayRef[2] stride=None, IntArrayRef[2] padding=0, IntArrayRef[2] dilation=1, bool ceil_mode=False)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::quantized_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
  
  auto dispatch_quantized_max_pool2d = [](const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::quantized_max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
  };
  return wrap(dispatch_quantized_max_pool2d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// mean
static PyObject * THPVariable_mean(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mean(Tensor input, *, ScalarType? dtype=None, Tensor out=None)",
    "mean(Tensor input, IntArrayRef[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
    "mean(Tensor input, DimnameList[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_mean = [](const at::Tensor & self, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.mean(dtype);
        };
        return wrap(dispatch_mean(_r.tensor(0), _r.scalartypeOptional(1)));
      } else {
        // aten::mean.dtype_out(Tensor self, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_mean_out = [](at::Tensor out, const at::Tensor & self, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::mean_out(out, self, dtype);
        };
        return wrap(dispatch_mean_out(_r.tensor(2), _r.tensor(0), _r.scalartypeOptional(1)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_mean = [](const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.mean(dim, keepdim, dtype);
        };
        return wrap(dispatch_mean(_r.tensor(0), _r.intlistOptional(1), _r.toBool(2), _r.scalartypeOptional(3)));
      } else {
        // aten::mean.out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_mean_out = [](at::Tensor out, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::mean_out(out, self, dim, keepdim, dtype);
        };
        return wrap(dispatch_mean_out(_r.tensor(4), _r.tensor(0), _r.intlistOptional(1), _r.toBool(2), _r.scalartypeOptional(3)));
      }
    }
    case 2: {
      if (_r.isNone(4)) {
        // aten::mean.names_dim(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_mean = [](const at::Tensor & self, at::DimnameList dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.mean(dim, keepdim, dtype);
        };
        return wrap(dispatch_mean(_r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.scalartypeOptional(3)));
      } else {
        // aten::mean.names_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_mean_out = [](at::Tensor out, const at::Tensor & self, at::DimnameList dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::mean_out(out, self, dim, keepdim, dtype);
        };
        return wrap(dispatch_mean_out(_r.tensor(4), _r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.scalartypeOptional(3)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// nanmean
static PyObject * THPVariable_nanmean(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nanmean(Tensor input, IntArrayRef[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::nanmean(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    
    auto dispatch_nanmean = [](const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.nanmean(dim, keepdim, dtype);
    };
    return wrap(dispatch_nanmean(_r.tensor(0), _r.intlistOptional(1), _r.toBool(2), _r.scalartypeOptional(3)));
  } else {
    // aten::nanmean.out(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_nanmean_out = [](at::Tensor out, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::nanmean_out(out, self, dim, keepdim, dtype);
    };
    return wrap(dispatch_nanmean_out(_r.tensor(4), _r.tensor(0), _r.intlistOptional(1), _r.toBool(2), _r.scalartypeOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// nanmedian
static PyObject * THPVariable_nanmedian(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_nanmedian_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_nanmedian_out_structseq();
  static PythonArgParser parser({
    "nanmedian(Tensor input)",
    "nanmedian(Tensor input, int64_t dim, bool keepdim=False, *, TensorList[2] out=None)",
    "nanmedian(Tensor input, Dimname dim, bool keepdim=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::nanmedian(Tensor self) -> Tensor
      
      auto dispatch_nanmedian = [](const at::Tensor & self) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.nanmedian();
      };
      return wrap(dispatch_nanmedian(_r.tensor(0)));
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::nanmedian.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        
        auto dispatch_nanmedian = [](const at::Tensor & self, int64_t dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.nanmedian(dim, keepdim);
        };
        return wrap(NamedTuple, dispatch_nanmedian(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      } else {
        // aten::nanmedian.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_nanmedian_out = [](at::Tensor & values, at::Tensor & indices, const at::Tensor & self, int64_t dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::nanmedian_out(values, indices, self, dim, keepdim);
        };
        return wrap(NamedTuple1, dispatch_nanmedian_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      }
    }
    case 2: {
      if (_r.isNone(3)) {
        // aten::nanmedian.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        
        auto dispatch_nanmedian = [](const at::Tensor & self, at::Dimname dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.nanmedian(dim, keepdim);
        };
        return wrap(NamedTuple, dispatch_nanmedian(_r.tensor(0), _r.dimname(1), _r.toBool(2)));
      } else {
        // aten::nanmedian.names_dim_values(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_nanmedian_out = [](at::Tensor & values, at::Tensor & indices, const at::Tensor & self, at::Dimname dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::nanmedian_out(values, indices, self, dim, keepdim);
        };
        return wrap(NamedTuple1, dispatch_nanmedian_out(out[0], out[1], _r.tensor(0), _r.dimname(1), _r.toBool(2)));
      }
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
  static PyTypeObject* NamedTuple1 = generated::get_min_out_structseq();
  static PythonArgParser parser({
    "min(Tensor input, *, Tensor out=None)",
    "min(Tensor input, Tensor other, *, Tensor out=None)",
    "min(Tensor input, int64_t dim, bool keepdim=False, *, TensorList[2] out=None)",
    "min(Tensor input, Dimname dim, bool keepdim=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(1)) {
        // aten::min(Tensor self) -> Tensor
        
        auto dispatch_min = [](const at::Tensor & self) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.min();
        };
        return wrap(dispatch_min(_r.tensor(0)));
      } else {
        // aten::min.unary_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_min_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::min_out(out, self);
        };
        return wrap(dispatch_min_out(_r.tensor(1), _r.tensor(0)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::min.other(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_min = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.min(other);
        };
        return wrap(dispatch_min(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::min.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_min_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::min_out(out, self, other);
        };
        return wrap(dispatch_min_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(3)) {
        // aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        
        auto dispatch_min = [](const at::Tensor & self, int64_t dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.min(dim, keepdim);
        };
        return wrap(NamedTuple, dispatch_min(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      } else {
        // aten::min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_min_out = [](at::Tensor & min, at::Tensor & min_indices, const at::Tensor & self, int64_t dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::min_out(min, min_indices, self, dim, keepdim);
        };
        return wrap(NamedTuple1, dispatch_min_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      }
    }
    case 3: {
      if (_r.isNone(3)) {
        // aten::min.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        
        auto dispatch_min = [](const at::Tensor & self, at::Dimname dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.min(dim, keepdim);
        };
        return wrap(NamedTuple, dispatch_min(_r.tensor(0), _r.dimname(1), _r.toBool(2)));
      } else {
        // aten::min.names_dim_min(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_min_out = [](at::Tensor & min, at::Tensor & min_indices, const at::Tensor & self, at::Dimname dim, bool keepdim) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::min_out(min, min_indices, self, dim, keepdim);
        };
        return wrap(NamedTuple1, dispatch_min_out(out[0], out[1], _r.tensor(0), _r.dimname(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// amin
static PyObject * THPVariable_amin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "amin(Tensor input, IntArrayRef[1] dim=None, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
    
    auto dispatch_amin = [](const at::Tensor & self, at::IntArrayRef dim, bool keepdim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.amin(dim, keepdim);
    };
    return wrap(dispatch_amin(_r.tensor(0), _r.intlist(1), _r.toBool(2)));
  } else {
    // aten::amin.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_amin_out = [](at::Tensor out, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::amin_out(out, self, dim, keepdim);
    };
    return wrap(dispatch_amin_out(_r.tensor(3), _r.tensor(0), _r.intlist(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mkldnn_convolution
static PyObject * THPVariable_mkldnn_convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mkldnn_convolution(Tensor input, Tensor weight, Tensor? bias, SymIntArrayRef padding, SymIntArrayRef stride, SymIntArrayRef dilation, SymInt groups)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::mkldnn_convolution(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups) -> Tensor
  
  auto dispatch_mkldnn_convolution = [](const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::mkldnn_convolution_symint(self, weight, bias, padding, stride, dilation, groups);
  };
  return wrap(dispatch_mkldnn_convolution(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.symintlist(4), _r.symintlist(5), _r.toSymInt(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// miopen_batch_norm
static PyObject * THPVariable_miopen_batch_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "miopen_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, double exponential_average_factor, double epsilon)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::miopen_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor)
  
  auto dispatch_miopen_batch_norm = [](const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double exponential_average_factor, double epsilon) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::miopen_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
  };
  return wrap(dispatch_miopen_batch_norm(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.optionalTensor(3), _r.optionalTensor(4), _r.toBool(5), _r.toDouble(6), _r.toDouble(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// miopen_depthwise_convolution
static PyObject * THPVariable_miopen_depthwise_convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "miopen_depthwise_convolution(Tensor input, Tensor weight, Tensor? bias, SymIntArrayRef padding, SymIntArrayRef stride, SymIntArrayRef dilation, SymInt groups, bool benchmark, bool deterministic)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::miopen_depthwise_convolution(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, bool deterministic) -> Tensor
  
  auto dispatch_miopen_depthwise_convolution = [](const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, bool benchmark, bool deterministic) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::miopen_depthwise_convolution_symint(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
  };
  return wrap(dispatch_miopen_depthwise_convolution(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.symintlist(4), _r.symintlist(5), _r.toSymInt(6), _r.toBool(7), _r.toBool(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _weight_int8pack_mm
static PyObject * THPVariable__weight_int8pack_mm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_weight_int8pack_mm(Tensor input, Tensor mat2, Tensor scales)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_weight_int8pack_mm(Tensor self, Tensor mat2, Tensor scales) -> Tensor
  
  auto dispatch__weight_int8pack_mm = [](const at::Tensor & self, const at::Tensor & mat2, const at::Tensor & scales) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_weight_int8pack_mm(self, mat2, scales);
  };
  return wrap(dispatch__weight_int8pack_mm(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mv
static PyObject * THPVariable_mv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mv(Tensor input, Tensor vec, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::mv(Tensor self, Tensor vec) -> Tensor
    
    auto dispatch_mv = [](const at::Tensor & self, const at::Tensor & vec) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.mv(vec);
    };
    return wrap(dispatch_mv(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::mv.out(Tensor self, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_mv_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & vec) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::mv_out(out, self, vec);
    };
    return wrap(dispatch_mv_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// narrow
static PyObject * THPVariable_narrow(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "narrow(Tensor input, int64_t dim, Tensor start, SymInt length)",
    "narrow(Tensor input, int64_t dim, SymInt start, SymInt length)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, SymInt length) -> Tensor(a)
      
      auto dispatch_narrow = [](const at::Tensor & self, int64_t dim, const at::Tensor & start, c10::SymInt length) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.narrow_symint(dim, start, length);
      };
      return wrap(dispatch_narrow(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.toSymInt(3)));
    }
    case 1: {
      // aten::narrow(Tensor(a) self, int dim, SymInt start, SymInt length) -> Tensor(a)
      
      auto dispatch_narrow = [](const at::Tensor & self, int64_t dim, c10::SymInt start, c10::SymInt length) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.narrow_symint(dim, start, length);
      };
      return wrap(dispatch_narrow(_r.tensor(0), _r.toInt64(1), _r.toSymInt(2), _r.toSymInt(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// batch_norm_stats
static PyObject * THPVariable_batch_norm_stats(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "batch_norm_stats(Tensor input, double eps)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::batch_norm_stats(Tensor input, float eps) -> (Tensor, Tensor)
  
  auto dispatch_batch_norm_stats = [](const at::Tensor & input, double eps) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::batch_norm_stats(input, eps);
  };
  return wrap(dispatch_batch_norm_stats(_r.tensor(0), _r.toDouble(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// batch_norm_update_stats
static PyObject * THPVariable_batch_norm_update_stats(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "batch_norm_update_stats(Tensor input, Tensor? running_mean, Tensor? running_var, double momentum)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::batch_norm_update_stats(Tensor input, Tensor? running_mean, Tensor? running_var, float momentum) -> (Tensor, Tensor)
  
  auto dispatch_batch_norm_update_stats = [](const at::Tensor & input, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, double momentum) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::batch_norm_update_stats(input, running_mean, running_var, momentum);
  };
  return wrap(dispatch_batch_norm_update_stats(_r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2), _r.toDouble(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nnpack_available
static PyObject * THPVariable__nnpack_available(PyObject* self_, PyObject* args)
{
  HANDLE_TH_ERRORS

  // aten::_nnpack_available() -> bool
  
  auto dispatch__nnpack_available = []() -> bool {
    pybind11::gil_scoped_release no_gil;
    return at::_nnpack_available();
  };
  return wrap(dispatch__nnpack_available());
  END_HANDLE_TH_ERRORS
}

// _nnpack_spatial_convolution
static PyObject * THPVariable__nnpack_spatial_convolution(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nnpack_spatial_convolution(Tensor input, Tensor weight, Tensor? bias, SymIntArrayRef[2] padding, SymIntArrayRef[2] stride=1)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_nnpack_spatial_convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[2] padding, SymInt[2] stride=1) -> Tensor
  
  auto dispatch__nnpack_spatial_convolution = [](const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_nnpack_spatial_convolution_symint(input, weight, bias, padding, stride);
  };
  return wrap(dispatch__nnpack_spatial_convolution(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.symintlist(3), _r.symintlist(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _euclidean_dist
static PyObject * THPVariable__euclidean_dist(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_euclidean_dist(Tensor x1, Tensor x2)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_euclidean_dist(Tensor x1, Tensor x2) -> Tensor
  
  auto dispatch__euclidean_dist = [](const at::Tensor & x1, const at::Tensor & x2) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_euclidean_dist(x1, x2);
  };
  return wrap(dispatch__euclidean_dist(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// permute
static PyObject * THPVariable_permute(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "permute(Tensor input, IntArrayRef dims)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)
  
  auto dispatch_permute = [](const at::Tensor & self, at::IntArrayRef dims) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.permute(dims);
  };
  return wrap(dispatch_permute(_r.tensor(0), _r.intlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// pixel_shuffle
static PyObject * THPVariable_pixel_shuffle(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pixel_shuffle(Tensor input, int64_t upscale_factor)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor
  
  auto dispatch_pixel_shuffle = [](const at::Tensor & self, int64_t upscale_factor) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::pixel_shuffle(self, upscale_factor);
  };
  return wrap(dispatch_pixel_shuffle(_r.tensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// native_channel_shuffle
static PyObject * THPVariable_native_channel_shuffle(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "native_channel_shuffle(Tensor input, SymInt groups)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::native_channel_shuffle(Tensor self, SymInt groups) -> Tensor
  
  auto dispatch_native_channel_shuffle = [](const at::Tensor & self, c10::SymInt groups) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::native_channel_shuffle_symint(self, groups);
  };
  return wrap(dispatch_native_channel_shuffle(_r.tensor(0), _r.toSymInt(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _pin_memory
static PyObject * THPVariable__pin_memory(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_pin_memory(Tensor input, Device? device=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_pin_memory(Tensor self, Device? device=None) -> Tensor
  
  auto dispatch__pin_memory = [](const at::Tensor & self, ::std::optional<at::Device> device) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_pin_memory(self, device);
  };
  return wrap(dispatch__pin_memory(_r.tensor(0), _r.deviceOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rad2deg
static PyObject * THPVariable_rad2deg(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rad2deg(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::rad2deg(Tensor self) -> Tensor
    
    auto dispatch_rad2deg = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.rad2deg();
    };
    return wrap(dispatch_rad2deg(_r.tensor(0)));
  } else {
    // aten::rad2deg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_rad2deg_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::rad2deg_out(out, self);
    };
    return wrap(dispatch_rad2deg_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rad2deg_
static PyObject * THPVariable_rad2deg_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rad2deg_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::rad2deg_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_rad2deg_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.rad2deg_();
  };
  return wrap(dispatch_rad2deg_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rand_like
static PyObject * THPVariable_rand_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rand_like(Tensor input, *, MemoryFormat? memory_format=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
  auto self = _r.tensor(0);
  const auto options = TensorOptions()
      .dtype(_r.scalartypeOptional(2))
      .device(_r.deviceOptional(4))
      .layout(_r.layoutOptional(3))
      .requires_grad(_r.toBool(6))
      .pinned_memory(_r.toBool(5));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch_rand_like = [](const at::Tensor & self, at::TensorOptions options, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::rand_like(self, options, memory_format);
  };
  return wrap(dispatch_rand_like(self, options, _r.memoryformatOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// randint
static PyObject * THPVariable_randint(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randint(SymInt high, SymIntArrayRef size, *, Generator? generator, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "randint(SymInt high, SymIntArrayRef size, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "randint(SymInt low, SymInt high, SymIntArrayRef size, *, Generator? generator, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "randint(SymInt low, SymInt high, SymIntArrayRef size, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<10> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::randint.generator(SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeWithDefault(4, at::kLong))
            .device(_r.deviceWithDefault(6, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(5))
            .requires_grad(_r.toBool(8))
            .pinned_memory(_r.toBool(7));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_randint = [](c10::SymInt high, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::randint_symint(high, size, generator, options);
        };
        return wrap(dispatch_randint(_r.toSymInt(0), _r.symintlist(1), _r.generator(2), options));
      } else {
        // aten::randint.generator_out(SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(3), _r.scalartypeWithDefault(4, at::kLong),
                               _r.isNone(4), _r.layoutOptional(5),
                               _r.deviceWithDefault(6, torch::tensors::get_default_device()), _r.isNone(6));
        
        auto dispatch_randint_out = [](at::Tensor out, c10::SymInt high, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::randint_symint_out(out, high, size, generator);
        };
        return wrap(dispatch_randint_out(_r.tensor(3), _r.toSymInt(0), _r.symintlist(1), _r.generator(2)).set_requires_grad(_r.toBool(8)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::randint(SymInt high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeWithDefault(3, at::kLong))
            .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(4))
            .requires_grad(_r.toBool(7))
            .pinned_memory(_r.toBool(6));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_randint = [](c10::SymInt high, c10::SymIntArrayRef size, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::randint_symint(high, size, options);
        };
        return wrap(dispatch_randint(_r.toSymInt(0), _r.symintlist(1), options));
      } else {
        // aten::randint.out(SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(2), _r.scalartypeWithDefault(3, at::kLong),
                               _r.isNone(3), _r.layoutOptional(4),
                               _r.deviceWithDefault(5, torch::tensors::get_default_device()), _r.isNone(5));
        
        auto dispatch_randint_out = [](at::Tensor out, c10::SymInt high, c10::SymIntArrayRef size) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::randint_symint_out(out, high, size);
        };
        return wrap(dispatch_randint_out(_r.tensor(2), _r.toSymInt(0), _r.symintlist(1)).set_requires_grad(_r.toBool(7)));
      }
    }
    case 2: {
      if (_r.isNone(4)) {
        // aten::randint.low_generator(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeWithDefault(5, at::kLong))
            .device(_r.deviceWithDefault(7, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(6))
            .requires_grad(_r.toBool(9))
            .pinned_memory(_r.toBool(8));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_randint = [](c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::randint_symint(low, high, size, generator, options);
        };
        return wrap(dispatch_randint(_r.toSymInt(0), _r.toSymInt(1), _r.symintlist(2), _r.generator(3), options));
      } else {
        // aten::randint.low_generator_out(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(4), _r.scalartypeWithDefault(5, at::kLong),
                               _r.isNone(5), _r.layoutOptional(6),
                               _r.deviceWithDefault(7, torch::tensors::get_default_device()), _r.isNone(7));
        
        auto dispatch_randint_out = [](at::Tensor out, c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::randint_symint_out(out, low, high, size, generator);
        };
        return wrap(dispatch_randint_out(_r.tensor(4), _r.toSymInt(0), _r.toSymInt(1), _r.symintlist(2), _r.generator(3)).set_requires_grad(_r.toBool(9)));
      }
    }
    case 3: {
      if (_r.isNone(3)) {
        // aten::randint.low(SymInt low, SymInt high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeWithDefault(4, at::kLong))
            .device(_r.deviceWithDefault(6, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(5))
            .requires_grad(_r.toBool(8))
            .pinned_memory(_r.toBool(7));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_randint = [](c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::randint_symint(low, high, size, options);
        };
        return wrap(dispatch_randint(_r.toSymInt(0), _r.toSymInt(1), _r.symintlist(2), options));
      } else {
        // aten::randint.low_out(SymInt low, SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(3), _r.scalartypeWithDefault(4, at::kLong),
                               _r.isNone(4), _r.layoutOptional(5),
                               _r.deviceWithDefault(6, torch::tensors::get_default_device()), _r.isNone(6));
        
        auto dispatch_randint_out = [](at::Tensor out, c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::randint_symint_out(out, low, high, size);
        };
        return wrap(dispatch_randint_out(_r.tensor(3), _r.toSymInt(0), _r.toSymInt(1), _r.symintlist(2)).set_requires_grad(_r.toBool(8)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// randint_like
static PyObject * THPVariable_randint_like(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randint_like(Tensor input, SymInt high, *, MemoryFormat? memory_format=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "randint_like(Tensor input, SymInt low, SymInt high, *, MemoryFormat? memory_format=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::randint_like(Tensor self, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
      auto self = _r.tensor(0);
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(3))
          .device(_r.deviceOptional(5))
          .layout(_r.layoutOptional(4))
          .requires_grad(_r.toBool(7))
          .pinned_memory(_r.toBool(6));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_randint_like = [](const at::Tensor & self, c10::SymInt high, at::TensorOptions options, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::randint_like_symint(self, high, options, memory_format);
      };
      return wrap(dispatch_randint_like(self, _r.toSymInt(1), options, _r.memoryformatOptional(2)));
    }
    case 1: {
      // aten::randint_like.low_dtype(Tensor self, SymInt low, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
      auto self = _r.tensor(0);
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(4))
          .device(_r.deviceOptional(6))
          .layout(_r.layoutOptional(5))
          .requires_grad(_r.toBool(8))
          .pinned_memory(_r.toBool(7));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_randint_like = [](const at::Tensor & self, c10::SymInt low, c10::SymInt high, at::TensorOptions options, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::randint_like_symint(self, low, high, options, memory_format);
      };
      return wrap(dispatch_randint_like(self, _r.toSymInt(1), _r.toSymInt(2), options, _r.memoryformatOptional(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// randn
static PyObject * THPVariable_randn(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randn(SymIntArrayRef size, *, Generator? generator, DimnameList? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "randn(SymIntArrayRef size, *, Generator? generator, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "randn(SymIntArrayRef size, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "randn(SymIntArrayRef size, *, DimnameList? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::randn.generator_with_names(SymInt[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      auto __names = _r.toDimnameListOptional(2);
      ::std::optional<DimnameList> names = __names ? ::std::make_optional(DimnameList(__names.value())) : ::std::nullopt;
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(3))
          .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(4))
          .requires_grad(_r.toBool(7))
          .pinned_memory(_r.toBool(6));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_randn = [](c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::randn_symint(size, generator, names, options);
      };
      return wrap(dispatch_randn(_r.symintlist(0), _r.generator(1), names, options));
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::randn.generator(SymInt[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(3))
            .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(4))
            .requires_grad(_r.toBool(7))
            .pinned_memory(_r.toBool(6));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_randn = [](c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::randn_symint(size, generator, options);
        };
        return wrap(dispatch_randn(_r.symintlist(0), _r.generator(1), options));
      } else {
        // aten::randn.generator_out(SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(2), _r.scalartypeOptional(3),
                               _r.isNone(3), _r.layoutOptional(4),
                               _r.deviceWithDefault(5, torch::tensors::get_default_device()), _r.isNone(5));
        
        auto dispatch_randn_out = [](at::Tensor out, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::randn_symint_out(out, size, generator);
        };
        return wrap(dispatch_randn_out(_r.tensor(2), _r.symintlist(0), _r.generator(1)).set_requires_grad(_r.toBool(7)));
      }
    }
    case 2: {
      if (_r.isNone(1)) {
        // aten::randn(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeOptional(2))
            .device(_r.deviceWithDefault(4, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(3))
            .requires_grad(_r.toBool(6))
            .pinned_memory(_r.toBool(5));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_randn = [](c10::SymIntArrayRef size, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::randn_symint(size, options);
        };
        return wrap(dispatch_randn(_r.symintlist(0), options));
      } else {
        // aten::randn.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(1), _r.scalartypeOptional(2),
                               _r.isNone(2), _r.layoutOptional(3),
                               _r.deviceWithDefault(4, torch::tensors::get_default_device()), _r.isNone(4));
        
        auto dispatch_randn_out = [](at::Tensor out, c10::SymIntArrayRef size) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::randn_symint_out(out, size);
        };
        return wrap(dispatch_randn_out(_r.tensor(1), _r.symintlist(0)).set_requires_grad(_r.toBool(6)));
      }
    }
    case 3: {
      // aten::randn.names(SymInt[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
      auto __names = _r.toDimnameListOptional(1);
      ::std::optional<DimnameList> names = __names ? ::std::make_optional(DimnameList(__names.value())) : ::std::nullopt;
      const auto options = TensorOptions()
          .dtype(_r.scalartypeOptional(2))
          .device(_r.deviceWithDefault(4, torch::tensors::get_default_device()))
          .layout(_r.layoutOptional(3))
          .requires_grad(_r.toBool(6))
          .pinned_memory(_r.toBool(5));
      torch::utils::maybe_initialize_device(options);
      
      auto dispatch_randn = [](c10::SymIntArrayRef size, ::std::optional<at::DimnameList> names, at::TensorOptions options) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return torch::randn_symint(size, names, options);
      };
      return wrap(dispatch_randn(_r.symintlist(0), names, options));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// randperm
static PyObject * THPVariable_randperm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "randperm(SymInt n, *, Generator? generator, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
    "randperm(SymInt n, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::randperm.generator(SymInt n, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeWithDefault(3, at::kLong))
            .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(4))
            .requires_grad(_r.toBool(7))
            .pinned_memory(_r.toBool(6));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_randperm = [](c10::SymInt n, ::std::optional<at::Generator> generator, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::randperm_symint(n, generator, options);
        };
        return wrap(dispatch_randperm(_r.toSymInt(0), _r.generator(1), options));
      } else {
        // aten::randperm.generator_out(SymInt n, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(2), _r.scalartypeWithDefault(3, at::kLong),
                               _r.isNone(3), _r.layoutOptional(4),
                               _r.deviceWithDefault(5, torch::tensors::get_default_device()), _r.isNone(5));
        
        auto dispatch_randperm_out = [](at::Tensor out, c10::SymInt n, ::std::optional<at::Generator> generator) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::randperm_symint_out(out, n, generator);
        };
        return wrap(dispatch_randperm_out(_r.tensor(2), _r.toSymInt(0), _r.generator(1)).set_requires_grad(_r.toBool(7)));
      }
    }
    case 1: {
      if (_r.isNone(1)) {
        // aten::randperm(SymInt n, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        const auto options = TensorOptions()
            .dtype(_r.scalartypeWithDefault(2, at::kLong))
            .device(_r.deviceWithDefault(4, torch::tensors::get_default_device()))
            .layout(_r.layoutOptional(3))
            .requires_grad(_r.toBool(6))
            .pinned_memory(_r.toBool(5));
        torch::utils::maybe_initialize_device(options);
        
        auto dispatch_randperm = [](c10::SymInt n, at::TensorOptions options) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return torch::randperm_symint(n, options);
        };
        return wrap(dispatch_randperm(_r.toSymInt(0), options));
      } else {
        // aten::randperm.out(SymInt n, *, Tensor(a!) out) -> Tensor(a!)
        check_out_type_matches(_r.tensor(1), _r.scalartypeWithDefault(2, at::kLong),
                               _r.isNone(2), _r.layoutOptional(3),
                               _r.deviceWithDefault(4, torch::tensors::get_default_device()), _r.isNone(4));
        
        auto dispatch_randperm_out = [](at::Tensor out, c10::SymInt n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::randperm_symint_out(out, n);
        };
        return wrap(dispatch_randperm_out(_r.tensor(1), _r.toSymInt(0)).set_requires_grad(_r.toBool(6)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// reciprocal
static PyObject * THPVariable_reciprocal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reciprocal(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::reciprocal(Tensor self) -> Tensor
    
    auto dispatch_reciprocal = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.reciprocal();
    };
    return wrap(dispatch_reciprocal(_r.tensor(0)));
  } else {
    // aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_reciprocal_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::reciprocal_out(out, self);
    };
    return wrap(dispatch_reciprocal_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// reciprocal_
static PyObject * THPVariable_reciprocal_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reciprocal_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_reciprocal_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.reciprocal_();
  };
  return wrap(dispatch_reciprocal_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// negative
static PyObject * THPVariable_negative(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "negative(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::negative(Tensor self) -> Tensor
    
    auto dispatch_negative = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.negative();
    };
    return wrap(dispatch_negative(_r.tensor(0)));
  } else {
    // aten::negative.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_negative_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::negative_out(out, self);
    };
    return wrap(dispatch_negative_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// negative_
static PyObject * THPVariable_negative_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "negative_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::negative_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_negative_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.negative_();
  };
  return wrap(dispatch_negative_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// reshape
static PyObject * THPVariable_reshape(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reshape(Tensor input, SymIntArrayRef shape)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)
  
  auto dispatch_reshape = [](const at::Tensor & self, c10::SymIntArrayRef shape) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.reshape_symint(shape);
  };
  return wrap(dispatch_reshape(_r.tensor(0), _r.symintlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _mkldnn_reshape
static PyObject * THPVariable__mkldnn_reshape(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_mkldnn_reshape(Tensor input, IntArrayRef shape)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_mkldnn_reshape(Tensor self, int[] shape) -> Tensor
  
  auto dispatch__mkldnn_reshape = [](const at::Tensor & self, at::IntArrayRef shape) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_mkldnn_reshape(self, shape);
  };
  return wrap(dispatch__mkldnn_reshape(_r.tensor(0), _r.intlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// relu
static PyObject * THPVariable_relu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "relu(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::relu(Tensor self) -> Tensor
  
  auto dispatch_relu = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.relu();
  };
  return wrap(dispatch_relu(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// relu_
static PyObject * THPVariable_relu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "relu_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::relu_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_relu_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.relu_();
  };
  return wrap(dispatch_relu_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// prelu
static PyObject * THPVariable_prelu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "prelu(Tensor input, Tensor weight)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::prelu(Tensor self, Tensor weight) -> Tensor
  
  auto dispatch_prelu = [](const at::Tensor & self, const at::Tensor & weight) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.prelu(weight);
  };
  return wrap(dispatch_prelu(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// select
static PyObject * THPVariable_select(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "select(Tensor input, Dimname dim, int64_t index)",
    "select(Tensor input, int64_t dim, SymInt index)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::select.Dimname(Tensor(a) self, Dimname dim, int index) -> Tensor(a)
      
      auto dispatch_select = [](const at::Tensor & self, at::Dimname dim, int64_t index) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.select(dim, index);
      };
      return wrap(dispatch_select(_r.tensor(0), _r.dimname(1), _r.toInt64(2)));
    }
    case 1: {
      // aten::select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a)
      
      auto dispatch_select = [](const at::Tensor & self, int64_t dim, c10::SymInt index) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.select_symint(dim, index);
      };
      return wrap(dispatch_select(_r.tensor(0), _r.toInt64(1), _r.toSymInt(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// selu
static PyObject * THPVariable_selu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "selu(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::selu(Tensor self) -> Tensor
  
  auto dispatch_selu = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::selu(self);
  };
  return wrap(dispatch_selu(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// selu_
static PyObject * THPVariable_selu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "selu_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::selu_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_selu_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::selu_(self);
  };
  return wrap(dispatch_selu_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logit
static PyObject * THPVariable_logit(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logit(Tensor input, double? eps=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::logit(Tensor self, float? eps=None) -> Tensor
    
    auto dispatch_logit = [](const at::Tensor & self, ::std::optional<double> eps) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.logit(eps);
    };
    return wrap(dispatch_logit(_r.tensor(0), _r.toDoubleOptional(1)));
  } else {
    // aten::logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_logit_out = [](at::Tensor out, const at::Tensor & self, ::std::optional<double> eps) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::logit_out(out, self, eps);
    };
    return wrap(dispatch_logit_out(_r.tensor(2), _r.tensor(0), _r.toDoubleOptional(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// logit_
static PyObject * THPVariable_logit_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "logit_(Tensor input, double? eps=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::logit_(Tensor(a!) self, float? eps=None) -> Tensor(a!)
  
  auto dispatch_logit_ = [](at::Tensor self, ::std::optional<double> eps) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.logit_(eps);
  };
  return wrap(dispatch_logit_(_r.tensor(0), _r.toDoubleOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sin
static PyObject * THPVariable_sin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sin(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::sin(Tensor self) -> Tensor
    
    auto dispatch_sin = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.sin();
    };
    return wrap(dispatch_sin(_r.tensor(0)));
  } else {
    // aten::sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_sin_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::sin_out(out, self);
    };
    return wrap(dispatch_sin_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sin_
static PyObject * THPVariable_sin_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sin_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::sin_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_sin_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.sin_();
  };
  return wrap(dispatch_sin_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// slice_inverse
static PyObject * THPVariable_slice_inverse(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "slice_inverse(Tensor input, Tensor src, int64_t dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::slice_inverse(Tensor(a) self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)
  
  auto dispatch_slice_inverse = [](const at::Tensor & self, const at::Tensor & src, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.slice_inverse_symint(src, dim, start, end, step);
  };
  return wrap(dispatch_slice_inverse(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toSymIntOptional(3), _r.toSymIntOptional(4), _r.toSymInt(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// softmax
static PyObject * THPVariable_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "softmax(Tensor input, int64_t dim, ScalarType? dtype=None, *, Tensor out=None)",
    "softmax(Tensor input, Dimname dim, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_softmax = [](const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.softmax(dim, dtype);
        };
        return wrap(dispatch_softmax(_r.tensor(0), _r.toInt64(1), _r.scalartypeOptional(2)));
      } else {
        // aten::softmax.int_out(Tensor self, int dim, ScalarType? dtype=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_softmax_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::softmax_out(out, self, dim, dtype);
        };
        return wrap(dispatch_softmax_out(_r.tensor(3), _r.tensor(0), _r.toInt64(1), _r.scalartypeOptional(2)));
      }
    }
    case 1: {
      // aten::softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_softmax = [](const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.softmax(dim, dtype);
      };
      return wrap(dispatch_softmax(_r.tensor(0), _r.dimname(1), _r.scalartypeOptional(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unsafe_split_with_sizes
static PyObject * THPVariable_unsafe_split_with_sizes(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unsafe_split_with_sizes(Tensor input, SymIntArrayRef split_sizes, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::unsafe_split_with_sizes(Tensor self, SymInt[] split_sizes, int dim=0) -> Tensor[]
  
  auto dispatch_unsafe_split_with_sizes = [](const at::Tensor & self, c10::SymIntArrayRef split_sizes, int64_t dim) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.unsafe_split_with_sizes_symint(split_sizes, dim);
  };
  return wrap(dispatch_unsafe_split_with_sizes(_r.tensor(0), _r.symintlist(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// split_with_sizes
static PyObject * THPVariable_split_with_sizes(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "split_with_sizes(Tensor input, SymIntArrayRef split_sizes, int64_t dim=0)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]
  
  auto dispatch_split_with_sizes = [](const at::Tensor & self, c10::SymIntArrayRef split_sizes, int64_t dim) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return self.split_with_sizes_symint(split_sizes, dim);
  };
  return wrap(dispatch_split_with_sizes(_r.tensor(0), _r.symintlist(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// vsplit
static PyObject * THPVariable_vsplit(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "vsplit(Tensor input, int64_t sections)",
    "vsplit(Tensor input, IntArrayRef indices)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::vsplit.int(Tensor(a -> *) self, int sections) -> Tensor(a)[]
      
      auto dispatch_vsplit = [](const at::Tensor & self, int64_t sections) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.vsplit(sections);
      };
      return wrap(dispatch_vsplit(_r.tensor(0), _r.toInt64(1)));
    }
    case 1: {
      // aten::vsplit.array(Tensor(a -> *) self, int[] indices) -> Tensor(a)[]
      
      auto dispatch_vsplit = [](const at::Tensor & self, at::IntArrayRef indices) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return self.vsplit(indices);
      };
      return wrap(dispatch_vsplit(_r.tensor(0), _r.intlist(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _chunk_cat
static PyObject * THPVariable__chunk_cat(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_chunk_cat(TensorList tensors, int64_t dim, int64_t num_chunks, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::_chunk_cat(Tensor[] tensors, int dim, int num_chunks) -> Tensor
    
    auto dispatch__chunk_cat = [](at::TensorList tensors, int64_t dim, int64_t num_chunks) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_chunk_cat(tensors, dim, num_chunks);
    };
    return wrap(dispatch__chunk_cat(_r.tensorlist(0), _r.toInt64(1), _r.toInt64(2)));
  } else {
    // aten::_chunk_cat.out(Tensor[] tensors, int dim, int num_chunks, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__chunk_cat_out = [](at::Tensor out, at::TensorList tensors, int64_t dim, int64_t num_chunks) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_chunk_cat_out(out, tensors, dim, num_chunks);
    };
    return wrap(dispatch__chunk_cat_out(_r.tensor(3), _r.tensorlist(0), _r.toInt64(1), _r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hstack
static PyObject * THPVariable_hstack(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hstack(TensorList tensors, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::hstack(Tensor[] tensors) -> Tensor
    
    auto dispatch_hstack = [](at::TensorList tensors) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::hstack(tensors);
    };
    return wrap(dispatch_hstack(_r.tensorlist(0)));
  } else {
    // aten::hstack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_hstack_out = [](at::Tensor out, at::TensorList tensors) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::hstack_out(out, tensors);
    };
    return wrap(dispatch_hstack_out(_r.tensor(1), _r.tensorlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// vstack
static PyObject * THPVariable_vstack(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "vstack(TensorList tensors, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::vstack(Tensor[] tensors) -> Tensor
    
    auto dispatch_vstack = [](at::TensorList tensors) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::vstack(tensors);
    };
    return wrap(dispatch_vstack(_r.tensorlist(0)));
  } else {
    // aten::vstack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_vstack_out = [](at::Tensor out, at::TensorList tensors) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::vstack_out(out, tensors);
    };
    return wrap(dispatch_vstack_out(_r.tensor(1), _r.tensorlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// dstack
static PyObject * THPVariable_dstack(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "dstack(TensorList tensors, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::dstack(Tensor[] tensors) -> Tensor
    
    auto dispatch_dstack = [](at::TensorList tensors) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::dstack(tensors);
    };
    return wrap(dispatch_dstack(_r.tensorlist(0)));
  } else {
    // aten::dstack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_dstack_out = [](at::Tensor out, at::TensorList tensors) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::dstack_out(out, tensors);
    };
    return wrap(dispatch_dstack_out(_r.tensor(1), _r.tensorlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// sum
static PyObject * THPVariable_sum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sum(Tensor input, *, ScalarType? dtype=None)",
    "sum(Tensor input, IntArrayRef[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
    "sum(Tensor input, DimnameList[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch_sum = [](const at::Tensor & self, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.sum(dtype);
      };
      return wrap(dispatch_sum(_r.tensor(0), _r.scalartypeOptional(1)));
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_sum = [](const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.sum(dim, keepdim, dtype);
        };
        return wrap(dispatch_sum(_r.tensor(0), _r.intlistOptional(1), _r.toBool(2), _r.scalartypeOptional(3)));
      } else {
        // aten::sum.IntList_out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_sum_out = [](at::Tensor out, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::sum_out(out, self, dim, keepdim, dtype);
        };
        return wrap(dispatch_sum_out(_r.tensor(4), _r.tensor(0), _r.intlistOptional(1), _r.toBool(2), _r.scalartypeOptional(3)));
      }
    }
    case 2: {
      if (_r.isNone(4)) {
        // aten::sum.dim_DimnameList(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_sum = [](const at::Tensor & self, at::DimnameList dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.sum(dim, keepdim, dtype);
        };
        return wrap(dispatch_sum(_r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.scalartypeOptional(3)));
      } else {
        // aten::sum.DimnameList_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_sum_out = [](at::Tensor out, const at::Tensor & self, at::DimnameList dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::sum_out(out, self, dim, keepdim, dtype);
        };
        return wrap(dispatch_sum_out(_r.tensor(4), _r.tensor(0), _r.dimnamelist(1), _r.toBool(2), _r.scalartypeOptional(3)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tan
static PyObject * THPVariable_tan(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tan(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::tan(Tensor self) -> Tensor
    
    auto dispatch_tan = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.tan();
    };
    return wrap(dispatch_tan(_r.tensor(0)));
  } else {
    // aten::tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_tan_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::tan_out(out, self);
    };
    return wrap(dispatch_tan_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// tan_
static PyObject * THPVariable_tan_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "tan_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::tan_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_tan_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.tan_();
  };
  return wrap(dispatch_tan_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// trapz
static PyObject * THPVariable_trapz(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "trapz(Tensor y, *, double dx=1, int64_t dim=-1)",
    "trapz(Tensor y, Tensor x, *, int64_t dim=-1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::trapz.dx(Tensor y, *, float dx=1, int dim=-1) -> Tensor
      
      auto dispatch_trapz = [](const at::Tensor & y, double dx, int64_t dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::trapz(y, dx, dim);
      };
      return wrap(dispatch_trapz(_r.tensor(0), _r.toDouble(1), _r.toInt64(2)));
    }
    case 1: {
      // aten::trapz.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor
      
      auto dispatch_trapz = [](const at::Tensor & y, const at::Tensor & x, int64_t dim) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::trapz(y, x, dim);
      };
      return wrap(dispatch_trapz(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_tensor_from_mask_left_aligned
static PyObject * THPVariable__nested_tensor_from_mask_left_aligned(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_tensor_from_mask_left_aligned(Tensor t, Tensor mask)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_nested_tensor_from_mask_left_aligned(Tensor t, Tensor mask) -> bool
  
  auto dispatch__nested_tensor_from_mask_left_aligned = [](const at::Tensor & t, const at::Tensor & mask) -> bool {
    pybind11::gil_scoped_release no_gil;
    return at::_nested_tensor_from_mask_left_aligned(t, mask);
  };
  return wrap(dispatch__nested_tensor_from_mask_left_aligned(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_from_padded_and_nested_example
static PyObject * THPVariable__nested_from_padded_and_nested_example(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_from_padded_and_nested_example(Tensor padded, Tensor nt_example)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_nested_from_padded_and_nested_example(Tensor padded, Tensor nt_example) -> Tensor
  
  auto dispatch__nested_from_padded_and_nested_example = [](const at::Tensor & padded, const at::Tensor & nt_example) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_nested_from_padded_and_nested_example(padded, nt_example);
  };
  return wrap(dispatch__nested_from_padded_and_nested_example(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_view_from_buffer
static PyObject * THPVariable__nested_view_from_buffer(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_view_from_buffer(Tensor input, Tensor nested_size, Tensor nested_strides, Tensor offsets)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_nested_view_from_buffer(Tensor(a) self, Tensor nested_size, Tensor nested_strides, Tensor offsets) -> Tensor(a)
  
  auto dispatch__nested_view_from_buffer = [](const at::Tensor & self, const at::Tensor & nested_size, const at::Tensor & nested_strides, const at::Tensor & offsets) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_nested_view_from_buffer(self, nested_size, nested_strides, offsets);
  };
  return wrap(dispatch__nested_view_from_buffer(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_get_values_copy
static PyObject * THPVariable__nested_get_values_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_get_values_copy(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::_nested_get_values_copy(Tensor self) -> Tensor
    
    auto dispatch__nested_get_values_copy = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_nested_get_values_copy(self);
    };
    return wrap(dispatch__nested_get_values_copy(_r.tensor(0)));
  } else {
    // aten::_nested_get_values_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__nested_get_values_copy_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_nested_get_values_copy_out(out, self);
    };
    return wrap(dispatch__nested_get_values_copy_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_get_lengths
static PyObject * THPVariable__nested_get_lengths(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_get_lengths(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_nested_get_lengths(Tensor self) -> Tensor
  
  auto dispatch__nested_get_lengths = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_nested_get_lengths(self);
  };
  return wrap(dispatch__nested_get_lengths(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// trunc
static PyObject * THPVariable_trunc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "trunc(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::trunc(Tensor self) -> Tensor
    
    auto dispatch_trunc = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.trunc();
    };
    return wrap(dispatch_trunc(_r.tensor(0)));
  } else {
    // aten::trunc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_trunc_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::trunc_out(out, self);
    };
    return wrap(dispatch_trunc_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// trunc_
static PyObject * THPVariable_trunc_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "trunc_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::trunc_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_trunc_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.trunc_();
  };
  return wrap(dispatch_trunc_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unique_dim
static PyObject * THPVariable_unique_dim(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unique_dim(Tensor input, int64_t dim, bool sorted=True, bool return_inverse=False, bool return_counts=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
  
  auto dispatch_unique_dim = [](const at::Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::unique_dim(self, dim, sorted, return_inverse, return_counts);
  };
  return wrap(dispatch_unique_dim(_r.tensor(0), _r.toInt64(1), _r.toBool(2), _r.toBool(3), _r.toBool(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unique_consecutive
static PyObject * THPVariable_unique_consecutive(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unique_consecutive(Tensor input, bool return_inverse=False, bool return_counts=False, int64_t? dim=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int? dim=None) -> (Tensor, Tensor, Tensor)
  
  auto dispatch_unique_consecutive = [](const at::Tensor & self, bool return_inverse, bool return_counts, ::std::optional<int64_t> dim) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::unique_consecutive(self, return_inverse, return_counts, dim);
  };
  return wrap(dispatch_unique_consecutive(_r.tensor(0), _r.toBool(1), _r.toBool(2), _r.toInt64Optional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// where
static PyObject * THPVariable_where(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "where(Tensor condition)",
    "where(Tensor condition, Tensor input, Tensor other, *, Tensor out=None)",
    "where(Tensor condition, Scalar self, Tensor other)",
    "where(Tensor condition, Tensor input, Scalar other)",
    "where(Tensor condition, Scalar self, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::where(Tensor condition) -> Tensor[]
      
      auto dispatch_where = [](const at::Tensor & condition) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::where(condition);
      };
      return wrap(dispatch_where(_r.tensor(0)));
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
        
        auto dispatch_where = [](const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.where(condition, other);
        };
        return wrap(dispatch_where(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
      } else {
        // aten::where.self_out(Tensor condition, Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_where_out = [](at::Tensor out, const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::where_out(out, condition, self, other);
        };
        return wrap(dispatch_where_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.tensor(2)));
      }
    }
    case 2: {
      // aten::where.ScalarSelf(Tensor condition, Scalar self, Tensor other) -> Tensor
      
      auto dispatch_where = [](const at::Tensor & condition, const at::Scalar & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::where(condition, self, other);
      };
      return wrap(dispatch_where(_r.tensor(0), _r.scalar(1), _r.tensor(2)));
    }
    case 3: {
      // aten::where.ScalarOther(Tensor condition, Tensor self, Scalar other) -> Tensor
      
      auto dispatch_where = [](const at::Tensor & condition, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.where(condition, other);
      };
      return wrap(dispatch_where(_r.tensor(0), _r.tensor(1), _r.scalar(2)));
    }
    case 4: {
      // aten::where.Scalar(Tensor condition, Scalar self, Scalar other) -> Tensor
      
      auto dispatch_where = [](const at::Tensor & condition, const at::Scalar & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::where(condition, self, other);
      };
      return wrap(dispatch_where(_r.tensor(0), _r.scalar(1), _r.scalar(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _standard_gamma
static PyObject * THPVariable__standard_gamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_standard_gamma(Tensor input, Generator? generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_standard_gamma(Tensor self, Generator? generator=None) -> Tensor
  
  auto dispatch__standard_gamma = [](const at::Tensor & self, ::std::optional<at::Generator> generator) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_standard_gamma(self, generator);
  };
  return wrap(dispatch__standard_gamma(_r.tensor(0), _r.generator(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _dirichlet_grad
static PyObject * THPVariable__dirichlet_grad(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_dirichlet_grad(Tensor x, Tensor alpha, Tensor total)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_dirichlet_grad(Tensor x, Tensor alpha, Tensor total) -> Tensor
  
  auto dispatch__dirichlet_grad = [](const at::Tensor & x, const at::Tensor & alpha, const at::Tensor & total) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_dirichlet_grad(x, alpha, total);
  };
  return wrap(dispatch__dirichlet_grad(_r.tensor(0), _r.tensor(1), _r.tensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// binomial
static PyObject * THPVariable_binomial(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "binomial(Tensor count, Tensor prob, Generator? generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::binomial(Tensor count, Tensor prob, Generator? generator=None) -> Tensor
  
  auto dispatch_binomial = [](const at::Tensor & count, const at::Tensor & prob, ::std::optional<at::Generator> generator) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::binomial(count, prob, generator);
  };
  return wrap(dispatch_binomial(_r.tensor(0), _r.tensor(1), _r.generator(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sparse_csr_sum
static PyObject * THPVariable__sparse_csr_sum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_csr_sum(Tensor input, IntArrayRef[1] dim, bool keepdim=False, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_sparse_csr_sum.dim_dtype(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
  
  auto dispatch__sparse_csr_sum = [](const at::Tensor & self, at::IntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sparse_csr_sum(self, dim, keepdim, dtype);
  };
  return wrap(dispatch__sparse_csr_sum(_r.tensor(0), _r.intlist(1), _r.toBool(2), _r.scalartypeOptional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sparse_log_softmax_backward_data
static PyObject * THPVariable__sparse_log_softmax_backward_data(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_log_softmax_backward_data(Tensor grad_output, Tensor output, int64_t dim, Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_sparse_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
  
  auto dispatch__sparse_log_softmax_backward_data = [](const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sparse_log_softmax_backward_data(grad_output, output, dim, self);
  };
  return wrap(dispatch__sparse_log_softmax_backward_data(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.tensor(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// nuclear_norm
static PyObject * THPVariable_nuclear_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nuclear_norm(Tensor input, IntArrayRef[2] dim, bool keepdim=False, *, Tensor out=None)",
    "nuclear_norm(Tensor input, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::nuclear_norm.dim(Tensor self, int[2] dim, bool keepdim=False) -> Tensor
        
        auto dispatch_nuclear_norm = [](const at::Tensor & self, at::IntArrayRef dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::nuclear_norm(self, dim, keepdim);
        };
        return wrap(dispatch_nuclear_norm(_r.tensor(0), _r.intlist(1), _r.toBool(2)));
      } else {
        // aten::nuclear_norm.dim_out(Tensor self, int[2] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_nuclear_norm_out = [](at::Tensor out, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::nuclear_norm_out(out, self, dim, keepdim);
        };
        return wrap(dispatch_nuclear_norm_out(_r.tensor(3), _r.tensor(0), _r.intlist(1), _r.toBool(2)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::nuclear_norm(Tensor self, bool keepdim=False) -> Tensor
        
        auto dispatch_nuclear_norm = [](const at::Tensor & self, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::nuclear_norm(self, keepdim);
        };
        return wrap(dispatch_nuclear_norm(_r.tensor(0), _r.toBool(1)));
      } else {
        // aten::nuclear_norm.out(Tensor self, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_nuclear_norm_out = [](at::Tensor out, const at::Tensor & self, bool keepdim) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::nuclear_norm_out(out, self, keepdim);
        };
        return wrap(dispatch_nuclear_norm_out(_r.tensor(2), _r.tensor(0), _r.toBool(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// clone
static PyObject * THPVariable_clone(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "clone(Tensor input, *, MemoryFormat? memory_format=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
  
  auto dispatch_clone = [](const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.clone(memory_format);
  };
  return wrap(dispatch_clone(_r.tensor(0), _r.memoryformatOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// heaviside
static PyObject * THPVariable_heaviside(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "heaviside(Tensor input, Tensor values, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::heaviside(Tensor self, Tensor values) -> Tensor
    
    auto dispatch_heaviside = [](const at::Tensor & self, const at::Tensor & values) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.heaviside(values);
    };
    return wrap(dispatch_heaviside(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::heaviside.out(Tensor self, Tensor values, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_heaviside_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & values) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::heaviside_out(out, self, values);
    };
    return wrap(dispatch_heaviside_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _scaled_mm
static PyObject * THPVariable__scaled_mm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_scaled_mm(Tensor input, Tensor mat2, Tensor scale_a, Tensor scale_b, Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, bool use_fast_accum=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(8)) {
    // aten::_scaled_mm(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b, Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, bool use_fast_accum=False) -> Tensor
    
    auto dispatch__scaled_mm = [](const at::Tensor & self, const at::Tensor & mat2, const at::Tensor & scale_a, const at::Tensor & scale_b, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & scale_result, ::std::optional<at::ScalarType> out_dtype, bool use_fast_accum) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_scaled_mm(self, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum);
    };
    return wrap(dispatch__scaled_mm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.optionalTensor(4), _r.optionalTensor(5), _r.scalartypeOptional(6), _r.toBool(7)));
  } else {
    // aten::_scaled_mm.out(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b, Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, bool use_fast_accum=False, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__scaled_mm_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & mat2, const at::Tensor & scale_a, const at::Tensor & scale_b, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & scale_result, ::std::optional<at::ScalarType> out_dtype, bool use_fast_accum) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_scaled_mm_out(out, self, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum);
    };
    return wrap(dispatch__scaled_mm_out(_r.tensor(8), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.optionalTensor(4), _r.optionalTensor(5), _r.scalartypeOptional(6), _r.toBool(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _validate_sparse_compressed_tensor_args
static PyObject * THPVariable__validate_sparse_compressed_tensor_args(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_validate_sparse_compressed_tensor_args(Tensor compressed_indices, Tensor plain_indices, Tensor values, IntArrayRef size, Layout layout)",
  }, /*traceable=*/false);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_validate_sparse_compressed_tensor_args(Tensor compressed_indices, Tensor plain_indices, Tensor values, int[] size, Layout layout) -> ()
  
  auto dispatch__validate_sparse_compressed_tensor_args = [](const at::Tensor & compressed_indices, const at::Tensor & plain_indices, const at::Tensor & values, at::IntArrayRef size, at::Layout layout) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_validate_sparse_compressed_tensor_args(compressed_indices, plain_indices, values, size, layout);
  };
  dispatch__validate_sparse_compressed_tensor_args(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3), _r.layout(4));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _validate_sparse_csr_tensor_args
static PyObject * THPVariable__validate_sparse_csr_tensor_args(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_validate_sparse_csr_tensor_args(Tensor crow_indices, Tensor col_indices, Tensor values, IntArrayRef size)",
  }, /*traceable=*/false);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_validate_sparse_csr_tensor_args(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size) -> ()
  
  auto dispatch__validate_sparse_csr_tensor_args = [](const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::IntArrayRef size) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_validate_sparse_csr_tensor_args(crow_indices, col_indices, values, size);
  };
  dispatch__validate_sparse_csr_tensor_args(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.intlist(3));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _to_cpu
static PyObject * THPVariable__to_cpu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_to_cpu(TensorList tensors)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_to_cpu(Tensor[] tensors) -> Tensor[]
  
  auto dispatch__to_cpu = [](at::TensorList tensors) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_to_cpu(tensors);
  };
  return wrap(dispatch__to_cpu(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hspmm
static PyObject * THPVariable_hspmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hspmm(Tensor mat1, Tensor mat2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::hspmm(Tensor mat1, Tensor mat2) -> Tensor
    
    auto dispatch_hspmm = [](const at::Tensor & mat1, const at::Tensor & mat2) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::hspmm(mat1, mat2);
    };
    return wrap(dispatch_hspmm(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::hspmm.out(Tensor mat1, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_hspmm_out = [](at::Tensor out, const at::Tensor & mat1, const at::Tensor & mat2) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::hspmm_out(out, mat1, mat2);
    };
    return wrap(dispatch_hspmm_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _to_sparse_semi_structured
static PyObject * THPVariable__to_sparse_semi_structured(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_to_sparse_semi_structured(Tensor dense)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_to_sparse_semi_structured(Tensor dense) -> (Tensor, Tensor)
  
  auto dispatch__to_sparse_semi_structured = [](const at::Tensor & dense) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_to_sparse_semi_structured(dense);
  };
  return wrap(dispatch__to_sparse_semi_structured(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// q_per_channel_scales
static PyObject * THPVariable_q_per_channel_scales(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "q_per_channel_scales(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::q_per_channel_scales(Tensor self) -> Tensor
  
  auto dispatch_q_per_channel_scales = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.q_per_channel_scales();
  };
  return wrap(dispatch_q_per_channel_scales(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// q_per_channel_zero_points
static PyObject * THPVariable_q_per_channel_zero_points(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "q_per_channel_zero_points(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::q_per_channel_zero_points(Tensor self) -> Tensor
  
  auto dispatch_q_per_channel_zero_points = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.q_per_channel_zero_points();
  };
  return wrap(dispatch_q_per_channel_zero_points(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// q_per_channel_axis
static PyObject * THPVariable_q_per_channel_axis(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "q_per_channel_axis(Tensor input)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::q_per_channel_axis(Tensor self) -> int
  
  auto dispatch_q_per_channel_axis = [](const at::Tensor & self) -> int64_t {
    pybind11::gil_scoped_release no_gil;
    return self.q_per_channel_axis();
  };
  return wrap(dispatch_q_per_channel_axis(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _fake_quantize_per_tensor_affine_cachemask_tensor_qparams
static PyObject * THPVariable__fake_quantize_per_tensor_affine_cachemask_tensor_qparams(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get__fake_quantize_per_tensor_affine_cachemask_tensor_qparams_structseq();
  static PythonArgParser parser({
    "_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(Tensor input, Tensor scale, Tensor zero_point, Tensor fake_quant_enabled, int64_t quant_min, int64_t quant_max)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, Tensor fake_quant_enabled, int quant_min, int quant_max) -> (Tensor output, Tensor mask)
  
  auto dispatch__fake_quantize_per_tensor_affine_cachemask_tensor_qparams = [](const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, const at::Tensor & fake_quant_enabled, int64_t quant_min, int64_t quant_max) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(self, scale, zero_point, fake_quant_enabled, quant_min, quant_max);
  };
  return wrap(NamedTuple, dispatch__fake_quantize_per_tensor_affine_cachemask_tensor_qparams(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.toInt64(4), _r.toInt64(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _fake_quantize_learnable_per_tensor_affine
static PyObject * THPVariable__fake_quantize_learnable_per_tensor_affine(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_fake_quantize_learnable_per_tensor_affine(Tensor input, Tensor scale, Tensor zero_point, int64_t quant_min, int64_t quant_max, double grad_factor=1.0)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_fake_quantize_learnable_per_tensor_affine(Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max, float grad_factor=1.0) -> Tensor
  
  auto dispatch__fake_quantize_learnable_per_tensor_affine = [](const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t quant_min, int64_t quant_max, double grad_factor) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_fake_quantize_learnable_per_tensor_affine(self, scale, zero_point, quant_min, quant_max, grad_factor);
  };
  return wrap(dispatch__fake_quantize_learnable_per_tensor_affine(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3), _r.toInt64(4), _r.toDouble(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fused_moving_avg_obs_fake_quant
static PyObject * THPVariable_fused_moving_avg_obs_fake_quant(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fused_moving_avg_obs_fake_quant(Tensor input, Tensor observer_on, Tensor fake_quant_on, Tensor running_min, Tensor running_max, Tensor scale, Tensor zero_point, double averaging_const, int64_t quant_min, int64_t quant_max, int64_t ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False)",
  }, /*traceable=*/true);

  ParsedArgs<13> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::fused_moving_avg_obs_fake_quant(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> Tensor
  
  auto dispatch_fused_moving_avg_obs_fake_quant = [](const at::Tensor & self, const at::Tensor & observer_on, const at::Tensor & fake_quant_on, at::Tensor running_min, at::Tensor running_max, at::Tensor scale, at::Tensor zero_point, double averaging_const, int64_t quant_min, int64_t quant_max, int64_t ch_axis, bool per_row_fake_quant, bool symmetric_quant) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::fused_moving_avg_obs_fake_quant(self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant);
  };
  return wrap(dispatch_fused_moving_avg_obs_fake_quant(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.tensor(5), _r.tensor(6), _r.toDouble(7), _r.toInt64(8), _r.toInt64(9), _r.toInt64(10), _r.toBool(11), _r.toBool(12)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _saturate_weight_to_fp16
static PyObject * THPVariable__saturate_weight_to_fp16(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_saturate_weight_to_fp16(Tensor weight)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_saturate_weight_to_fp16(Tensor weight) -> Tensor
  
  auto dispatch__saturate_weight_to_fp16 = [](const at::Tensor & weight) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_saturate_weight_to_fp16(weight);
  };
  return wrap(dispatch__saturate_weight_to_fp16(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// choose_qparams_optimized
static PyObject * THPVariable_choose_qparams_optimized(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "choose_qparams_optimized(Tensor input, int64_t numel, int64_t n_bins, double ratio, int64_t bit_width)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::choose_qparams_optimized(Tensor input, int numel, int n_bins, float ratio, int bit_width) -> (Tensor, Tensor)
  
  auto dispatch_choose_qparams_optimized = [](const at::Tensor & input, int64_t numel, int64_t n_bins, double ratio, int64_t bit_width) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::choose_qparams_optimized(input, numel, n_bins, ratio, bit_width);
  };
  return wrap(dispatch_choose_qparams_optimized(_r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toDouble(3), _r.toInt64(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// combinations
static PyObject * THPVariable_combinations(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "combinations(Tensor input, int64_t r=2, bool with_replacement=False)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::combinations(Tensor self, int r=2, bool with_replacement=False) -> Tensor
  
  auto dispatch_combinations = [](const at::Tensor & self, int64_t r, bool with_replacement) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::combinations(self, r, with_replacement);
  };
  return wrap(dispatch_combinations(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// result_type
static PyObject * THPVariable_result_type(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "result_type(Tensor tensor, Tensor other)",
    "result_type(Scalar scalar, Tensor tensor)",
    "result_type(Tensor tensor, Scalar other)",
    "result_type(Scalar scalar1, Scalar scalar2)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::result_type.Tensor(Tensor tensor, Tensor other) -> ScalarType
      
      auto dispatch_result_type = [](const at::Tensor & tensor, const at::Tensor & other) -> at::ScalarType {
        pybind11::gil_scoped_release no_gil;
        return at::result_type(tensor, other);
      };
      return wrap(dispatch_result_type(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::result_type.Scalar_Tensor(Scalar scalar, Tensor tensor) -> ScalarType
      
      auto dispatch_result_type = [](const at::Scalar & scalar, const at::Tensor & tensor) -> at::ScalarType {
        pybind11::gil_scoped_release no_gil;
        return at::result_type(scalar, tensor);
      };
      return wrap(dispatch_result_type(_r.scalar(0), _r.tensor(1)));
    }
    case 2: {
      // aten::result_type.Scalar(Tensor tensor, Scalar other) -> ScalarType
      
      auto dispatch_result_type = [](const at::Tensor & tensor, const at::Scalar & other) -> at::ScalarType {
        pybind11::gil_scoped_release no_gil;
        return at::result_type(tensor, other);
      };
      return wrap(dispatch_result_type(_r.tensor(0), _r.scalar(1)));
    }
    case 3: {
      // aten::result_type.Scalar_Scalar(Scalar scalar1, Scalar scalar2) -> ScalarType
      
      auto dispatch_result_type = [](const at::Scalar & scalar1, const at::Scalar & scalar2) -> at::ScalarType {
        pybind11::gil_scoped_release no_gil;
        return at::result_type(scalar1, scalar2);
      };
      return wrap(dispatch_result_type(_r.scalar(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// can_cast
static PyObject * THPVariable_can_cast(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "can_cast(ScalarType from_, ScalarType to)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::can_cast(ScalarType from_, ScalarType to) -> bool
  
  auto dispatch_can_cast = [](at::ScalarType from_, at::ScalarType to) -> bool {
    pybind11::gil_scoped_release no_gil;
    return at::can_cast(from_, to);
  };
  return wrap(dispatch_can_cast(_r.scalartype(0), _r.scalartype(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _lstm_mps
static PyObject * THPVariable__lstm_mps(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_lstm_mps(Tensor input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_lstm_mps(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
  
  auto dispatch__lstm_mps = [](const at::Tensor & input, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_lstm_mps(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  };
  return wrap(dispatch__lstm_mps(_r.tensor(0), _r.tensorlist(1), _r.tensorlist(2), _r.toBool(3), _r.toInt64(4), _r.toDouble(5), _r.toBool(6), _r.toBool(7), _r.toBool(8)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// lstm
static PyObject * THPVariable_lstm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lstm(Tensor data, Tensor batch_sizes, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional)",
    "lstm(Tensor input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor, Tensor)
      
      auto dispatch_lstm = [](const at::Tensor & data, const at::Tensor & batch_sizes, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::lstm(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
      };
      return wrap(dispatch_lstm(_r.tensor(0), _r.tensor(1), _r.tensorlist(2), _r.tensorlist(3), _r.toBool(4), _r.toInt64(5), _r.toDouble(6), _r.toBool(7), _r.toBool(8)));
    }
    case 1: {
      // aten::lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)
      
      auto dispatch_lstm = [](const at::Tensor & input, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
      };
      return wrap(dispatch_lstm(_r.tensor(0), _r.tensorlist(1), _r.tensorlist(2), _r.toBool(3), _r.toInt64(4), _r.toDouble(5), _r.toBool(6), _r.toBool(7), _r.toBool(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// lstm_cell
static PyObject * THPVariable_lstm_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "lstm_cell(Tensor input, TensorList hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None)",
  }, /*traceable=*/false);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)
  
  auto dispatch_lstm_cell = [](const at::Tensor & input, at::TensorList hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const ::std::optional<at::Tensor> & b_ih, const ::std::optional<at::Tensor> & b_hh) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
  };
  return wrap(dispatch_lstm_cell(_r.tensor(0), _r.tensorlist(1), _r.tensor(2), _r.tensor(3), _r.optionalTensor(4), _r.optionalTensor(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// quantized_rnn_relu_cell
static PyObject * THPVariable_quantized_rnn_relu_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantized_rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh)",
  }, /*traceable=*/true);

  ParsedArgs<14> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::quantized_rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
  
  auto dispatch_quantized_rnn_relu_cell = [](const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::quantized_rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  };
  return wrap(dispatch_quantized_rnn_relu_cell(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.tensor(5), _r.tensor(6), _r.tensor(7), _r.tensor(8), _r.tensor(9), _r.scalar(10), _r.scalar(11), _r.scalar(12), _r.scalar(13)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// quantized_rnn_tanh_cell
static PyObject * THPVariable_quantized_rnn_tanh_cell(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "quantized_rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh)",
  }, /*traceable=*/true);

  ParsedArgs<14> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::quantized_rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
  
  auto dispatch_quantized_rnn_tanh_cell = [](const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::quantized_rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  };
  return wrap(dispatch_quantized_rnn_tanh_cell(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.tensor(3), _r.tensor(4), _r.tensor(5), _r.tensor(6), _r.tensor(7), _r.tensor(8), _r.tensor(9), _r.scalar(10), _r.scalar(11), _r.scalar(12), _r.scalar(13)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// index_add
static PyObject * THPVariable_index_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_add(Tensor input, int64_t dim, Tensor index, Tensor source, *, Scalar alpha=1, Tensor out=None)",
    "index_add(Tensor input, Dimname dim, Tensor index, Tensor source, *, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(5)) {
        // aten::index_add(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor
        
        auto dispatch_index_add = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.index_add(dim, index, source, alpha);
        };
        return wrap(dispatch_index_add(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3), _r.scalar(4)));
      } else {
        // aten::index_add.out(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_index_add_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::index_add_out(out, self, dim, index, source, alpha);
        };
        return wrap(dispatch_index_add_out(_r.tensor(5), _r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3), _r.scalar(4)));
      }
    }
    case 1: {
      // aten::index_add.dimname(Tensor self, Dimname dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor
      
      auto dispatch_index_add = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.index_add(dim, index, source, alpha);
      };
      return wrap(dispatch_index_add(_r.tensor(0), _r.dimname(1), _r.tensor(2), _r.tensor(3), _r.scalar(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// index_reduce
static PyObject * THPVariable_index_reduce(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "index_reduce(Tensor input, int64_t dim, Tensor index, Tensor source, c10::string_view reduce, *, bool include_self=True, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(6)) {
    // aten::index_reduce(Tensor self, int dim, Tensor index, Tensor source, str reduce, *, bool include_self=True) -> Tensor
    
    auto dispatch_index_reduce = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, c10::string_view reduce, bool include_self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.index_reduce(dim, index, source, reduce, include_self);
    };
    return wrap(dispatch_index_reduce(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3), _r.stringView(4), _r.toBool(5)));
  } else {
    // aten::index_reduce.out(Tensor self, int dim, Tensor index, Tensor source, str reduce, *, bool include_self=True, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_index_reduce_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, c10::string_view reduce, bool include_self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::index_reduce_out(out, self, dim, index, source, reduce, include_self);
    };
    return wrap(dispatch_index_reduce_out(_r.tensor(6), _r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3), _r.stringView(4), _r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// scatter_add
static PyObject * THPVariable_scatter_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "scatter_add(Tensor input, int64_t dim, Tensor index, Tensor src, *, Tensor out=None)",
    "scatter_add(Tensor input, Dimname dim, Tensor index, Tensor src)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
        
        auto dispatch_scatter_add = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.scatter_add(dim, index, src);
        };
        return wrap(dispatch_scatter_add(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3)));
      } else {
        // aten::scatter_add.out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_scatter_add_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::scatter_add_out(out, self, dim, index, src);
        };
        return wrap(dispatch_scatter_add_out(_r.tensor(4), _r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3)));
      }
    }
    case 1: {
      // aten::scatter_add.dimname(Tensor self, Dimname dim, Tensor index, Tensor src) -> Tensor
      
      auto dispatch_scatter_add = [](const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.scatter_add(dim, index, src);
      };
      return wrap(dispatch_scatter_add(_r.tensor(0), _r.dimname(1), _r.tensor(2), _r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// scatter_reduce
static PyObject * THPVariable_scatter_reduce(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "scatter_reduce(Tensor input, int64_t dim, Tensor index, Tensor src, c10::string_view reduce, *, bool include_self=True, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(6)) {
    // aten::scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor
    
    auto dispatch_scatter_reduce = [](const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, bool include_self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.scatter_reduce(dim, index, src, reduce, include_self);
    };
    return wrap(dispatch_scatter_reduce(_r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3), _r.stringView(4), _r.toBool(5)));
  } else {
    // aten::scatter_reduce.two_out(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_scatter_reduce_out = [](at::Tensor out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, bool include_self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::scatter_reduce_out(out, self, dim, index, src, reduce, include_self);
    };
    return wrap(dispatch_scatter_reduce_out(_r.tensor(6), _r.tensor(0), _r.toInt64(1), _r.tensor(2), _r.tensor(3), _r.stringView(4), _r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bitwise_and
static PyObject * THPVariable_bitwise_and(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bitwise_and(Tensor input, Tensor other, *, Tensor out=None)",
    "bitwise_and(Scalar self, Tensor other)",
    "bitwise_and(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_bitwise_and = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.bitwise_and(other);
        };
        return wrap(dispatch_bitwise_and(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::bitwise_and.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_bitwise_and_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bitwise_and_out(out, self, other);
        };
        return wrap(dispatch_bitwise_and_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      // aten::bitwise_and.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
      
      auto dispatch_bitwise_and = [](const at::Scalar & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::bitwise_and(self, other);
      };
      return wrap(dispatch_bitwise_and(_r.scalar(0), _r.tensor(1)));
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_bitwise_and = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.bitwise_and(other);
        };
        return wrap(dispatch_bitwise_and(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::bitwise_and.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_bitwise_and_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bitwise_and_out(out, self, other);
        };
        return wrap(dispatch_bitwise_and_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
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
  static PythonArgParser parser({
    "bitwise_xor(Tensor input, Tensor other, *, Tensor out=None)",
    "bitwise_xor(Scalar self, Tensor other)",
    "bitwise_xor(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_bitwise_xor = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.bitwise_xor(other);
        };
        return wrap(dispatch_bitwise_xor(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::bitwise_xor.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_bitwise_xor_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bitwise_xor_out(out, self, other);
        };
        return wrap(dispatch_bitwise_xor_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      // aten::bitwise_xor.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
      
      auto dispatch_bitwise_xor = [](const at::Scalar & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::bitwise_xor(self, other);
      };
      return wrap(dispatch_bitwise_xor(_r.scalar(0), _r.tensor(1)));
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::bitwise_xor.Scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_bitwise_xor = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.bitwise_xor(other);
        };
        return wrap(dispatch_bitwise_xor(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::bitwise_xor.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_bitwise_xor_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bitwise_xor_out(out, self, other);
        };
        return wrap(dispatch_bitwise_xor_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
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
  static PythonArgParser parser({
    "bitwise_left_shift(Tensor input, Tensor other, *, Tensor out=None)",
    "bitwise_left_shift(Scalar self, Tensor other)",
    "bitwise_left_shift(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_bitwise_left_shift = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.bitwise_left_shift(other);
        };
        return wrap(dispatch_bitwise_left_shift(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::bitwise_left_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_bitwise_left_shift_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bitwise_left_shift_out(out, self, other);
        };
        return wrap(dispatch_bitwise_left_shift_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      // aten::bitwise_left_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
      
      auto dispatch_bitwise_left_shift = [](const at::Scalar & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::bitwise_left_shift(self, other);
      };
      return wrap(dispatch_bitwise_left_shift(_r.scalar(0), _r.tensor(1)));
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::bitwise_left_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_bitwise_left_shift = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.bitwise_left_shift(other);
        };
        return wrap(dispatch_bitwise_left_shift(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::bitwise_left_shift.Tensor_Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_bitwise_left_shift_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bitwise_left_shift_out(out, self, other);
        };
        return wrap(dispatch_bitwise_left_shift_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
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
  static PythonArgParser parser({
    "__rshift__(Tensor input, Tensor other)",
    "__rshift__(Tensor input, Scalar other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::__rshift__.Tensor(Tensor self, Tensor other) -> Tensor
      
      auto dispatch___rshift__ = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__rshift__(other);
      };
      return wrap(dispatch___rshift__(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::__rshift__.Scalar(Tensor self, Scalar other) -> Tensor
      
      auto dispatch___rshift__ = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.__rshift__(other);
      };
      return wrap(dispatch___rshift__(_r.tensor(0), _r.scalar(1)));
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
  static PythonArgParser parser({
    "addbmm(Scalar beta, Tensor input, Scalar alpha, Tensor batch1, Tensor batch2, *, Tensor out=None)|deprecated",
    "addbmm(Scalar beta, Tensor input, Tensor batch1, Tensor batch2, *, Tensor out=None)|deprecated",
    "addbmm(Tensor input, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(5)) {
        // [deprecated] aten::addbmm(Scalar beta, Tensor self, Scalar alpha, Tensor batch1, Tensor batch2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addbmm = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & batch1, const at::Tensor & batch2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addbmm(batch1, batch2, beta, alpha);
        };
        return wrap(dispatch_addbmm(_r.tensor(5), _r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4)));
      } else {
        // [deprecated] aten::addbmm(Scalar beta, Tensor self, Scalar alpha, Tensor batch1, Tensor batch2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addbmm_out = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & batch1, const at::Tensor & batch2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addbmm_out(out, self, batch1, batch2, beta, alpha);
        };
        return wrap(dispatch_addbmm_out(_r.tensor(5), _r.scalar(0), _r.tensor(1), _r.scalar(2), _r.tensor(3), _r.tensor(4)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // [deprecated] aten::addbmm(Scalar beta, Tensor self, Tensor batch1, Tensor batch2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addbmm = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addbmm(batch1, batch2, beta, 1);
        };
        return wrap(dispatch_addbmm(_r.tensor(4), _r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
      } else {
        // [deprecated] aten::addbmm(Scalar beta, Tensor self, Tensor batch1, Tensor batch2, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addbmm_out = [](at::Tensor out, const at::Scalar & beta, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addbmm_out(out, self, batch1, batch2, beta, 1);
        };
        return wrap(dispatch_addbmm_out(_r.tensor(4), _r.scalar(0), _r.tensor(1), _r.tensor(2), _r.tensor(3)));
      }
    }
    case 2: {
      if (_r.isNone(5)) {
        // aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        
        auto dispatch_addbmm = [](const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.addbmm(batch1, batch2, beta, alpha);
        };
        return wrap(dispatch_addbmm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      } else {
        // aten::addbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_addbmm_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::addbmm_out(out, self, batch1, batch2, beta, alpha);
        };
        return wrap(dispatch_addbmm_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// diag
static PyObject * THPVariable_diag(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "diag(Tensor input, int64_t diagonal=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::diag(Tensor self, int diagonal=0) -> Tensor
    
    auto dispatch_diag = [](const at::Tensor & self, int64_t diagonal) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.diag(diagonal);
    };
    return wrap(dispatch_diag(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::diag.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_diag_out = [](at::Tensor out, const at::Tensor & self, int64_t diagonal) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::diag_out(out, self, diagonal);
    };
    return wrap(dispatch_diag_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cross
static PyObject * THPVariable_cross(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cross(Tensor input, Tensor other, int64_t? dim=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor
    
    auto dispatch_cross = [](const at::Tensor & self, const at::Tensor & other, ::std::optional<int64_t> dim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.cross(other, dim);
    };
    return wrap(dispatch_cross(_r.tensor(0), _r.tensor(1), _r.toInt64Optional(2)));
  } else {
    // aten::cross.out(Tensor self, Tensor other, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_cross_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other, ::std::optional<int64_t> dim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::cross_out(out, self, other, dim);
    };
    return wrap(dispatch_cross_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toInt64Optional(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// triu_indices
static PyObject * THPVariable_triu_indices(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "triu_indices(int64_t row, int64_t col, int64_t offset=0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  const auto options = TensorOptions()
      .dtype(_r.scalartypeWithDefault(3, at::kLong))
      .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
      .layout(_r.layoutOptional(4))
      .requires_grad(_r.toBool(7))
      .pinned_memory(_r.toBool(6));
  torch::utils::maybe_initialize_device(options);
  
  auto dispatch_triu_indices = [](int64_t row, int64_t col, int64_t offset, at::TensorOptions options) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return torch::triu_indices(row, col, offset, options);
  };
  return wrap(dispatch_triu_indices(_r.toInt64(0), _r.toInt64(1), _r.toInt64(2), options));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// trace
static PyObject * THPVariable_trace(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "trace(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::trace(Tensor self) -> Tensor
  
  auto dispatch_trace = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.trace();
  };
  return wrap(dispatch_trace(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// ne
static PyObject * THPVariable_ne(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "ne(Tensor input, Tensor other, *, Tensor out=None)",
    "ne(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::ne.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_ne = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.ne(other);
        };
        return wrap(dispatch_ne(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_ne_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::ne_out(out, self, other);
        };
        return wrap(dispatch_ne_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::ne.Scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_ne = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.ne(other);
        };
        return wrap(dispatch_ne(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_ne_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::ne_out(out, self, other);
        };
        return wrap(dispatch_ne_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
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
  static PythonArgParser parser({
    "eq(Tensor input, Tensor other, *, Tensor out=None)",
    "eq(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::eq.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_eq = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.eq(other);
        };
        return wrap(dispatch_eq(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_eq_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::eq_out(out, self, other);
        };
        return wrap(dispatch_eq_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::eq.Scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_eq = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.eq(other);
        };
        return wrap(dispatch_eq(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_eq_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::eq_out(out, self, other);
        };
        return wrap(dispatch_eq_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// greater
static PyObject * THPVariable_greater(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "greater(Tensor input, Tensor other, *, Tensor out=None)",
    "greater(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::greater.Tensor(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_greater = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.greater(other);
        };
        return wrap(dispatch_greater(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::greater.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_greater_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::greater_out(out, self, other);
        };
        return wrap(dispatch_greater_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::greater.Scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_greater = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return self.greater(other);
        };
        return wrap(dispatch_greater(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::greater.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_greater_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::greater_out(out, self, other);
        };
        return wrap(dispatch_greater_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// take
static PyObject * THPVariable_take(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "take(Tensor input, Tensor index, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::take(Tensor self, Tensor index) -> Tensor
    
    auto dispatch_take = [](const at::Tensor & self, const at::Tensor & index) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.take(index);
    };
    return wrap(dispatch_take(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::take.out(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_take_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & index) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::take_out(out, self, index);
    };
    return wrap(dispatch_take_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _linalg_check_errors
static PyObject * THPVariable__linalg_check_errors(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_linalg_check_errors(Tensor info, c10::string_view api_name, *, bool is_matrix)",
  }, /*traceable=*/false);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_linalg_check_errors(Tensor info, str api_name, *, bool is_matrix) -> ()
  
  auto dispatch__linalg_check_errors = [](const at::Tensor & info, c10::string_view api_name, bool is_matrix) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_linalg_check_errors(info, api_name, is_matrix);
  };
  dispatch__linalg_check_errors(_r.tensor(0), _r.stringView(1), _r.toBool(2));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// svd
static PyObject * THPVariable_svd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_svd_out_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_svd_structseq();
  static PythonArgParser parser({
    "svd(Tensor input, bool some=True, bool compute_uv=True, *, TensorList[3] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)
    
    auto dispatch_svd = [](const at::Tensor & self, bool some, bool compute_uv) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return self.svd(some, compute_uv);
    };
    return wrap(NamedTuple1, dispatch_svd(_r.tensor(0), _r.toBool(1), _r.toBool(2)));
  } else {
    // aten::svd.U(Tensor self, bool some=True, bool compute_uv=True, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) V) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) V)
    auto out = _r.tensorlist_n<3>(3);
    auto dispatch_svd_out = [](at::Tensor & U, at::Tensor & S, at::Tensor & V, const at::Tensor & self, bool some, bool compute_uv) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::svd_out(U, S, V, self, some, compute_uv);
    };
    return wrap(NamedTuple, dispatch_svd_out(out[0], out[1], out[2], _r.tensor(0), _r.toBool(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cholesky
static PyObject * THPVariable_cholesky(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cholesky(Tensor input, bool upper=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::cholesky(Tensor self, bool upper=False) -> Tensor
    
    auto dispatch_cholesky = [](const at::Tensor & self, bool upper) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.cholesky(upper);
    };
    return wrap(dispatch_cholesky(_r.tensor(0), _r.toBool(1)));
  } else {
    // aten::cholesky.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_cholesky_out = [](at::Tensor out, const at::Tensor & self, bool upper) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::cholesky_out(out, self, upper);
    };
    return wrap(dispatch_cholesky_out(_r.tensor(2), _r.tensor(0), _r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _lu_with_info
static PyObject * THPVariable__lu_with_info(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get__lu_with_info_structseq();
  static PythonArgParser parser({
    "_lu_with_info(Tensor input, bool pivot=True, bool check_errors=True)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_lu_with_info(Tensor self, bool pivot=True, bool check_errors=True) -> (Tensor LU, Tensor pivots, Tensor info)
  
  auto dispatch__lu_with_info = [](const at::Tensor & self, bool pivot, bool check_errors) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_lu_with_info(self, pivot, check_errors);
  };
  return wrap(NamedTuple, dispatch__lu_with_info(_r.tensor(0), _r.toBool(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// lu_unpack
static PyObject * THPVariable_lu_unpack(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_lu_unpack_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_lu_unpack_out_structseq();
  static PythonArgParser parser({
    "lu_unpack(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True, *, TensorList[3] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::lu_unpack(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True) -> (Tensor P, Tensor L, Tensor U)
    
    auto dispatch_lu_unpack = [](const at::Tensor & LU_data, const at::Tensor & LU_pivots, bool unpack_data, bool unpack_pivots) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::lu_unpack(LU_data, LU_pivots, unpack_data, unpack_pivots);
    };
    return wrap(NamedTuple, dispatch_lu_unpack(_r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3)));
  } else {
    // aten::lu_unpack.out(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True, *, Tensor(a!) P, Tensor(b!) L, Tensor(c!) U) -> (Tensor(a!) P, Tensor(b!) L, Tensor(c!) U)
    auto out = _r.tensorlist_n<3>(4);
    auto dispatch_lu_unpack_out = [](at::Tensor & P, at::Tensor & L, at::Tensor & U, const at::Tensor & LU_data, const at::Tensor & LU_pivots, bool unpack_data, bool unpack_pivots) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::lu_unpack_out(P, L, U, LU_data, LU_pivots, unpack_data, unpack_pivots);
    };
    return wrap(NamedTuple1, dispatch_lu_unpack_out(out[0], out[1], out[2], _r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// multinomial
static PyObject * THPVariable_multinomial(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "multinomial(Tensor input, int64_t num_samples, bool replacement=False, *, Generator? generator=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor
    
    auto dispatch_multinomial = [](const at::Tensor & self, int64_t num_samples, bool replacement, ::std::optional<at::Generator> generator) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.multinomial(num_samples, replacement, generator);
    };
    return wrap(dispatch_multinomial(_r.tensor(0), _r.toInt64(1), _r.toBool(2), _r.generator(3)));
  } else {
    // aten::multinomial.out(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_multinomial_out = [](at::Tensor out, const at::Tensor & self, int64_t num_samples, bool replacement, ::std::optional<at::Generator> generator) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::multinomial_out(out, self, num_samples, replacement, generator);
    };
    return wrap(dispatch_multinomial_out(_r.tensor(4), _r.tensor(0), _r.toInt64(1), _r.toBool(2), _r.generator(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// i0
static PyObject * THPVariable_i0(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "i0(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::i0(Tensor self) -> Tensor
    
    auto dispatch_i0 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.i0();
    };
    return wrap(dispatch_i0(_r.tensor(0)));
  } else {
    // aten::i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_i0_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::i0_out(out, self);
    };
    return wrap(dispatch_i0_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// i0_
static PyObject * THPVariable_i0_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "i0_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::i0_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_i0_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.i0_();
  };
  return wrap(dispatch_i0_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// atan2
static PyObject * THPVariable_atan2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "atan2(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::atan2(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_atan2 = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.atan2(other);
    };
    return wrap(dispatch_atan2(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_atan2_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::atan2_out(out, self, other);
    };
    return wrap(dispatch_atan2_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hypot
static PyObject * THPVariable_hypot(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hypot(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::hypot(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_hypot = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.hypot(other);
    };
    return wrap(dispatch_hypot(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::hypot.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_hypot_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::hypot_out(out, self, other);
    };
    return wrap(dispatch_hypot_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// igamma
static PyObject * THPVariable_igamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "igamma(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::igamma(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_igamma = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.igamma(other);
    };
    return wrap(dispatch_igamma(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::igamma.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_igamma_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::igamma_out(out, self, other);
    };
    return wrap(dispatch_igamma_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// nextafter
static PyObject * THPVariable_nextafter(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nextafter(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::nextafter(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_nextafter = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.nextafter(other);
    };
    return wrap(dispatch_nextafter(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::nextafter.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_nextafter_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::nextafter_out(out, self, other);
    };
    return wrap(dispatch_nextafter_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fmin
static PyObject * THPVariable_fmin(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fmin(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::fmin(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_fmin = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.fmin(other);
    };
    return wrap(dispatch_fmin(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::fmin.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fmin_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fmin_out(out, self, other);
    };
    return wrap(dispatch_fmin_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// sort
static PyObject * THPVariable_sort(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_sort_out_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_sort_structseq();
  static PythonArgParser parser({
    "sort(Tensor input, *, bool? stable, int64_t dim=-1, bool descending=False, TensorList[2] out=None)",
    "sort(Tensor input, int64_t dim=-1, bool descending=False, *, TensorList[2] out=None)",
    "sort(Tensor input, *, bool? stable, Dimname dim, bool descending=False, TensorList[2] out=None)",
    "sort(Tensor input, Dimname dim, bool descending=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // aten::sort.stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
        
        auto dispatch_sort = [](const at::Tensor & self, ::std::optional<bool> stable, int64_t dim, bool descending) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.sort(stable, dim, descending);
        };
        return wrap(NamedTuple1, dispatch_sort(_r.tensor(0), _r.toBoolOptional(1), _r.toInt64(2), _r.toBool(3)));
      } else {
        // aten::sort.values_stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(4);
        auto dispatch_sort_out = [](at::Tensor & values, at::Tensor & indices, const at::Tensor & self, ::std::optional<bool> stable, int64_t dim, bool descending) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::sort_out(values, indices, self, stable, dim, descending);
        };
        return wrap(NamedTuple, dispatch_sort_out(out[0], out[1], _r.tensor(0), _r.toBoolOptional(1), _r.toInt64(2), _r.toBool(3)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
        
        auto dispatch_sort = [](const at::Tensor & self, int64_t dim, bool descending) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.sort(dim, descending);
        };
        return wrap(NamedTuple1, dispatch_sort(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      } else {
        // aten::sort.values(Tensor self, int dim=-1, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_sort_out = [](at::Tensor & values, at::Tensor & indices, const at::Tensor & self, int64_t dim, bool descending) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::sort_out(values, indices, self, dim, descending);
        };
        return wrap(NamedTuple, dispatch_sort_out(out[0], out[1], _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
      }
    }
    case 2: {
      if (_r.isNone(4)) {
        // aten::sort.dimname_stable(Tensor self, *, bool? stable, Dimname dim, bool descending=False) -> (Tensor values, Tensor indices)
        
        auto dispatch_sort = [](const at::Tensor & self, ::std::optional<bool> stable, at::Dimname dim, bool descending) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.sort(stable, dim, descending);
        };
        return wrap(NamedTuple1, dispatch_sort(_r.tensor(0), _r.toBoolOptional(1), _r.dimname(2), _r.toBool(3)));
      } else {
        // aten::sort.dimname_values_stable(Tensor self, *, bool? stable, Dimname dim, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(4);
        auto dispatch_sort_out = [](at::Tensor & values, at::Tensor & indices, const at::Tensor & self, ::std::optional<bool> stable, at::Dimname dim, bool descending) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::sort_out(values, indices, self, stable, dim, descending);
        };
        return wrap(NamedTuple, dispatch_sort_out(out[0], out[1], _r.tensor(0), _r.toBoolOptional(1), _r.dimname(2), _r.toBool(3)));
      }
    }
    case 3: {
      if (_r.isNone(3)) {
        // aten::sort.dimname(Tensor self, Dimname dim, bool descending=False) -> (Tensor values, Tensor indices)
        
        auto dispatch_sort = [](const at::Tensor & self, at::Dimname dim, bool descending) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return self.sort(dim, descending);
        };
        return wrap(NamedTuple1, dispatch_sort(_r.tensor(0), _r.dimname(1), _r.toBool(2)));
      } else {
        // aten::sort.dimname_values(Tensor self, Dimname dim, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
        auto out = _r.tensorlist_n<2>(3);
        auto dispatch_sort_out = [](at::Tensor & values, at::Tensor & indices, const at::Tensor & self, at::Dimname dim, bool descending) -> ::std::tuple<at::Tensor,at::Tensor> {
          pybind11::gil_scoped_release no_gil;
          return at::sort_out(values, indices, self, dim, descending);
        };
        return wrap(NamedTuple, dispatch_sort_out(out[0], out[1], _r.tensor(0), _r.dimname(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// renorm
static PyObject * THPVariable_renorm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "renorm(Tensor input, Scalar p, int64_t dim, Scalar maxnorm, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(4)) {
    // aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor
    
    auto dispatch_renorm = [](const at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.renorm(p, dim, maxnorm);
    };
    return wrap(dispatch_renorm(_r.tensor(0), _r.scalar(1), _r.toInt64(2), _r.scalar(3)));
  } else {
    // aten::renorm.out(Tensor self, Scalar p, int dim, Scalar maxnorm, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_renorm_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::renorm_out(out, self, p, dim, maxnorm);
    };
    return wrap(dispatch_renorm_out(_r.tensor(4), _r.tensor(0), _r.scalar(1), _r.toInt64(2), _r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _amp_foreach_non_finite_check_and_unscale_
static PyObject * THPVariable__amp_foreach_non_finite_check_and_unscale_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_amp_foreach_non_finite_check_and_unscale_(TensorList self, Tensor found_inf, Tensor inv_scale)",
  }, /*traceable=*/false);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_amp_foreach_non_finite_check_and_unscale_(Tensor(a!)[] self, Tensor(b!) found_inf, Tensor inv_scale) -> ()
  
  auto dispatch__amp_foreach_non_finite_check_and_unscale_ = [](at::TensorList self, at::Tensor found_inf, const at::Tensor & inv_scale) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_amp_foreach_non_finite_check_and_unscale_(self, found_inf, inv_scale);
  };
  dispatch__amp_foreach_non_finite_check_and_unscale_(_r.tensorlist(0), _r.tensor(1), _r.tensor(2));
  Py_RETURN_NONE;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _amp_update_scale_
static PyObject * THPVariable__amp_update_scale_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_amp_update_scale_(Tensor input, Tensor growth_tracker, Tensor found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_amp_update_scale_(Tensor(a!) self, Tensor(b!) growth_tracker, Tensor found_inf, float scale_growth_factor, float scale_backoff_factor, int growth_interval) -> Tensor(a!)
  
  auto dispatch__amp_update_scale_ = [](at::Tensor self, at::Tensor growth_tracker, const at::Tensor & found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_amp_update_scale_(self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);
  };
  return wrap(dispatch__amp_update_scale_(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toDouble(3), _r.toDouble(4), _r.toInt64(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_mul
static PyObject * THPVariable__foreach_mul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_mul(TensorList self, ScalarList scalars)",
    "_foreach_mul(TensorList self, Tensor other)",
    "_foreach_mul(TensorList self, TensorList other)",
    "_foreach_mul(TensorList self, Scalar scalar)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_mul.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
      
      auto dispatch__foreach_mul = [](at::TensorList self, at::ArrayRef<at::Scalar> scalars) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_mul(self, scalars);
      };
      return wrap(dispatch__foreach_mul(_r.tensorlist(0), _r.scalarlist(1)));
    }
    case 1: {
      // aten::_foreach_mul.Tensor(Tensor[] self, Tensor other) -> Tensor[]
      
      auto dispatch__foreach_mul = [](at::TensorList self, const at::Tensor & other) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_mul(self, other);
      };
      return wrap(dispatch__foreach_mul(_r.tensorlist(0), _r.tensor(1)));
    }
    case 2: {
      // aten::_foreach_mul.List(Tensor[] self, Tensor[] other) -> Tensor[]
      
      auto dispatch__foreach_mul = [](at::TensorList self, at::TensorList other) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_mul(self, other);
      };
      return wrap(dispatch__foreach_mul(_r.tensorlist(0), _r.tensorlist(1)));
    }
    case 3: {
      // aten::_foreach_mul.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
      
      auto dispatch__foreach_mul = [](at::TensorList self, const at::Scalar & scalar) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_mul(self, scalar);
      };
      return wrap(dispatch__foreach_mul(_r.tensorlist(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_mul_
static PyObject * THPVariable__foreach_mul_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_mul_(TensorList self, ScalarList scalars)",
    "_foreach_mul_(TensorList self, Tensor other)",
    "_foreach_mul_(TensorList self, TensorList other)",
    "_foreach_mul_(TensorList self, Scalar scalar)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_mul_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
      
      auto dispatch__foreach_mul_ = [](at::TensorList self, at::ArrayRef<at::Scalar> scalars) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_mul_(self, scalars);
      };
      dispatch__foreach_mul_(_r.tensorlist(0), _r.scalarlist(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 1: {
      // aten::_foreach_mul_.Tensor(Tensor(a!)[] self, Tensor other) -> ()
      
      auto dispatch__foreach_mul_ = [](at::TensorList self, const at::Tensor & other) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_mul_(self, other);
      };
      dispatch__foreach_mul_(_r.tensorlist(0), _r.tensor(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 2: {
      // aten::_foreach_mul_.List(Tensor(a!)[] self, Tensor[] other) -> ()
      
      auto dispatch__foreach_mul_ = [](at::TensorList self, at::TensorList other) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_mul_(self, other);
      };
      dispatch__foreach_mul_(_r.tensorlist(0), _r.tensorlist(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 3: {
      // aten::_foreach_mul_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
      
      auto dispatch__foreach_mul_ = [](at::TensorList self, const at::Scalar & scalar) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_mul_(self, scalar);
      };
      dispatch__foreach_mul_(_r.tensorlist(0), _r.scalar(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_maximum
static PyObject * THPVariable__foreach_maximum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_maximum(TensorList self, Scalar scalar)",
    "_foreach_maximum(TensorList self, ScalarList scalars)",
    "_foreach_maximum(TensorList self, TensorList other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_maximum.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
      
      auto dispatch__foreach_maximum = [](at::TensorList self, const at::Scalar & scalar) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_maximum(self, scalar);
      };
      return wrap(dispatch__foreach_maximum(_r.tensorlist(0), _r.scalar(1)));
    }
    case 1: {
      // aten::_foreach_maximum.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
      
      auto dispatch__foreach_maximum = [](at::TensorList self, at::ArrayRef<at::Scalar> scalars) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_maximum(self, scalars);
      };
      return wrap(dispatch__foreach_maximum(_r.tensorlist(0), _r.scalarlist(1)));
    }
    case 2: {
      // aten::_foreach_maximum.List(Tensor[] self, Tensor[] other) -> Tensor[]
      
      auto dispatch__foreach_maximum = [](at::TensorList self, at::TensorList other) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_maximum(self, other);
      };
      return wrap(dispatch__foreach_maximum(_r.tensorlist(0), _r.tensorlist(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_maximum_
static PyObject * THPVariable__foreach_maximum_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_maximum_(TensorList self, Scalar scalar)",
    "_foreach_maximum_(TensorList self, ScalarList scalars)",
    "_foreach_maximum_(TensorList self, TensorList other)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_maximum_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
      
      auto dispatch__foreach_maximum_ = [](at::TensorList self, const at::Scalar & scalar) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_maximum_(self, scalar);
      };
      dispatch__foreach_maximum_(_r.tensorlist(0), _r.scalar(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 1: {
      // aten::_foreach_maximum_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
      
      auto dispatch__foreach_maximum_ = [](at::TensorList self, at::ArrayRef<at::Scalar> scalars) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_maximum_(self, scalars);
      };
      dispatch__foreach_maximum_(_r.tensorlist(0), _r.scalarlist(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 2: {
      // aten::_foreach_maximum_.List(Tensor(a!)[] self, Tensor[] other) -> ()
      
      auto dispatch__foreach_maximum_ = [](at::TensorList self, at::TensorList other) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_maximum_(self, other);
      };
      dispatch__foreach_maximum_(_r.tensorlist(0), _r.tensorlist(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_minimum
static PyObject * THPVariable__foreach_minimum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_minimum(TensorList self, Scalar scalar)",
    "_foreach_minimum(TensorList self, ScalarList scalars)",
    "_foreach_minimum(TensorList self, TensorList other)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_minimum.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
      
      auto dispatch__foreach_minimum = [](at::TensorList self, const at::Scalar & scalar) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_minimum(self, scalar);
      };
      return wrap(dispatch__foreach_minimum(_r.tensorlist(0), _r.scalar(1)));
    }
    case 1: {
      // aten::_foreach_minimum.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
      
      auto dispatch__foreach_minimum = [](at::TensorList self, at::ArrayRef<at::Scalar> scalars) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_minimum(self, scalars);
      };
      return wrap(dispatch__foreach_minimum(_r.tensorlist(0), _r.scalarlist(1)));
    }
    case 2: {
      // aten::_foreach_minimum.List(Tensor[] self, Tensor[] other) -> Tensor[]
      
      auto dispatch__foreach_minimum = [](at::TensorList self, at::TensorList other) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_minimum(self, other);
      };
      return wrap(dispatch__foreach_minimum(_r.tensorlist(0), _r.tensorlist(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_minimum_
static PyObject * THPVariable__foreach_minimum_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_minimum_(TensorList self, Scalar scalar)",
    "_foreach_minimum_(TensorList self, ScalarList scalars)",
    "_foreach_minimum_(TensorList self, TensorList other)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_minimum_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
      
      auto dispatch__foreach_minimum_ = [](at::TensorList self, const at::Scalar & scalar) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_minimum_(self, scalar);
      };
      dispatch__foreach_minimum_(_r.tensorlist(0), _r.scalar(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 1: {
      // aten::_foreach_minimum_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
      
      auto dispatch__foreach_minimum_ = [](at::TensorList self, at::ArrayRef<at::Scalar> scalars) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_minimum_(self, scalars);
      };
      dispatch__foreach_minimum_(_r.tensorlist(0), _r.scalarlist(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 2: {
      // aten::_foreach_minimum_.List(Tensor(a!)[] self, Tensor[] other) -> ()
      
      auto dispatch__foreach_minimum_ = [](at::TensorList self, at::TensorList other) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_minimum_(self, other);
      };
      dispatch__foreach_minimum_(_r.tensorlist(0), _r.tensorlist(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_addcdiv
static PyObject * THPVariable__foreach_addcdiv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_addcdiv(TensorList self, TensorList tensor1, TensorList tensor2, ScalarList scalars)",
    "_foreach_addcdiv(TensorList self, TensorList tensor1, TensorList tensor2, Tensor scalars)",
    "_foreach_addcdiv(TensorList self, TensorList tensor1, TensorList tensor2, Scalar value=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_addcdiv.ScalarList(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> Tensor[]
      
      auto dispatch__foreach_addcdiv = [](at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_addcdiv(self, tensor1, tensor2, scalars);
      };
      return wrap(dispatch__foreach_addcdiv(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.scalarlist(3)));
    }
    case 1: {
      // aten::_foreach_addcdiv.Tensor(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars) -> Tensor[]
      
      auto dispatch__foreach_addcdiv = [](at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_addcdiv(self, tensor1, tensor2, scalars);
      };
      return wrap(dispatch__foreach_addcdiv(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.tensor(3)));
    }
    case 2: {
      // aten::_foreach_addcdiv.Scalar(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> Tensor[]
      
      auto dispatch__foreach_addcdiv = [](at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_addcdiv(self, tensor1, tensor2, value);
      };
      return wrap(dispatch__foreach_addcdiv(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.scalar(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_addcdiv_
static PyObject * THPVariable__foreach_addcdiv_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_addcdiv_(TensorList self, TensorList tensor1, TensorList tensor2, ScalarList scalars)",
    "_foreach_addcdiv_(TensorList self, TensorList tensor1, TensorList tensor2, Tensor scalars)",
    "_foreach_addcdiv_(TensorList self, TensorList tensor1, TensorList tensor2, Scalar value=1)",
  }, /*traceable=*/false);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_addcdiv_.ScalarList(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> ()
      
      auto dispatch__foreach_addcdiv_ = [](at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_addcdiv_(self, tensor1, tensor2, scalars);
      };
      dispatch__foreach_addcdiv_(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.scalarlist(3));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 1: {
      // aten::_foreach_addcdiv_.Tensor(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars) -> ()
      
      auto dispatch__foreach_addcdiv_ = [](at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_addcdiv_(self, tensor1, tensor2, scalars);
      };
      dispatch__foreach_addcdiv_(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.tensor(3));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 2: {
      // aten::_foreach_addcdiv_.Scalar(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> ()
      
      auto dispatch__foreach_addcdiv_ = [](at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_addcdiv_(self, tensor1, tensor2, value);
      };
      dispatch__foreach_addcdiv_(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.scalar(3));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_abs
static PyObject * THPVariable__foreach_abs(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_abs(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_abs(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_abs = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_abs(self);
  };
  return wrap(dispatch__foreach_abs(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_abs_
static PyObject * THPVariable__foreach_abs_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_abs_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_abs_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_abs_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_abs_(self);
  };
  dispatch__foreach_abs_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_erf
static PyObject * THPVariable__foreach_erf(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_erf(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_erf(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_erf = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_erf(self);
  };
  return wrap(dispatch__foreach_erf(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_erf_
static PyObject * THPVariable__foreach_erf_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_erf_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_erf_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_erf_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_erf_(self);
  };
  dispatch__foreach_erf_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_frac
static PyObject * THPVariable__foreach_frac(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_frac(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_frac(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_frac = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_frac(self);
  };
  return wrap(dispatch__foreach_frac(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_frac_
static PyObject * THPVariable__foreach_frac_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_frac_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_frac_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_frac_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_frac_(self);
  };
  dispatch__foreach_frac_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_lgamma
static PyObject * THPVariable__foreach_lgamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_lgamma(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_lgamma(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_lgamma = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_lgamma(self);
  };
  return wrap(dispatch__foreach_lgamma(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_lgamma_
static PyObject * THPVariable__foreach_lgamma_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_lgamma_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_lgamma_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_lgamma_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_lgamma_(self);
  };
  dispatch__foreach_lgamma_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_pow
static PyObject * THPVariable__foreach_pow(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_pow(Scalar self, TensorList exponent)",
    "_foreach_pow(TensorList self, Scalar exponent)",
    "_foreach_pow(TensorList self, ScalarList exponent)",
    "_foreach_pow(TensorList self, TensorList exponent)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_pow.ScalarAndTensor(Scalar self, Tensor[] exponent) -> Tensor[]
      
      auto dispatch__foreach_pow = [](const at::Scalar & self, at::TensorList exponent) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_pow(self, exponent);
      };
      return wrap(dispatch__foreach_pow(_r.scalar(0), _r.tensorlist(1)));
    }
    case 1: {
      // aten::_foreach_pow.Scalar(Tensor[] self, Scalar exponent) -> Tensor[]
      
      auto dispatch__foreach_pow = [](at::TensorList self, const at::Scalar & exponent) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_pow(self, exponent);
      };
      return wrap(dispatch__foreach_pow(_r.tensorlist(0), _r.scalar(1)));
    }
    case 2: {
      // aten::_foreach_pow.ScalarList(Tensor[] self, Scalar[] exponent) -> Tensor[]
      
      auto dispatch__foreach_pow = [](at::TensorList self, at::ArrayRef<at::Scalar> exponent) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_pow(self, exponent);
      };
      return wrap(dispatch__foreach_pow(_r.tensorlist(0), _r.scalarlist(1)));
    }
    case 3: {
      // aten::_foreach_pow.List(Tensor[] self, Tensor[] exponent) -> Tensor[]
      
      auto dispatch__foreach_pow = [](at::TensorList self, at::TensorList exponent) -> ::std::vector<at::Tensor> {
        pybind11::gil_scoped_release no_gil;
        return at::_foreach_pow(self, exponent);
      };
      return wrap(dispatch__foreach_pow(_r.tensorlist(0), _r.tensorlist(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _foreach_pow_
static PyObject * THPVariable__foreach_pow_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_pow_(TensorList self, Scalar exponent)",
    "_foreach_pow_(TensorList self, ScalarList exponent)",
    "_foreach_pow_(TensorList self, TensorList exponent)",
  }, /*traceable=*/false);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_foreach_pow_.Scalar(Tensor(a!)[] self, Scalar exponent) -> ()
      
      auto dispatch__foreach_pow_ = [](at::TensorList self, const at::Scalar & exponent) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_pow_(self, exponent);
      };
      dispatch__foreach_pow_(_r.tensorlist(0), _r.scalar(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 1: {
      // aten::_foreach_pow_.ScalarList(Tensor(a!)[] self, Scalar[] exponent) -> ()
      
      auto dispatch__foreach_pow_ = [](at::TensorList self, at::ArrayRef<at::Scalar> exponent) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_pow_(self, exponent);
      };
      dispatch__foreach_pow_(_r.tensorlist(0), _r.scalarlist(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
    case 2: {
      // aten::_foreach_pow_.List(Tensor(a!)[] self, Tensor[] exponent) -> ()
      
      auto dispatch__foreach_pow_ = [](at::TensorList self, at::TensorList exponent) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_foreach_pow_(self, exponent);
      };
      dispatch__foreach_pow_(_r.tensorlist(0), _r.tensorlist(1));
      PyObject* self_tensorlist = _r.args[0];
      Py_INCREF(self_tensorlist);
      return self_tensorlist;
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_reciprocal
static PyObject * THPVariable__foreach_reciprocal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_reciprocal(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_reciprocal(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_reciprocal = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_reciprocal(self);
  };
  return wrap(dispatch__foreach_reciprocal(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_reciprocal_
static PyObject * THPVariable__foreach_reciprocal_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_reciprocal_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_reciprocal_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_reciprocal_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_reciprocal_(self);
  };
  dispatch__foreach_reciprocal_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_round
static PyObject * THPVariable__foreach_round(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_round(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_round(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_round = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_round(self);
  };
  return wrap(dispatch__foreach_round(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_round_
static PyObject * THPVariable__foreach_round_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_round_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_round_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_round_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_round_(self);
  };
  dispatch__foreach_round_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_sinh
static PyObject * THPVariable__foreach_sinh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_sinh(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_sinh(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_sinh = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_sinh(self);
  };
  return wrap(dispatch__foreach_sinh(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_sinh_
static PyObject * THPVariable__foreach_sinh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_sinh_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_sinh_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_sinh_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_sinh_(self);
  };
  dispatch__foreach_sinh_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_tan
static PyObject * THPVariable__foreach_tan(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_tan(TensorList self)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_tan(Tensor[] self) -> Tensor[]
  
  auto dispatch__foreach_tan = [](at::TensorList self) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_foreach_tan(self);
  };
  return wrap(dispatch__foreach_tan(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_tan_
static PyObject * THPVariable__foreach_tan_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_tan_(TensorList self)",
  }, /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_tan_(Tensor(a!)[] self) -> ()
  
  auto dispatch__foreach_tan_ = [](at::TensorList self) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_tan_(self);
  };
  dispatch__foreach_tan_(_r.tensorlist(0));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _foreach_copy_
static PyObject * THPVariable__foreach_copy_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_foreach_copy_(TensorList self, TensorList src, bool non_blocking=False)",
  }, /*traceable=*/false);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_foreach_copy_(Tensor(a!)[] self, Tensor[] src, bool non_blocking=False) -> ()
  
  auto dispatch__foreach_copy_ = [](at::TensorList self, at::TensorList src, bool non_blocking) -> void {
    pybind11::gil_scoped_release no_gil;
    at::_foreach_copy_(self, src, non_blocking);
  };
  dispatch__foreach_copy_(_r.tensorlist(0), _r.tensorlist(1), _r.toBool(2));
  PyObject* self_tensorlist = _r.args[0];
  Py_INCREF(self_tensorlist);
  return self_tensorlist;
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// bucketize
static PyObject * THPVariable_bucketize(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "bucketize(Tensor input, Tensor boundaries, *, bool out_int32=False, bool right=False, Tensor out=None)",
    "bucketize(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // aten::bucketize.Tensor(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor
        
        auto dispatch_bucketize = [](const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bucketize(self, boundaries, out_int32, right);
        };
        return wrap(dispatch_bucketize(_r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3)));
      } else {
        // aten::bucketize.Tensor_out(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_bucketize_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::bucketize_out(out, self, boundaries, out_int32, right);
        };
        return wrap(dispatch_bucketize_out(_r.tensor(4), _r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3)));
      }
    }
    case 1: {
      // aten::bucketize.Scalar(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor
      
      auto dispatch_bucketize = [](const at::Scalar & self, const at::Tensor & boundaries, bool out_int32, bool right) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::bucketize(self, boundaries, out_int32, right);
      };
      return wrap(dispatch_bucketize(_r.scalar(0), _r.tensor(1), _r.toBool(2), _r.toBool(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// searchsorted
static PyObject * THPVariable_searchsorted(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "searchsorted(Tensor sorted_sequence, Tensor input, *, bool out_int32=False, bool right=False, c10::string_view? side=None, Tensor? sorter=None, Tensor out=None)",
    "searchsorted(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False, c10::string_view? side=None, Tensor? sorter=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(6)) {
        // aten::searchsorted.Tensor(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor
        
        auto dispatch_searchsorted = [](const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right, ::std::optional<c10::string_view> side, const ::std::optional<at::Tensor> & sorter) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::searchsorted(sorted_sequence, self, out_int32, right, side, sorter);
        };
        return wrap(dispatch_searchsorted(_r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3), _r.stringViewOptional(4), _r.optionalTensor(5)));
      } else {
        // aten::searchsorted.Tensor_out(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_searchsorted_out = [](at::Tensor out, const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right, ::std::optional<c10::string_view> side, const ::std::optional<at::Tensor> & sorter) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::searchsorted_out(out, sorted_sequence, self, out_int32, right, side, sorter);
        };
        return wrap(dispatch_searchsorted_out(_r.tensor(6), _r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3), _r.stringViewOptional(4), _r.optionalTensor(5)));
      }
    }
    case 1: {
      if (_r.isNone(6)) {
        // aten::searchsorted.Scalar(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor
        
        auto dispatch_searchsorted = [](const at::Tensor & sorted_sequence, const at::Scalar & self, bool out_int32, bool right, ::std::optional<c10::string_view> side, const ::std::optional<at::Tensor> & sorter) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::searchsorted(sorted_sequence, self, out_int32, right, side, sorter);
        };
        return wrap(dispatch_searchsorted(_r.tensor(0), _r.scalar(1), _r.toBool(2), _r.toBool(3), _r.stringViewOptional(4), _r.optionalTensor(5)));
      } else {
        // aten::searchsorted.Scalar_out(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_searchsorted_out = [](at::Tensor out, const at::Tensor & sorted_sequence, const at::Scalar & self, bool out_int32, bool right, ::std::optional<c10::string_view> side, const ::std::optional<at::Tensor> & sorter) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::searchsorted_out(out, sorted_sequence, self, out_int32, right, side, sorter);
        };
        return wrap(dispatch_searchsorted_out(_r.tensor(6), _r.tensor(0), _r.scalar(1), _r.toBool(2), _r.toBool(3), _r.stringViewOptional(4), _r.optionalTensor(5)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _convert_indices_from_coo_to_csr
static PyObject * THPVariable__convert_indices_from_coo_to_csr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_convert_indices_from_coo_to_csr(Tensor input, int64_t size, *, bool out_int32=False, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::_convert_indices_from_coo_to_csr(Tensor self, int size, *, bool out_int32=False) -> Tensor
    
    auto dispatch__convert_indices_from_coo_to_csr = [](const at::Tensor & self, int64_t size, bool out_int32) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_convert_indices_from_coo_to_csr(self, size, out_int32);
    };
    return wrap(dispatch__convert_indices_from_coo_to_csr(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
  } else {
    // aten::_convert_indices_from_coo_to_csr.out(Tensor self, int size, *, bool out_int32=False, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__convert_indices_from_coo_to_csr_out = [](at::Tensor out, const at::Tensor & self, int64_t size, bool out_int32) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_convert_indices_from_coo_to_csr_out(out, self, size, out_int32);
    };
    return wrap(dispatch__convert_indices_from_coo_to_csr_out(_r.tensor(3), _r.tensor(0), _r.toInt64(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// isfinite
static PyObject * THPVariable_isfinite(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "isfinite(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::isfinite(Tensor self) -> Tensor
  
  auto dispatch_isfinite = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return self.isfinite();
  };
  return wrap(dispatch_isfinite(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _remove_batch_dim
static PyObject * THPVariable__remove_batch_dim(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_remove_batch_dim(Tensor input, int64_t level, SymInt batch_size, int64_t out_dim)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_remove_batch_dim(Tensor self, int level, SymInt batch_size, int out_dim) -> Tensor
  
  auto dispatch__remove_batch_dim = [](const at::Tensor & self, int64_t level, c10::SymInt batch_size, int64_t out_dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_remove_batch_dim_symint(self, level, batch_size, out_dim);
  };
  return wrap(dispatch__remove_batch_dim(_r.tensor(0), _r.toInt64(1), _r.toSymInt(2), _r.toInt64(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _linalg_eigh
static PyObject * THPVariable__linalg_eigh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get__linalg_eigh_structseq();
  static PyTypeObject* NamedTuple1 = generated::get__linalg_eigh_out_structseq();
  static PythonArgParser parser({
    "_linalg_eigh(Tensor A, c10::string_view UPLO=\"L\", bool compute_v=True, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::_linalg_eigh(Tensor A, str UPLO="L", bool compute_v=True) -> (Tensor eigenvalues, Tensor eigenvectors)
    
    auto dispatch__linalg_eigh = [](const at::Tensor & A, c10::string_view UPLO, bool compute_v) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::_linalg_eigh(A, UPLO, compute_v);
    };
    return wrap(NamedTuple, dispatch__linalg_eigh(_r.tensor(0), _r.stringView(1), _r.toBool(2)));
  } else {
    // aten::_linalg_eigh.eigenvalues(Tensor A, str UPLO="L", bool compute_v=True, *, Tensor(a!) eigenvalues, Tensor(b!) eigenvectors) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
    auto out = _r.tensorlist_n<2>(3);
    auto dispatch__linalg_eigh_out = [](at::Tensor & eigenvalues, at::Tensor & eigenvectors, const at::Tensor & A, c10::string_view UPLO, bool compute_v) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::_linalg_eigh_out(eigenvalues, eigenvectors, A, UPLO, compute_v);
    };
    return wrap(NamedTuple1, dispatch__linalg_eigh_out(out[0], out[1], _r.tensor(0), _r.stringView(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// outer
static PyObject * THPVariable_outer(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "outer(Tensor input, Tensor vec2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::outer(Tensor self, Tensor vec2) -> Tensor
    
    auto dispatch_outer = [](const at::Tensor & self, const at::Tensor & vec2) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return self.outer(vec2);
    };
    return wrap(dispatch_outer(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::outer.out(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_outer_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & vec2) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::outer_out(out, self, vec2);
    };
    return wrap(dispatch_outer_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _test_serialization_subcmul
static PyObject * THPVariable__test_serialization_subcmul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_test_serialization_subcmul(Tensor input, Tensor other, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_test_serialization_subcmul(Tensor self, Tensor other, Scalar alpha=1) -> Tensor
  
  auto dispatch__test_serialization_subcmul = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_test_serialization_subcmul(self, other, alpha);
  };
  return wrap(dispatch__test_serialization_subcmul(_r.tensor(0), _r.tensor(1), _r.scalar(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _test_autograd_multiple_dispatch_view
static PyObject * THPVariable__test_autograd_multiple_dispatch_view(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_test_autograd_multiple_dispatch_view(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_test_autograd_multiple_dispatch_view(Tensor(a) self) -> Tensor(a)
  
  auto dispatch__test_autograd_multiple_dispatch_view = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_test_autograd_multiple_dispatch_view(self);
  };
  return wrap(dispatch__test_autograd_multiple_dispatch_view(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// view_as_real_copy
static PyObject * THPVariable_view_as_real_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "view_as_real_copy(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::view_as_real_copy(Tensor self) -> Tensor
    
    auto dispatch_view_as_real_copy = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::view_as_real_copy(self);
    };
    return wrap(dispatch_view_as_real_copy(_r.tensor(0)));
  } else {
    // aten::view_as_real_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_view_as_real_copy_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::view_as_real_copy_out(out, self);
    };
    return wrap(dispatch_view_as_real_copy_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// permute_copy
static PyObject * THPVariable_permute_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "permute_copy(Tensor input, IntArrayRef dims, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::permute_copy(Tensor self, int[] dims) -> Tensor
    
    auto dispatch_permute_copy = [](const at::Tensor & self, at::IntArrayRef dims) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::permute_copy(self, dims);
    };
    return wrap(dispatch_permute_copy(_r.tensor(0), _r.intlist(1)));
  } else {
    // aten::permute_copy.out(Tensor self, int[] dims, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_permute_copy_out = [](at::Tensor out, const at::Tensor & self, at::IntArrayRef dims) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::permute_copy_out(out, self, dims);
    };
    return wrap(dispatch_permute_copy_out(_r.tensor(2), _r.tensor(0), _r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// detach_copy
static PyObject * THPVariable_detach_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "detach_copy(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::detach_copy(Tensor self) -> Tensor
    
    auto dispatch_detach_copy = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::detach_copy(self);
    };
    return wrap(dispatch_detach_copy(_r.tensor(0)));
  } else {
    // aten::detach_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_detach_copy_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::detach_copy_out(out, self);
    };
    return wrap(dispatch_detach_copy_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// split_copy
static PyObject * THPVariable_split_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "split_copy(Tensor input, SymInt split_size, int64_t dim=0, *, TensorList out=None)",
  }, /*traceable=*/false);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(3)) {
    // aten::split_copy.Tensor(Tensor self, SymInt split_size, int dim=0) -> Tensor[]
    
    auto dispatch_split_copy = [](const at::Tensor & self, c10::SymInt split_size, int64_t dim) -> ::std::vector<at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::split_copy_symint(self, split_size, dim);
    };
    return wrap(dispatch_split_copy(_r.tensor(0), _r.toSymInt(1), _r.toInt64(2)));
  } else {
    // aten::split_copy.Tensor_out(Tensor self, SymInt split_size, int dim=0, *, Tensor(a!)[] out) -> ()
    
    auto dispatch_split_copy_out = [](at::TensorList out, const at::Tensor & self, c10::SymInt split_size, int64_t dim) -> void {
      pybind11::gil_scoped_release no_gil;
      at::split_copy_symint_out(out, self, split_size, dim);
    };
    dispatch_split_copy_out(_r.tensorlist(3), _r.tensor(0), _r.toSymInt(1), _r.toInt64(2));
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unsqueeze_copy
static PyObject * THPVariable_unsqueeze_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unsqueeze_copy(Tensor input, int64_t dim, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::unsqueeze_copy(Tensor self, int dim) -> Tensor
    
    auto dispatch_unsqueeze_copy = [](const at::Tensor & self, int64_t dim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::unsqueeze_copy(self, dim);
    };
    return wrap(dispatch_unsqueeze_copy(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::unsqueeze_copy.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_unsqueeze_copy_out = [](at::Tensor out, const at::Tensor & self, int64_t dim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::unsqueeze_copy_out(out, self, dim);
    };
    return wrap(dispatch_unsqueeze_copy_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _indices_copy
static PyObject * THPVariable__indices_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_indices_copy(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::_indices_copy(Tensor self) -> Tensor
    
    auto dispatch__indices_copy = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_indices_copy(self);
    };
    return wrap(dispatch__indices_copy(_r.tensor(0)));
  } else {
    // aten::_indices_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__indices_copy_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_indices_copy_out(out, self);
    };
    return wrap(dispatch__indices_copy_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _values_copy
static PyObject * THPVariable__values_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_values_copy(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::_values_copy(Tensor self) -> Tensor
    
    auto dispatch__values_copy = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_values_copy(self);
    };
    return wrap(dispatch__values_copy(_r.tensor(0)));
  } else {
    // aten::_values_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__values_copy_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_values_copy_out(out, self);
    };
    return wrap(dispatch__values_copy_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unbind_copy
static PyObject * THPVariable_unbind_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unbind_copy(Tensor input, int64_t dim=0, *, TensorList out=None)",
  }, /*traceable=*/false);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(2)) {
    // aten::unbind_copy.int(Tensor self, int dim=0) -> Tensor[]
    
    auto dispatch_unbind_copy = [](const at::Tensor & self, int64_t dim) -> ::std::vector<at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::unbind_copy(self, dim);
    };
    return wrap(dispatch_unbind_copy(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::unbind_copy.int_out(Tensor self, int dim=0, *, Tensor(a!)[] out) -> ()
    
    auto dispatch_unbind_copy_out = [](at::TensorList out, const at::Tensor & self, int64_t dim) -> void {
      pybind11::gil_scoped_release no_gil;
      at::unbind_copy_out(out, self, dim);
    };
    dispatch_unbind_copy_out(_r.tensorlist(2), _r.tensor(0), _r.toInt64(1));
    Py_RETURN_NONE;
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// alias_copy
static PyObject * THPVariable_alias_copy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "alias_copy(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  if (_r.isNone(1)) {
    // aten::alias_copy(Tensor self) -> Tensor
    
    auto dispatch_alias_copy = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::alias_copy(self);
    };
    return wrap(dispatch_alias_copy(_r.tensor(0)));
  } else {
    // aten::alias_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_alias_copy_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::alias_copy_out(out, self);
    };
    return wrap(dispatch_alias_copy_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _nested_tensor_softmax_with_shape
static PyObject * THPVariable__nested_tensor_softmax_with_shape(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_nested_tensor_softmax_with_shape(Tensor input, Tensor query)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_nested_tensor_softmax_with_shape(Tensor self, Tensor query) -> Tensor
  
  auto dispatch__nested_tensor_softmax_with_shape = [](const at::Tensor & self, const at::Tensor & query) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_nested_tensor_softmax_with_shape(self, query);
  };
  return wrap(dispatch__nested_tensor_softmax_with_shape(_r.tensor(0), _r.tensor(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _safe_softmax
static PyObject * THPVariable__safe_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_safe_softmax(Tensor input, int64_t dim, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_safe_softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
  
  auto dispatch__safe_softmax = [](const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_safe_softmax(self, dim, dtype);
  };
  return wrap(dispatch__safe_softmax(_r.tensor(0), _r.toInt64(1), _r.scalartypeOptional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _native_multi_head_attention
static PyObject * THPVariable__native_multi_head_attention(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_native_multi_head_attention(Tensor query, Tensor key, Tensor value, int64_t embed_dim, int64_t num_head, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, Tensor proj_bias, Tensor? mask=None, bool need_weights=True, bool average_attn_weights=True, int64_t? mask_type=None)",
  }, /*traceable=*/true);

  ParsedArgs<13> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_native_multi_head_attention(Tensor query, Tensor key, Tensor value, int embed_dim, int num_head, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, Tensor proj_bias, Tensor? mask=None, bool need_weights=True, bool average_attn_weights=True, int? mask_type=None) -> (Tensor, Tensor)
  
  auto dispatch__native_multi_head_attention = [](const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t embed_dim, int64_t num_head, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, const ::std::optional<at::Tensor> & mask, bool need_weights, bool average_attn_weights, ::std::optional<int64_t> mask_type) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_native_multi_head_attention(query, key, value, embed_dim, num_head, qkv_weight, qkv_bias, proj_weight, proj_bias, mask, need_weights, average_attn_weights, mask_type);
  };
  return wrap(dispatch__native_multi_head_attention(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toInt64(3), _r.toInt64(4), _r.tensor(5), _r.tensor(6), _r.tensor(7), _r.tensor(8), _r.optionalTensor(9), _r.toBool(10), _r.toBool(11), _r.toInt64Optional(12)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _scaled_dot_product_attention_math_for_mps
static PyObject * THPVariable__scaled_dot_product_attention_math_for_mps(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_scaled_dot_product_attention_math_for_mps(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, double dropout_p=0.0, bool is_causal=False, Tensor? dropout_mask=None, *, double? scale=None)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_scaled_dot_product_attention_math_for_mps(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, Tensor? dropout_mask=None, *, float? scale=None) -> (Tensor, Tensor)
  
  auto dispatch__scaled_dot_product_attention_math_for_mps = [](const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & attn_mask, double dropout_p, bool is_causal, const ::std::optional<at::Tensor> & dropout_mask, ::std::optional<double> scale) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_scaled_dot_product_attention_math_for_mps(query, key, value, attn_mask, dropout_p, is_causal, dropout_mask, scale);
  };
  return wrap(dispatch__scaled_dot_product_attention_math_for_mps(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.optionalTensor(3), _r.toDouble(4), _r.toBool(5), _r.optionalTensor(6), _r.toDoubleOptional(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _scaled_dot_product_flash_attention
static PyObject * THPVariable__scaled_dot_product_flash_attention(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get__scaled_dot_product_flash_attention_structseq();
  static PythonArgParser parser({
    "_scaled_dot_product_flash_attention(Tensor query, Tensor key, Tensor value, double dropout_p=0.0, bool is_causal=False, bool return_debug_mask=False, *, double? scale=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_scaled_dot_product_flash_attention(Tensor query, Tensor key, Tensor value, float dropout_p=0.0, bool is_causal=False, bool return_debug_mask=False, *, float? scale=None) -> (Tensor output, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, Tensor rng_state, Tensor unused, Tensor debug_attn_mask)
  
  auto dispatch__scaled_dot_product_flash_attention = [](const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, double dropout_p, bool is_causal, bool return_debug_mask, ::std::optional<double> scale) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,c10::SymInt,c10::SymInt,at::Tensor,at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_scaled_dot_product_flash_attention(query, key, value, dropout_p, is_causal, return_debug_mask, scale);
  };
  return wrap(NamedTuple, dispatch__scaled_dot_product_flash_attention(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toDouble(3), _r.toBool(4), _r.toBool(5), _r.toDoubleOptional(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _scaled_dot_product_flash_attention_for_cpu
static PyObject * THPVariable__scaled_dot_product_flash_attention_for_cpu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get__scaled_dot_product_flash_attention_for_cpu_structseq();
  static PythonArgParser parser({
    "_scaled_dot_product_flash_attention_for_cpu(Tensor query, Tensor key, Tensor value, double dropout_p=0.0, bool is_causal=False, *, Tensor? attn_mask=None, double? scale=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_scaled_dot_product_flash_attention_for_cpu(Tensor query, Tensor key, Tensor value, float dropout_p=0.0, bool is_causal=False, *, Tensor? attn_mask=None, float? scale=None) -> (Tensor output, Tensor logsumexp)
  
  auto dispatch__scaled_dot_product_flash_attention_for_cpu = [](const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, double dropout_p, bool is_causal, const ::std::optional<at::Tensor> & attn_mask, ::std::optional<double> scale) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_scaled_dot_product_flash_attention_for_cpu(query, key, value, dropout_p, is_causal, attn_mask, scale);
  };
  return wrap(NamedTuple, dispatch__scaled_dot_product_flash_attention_for_cpu(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toDouble(3), _r.toBool(4), _r.optionalTensor(5), _r.toDoubleOptional(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _triton_scaled_dot_attention
static PyObject * THPVariable__triton_scaled_dot_attention(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_triton_scaled_dot_attention(Tensor q, Tensor k, Tensor v, double dropout_p=0.0)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // aten::_triton_scaled_dot_attention(Tensor q, Tensor k, Tensor v, float dropout_p=0.0) -> Tensor
  
  auto dispatch__triton_scaled_dot_attention = [](const at::Tensor & q, const at::Tensor & k, const at::Tensor & v, double dropout_p) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_triton_scaled_dot_attention(q, k, v, dropout_p);
  };
  return wrap(dispatch__triton_scaled_dot_attention(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toDouble(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _fused_adamw_
static PyObject * THPVariable__fused_adamw_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_fused_adamw_(TensorList self, TensorList grads, TensorList exp_avgs, TensorList exp_avg_sqs, TensorList max_exp_avg_sqs, TensorList state_steps, *, Tensor lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None)",
    "_fused_adamw_(TensorList self, TensorList grads, TensorList exp_avgs, TensorList exp_avg_sqs, TensorList max_exp_avg_sqs, TensorList state_steps, *, double lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None)",
  }, /*traceable=*/false);

  ParsedArgs<15> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_fused_adamw_.tensor_lr(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, Tensor lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
      
      auto dispatch__fused_adamw_ = [](at::TensorList self, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps, const at::Tensor & lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_fused_adamw_(self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
      };
      dispatch__fused_adamw_(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.tensorlist(3), _r.tensorlist(4), _r.tensorlist(5), _r.tensor(6), _r.toDouble(7), _r.toDouble(8), _r.toDouble(9), _r.toDouble(10), _r.toBool(11), _r.toBool(12), _r.optionalTensor(13), _r.optionalTensor(14));
      Py_RETURN_NONE;
    }
    case 1: {
      // aten::_fused_adamw_(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
      
      auto dispatch__fused_adamw_ = [](at::TensorList self, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps, double lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) -> void {
        pybind11::gil_scoped_release no_gil;
        at::_fused_adamw_(self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
      };
      dispatch__fused_adamw_(_r.tensorlist(0), _r.tensorlist(1), _r.tensorlist(2), _r.tensorlist(3), _r.tensorlist(4), _r.tensorlist(5), _r.toDouble(6), _r.toDouble(7), _r.toDouble(8), _r.toDouble(9), _r.toDouble(10), _r.toBool(11), _r.toBool(12), _r.optionalTensor(13), _r.optionalTensor(14));
      Py_RETURN_NONE;
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

} // namespace torch::autograd
