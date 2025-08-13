#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include "torch/csrc/jit/frontend/tracer.h"

#include <torch/library.h>

#include "torch/csrc/autograd/function.h"

#include "ATen/quantized/Quantizer.h"

// @generated from ../tools/autograd/templates/TraceType.cpp

// See the `Tracer` section in `torch/csrc/jit/OVERVIEW.md`.
// NOTE See [Sharded File] comment in VariableType

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>
#else
#include <ATen/ops/_cast_Double_ops.h>
#include <ATen/ops/_cast_Float_ops.h>
#include <ATen/ops/_cast_Half_ops.h>
#include <ATen/ops/align_as_ops.h>
#include <ATen/ops/_assert_tensor_metadata_ops.h>
#include <ATen/ops/sym_constrain_range_for_size_ops.h>
#include <ATen/ops/refine_names_ops.h>
#include <ATen/ops/_use_cudnn_rnn_flatten_weight_ops.h>
#include <ATen/ops/_cudnn_rnn_flatten_weight_ops.h>
#include <ATen/ops/_sobol_engine_ff_ops.h>
#include <ATen/ops/_sobol_engine_scramble_ops.h>
#include <ATen/ops/feature_alpha_dropout_ops.h>
#include <ATen/ops/feature_alpha_dropout_ops.h>
#include <ATen/ops/abs_ops.h>
#include <ATen/ops/abs_ops.h>
#include <ATen/ops/abs_ops.h>
#include <ATen/ops/imag_ops.h>
#include <ATen/ops/resolve_conj_ops.h>
#include <ATen/ops/resolve_neg_ops.h>
#include <ATen/ops/adaptive_max_pool1d_ops.h>
#include <ATen/ops/addmv_ops.h>
#include <ATen/ops/addmv_ops.h>
#include <ATen/ops/addmv_ops.h>
#include <ATen/ops/addr_ops.h>
#include <ATen/ops/addr_ops.h>
#include <ATen/ops/addr_ops.h>
#include <ATen/ops/affine_grid_generator_backward_ops.h>
#include <ATen/ops/argmin_ops.h>
#include <ATen/ops/argmin_ops.h>
#include <ATen/ops/atan_ops.h>
#include <ATen/ops/atan_ops.h>
#include <ATen/ops/atan_ops.h>
#include <ATen/ops/arctan_ops.h>
#include <ATen/ops/arctan_ops.h>
#include <ATen/ops/arctan_ops.h>
#include <ATen/ops/quantized_batch_norm_ops.h>
#include <ATen/ops/binary_cross_entropy_backward_ops.h>
#include <ATen/ops/binary_cross_entropy_backward_ops.h>
#include <ATen/ops/bitwise_not_ops.h>
#include <ATen/ops/bitwise_not_ops.h>
#include <ATen/ops/bitwise_not_ops.h>
#include <ATen/ops/logical_not_ops.h>
#include <ATen/ops/logical_not_ops.h>
#include <ATen/ops/logical_not_ops.h>
#include <ATen/ops/concatenate_ops.h>
#include <ATen/ops/concatenate_ops.h>
#include <ATen/ops/concatenate_ops.h>
#include <ATen/ops/concatenate_ops.h>
#include <ATen/ops/ceil_ops.h>
#include <ATen/ops/ceil_ops.h>
#include <ATen/ops/ceil_ops.h>
#include <ATen/ops/conv_tbc_ops.h>
#include <ATen/ops/cosh_ops.h>
#include <ATen/ops/cosh_ops.h>
#include <ATen/ops/cosh_ops.h>
#include <ATen/ops/cosine_embedding_loss_ops.h>
#include <ATen/ops/cudnn_affine_grid_generator_backward_ops.h>
#include <ATen/ops/cudnn_grid_sampler_ops.h>
#include <ATen/ops/cummin_ops.h>
#include <ATen/ops/cummin_ops.h>
#include <ATen/ops/cummin_ops.h>
#include <ATen/ops/cummin_ops.h>
#include <ATen/ops/_cummin_helper_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/_embedding_bag_forward_only_ops.h>
#include <ATen/ops/embedding_bag_ops.h>
#include <ATen/ops/embedding_bag_ops.h>
#include <ATen/ops/new_zeros_ops.h>
#include <ATen/ops/erf_ops.h>
#include <ATen/ops/erf_ops.h>
#include <ATen/ops/erf_ops.h>
#include <ATen/ops/grid_sampler_ops.h>
#include <ATen/ops/_grid_sampler_2d_cpu_fallback_ops.h>
#include <ATen/ops/grid_sampler_3d_ops.h>
#include <ATen/ops/hann_window_ops.h>
#include <ATen/ops/hann_window_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/native_group_norm_backward_ops.h>
#include <ATen/ops/_fft_c2c_ops.h>
#include <ATen/ops/_fft_c2c_ops.h>
#include <ATen/ops/_validate_compressed_sparse_indices_ops.h>
#include <ATen/ops/_cufft_get_plan_cache_size_ops.h>
#include <ATen/ops/_cufft_get_plan_cache_max_size_ops.h>
#include <ATen/ops/index_ops.h>
#include <ATen/ops/index_ops.h>
#include <ATen/ops/isnan_ops.h>
#include <ATen/ops/kthvalue_ops.h>
#include <ATen/ops/kthvalue_ops.h>
#include <ATen/ops/kthvalue_ops.h>
#include <ATen/ops/kthvalue_ops.h>
#include <ATen/ops/native_layer_norm_ops.h>
#include <ATen/ops/nan_to_num_ops.h>
#include <ATen/ops/nan_to_num_ops.h>
#include <ATen/ops/nan_to_num_ops.h>
#include <ATen/ops/_sparse_semi_structured_apply_ops.h>
#include <ATen/ops/_sparse_semi_structured_mm_ops.h>
#include <ATen/ops/fbgemm_linear_int8_weight_fp32_activation_ops.h>
#include <ATen/ops/fbgemm_linear_int8_weight_ops.h>
#include <ATen/ops/fbgemm_linear_fp16_weight_fp32_activation_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/_log_softmax_backward_data_ops.h>
#include <ATen/ops/_log_softmax_backward_data_ops.h>
#include <ATen/ops/logcumsumexp_ops.h>
#include <ATen/ops/logcumsumexp_ops.h>
#include <ATen/ops/logcumsumexp_ops.h>
#include <ATen/ops/logcumsumexp_ops.h>
#include <ATen/ops/matrix_exp_backward_ops.h>
#include <ATen/ops/amax_ops.h>
#include <ATen/ops/amax_ops.h>
#include <ATen/ops/mkldnn_max_pool2d_ops.h>
#include <ATen/ops/quantized_max_pool2d_ops.h>
#include <ATen/ops/amin_ops.h>
#include <ATen/ops/amin_ops.h>
#include <ATen/ops/_mps_convolution_ops.h>
#include <ATen/ops/mkldnn_rnn_layer_backward_ops.h>
#include <ATen/ops/miopen_depthwise_convolution_ops.h>
#include <ATen/ops/_int_mm_ops.h>
#include <ATen/ops/_int_mm_ops.h>
#include <ATen/ops/_dyn_quant_pack_4bit_weight_ops.h>
#include <ATen/ops/_weight_int8pack_mm_ops.h>
#include <ATen/ops/native_batch_norm_ops.h>
#include <ATen/ops/native_batch_norm_ops.h>
#include <ATen/ops/batch_norm_stats_ops.h>
#include <ATen/ops/batch_norm_gather_stats_ops.h>
#include <ATen/ops/native_batch_norm_backward_ops.h>
#include <ATen/ops/batch_norm_backward_reduce_ops.h>
#include <ATen/ops/is_vulkan_available_ops.h>
#include <ATen/ops/_nnpack_spatial_convolution_ops.h>
#include <ATen/ops/ones_ops.h>
#include <ATen/ops/ones_ops.h>
#include <ATen/ops/ones_ops.h>
#include <ATen/ops/_cdist_forward_ops.h>
#include <ATen/ops/cosine_similarity_ops.h>
#include <ATen/ops/movedim_ops.h>
#include <ATen/ops/movedim_ops.h>
#include <ATen/ops/numpy_T_ops.h>
#include <ATen/ops/mH_ops.h>
#include <ATen/ops/rand_like_ops.h>
#include <ATen/ops/randint_like_ops.h>
#include <ATen/ops/randint_like_ops.h>
#include <ATen/ops/round_ops.h>
#include <ATen/ops/round_ops.h>
#include <ATen/ops/round_ops.h>
#include <ATen/ops/round_ops.h>
#include <ATen/ops/round_ops.h>
#include <ATen/ops/round_ops.h>
#include <ATen/ops/gelu_ops.h>
#include <ATen/ops/gelu_ops.h>
#include <ATen/ops/gelu_ops.h>
#include <ATen/ops/hardshrink_ops.h>
#include <ATen/ops/hardshrink_ops.h>
#include <ATen/ops/select_backward_ops.h>
#include <ATen/ops/mish_ops.h>
#include <ATen/ops/mish_ops.h>
#include <ATen/ops/mish_ops.h>
#include <ATen/ops/sigmoid_ops.h>
#include <ATen/ops/sigmoid_ops.h>
#include <ATen/ops/sigmoid_ops.h>
#include <ATen/ops/size_ops.h>
#include <ATen/ops/size_ops.h>
#include <ATen/ops/sym_numel_ops.h>
#include <ATen/ops/slice_scatter_ops.h>
#include <ATen/ops/_softmax_backward_data_ops.h>
#include <ATen/ops/_softmax_backward_data_ops.h>
#include <ATen/ops/split_with_sizes_ops.h>
#include <ATen/ops/hsplit_ops.h>
#include <ATen/ops/hsplit_ops.h>
#include <ATen/ops/stack_ops.h>
#include <ATen/ops/stack_ops.h>
#include <ATen/ops/_stack_ops.h>
#include <ATen/ops/_stack_ops.h>
#include <ATen/ops/square_ops.h>
#include <ATen/ops/square_ops.h>
#include <ATen/ops/square_ops.h>
#include <ATen/ops/tanh_ops.h>
#include <ATen/ops/tanh_ops.h>
#include <ATen/ops/tanh_ops.h>
#include <ATen/ops/tensordot_ops.h>
#include <ATen/ops/tensordot_ops.h>
#include <ATen/ops/tile_ops.h>
#include <ATen/ops/_mkldnn_transpose_ops.h>
#include <ATen/ops/_mkldnn_transpose_ops.h>
#include <ATen/ops/fliplr_ops.h>
#include <ATen/ops/_nested_from_padded_and_nested_example_ops.h>
#include <ATen/ops/_nested_get_max_seqlen_ops.h>
#include <ATen/ops/_nested_compute_contiguous_strides_offsets_ops.h>
#include <ATen/ops/fix_ops.h>
#include <ATen/ops/fix_ops.h>
#include <ATen/ops/fix_ops.h>
#include <ATen/ops/unique_dim_ops.h>
#include <ATen/ops/unique_consecutive_ops.h>
#include <ATen/ops/vander_ops.h>
#include <ATen/ops/view_as_ops.h>
#include <ATen/ops/_dirichlet_grad_ops.h>
#include <ATen/ops/batch_norm_backward_ops.h>
#include <ATen/ops/frobenius_norm_ops.h>
#include <ATen/ops/frobenius_norm_ops.h>
#include <ATen/ops/clone_ops.h>
#include <ATen/ops/positive_ops.h>
#include <ATen/ops/resize_as_sparse_ops.h>
#include <ATen/ops/sparse_sampled_addmm_ops.h>
#include <ATen/ops/sparse_sampled_addmm_ops.h>
#include <ATen/ops/sparse_csr_tensor_ops.h>
#include <ATen/ops/sparse_csr_tensor_ops.h>
#include <ATen/ops/_sparse_bsc_tensor_unsafe_ops.h>
#include <ATen/ops/dense_dim_ops.h>
#include <ATen/ops/_dimV_ops.h>
#include <ATen/ops/coalesce_ops.h>
#include <ATen/ops/_indices_ops.h>
#include <ATen/ops/to_sparse_csc_ops.h>
#include <ATen/ops/_to_sparse_csc_ops.h>
#include <ATen/ops/_to_sparse_bsc_ops.h>
#include <ATen/ops/mkldnn_reorder_conv2d_weight_ops.h>
#include <ATen/ops/quantize_per_channel_ops.h>
#include <ATen/ops/dequantize_ops.h>
#include <ATen/ops/dequantize_ops.h>
#include <ATen/ops/q_per_channel_zero_points_ops.h>
#include <ATen/ops/fake_quantize_per_tensor_affine_ops.h>
#include <ATen/ops/fake_quantize_per_tensor_affine_ops.h>
#include <ATen/ops/_fake_quantize_learnable_per_channel_affine_ops.h>
#include <ATen/ops/_autocast_to_full_precision_ops.h>
#include <ATen/ops/to_ops.h>
#include <ATen/ops/to_ops.h>
#include <ATen/ops/to_ops.h>
#include <ATen/ops/to_ops.h>
#include <ATen/ops/combinations_ops.h>
#include <ATen/ops/item_ops.h>
#include <ATen/ops/_lstm_mps_ops.h>
#include <ATen/ops/_thnn_fused_lstm_cell_ops.h>
#include <ATen/ops/lstm_ops.h>
#include <ATen/ops/lstm_ops.h>
#include <ATen/ops/gru_ops.h>
#include <ATen/ops/gru_ops.h>
#include <ATen/ops/rnn_tanh_ops.h>
#include <ATen/ops/rnn_tanh_ops.h>
#include <ATen/ops/rnn_relu_cell_ops.h>
#include <ATen/ops/_pad_packed_sequence_ops.h>
#include <ATen/ops/lift_fresh_copy_ops.h>
#include <ATen/ops/index_reduce_ops.h>
#include <ATen/ops/index_reduce_ops.h>
#include <ATen/ops/index_reduce_ops.h>
#include <ATen/ops/index_fill_ops.h>
#include <ATen/ops/index_fill_ops.h>
#include <ATen/ops/index_fill_ops.h>
#include <ATen/ops/index_fill_ops.h>
#include <ATen/ops/index_fill_ops.h>
#include <ATen/ops/index_fill_ops.h>
#include <ATen/ops/index_fill_ops.h>
#include <ATen/ops/index_fill_ops.h>
#include <ATen/ops/scatter_add_ops.h>
#include <ATen/ops/scatter_add_ops.h>
#include <ATen/ops/scatter_add_ops.h>
#include <ATen/ops/scatter_add_ops.h>
#include <ATen/ops/digamma_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/cauchy_ops.h>
#include <ATen/ops/log_normal_ops.h>
#include <ATen/ops/cross_ops.h>
#include <ATen/ops/cross_ops.h>
#include <ATen/ops/ne_ops.h>
#include <ATen/ops/ne_ops.h>
#include <ATen/ops/ne_ops.h>
#include <ATen/ops/ne_ops.h>
#include <ATen/ops/ne_ops.h>
#include <ATen/ops/ne_ops.h>
#include <ATen/ops/ge_ops.h>
#include <ATen/ops/ge_ops.h>
#include <ATen/ops/ge_ops.h>
#include <ATen/ops/ge_ops.h>
#include <ATen/ops/ge_ops.h>
#include <ATen/ops/ge_ops.h>
#include <ATen/ops/_gather_sparse_backward_ops.h>
#include <ATen/ops/linalg_vander_ops.h>
#include <ATen/ops/swapaxes_ops.h>
#include <ATen/ops/swapaxes_ops.h>
#include <ATen/ops/cholesky_solve_ops.h>
#include <ATen/ops/cholesky_solve_ops.h>
#include <ATen/ops/qr_ops.h>
#include <ATen/ops/qr_ops.h>
#include <ATen/ops/digamma_ops.h>
#include <ATen/ops/digamma_ops.h>
#include <ATen/ops/polygamma_ops.h>
#include <ATen/ops/polygamma_ops.h>
#include <ATen/ops/polygamma_ops.h>
#include <ATen/ops/histc_ops.h>
#include <ATen/ops/histc_ops.h>
#include <ATen/ops/_histogramdd_bin_edges_ops.h>
#include <ATen/ops/_histogramdd_from_bin_tensors_ops.h>
#include <ATen/ops/nextafter_ops.h>
#include <ATen/ops/nextafter_ops.h>
#include <ATen/ops/nextafter_ops.h>
#include <ATen/ops/maximum_ops.h>
#include <ATen/ops/maximum_ops.h>
#include <ATen/ops/minimum_ops.h>
#include <ATen/ops/minimum_ops.h>
#include <ATen/ops/quantile_ops.h>
#include <ATen/ops/quantile_ops.h>
#include <ATen/ops/quantile_ops.h>
#include <ATen/ops/quantile_ops.h>
#include <ATen/ops/msort_ops.h>
#include <ATen/ops/msort_ops.h>
#include <ATen/ops/argsort_ops.h>
#include <ATen/ops/argsort_ops.h>
#include <ATen/ops/argsort_ops.h>
#include <ATen/ops/argsort_ops.h>
#include <ATen/ops/topk_ops.h>
#include <ATen/ops/topk_ops.h>
#include <ATen/ops/unfold_backward_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/alias_ops.h>
#include <ATen/ops/_foreach_sub_ops.h>
#include <ATen/ops/_foreach_sub_ops.h>
#include <ATen/ops/_foreach_sub_ops.h>
#include <ATen/ops/_foreach_sub_ops.h>
#include <ATen/ops/_foreach_sub_ops.h>
#include <ATen/ops/_foreach_sub_ops.h>
#include <ATen/ops/_foreach_maximum_ops.h>
#include <ATen/ops/_foreach_maximum_ops.h>
#include <ATen/ops/_foreach_maximum_ops.h>
#include <ATen/ops/_foreach_maximum_ops.h>
#include <ATen/ops/_foreach_maximum_ops.h>
#include <ATen/ops/_foreach_maximum_ops.h>
#include <ATen/ops/_foreach_acos_ops.h>
#include <ATen/ops/_foreach_acos_ops.h>
#include <ATen/ops/_foreach_atan_ops.h>
#include <ATen/ops/_foreach_atan_ops.h>
#include <ATen/ops/_foreach_ceil_ops.h>
#include <ATen/ops/_foreach_ceil_ops.h>
#include <ATen/ops/_foreach_erf_ops.h>
#include <ATen/ops/_foreach_erf_ops.h>
#include <ATen/ops/_foreach_log2_ops.h>
#include <ATen/ops/_foreach_log2_ops.h>
#include <ATen/ops/bucketize_ops.h>
#include <ATen/ops/bucketize_ops.h>
#include <ATen/ops/bucketize_ops.h>
#include <ATen/ops/mse_loss_ops.h>
#include <ATen/ops/mse_loss_ops.h>
#include <ATen/ops/l1_loss_ops.h>
#include <ATen/ops/nll_loss_nd_ops.h>
#include <ATen/ops/nll_loss2d_ops.h>
#include <ATen/ops/nll_loss2d_ops.h>
#include <ATen/ops/nll_loss2d_forward_ops.h>
#include <ATen/ops/nll_loss2d_forward_ops.h>
#include <ATen/ops/nll_loss2d_backward_ops.h>
#include <ATen/ops/nll_loss2d_backward_ops.h>
#include <ATen/ops/soft_margin_loss_ops.h>
#include <ATen/ops/soft_margin_loss_ops.h>
#include <ATen/ops/glu_ops.h>
#include <ATen/ops/glu_ops.h>
#include <ATen/ops/glu_backward_jvp_ops.h>
#include <ATen/ops/hardtanh_ops.h>
#include <ATen/ops/hardtanh_ops.h>
#include <ATen/ops/hardtanh_ops.h>
#include <ATen/ops/hardswish_backward_ops.h>
#include <ATen/ops/leaky_relu_ops.h>
#include <ATen/ops/leaky_relu_ops.h>
#include <ATen/ops/leaky_relu_ops.h>
#include <ATen/ops/log_sigmoid_forward_ops.h>
#include <ATen/ops/log_sigmoid_forward_ops.h>
#include <ATen/ops/log_sigmoid_backward_ops.h>
#include <ATen/ops/log_sigmoid_backward_ops.h>
#include <ATen/ops/softshrink_ops.h>
#include <ATen/ops/softshrink_ops.h>
#include <ATen/ops/adaptive_avg_pool3d_backward_ops.h>
#include <ATen/ops/_adaptive_avg_pool3d_backward_ops.h>
#include <ATen/ops/adaptive_max_pool2d_backward_ops.h>
#include <ATen/ops/adaptive_max_pool2d_backward_ops.h>
#include <ATen/ops/adaptive_max_pool3d_backward_ops.h>
#include <ATen/ops/adaptive_max_pool3d_backward_ops.h>
#include <ATen/ops/fractional_max_pool3d_ops.h>
#include <ATen/ops/fractional_max_pool3d_ops.h>
#include <ATen/ops/reflection_pad3d_ops.h>
#include <ATen/ops/reflection_pad3d_ops.h>
#include <ATen/ops/replication_pad1d_ops.h>
#include <ATen/ops/replication_pad1d_ops.h>
#include <ATen/ops/replication_pad2d_ops.h>
#include <ATen/ops/replication_pad2d_ops.h>
#include <ATen/ops/_pad_circular_ops.h>
#include <ATen/ops/pad_ops.h>
#include <ATen/ops/upsample_nearest1d_ops.h>
#include <ATen/ops/_upsample_nearest_exact1d_ops.h>
#include <ATen/ops/upsample_nearest1d_ops.h>
#include <ATen/ops/_upsample_nearest_exact1d_ops.h>
#include <ATen/ops/upsample_nearest1d_ops.h>
#include <ATen/ops/_upsample_nearest_exact1d_ops.h>
#include <ATen/ops/_conv_depthwise2d_ops.h>
#include <ATen/ops/_conv_depthwise2d_ops.h>
#include <ATen/ops/slow_conv3d_ops.h>
#include <ATen/ops/slow_conv3d_ops.h>
#include <ATen/ops/_remove_batch_dim_ops.h>
#include <ATen/ops/special_log_ndtr_ops.h>
#include <ATen/ops/special_log_ndtr_ops.h>
#include <ATen/ops/special_erf_ops.h>
#include <ATen/ops/special_erf_ops.h>
#include <ATen/ops/special_xlogy_ops.h>
#include <ATen/ops/special_xlogy_ops.h>
#include <ATen/ops/special_xlogy_ops.h>
#include <ATen/ops/special_xlogy_ops.h>
#include <ATen/ops/special_xlogy_ops.h>
#include <ATen/ops/special_xlogy_ops.h>
#include <ATen/ops/special_expit_ops.h>
#include <ATen/ops/special_expit_ops.h>
#include <ATen/ops/special_sinc_ops.h>
#include <ATen/ops/special_sinc_ops.h>
#include <ATen/ops/special_softmax_ops.h>
#include <ATen/ops/fft_fft_ops.h>
#include <ATen/ops/fft_fft_ops.h>
#include <ATen/ops/fft_rfft_ops.h>
#include <ATen/ops/fft_rfft_ops.h>
#include <ATen/ops/fft_hfft2_ops.h>
#include <ATen/ops/fft_hfft2_ops.h>
#include <ATen/ops/fft_ifftn_ops.h>
#include <ATen/ops/fft_ifftn_ops.h>
#include <ATen/ops/fft_ihfftn_ops.h>
#include <ATen/ops/fft_ihfftn_ops.h>
#include <ATen/ops/fft_fftfreq_ops.h>
#include <ATen/ops/fft_fftfreq_ops.h>
#include <ATen/ops/fft_rfftfreq_ops.h>
#include <ATen/ops/fft_rfftfreq_ops.h>
#include <ATen/ops/linalg_cholesky_ex_ops.h>
#include <ATen/ops/linalg_cholesky_ex_ops.h>
#include <ATen/ops/linalg_cross_ops.h>
#include <ATen/ops/linalg_cross_ops.h>
#include <ATen/ops/linalg_lu_factor_ex_ops.h>
#include <ATen/ops/linalg_lu_factor_ex_ops.h>
#include <ATen/ops/det_ops.h>
#include <ATen/ops/inverse_ops.h>
#include <ATen/ops/inverse_ops.h>
#include <ATen/ops/linalg_cond_ops.h>
#include <ATen/ops/linalg_cond_ops.h>
#include <ATen/ops/linalg_cond_ops.h>
#include <ATen/ops/linalg_cond_ops.h>
#include <ATen/ops/linalg_pinv_ops.h>
#include <ATen/ops/linalg_pinv_ops.h>
#include <ATen/ops/linalg_pinv_ops.h>
#include <ATen/ops/linalg_pinv_ops.h>
#include <ATen/ops/linalg_pinv_ops.h>
#include <ATen/ops/linalg_pinv_ops.h>
#include <ATen/ops/linalg_pinv_ops.h>
#include <ATen/ops/linalg_pinv_ops.h>
#include <ATen/ops/linalg_solve_ex_ops.h>
#include <ATen/ops/linalg_solve_ex_ops.h>
#include <ATen/ops/_spsolve_ops.h>
#include <ATen/ops/linalg_tensorsolve_ops.h>
#include <ATen/ops/linalg_tensorsolve_ops.h>
#include <ATen/ops/linalg_multi_dot_ops.h>
#include <ATen/ops/linalg_multi_dot_ops.h>
#include <ATen/ops/_test_parallel_materialize_ops.h>
#include <ATen/ops/_test_string_default_ops.h>
#include <ATen/ops/flatten_dense_tensors_ops.h>
#include <ATen/ops/_conj_copy_ops.h>
#include <ATen/ops/detach_copy_ops.h>
#include <ATen/ops/row_indices_copy_ops.h>
#include <ATen/ops/_transformer_encoder_layer_fwd_ops.h>
#include <ATen/ops/_native_multi_head_attention_ops.h>
#include <ATen/ops/_fused_sdp_choice_ops.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_ops.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_for_cpu_ops.h>
#include <ATen/ops/_scaled_dot_product_efficient_attention_backward_ops.h>
#include <ATen/ops/_flash_attention_backward_ops.h>
#include <ATen/ops/_efficient_attention_backward_ops.h>
#include <ATen/ops/_fill_mem_eff_dropout_mask_ops.h>
#include <ATen/ops/_triton_multi_head_attention_ops.h>
#include <ATen/ops/special_airy_ai_ops.h>
#include <ATen/ops/special_airy_ai_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_w_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_w_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_w_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_w_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_w_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_w_ops.h>
#include <ATen/ops/special_hermite_polynomial_h_ops.h>
#include <ATen/ops/special_hermite_polynomial_h_ops.h>
#include <ATen/ops/special_hermite_polynomial_h_ops.h>
#include <ATen/ops/special_hermite_polynomial_h_ops.h>
#include <ATen/ops/special_hermite_polynomial_h_ops.h>
#include <ATen/ops/special_hermite_polynomial_h_ops.h>
#include <ATen/ops/special_modified_bessel_i0_ops.h>
#include <ATen/ops/special_modified_bessel_i0_ops.h>
#include <ATen/ops/special_modified_bessel_k0_ops.h>
#include <ATen/ops/special_modified_bessel_k0_ops.h>
#include <ATen/ops/special_modified_bessel_k1_ops.h>
#include <ATen/ops/special_modified_bessel_k1_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/_fused_adamw_ops.h>
#include <ATen/ops/_fused_adamw_ops.h>
#include <ATen/ops/_cudnn_rnn_flatten_weight_ops.h>
#include <ATen/ops/quantized_batch_norm_ops.h>
#include <ATen/ops/conv_tbc_ops.h>
#include <ATen/ops/cudnn_affine_grid_generator_backward_ops.h>
#include <ATen/ops/cudnn_grid_sampler_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/_embedding_bag_forward_only_ops.h>
#include <ATen/ops/new_zeros_ops.h>
#include <ATen/ops/_grid_sampler_2d_cpu_fallback_ops.h>
#include <ATen/ops/grid_sampler_3d_ops.h>
#include <ATen/ops/hann_window_ops.h>
#include <ATen/ops/hann_window_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/native_group_norm_backward_ops.h>
#include <ATen/ops/isnan_ops.h>
#include <ATen/ops/native_layer_norm_ops.h>
#include <ATen/ops/mkldnn_max_pool2d_ops.h>
#include <ATen/ops/quantized_max_pool2d_ops.h>
#include <ATen/ops/_mps_convolution_ops.h>
#include <ATen/ops/mkldnn_rnn_layer_backward_ops.h>
#include <ATen/ops/miopen_depthwise_convolution_ops.h>
#include <ATen/ops/batch_norm_stats_ops.h>
#include <ATen/ops/batch_norm_gather_stats_ops.h>
#include <ATen/ops/native_batch_norm_backward_ops.h>
#include <ATen/ops/batch_norm_backward_reduce_ops.h>
#include <ATen/ops/_nnpack_spatial_convolution_ops.h>
#include <ATen/ops/ones_ops.h>
#include <ATen/ops/_cdist_forward_ops.h>
#include <ATen/ops/rand_like_ops.h>
#include <ATen/ops/randint_like_ops.h>
#include <ATen/ops/randint_like_ops.h>
#include <ATen/ops/select_backward_ops.h>
#include <ATen/ops/slice_scatter_ops.h>
#include <ATen/ops/_mkldnn_transpose_ops.h>
#include <ATen/ops/_nested_from_padded_and_nested_example_ops.h>
#include <ATen/ops/unique_dim_ops.h>
#include <ATen/ops/unique_consecutive_ops.h>
#include <ATen/ops/_dirichlet_grad_ops.h>
#include <ATen/ops/clone_ops.h>
#include <ATen/ops/resize_as_ops.h>
#include <ATen/ops/resize_as_ops.h>
#include <ATen/ops/resize_as_sparse_ops.h>
#include <ATen/ops/resize_as_sparse_ops.h>
#include <ATen/ops/_to_sparse_csc_ops.h>
#include <ATen/ops/_to_sparse_bsc_ops.h>
#include <ATen/ops/mkldnn_reorder_conv2d_weight_ops.h>
#include <ATen/ops/quantize_per_channel_ops.h>
#include <ATen/ops/dequantize_ops.h>
#include <ATen/ops/dequantize_ops.h>
#include <ATen/ops/q_per_channel_zero_points_ops.h>
#include <ATen/ops/_fake_quantize_learnable_per_channel_affine_ops.h>
#include <ATen/ops/_lstm_mps_ops.h>
#include <ATen/ops/_thnn_fused_lstm_cell_ops.h>
#include <ATen/ops/lift_fresh_copy_ops.h>
#include <ATen/ops/index_fill_ops.h>
#include <ATen/ops/index_fill_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/cauchy_ops.h>
#include <ATen/ops/cauchy_ops.h>
#include <ATen/ops/log_normal_ops.h>
#include <ATen/ops/log_normal_ops.h>
#include <ATen/ops/_histogramdd_bin_edges_ops.h>
#include <ATen/ops/_histogramdd_from_bin_tensors_ops.h>
#include <ATen/ops/unfold_backward_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/_foreach_sub_ops.h>
#include <ATen/ops/_foreach_sub_ops.h>
#include <ATen/ops/_foreach_sub_ops.h>
#include <ATen/ops/_foreach_maximum_ops.h>
#include <ATen/ops/_foreach_maximum_ops.h>
#include <ATen/ops/_foreach_maximum_ops.h>
#include <ATen/ops/_foreach_acos_ops.h>
#include <ATen/ops/_foreach_atan_ops.h>
#include <ATen/ops/_foreach_ceil_ops.h>
#include <ATen/ops/_foreach_erf_ops.h>
#include <ATen/ops/_foreach_log2_ops.h>
#include <ATen/ops/bucketize_ops.h>
#include <ATen/ops/glu_backward_jvp_ops.h>
#include <ATen/ops/hardswish_backward_ops.h>
#include <ATen/ops/_adaptive_avg_pool3d_backward_ops.h>
#include <ATen/ops/_conj_copy_ops.h>
#include <ATen/ops/detach_copy_ops.h>
#include <ATen/ops/row_indices_copy_ops.h>
#include <ATen/ops/_transformer_encoder_layer_fwd_ops.h>
#include <ATen/ops/_native_multi_head_attention_ops.h>
#include <ATen/ops/_triton_multi_head_attention_ops.h>
#include <ATen/ops/_fused_adamw_ops.h>
#include <ATen/ops/_fused_adamw_ops.h>
#include <ATen/ops/_fused_adamw_ops.h>
#include <ATen/ops/_fused_adamw_ops.h>
#endif

using namespace at;

namespace torch {

namespace TraceType {

namespace {
at::Tensor _cast_Double(c10::DispatchKeySet ks, const at::Tensor & self, bool non_blocking) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cast_Double");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_cast_Double::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _cast_Float(c10::DispatchKeySet ks, const at::Tensor & self, bool non_blocking) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cast_Float");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_cast_Float::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _cast_Half(c10::DispatchKeySet ks, const at::Tensor & self, bool non_blocking) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cast_Half");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_cast_Half::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor align_as(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::align_as");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::align_as::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _assert_tensor_metadata(c10::DispatchKeySet ks, const at::Tensor & a, at::OptionalSymIntArrayRef size, at::OptionalSymIntArrayRef stride, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Device> device, ::std::optional<at::Layout> layout) {
  at::_ops::_assert_tensor_metadata::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), a, size, stride, dtype, device, layout);
}
void sym_constrain_range_for_size(c10::DispatchKeySet ks, const at::Scalar & size, ::std::optional<int64_t> min, ::std::optional<int64_t> max) {
  at::_ops::sym_constrain_range_for_size::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, min, max);
}
at::Tensor refine_names(c10::DispatchKeySet ks, const at::Tensor & self, at::DimnameList names) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::refine_names");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "names", names);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::refine_names::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, names);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
bool _use_cudnn_rnn_flatten_weight(c10::DispatchKeySet ks) {
  auto result =at::_ops::_use_cudnn_rnn_flatten_weight::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer));
  return result;
}
at::Tensor _cudnn_rnn_flatten_weight(c10::DispatchKeySet ks, at::TensorList weight_arr, int64_t weight_stride0, c10::SymInt input_size, int64_t mode, c10::SymInt hidden_size, c10::SymInt proj_size, int64_t num_layers, bool batch_first, bool bidirectional) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cudnn_rnn_flatten_weight");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight_arr", weight_arr);
    jit::tracer::addInputs(node, "weight_stride0", weight_stride0);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "hidden_size", hidden_size);
    jit::tracer::addInputs(node, "proj_size", proj_size);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_cudnn_rnn_flatten_weight::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), weight_arr, weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _sobol_engine_ff_(c10::DispatchKeySet ks, at::Tensor & self, int64_t n, const at::Tensor & sobolstate, int64_t dimension, int64_t num_generated) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::_sobol_engine_ff");
    } else {
      op_name = c10::Symbol::fromQualString("aten::_sobol_engine_ff_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "sobolstate", sobolstate);
    jit::tracer::addInputs(node, "dimension", dimension);
    jit::tracer::addInputs(node, "num_generated", num_generated);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_sobol_engine_ff_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_sobol_engine_ff_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n, sobolstate, dimension, num_generated);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & _sobol_engine_scramble_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & ltm, int64_t dimension) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::_sobol_engine_scramble");
    } else {
      op_name = c10::Symbol::fromQualString("aten::_sobol_engine_scramble_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "ltm", ltm);
    jit::tracer::addInputs(node, "dimension", dimension);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_sobol_engine_scramble_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_sobol_engine_scramble_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, ltm, dimension);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor feature_alpha_dropout(c10::DispatchKeySet ks, const at::Tensor & input, double p, bool train) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::feature_alpha_dropout");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::feature_alpha_dropout::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, p, train);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & feature_alpha_dropout_(c10::DispatchKeySet ks, at::Tensor & self, double p, bool train) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::feature_alpha_dropout");
    } else {
      op_name = c10::Symbol::fromQualString("aten::feature_alpha_dropout_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("feature_alpha_dropout_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::feature_alpha_dropout_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, train);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor abs(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::abs");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::abs::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & abs_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::abs");
    } else {
      op_name = c10::Symbol::fromQualString("aten::abs_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("abs_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::abs_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & abs_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::abs");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("abs_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::abs_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor imag(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::imag");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::imag::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor resolve_conj(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::resolve_conj");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::resolve_conj::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor resolve_neg(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::resolve_neg");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::resolve_neg::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> adaptive_max_pool1d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::adaptive_max_pool1d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::adaptive_max_pool1d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor addmv(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::addmv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat", mat);
    jit::tracer::addInputs(node, "vec", vec);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::addmv::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat, vec, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & addmv_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::addmv");
    } else {
      op_name = c10::Symbol::fromQualString("aten::addmv_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat", mat);
    jit::tracer::addInputs(node, "vec", vec);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addmv_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::addmv_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat, vec, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & addmv_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::addmv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat", mat);
    jit::tracer::addInputs(node, "vec", vec);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addmv_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::addmv_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat, vec, beta, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor addr(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::addr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec1", vec1);
    jit::tracer::addInputs(node, "vec2", vec2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::addr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, vec1, vec2, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & addr_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::addr");
    } else {
      op_name = c10::Symbol::fromQualString("aten::addr_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec1", vec1);
    jit::tracer::addInputs(node, "vec2", vec2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addr_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::addr_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, vec1, vec2, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & addr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::addr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec1", vec1);
    jit::tracer::addInputs(node, "vec2", vec2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addr_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::addr_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, vec1, vec2, beta, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor affine_grid_generator_backward(c10::DispatchKeySet ks, const at::Tensor & grad, c10::SymIntArrayRef size, bool align_corners) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::affine_grid_generator_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::affine_grid_generator_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, size, align_corners);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor argmin(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<int64_t> dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::argmin");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::argmin::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & argmin_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::argmin");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("argmin_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::argmin_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor atan(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::atan");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::atan::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & atan_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::atan");
    } else {
      op_name = c10::Symbol::fromQualString("aten::atan_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("atan_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::atan_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & atan_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::atan");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("atan_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::atan_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor arctan(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::arctan");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::arctan::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & arctan_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::arctan");
    } else {
      op_name = c10::Symbol::fromQualString("aten::arctan_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("arctan_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::arctan_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & arctan_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::arctan");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("arctan_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::arctan_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor quantized_batch_norm(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & var, double eps, double output_scale, int64_t output_zero_point) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantized_batch_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "var", var);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "output_scale", output_scale);
    jit::tracer::addInputs(node, "output_zero_point", output_zero_point);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::quantized_batch_norm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, mean, var, eps, output_scale, output_zero_point);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor binary_cross_entropy_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::binary_cross_entropy_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::binary_cross_entropy_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, target, weight, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & binary_cross_entropy_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::binary_cross_entropy_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("binary_cross_entropy_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::binary_cross_entropy_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, target, weight, reduction, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor bitwise_not(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_not");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bitwise_not::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & bitwise_not_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::bitwise_not");
    } else {
      op_name = c10::Symbol::fromQualString("aten::bitwise_not_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_not_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bitwise_not_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & bitwise_not_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_not");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_not_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bitwise_not_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor logical_not(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logical_not");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::logical_not::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & logical_not_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::logical_not");
    } else {
      op_name = c10::Symbol::fromQualString("aten::logical_not_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logical_not_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::logical_not_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & logical_not_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logical_not");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logical_not_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::logical_not_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor concatenate(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::concatenate");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::concatenate::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & concatenate_out_out(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::concatenate");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("concatenate_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::concatenate_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor concatenate_names(c10::DispatchKeySet ks, at::TensorList tensors, at::Dimname dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::concatenate");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::concatenate_names::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & concatenate_out_names_out(c10::DispatchKeySet ks, at::TensorList tensors, at::Dimname dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::concatenate");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("concatenate_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::concatenate_names_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor ceil(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ceil");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::ceil::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & ceil_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::ceil");
    } else {
      op_name = c10::Symbol::fromQualString("aten::ceil_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ceil_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ceil_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & ceil_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ceil");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ceil_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ceil_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor conv_tbc(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const at::Tensor & bias, int64_t pad) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::conv_tbc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "pad", pad);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::conv_tbc::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, bias, pad);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor cosh(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cosh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cosh::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & cosh_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::cosh");
    } else {
      op_name = c10::Symbol::fromQualString("aten::cosh_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cosh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cosh_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & cosh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cosh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cosh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cosh_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor cosine_embedding_loss(c10::DispatchKeySet ks, const at::Tensor & input1, const at::Tensor & input2, const at::Tensor & target, double margin, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cosine_embedding_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input1", input1);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cosine_embedding_loss::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input1, input2, target, margin, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor cudnn_affine_grid_generator_backward(c10::DispatchKeySet ks, const at::Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cudnn_affine_grid_generator_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "N", N);
    jit::tracer::addInputs(node, "C", C);
    jit::tracer::addInputs(node, "H", H);
    jit::tracer::addInputs(node, "W", W);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto grad_theta =at::_ops::cudnn_affine_grid_generator_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, N, C, H, W);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_theta);
  }
  return grad_theta;
}
at::Tensor cudnn_grid_sampler(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & grid) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cudnn_grid_sampler");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grid", grid);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto output =at::_ops::cudnn_grid_sampler::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grid);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
::std::tuple<at::Tensor,at::Tensor> cummin(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cummin");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::cummin::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor &,at::Tensor &> cummin_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & values, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cummin");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
      jit::tracer::addInputs(node, "indices", indices);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cummin_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cummin_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor,at::Tensor> cummin_dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cummin");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::cummin_dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor &,at::Tensor &> cummin_out_dimname_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, at::Tensor & values, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cummin");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
      jit::tracer::addInputs(node, "indices", indices);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cummin_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cummin_dimname_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
void _cummin_helper(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & values, at::Tensor & indices, int64_t dim) {
  at::_ops::_cummin_helper::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, values, indices, dim);
}
at::Tensor div_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::div");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::div_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & div__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::div");
    } else {
      op_name = c10::Symbol::fromQualString("aten::div_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::div__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & div_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::div");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::div_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor div_Tensor_mode(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::div");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "rounding_mode", rounding_mode);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::div_Tensor_mode::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, rounding_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & div__Tensor_mode(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::div");
    } else {
      op_name = c10::Symbol::fromQualString("aten::div_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "rounding_mode", rounding_mode);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::div__Tensor_mode::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, rounding_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & div_out_out_mode(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::div");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "rounding_mode", rounding_mode);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::div_out_mode::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, rounding_mode, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor div_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::div");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::div_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & div__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::div");
    } else {
      op_name = c10::Symbol::fromQualString("aten::div_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::div__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor div_Scalar_mode(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, ::std::optional<c10::string_view> rounding_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::div");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "rounding_mode", rounding_mode);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::div_Scalar_mode::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, rounding_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & div__Scalar_mode(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other, ::std::optional<c10::string_view> rounding_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::div");
    } else {
      op_name = c10::Symbol::fromQualString("aten::div_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "rounding_mode", rounding_mode);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::div__Scalar_mode::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, rounding_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _embedding_bag_forward_only(c10::DispatchKeySet ks, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const ::std::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_embedding_bag_forward_only");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    jit::tracer::addInputs(node, "include_last_offset", include_last_offset);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3] =at::_ops::_embedding_bag_forward_only::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> embedding_bag(c10::DispatchKeySet ks, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const ::std::optional<at::Tensor> & per_sample_weights, bool include_last_offset) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::embedding_bag");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    jit::tracer::addInputs(node, "include_last_offset", include_last_offset);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3] =at::_ops::embedding_bag::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> embedding_bag_padding_idx(c10::DispatchKeySet ks, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const ::std::optional<at::Tensor> & per_sample_weights, bool include_last_offset, ::std::optional<int64_t> padding_idx) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::embedding_bag");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    jit::tracer::addInputs(node, "include_last_offset", include_last_offset);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3] =at::_ops::embedding_bag_padding_idx::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
at::Tensor new_zeros(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::new_zeros");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::new_zeros::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor erf(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::erf");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::erf::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & erf_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::erf");
    } else {
      op_name = c10::Symbol::fromQualString("aten::erf_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erf_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::erf_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & erf_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::erf");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erf_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::erf_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor grid_sampler(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::grid_sampler");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::grid_sampler::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, grid, interpolation_mode, padding_mode, align_corners);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _grid_sampler_2d_cpu_fallback(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_grid_sampler_2d_cpu_fallback");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_grid_sampler_2d_cpu_fallback::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, grid, interpolation_mode, padding_mode, align_corners);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor grid_sampler_3d(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::grid_sampler_3d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::grid_sampler_3d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, grid, interpolation_mode, padding_mode, align_corners);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor hann_window(c10::DispatchKeySet ks, int64_t window_length, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hann_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hann_window::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor hann_window_periodic(c10::DispatchKeySet ks, int64_t window_length, bool periodic, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hann_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hann_window_periodic::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor hamming_window(c10::DispatchKeySet ks, int64_t window_length, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hamming_window::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor hamming_window_periodic(c10::DispatchKeySet ks, int64_t window_length, bool periodic, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hamming_window_periodic::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor hamming_window_periodic_alpha(c10::DispatchKeySet ks, int64_t window_length, bool periodic, double alpha, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hamming_window_periodic_alpha::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, alpha, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor hamming_window_periodic_alpha_beta(c10::DispatchKeySet ks, int64_t window_length, bool periodic, double alpha, double beta, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hamming_window_periodic_alpha_beta::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, alpha, beta, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_group_norm_backward(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & rstd, const ::std::optional<at::Tensor> & weight, c10::SymInt N, c10::SymInt C, c10::SymInt HxW, int64_t group, ::std::array<bool,3> output_mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::native_group_norm_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "rstd", rstd);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "N", N);
    jit::tracer::addInputs(node, "C", C);
    jit::tracer::addInputs(node, "HxW", HxW);
    jit::tracer::addInputs(node, "group", group);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::native_group_norm_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
at::Tensor _fft_c2c(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef dim, int64_t normalization, bool forward) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fft_c2c");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "normalization", normalization);
    jit::tracer::addInputs(node, "forward", forward);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_fft_c2c::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, normalization, forward);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _fft_c2c_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef dim, int64_t normalization, bool forward, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fft_c2c");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "normalization", normalization);
    jit::tracer::addInputs(node, "forward", forward);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_fft_c2c_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_fft_c2c_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, normalization, forward, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void _validate_compressed_sparse_indices(c10::DispatchKeySet ks, bool is_crow, const at::Tensor & compressed_idx, const at::Tensor & plain_idx, int64_t cdim, int64_t dim, int64_t nnz) {
  at::_ops::_validate_compressed_sparse_indices::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), is_crow, compressed_idx, plain_idx, cdim, dim, nnz);
}
int64_t _cufft_get_plan_cache_size(c10::DispatchKeySet ks, at::DeviceIndex device_index) {
  auto result =at::_ops::_cufft_get_plan_cache_size::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), device_index);
  return result;
}
int64_t _cufft_get_plan_cache_max_size(c10::DispatchKeySet ks, at::DeviceIndex device_index) {
  auto result =at::_ops::_cufft_get_plan_cache_max_size::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), device_index);
  return result;
}
at::Tensor index_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::index");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::index_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & index_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::index");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::index_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, indices, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor isnan(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::isnan");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::isnan::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> kthvalue(c10::DispatchKeySet ks, const at::Tensor & self, int64_t k, int64_t dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::kthvalue");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::kthvalue::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, k, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor &,at::Tensor &> kthvalue_out_values(c10::DispatchKeySet ks, const at::Tensor & self, int64_t k, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::kthvalue");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
      jit::tracer::addInputs(node, "indices", indices);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("kthvalue_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::kthvalue_values::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, k, dim, keepdim, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor,at::Tensor> kthvalue_dimname(c10::DispatchKeySet ks, const at::Tensor & self, int64_t k, at::Dimname dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::kthvalue");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::kthvalue_dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, k, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor &,at::Tensor &> kthvalue_out_dimname_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t k, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::kthvalue");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
      jit::tracer::addInputs(node, "indices", indices);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("kthvalue_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::kthvalue_dimname_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, k, dim, keepdim, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm(c10::DispatchKeySet ks, const at::Tensor & input, c10::SymIntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::native_layer_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "normalized_shape", normalized_shape);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::native_layer_norm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, normalized_shape, weight, bias, eps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
at::Tensor nan_to_num(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<double> nan, ::std::optional<double> posinf, ::std::optional<double> neginf) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nan_to_num");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "nan", nan);
    jit::tracer::addInputs(node, "posinf", posinf);
    jit::tracer::addInputs(node, "neginf", neginf);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::nan_to_num::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, nan, posinf, neginf);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & nan_to_num_(c10::DispatchKeySet ks, at::Tensor & self, ::std::optional<double> nan, ::std::optional<double> posinf, ::std::optional<double> neginf) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::nan_to_num");
    } else {
      op_name = c10::Symbol::fromQualString("aten::nan_to_num_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "nan", nan);
    jit::tracer::addInputs(node, "posinf", posinf);
    jit::tracer::addInputs(node, "neginf", neginf);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nan_to_num_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::nan_to_num_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, nan, posinf, neginf);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & nan_to_num_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<double> nan, ::std::optional<double> posinf, ::std::optional<double> neginf, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nan_to_num");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "nan", nan);
    jit::tracer::addInputs(node, "posinf", posinf);
    jit::tracer::addInputs(node, "neginf", neginf);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nan_to_num_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::nan_to_num_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, nan, posinf, neginf, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor> _sparse_semi_structured_apply(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & thread_masks) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_semi_structured_apply");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "thread_masks", thread_masks);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_sparse_semi_structured_apply::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, thread_masks);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor _sparse_semi_structured_mm(c10::DispatchKeySet ks, const at::Tensor & mat1, const at::Tensor & mat1_meta, const at::Tensor & mat2, ::std::optional<at::ScalarType> out_dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_semi_structured_mm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat1_meta", mat1_meta);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "out_dtype", out_dtype);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_sparse_semi_structured_mm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), mat1, mat1_meta, mat2, out_dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor fbgemm_linear_int8_weight_fp32_activation(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & packed, const at::Tensor & col_offsets, const at::Scalar & weight_scale, const at::Scalar & weight_zero_point, const at::Tensor & bias) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fbgemm_linear_int8_weight_fp32_activation");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "packed", packed);
    jit::tracer::addInputs(node, "col_offsets", col_offsets);
    jit::tracer::addInputs(node, "weight_scale", weight_scale);
    jit::tracer::addInputs(node, "weight_zero_point", weight_zero_point);
    jit::tracer::addInputs(node, "bias", bias);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fbgemm_linear_int8_weight_fp32_activation::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor fbgemm_linear_int8_weight(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & packed, const at::Tensor & col_offsets, const at::Scalar & weight_scale, const at::Scalar & weight_zero_point, const at::Tensor & bias) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fbgemm_linear_int8_weight");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "packed", packed);
    jit::tracer::addInputs(node, "col_offsets", col_offsets);
    jit::tracer::addInputs(node, "weight_scale", weight_scale);
    jit::tracer::addInputs(node, "weight_zero_point", weight_zero_point);
    jit::tracer::addInputs(node, "bias", bias);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fbgemm_linear_int8_weight::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor fbgemm_linear_fp16_weight_fp32_activation(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & packed_weight, const at::Tensor & bias) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fbgemm_linear_fp16_weight_fp32_activation");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "packed_weight", packed_weight);
    jit::tracer::addInputs(node, "bias", bias);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fbgemm_linear_fp16_weight_fp32_activation::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, packed_weight, bias);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor xlogy_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::xlogy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::xlogy_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor xlogy_Scalar_Self(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::xlogy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::xlogy_Scalar_Self::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor xlogy_Scalar_Other(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::xlogy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::xlogy_Scalar_Other::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & xlogy__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::xlogy");
    } else {
      op_name = c10::Symbol::fromQualString("aten::xlogy_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("xlogy_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::xlogy__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & xlogy__Scalar_Other(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::xlogy");
    } else {
      op_name = c10::Symbol::fromQualString("aten::xlogy_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("xlogy_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::xlogy__Scalar_Other::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & xlogy_out_OutTensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::xlogy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("xlogy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::xlogy_OutTensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & xlogy_out_OutScalar_Self(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::xlogy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("xlogy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::xlogy_OutScalar_Self::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & xlogy_out_OutScalar_Other(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::xlogy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("xlogy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::xlogy_OutScalar_Other::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _log_softmax_backward_data(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_log_softmax_backward_data");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "input_dtype", input_dtype);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_log_softmax_backward_data::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output, dim, input_dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _log_softmax_backward_data_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_log_softmax_backward_data");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "input_dtype", input_dtype);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_log_softmax_backward_data_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_log_softmax_backward_data_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output, dim, input_dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor logcumsumexp(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logcumsumexp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::logcumsumexp::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & logcumsumexp_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logcumsumexp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logcumsumexp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::logcumsumexp_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor logcumsumexp_dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logcumsumexp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::logcumsumexp_dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & logcumsumexp_out_dimname_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logcumsumexp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logcumsumexp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::logcumsumexp_dimname_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor matrix_exp_backward(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & grad) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::matrix_exp_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grad", grad);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::matrix_exp_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grad);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor amax(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::amax");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::amax::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & amax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::amax");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("amax_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::amax_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor mkldnn_max_pool2d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_max_pool2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::mkldnn_max_pool2d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, dilation, ceil_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor quantized_max_pool2d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantized_max_pool2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::quantized_max_pool2d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, dilation, ceil_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor amin(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::amin");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::amin::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & amin_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::amin");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("amin_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::amin_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _mps_convolution(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_mps_convolution");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_mps_convolution::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, bias, padding, stride, dilation, groups);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> mkldnn_rnn_layer_backward(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight1, const at::Tensor & weight2, const at::Tensor & weight3, const at::Tensor & weight4, const at::Tensor & hx_, const at::Tensor & cx_tmp, const at::Tensor & output, const at::Tensor & hy_, const at::Tensor & cy_, const ::std::optional<at::Tensor> & grad_output, const ::std::optional<at::Tensor> & grad_hy, const ::std::optional<at::Tensor> & grad_cy, bool reverse, int64_t mode, int64_t hidden_size, int64_t num_layers, bool has_biases, bool train, bool bidirectional, at::IntArrayRef batch_sizes, bool batch_first, const at::Tensor & workspace) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_rnn_layer_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight1", weight1);
    jit::tracer::addInputs(node, "weight2", weight2);
    jit::tracer::addInputs(node, "weight3", weight3);
    jit::tracer::addInputs(node, "weight4", weight4);
    jit::tracer::addInputs(node, "hx_", hx_);
    jit::tracer::addInputs(node, "cx_tmp", cx_tmp);
    jit::tracer::addInputs(node, "output", output);
    jit::tracer::addInputs(node, "hy_", hy_);
    jit::tracer::addInputs(node, "cy_", cy_);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "grad_hy", grad_hy);
    jit::tracer::addInputs(node, "grad_cy", grad_cy);
    jit::tracer::addInputs(node, "reverse", reverse);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "hidden_size", hidden_size);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "workspace", workspace);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3, result4, result5, result6] =at::_ops::mkldnn_rnn_layer_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight1, weight2, weight3, weight4, hx_, cx_tmp, output, hy_, cy_, grad_output, grad_hy, grad_cy, reverse, mode, hidden_size, num_layers, has_biases, train, bidirectional, batch_sizes, batch_first, workspace);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
    jit::tracer::addOutput(node, result4);
    jit::tracer::addOutput(node, result5);
    jit::tracer::addOutput(node, result6);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4), std::move(result5), std::move(result6));
}
at::Tensor miopen_depthwise_convolution(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, bool benchmark, bool deterministic) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::miopen_depthwise_convolution");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::miopen_depthwise_convolution::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _int_mm(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_int_mm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_int_mm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _int_mm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_int_mm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_int_mm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_int_mm_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat2, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _dyn_quant_pack_4bit_weight(c10::DispatchKeySet ks, const at::Tensor & weights, const at::Tensor & scales_zeros, const ::std::optional<at::Tensor> & bias, int64_t block_size, int64_t in_features, int64_t out_features) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_dyn_quant_pack_4bit_weight");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weights", weights);
    jit::tracer::addInputs(node, "scales_zeros", scales_zeros);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "block_size", block_size);
    jit::tracer::addInputs(node, "in_features", in_features);
    jit::tracer::addInputs(node, "out_features", out_features);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_dyn_quant_pack_4bit_weight::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), weights, scales_zeros, bias, block_size, in_features, out_features);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _weight_int8pack_mm(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2, const at::Tensor & scales) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_weight_int8pack_mm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "scales", scales);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_weight_int8pack_mm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat2, scales);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double momentum, double eps) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::native_batch_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::native_batch_norm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, running_mean, running_var, training, momentum, eps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_batch_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double momentum, double eps, at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::native_batch_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
      jit::tracer::addInputs(node, "save_mean", save_mean);
      jit::tracer::addInputs(node, "save_invstd", save_invstd);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("native_batch_norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::native_batch_norm_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, running_mean, running_var, training, momentum, eps, out, save_mean, save_invstd);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
    jit::tracer::addOutput(node, save_mean);
    jit::tracer::addOutput(node, save_invstd);
  }
  return std::forward_as_tuple(out, save_mean, save_invstd);
}
::std::tuple<at::Tensor,at::Tensor> batch_norm_stats(c10::DispatchKeySet ks, const at::Tensor & input, double eps) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::batch_norm_stats");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::batch_norm_stats::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, eps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor> batch_norm_gather_stats(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, double momentum, double eps, int64_t count) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::batch_norm_gather_stats");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "invstd", invstd);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "count", count);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::batch_norm_gather_stats::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, mean, invstd, running_mean, running_var, momentum, eps, count);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm_backward(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, const ::std::optional<at::Tensor> & save_mean, const ::std::optional<at::Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::native_batch_norm_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "save_mean", save_mean);
    jit::tracer::addInputs(node, "save_invstd", save_invstd);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::native_batch_norm_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> batch_norm_backward_reduce(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const ::std::optional<at::Tensor> & weight, bool input_g, bool weight_g, bool bias_g) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::batch_norm_backward_reduce");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "invstd", invstd);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "input_g", input_g);
    jit::tracer::addInputs(node, "weight_g", weight_g);
    jit::tracer::addInputs(node, "bias_g", bias_g);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3] =at::_ops::batch_norm_backward_reduce::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
bool is_vulkan_available(c10::DispatchKeySet ks) {
  auto result =at::_ops::is_vulkan_available::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer));
  return result;
}
at::Tensor _nnpack_spatial_convolution(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nnpack_spatial_convolution");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_nnpack_spatial_convolution::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, padding, stride);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor ones_names(c10::DispatchKeySet ks, at::IntArrayRef size, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ones");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::ones_names::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, names, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor ones(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ones");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::ones::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & ones_out_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ones");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ones_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ones_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _cdist_forward(c10::DispatchKeySet ks, const at::Tensor & x1, const at::Tensor & x2, double p, ::std::optional<int64_t> compute_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cdist_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "compute_mode", compute_mode);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_cdist_forward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x1, x2, p, compute_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor cosine_similarity(c10::DispatchKeySet ks, const at::Tensor & x1, const at::Tensor & x2, int64_t dim, double eps) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cosine_similarity");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cosine_similarity::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x1, x2, dim, eps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor movedim_intlist(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef source, at::IntArrayRef destination) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::movedim");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "destination", destination);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::movedim_intlist::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, source, destination);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor movedim_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t source, int64_t destination) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::movedim");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "destination", destination);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::movedim_int::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, source, destination);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor numpy_T(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::numpy_T");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::numpy_T::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor mH(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mH");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::mH::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor rand_like(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rand_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::rand_like::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor randint_like(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt high, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::randint_like::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, high, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor randint_like_low_dtype(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt low, c10::SymInt high, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::randint_like_low_dtype::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, low, high, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor round(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::round");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::round::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & round_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::round");
    } else {
      op_name = c10::Symbol::fromQualString("aten::round_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("round_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::round_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & round_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::round");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("round_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::round_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor round_decimals(c10::DispatchKeySet ks, const at::Tensor & self, int64_t decimals) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::round");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "decimals", decimals);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::round_decimals::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, decimals);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & round__decimals(c10::DispatchKeySet ks, at::Tensor & self, int64_t decimals) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::round");
    } else {
      op_name = c10::Symbol::fromQualString("aten::round_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "decimals", decimals);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("round_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::round__decimals::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, decimals);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & round_out_decimals_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t decimals, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::round");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "decimals", decimals);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("round_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::round_decimals_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, decimals, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & gelu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::string_view approximate, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::gelu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "approximate", approximate);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gelu_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::gelu_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, approximate, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & gelu_(c10::DispatchKeySet ks, at::Tensor & self, c10::string_view approximate) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::gelu");
    } else {
      op_name = c10::Symbol::fromQualString("aten::gelu_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "approximate", approximate);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gelu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::gelu_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, approximate);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor gelu(c10::DispatchKeySet ks, const at::Tensor & self, c10::string_view approximate) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::gelu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "approximate", approximate);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::gelu::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, approximate);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & hardshrink_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hardshrink");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hardshrink_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hardshrink_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, lambd, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor hardshrink(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & lambd) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hardshrink");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hardshrink::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, lambd);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor select_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef input_sizes, int64_t dim, c10::SymInt index) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::select_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input_sizes", input_sizes);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::select_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, input_sizes, dim, index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor mish(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mish");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::mish::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & mish_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::mish");
    } else {
      op_name = c10::Symbol::fromQualString("aten::mish_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mish_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mish_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & mish_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mish");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mish_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mish_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor sigmoid(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sigmoid");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sigmoid::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & sigmoid_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::sigmoid");
    } else {
      op_name = c10::Symbol::fromQualString("aten::sigmoid_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sigmoid_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sigmoid_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & sigmoid_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sigmoid");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sigmoid_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sigmoid_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
int64_t size_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  auto result =at::_ops::size_int::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  return result;
}
int64_t size_Dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim) {
  auto result =at::_ops::size_Dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  return result;
}
c10::SymInt sym_numel(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::sym_numel::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
at::Tensor slice_scatter(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & src, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::slice_scatter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::slice_scatter::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, src, dim, start, end, step);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _softmax_backward_data(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_softmax_backward_data");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "input_dtype", input_dtype);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_softmax_backward_data::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output, dim, input_dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _softmax_backward_data_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_softmax_backward_data");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "input_dtype", input_dtype);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_softmax_backward_data_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_softmax_backward_data_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output, dim, input_dtype, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
::std::vector<at::Tensor> split_with_sizes(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef split_sizes, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::split_with_sizes");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "split_sizes", split_sizes);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::split_with_sizes::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, split_sizes, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> hsplit_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t sections) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hsplit");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "sections", sections);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hsplit_int::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, sections);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> hsplit_array(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hsplit");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hsplit_array::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor stack(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::stack");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::stack::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & stack_out_out(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::stack");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("stack_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::stack_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _stack(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_stack");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_stack::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _stack_out_out(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_stack");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_stack_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_stack_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor square(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::square");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::square::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & square_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::square");
    } else {
      op_name = c10::Symbol::fromQualString("aten::square_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("square_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::square_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & square_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::square");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("square_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::square_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor tanh(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::tanh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::tanh::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & tanh_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::tanh");
    } else {
      op_name = c10::Symbol::fromQualString("aten::tanh_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tanh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::tanh_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & tanh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::tanh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tanh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::tanh_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor tensordot(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::IntArrayRef dims_self, at::IntArrayRef dims_other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::tensordot");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dims_self", dims_self);
    jit::tracer::addInputs(node, "dims_other", dims_other);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::tensordot::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, dims_self, dims_other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & tensordot_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::IntArrayRef dims_self, at::IntArrayRef dims_other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::tensordot");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dims_self", dims_self);
    jit::tracer::addInputs(node, "dims_other", dims_other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tensordot_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::tensordot_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, dims_self, dims_other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor tile(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef dims) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::tile");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dims", dims);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::tile::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dims);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _mkldnn_transpose(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim0, int64_t dim1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_mkldnn_transpose");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim0", dim0);
    jit::tracer::addInputs(node, "dim1", dim1);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_mkldnn_transpose::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim0, dim1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _mkldnn_transpose_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim0, int64_t dim1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::_mkldnn_transpose");
    } else {
      op_name = c10::Symbol::fromQualString("aten::_mkldnn_transpose_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim0", dim0);
    jit::tracer::addInputs(node, "dim1", dim1);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_mkldnn_transpose_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_mkldnn_transpose_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim0, dim1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor fliplr(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fliplr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fliplr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _nested_from_padded_and_nested_example(c10::DispatchKeySet ks, const at::Tensor & padded, const at::Tensor & nt_example) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_from_padded_and_nested_example");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "padded", padded);
    jit::tracer::addInputs(node, "nt_example", nt_example);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_nested_from_padded_and_nested_example::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), padded, nt_example);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _nested_get_max_seqlen(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_get_max_seqlen");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_nested_get_max_seqlen::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> _nested_compute_contiguous_strides_offsets(c10::DispatchKeySet ks, const at::Tensor & nested_size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_compute_contiguous_strides_offsets");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "nested_size", nested_size);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_nested_compute_contiguous_strides_offsets::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), nested_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor fix(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fix");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fix::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fix_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::fix");
    } else {
      op_name = c10::Symbol::fromQualString("aten::fix_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fix_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fix_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & fix_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fix");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fix_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fix_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_dim(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unique_dim");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "sorted", sorted);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);
    jit::tracer::addInputs(node, "return_counts", return_counts);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::unique_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, sorted, return_inverse, return_counts);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_consecutive(c10::DispatchKeySet ks, const at::Tensor & self, bool return_inverse, bool return_counts, ::std::optional<int64_t> dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unique_consecutive");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);
    jit::tracer::addInputs(node, "return_counts", return_counts);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::unique_consecutive::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, return_inverse, return_counts, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
at::Tensor vander(c10::DispatchKeySet ks, const at::Tensor & x, ::std::optional<int64_t> N, bool increasing) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::vander");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "N", N);
    jit::tracer::addInputs(node, "increasing", increasing);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::vander::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, N, increasing);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor view_as(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::view_as");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::view_as::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _dirichlet_grad(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & alpha, const at::Tensor & total) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_dirichlet_grad");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "total", total);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_dirichlet_grad::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, alpha, total);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> batch_norm_backward(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, const ::std::optional<at::Tensor> & save_mean, const ::std::optional<at::Tensor> & save_var, bool update, double eps, ::std::array<bool,3> output_mask, const at::Tensor & reserve) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::batch_norm_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "save_mean", save_mean);
    jit::tracer::addInputs(node, "save_var", save_var);
    jit::tracer::addInputs(node, "update", update);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    jit::tracer::addInputs(node, "reserve", reserve);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::batch_norm_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, input, weight, running_mean, running_var, save_mean, save_var, update, eps, output_mask, reserve);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
at::Tensor frobenius_norm_dim(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::frobenius_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::frobenius_norm_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & frobenius_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::frobenius_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("frobenius_norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::frobenius_norm_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor clone(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::clone");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::clone::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor positive(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::positive");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::positive::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
const at::Tensor & resize_as_sparse_(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & the_template) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::resize_as_sparse");
    } else {
      op_name = c10::Symbol::fromQualString("aten::resize_as_sparse_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "the_template", the_template);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("resize_as_sparse_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::resize_as_sparse_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, the_template);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & sparse_sampled_addmm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sparse_sampled_addmm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sparse_sampled_addmm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sparse_sampled_addmm_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat1, mat2, beta, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor sparse_sampled_addmm(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sparse_sampled_addmm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sparse_sampled_addmm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat1, mat2, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor sparse_csr_tensor_crow_col_value_size(c10::DispatchKeySet ks, const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sparse_csr_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "crow_indices", crow_indices);
    jit::tracer::addInputs(node, "col_indices", col_indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sparse_csr_tensor_crow_col_value_size::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), crow_indices, col_indices, values, size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor sparse_csr_tensor_crow_col_value(c10::DispatchKeySet ks, const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sparse_csr_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "crow_indices", crow_indices);
    jit::tracer::addInputs(node, "col_indices", col_indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sparse_csr_tensor_crow_col_value::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), crow_indices, col_indices, values, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _sparse_bsc_tensor_unsafe(c10::DispatchKeySet ks, const at::Tensor & ccol_indices, const at::Tensor & row_indices, const at::Tensor & values, at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_bsc_tensor_unsafe");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "ccol_indices", ccol_indices);
    jit::tracer::addInputs(node, "row_indices", row_indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_sparse_bsc_tensor_unsafe::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), ccol_indices, row_indices, values, size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
int64_t dense_dim(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::dense_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
int64_t _dimV(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::_dimV::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
at::Tensor coalesce(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::coalesce");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::coalesce::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _indices(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_indices");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_indices::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor to_sparse_csc(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<int64_t> dense_dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::to_sparse_csc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::to_sparse_csc::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dense_dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _to_sparse_csc(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<int64_t> dense_dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_to_sparse_csc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_to_sparse_csc::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dense_dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _to_sparse_bsc(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef blocksize, ::std::optional<int64_t> dense_dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_to_sparse_bsc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "blocksize", blocksize);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_to_sparse_bsc::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, blocksize, dense_dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor mkldnn_reorder_conv2d_weight(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, at::OptionalSymIntArrayRef input_size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_reorder_conv2d_weight");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "input_size", input_size);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::mkldnn_reorder_conv2d_weight::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding, stride, dilation, groups, input_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor quantize_per_channel(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, at::ScalarType dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantize_per_channel");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scales", scales);
    jit::tracer::addInputs(node, "zero_points", zero_points);
    jit::tracer::addInputs(node, "axis", axis);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::quantize_per_channel::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scales, zero_points, axis, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor dequantize_self(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::dequantize");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::dequantize_self::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> dequantize_tensors(c10::DispatchKeySet ks, at::TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::dequantize");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::dequantize_tensors::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor q_per_channel_zero_points(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::q_per_channel_zero_points");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::q_per_channel_zero_points::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor fake_quantize_per_tensor_affine(c10::DispatchKeySet ks, const at::Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fake_quantize_per_tensor_affine");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fake_quantize_per_tensor_affine::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, quant_min, quant_max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor fake_quantize_per_tensor_affine_tensor_qparams(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t quant_min, int64_t quant_max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fake_quantize_per_tensor_affine");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fake_quantize_per_tensor_affine_tensor_qparams::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, quant_min, quant_max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _fake_quantize_learnable_per_channel_affine(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fake_quantize_learnable_per_channel_affine");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "axis", axis);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    jit::tracer::addInputs(node, "grad_factor", grad_factor);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_fake_quantize_learnable_per_channel_affine::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, axis, quant_min, quant_max, grad_factor);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _autocast_to_full_precision(c10::DispatchKeySet ks, const at::Tensor & self, bool cuda_enabled, bool cpu_enabled) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_autocast_to_full_precision");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "cuda_enabled", cuda_enabled);
    jit::tracer::addInputs(node, "cpu_enabled", cpu_enabled);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_autocast_to_full_precision::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, cuda_enabled, cpu_enabled);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor to_dtype_layout(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, bool non_blocking, bool copy, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::to");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    jit::tracer::addInputs(node, "copy", copy);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::to_dtype_layout::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor to_device(c10::DispatchKeySet ks, const at::Tensor & self, at::Device device, at::ScalarType dtype, bool non_blocking, bool copy, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::to");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    jit::tracer::addInputs(node, "copy", copy);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::to_device::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, device, dtype, non_blocking, copy, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor to_dtype(c10::DispatchKeySet ks, const at::Tensor & self, at::ScalarType dtype, bool non_blocking, bool copy, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::to");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    jit::tracer::addInputs(node, "copy", copy);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::to_dtype::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dtype, non_blocking, copy, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor to_other(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, bool non_blocking, bool copy, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::to");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    jit::tracer::addInputs(node, "copy", copy);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::to_other::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, non_blocking, copy, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor combinations(c10::DispatchKeySet ks, const at::Tensor & self, int64_t r, bool with_replacement) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::combinations");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "r", r);
    jit::tracer::addInputs(node, "with_replacement", with_replacement);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::combinations::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, r, with_replacement);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Scalar item(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::item::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _lstm_mps(c10::DispatchKeySet ks, const at::Tensor & input, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_lstm_mps");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3, result4, result5] =at::_ops::_lstm_mps::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
    jit::tracer::addOutput(node, result4);
    jit::tracer::addOutput(node, result5);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4), std::move(result5));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _thnn_fused_lstm_cell(c10::DispatchKeySet ks, const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & cx, const ::std::optional<at::Tensor> & input_bias, const ::std::optional<at::Tensor> & hidden_bias) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_thnn_fused_lstm_cell");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input_gates", input_gates);
    jit::tracer::addInputs(node, "hidden_gates", hidden_gates);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "input_bias", input_bias);
    jit::tracer::addInputs(node, "hidden_bias", hidden_bias);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::_thnn_fused_lstm_cell::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input_gates, hidden_gates, cx, input_bias, hidden_bias);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> lstm_input(c10::DispatchKeySet ks, const at::Tensor & input, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::lstm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::lstm_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> lstm_data(c10::DispatchKeySet ks, const at::Tensor & data, const at::Tensor & batch_sizes, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::lstm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "data", data);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::lstm_data::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
::std::tuple<at::Tensor,at::Tensor> gru_input(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::gru");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::gru_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor> gru_data(c10::DispatchKeySet ks, const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::gru");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "data", data);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::gru_data::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor> rnn_tanh_input(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rnn_tanh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::rnn_tanh_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor> rnn_tanh_data(c10::DispatchKeySet ks, const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rnn_tanh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "data", data);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::rnn_tanh_data::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor rnn_relu_cell(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const ::std::optional<at::Tensor> & b_ih, const ::std::optional<at::Tensor> & b_hh) {
  auto result =at::_ops::rnn_relu_cell::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, hx, w_ih, w_hh, b_ih, b_hh);
  return result;
}
::std::tuple<at::Tensor,at::Tensor> _pad_packed_sequence(c10::DispatchKeySet ks, const at::Tensor & data, const at::Tensor & batch_sizes, bool batch_first, const at::Scalar & padding_value, int64_t total_length) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_pad_packed_sequence");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "data", data);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "padding_value", padding_value);
    jit::tracer::addInputs(node, "total_length", total_length);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_pad_packed_sequence::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), data, batch_sizes, batch_first, padding_value, total_length);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor lift_fresh_copy(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::lift_fresh_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::lift_fresh_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & index_reduce_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, c10::string_view reduce, bool include_self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::index_reduce");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "reduce", reduce);
    jit::tracer::addInputs(node, "include_self", include_self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_reduce_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::index_reduce_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, source, reduce, include_self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & index_reduce_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, c10::string_view reduce, bool include_self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::index_reduce");
    } else {
      op_name = c10::Symbol::fromQualString("aten::index_reduce_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "reduce", reduce);
    jit::tracer::addInputs(node, "include_self", include_self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_reduce_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::index_reduce_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, source, reduce, include_self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor index_reduce(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, c10::string_view reduce, bool include_self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::index_reduce");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "reduce", reduce);
    jit::tracer::addInputs(node, "include_self", include_self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::index_reduce::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, source, reduce, include_self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & index_fill__int_Scalar(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::index_fill");
    } else {
      op_name = c10::Symbol::fromQualString("aten::index_fill_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::index_fill__int_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor index_fill_int_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::index_fill");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::index_fill_int_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & index_fill__int_Tensor(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::index_fill");
    } else {
      op_name = c10::Symbol::fromQualString("aten::index_fill_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::index_fill__int_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor index_fill_int_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::index_fill");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::index_fill_int_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & index_fill__Dimname_Scalar(c10::DispatchKeySet ks, at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::index_fill");
    } else {
      op_name = c10::Symbol::fromQualString("aten::index_fill_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::index_fill__Dimname_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & index_fill__Dimname_Tensor(c10::DispatchKeySet ks, at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::index_fill");
    } else {
      op_name = c10::Symbol::fromQualString("aten::index_fill_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::index_fill__Dimname_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor index_fill_Dimname_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::index_fill");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::index_fill_Dimname_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor index_fill_Dimname_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::index_fill");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::index_fill_Dimname_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor scatter_add(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::scatter_add");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::scatter_add::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, src);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & scatter_add_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::scatter_add");
    } else {
      op_name = c10::Symbol::fromQualString("aten::scatter_add_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_add_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::scatter_add_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, src);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & scatter_add_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::scatter_add");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_add_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::scatter_add_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, src, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor scatter_add_dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::scatter_add");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::scatter_add_dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, src);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & digamma_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::digamma");
    } else {
      op_name = c10::Symbol::fromQualString("aten::digamma_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("digamma_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::digamma_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & random__from(c10::DispatchKeySet ks, at::Tensor & self, int64_t from, ::std::optional<int64_t> to, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::random");
    } else {
      op_name = c10::Symbol::fromQualString("aten::random_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "from", from);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("random_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::random__from::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, from, to, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & random__to(c10::DispatchKeySet ks, at::Tensor & self, int64_t to, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::random");
    } else {
      op_name = c10::Symbol::fromQualString("aten::random_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("random_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::random__to::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, to, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & random_(c10::DispatchKeySet ks, at::Tensor & self, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::random");
    } else {
      op_name = c10::Symbol::fromQualString("aten::random_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("random_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::random_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & cauchy_(c10::DispatchKeySet ks, at::Tensor & self, double median, double sigma, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::cauchy");
    } else {
      op_name = c10::Symbol::fromQualString("aten::cauchy_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "median", median);
    jit::tracer::addInputs(node, "sigma", sigma);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cauchy_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cauchy_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, median, sigma, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & log_normal_(c10::DispatchKeySet ks, at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::log_normal");
    } else {
      op_name = c10::Symbol::fromQualString("aten::log_normal_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_normal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::log_normal_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mean, std, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & cross_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, ::std::optional<int64_t> dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cross");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cross_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cross_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor cross(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, ::std::optional<int64_t> dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cross");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cross::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & ne_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ne");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ne_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor ne_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ne");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::ne_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & ne_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ne");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ne_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor ne_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ne");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::ne_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & ne__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::ne");
    } else {
      op_name = c10::Symbol::fromQualString("aten::ne_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ne__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & ne__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::ne");
    } else {
      op_name = c10::Symbol::fromQualString("aten::ne_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ne__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & ge_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ge");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ge_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ge_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor ge_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ge");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::ge_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & ge_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ge");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ge_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ge_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor ge_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ge");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::ge_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & ge__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::ge");
    } else {
      op_name = c10::Symbol::fromQualString("aten::ge_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ge_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ge__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & ge__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::ge");
    } else {
      op_name = c10::Symbol::fromQualString("aten::ge_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ge_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ge__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor _gather_sparse_backward(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & grad) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_gather_sparse_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "grad", grad);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_gather_sparse_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, grad);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor linalg_vander(c10::DispatchKeySet ks, const at::Tensor & x, ::std::optional<c10::SymInt> N) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_vander");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "N", N);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_vander::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, N);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor swapaxes(c10::DispatchKeySet ks, const at::Tensor & self, int64_t axis0, int64_t axis1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::swapaxes");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "axis0", axis0);
    jit::tracer::addInputs(node, "axis1", axis1);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::swapaxes::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, axis0, axis1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & swapaxes_(c10::DispatchKeySet ks, at::Tensor & self, int64_t axis0, int64_t axis1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::swapaxes");
    } else {
      op_name = c10::Symbol::fromQualString("aten::swapaxes_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "axis0", axis0);
    jit::tracer::addInputs(node, "axis1", axis1);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("swapaxes_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::swapaxes_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, axis0, axis1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & cholesky_solve_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & input2, bool upper, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cholesky_solve");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "upper", upper);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cholesky_solve_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cholesky_solve_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, input2, upper, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor cholesky_solve(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & input2, bool upper) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cholesky_solve");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cholesky_solve::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, input2, upper);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor &,at::Tensor &> qr_out_Q(c10::DispatchKeySet ks, const at::Tensor & self, bool some, at::Tensor & Q, at::Tensor & R) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::qr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "some", some);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "Q", Q);
      jit::tracer::addInputs(node, "R", R);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("qr_out", Q);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::qr_Q::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, some, Q, R);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, Q);
    jit::tracer::addOutput(node, R);
  }
  return std::forward_as_tuple(Q, R);
}
::std::tuple<at::Tensor,at::Tensor> qr(c10::DispatchKeySet ks, const at::Tensor & self, bool some) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::qr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "some", some);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [Q, R] =at::_ops::qr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, some);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, Q);
    jit::tracer::addOutput(node, R);
  }
  return std::make_tuple(std::move(Q), std::move(R));
}
at::Tensor & digamma_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::digamma");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("digamma_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::digamma_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor digamma(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::digamma");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::digamma::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & polygamma_out_out(c10::DispatchKeySet ks, int64_t n, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::polygamma");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("polygamma_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::polygamma_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), n, self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor polygamma(c10::DispatchKeySet ks, int64_t n, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::polygamma");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::polygamma::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), n, self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & polygamma_(c10::DispatchKeySet ks, at::Tensor & self, int64_t n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::polygamma");
    } else {
      op_name = c10::Symbol::fromQualString("aten::polygamma_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("polygamma_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::polygamma_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & histc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::histc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "bins", bins);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("histc_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::histc_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, bins, min, max, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor histc(c10::DispatchKeySet ks, const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::histc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "bins", bins);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::histc::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, bins, min, max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> _histogramdd_bin_edges(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_histogramdd_bin_edges");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "bins", bins);
    jit::tracer::addInputs(node, "range", range);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "density", density);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_histogramdd_bin_edges::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, bins, range, weight, density);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _histogramdd_from_bin_tensors(c10::DispatchKeySet ks, const at::Tensor & self, at::TensorList bins, const ::std::optional<at::Tensor> & weight, bool density) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_histogramdd_from_bin_tensors");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "bins", bins);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "density", density);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_histogramdd_from_bin_tensors::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, bins, weight, density);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & nextafter_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nextafter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nextafter_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::nextafter_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor nextafter(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nextafter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::nextafter::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & nextafter_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::nextafter");
    } else {
      op_name = c10::Symbol::fromQualString("aten::nextafter_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nextafter_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::nextafter_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor maximum(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::maximum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::maximum::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & maximum_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::maximum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("maximum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::maximum_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor minimum(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::minimum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::minimum::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & minimum_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::minimum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("minimum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::minimum_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor quantile(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantile");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "q", q);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "interpolation", interpolation);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::quantile::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, q, dim, keepdim, interpolation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & quantile_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantile");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "q", q);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "interpolation", interpolation);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("quantile_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::quantile_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, q, dim, keepdim, interpolation, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor quantile_scalar(c10::DispatchKeySet ks, const at::Tensor & self, double q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantile");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "q", q);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "interpolation", interpolation);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::quantile_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, q, dim, keepdim, interpolation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & quantile_out_scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, double q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantile");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "q", q);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "interpolation", interpolation);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("quantile_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::quantile_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, q, dim, keepdim, interpolation, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & msort_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::msort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("msort_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::msort_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor msort(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::msort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::msort::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor argsort(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool descending) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::argsort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::argsort::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, descending);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor argsort_stable(c10::DispatchKeySet ks, const at::Tensor & self, bool stable, int64_t dim, bool descending) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::argsort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "stable", stable);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::argsort_stable::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, stable, dim, descending);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & argsort_out_stable_out(c10::DispatchKeySet ks, const at::Tensor & self, bool stable, int64_t dim, bool descending, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::argsort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "stable", stable);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("argsort_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::argsort_stable_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, stable, dim, descending, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor argsort_dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, bool descending) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::argsort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::argsort_dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, descending);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor &,at::Tensor &> topk_out_values(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt k, int64_t dim, bool largest, bool sorted, at::Tensor & values, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::topk");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "largest", largest);
    jit::tracer::addInputs(node, "sorted", sorted);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "values", values);
      jit::tracer::addInputs(node, "indices", indices);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("topk_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::topk_values::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, k, dim, largest, sorted, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor,at::Tensor> topk(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt k, int64_t dim, bool largest, bool sorted) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::topk");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "largest", largest);
    jit::tracer::addInputs(node, "sorted", sorted);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::topk::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, k, dim, largest, sorted);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
at::Tensor unfold_backward(c10::DispatchKeySet ks, const at::Tensor & grad_in, c10::SymIntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unfold_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_in", grad_in);
    jit::tracer::addInputs(node, "input_sizes", input_sizes);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "step", step);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::unfold_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_in, input_sizes, dim, size, step);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & normal_(c10::DispatchKeySet ks, at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::normal");
    } else {
      op_name = c10::Symbol::fromQualString("aten::normal_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::normal_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mean, std, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor normal_functional(c10::DispatchKeySet ks, const at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal_functional");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::normal_functional::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mean, std, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & normal_out_Tensor_float_out(c10::DispatchKeySet ks, const at::Tensor & mean, double std, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::normal_Tensor_float_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), mean, std, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor normal_Tensor_float(c10::DispatchKeySet ks, const at::Tensor & mean, double std, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::normal_Tensor_float::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), mean, std, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & normal_out_float_Tensor_out(c10::DispatchKeySet ks, double mean, const at::Tensor & std, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::normal_float_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), mean, std, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor normal_float_Tensor(c10::DispatchKeySet ks, double mean, const at::Tensor & std, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::normal_float_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), mean, std, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & normal_out_Tensor_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & mean, const at::Tensor & std, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::normal_Tensor_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), mean, std, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor normal_Tensor_Tensor(c10::DispatchKeySet ks, const at::Tensor & mean, const at::Tensor & std, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::normal_Tensor_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), mean, std, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor normal_float_float(c10::DispatchKeySet ks, double mean, double std, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::normal_float_float::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), mean, std, size, generator, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & normal_out_float_float_out(c10::DispatchKeySet ks, double mean, double std, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::normal_float_float_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), mean, std, size, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor alias(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::alias");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::alias::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> _foreach_sub_Scalar(c10::DispatchKeySet ks, at::TensorList self, const at::Scalar & scalar) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_sub");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scalar", scalar);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_sub_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalar);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_sub__Scalar(c10::DispatchKeySet ks, at::TensorList self, const at::Scalar & scalar) {
  at::_ops::_foreach_sub__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalar);
}
::std::vector<at::Tensor> _foreach_sub_List(c10::DispatchKeySet ks, at::TensorList self, at::TensorList other, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_sub");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_sub_List::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_sub__List(c10::DispatchKeySet ks, at::TensorList self, at::TensorList other, const at::Scalar & alpha) {
  at::_ops::_foreach_sub__List::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha);
}
::std::vector<at::Tensor> _foreach_sub_ScalarList(c10::DispatchKeySet ks, at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_sub");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scalars", scalars);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_sub_ScalarList::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalars);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_sub__ScalarList(c10::DispatchKeySet ks, at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  at::_ops::_foreach_sub__ScalarList::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalars);
}
::std::vector<at::Tensor> _foreach_maximum_Scalar(c10::DispatchKeySet ks, at::TensorList self, const at::Scalar & scalar) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_maximum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scalar", scalar);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_maximum_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalar);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_maximum__Scalar(c10::DispatchKeySet ks, at::TensorList self, const at::Scalar & scalar) {
  at::_ops::_foreach_maximum__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalar);
}
::std::vector<at::Tensor> _foreach_maximum_List(c10::DispatchKeySet ks, at::TensorList self, at::TensorList other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_maximum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_maximum_List::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_maximum__List(c10::DispatchKeySet ks, at::TensorList self, at::TensorList other) {
  at::_ops::_foreach_maximum__List::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
}
::std::vector<at::Tensor> _foreach_maximum_ScalarList(c10::DispatchKeySet ks, at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_maximum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scalars", scalars);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_maximum_ScalarList::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalars);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_maximum__ScalarList(c10::DispatchKeySet ks, at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  at::_ops::_foreach_maximum__ScalarList::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalars);
}
::std::vector<at::Tensor> _foreach_acos(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_acos");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_acos::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_acos_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_acos_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_atan(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_atan");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_atan::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_atan_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_atan_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_ceil(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_ceil");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_ceil::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_ceil_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_ceil_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_erf(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_erf");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_erf::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_erf_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_erf_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_log2(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_log2");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_log2::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_log2_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_log2_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
at::Tensor bucketize_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bucketize");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "boundaries", boundaries);
    jit::tracer::addInputs(node, "out_int32", out_int32);
    jit::tracer::addInputs(node, "right", right);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bucketize_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, boundaries, out_int32, right);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & bucketize_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bucketize");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "boundaries", boundaries);
    jit::tracer::addInputs(node, "out_int32", out_int32);
    jit::tracer::addInputs(node, "right", right);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bucketize_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bucketize_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, boundaries, out_int32, right, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor bucketize_Scalar(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & boundaries, bool out_int32, bool right) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bucketize");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "boundaries", boundaries);
    jit::tracer::addInputs(node, "out_int32", out_int32);
    jit::tracer::addInputs(node, "right", right);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bucketize_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, boundaries, out_int32, right);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & mse_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mse_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mse_loss_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mse_loss_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor mse_loss(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mse_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::mse_loss::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor l1_loss(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::l1_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::l1_loss::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor nll_loss_nd(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nll_loss_nd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::nll_loss_nd::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, weight, reduction, ignore_index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & nll_loss2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nll_loss2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::nll_loss2d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, weight, reduction, ignore_index, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor nll_loss2d(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nll_loss2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::nll_loss2d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, weight, reduction, ignore_index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor &,at::Tensor &> nll_loss2d_forward_out_output(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, at::Tensor & output, at::Tensor & total_weight) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nll_loss2d_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
      jit::tracer::addInputs(node, "total_weight", total_weight);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss2d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::nll_loss2d_forward_output::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, weight, reduction, ignore_index, output, total_weight);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, total_weight);
  }
  return std::forward_as_tuple(output, total_weight);
}
::std::tuple<at::Tensor,at::Tensor> nll_loss2d_forward(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nll_loss2d_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [output, total_weight] =at::_ops::nll_loss2d_forward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, weight, reduction, ignore_index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, total_weight);
  }
  return std::make_tuple(std::move(output), std::move(total_weight));
}
at::Tensor & nll_loss2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nll_loss2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    jit::tracer::addInputs(node, "total_weight", total_weight);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::nll_loss2d_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor nll_loss2d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, const at::Tensor & total_weight) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nll_loss2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    jit::tracer::addInputs(node, "total_weight", total_weight);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::nll_loss2d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, target, weight, reduction, ignore_index, total_weight);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & soft_margin_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::soft_margin_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("soft_margin_loss_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::soft_margin_loss_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor soft_margin_loss(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::soft_margin_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::soft_margin_loss::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & glu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::glu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("glu_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::glu_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor glu(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::glu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::glu::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor glu_backward_jvp(c10::DispatchKeySet ks, const at::Tensor & grad_x, const at::Tensor & grad_glu, const at::Tensor & x, const at::Tensor & dgrad_glu, const at::Tensor & dx, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::glu_backward_jvp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_x", grad_x);
    jit::tracer::addInputs(node, "grad_glu", grad_glu);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "dgrad_glu", dgrad_glu);
    jit::tracer::addInputs(node, "dx", dx);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::glu_backward_jvp::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_x, grad_glu, x, dgrad_glu, dx, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & hardtanh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hardtanh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hardtanh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hardtanh_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, min_val, max_val, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor hardtanh(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hardtanh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hardtanh::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, min_val, max_val);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & hardtanh_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::hardtanh");
    } else {
      op_name = c10::Symbol::fromQualString("aten::hardtanh_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hardtanh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hardtanh_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, min_val, max_val);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor hardswish_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hardswish_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hardswish_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & leaky_relu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & negative_slope, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::leaky_relu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("leaky_relu_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::leaky_relu_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, negative_slope, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor leaky_relu(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & negative_slope) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::leaky_relu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::leaky_relu::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, negative_slope);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & leaky_relu_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & negative_slope) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::leaky_relu");
    } else {
      op_name = c10::Symbol::fromQualString("aten::leaky_relu_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("leaky_relu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::leaky_relu_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, negative_slope);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
::std::tuple<at::Tensor &,at::Tensor &> log_sigmoid_forward_out_output(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & output, at::Tensor & buffer) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::log_sigmoid_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
      jit::tracer::addInputs(node, "buffer", buffer);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_sigmoid_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::log_sigmoid_forward_output::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output, buffer);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, buffer);
  }
  return std::forward_as_tuple(output, buffer);
}
::std::tuple<at::Tensor,at::Tensor> log_sigmoid_forward(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::log_sigmoid_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [output, buffer] =at::_ops::log_sigmoid_forward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, buffer);
  }
  return std::make_tuple(std::move(output), std::move(buffer));
}
at::Tensor & log_sigmoid_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::log_sigmoid_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "buffer", buffer);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_sigmoid_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::log_sigmoid_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, buffer, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor log_sigmoid_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::log_sigmoid_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "buffer", buffer);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::log_sigmoid_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, buffer);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & softshrink_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::softshrink");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("softshrink_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::softshrink_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, lambd, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor softshrink(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & lambd) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::softshrink");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::softshrink::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, lambd);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & adaptive_avg_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::adaptive_avg_pool3d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_avg_pool3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::adaptive_avg_pool3d_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor _adaptive_avg_pool3d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_adaptive_avg_pool3d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_adaptive_avg_pool3d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & adaptive_max_pool2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::adaptive_max_pool2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_max_pool2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::adaptive_max_pool2d_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, indices, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor adaptive_max_pool2d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::adaptive_max_pool2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::adaptive_max_pool2d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & adaptive_max_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::adaptive_max_pool3d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_max_pool3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::adaptive_max_pool3d_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, indices, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor adaptive_max_pool3d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::adaptive_max_pool3d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::adaptive_max_pool3d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor &,at::Tensor &> fractional_max_pool3d_out_output(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples, at::Tensor & output, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fractional_max_pool3d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "random_samples", random_samples);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "output", output);
      jit::tracer::addInputs(node, "indices", indices);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fractional_max_pool3d_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fractional_max_pool3d_output::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, output_size, random_samples, output, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(output, indices);
}
::std::tuple<at::Tensor,at::Tensor> fractional_max_pool3d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fractional_max_pool3d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "random_samples", random_samples);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::fractional_max_pool3d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, output_size, random_samples);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor & reflection_pad3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::reflection_pad3d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reflection_pad3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::reflection_pad3d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor reflection_pad3d(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::reflection_pad3d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::reflection_pad3d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & replication_pad1d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::replication_pad1d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("replication_pad1d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::replication_pad1d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor replication_pad1d(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::replication_pad1d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::replication_pad1d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & replication_pad2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::replication_pad2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("replication_pad2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::replication_pad2d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor replication_pad2d(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::replication_pad2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::replication_pad2d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _pad_circular(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef pad) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_pad_circular");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "pad", pad);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_pad_circular::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, pad);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor pad(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef pad, c10::string_view mode, ::std::optional<double> value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::pad");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "pad", pad);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::pad::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, pad, mode, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor upsample_nearest1d_vec(c10::DispatchKeySet ks, const at::Tensor & input, at::OptionalSymIntArrayRef output_size, ::std::optional<at::ArrayRef<double>> scale_factors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::upsample_nearest1d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scale_factors", scale_factors);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::upsample_nearest1d_vec::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, output_size, scale_factors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _upsample_nearest_exact1d_vec(c10::DispatchKeySet ks, const at::Tensor & input, at::OptionalSymIntArrayRef output_size, ::std::optional<at::ArrayRef<double>> scale_factors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_nearest_exact1d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scale_factors", scale_factors);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_upsample_nearest_exact1d_vec::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, output_size, scale_factors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & upsample_nearest1d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::upsample_nearest1d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scales", scales);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest1d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::upsample_nearest1d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, scales, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _upsample_nearest_exact1d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_nearest_exact1d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scales", scales);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_upsample_nearest_exact1d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_upsample_nearest_exact1d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, scales, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor upsample_nearest1d(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::upsample_nearest1d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scales", scales);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::upsample_nearest1d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, scales);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _upsample_nearest_exact1d(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_nearest_exact1d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scales", scales);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_upsample_nearest_exact1d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, scales);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _conv_depthwise2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_conv_depthwise2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_conv_depthwise2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_conv_depthwise2d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, kernel_size, bias, stride, padding, dilation, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _conv_depthwise2d(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_conv_depthwise2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_conv_depthwise2d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, kernel_size, bias, stride, padding, dilation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & slow_conv3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::slow_conv3d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("slow_conv3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::slow_conv3d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, kernel_size, bias, stride, padding, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor slow_conv3d(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::slow_conv3d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::slow_conv3d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, kernel_size, bias, stride, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _remove_batch_dim(c10::DispatchKeySet ks, const at::Tensor & self, int64_t level, c10::SymInt batch_size, int64_t out_dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_remove_batch_dim");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "level", level);
    jit::tracer::addInputs(node, "batch_size", batch_size);
    jit::tracer::addInputs(node, "out_dim", out_dim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_remove_batch_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, level, batch_size, out_dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_log_ndtr(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_log_ndtr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_log_ndtr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_log_ndtr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_log_ndtr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_log_ndtr_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_log_ndtr_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_erf(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_erf");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_erf::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_erf_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_erf");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_erf_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_erf_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_xlogy(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_xlogy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_xlogy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_xlogy_self_scalar(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_xlogy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_xlogy_self_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_xlogy_other_scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_xlogy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_xlogy_other_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_xlogy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_xlogy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_xlogy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_xlogy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_xlogy_out_self_scalar_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_xlogy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_xlogy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_xlogy_self_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_xlogy_out_other_scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_xlogy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_xlogy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_xlogy_other_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_expit(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_expit");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_expit::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_expit_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_expit");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_expit_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_expit_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_sinc(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_sinc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_sinc::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_sinc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_sinc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_sinc_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_sinc_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_softmax(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_softmax");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_softmax::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor fft_fft(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_fft");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_fft::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n, dim, norm);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fft_fft_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_fft");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fft_fft_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fft_fft_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n, dim, norm, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fft_rfft(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_rfft");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_rfft::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n, dim, norm);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fft_rfft_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_rfft");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fft_rfft_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fft_rfft_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n, dim, norm, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fft_hfft2(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_hfft2");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "s", s);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_hfft2::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, s, dim, norm);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fft_hfft2_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_hfft2");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "s", s);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fft_hfft2_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fft_hfft2_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, s, dim, norm, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fft_ifftn(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_ifftn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "s", s);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_ifftn::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, s, dim, norm);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fft_ifftn_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_ifftn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "s", s);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fft_ifftn_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fft_ifftn_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, s, dim, norm, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fft_ihfftn(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_ihfftn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "s", s);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_ihfftn::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, s, dim, norm);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fft_ihfftn_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_ihfftn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "s", s);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fft_ihfftn_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fft_ihfftn_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, s, dim, norm, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fft_fftfreq(c10::DispatchKeySet ks, int64_t n, double d, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_fftfreq");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "d", d);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_fftfreq::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), n, d, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fft_fftfreq_out_out(c10::DispatchKeySet ks, int64_t n, double d, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_fftfreq");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "d", d);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fft_fftfreq_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fft_fftfreq_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), n, d, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fft_rfftfreq(c10::DispatchKeySet ks, int64_t n, double d, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_rfftfreq");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "d", d);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_rfftfreq::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), n, d, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fft_rfftfreq_out_out(c10::DispatchKeySet ks, int64_t n, double d, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_rfftfreq");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "d", d);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fft_rfftfreq_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fft_rfftfreq_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), n, d, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor> linalg_cholesky_ex(c10::DispatchKeySet ks, const at::Tensor & self, bool upper, bool check_errors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_cholesky_ex");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "check_errors", check_errors);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [L, info] =at::_ops::linalg_cholesky_ex::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, upper, check_errors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, L);
    jit::tracer::addOutput(node, info);
  }
  return std::make_tuple(std::move(L), std::move(info));
}
::std::tuple<at::Tensor &,at::Tensor &> linalg_cholesky_ex_out_L(c10::DispatchKeySet ks, const at::Tensor & self, bool upper, bool check_errors, at::Tensor & L, at::Tensor & info) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_cholesky_ex");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "check_errors", check_errors);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "L", L);
      jit::tracer::addInputs(node, "info", info);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_cholesky_ex_out", L);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_cholesky_ex_L::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, upper, check_errors, L, info);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, L);
    jit::tracer::addOutput(node, info);
  }
  return std::forward_as_tuple(L, info);
}
at::Tensor linalg_cross(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_cross");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_cross::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linalg_cross_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, int64_t dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_cross");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_cross_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_cross_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> linalg_lu_factor_ex(c10::DispatchKeySet ks, const at::Tensor & A, bool pivot, bool check_errors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_lu_factor_ex");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "pivot", pivot);
    jit::tracer::addInputs(node, "check_errors", check_errors);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [LU, pivots, info] =at::_ops::linalg_lu_factor_ex::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, pivot, check_errors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, LU);
    jit::tracer::addOutput(node, pivots);
    jit::tracer::addOutput(node, info);
  }
  return std::make_tuple(std::move(LU), std::move(pivots), std::move(info));
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> linalg_lu_factor_ex_out_out(c10::DispatchKeySet ks, const at::Tensor & A, bool pivot, bool check_errors, at::Tensor & LU, at::Tensor & pivots, at::Tensor & info) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_lu_factor_ex");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "pivot", pivot);
    jit::tracer::addInputs(node, "check_errors", check_errors);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "LU", LU);
      jit::tracer::addInputs(node, "pivots", pivots);
      jit::tracer::addInputs(node, "info", info);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_lu_factor_ex_out", LU);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_lu_factor_ex_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, pivot, check_errors, LU, pivots, info);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, LU);
    jit::tracer::addOutput(node, pivots);
    jit::tracer::addOutput(node, info);
  }
  return std::forward_as_tuple(LU, pivots, info);
}
at::Tensor det(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::det");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::det::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor inverse(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::inverse");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::inverse::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & inverse_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::inverse");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("inverse_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::inverse_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor linalg_cond(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & p) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_cond");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_cond::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linalg_cond_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_cond");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_cond_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_cond_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor linalg_cond_p_str(c10::DispatchKeySet ks, const at::Tensor & self, c10::string_view p) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_cond");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_cond_p_str::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linalg_cond_out_p_str_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::string_view p, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_cond");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_cond_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_cond_p_str_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor linalg_pinv_atol_rtol_tensor(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Tensor> & atol, const ::std::optional<at::Tensor> & rtol, bool hermitian) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_pinv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "atol", atol);
    jit::tracer::addInputs(node, "rtol", rtol);
    jit::tracer::addInputs(node, "hermitian", hermitian);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_pinv_atol_rtol_tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, atol, rtol, hermitian);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linalg_pinv_out_atol_rtol_tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Tensor> & atol, const ::std::optional<at::Tensor> & rtol, bool hermitian, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_pinv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "atol", atol);
    jit::tracer::addInputs(node, "rtol", rtol);
    jit::tracer::addInputs(node, "hermitian", hermitian);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_pinv_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_pinv_atol_rtol_tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, atol, rtol, hermitian, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor linalg_pinv_atol_rtol_float(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<double> atol, ::std::optional<double> rtol, bool hermitian) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_pinv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "atol", atol);
    jit::tracer::addInputs(node, "rtol", rtol);
    jit::tracer::addInputs(node, "hermitian", hermitian);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_pinv_atol_rtol_float::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, atol, rtol, hermitian);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linalg_pinv_out_atol_rtol_float_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<double> atol, ::std::optional<double> rtol, bool hermitian, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_pinv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "atol", atol);
    jit::tracer::addInputs(node, "rtol", rtol);
    jit::tracer::addInputs(node, "hermitian", hermitian);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_pinv_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_pinv_atol_rtol_float_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, atol, rtol, hermitian, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor linalg_pinv(c10::DispatchKeySet ks, const at::Tensor & self, double rcond, bool hermitian) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_pinv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "rcond", rcond);
    jit::tracer::addInputs(node, "hermitian", hermitian);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_pinv::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, rcond, hermitian);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor linalg_pinv_rcond_tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & rcond, bool hermitian) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_pinv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "rcond", rcond);
    jit::tracer::addInputs(node, "hermitian", hermitian);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_pinv_rcond_tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, rcond, hermitian);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linalg_pinv_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double rcond, bool hermitian, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_pinv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "rcond", rcond);
    jit::tracer::addInputs(node, "hermitian", hermitian);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_pinv_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_pinv_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, rcond, hermitian, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & linalg_pinv_out_out_rcond_tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & rcond, bool hermitian, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_pinv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "rcond", rcond);
    jit::tracer::addInputs(node, "hermitian", hermitian);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_pinv_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_pinv_out_rcond_tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, rcond, hermitian, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor> linalg_solve_ex(c10::DispatchKeySet ks, const at::Tensor & A, const at::Tensor & B, bool left, bool check_errors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_solve_ex");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "B", B);
    jit::tracer::addInputs(node, "left", left);
    jit::tracer::addInputs(node, "check_errors", check_errors);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result, info] =at::_ops::linalg_solve_ex::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, B, left, check_errors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
    jit::tracer::addOutput(node, info);
  }
  return std::make_tuple(std::move(result), std::move(info));
}
::std::tuple<at::Tensor &,at::Tensor &> linalg_solve_ex_out_out(c10::DispatchKeySet ks, const at::Tensor & A, const at::Tensor & B, bool left, bool check_errors, at::Tensor & result, at::Tensor & info) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_solve_ex");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "B", B);
    jit::tracer::addInputs(node, "left", left);
    jit::tracer::addInputs(node, "check_errors", check_errors);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "result", result);
      jit::tracer::addInputs(node, "info", info);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_solve_ex_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_solve_ex_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, B, left, check_errors, result, info);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
    jit::tracer::addOutput(node, info);
  }
  return std::forward_as_tuple(result, info);
}
at::Tensor _spsolve(c10::DispatchKeySet ks, const at::Tensor & A, const at::Tensor & B, bool left) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_spsolve");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "B", B);
    jit::tracer::addInputs(node, "left", left);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_spsolve::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, B, left);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor linalg_tensorsolve(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::OptionalIntArrayRef dims) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_tensorsolve");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dims", dims);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_tensorsolve::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, dims);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linalg_tensorsolve_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::OptionalIntArrayRef dims, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_tensorsolve");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dims", dims);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_tensorsolve_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_tensorsolve_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, dims, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor linalg_multi_dot(c10::DispatchKeySet ks, at::TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_multi_dot");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_multi_dot::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linalg_multi_dot_out_out(c10::DispatchKeySet ks, at::TensorList tensors, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_multi_dot");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_multi_dot_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_multi_dot_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _test_parallel_materialize(c10::DispatchKeySet ks, const at::Tensor & self, int64_t num_parallel, bool skip_first) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_test_parallel_materialize");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "num_parallel", num_parallel);
    jit::tracer::addInputs(node, "skip_first", skip_first);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_test_parallel_materialize::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, num_parallel, skip_first);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _test_string_default(c10::DispatchKeySet ks, const at::Tensor & dummy, c10::string_view a, c10::string_view b) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_test_string_default");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "dummy", dummy);
    jit::tracer::addInputs(node, "a", a);
    jit::tracer::addInputs(node, "b", b);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_test_string_default::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), dummy, a, b);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor flatten_dense_tensors(c10::DispatchKeySet ks, at::TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::flatten_dense_tensors");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::flatten_dense_tensors::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _conj_copy(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_conj_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_conj_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor detach_copy(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::detach_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::detach_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor row_indices_copy(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::row_indices_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::row_indices_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _transformer_encoder_layer_fwd(c10::DispatchKeySet ks, const at::Tensor & src, int64_t embed_dim, int64_t num_heads, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, bool use_gelu, bool norm_first, double eps, const at::Tensor & norm_weight_1, const at::Tensor & norm_bias_1, const at::Tensor & norm_weight_2, const at::Tensor & norm_bias_2, const at::Tensor & ffn_weight_1, const at::Tensor & ffn_bias_1, const at::Tensor & ffn_weight_2, const at::Tensor & ffn_bias_2, const ::std::optional<at::Tensor> & mask, ::std::optional<int64_t> mask_type) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_transformer_encoder_layer_fwd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "embed_dim", embed_dim);
    jit::tracer::addInputs(node, "num_heads", num_heads);
    jit::tracer::addInputs(node, "qkv_weight", qkv_weight);
    jit::tracer::addInputs(node, "qkv_bias", qkv_bias);
    jit::tracer::addInputs(node, "proj_weight", proj_weight);
    jit::tracer::addInputs(node, "proj_bias", proj_bias);
    jit::tracer::addInputs(node, "use_gelu", use_gelu);
    jit::tracer::addInputs(node, "norm_first", norm_first);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "norm_weight_1", norm_weight_1);
    jit::tracer::addInputs(node, "norm_bias_1", norm_bias_1);
    jit::tracer::addInputs(node, "norm_weight_2", norm_weight_2);
    jit::tracer::addInputs(node, "norm_bias_2", norm_bias_2);
    jit::tracer::addInputs(node, "ffn_weight_1", ffn_weight_1);
    jit::tracer::addInputs(node, "ffn_bias_1", ffn_bias_1);
    jit::tracer::addInputs(node, "ffn_weight_2", ffn_weight_2);
    jit::tracer::addInputs(node, "ffn_bias_2", ffn_bias_2);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "mask_type", mask_type);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_transformer_encoder_layer_fwd::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), src, embed_dim, num_heads, qkv_weight, qkv_bias, proj_weight, proj_bias, use_gelu, norm_first, eps, norm_weight_1, norm_bias_1, norm_weight_2, norm_bias_2, ffn_weight_1, ffn_bias_1, ffn_weight_2, ffn_bias_2, mask, mask_type);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> _native_multi_head_attention(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t embed_dim, int64_t num_head, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, const ::std::optional<at::Tensor> & mask, bool need_weights, bool average_attn_weights, ::std::optional<int64_t> mask_type) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_native_multi_head_attention");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "embed_dim", embed_dim);
    jit::tracer::addInputs(node, "num_head", num_head);
    jit::tracer::addInputs(node, "qkv_weight", qkv_weight);
    jit::tracer::addInputs(node, "qkv_bias", qkv_bias);
    jit::tracer::addInputs(node, "proj_weight", proj_weight);
    jit::tracer::addInputs(node, "proj_bias", proj_bias);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "need_weights", need_weights);
    jit::tracer::addInputs(node, "average_attn_weights", average_attn_weights);
    jit::tracer::addInputs(node, "mask_type", mask_type);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_native_multi_head_attention::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, embed_dim, num_head, qkv_weight, qkv_bias, proj_weight, proj_bias, mask, need_weights, average_attn_weights, mask_type);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
int64_t _fused_sdp_choice(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & attn_mask, double dropout_p, bool is_causal, ::std::optional<double> scale, bool enable_gqa) {
  auto result =at::_ops::_fused_sdp_choice::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,c10::SymInt,c10::SymInt,at::Tensor,at::Tensor,at::Tensor> _scaled_dot_product_flash_attention(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, double dropout_p, bool is_causal, bool return_debug_mask, ::std::optional<double> scale) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_scaled_dot_product_flash_attention");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);
    jit::tracer::addInputs(node, "is_causal", is_causal);
    jit::tracer::addInputs(node, "return_debug_mask", return_debug_mask);
    jit::tracer::addInputs(node, "scale", scale);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, rng_state, unused, debug_attn_mask] =at::_ops::_scaled_dot_product_flash_attention::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, dropout_p, is_causal, return_debug_mask, scale);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, logsumexp);
    jit::tracer::addOutput(node, cum_seq_q);
    jit::tracer::addOutput(node, cum_seq_k);
    jit::tracer::addOutput(node, max_q);
    jit::tracer::addOutput(node, max_k);
    jit::tracer::addOutput(node, rng_state);
    jit::tracer::addOutput(node, unused);
    jit::tracer::addOutput(node, debug_attn_mask);
  }
  return std::make_tuple(std::move(output), std::move(logsumexp), std::move(cum_seq_q), std::move(cum_seq_k), std::move(max_q), std::move(max_k), std::move(rng_state), std::move(unused), std::move(debug_attn_mask));
}
::std::tuple<at::Tensor,at::Tensor> _scaled_dot_product_flash_attention_for_cpu(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, double dropout_p, bool is_causal, const ::std::optional<at::Tensor> & attn_mask, ::std::optional<double> scale) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_scaled_dot_product_flash_attention_for_cpu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);
    jit::tracer::addInputs(node, "is_causal", is_causal);
    jit::tracer::addInputs(node, "attn_mask", attn_mask);
    jit::tracer::addInputs(node, "scale", scale);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [output, logsumexp] =at::_ops::_scaled_dot_product_flash_attention_for_cpu::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, dropout_p, is_causal, attn_mask, scale);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, logsumexp);
  }
  return std::make_tuple(std::move(output), std::move(logsumexp));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _scaled_dot_product_efficient_attention_backward(c10::DispatchKeySet ks, const at::Tensor & grad_out_, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & attn_bias, const at::Tensor & out, const at::Tensor & logsumexp, const at::Tensor & philox_seed, const at::Tensor & philox_offset, double dropout_p, ::std::array<bool,4> grad_input_mask, bool is_causal, ::std::optional<double> scale) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_scaled_dot_product_efficient_attention_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out_", grad_out_);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "attn_bias", attn_bias);
    jit::tracer::addInputs(node, "out", out);
    jit::tracer::addInputs(node, "logsumexp", logsumexp);
    jit::tracer::addInputs(node, "philox_seed", philox_seed);
    jit::tracer::addInputs(node, "philox_offset", philox_offset);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);
    jit::tracer::addInputs(node, "grad_input_mask", grad_input_mask);
    jit::tracer::addInputs(node, "is_causal", is_causal);
    jit::tracer::addInputs(node, "scale", scale);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3] =at::_ops::_scaled_dot_product_efficient_attention_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out_, query, key, value, attn_bias, out, logsumexp, philox_seed, philox_offset, dropout_p, grad_input_mask, is_causal, scale);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _flash_attention_backward(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & out, const at::Tensor & logsumexp, const at::Tensor & cum_seq_q, const at::Tensor & cum_seq_k, c10::SymInt max_q, c10::SymInt max_k, double dropout_p, bool is_causal, const at::Tensor & rng_state, const at::Tensor & unused, ::std::optional<double> scale, ::std::optional<c10::SymInt> window_size_left, ::std::optional<c10::SymInt> window_size_right) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_flash_attention_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "out", out);
    jit::tracer::addInputs(node, "logsumexp", logsumexp);
    jit::tracer::addInputs(node, "cum_seq_q", cum_seq_q);
    jit::tracer::addInputs(node, "cum_seq_k", cum_seq_k);
    jit::tracer::addInputs(node, "max_q", max_q);
    jit::tracer::addInputs(node, "max_k", max_k);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);
    jit::tracer::addInputs(node, "is_causal", is_causal);
    jit::tracer::addInputs(node, "rng_state", rng_state);
    jit::tracer::addInputs(node, "unused", unused);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "window_size_left", window_size_left);
    jit::tracer::addInputs(node, "window_size_right", window_size_right);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::_flash_attention_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, query, key, value, out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, is_causal, rng_state, unused, scale, window_size_left, window_size_right);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _efficient_attention_backward(c10::DispatchKeySet ks, const at::Tensor & grad_out_, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & bias, const at::Tensor & out, const ::std::optional<at::Tensor> & cu_seqlens_q, const ::std::optional<at::Tensor> & cu_seqlens_k, c10::SymInt max_seqlen_q, c10::SymInt max_seqlen_k, const at::Tensor & logsumexp, double dropout_p, const at::Tensor & philox_seed, const at::Tensor & philox_offset, int64_t custom_mask_type, bool bias_requires_grad, ::std::optional<double> scale, ::std::optional<int64_t> num_splits_key, ::std::optional<int64_t> window_size, bool shared_storage_dqdkdv) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_efficient_attention_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out_", grad_out_);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "out", out);
    jit::tracer::addInputs(node, "cu_seqlens_q", cu_seqlens_q);
    jit::tracer::addInputs(node, "cu_seqlens_k", cu_seqlens_k);
    jit::tracer::addInputs(node, "max_seqlen_q", max_seqlen_q);
    jit::tracer::addInputs(node, "max_seqlen_k", max_seqlen_k);
    jit::tracer::addInputs(node, "logsumexp", logsumexp);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);
    jit::tracer::addInputs(node, "philox_seed", philox_seed);
    jit::tracer::addInputs(node, "philox_offset", philox_offset);
    jit::tracer::addInputs(node, "custom_mask_type", custom_mask_type);
    jit::tracer::addInputs(node, "bias_requires_grad", bias_requires_grad);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "num_splits_key", num_splits_key);
    jit::tracer::addInputs(node, "window_size", window_size);
    jit::tracer::addInputs(node, "shared_storage_dqdkdv", shared_storage_dqdkdv);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3] =at::_ops::_efficient_attention_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out_, query, key, value, bias, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, logsumexp, dropout_p, philox_seed, philox_offset, custom_mask_type, bias_requires_grad, scale, num_splits_key, window_size, shared_storage_dqdkdv);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
at::Tensor & _fill_mem_eff_dropout_mask_(c10::DispatchKeySet ks, at::Tensor & self, double dropout_p, int64_t seed, int64_t offset) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    
    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::_fill_mem_eff_dropout_mask");
    } else {
      op_name = c10::Symbol::fromQualString("aten::_fill_mem_eff_dropout_mask_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);
    jit::tracer::addInputs(node, "seed", seed);
    jit::tracer::addInputs(node, "offset", offset);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_fill_mem_eff_dropout_mask_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_fill_mem_eff_dropout_mask_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dropout_p, seed, offset);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor _triton_multi_head_attention(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t embed_dim, int64_t num_head, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, const ::std::optional<at::Tensor> & mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_triton_multi_head_attention");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "embed_dim", embed_dim);
    jit::tracer::addInputs(node, "num_head", num_head);
    jit::tracer::addInputs(node, "qkv_weight", qkv_weight);
    jit::tracer::addInputs(node, "qkv_bias", qkv_bias);
    jit::tracer::addInputs(node, "proj_weight", proj_weight);
    jit::tracer::addInputs(node, "proj_bias", proj_bias);
    jit::tracer::addInputs(node, "mask", mask);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_triton_multi_head_attention::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, embed_dim, num_head, qkv_weight, qkv_bias, proj_weight, proj_bias, mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_airy_ai(c10::DispatchKeySet ks, const at::Tensor & x) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_airy_ai");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_airy_ai::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_airy_ai_out_out(c10::DispatchKeySet ks, const at::Tensor & x, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_airy_ai");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_airy_ai_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_airy_ai_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_chebyshev_polynomial_w(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_chebyshev_polynomial_w");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_chebyshev_polynomial_w::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_chebyshev_polynomial_w_x_scalar(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_chebyshev_polynomial_w");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_chebyshev_polynomial_w_x_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_chebyshev_polynomial_w_n_scalar(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_chebyshev_polynomial_w");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_chebyshev_polynomial_w_n_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_chebyshev_polynomial_w_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_chebyshev_polynomial_w");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_chebyshev_polynomial_w_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_chebyshev_polynomial_w_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_chebyshev_polynomial_w_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_chebyshev_polynomial_w");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_chebyshev_polynomial_w_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_chebyshev_polynomial_w_x_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_chebyshev_polynomial_w_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_chebyshev_polynomial_w");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_chebyshev_polynomial_w_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_chebyshev_polynomial_w_n_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_hermite_polynomial_h(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_hermite_polynomial_h");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_hermite_polynomial_h::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_hermite_polynomial_h_x_scalar(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_hermite_polynomial_h");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_hermite_polynomial_h_x_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_hermite_polynomial_h_n_scalar(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_hermite_polynomial_h");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_hermite_polynomial_h_n_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_hermite_polynomial_h_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_hermite_polynomial_h");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_hermite_polynomial_h_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_hermite_polynomial_h_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_hermite_polynomial_h_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_hermite_polynomial_h");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_hermite_polynomial_h_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_hermite_polynomial_h_x_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_hermite_polynomial_h_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_hermite_polynomial_h");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_hermite_polynomial_h_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_hermite_polynomial_h_n_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_modified_bessel_i0(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_modified_bessel_i0");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_modified_bessel_i0::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_modified_bessel_i0_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_modified_bessel_i0");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_modified_bessel_i0_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_modified_bessel_i0_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_modified_bessel_k0(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_modified_bessel_k0");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_modified_bessel_k0::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_modified_bessel_k0_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_modified_bessel_k0");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_modified_bessel_k0_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_modified_bessel_k0_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_modified_bessel_k1(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_modified_bessel_k1");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_modified_bessel_k1::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_modified_bessel_k1_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_modified_bessel_k1");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_modified_bessel_k1_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_modified_bessel_k1_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_shifted_chebyshev_polynomial_t(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_t");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_shifted_chebyshev_polynomial_t::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_shifted_chebyshev_polynomial_t_x_scalar(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_t");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_shifted_chebyshev_polynomial_t_x_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_shifted_chebyshev_polynomial_t_n_scalar(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_t");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_shifted_chebyshev_polynomial_t_n_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_shifted_chebyshev_polynomial_t_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_t");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_shifted_chebyshev_polynomial_t_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_shifted_chebyshev_polynomial_t_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_t_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_t");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_shifted_chebyshev_polynomial_t_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_shifted_chebyshev_polynomial_t_x_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_t_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_t");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_shifted_chebyshev_polynomial_t_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_shifted_chebyshev_polynomial_t_n_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void _fused_adamw_(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps, double lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) {
  at::_ops::_fused_adamw_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
}
void _fused_adamw__tensor_lr(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps, const at::Tensor & lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) {
  at::_ops::_fused_adamw__tensor_lr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
}
at::Tensor & _cudnn_rnn_flatten_weight_out_out(c10::DispatchKeySet ks, at::TensorList weight_arr, int64_t weight_stride0, c10::SymInt input_size, int64_t mode, c10::SymInt hidden_size, c10::SymInt proj_size, int64_t num_layers, bool batch_first, bool bidirectional, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cudnn_rnn_flatten_weight");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight_arr", weight_arr);
    jit::tracer::addInputs(node, "weight_stride0", weight_stride0);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "hidden_size", hidden_size);
    jit::tracer::addInputs(node, "proj_size", proj_size);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_cudnn_rnn_flatten_weight_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_cudnn_rnn_flatten_weight_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), weight_arr, weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & quantized_batch_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & var, double eps, double output_scale, int64_t output_zero_point, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantized_batch_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "var", var);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "output_scale", output_scale);
    jit::tracer::addInputs(node, "output_zero_point", output_zero_point);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("quantized_batch_norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::quantized_batch_norm_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, mean, var, eps, output_scale, output_zero_point, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & conv_tbc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const at::Tensor & bias, int64_t pad, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::conv_tbc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "pad", pad);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("conv_tbc_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::conv_tbc_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, bias, pad, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & cudnn_affine_grid_generator_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cudnn_affine_grid_generator_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "N", N);
    jit::tracer::addInputs(node, "C", C);
    jit::tracer::addInputs(node, "H", H);
    jit::tracer::addInputs(node, "W", W);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cudnn_affine_grid_generator_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cudnn_affine_grid_generator_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, N, C, H, W, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & cudnn_grid_sampler_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & grid, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cudnn_grid_sampler");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grid", grid);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cudnn_grid_sampler_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cudnn_grid_sampler_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grid, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & div_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::div");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::div_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & div_out_Scalar_mode_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, ::std::optional<c10::string_view> rounding_mode, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::div");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "rounding_mode", rounding_mode);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::div_Scalar_mode_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, rounding_mode, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _embedding_bag_forward_only_out_out(c10::DispatchKeySet ks, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const ::std::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_embedding_bag_forward_only");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    jit::tracer::addInputs(node, "include_last_offset", include_last_offset);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
      jit::tracer::addInputs(node, "out3", out3);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_embedding_bag_forward_only_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_embedding_bag_forward_only_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx, out0, out1, out2, out3);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
    jit::tracer::addOutput(node, out3);
  }
  return std::forward_as_tuple(out0, out1, out2, out3);
}
at::Tensor & new_zeros_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::new_zeros");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("new_zeros_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::new_zeros_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _grid_sampler_2d_cpu_fallback_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_grid_sampler_2d_cpu_fallback");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_grid_sampler_2d_cpu_fallback_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_grid_sampler_2d_cpu_fallback_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, grid, interpolation_mode, padding_mode, align_corners, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & grid_sampler_3d_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::grid_sampler_3d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("grid_sampler_3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::grid_sampler_3d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, grid, interpolation_mode, padding_mode, align_corners, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & hann_window_out_out(c10::DispatchKeySet ks, int64_t window_length, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hann_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hann_window_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hann_window_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & hann_window_out_periodic_out(c10::DispatchKeySet ks, int64_t window_length, bool periodic, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hann_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hann_window_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hann_window_periodic_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & hamming_window_out_out(c10::DispatchKeySet ks, int64_t window_length, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hamming_window_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hamming_window_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & hamming_window_out_periodic_out(c10::DispatchKeySet ks, int64_t window_length, bool periodic, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hamming_window_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hamming_window_periodic_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & hamming_window_out_periodic_alpha_out(c10::DispatchKeySet ks, int64_t window_length, bool periodic, double alpha, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "alpha", alpha);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hamming_window_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hamming_window_periodic_alpha_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & hamming_window_out_periodic_alpha_beta_out(c10::DispatchKeySet ks, int64_t window_length, bool periodic, double alpha, double beta, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "beta", beta);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hamming_window_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hamming_window_periodic_alpha_beta_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, alpha, beta, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_group_norm_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & rstd, const ::std::optional<at::Tensor> & weight, c10::SymInt N, c10::SymInt C, c10::SymInt HxW, int64_t group, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::native_group_norm_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "rstd", rstd);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "N", N);
    jit::tracer::addInputs(node, "C", C);
    jit::tracer::addInputs(node, "HxW", HxW);
    jit::tracer::addInputs(node, "group", group);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("native_group_norm_backward_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::native_group_norm_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask, out0, out1, out2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
  }
  return std::forward_as_tuple(out0, out1, out2);
}
at::Tensor & isnan_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::isnan");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("isnan_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::isnan_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_layer_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & input, c10::SymIntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::native_layer_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "normalized_shape", normalized_shape);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "eps", eps);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("native_layer_norm_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::native_layer_norm_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, normalized_shape, weight, bias, eps, out0, out1, out2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
  }
  return std::forward_as_tuple(out0, out1, out2);
}
at::Tensor & mkldnn_max_pool2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_max_pool2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mkldnn_max_pool2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mkldnn_max_pool2d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, dilation, ceil_mode, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & quantized_max_pool2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantized_max_pool2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("quantized_max_pool2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::quantized_max_pool2d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, dilation, ceil_mode, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _mps_convolution_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_mps_convolution");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_mps_convolution_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_mps_convolution_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, bias, padding, stride, dilation, groups, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> mkldnn_rnn_layer_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight1, const at::Tensor & weight2, const at::Tensor & weight3, const at::Tensor & weight4, const at::Tensor & hx_, const at::Tensor & cx_tmp, const at::Tensor & output, const at::Tensor & hy_, const at::Tensor & cy_, const ::std::optional<at::Tensor> & grad_output, const ::std::optional<at::Tensor> & grad_hy, const ::std::optional<at::Tensor> & grad_cy, bool reverse, int64_t mode, int64_t hidden_size, int64_t num_layers, bool has_biases, bool train, bool bidirectional, at::IntArrayRef batch_sizes, bool batch_first, const at::Tensor & workspace, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3, at::Tensor & out4, at::Tensor & out5, at::Tensor & out6) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_rnn_layer_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight1", weight1);
    jit::tracer::addInputs(node, "weight2", weight2);
    jit::tracer::addInputs(node, "weight3", weight3);
    jit::tracer::addInputs(node, "weight4", weight4);
    jit::tracer::addInputs(node, "hx_", hx_);
    jit::tracer::addInputs(node, "cx_tmp", cx_tmp);
    jit::tracer::addInputs(node, "output", output);
    jit::tracer::addInputs(node, "hy_", hy_);
    jit::tracer::addInputs(node, "cy_", cy_);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "grad_hy", grad_hy);
    jit::tracer::addInputs(node, "grad_cy", grad_cy);
    jit::tracer::addInputs(node, "reverse", reverse);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "hidden_size", hidden_size);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "workspace", workspace);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
      jit::tracer::addInputs(node, "out3", out3);
      jit::tracer::addInputs(node, "out4", out4);
      jit::tracer::addInputs(node, "out5", out5);
      jit::tracer::addInputs(node, "out6", out6);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mkldnn_rnn_layer_backward_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mkldnn_rnn_layer_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight1, weight2, weight3, weight4, hx_, cx_tmp, output, hy_, cy_, grad_output, grad_hy, grad_cy, reverse, mode, hidden_size, num_layers, has_biases, train, bidirectional, batch_sizes, batch_first, workspace, out0, out1, out2, out3, out4, out5, out6);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
    jit::tracer::addOutput(node, out3);
    jit::tracer::addOutput(node, out4);
    jit::tracer::addOutput(node, out5);
    jit::tracer::addOutput(node, out6);
  }
  return std::forward_as_tuple(out0, out1, out2, out3, out4, out5, out6);
}
at::Tensor & miopen_depthwise_convolution_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, bool benchmark, bool deterministic, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::miopen_depthwise_convolution");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("miopen_depthwise_convolution_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::miopen_depthwise_convolution_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> batch_norm_stats_out_out(c10::DispatchKeySet ks, const at::Tensor & input, double eps, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::batch_norm_stats");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "eps", eps);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("batch_norm_stats_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::batch_norm_stats_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, eps, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
::std::tuple<at::Tensor &,at::Tensor &> batch_norm_gather_stats_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, double momentum, double eps, int64_t count, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::batch_norm_gather_stats");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "invstd", invstd);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "count", count);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("batch_norm_gather_stats_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::batch_norm_gather_stats_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, mean, invstd, running_mean, running_var, momentum, eps, count, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_batch_norm_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, const ::std::optional<at::Tensor> & save_mean, const ::std::optional<at::Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::native_batch_norm_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "save_mean", save_mean);
    jit::tracer::addInputs(node, "save_invstd", save_invstd);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("native_batch_norm_backward_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::native_batch_norm_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask, out0, out1, out2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
  }
  return std::forward_as_tuple(out0, out1, out2);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> batch_norm_backward_reduce_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const ::std::optional<at::Tensor> & weight, bool input_g, bool weight_g, bool bias_g, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::batch_norm_backward_reduce");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "invstd", invstd);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "input_g", input_g);
    jit::tracer::addInputs(node, "weight_g", weight_g);
    jit::tracer::addInputs(node, "bias_g", bias_g);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
      jit::tracer::addInputs(node, "out3", out3);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("batch_norm_backward_reduce_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::batch_norm_backward_reduce_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g, out0, out1, out2, out3);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
    jit::tracer::addOutput(node, out3);
  }
  return std::forward_as_tuple(out0, out1, out2, out3);
}
at::Tensor & _nnpack_spatial_convolution_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nnpack_spatial_convolution");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_nnpack_spatial_convolution_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_nnpack_spatial_convolution_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, padding, stride, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & ones_out_names_out(c10::DispatchKeySet ks, at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ones");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "names", names);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ones_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ones_names_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, names, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _cdist_forward_out_out(c10::DispatchKeySet ks, const at::Tensor & x1, const at::Tensor & x2, double p, ::std::optional<int64_t> compute_mode, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cdist_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "compute_mode", compute_mode);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_cdist_forward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_cdist_forward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x1, x2, p, compute_mode, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & rand_like_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rand_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rand_like_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::rand_like_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & randint_like_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt high, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_like_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randint_like_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, high, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & randint_like_out_low_dtype_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt low, c10::SymInt high, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_like_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randint_like_low_dtype_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, low, high, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & select_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef input_sizes, int64_t dim, c10::SymInt index, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::select_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input_sizes", input_sizes);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("select_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::select_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, input_sizes, dim, index, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & slice_scatter_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & src, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::slice_scatter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("slice_scatter_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::slice_scatter_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, src, dim, start, end, step, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _mkldnn_transpose_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim0, int64_t dim1, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_mkldnn_transpose");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim0", dim0);
    jit::tracer::addInputs(node, "dim1", dim1);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_mkldnn_transpose_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_mkldnn_transpose_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim0, dim1, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _nested_from_padded_and_nested_example_out_out(c10::DispatchKeySet ks, const at::Tensor & padded, const at::Tensor & nt_example, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_from_padded_and_nested_example");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "padded", padded);
    jit::tracer::addInputs(node, "nt_example", nt_example);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_nested_from_padded_and_nested_example_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_nested_from_padded_and_nested_example_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), padded, nt_example, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> unique_dim_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unique_dim");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "sorted", sorted);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);
    jit::tracer::addInputs(node, "return_counts", return_counts);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("unique_dim_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::unique_dim_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, sorted, return_inverse, return_counts, out0, out1, out2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
  }
  return std::forward_as_tuple(out0, out1, out2);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> unique_consecutive_out_out(c10::DispatchKeySet ks, const at::Tensor & self, bool return_inverse, bool return_counts, ::std::optional<int64_t> dim, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unique_consecutive");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);
    jit::tracer::addInputs(node, "return_counts", return_counts);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("unique_consecutive_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::unique_consecutive_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, return_inverse, return_counts, dim, out0, out1, out2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
  }
  return std::forward_as_tuple(out0, out1, out2);
}
at::Tensor & _dirichlet_grad_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & alpha, const at::Tensor & total, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_dirichlet_grad");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "total", total);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_dirichlet_grad_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_dirichlet_grad_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, alpha, total, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & clone_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::clone");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clone_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::clone_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
const at::Tensor & resize_as_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & the_template, ::std::optional<at::MemoryFormat> memory_format, const at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::resize_as");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "the_template", the_template);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("resize_as_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::resize_as_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, the_template, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor resize_as(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & the_template, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::resize_as");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "the_template", the_template);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::resize_as::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, the_template, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
const at::Tensor & resize_as_sparse_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & the_template, const at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::resize_as_sparse");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "the_template", the_template);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("resize_as_sparse_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::resize_as_sparse_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, the_template, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor resize_as_sparse(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & the_template) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::resize_as_sparse");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "the_template", the_template);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::resize_as_sparse::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, the_template);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _to_sparse_csc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<int64_t> dense_dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_to_sparse_csc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_to_sparse_csc_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_to_sparse_csc_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dense_dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _to_sparse_bsc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef blocksize, ::std::optional<int64_t> dense_dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_to_sparse_bsc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "blocksize", blocksize);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_to_sparse_bsc_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_to_sparse_bsc_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, blocksize, dense_dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & mkldnn_reorder_conv2d_weight_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, at::OptionalSymIntArrayRef input_size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_reorder_conv2d_weight");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "input_size", input_size);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mkldnn_reorder_conv2d_weight_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mkldnn_reorder_conv2d_weight_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding, stride, dilation, groups, input_size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & quantize_per_channel_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, at::ScalarType dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantize_per_channel");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scales", scales);
    jit::tracer::addInputs(node, "zero_points", zero_points);
    jit::tracer::addInputs(node, "axis", axis);
    jit::tracer::addInputs(node, "dtype", dtype);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("quantize_per_channel_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::quantize_per_channel_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scales, zero_points, axis, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & dequantize_out_self_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::dequantize");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("dequantize_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::dequantize_self_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void dequantize_out_tensors_out(c10::DispatchKeySet ks, at::TensorList tensors, at::TensorList out) {
  at::_ops::dequantize_tensors_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, out);
}
at::Tensor & q_per_channel_zero_points_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::q_per_channel_zero_points");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("q_per_channel_zero_points_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::q_per_channel_zero_points_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _fake_quantize_learnable_per_channel_affine_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fake_quantize_learnable_per_channel_affine");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "axis", axis);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    jit::tracer::addInputs(node, "grad_factor", grad_factor);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_fake_quantize_learnable_per_channel_affine_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_fake_quantize_learnable_per_channel_affine_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, axis, quant_min, quant_max, grad_factor, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _lstm_mps_out_out(c10::DispatchKeySet ks, const at::Tensor & input, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3, at::Tensor & out4, at::Tensor & out5) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_lstm_mps");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
      jit::tracer::addInputs(node, "out3", out3);
      jit::tracer::addInputs(node, "out4", out4);
      jit::tracer::addInputs(node, "out5", out5);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_lstm_mps_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_lstm_mps_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first, out0, out1, out2, out3, out4, out5);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
    jit::tracer::addOutput(node, out3);
    jit::tracer::addOutput(node, out4);
    jit::tracer::addOutput(node, out5);
  }
  return std::forward_as_tuple(out0, out1, out2, out3, out4, out5);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> _thnn_fused_lstm_cell_out_out(c10::DispatchKeySet ks, const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & cx, const ::std::optional<at::Tensor> & input_bias, const ::std::optional<at::Tensor> & hidden_bias, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_thnn_fused_lstm_cell");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input_gates", input_gates);
    jit::tracer::addInputs(node, "hidden_gates", hidden_gates);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "input_bias", input_bias);
    jit::tracer::addInputs(node, "hidden_bias", hidden_bias);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_fused_lstm_cell_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_thnn_fused_lstm_cell_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input_gates, hidden_gates, cx, input_bias, hidden_bias, out0, out1, out2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
  }
  return std::forward_as_tuple(out0, out1, out2);
}
at::Tensor & lift_fresh_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::lift_fresh_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lift_fresh_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::lift_fresh_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & index_fill_out_int_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::index_fill");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_fill_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::index_fill_int_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, value, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & index_fill_out_int_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::index_fill");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_fill_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::index_fill_int_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, value, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & random_out_from_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t from, ::std::optional<int64_t> to, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::random");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "from", from);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("random_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::random_from_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, from, to, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor random_from(c10::DispatchKeySet ks, const at::Tensor & self, int64_t from, ::std::optional<int64_t> to, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::random");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "from", from);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::random_from::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, from, to, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & random_out_to_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t to, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::random");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("random_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::random_to_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, to, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor random_to(c10::DispatchKeySet ks, const at::Tensor & self, int64_t to, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::random");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::random_to::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, to, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & random_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::random");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("random_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::random_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor random(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::random");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::random::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & cauchy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double median, double sigma, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cauchy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "median", median);
    jit::tracer::addInputs(node, "sigma", sigma);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cauchy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cauchy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, median, sigma, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor cauchy(c10::DispatchKeySet ks, const at::Tensor & self, double median, double sigma, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cauchy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "median", median);
    jit::tracer::addInputs(node, "sigma", sigma);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cauchy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, median, sigma, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & log_normal_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::log_normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_normal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::log_normal_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mean, std, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor log_normal(c10::DispatchKeySet ks, const at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::log_normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::log_normal::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mean, std, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _histogramdd_bin_edges_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density, at::TensorList out) {
  at::_ops::_histogramdd_bin_edges_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, bins, range, weight, density, out);
}
at::Tensor & _histogramdd_from_bin_tensors_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::TensorList bins, const ::std::optional<at::Tensor> & weight, bool density, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_histogramdd_from_bin_tensors");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "bins", bins);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "density", density);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_histogramdd_from_bin_tensors_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_histogramdd_from_bin_tensors_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, bins, weight, density, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & unfold_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_in, c10::SymIntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unfold_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_in", grad_in);
    jit::tracer::addInputs(node, "input_sizes", input_sizes);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "step", step);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("unfold_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::unfold_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_in, input_sizes, dim, size, step, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & normal_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    
    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::normal_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mean, std, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void _foreach_sub_out_Scalar_out(c10::DispatchKeySet ks, at::TensorList self, const at::Scalar & scalar, at::TensorList out) {
  at::_ops::_foreach_sub_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalar, out);
}
void _foreach_sub_out_List_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList other, const at::Scalar & alpha, at::TensorList out) {
  at::_ops::_foreach_sub_List_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha, out);
}
void _foreach_sub_out_ScalarList_out(c10::DispatchKeySet ks, at::TensorList self, at::ArrayRef<at::Scalar> scalars, at::TensorList out) {
  at::_ops::_foreach_sub_ScalarList_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalars, out);
}
void _foreach_maximum_out_Scalar_out(c10::DispatchKeySet ks, at::TensorList self, const at::Scalar & scalar, at::TensorList out) {
  at::_ops::_foreach_maximum_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalar, out);
}
void _foreach_maximum_out_List_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList other, at::TensorList out) {
  at::_ops::_foreach_maximum_List_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
}
void _foreach_maximum_out_ScalarList_out(c10::DispatchKeySet ks, at::TensorList self, at::ArrayRef<at::Scalar> scalars, at::TensorList out) {
  at::_ops::_foreach_maximum_ScalarList_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalars, out);
}
void _foreach_acos_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_acos_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_atan_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_atan_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_ceil_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_ceil_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_erf_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_erf_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_log2_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_log2_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
at::Tensor & bucketize_out_Scalar_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & boundaries, bool out_int32, bool right, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bucketize");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "boundaries", boundaries);
    jit::tracer::addInputs(node, "out_int32", out_int32);
    jit::tracer::addInputs(node, "right", right);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bucketize_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bucketize_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, boundaries, out_int32, right, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & glu_backward_jvp_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_x, const at::Tensor & grad_glu, const at::Tensor & x, const at::Tensor & dgrad_glu, const at::Tensor & dx, int64_t dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::glu_backward_jvp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_x", grad_x);
    jit::tracer::addInputs(node, "grad_glu", grad_glu);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "dgrad_glu", dgrad_glu);
    jit::tracer::addInputs(node, "dx", dx);
    jit::tracer::addInputs(node, "dim", dim);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("glu_backward_jvp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::glu_backward_jvp_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_x, grad_glu, x, dgrad_glu, dx, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & hardswish_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hardswish_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hardswish_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hardswish_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _adaptive_avg_pool3d_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_adaptive_avg_pool3d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_adaptive_avg_pool3d_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_adaptive_avg_pool3d_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _conj_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_conj_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_conj_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_conj_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & detach_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::detach_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("detach_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::detach_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & row_indices_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::row_indices_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("row_indices_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::row_indices_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _transformer_encoder_layer_fwd_out_out(c10::DispatchKeySet ks, const at::Tensor & src, int64_t embed_dim, int64_t num_heads, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, bool use_gelu, bool norm_first, double eps, const at::Tensor & norm_weight_1, const at::Tensor & norm_bias_1, const at::Tensor & norm_weight_2, const at::Tensor & norm_bias_2, const at::Tensor & ffn_weight_1, const at::Tensor & ffn_bias_1, const at::Tensor & ffn_weight_2, const at::Tensor & ffn_bias_2, const ::std::optional<at::Tensor> & mask, ::std::optional<int64_t> mask_type, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_transformer_encoder_layer_fwd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "embed_dim", embed_dim);
    jit::tracer::addInputs(node, "num_heads", num_heads);
    jit::tracer::addInputs(node, "qkv_weight", qkv_weight);
    jit::tracer::addInputs(node, "qkv_bias", qkv_bias);
    jit::tracer::addInputs(node, "proj_weight", proj_weight);
    jit::tracer::addInputs(node, "proj_bias", proj_bias);
    jit::tracer::addInputs(node, "use_gelu", use_gelu);
    jit::tracer::addInputs(node, "norm_first", norm_first);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "norm_weight_1", norm_weight_1);
    jit::tracer::addInputs(node, "norm_bias_1", norm_bias_1);
    jit::tracer::addInputs(node, "norm_weight_2", norm_weight_2);
    jit::tracer::addInputs(node, "norm_bias_2", norm_bias_2);
    jit::tracer::addInputs(node, "ffn_weight_1", ffn_weight_1);
    jit::tracer::addInputs(node, "ffn_bias_1", ffn_bias_1);
    jit::tracer::addInputs(node, "ffn_weight_2", ffn_weight_2);
    jit::tracer::addInputs(node, "ffn_bias_2", ffn_bias_2);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "mask_type", mask_type);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_transformer_encoder_layer_fwd_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_transformer_encoder_layer_fwd_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), src, embed_dim, num_heads, qkv_weight, qkv_bias, proj_weight, proj_bias, use_gelu, norm_first, eps, norm_weight_1, norm_bias_1, norm_weight_2, norm_bias_2, ffn_weight_1, ffn_bias_1, ffn_weight_2, ffn_bias_2, mask, mask_type, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> _native_multi_head_attention_out_out(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t embed_dim, int64_t num_head, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, const ::std::optional<at::Tensor> & mask, bool need_weights, bool average_attn_weights, ::std::optional<int64_t> mask_type, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_native_multi_head_attention");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "embed_dim", embed_dim);
    jit::tracer::addInputs(node, "num_head", num_head);
    jit::tracer::addInputs(node, "qkv_weight", qkv_weight);
    jit::tracer::addInputs(node, "qkv_bias", qkv_bias);
    jit::tracer::addInputs(node, "proj_weight", proj_weight);
    jit::tracer::addInputs(node, "proj_bias", proj_bias);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "need_weights", need_weights);
    jit::tracer::addInputs(node, "average_attn_weights", average_attn_weights);
    jit::tracer::addInputs(node, "mask_type", mask_type);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_native_multi_head_attention_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_native_multi_head_attention_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, embed_dim, num_head, qkv_weight, qkv_bias, proj_weight, proj_bias, mask, need_weights, average_attn_weights, mask_type, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & _triton_multi_head_attention_out_out(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t embed_dim, int64_t num_head, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, const ::std::optional<at::Tensor> & mask, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_triton_multi_head_attention");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "embed_dim", embed_dim);
    jit::tracer::addInputs(node, "num_head", num_head);
    jit::tracer::addInputs(node, "qkv_weight", qkv_weight);
    jit::tracer::addInputs(node, "qkv_bias", qkv_bias);
    jit::tracer::addInputs(node, "proj_weight", proj_weight);
    jit::tracer::addInputs(node, "proj_bias", proj_bias);
    jit::tracer::addInputs(node, "mask", mask);
    
    if (tracer_state->force_outplace) {
    
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_triton_multi_head_attention_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_triton_multi_head_attention_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, embed_dim, num_head, qkv_weight, qkv_bias, proj_weight, proj_bias, mask, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void _fused_adamw_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps, double lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf, at::TensorList out) {
  at::_ops::_fused_adamw_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf, out);
}
::std::tuple<::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>> _fused_adamw(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps, double lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fused_adamw");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grads", grads);
    jit::tracer::addInputs(node, "exp_avgs", exp_avgs);
    jit::tracer::addInputs(node, "exp_avg_sqs", exp_avg_sqs);
    jit::tracer::addInputs(node, "max_exp_avg_sqs", max_exp_avg_sqs);
    jit::tracer::addInputs(node, "state_steps", state_steps);
    jit::tracer::addInputs(node, "lr", lr);
    jit::tracer::addInputs(node, "beta1", beta1);
    jit::tracer::addInputs(node, "beta2", beta2);
    jit::tracer::addInputs(node, "weight_decay", weight_decay);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "amsgrad", amsgrad);
    jit::tracer::addInputs(node, "maximize", maximize);
    jit::tracer::addInputs(node, "grad_scale", grad_scale);
    jit::tracer::addInputs(node, "found_inf", found_inf);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [self_out, grads_out, exp_avgs_out, exp_avg_sqs_out, max_exp_avg_sqs_out] =at::_ops::_fused_adamw::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self_out);
    jit::tracer::addOutput(node, grads_out);
    jit::tracer::addOutput(node, exp_avgs_out);
    jit::tracer::addOutput(node, exp_avg_sqs_out);
    jit::tracer::addOutput(node, max_exp_avg_sqs_out);
  }
  return std::make_tuple(std::move(self_out), std::move(grads_out), std::move(exp_avgs_out), std::move(exp_avg_sqs_out), std::move(max_exp_avg_sqs_out));
}
void _fused_adamw_out_tensor_lr_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps, const at::Tensor & lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf, at::TensorList out) {
  at::_ops::_fused_adamw_tensor_lr_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf, out);
}
::std::tuple<::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>> _fused_adamw_tensor_lr(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps, const at::Tensor & lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fused_adamw");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grads", grads);
    jit::tracer::addInputs(node, "exp_avgs", exp_avgs);
    jit::tracer::addInputs(node, "exp_avg_sqs", exp_avg_sqs);
    jit::tracer::addInputs(node, "max_exp_avg_sqs", max_exp_avg_sqs);
    jit::tracer::addInputs(node, "state_steps", state_steps);
    jit::tracer::addInputs(node, "lr", lr);
    jit::tracer::addInputs(node, "beta1", beta1);
    jit::tracer::addInputs(node, "beta2", beta2);
    jit::tracer::addInputs(node, "weight_decay", weight_decay);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "amsgrad", amsgrad);
    jit::tracer::addInputs(node, "maximize", maximize);
    jit::tracer::addInputs(node, "grad_scale", grad_scale);
    jit::tracer::addInputs(node, "found_inf", found_inf);
    tracer_state->insertNode(node);
  
    jit::tracer::setTracingState(nullptr);
  }
  auto [self_out, grads_out, exp_avgs_out, exp_avg_sqs_out, max_exp_avg_sqs_out] =at::_ops::_fused_adamw_tensor_lr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self_out);
    jit::tracer::addOutput(node, grads_out);
    jit::tracer::addOutput(node, exp_avgs_out);
    jit::tracer::addOutput(node, exp_avg_sqs_out);
    jit::tracer::addOutput(node, max_exp_avg_sqs_out);
  }
  return std::make_tuple(std::move(self_out), std::move(grads_out), std::move(exp_avgs_out), std::move(exp_avg_sqs_out), std::move(max_exp_avg_sqs_out));
}
}  // namespace
}  // namespace TraceType

namespace {

TORCH_LIBRARY_IMPL(aten, Tracer, m) {
  m.impl("_cast_Double",
         TORCH_FN(TraceType::_cast_Double)
  );
  m.impl("_cast_Float",
         TORCH_FN(TraceType::_cast_Float)
  );
  m.impl("_cast_Half",
         TORCH_FN(TraceType::_cast_Half)
  );
  m.impl("align_as",
         TORCH_FN(TraceType::align_as)
  );
  m.impl("_assert_tensor_metadata",
         TORCH_FN(TraceType::_assert_tensor_metadata)
  );
  m.impl("sym_constrain_range_for_size",
         TORCH_FN(TraceType::sym_constrain_range_for_size)
  );
  m.impl("refine_names",
         TORCH_FN(TraceType::refine_names)
  );
  m.impl("_use_cudnn_rnn_flatten_weight",
         TORCH_FN(TraceType::_use_cudnn_rnn_flatten_weight)
  );
  m.impl("_cudnn_rnn_flatten_weight",
         TORCH_FN(TraceType::_cudnn_rnn_flatten_weight)
  );
  m.impl("_sobol_engine_ff_",
         TORCH_FN(TraceType::_sobol_engine_ff_)
  );
  m.impl("_sobol_engine_scramble_",
         TORCH_FN(TraceType::_sobol_engine_scramble_)
  );
  m.impl("feature_alpha_dropout",
         TORCH_FN(TraceType::feature_alpha_dropout)
  );
  m.impl("feature_alpha_dropout_",
         TORCH_FN(TraceType::feature_alpha_dropout_)
  );
  m.impl("abs",
         TORCH_FN(TraceType::abs)
  );
  m.impl("abs_",
         TORCH_FN(TraceType::abs_)
  );
  m.impl("abs.out",
         TORCH_FN(TraceType::abs_out_out)
  );
  m.impl("imag",
         TORCH_FN(TraceType::imag)
  );
  m.impl("resolve_conj",
         TORCH_FN(TraceType::resolve_conj)
  );
  m.impl("resolve_neg",
         TORCH_FN(TraceType::resolve_neg)
  );
  m.impl("adaptive_max_pool1d",
         TORCH_FN(TraceType::adaptive_max_pool1d)
  );
  m.impl("addmv",
         TORCH_FN(TraceType::addmv)
  );
  m.impl("addmv_",
         TORCH_FN(TraceType::addmv_)
  );
  m.impl("addmv.out",
         TORCH_FN(TraceType::addmv_out_out)
  );
  m.impl("addr",
         TORCH_FN(TraceType::addr)
  );
  m.impl("addr_",
         TORCH_FN(TraceType::addr_)
  );
  m.impl("addr.out",
         TORCH_FN(TraceType::addr_out_out)
  );
  m.impl("affine_grid_generator_backward",
         TORCH_FN(TraceType::affine_grid_generator_backward)
  );
  m.impl("argmin",
         TORCH_FN(TraceType::argmin)
  );
  m.impl("argmin.out",
         TORCH_FN(TraceType::argmin_out_out)
  );
  m.impl("atan",
         TORCH_FN(TraceType::atan)
  );
  m.impl("atan_",
         TORCH_FN(TraceType::atan_)
  );
  m.impl("atan.out",
         TORCH_FN(TraceType::atan_out_out)
  );
  m.impl("arctan",
         TORCH_FN(TraceType::arctan)
  );
  m.impl("arctan_",
         TORCH_FN(TraceType::arctan_)
  );
  m.impl("arctan.out",
         TORCH_FN(TraceType::arctan_out_out)
  );
  m.impl("quantized_batch_norm",
         TORCH_FN(TraceType::quantized_batch_norm)
  );
  m.impl("binary_cross_entropy_backward",
         TORCH_FN(TraceType::binary_cross_entropy_backward)
  );
  m.impl("binary_cross_entropy_backward.grad_input",
         TORCH_FN(TraceType::binary_cross_entropy_backward_out_grad_input)
  );
  m.impl("bitwise_not",
         TORCH_FN(TraceType::bitwise_not)
  );
  m.impl("bitwise_not_",
         TORCH_FN(TraceType::bitwise_not_)
  );
  m.impl("bitwise_not.out",
         TORCH_FN(TraceType::bitwise_not_out_out)
  );
  m.impl("logical_not",
         TORCH_FN(TraceType::logical_not)
  );
  m.impl("logical_not_",
         TORCH_FN(TraceType::logical_not_)
  );
  m.impl("logical_not.out",
         TORCH_FN(TraceType::logical_not_out_out)
  );
  m.impl("concatenate",
         TORCH_FN(TraceType::concatenate)
  );
  m.impl("concatenate.out",
         TORCH_FN(TraceType::concatenate_out_out)
  );
  m.impl("concatenate.names",
         TORCH_FN(TraceType::concatenate_names)
  );
  m.impl("concatenate.names_out",
         TORCH_FN(TraceType::concatenate_out_names_out)
  );
  m.impl("ceil",
         TORCH_FN(TraceType::ceil)
  );
  m.impl("ceil_",
         TORCH_FN(TraceType::ceil_)
  );
  m.impl("ceil.out",
         TORCH_FN(TraceType::ceil_out_out)
  );
  m.impl("conv_tbc",
         TORCH_FN(TraceType::conv_tbc)
  );
  m.impl("cosh",
         TORCH_FN(TraceType::cosh)
  );
  m.impl("cosh_",
         TORCH_FN(TraceType::cosh_)
  );
  m.impl("cosh.out",
         TORCH_FN(TraceType::cosh_out_out)
  );
  m.impl("cosine_embedding_loss",
         TORCH_FN(TraceType::cosine_embedding_loss)
  );
  m.impl("cudnn_affine_grid_generator_backward",
         TORCH_FN(TraceType::cudnn_affine_grid_generator_backward)
  );
  m.impl("cudnn_grid_sampler",
         TORCH_FN(TraceType::cudnn_grid_sampler)
  );
  m.impl("cummin",
         TORCH_FN(TraceType::cummin)
  );
  m.impl("cummin.out",
         TORCH_FN(TraceType::cummin_out_out)
  );
  m.impl("cummin.dimname",
         TORCH_FN(TraceType::cummin_dimname)
  );
  m.impl("cummin.dimname_out",
         TORCH_FN(TraceType::cummin_out_dimname_out)
  );
  m.impl("_cummin_helper",
         TORCH_FN(TraceType::_cummin_helper)
  );
  m.impl("div.Tensor",
         TORCH_FN(TraceType::div_Tensor)
  );
  m.impl("div_.Tensor",
         TORCH_FN(TraceType::div__Tensor)
  );
  m.impl("div.out",
         TORCH_FN(TraceType::div_out_out)
  );
  m.impl("div.Tensor_mode",
         TORCH_FN(TraceType::div_Tensor_mode)
  );
  m.impl("div_.Tensor_mode",
         TORCH_FN(TraceType::div__Tensor_mode)
  );
  m.impl("div.out_mode",
         TORCH_FN(TraceType::div_out_out_mode)
  );
  m.impl("div.Scalar",
         TORCH_FN(TraceType::div_Scalar)
  );
  m.impl("div_.Scalar",
         TORCH_FN(TraceType::div__Scalar)
  );
  m.impl("div.Scalar_mode",
         TORCH_FN(TraceType::div_Scalar_mode)
  );
  m.impl("div_.Scalar_mode",
         TORCH_FN(TraceType::div__Scalar_mode)
  );
  m.impl("_embedding_bag_forward_only",
         TORCH_FN(TraceType::_embedding_bag_forward_only)
  );
  m.impl("embedding_bag",
         TORCH_FN(TraceType::embedding_bag)
  );
  m.impl("embedding_bag.padding_idx",
         TORCH_FN(TraceType::embedding_bag_padding_idx)
  );
  m.impl("new_zeros",
         TORCH_FN(TraceType::new_zeros)
  );
  m.impl("erf",
         TORCH_FN(TraceType::erf)
  );
  m.impl("erf_",
         TORCH_FN(TraceType::erf_)
  );
  m.impl("erf.out",
         TORCH_FN(TraceType::erf_out_out)
  );
  m.impl("grid_sampler",
         TORCH_FN(TraceType::grid_sampler)
  );
  m.impl("_grid_sampler_2d_cpu_fallback",
         TORCH_FN(TraceType::_grid_sampler_2d_cpu_fallback)
  );
  m.impl("grid_sampler_3d",
         TORCH_FN(TraceType::grid_sampler_3d)
  );
  m.impl("hann_window",
         TORCH_FN(TraceType::hann_window)
  );
  m.impl("hann_window.periodic",
         TORCH_FN(TraceType::hann_window_periodic)
  );
  m.impl("hamming_window",
         TORCH_FN(TraceType::hamming_window)
  );
  m.impl("hamming_window.periodic",
         TORCH_FN(TraceType::hamming_window_periodic)
  );
  m.impl("hamming_window.periodic_alpha",
         TORCH_FN(TraceType::hamming_window_periodic_alpha)
  );
  m.impl("hamming_window.periodic_alpha_beta",
         TORCH_FN(TraceType::hamming_window_periodic_alpha_beta)
  );
  m.impl("native_group_norm_backward",
         TORCH_FN(TraceType::native_group_norm_backward)
  );
  m.impl("_fft_c2c",
         TORCH_FN(TraceType::_fft_c2c)
  );
  m.impl("_fft_c2c.out",
         TORCH_FN(TraceType::_fft_c2c_out_out)
  );
  m.impl("_validate_compressed_sparse_indices",
         TORCH_FN(TraceType::_validate_compressed_sparse_indices)
  );
  m.impl("_cufft_get_plan_cache_size",
         TORCH_FN(TraceType::_cufft_get_plan_cache_size)
  );
  m.impl("_cufft_get_plan_cache_max_size",
         TORCH_FN(TraceType::_cufft_get_plan_cache_max_size)
  );
  m.impl("index.Tensor",
         TORCH_FN(TraceType::index_Tensor)
  );
  m.impl("index.Tensor_out",
         TORCH_FN(TraceType::index_out_Tensor_out)
  );
  m.impl("isnan",
         TORCH_FN(TraceType::isnan)
  );
  m.impl("kthvalue",
         TORCH_FN(TraceType::kthvalue)
  );
  m.impl("kthvalue.values",
         TORCH_FN(TraceType::kthvalue_out_values)
  );
  m.impl("kthvalue.dimname",
         TORCH_FN(TraceType::kthvalue_dimname)
  );
  m.impl("kthvalue.dimname_out",
         TORCH_FN(TraceType::kthvalue_out_dimname_out)
  );
  m.impl("native_layer_norm",
         TORCH_FN(TraceType::native_layer_norm)
  );
  m.impl("nan_to_num",
         TORCH_FN(TraceType::nan_to_num)
  );
  m.impl("nan_to_num_",
         TORCH_FN(TraceType::nan_to_num_)
  );
  m.impl("nan_to_num.out",
         TORCH_FN(TraceType::nan_to_num_out_out)
  );
  m.impl("_sparse_semi_structured_apply",
         TORCH_FN(TraceType::_sparse_semi_structured_apply)
  );
  m.impl("_sparse_semi_structured_mm",
         TORCH_FN(TraceType::_sparse_semi_structured_mm)
  );
  m.impl("fbgemm_linear_int8_weight_fp32_activation",
         TORCH_FN(TraceType::fbgemm_linear_int8_weight_fp32_activation)
  );
  m.impl("fbgemm_linear_int8_weight",
         TORCH_FN(TraceType::fbgemm_linear_int8_weight)
  );
  m.impl("fbgemm_linear_fp16_weight_fp32_activation",
         TORCH_FN(TraceType::fbgemm_linear_fp16_weight_fp32_activation)
  );
  m.impl("xlogy.Tensor",
         TORCH_FN(TraceType::xlogy_Tensor)
  );
  m.impl("xlogy.Scalar_Self",
         TORCH_FN(TraceType::xlogy_Scalar_Self)
  );
  m.impl("xlogy.Scalar_Other",
         TORCH_FN(TraceType::xlogy_Scalar_Other)
  );
  m.impl("xlogy_.Tensor",
         TORCH_FN(TraceType::xlogy__Tensor)
  );
  m.impl("xlogy_.Scalar_Other",
         TORCH_FN(TraceType::xlogy__Scalar_Other)
  );
  m.impl("xlogy.OutTensor",
         TORCH_FN(TraceType::xlogy_out_OutTensor)
  );
  m.impl("xlogy.OutScalar_Self",
         TORCH_FN(TraceType::xlogy_out_OutScalar_Self)
  );
  m.impl("xlogy.OutScalar_Other",
         TORCH_FN(TraceType::xlogy_out_OutScalar_Other)
  );
  m.impl("_log_softmax_backward_data",
         TORCH_FN(TraceType::_log_softmax_backward_data)
  );
  m.impl("_log_softmax_backward_data.out",
         TORCH_FN(TraceType::_log_softmax_backward_data_out_out)
  );
  m.impl("logcumsumexp",
         TORCH_FN(TraceType::logcumsumexp)
  );
  m.impl("logcumsumexp.out",
         TORCH_FN(TraceType::logcumsumexp_out_out)
  );
  m.impl("logcumsumexp.dimname",
         TORCH_FN(TraceType::logcumsumexp_dimname)
  );
  m.impl("logcumsumexp.dimname_out",
         TORCH_FN(TraceType::logcumsumexp_out_dimname_out)
  );
  m.impl("matrix_exp_backward",
         TORCH_FN(TraceType::matrix_exp_backward)
  );
  m.impl("amax",
         TORCH_FN(TraceType::amax)
  );
  m.impl("amax.out",
         TORCH_FN(TraceType::amax_out_out)
  );
  m.impl("mkldnn_max_pool2d",
         TORCH_FN(TraceType::mkldnn_max_pool2d)
  );
  m.impl("quantized_max_pool2d",
         TORCH_FN(TraceType::quantized_max_pool2d)
  );
  m.impl("amin",
         TORCH_FN(TraceType::amin)
  );
  m.impl("amin.out",
         TORCH_FN(TraceType::amin_out_out)
  );
  m.impl("_mps_convolution",
         TORCH_FN(TraceType::_mps_convolution)
  );
  m.impl("mkldnn_rnn_layer_backward",
         TORCH_FN(TraceType::mkldnn_rnn_layer_backward)
  );
  m.impl("miopen_depthwise_convolution",
         TORCH_FN(TraceType::miopen_depthwise_convolution)
  );
  m.impl("_int_mm",
         TORCH_FN(TraceType::_int_mm)
  );
  m.impl("_int_mm.out",
         TORCH_FN(TraceType::_int_mm_out_out)
  );
  m.impl("_dyn_quant_pack_4bit_weight",
         TORCH_FN(TraceType::_dyn_quant_pack_4bit_weight)
  );
  m.impl("_weight_int8pack_mm",
         TORCH_FN(TraceType::_weight_int8pack_mm)
  );
  m.impl("native_batch_norm",
         TORCH_FN(TraceType::native_batch_norm)
  );
  m.impl("native_batch_norm.out",
         TORCH_FN(TraceType::native_batch_norm_out_out)
  );
  m.impl("batch_norm_stats",
         TORCH_FN(TraceType::batch_norm_stats)
  );
  m.impl("batch_norm_gather_stats",
         TORCH_FN(TraceType::batch_norm_gather_stats)
  );
  m.impl("native_batch_norm_backward",
         TORCH_FN(TraceType::native_batch_norm_backward)
  );
  m.impl("batch_norm_backward_reduce",
         TORCH_FN(TraceType::batch_norm_backward_reduce)
  );
  m.impl("is_vulkan_available",
         TORCH_FN(TraceType::is_vulkan_available)
  );
  m.impl("_nnpack_spatial_convolution",
         TORCH_FN(TraceType::_nnpack_spatial_convolution)
  );
  m.impl("ones.names",
         TORCH_FN(TraceType::ones_names)
  );
  m.impl("ones",
         TORCH_FN(TraceType::ones)
  );
  m.impl("ones.out",
         TORCH_FN(TraceType::ones_out_out)
  );
  m.impl("_cdist_forward",
         TORCH_FN(TraceType::_cdist_forward)
  );
  m.impl("cosine_similarity",
         TORCH_FN(TraceType::cosine_similarity)
  );
  m.impl("movedim.intlist",
         TORCH_FN(TraceType::movedim_intlist)
  );
  m.impl("movedim.int",
         TORCH_FN(TraceType::movedim_int)
  );
  m.impl("numpy_T",
         TORCH_FN(TraceType::numpy_T)
  );
  m.impl("mH",
         TORCH_FN(TraceType::mH)
  );
  m.impl("rand_like",
         TORCH_FN(TraceType::rand_like)
  );
  m.impl("randint_like",
         TORCH_FN(TraceType::randint_like)
  );
  m.impl("randint_like.low_dtype",
         TORCH_FN(TraceType::randint_like_low_dtype)
  );
  m.impl("round",
         TORCH_FN(TraceType::round)
  );
  m.impl("round_",
         TORCH_FN(TraceType::round_)
  );
  m.impl("round.out",
         TORCH_FN(TraceType::round_out_out)
  );
  m.impl("round.decimals",
         TORCH_FN(TraceType::round_decimals)
  );
  m.impl("round_.decimals",
         TORCH_FN(TraceType::round__decimals)
  );
  m.impl("round.decimals_out",
         TORCH_FN(TraceType::round_out_decimals_out)
  );
  m.impl("gelu.out",
         TORCH_FN(TraceType::gelu_out_out)
  );
  m.impl("gelu_",
         TORCH_FN(TraceType::gelu_)
  );
  m.impl("gelu",
         TORCH_FN(TraceType::gelu)
  );
  m.impl("hardshrink.out",
         TORCH_FN(TraceType::hardshrink_out_out)
  );
  m.impl("hardshrink",
         TORCH_FN(TraceType::hardshrink)
  );
  m.impl("select_backward",
         TORCH_FN(TraceType::select_backward)
  );
  m.impl("mish",
         TORCH_FN(TraceType::mish)
  );
  m.impl("mish_",
         TORCH_FN(TraceType::mish_)
  );
  m.impl("mish.out",
         TORCH_FN(TraceType::mish_out_out)
  );
  m.impl("sigmoid",
         TORCH_FN(TraceType::sigmoid)
  );
  m.impl("sigmoid_",
         TORCH_FN(TraceType::sigmoid_)
  );
  m.impl("sigmoid.out",
         TORCH_FN(TraceType::sigmoid_out_out)
  );
  m.impl("size.int",
         TORCH_FN(TraceType::size_int)
  );
  m.impl("size.Dimname",
         TORCH_FN(TraceType::size_Dimname)
  );
  m.impl("sym_numel",
         TORCH_FN(TraceType::sym_numel)
  );
  m.impl("slice_scatter",
         TORCH_FN(TraceType::slice_scatter)
  );
  m.impl("_softmax_backward_data",
         TORCH_FN(TraceType::_softmax_backward_data)
  );
  m.impl("_softmax_backward_data.out",
         TORCH_FN(TraceType::_softmax_backward_data_out_out)
  );
  m.impl("split_with_sizes",
         TORCH_FN(TraceType::split_with_sizes)
  );
  m.impl("hsplit.int",
         TORCH_FN(TraceType::hsplit_int)
  );
  m.impl("hsplit.array",
         TORCH_FN(TraceType::hsplit_array)
  );
  m.impl("stack",
         TORCH_FN(TraceType::stack)
  );
  m.impl("stack.out",
         TORCH_FN(TraceType::stack_out_out)
  );
  m.impl("_stack",
         TORCH_FN(TraceType::_stack)
  );
  m.impl("_stack.out",
         TORCH_FN(TraceType::_stack_out_out)
  );
  m.impl("square",
         TORCH_FN(TraceType::square)
  );
  m.impl("square_",
         TORCH_FN(TraceType::square_)
  );
  m.impl("square.out",
         TORCH_FN(TraceType::square_out_out)
  );
  m.impl("tanh",
         TORCH_FN(TraceType::tanh)
  );
  m.impl("tanh_",
         TORCH_FN(TraceType::tanh_)
  );
  m.impl("tanh.out",
         TORCH_FN(TraceType::tanh_out_out)
  );
  m.impl("tensordot",
         TORCH_FN(TraceType::tensordot)
  );
  m.impl("tensordot.out",
         TORCH_FN(TraceType::tensordot_out_out)
  );
  m.impl("tile",
         TORCH_FN(TraceType::tile)
  );
  m.impl("_mkldnn_transpose",
         TORCH_FN(TraceType::_mkldnn_transpose)
  );
  m.impl("_mkldnn_transpose_",
         TORCH_FN(TraceType::_mkldnn_transpose_)
  );
  m.impl("fliplr",
         TORCH_FN(TraceType::fliplr)
  );
  m.impl("_nested_from_padded_and_nested_example",
         TORCH_FN(TraceType::_nested_from_padded_and_nested_example)
  );
  m.impl("_nested_get_max_seqlen",
         TORCH_FN(TraceType::_nested_get_max_seqlen)
  );
  m.impl("_nested_compute_contiguous_strides_offsets",
         TORCH_FN(TraceType::_nested_compute_contiguous_strides_offsets)
  );
  m.impl("fix",
         TORCH_FN(TraceType::fix)
  );
  m.impl("fix_",
         TORCH_FN(TraceType::fix_)
  );
  m.impl("fix.out",
         TORCH_FN(TraceType::fix_out_out)
  );
  m.impl("unique_dim",
         TORCH_FN(TraceType::unique_dim)
  );
  m.impl("unique_consecutive",
         TORCH_FN(TraceType::unique_consecutive)
  );
  m.impl("vander",
         TORCH_FN(TraceType::vander)
  );
  m.impl("view_as",
         TORCH_FN(TraceType::view_as)
  );
  m.impl("_dirichlet_grad",
         TORCH_FN(TraceType::_dirichlet_grad)
  );
  m.impl("batch_norm_backward",
         TORCH_FN(TraceType::batch_norm_backward)
  );
  m.impl("frobenius_norm.dim",
         TORCH_FN(TraceType::frobenius_norm_dim)
  );
  m.impl("frobenius_norm.out",
         TORCH_FN(TraceType::frobenius_norm_out_out)
  );
  m.impl("clone",
         TORCH_FN(TraceType::clone)
  );
  m.impl("positive",
         TORCH_FN(TraceType::positive)
  );
  m.impl("resize_as_sparse_",
         TORCH_FN(TraceType::resize_as_sparse_)
  );
  m.impl("sparse_sampled_addmm.out",
         TORCH_FN(TraceType::sparse_sampled_addmm_out_out)
  );
  m.impl("sparse_sampled_addmm",
         TORCH_FN(TraceType::sparse_sampled_addmm)
  );
  m.impl("sparse_csr_tensor.crow_col_value_size",
         TORCH_FN(TraceType::sparse_csr_tensor_crow_col_value_size)
  );
  m.impl("sparse_csr_tensor.crow_col_value",
         TORCH_FN(TraceType::sparse_csr_tensor_crow_col_value)
  );
  m.impl("_sparse_bsc_tensor_unsafe",
         TORCH_FN(TraceType::_sparse_bsc_tensor_unsafe)
  );
  m.impl("dense_dim",
         TORCH_FN(TraceType::dense_dim)
  );
  m.impl("_dimV",
         TORCH_FN(TraceType::_dimV)
  );
  m.impl("coalesce",
         TORCH_FN(TraceType::coalesce)
  );
  m.impl("_indices",
         TORCH_FN(TraceType::_indices)
  );
  m.impl("to_sparse_csc",
         TORCH_FN(TraceType::to_sparse_csc)
  );
  m.impl("_to_sparse_csc",
         TORCH_FN(TraceType::_to_sparse_csc)
  );
  m.impl("_to_sparse_bsc",
         TORCH_FN(TraceType::_to_sparse_bsc)
  );
  m.impl("mkldnn_reorder_conv2d_weight",
         TORCH_FN(TraceType::mkldnn_reorder_conv2d_weight)
  );
  m.impl("quantize_per_channel",
         TORCH_FN(TraceType::quantize_per_channel)
  );
  m.impl("dequantize.self",
         TORCH_FN(TraceType::dequantize_self)
  );
  m.impl("dequantize.tensors",
         TORCH_FN(TraceType::dequantize_tensors)
  );
  m.impl("q_per_channel_zero_points",
         TORCH_FN(TraceType::q_per_channel_zero_points)
  );
  m.impl("fake_quantize_per_tensor_affine",
         TORCH_FN(TraceType::fake_quantize_per_tensor_affine)
  );
  m.impl("fake_quantize_per_tensor_affine.tensor_qparams",
         TORCH_FN(TraceType::fake_quantize_per_tensor_affine_tensor_qparams)
  );
  m.impl("_fake_quantize_learnable_per_channel_affine",
         TORCH_FN(TraceType::_fake_quantize_learnable_per_channel_affine)
  );
  m.impl("_autocast_to_full_precision",
         TORCH_FN(TraceType::_autocast_to_full_precision)
  );
  m.impl("to.dtype_layout",
         TORCH_FN(TraceType::to_dtype_layout)
  );
  m.impl("to.device",
         TORCH_FN(TraceType::to_device)
  );
  m.impl("to.dtype",
         TORCH_FN(TraceType::to_dtype)
  );
  m.impl("to.other",
         TORCH_FN(TraceType::to_other)
  );
  m.impl("combinations",
         TORCH_FN(TraceType::combinations)
  );
  m.impl("item",
         TORCH_FN(TraceType::item)
  );
  m.impl("_lstm_mps",
         TORCH_FN(TraceType::_lstm_mps)
  );
  m.impl("_thnn_fused_lstm_cell",
         TORCH_FN(TraceType::_thnn_fused_lstm_cell)
  );
  m.impl("lstm.input",
         TORCH_FN(TraceType::lstm_input)
  );
  m.impl("lstm.data",
         TORCH_FN(TraceType::lstm_data)
  );
  m.impl("gru.input",
         TORCH_FN(TraceType::gru_input)
  );
  m.impl("gru.data",
         TORCH_FN(TraceType::gru_data)
  );
  m.impl("rnn_tanh.input",
         TORCH_FN(TraceType::rnn_tanh_input)
  );
  m.impl("rnn_tanh.data",
         TORCH_FN(TraceType::rnn_tanh_data)
  );
  m.impl("rnn_relu_cell",
         TORCH_FN(TraceType::rnn_relu_cell)
  );
  m.impl("_pad_packed_sequence",
         TORCH_FN(TraceType::_pad_packed_sequence)
  );
  m.impl("lift_fresh_copy",
         TORCH_FN(TraceType::lift_fresh_copy)
  );
  m.impl("index_reduce.out",
         TORCH_FN(TraceType::index_reduce_out_out)
  );
  m.impl("index_reduce_",
         TORCH_FN(TraceType::index_reduce_)
  );
  m.impl("index_reduce",
         TORCH_FN(TraceType::index_reduce)
  );
  m.impl("index_fill_.int_Scalar",
         TORCH_FN(TraceType::index_fill__int_Scalar)
  );
  m.impl("index_fill.int_Scalar",
         TORCH_FN(TraceType::index_fill_int_Scalar)
  );
  m.impl("index_fill_.int_Tensor",
         TORCH_FN(TraceType::index_fill__int_Tensor)
  );
  m.impl("index_fill.int_Tensor",
         TORCH_FN(TraceType::index_fill_int_Tensor)
  );
  m.impl("index_fill_.Dimname_Scalar",
         TORCH_FN(TraceType::index_fill__Dimname_Scalar)
  );
  m.impl("index_fill_.Dimname_Tensor",
         TORCH_FN(TraceType::index_fill__Dimname_Tensor)
  );
  m.impl("index_fill.Dimname_Scalar",
         TORCH_FN(TraceType::index_fill_Dimname_Scalar)
  );
  m.impl("index_fill.Dimname_Tensor",
         TORCH_FN(TraceType::index_fill_Dimname_Tensor)
  );
  m.impl("scatter_add",
         TORCH_FN(TraceType::scatter_add)
  );
  m.impl("scatter_add_",
         TORCH_FN(TraceType::scatter_add_)
  );
  m.impl("scatter_add.out",
         TORCH_FN(TraceType::scatter_add_out_out)
  );
  m.impl("scatter_add.dimname",
         TORCH_FN(TraceType::scatter_add_dimname)
  );
  m.impl("digamma_",
         TORCH_FN(TraceType::digamma_)
  );
  m.impl("random_.from",
         TORCH_FN(TraceType::random__from)
  );
  m.impl("random_.to",
         TORCH_FN(TraceType::random__to)
  );
  m.impl("random_",
         TORCH_FN(TraceType::random_)
  );
  m.impl("cauchy_",
         TORCH_FN(TraceType::cauchy_)
  );
  m.impl("log_normal_",
         TORCH_FN(TraceType::log_normal_)
  );
  m.impl("cross.out",
         TORCH_FN(TraceType::cross_out_out)
  );
  m.impl("cross",
         TORCH_FN(TraceType::cross)
  );
  m.impl("ne.Scalar_out",
         TORCH_FN(TraceType::ne_out_Scalar_out)
  );
  m.impl("ne.Scalar",
         TORCH_FN(TraceType::ne_Scalar)
  );
  m.impl("ne.Tensor_out",
         TORCH_FN(TraceType::ne_out_Tensor_out)
  );
  m.impl("ne.Tensor",
         TORCH_FN(TraceType::ne_Tensor)
  );
  m.impl("ne_.Scalar",
         TORCH_FN(TraceType::ne__Scalar)
  );
  m.impl("ne_.Tensor",
         TORCH_FN(TraceType::ne__Tensor)
  );
  m.impl("ge.Scalar_out",
         TORCH_FN(TraceType::ge_out_Scalar_out)
  );
  m.impl("ge.Scalar",
         TORCH_FN(TraceType::ge_Scalar)
  );
  m.impl("ge.Tensor_out",
         TORCH_FN(TraceType::ge_out_Tensor_out)
  );
  m.impl("ge.Tensor",
         TORCH_FN(TraceType::ge_Tensor)
  );
  m.impl("ge_.Scalar",
         TORCH_FN(TraceType::ge__Scalar)
  );
  m.impl("ge_.Tensor",
         TORCH_FN(TraceType::ge__Tensor)
  );
  m.impl("_gather_sparse_backward",
         TORCH_FN(TraceType::_gather_sparse_backward)
  );
  m.impl("linalg_vander",
         TORCH_FN(TraceType::linalg_vander)
  );
  m.impl("swapaxes",
         TORCH_FN(TraceType::swapaxes)
  );
  m.impl("swapaxes_",
         TORCH_FN(TraceType::swapaxes_)
  );
  m.impl("cholesky_solve.out",
         TORCH_FN(TraceType::cholesky_solve_out_out)
  );
  m.impl("cholesky_solve",
         TORCH_FN(TraceType::cholesky_solve)
  );
  m.impl("qr.Q",
         TORCH_FN(TraceType::qr_out_Q)
  );
  m.impl("qr",
         TORCH_FN(TraceType::qr)
  );
  m.impl("digamma.out",
         TORCH_FN(TraceType::digamma_out_out)
  );
  m.impl("digamma",
         TORCH_FN(TraceType::digamma)
  );
  m.impl("polygamma.out",
         TORCH_FN(TraceType::polygamma_out_out)
  );
  m.impl("polygamma",
         TORCH_FN(TraceType::polygamma)
  );
  m.impl("polygamma_",
         TORCH_FN(TraceType::polygamma_)
  );
  m.impl("histc.out",
         TORCH_FN(TraceType::histc_out_out)
  );
  m.impl("histc",
         TORCH_FN(TraceType::histc)
  );
  m.impl("_histogramdd_bin_edges",
         TORCH_FN(TraceType::_histogramdd_bin_edges)
  );
  m.impl("_histogramdd_from_bin_tensors",
         TORCH_FN(TraceType::_histogramdd_from_bin_tensors)
  );
  m.impl("nextafter.out",
         TORCH_FN(TraceType::nextafter_out_out)
  );
  m.impl("nextafter",
         TORCH_FN(TraceType::nextafter)
  );
  m.impl("nextafter_",
         TORCH_FN(TraceType::nextafter_)
  );
  m.impl("maximum",
         TORCH_FN(TraceType::maximum)
  );
  m.impl("maximum.out",
         TORCH_FN(TraceType::maximum_out_out)
  );
  m.impl("minimum",
         TORCH_FN(TraceType::minimum)
  );
  m.impl("minimum.out",
         TORCH_FN(TraceType::minimum_out_out)
  );
  m.impl("quantile",
         TORCH_FN(TraceType::quantile)
  );
  m.impl("quantile.out",
         TORCH_FN(TraceType::quantile_out_out)
  );
  m.impl("quantile.scalar",
         TORCH_FN(TraceType::quantile_scalar)
  );
  m.impl("quantile.scalar_out",
         TORCH_FN(TraceType::quantile_out_scalar_out)
  );
  m.impl("msort.out",
         TORCH_FN(TraceType::msort_out_out)
  );
  m.impl("msort",
         TORCH_FN(TraceType::msort)
  );
  m.impl("argsort",
         TORCH_FN(TraceType::argsort)
  );
  m.impl("argsort.stable",
         TORCH_FN(TraceType::argsort_stable)
  );
  m.impl("argsort.stable_out",
         TORCH_FN(TraceType::argsort_out_stable_out)
  );
  m.impl("argsort.dimname",
         TORCH_FN(TraceType::argsort_dimname)
  );
  m.impl("topk.values",
         TORCH_FN(TraceType::topk_out_values)
  );
  m.impl("topk",
         TORCH_FN(TraceType::topk)
  );
  m.impl("unfold_backward",
         TORCH_FN(TraceType::unfold_backward)
  );
  m.impl("normal_",
         TORCH_FN(TraceType::normal_)
  );
  m.impl("normal_functional",
         TORCH_FN(TraceType::normal_functional)
  );
  m.impl("normal.Tensor_float_out",
         TORCH_FN(TraceType::normal_out_Tensor_float_out)
  );
  m.impl("normal.Tensor_float",
         TORCH_FN(TraceType::normal_Tensor_float)
  );
  m.impl("normal.float_Tensor_out",
         TORCH_FN(TraceType::normal_out_float_Tensor_out)
  );
  m.impl("normal.float_Tensor",
         TORCH_FN(TraceType::normal_float_Tensor)
  );
  m.impl("normal.Tensor_Tensor_out",
         TORCH_FN(TraceType::normal_out_Tensor_Tensor_out)
  );
  m.impl("normal.Tensor_Tensor",
         TORCH_FN(TraceType::normal_Tensor_Tensor)
  );
  m.impl("normal.float_float",
         TORCH_FN(TraceType::normal_float_float)
  );
  m.impl("normal.float_float_out",
         TORCH_FN(TraceType::normal_out_float_float_out)
  );
  m.impl("alias",
         TORCH_FN(TraceType::alias)
  );
  m.impl("_foreach_sub.Scalar",
         TORCH_FN(TraceType::_foreach_sub_Scalar)
  );
  m.impl("_foreach_sub_.Scalar",
         TORCH_FN(TraceType::_foreach_sub__Scalar)
  );
  m.impl("_foreach_sub.List",
         TORCH_FN(TraceType::_foreach_sub_List)
  );
  m.impl("_foreach_sub_.List",
         TORCH_FN(TraceType::_foreach_sub__List)
  );
  m.impl("_foreach_sub.ScalarList",
         TORCH_FN(TraceType::_foreach_sub_ScalarList)
  );
  m.impl("_foreach_sub_.ScalarList",
         TORCH_FN(TraceType::_foreach_sub__ScalarList)
  );
  m.impl("_foreach_maximum.Scalar",
         TORCH_FN(TraceType::_foreach_maximum_Scalar)
  );
  m.impl("_foreach_maximum_.Scalar",
         TORCH_FN(TraceType::_foreach_maximum__Scalar)
  );
  m.impl("_foreach_maximum.List",
         TORCH_FN(TraceType::_foreach_maximum_List)
  );
  m.impl("_foreach_maximum_.List",
         TORCH_FN(TraceType::_foreach_maximum__List)
  );
  m.impl("_foreach_maximum.ScalarList",
         TORCH_FN(TraceType::_foreach_maximum_ScalarList)
  );
  m.impl("_foreach_maximum_.ScalarList",
         TORCH_FN(TraceType::_foreach_maximum__ScalarList)
  );
  m.impl("_foreach_acos",
         TORCH_FN(TraceType::_foreach_acos)
  );
  m.impl("_foreach_acos_",
         TORCH_FN(TraceType::_foreach_acos_)
  );
  m.impl("_foreach_atan",
         TORCH_FN(TraceType::_foreach_atan)
  );
  m.impl("_foreach_atan_",
         TORCH_FN(TraceType::_foreach_atan_)
  );
  m.impl("_foreach_ceil",
         TORCH_FN(TraceType::_foreach_ceil)
  );
  m.impl("_foreach_ceil_",
         TORCH_FN(TraceType::_foreach_ceil_)
  );
  m.impl("_foreach_erf",
         TORCH_FN(TraceType::_foreach_erf)
  );
  m.impl("_foreach_erf_",
         TORCH_FN(TraceType::_foreach_erf_)
  );
  m.impl("_foreach_log2",
         TORCH_FN(TraceType::_foreach_log2)
  );
  m.impl("_foreach_log2_",
         TORCH_FN(TraceType::_foreach_log2_)
  );
  m.impl("bucketize.Tensor",
         TORCH_FN(TraceType::bucketize_Tensor)
  );
  m.impl("bucketize.Tensor_out",
         TORCH_FN(TraceType::bucketize_out_Tensor_out)
  );
  m.impl("bucketize.Scalar",
         TORCH_FN(TraceType::bucketize_Scalar)
  );
  m.impl("mse_loss.out",
         TORCH_FN(TraceType::mse_loss_out_out)
  );
  m.impl("mse_loss",
         TORCH_FN(TraceType::mse_loss)
  );
  m.impl("l1_loss",
         TORCH_FN(TraceType::l1_loss)
  );
  m.impl("nll_loss_nd",
         TORCH_FN(TraceType::nll_loss_nd)
  );
  m.impl("nll_loss2d.out",
         TORCH_FN(TraceType::nll_loss2d_out_out)
  );
  m.impl("nll_loss2d",
         TORCH_FN(TraceType::nll_loss2d)
  );
  m.impl("nll_loss2d_forward.output",
         TORCH_FN(TraceType::nll_loss2d_forward_out_output)
  );
  m.impl("nll_loss2d_forward",
         TORCH_FN(TraceType::nll_loss2d_forward)
  );
  m.impl("nll_loss2d_backward.grad_input",
         TORCH_FN(TraceType::nll_loss2d_backward_out_grad_input)
  );
  m.impl("nll_loss2d_backward",
         TORCH_FN(TraceType::nll_loss2d_backward)
  );
  m.impl("soft_margin_loss.out",
         TORCH_FN(TraceType::soft_margin_loss_out_out)
  );
  m.impl("soft_margin_loss",
         TORCH_FN(TraceType::soft_margin_loss)
  );
  m.impl("glu.out",
         TORCH_FN(TraceType::glu_out_out)
  );
  m.impl("glu",
         TORCH_FN(TraceType::glu)
  );
  m.impl("glu_backward_jvp",
         TORCH_FN(TraceType::glu_backward_jvp)
  );
  m.impl("hardtanh.out",
         TORCH_FN(TraceType::hardtanh_out_out)
  );
  m.impl("hardtanh",
         TORCH_FN(TraceType::hardtanh)
  );
  m.impl("hardtanh_",
         TORCH_FN(TraceType::hardtanh_)
  );
  m.impl("hardswish_backward",
         TORCH_FN(TraceType::hardswish_backward)
  );
  m.impl("leaky_relu.out",
         TORCH_FN(TraceType::leaky_relu_out_out)
  );
  m.impl("leaky_relu",
         TORCH_FN(TraceType::leaky_relu)
  );
  m.impl("leaky_relu_",
         TORCH_FN(TraceType::leaky_relu_)
  );
  m.impl("log_sigmoid_forward.output",
         TORCH_FN(TraceType::log_sigmoid_forward_out_output)
  );
  m.impl("log_sigmoid_forward",
         TORCH_FN(TraceType::log_sigmoid_forward)
  );
  m.impl("log_sigmoid_backward.grad_input",
         TORCH_FN(TraceType::log_sigmoid_backward_out_grad_input)
  );
  m.impl("log_sigmoid_backward",
         TORCH_FN(TraceType::log_sigmoid_backward)
  );
  m.impl("softshrink.out",
         TORCH_FN(TraceType::softshrink_out_out)
  );
  m.impl("softshrink",
         TORCH_FN(TraceType::softshrink)
  );
  m.impl("adaptive_avg_pool3d_backward.grad_input",
         TORCH_FN(TraceType::adaptive_avg_pool3d_backward_out_grad_input)
  );
  m.impl("_adaptive_avg_pool3d_backward",
         TORCH_FN(TraceType::_adaptive_avg_pool3d_backward)
  );
  m.impl("adaptive_max_pool2d_backward.grad_input",
         TORCH_FN(TraceType::adaptive_max_pool2d_backward_out_grad_input)
  );
  m.impl("adaptive_max_pool2d_backward",
         TORCH_FN(TraceType::adaptive_max_pool2d_backward)
  );
  m.impl("adaptive_max_pool3d_backward.grad_input",
         TORCH_FN(TraceType::adaptive_max_pool3d_backward_out_grad_input)
  );
  m.impl("adaptive_max_pool3d_backward",
         TORCH_FN(TraceType::adaptive_max_pool3d_backward)
  );
  m.impl("fractional_max_pool3d.output",
         TORCH_FN(TraceType::fractional_max_pool3d_out_output)
  );
  m.impl("fractional_max_pool3d",
         TORCH_FN(TraceType::fractional_max_pool3d)
  );
  m.impl("reflection_pad3d.out",
         TORCH_FN(TraceType::reflection_pad3d_out_out)
  );
  m.impl("reflection_pad3d",
         TORCH_FN(TraceType::reflection_pad3d)
  );
  m.impl("replication_pad1d.out",
         TORCH_FN(TraceType::replication_pad1d_out_out)
  );
  m.impl("replication_pad1d",
         TORCH_FN(TraceType::replication_pad1d)
  );
  m.impl("replication_pad2d.out",
         TORCH_FN(TraceType::replication_pad2d_out_out)
  );
  m.impl("replication_pad2d",
         TORCH_FN(TraceType::replication_pad2d)
  );
  m.impl("_pad_circular",
         TORCH_FN(TraceType::_pad_circular)
  );
  m.impl("pad",
         TORCH_FN(TraceType::pad)
  );
  m.impl("upsample_nearest1d.vec",
         TORCH_FN(TraceType::upsample_nearest1d_vec)
  );
  m.impl("_upsample_nearest_exact1d.vec",
         TORCH_FN(TraceType::_upsample_nearest_exact1d_vec)
  );
  m.impl("upsample_nearest1d.out",
         TORCH_FN(TraceType::upsample_nearest1d_out_out)
  );
  m.impl("_upsample_nearest_exact1d.out",
         TORCH_FN(TraceType::_upsample_nearest_exact1d_out_out)
  );
  m.impl("upsample_nearest1d",
         TORCH_FN(TraceType::upsample_nearest1d)
  );
  m.impl("_upsample_nearest_exact1d",
         TORCH_FN(TraceType::_upsample_nearest_exact1d)
  );
  m.impl("_conv_depthwise2d.out",
         TORCH_FN(TraceType::_conv_depthwise2d_out_out)
  );
  m.impl("_conv_depthwise2d",
         TORCH_FN(TraceType::_conv_depthwise2d)
  );
  m.impl("slow_conv3d.out",
         TORCH_FN(TraceType::slow_conv3d_out_out)
  );
  m.impl("slow_conv3d",
         TORCH_FN(TraceType::slow_conv3d)
  );
  m.impl("_remove_batch_dim",
         TORCH_FN(TraceType::_remove_batch_dim)
  );
  m.impl("special_log_ndtr",
         TORCH_FN(TraceType::special_log_ndtr)
  );
  m.impl("special_log_ndtr.out",
         TORCH_FN(TraceType::special_log_ndtr_out_out)
  );
  m.impl("special_erf",
         TORCH_FN(TraceType::special_erf)
  );
  m.impl("special_erf.out",
         TORCH_FN(TraceType::special_erf_out_out)
  );
  m.impl("special_xlogy",
         TORCH_FN(TraceType::special_xlogy)
  );
  m.impl("special_xlogy.self_scalar",
         TORCH_FN(TraceType::special_xlogy_self_scalar)
  );
  m.impl("special_xlogy.other_scalar",
         TORCH_FN(TraceType::special_xlogy_other_scalar)
  );
  m.impl("special_xlogy.out",
         TORCH_FN(TraceType::special_xlogy_out_out)
  );
  m.impl("special_xlogy.self_scalar_out",
         TORCH_FN(TraceType::special_xlogy_out_self_scalar_out)
  );
  m.impl("special_xlogy.other_scalar_out",
         TORCH_FN(TraceType::special_xlogy_out_other_scalar_out)
  );
  m.impl("special_expit",
         TORCH_FN(TraceType::special_expit)
  );
  m.impl("special_expit.out",
         TORCH_FN(TraceType::special_expit_out_out)
  );
  m.impl("special_sinc",
         TORCH_FN(TraceType::special_sinc)
  );
  m.impl("special_sinc.out",
         TORCH_FN(TraceType::special_sinc_out_out)
  );
  m.impl("special_softmax",
         TORCH_FN(TraceType::special_softmax)
  );
  m.impl("fft_fft",
         TORCH_FN(TraceType::fft_fft)
  );
  m.impl("fft_fft.out",
         TORCH_FN(TraceType::fft_fft_out_out)
  );
  m.impl("fft_rfft",
         TORCH_FN(TraceType::fft_rfft)
  );
  m.impl("fft_rfft.out",
         TORCH_FN(TraceType::fft_rfft_out_out)
  );
  m.impl("fft_hfft2",
         TORCH_FN(TraceType::fft_hfft2)
  );
  m.impl("fft_hfft2.out",
         TORCH_FN(TraceType::fft_hfft2_out_out)
  );
  m.impl("fft_ifftn",
         TORCH_FN(TraceType::fft_ifftn)
  );
  m.impl("fft_ifftn.out",
         TORCH_FN(TraceType::fft_ifftn_out_out)
  );
  m.impl("fft_ihfftn",
         TORCH_FN(TraceType::fft_ihfftn)
  );
  m.impl("fft_ihfftn.out",
         TORCH_FN(TraceType::fft_ihfftn_out_out)
  );
  m.impl("fft_fftfreq",
         TORCH_FN(TraceType::fft_fftfreq)
  );
  m.impl("fft_fftfreq.out",
         TORCH_FN(TraceType::fft_fftfreq_out_out)
  );
  m.impl("fft_rfftfreq",
         TORCH_FN(TraceType::fft_rfftfreq)
  );
  m.impl("fft_rfftfreq.out",
         TORCH_FN(TraceType::fft_rfftfreq_out_out)
  );
  m.impl("linalg_cholesky_ex",
         TORCH_FN(TraceType::linalg_cholesky_ex)
  );
  m.impl("linalg_cholesky_ex.L",
         TORCH_FN(TraceType::linalg_cholesky_ex_out_L)
  );
  m.impl("linalg_cross",
         TORCH_FN(TraceType::linalg_cross)
  );
  m.impl("linalg_cross.out",
         TORCH_FN(TraceType::linalg_cross_out_out)
  );
  m.impl("linalg_lu_factor_ex",
         TORCH_FN(TraceType::linalg_lu_factor_ex)
  );
  m.impl("linalg_lu_factor_ex.out",
         TORCH_FN(TraceType::linalg_lu_factor_ex_out_out)
  );
  m.impl("det",
         TORCH_FN(TraceType::det)
  );
  m.impl("inverse",
         TORCH_FN(TraceType::inverse)
  );
  m.impl("inverse.out",
         TORCH_FN(TraceType::inverse_out_out)
  );
  m.impl("linalg_cond",
         TORCH_FN(TraceType::linalg_cond)
  );
  m.impl("linalg_cond.out",
         TORCH_FN(TraceType::linalg_cond_out_out)
  );
  m.impl("linalg_cond.p_str",
         TORCH_FN(TraceType::linalg_cond_p_str)
  );
  m.impl("linalg_cond.p_str_out",
         TORCH_FN(TraceType::linalg_cond_out_p_str_out)
  );
  m.impl("linalg_pinv.atol_rtol_tensor",
         TORCH_FN(TraceType::linalg_pinv_atol_rtol_tensor)
  );
  m.impl("linalg_pinv.atol_rtol_tensor_out",
         TORCH_FN(TraceType::linalg_pinv_out_atol_rtol_tensor_out)
  );
  m.impl("linalg_pinv.atol_rtol_float",
         TORCH_FN(TraceType::linalg_pinv_atol_rtol_float)
  );
  m.impl("linalg_pinv.atol_rtol_float_out",
         TORCH_FN(TraceType::linalg_pinv_out_atol_rtol_float_out)
  );
  m.impl("linalg_pinv",
         TORCH_FN(TraceType::linalg_pinv)
  );
  m.impl("linalg_pinv.rcond_tensor",
         TORCH_FN(TraceType::linalg_pinv_rcond_tensor)
  );
  m.impl("linalg_pinv.out",
         TORCH_FN(TraceType::linalg_pinv_out_out)
  );
  m.impl("linalg_pinv.out_rcond_tensor",
         TORCH_FN(TraceType::linalg_pinv_out_out_rcond_tensor)
  );
  m.impl("linalg_solve_ex",
         TORCH_FN(TraceType::linalg_solve_ex)
  );
  m.impl("linalg_solve_ex.out",
         TORCH_FN(TraceType::linalg_solve_ex_out_out)
  );
  m.impl("_spsolve",
         TORCH_FN(TraceType::_spsolve)
  );
  m.impl("linalg_tensorsolve",
         TORCH_FN(TraceType::linalg_tensorsolve)
  );
  m.impl("linalg_tensorsolve.out",
         TORCH_FN(TraceType::linalg_tensorsolve_out_out)
  );
  m.impl("linalg_multi_dot",
         TORCH_FN(TraceType::linalg_multi_dot)
  );
  m.impl("linalg_multi_dot.out",
         TORCH_FN(TraceType::linalg_multi_dot_out_out)
  );
  m.impl("_test_parallel_materialize",
         TORCH_FN(TraceType::_test_parallel_materialize)
  );
  m.impl("_test_string_default",
         TORCH_FN(TraceType::_test_string_default)
  );
  m.impl("flatten_dense_tensors",
         TORCH_FN(TraceType::flatten_dense_tensors)
  );
  m.impl("_conj_copy",
         TORCH_FN(TraceType::_conj_copy)
  );
  m.impl("detach_copy",
         TORCH_FN(TraceType::detach_copy)
  );
  m.impl("row_indices_copy",
         TORCH_FN(TraceType::row_indices_copy)
  );
  m.impl("_transformer_encoder_layer_fwd",
         TORCH_FN(TraceType::_transformer_encoder_layer_fwd)
  );
  m.impl("_native_multi_head_attention",
         TORCH_FN(TraceType::_native_multi_head_attention)
  );
  m.impl("_fused_sdp_choice",
         TORCH_FN(TraceType::_fused_sdp_choice)
  );
  m.impl("_scaled_dot_product_flash_attention",
         TORCH_FN(TraceType::_scaled_dot_product_flash_attention)
  );
  m.impl("_scaled_dot_product_flash_attention_for_cpu",
         TORCH_FN(TraceType::_scaled_dot_product_flash_attention_for_cpu)
  );
  m.impl("_scaled_dot_product_efficient_attention_backward",
         TORCH_FN(TraceType::_scaled_dot_product_efficient_attention_backward)
  );
  m.impl("_flash_attention_backward",
         TORCH_FN(TraceType::_flash_attention_backward)
  );
  m.impl("_efficient_attention_backward",
         TORCH_FN(TraceType::_efficient_attention_backward)
  );
  m.impl("_fill_mem_eff_dropout_mask_",
         TORCH_FN(TraceType::_fill_mem_eff_dropout_mask_)
  );
  m.impl("_triton_multi_head_attention",
         TORCH_FN(TraceType::_triton_multi_head_attention)
  );
  m.impl("special_airy_ai",
         TORCH_FN(TraceType::special_airy_ai)
  );
  m.impl("special_airy_ai.out",
         TORCH_FN(TraceType::special_airy_ai_out_out)
  );
  m.impl("special_chebyshev_polynomial_w",
         TORCH_FN(TraceType::special_chebyshev_polynomial_w)
  );
  m.impl("special_chebyshev_polynomial_w.x_scalar",
         TORCH_FN(TraceType::special_chebyshev_polynomial_w_x_scalar)
  );
  m.impl("special_chebyshev_polynomial_w.n_scalar",
         TORCH_FN(TraceType::special_chebyshev_polynomial_w_n_scalar)
  );
  m.impl("special_chebyshev_polynomial_w.out",
         TORCH_FN(TraceType::special_chebyshev_polynomial_w_out_out)
  );
  m.impl("special_chebyshev_polynomial_w.x_scalar_out",
         TORCH_FN(TraceType::special_chebyshev_polynomial_w_out_x_scalar_out)
  );
  m.impl("special_chebyshev_polynomial_w.n_scalar_out",
         TORCH_FN(TraceType::special_chebyshev_polynomial_w_out_n_scalar_out)
  );
  m.impl("special_hermite_polynomial_h",
         TORCH_FN(TraceType::special_hermite_polynomial_h)
  );
  m.impl("special_hermite_polynomial_h.x_scalar",
         TORCH_FN(TraceType::special_hermite_polynomial_h_x_scalar)
  );
  m.impl("special_hermite_polynomial_h.n_scalar",
         TORCH_FN(TraceType::special_hermite_polynomial_h_n_scalar)
  );
  m.impl("special_hermite_polynomial_h.out",
         TORCH_FN(TraceType::special_hermite_polynomial_h_out_out)
  );
  m.impl("special_hermite_polynomial_h.x_scalar_out",
         TORCH_FN(TraceType::special_hermite_polynomial_h_out_x_scalar_out)
  );
  m.impl("special_hermite_polynomial_h.n_scalar_out",
         TORCH_FN(TraceType::special_hermite_polynomial_h_out_n_scalar_out)
  );
  m.impl("special_modified_bessel_i0",
         TORCH_FN(TraceType::special_modified_bessel_i0)
  );
  m.impl("special_modified_bessel_i0.out",
         TORCH_FN(TraceType::special_modified_bessel_i0_out_out)
  );
  m.impl("special_modified_bessel_k0",
         TORCH_FN(TraceType::special_modified_bessel_k0)
  );
  m.impl("special_modified_bessel_k0.out",
         TORCH_FN(TraceType::special_modified_bessel_k0_out_out)
  );
  m.impl("special_modified_bessel_k1",
         TORCH_FN(TraceType::special_modified_bessel_k1)
  );
  m.impl("special_modified_bessel_k1.out",
         TORCH_FN(TraceType::special_modified_bessel_k1_out_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_t",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_t)
  );
  m.impl("special_shifted_chebyshev_polynomial_t.x_scalar",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_t_x_scalar)
  );
  m.impl("special_shifted_chebyshev_polynomial_t.n_scalar",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_t_n_scalar)
  );
  m.impl("special_shifted_chebyshev_polynomial_t.out",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_t_out_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_t.x_scalar_out",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_t_out_x_scalar_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_t.n_scalar_out",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_t_out_n_scalar_out)
  );
  m.impl("_fused_adamw_",
         TORCH_FN(TraceType::_fused_adamw_)
  );
  m.impl("_fused_adamw_.tensor_lr",
         TORCH_FN(TraceType::_fused_adamw__tensor_lr)
  );
  m.impl("_cudnn_rnn_flatten_weight.out",
         TORCH_FN(TraceType::_cudnn_rnn_flatten_weight_out_out)
  );
  m.impl("quantized_batch_norm.out",
         TORCH_FN(TraceType::quantized_batch_norm_out_out)
  );
  m.impl("conv_tbc.out",
         TORCH_FN(TraceType::conv_tbc_out_out)
  );
  m.impl("cudnn_affine_grid_generator_backward.out",
         TORCH_FN(TraceType::cudnn_affine_grid_generator_backward_out_out)
  );
  m.impl("cudnn_grid_sampler.out",
         TORCH_FN(TraceType::cudnn_grid_sampler_out_out)
  );
  m.impl("div.Scalar_out",
         TORCH_FN(TraceType::div_out_Scalar_out)
  );
  m.impl("div.Scalar_mode_out",
         TORCH_FN(TraceType::div_out_Scalar_mode_out)
  );
  m.impl("_embedding_bag_forward_only.out",
         TORCH_FN(TraceType::_embedding_bag_forward_only_out_out)
  );
  m.impl("new_zeros.out",
         TORCH_FN(TraceType::new_zeros_out_out)
  );
  m.impl("_grid_sampler_2d_cpu_fallback.out",
         TORCH_FN(TraceType::_grid_sampler_2d_cpu_fallback_out_out)
  );
  m.impl("grid_sampler_3d.out",
         TORCH_FN(TraceType::grid_sampler_3d_out_out)
  );
  m.impl("hann_window.out",
         TORCH_FN(TraceType::hann_window_out_out)
  );
  m.impl("hann_window.periodic_out",
         TORCH_FN(TraceType::hann_window_out_periodic_out)
  );
  m.impl("hamming_window.out",
         TORCH_FN(TraceType::hamming_window_out_out)
  );
  m.impl("hamming_window.periodic_out",
         TORCH_FN(TraceType::hamming_window_out_periodic_out)
  );
  m.impl("hamming_window.periodic_alpha_out",
         TORCH_FN(TraceType::hamming_window_out_periodic_alpha_out)
  );
  m.impl("hamming_window.periodic_alpha_beta_out",
         TORCH_FN(TraceType::hamming_window_out_periodic_alpha_beta_out)
  );
  m.impl("native_group_norm_backward.out",
         TORCH_FN(TraceType::native_group_norm_backward_out_out)
  );
  m.impl("isnan.out",
         TORCH_FN(TraceType::isnan_out_out)
  );
  m.impl("native_layer_norm.out",
         TORCH_FN(TraceType::native_layer_norm_out_out)
  );
  m.impl("mkldnn_max_pool2d.out",
         TORCH_FN(TraceType::mkldnn_max_pool2d_out_out)
  );
  m.impl("quantized_max_pool2d.out",
         TORCH_FN(TraceType::quantized_max_pool2d_out_out)
  );
  m.impl("_mps_convolution.out",
         TORCH_FN(TraceType::_mps_convolution_out_out)
  );
  m.impl("mkldnn_rnn_layer_backward.out",
         TORCH_FN(TraceType::mkldnn_rnn_layer_backward_out_out)
  );
  m.impl("miopen_depthwise_convolution.out",
         TORCH_FN(TraceType::miopen_depthwise_convolution_out_out)
  );
  m.impl("batch_norm_stats.out",
         TORCH_FN(TraceType::batch_norm_stats_out_out)
  );
  m.impl("batch_norm_gather_stats.out",
         TORCH_FN(TraceType::batch_norm_gather_stats_out_out)
  );
  m.impl("native_batch_norm_backward.out",
         TORCH_FN(TraceType::native_batch_norm_backward_out_out)
  );
  m.impl("batch_norm_backward_reduce.out",
         TORCH_FN(TraceType::batch_norm_backward_reduce_out_out)
  );
  m.impl("_nnpack_spatial_convolution.out",
         TORCH_FN(TraceType::_nnpack_spatial_convolution_out_out)
  );
  m.impl("ones.names_out",
         TORCH_FN(TraceType::ones_out_names_out)
  );
  m.impl("_cdist_forward.out",
         TORCH_FN(TraceType::_cdist_forward_out_out)
  );
  m.impl("rand_like.out",
         TORCH_FN(TraceType::rand_like_out_out)
  );
  m.impl("randint_like.out",
         TORCH_FN(TraceType::randint_like_out_out)
  );
  m.impl("randint_like.low_dtype_out",
         TORCH_FN(TraceType::randint_like_out_low_dtype_out)
  );
  m.impl("select_backward.out",
         TORCH_FN(TraceType::select_backward_out_out)
  );
  m.impl("slice_scatter.out",
         TORCH_FN(TraceType::slice_scatter_out_out)
  );
  m.impl("_mkldnn_transpose.out",
         TORCH_FN(TraceType::_mkldnn_transpose_out_out)
  );
  m.impl("_nested_from_padded_and_nested_example.out",
         TORCH_FN(TraceType::_nested_from_padded_and_nested_example_out_out)
  );
  m.impl("unique_dim.out",
         TORCH_FN(TraceType::unique_dim_out_out)
  );
  m.impl("unique_consecutive.out",
         TORCH_FN(TraceType::unique_consecutive_out_out)
  );
  m.impl("_dirichlet_grad.out",
         TORCH_FN(TraceType::_dirichlet_grad_out_out)
  );
  m.impl("clone.out",
         TORCH_FN(TraceType::clone_out_out)
  );
  m.impl("resize_as.out",
         TORCH_FN(TraceType::resize_as_out_out)
  );
  m.impl("resize_as",
         TORCH_FN(TraceType::resize_as)
  );
  m.impl("resize_as_sparse.out",
         TORCH_FN(TraceType::resize_as_sparse_out_out)
  );
  m.impl("resize_as_sparse",
         TORCH_FN(TraceType::resize_as_sparse)
  );
  m.impl("_to_sparse_csc.out",
         TORCH_FN(TraceType::_to_sparse_csc_out_out)
  );
  m.impl("_to_sparse_bsc.out",
         TORCH_FN(TraceType::_to_sparse_bsc_out_out)
  );
  m.impl("mkldnn_reorder_conv2d_weight.out",
         TORCH_FN(TraceType::mkldnn_reorder_conv2d_weight_out_out)
  );
  m.impl("quantize_per_channel.out",
         TORCH_FN(TraceType::quantize_per_channel_out_out)
  );
  m.impl("dequantize.self_out",
         TORCH_FN(TraceType::dequantize_out_self_out)
  );
  m.impl("dequantize.tensors_out",
         TORCH_FN(TraceType::dequantize_out_tensors_out)
  );
  m.impl("q_per_channel_zero_points.out",
         TORCH_FN(TraceType::q_per_channel_zero_points_out_out)
  );
  m.impl("_fake_quantize_learnable_per_channel_affine.out",
         TORCH_FN(TraceType::_fake_quantize_learnable_per_channel_affine_out_out)
  );
  m.impl("_lstm_mps.out",
         TORCH_FN(TraceType::_lstm_mps_out_out)
  );
  m.impl("_thnn_fused_lstm_cell.out",
         TORCH_FN(TraceType::_thnn_fused_lstm_cell_out_out)
  );
  m.impl("lift_fresh_copy.out",
         TORCH_FN(TraceType::lift_fresh_copy_out_out)
  );
  m.impl("index_fill.int_Scalar_out",
         TORCH_FN(TraceType::index_fill_out_int_Scalar_out)
  );
  m.impl("index_fill.int_Tensor_out",
         TORCH_FN(TraceType::index_fill_out_int_Tensor_out)
  );
  m.impl("random.from_out",
         TORCH_FN(TraceType::random_out_from_out)
  );
  m.impl("random.from",
         TORCH_FN(TraceType::random_from)
  );
  m.impl("random.to_out",
         TORCH_FN(TraceType::random_out_to_out)
  );
  m.impl("random.to",
         TORCH_FN(TraceType::random_to)
  );
  m.impl("random.out",
         TORCH_FN(TraceType::random_out_out)
  );
  m.impl("random",
         TORCH_FN(TraceType::random)
  );
  m.impl("cauchy.out",
         TORCH_FN(TraceType::cauchy_out_out)
  );
  m.impl("cauchy",
         TORCH_FN(TraceType::cauchy)
  );
  m.impl("log_normal.out",
         TORCH_FN(TraceType::log_normal_out_out)
  );
  m.impl("log_normal",
         TORCH_FN(TraceType::log_normal)
  );
  m.impl("_histogramdd_bin_edges.out",
         TORCH_FN(TraceType::_histogramdd_bin_edges_out_out)
  );
  m.impl("_histogramdd_from_bin_tensors.out",
         TORCH_FN(TraceType::_histogramdd_from_bin_tensors_out_out)
  );
  m.impl("unfold_backward.out",
         TORCH_FN(TraceType::unfold_backward_out_out)
  );
  m.impl("normal.out",
         TORCH_FN(TraceType::normal_out_out)
  );
  m.impl("_foreach_sub.Scalar_out",
         TORCH_FN(TraceType::_foreach_sub_out_Scalar_out)
  );
  m.impl("_foreach_sub.List_out",
         TORCH_FN(TraceType::_foreach_sub_out_List_out)
  );
  m.impl("_foreach_sub.ScalarList_out",
         TORCH_FN(TraceType::_foreach_sub_out_ScalarList_out)
  );
  m.impl("_foreach_maximum.Scalar_out",
         TORCH_FN(TraceType::_foreach_maximum_out_Scalar_out)
  );
  m.impl("_foreach_maximum.List_out",
         TORCH_FN(TraceType::_foreach_maximum_out_List_out)
  );
  m.impl("_foreach_maximum.ScalarList_out",
         TORCH_FN(TraceType::_foreach_maximum_out_ScalarList_out)
  );
  m.impl("_foreach_acos.out",
         TORCH_FN(TraceType::_foreach_acos_out_out)
  );
  m.impl("_foreach_atan.out",
         TORCH_FN(TraceType::_foreach_atan_out_out)
  );
  m.impl("_foreach_ceil.out",
         TORCH_FN(TraceType::_foreach_ceil_out_out)
  );
  m.impl("_foreach_erf.out",
         TORCH_FN(TraceType::_foreach_erf_out_out)
  );
  m.impl("_foreach_log2.out",
         TORCH_FN(TraceType::_foreach_log2_out_out)
  );
  m.impl("bucketize.Scalar_out",
         TORCH_FN(TraceType::bucketize_out_Scalar_out)
  );
  m.impl("glu_backward_jvp.out",
         TORCH_FN(TraceType::glu_backward_jvp_out_out)
  );
  m.impl("hardswish_backward.out",
         TORCH_FN(TraceType::hardswish_backward_out_out)
  );
  m.impl("_adaptive_avg_pool3d_backward.out",
         TORCH_FN(TraceType::_adaptive_avg_pool3d_backward_out_out)
  );
  m.impl("_conj_copy.out",
         TORCH_FN(TraceType::_conj_copy_out_out)
  );
  m.impl("detach_copy.out",
         TORCH_FN(TraceType::detach_copy_out_out)
  );
  m.impl("row_indices_copy.out",
         TORCH_FN(TraceType::row_indices_copy_out_out)
  );
  m.impl("_transformer_encoder_layer_fwd.out",
         TORCH_FN(TraceType::_transformer_encoder_layer_fwd_out_out)
  );
  m.impl("_native_multi_head_attention.out",
         TORCH_FN(TraceType::_native_multi_head_attention_out_out)
  );
  m.impl("_triton_multi_head_attention.out",
         TORCH_FN(TraceType::_triton_multi_head_attention_out_out)
  );
  m.impl("_fused_adamw.out",
         TORCH_FN(TraceType::_fused_adamw_out_out)
  );
  m.impl("_fused_adamw",
         TORCH_FN(TraceType::_fused_adamw)
  );
  m.impl("_fused_adamw.tensor_lr_out",
         TORCH_FN(TraceType::_fused_adamw_out_tensor_lr_out)
  );
  m.impl("_fused_adamw.tensor_lr",
         TORCH_FN(TraceType::_fused_adamw_tensor_lr)
  );;
}

}  // namespace

} // namespace torch
