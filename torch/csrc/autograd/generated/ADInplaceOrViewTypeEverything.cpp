#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include "torch/csrc/autograd/VariableTypeUtils.h"
#include "torch/csrc/autograd/generated/ViewFuncs.h"

#include <torch/library.h>
#include <ATen/FunctionalInverses.h>
#include <ATen/FunctionalTensorWrapper.h>

// @generated from ../tools/autograd/templates/ADInplaceOrViewType.cpp

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>
#else
#include <ATen/ops/lshift_ops.h>
#include <ATen/ops/lshift_ops.h>
#include <ATen/ops/rshift_ops.h>
#include <ATen/ops/rshift_ops.h>
#include <ATen/ops/lshift_ops.h>
#include <ATen/ops/lshift_ops.h>
#include <ATen/ops/rshift_ops.h>
#include <ATen/ops/rshift_ops.h>
#include <ATen/ops/_adaptive_avg_pool2d_backward_ops.h>
#include <ATen/ops/_adaptive_avg_pool2d_ops.h>
#include <ATen/ops/_adaptive_avg_pool3d_backward_ops.h>
#include <ATen/ops/_adaptive_avg_pool3d_ops.h>
#include <ATen/ops/_add_relu_ops.h>
#include <ATen/ops/_add_relu_ops.h>
#include <ATen/ops/_add_relu_ops.h>
#include <ATen/ops/_add_relu_ops.h>
#include <ATen/ops/_addmm_activation_ops.h>
#include <ATen/ops/_aminmax_ops.h>
#include <ATen/ops/_aminmax_ops.h>
#include <ATen/ops/_amp_update_scale_ops.h>
#include <ATen/ops/_amp_update_scale_ops.h>
#include <ATen/ops/_batch_norm_no_update_ops.h>
#include <ATen/ops/_batch_norm_with_update_ops.h>
#include <ATen/ops/_cdist_backward_ops.h>
#include <ATen/ops/_cdist_forward_ops.h>
#include <ATen/ops/_cholesky_solve_helper_ops.h>
#include <ATen/ops/_chunk_cat_ops.h>
#include <ATen/ops/_coalesce_ops.h>
#include <ATen/ops/_coalesced_ops.h>
#include <ATen/ops/_coalesced_ops.h>
#include <ATen/ops/_compute_linear_combination_ops.h>
#include <ATen/ops/_conj_ops.h>
#include <ATen/ops/_conj_copy_ops.h>
#include <ATen/ops/_conj_physical_ops.h>
#include <ATen/ops/_conv_depthwise2d_ops.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr_ops.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo_ops.h>
#include <ATen/ops/_convolution_ops.h>
#include <ATen/ops/_copy_from_and_resize_ops.h>
#include <ATen/ops/_copy_from_ops.h>
#include <ATen/ops/_ctc_loss_backward_ops.h>
#include <ATen/ops/_ctc_loss_ops.h>
#include <ATen/ops/_ctc_loss_ops.h>
#include <ATen/ops/_cudnn_ctc_loss_ops.h>
#include <ATen/ops/_cudnn_init_dropout_state_ops.h>
#include <ATen/ops/_cudnn_rnn_flatten_weight_ops.h>
#include <ATen/ops/_cudnn_rnn_ops.h>
#include <ATen/ops/_dirichlet_grad_ops.h>
#include <ATen/ops/_efficientzerotensor_ops.h>
#include <ATen/ops/_embedding_bag_dense_backward_ops.h>
#include <ATen/ops/_embedding_bag_forward_only_ops.h>
#include <ATen/ops/_embedding_bag_ops.h>
#include <ATen/ops/_embedding_bag_per_sample_weights_backward_ops.h>
#include <ATen/ops/_empty_affine_quantized_ops.h>
#include <ATen/ops/_empty_per_channel_affine_quantized_ops.h>
#include <ATen/ops/_euclidean_dist_ops.h>
#include <ATen/ops/_fake_quantize_learnable_per_channel_affine_ops.h>
#include <ATen/ops/_fake_quantize_learnable_per_tensor_affine_ops.h>
#include <ATen/ops/_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_ops.h>
#include <ATen/ops/_fft_c2c_ops.h>
#include <ATen/ops/_fft_c2r_ops.h>
#include <ATen/ops/_fft_r2c_ops.h>
#include <ATen/ops/_fill_mem_eff_dropout_mask_ops.h>
#include <ATen/ops/_foobar_ops.h>
#include <ATen/ops/_fused_dropout_ops.h>
#include <ATen/ops/_fused_moving_avg_obs_fq_helper_ops.h>
#include <ATen/ops/_fw_primal_copy_ops.h>
#include <ATen/ops/_grid_sampler_2d_cpu_fallback_ops.h>
#include <ATen/ops/_histogramdd_from_bin_cts_ops.h>
#include <ATen/ops/_histogramdd_from_bin_tensors_ops.h>
#include <ATen/ops/_index_put_impl_ops.h>
#include <ATen/ops/_index_put_impl_ops.h>
#include <ATen/ops/_indices_ops.h>
#include <ATen/ops/_indices_copy_ops.h>
#include <ATen/ops/_int_mm_ops.h>
#include <ATen/ops/_linalg_det_ops.h>
#include <ATen/ops/_linalg_eigh_ops.h>
#include <ATen/ops/_linalg_slogdet_ops.h>
#include <ATen/ops/_linalg_solve_ex_ops.h>
#include <ATen/ops/_linalg_svd_ops.h>
#include <ATen/ops/_log_softmax_backward_data_ops.h>
#include <ATen/ops/_log_softmax_ops.h>
#include <ATen/ops/_logcumsumexp_ops.h>
#include <ATen/ops/_lstm_mps_ops.h>
#include <ATen/ops/_make_dual_copy_ops.h>
#include <ATen/ops/_make_per_channel_quantized_tensor_ops.h>
#include <ATen/ops/_make_per_tensor_quantized_tensor_ops.h>
#include <ATen/ops/_masked_scale_ops.h>
#include <ATen/ops/_masked_softmax_backward_ops.h>
#include <ATen/ops/_masked_softmax_ops.h>
#include <ATen/ops/_mkldnn_reshape_ops.h>
#include <ATen/ops/_mkldnn_transpose_ops.h>
#include <ATen/ops/_mkldnn_transpose_ops.h>
#include <ATen/ops/_mps_convolution_ops.h>
#include <ATen/ops/_mps_convolution_transpose_ops.h>
#include <ATen/ops/_native_batch_norm_legit_no_training_ops.h>
#include <ATen/ops/_native_batch_norm_legit_ops.h>
#include <ATen/ops/_native_batch_norm_legit_ops.h>
#include <ATen/ops/_native_multi_head_attention_ops.h>
#include <ATen/ops/_neg_view_ops.h>
#include <ATen/ops/_neg_view_copy_ops.h>
#include <ATen/ops/_nested_from_padded_and_nested_example_ops.h>
#include <ATen/ops/_nested_from_padded_ops.h>
#include <ATen/ops/_nested_get_values_ops.h>
#include <ATen/ops/_nested_get_values_copy_ops.h>
#include <ATen/ops/_nested_tensor_from_mask_ops.h>
#include <ATen/ops/_nested_tensor_from_tensor_list_ops.h>
#include <ATen/ops/_nested_tensor_size_ops.h>
#include <ATen/ops/_nested_tensor_storage_offsets_ops.h>
#include <ATen/ops/_nested_tensor_strides_ops.h>
#include <ATen/ops/_nested_view_from_buffer_ops.h>
#include <ATen/ops/_nested_view_from_buffer_copy_ops.h>
#include <ATen/ops/_nested_view_from_jagged_ops.h>
#include <ATen/ops/_nested_view_from_jagged_copy_ops.h>
#include <ATen/ops/_new_zeros_with_same_feature_meta_ops.h>
#include <ATen/ops/_nnpack_spatial_convolution_ops.h>
#include <ATen/ops/_pack_padded_sequence_ops.h>
#include <ATen/ops/_pdist_backward_ops.h>
#include <ATen/ops/_pdist_forward_ops.h>
#include <ATen/ops/_pin_memory_ops.h>
#include <ATen/ops/_reshape_alias_ops.h>
#include <ATen/ops/_reshape_alias_copy_ops.h>
#include <ATen/ops/_resize_output_ops.h>
#include <ATen/ops/_resize_output_ops.h>
#include <ATen/ops/_sample_dirichlet_ops.h>
#include <ATen/ops/_scaled_mm_ops.h>
#include <ATen/ops/_segment_reduce_backward_ops.h>
#include <ATen/ops/_slow_conv2d_backward_ops.h>
#include <ATen/ops/_slow_conv2d_backward_ops.h>
#include <ATen/ops/_slow_conv2d_forward_ops.h>
#include <ATen/ops/_softmax_backward_data_ops.h>
#include <ATen/ops/_softmax_ops.h>
#include <ATen/ops/_sparse_addmm_ops.h>
#include <ATen/ops/_sparse_broadcast_to_copy_ops.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors_ops.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_ops.h>
#include <ATen/ops/_sparse_csr_prod_ops.h>
#include <ATen/ops/_sparse_csr_sum_ops.h>
#include <ATen/ops/_sparse_log_softmax_backward_data_ops.h>
#include <ATen/ops/_sparse_log_softmax_ops.h>
#include <ATen/ops/_sparse_mask_projection_ops.h>
#include <ATen/ops/_sparse_softmax_backward_data_ops.h>
#include <ATen/ops/_sparse_softmax_ops.h>
#include <ATen/ops/_sparse_sparse_matmul_ops.h>
#include <ATen/ops/_sparse_sum_backward_ops.h>
#include <ATen/ops/_sparse_sum_ops.h>
#include <ATen/ops/_spdiags_ops.h>
#include <ATen/ops/_stack_ops.h>
#include <ATen/ops/_standard_gamma_grad_ops.h>
#include <ATen/ops/_standard_gamma_ops.h>
#include <ATen/ops/_test_autograd_multiple_dispatch_ops.h>
#include <ATen/ops/_test_autograd_multiple_dispatch_view_ops.h>
#include <ATen/ops/_test_autograd_multiple_dispatch_view_copy_ops.h>
#include <ATen/ops/_test_functorch_fallback_ops.h>
#include <ATen/ops/_test_optional_filled_intlist_ops.h>
#include <ATen/ops/_test_optional_floatlist_ops.h>
#include <ATen/ops/_test_optional_intlist_ops.h>
#include <ATen/ops/_test_warn_in_autograd_ops.h>
#include <ATen/ops/_thnn_fused_gru_cell_backward_ops.h>
#include <ATen/ops/_thnn_fused_gru_cell_ops.h>
#include <ATen/ops/_thnn_fused_lstm_cell_backward_impl_ops.h>
#include <ATen/ops/_thnn_fused_lstm_cell_ops.h>
#include <ATen/ops/_to_copy_ops.h>
#include <ATen/ops/_to_dense_ops.h>
#include <ATen/ops/_to_sparse_bsc_ops.h>
#include <ATen/ops/_to_sparse_bsr_ops.h>
#include <ATen/ops/_to_sparse_csc_ops.h>
#include <ATen/ops/_to_sparse_csr_ops.h>
#include <ATen/ops/_to_sparse_ops.h>
#include <ATen/ops/_to_sparse_ops.h>
#include <ATen/ops/_transform_bias_rescale_qkv_ops.h>
#include <ATen/ops/_transformer_encoder_layer_fwd_ops.h>
#include <ATen/ops/_trilinear_ops.h>
#include <ATen/ops/_triton_multi_head_attention_ops.h>
#include <ATen/ops/_triton_scaled_dot_attention_ops.h>
#include <ATen/ops/_unique2_ops.h>
#include <ATen/ops/_unique_ops.h>
#include <ATen/ops/_unsafe_view_ops.h>
#include <ATen/ops/_upsample_bicubic2d_aa_backward_ops.h>
#include <ATen/ops/_upsample_bicubic2d_aa_ops.h>
#include <ATen/ops/_upsample_bilinear2d_aa_backward_ops.h>
#include <ATen/ops/_upsample_bilinear2d_aa_ops.h>
#include <ATen/ops/_upsample_nearest_exact1d_backward_ops.h>
#include <ATen/ops/_upsample_nearest_exact1d_ops.h>
#include <ATen/ops/_upsample_nearest_exact2d_backward_ops.h>
#include <ATen/ops/_upsample_nearest_exact2d_ops.h>
#include <ATen/ops/_upsample_nearest_exact3d_backward_ops.h>
#include <ATen/ops/_upsample_nearest_exact3d_ops.h>
#include <ATen/ops/_values_ops.h>
#include <ATen/ops/_values_copy_ops.h>
#include <ATen/ops/_weight_norm_interface_backward_ops.h>
#include <ATen/ops/_weight_norm_interface_ops.h>
#include <ATen/ops/abs_ops.h>
#include <ATen/ops/abs_ops.h>
#include <ATen/ops/acos_ops.h>
#include <ATen/ops/acos_ops.h>
#include <ATen/ops/acosh_ops.h>
#include <ATen/ops/acosh_ops.h>
#include <ATen/ops/adaptive_avg_pool2d_ops.h>
#include <ATen/ops/adaptive_avg_pool3d_backward_ops.h>
#include <ATen/ops/adaptive_avg_pool3d_ops.h>
#include <ATen/ops/adaptive_max_pool2d_backward_ops.h>
#include <ATen/ops/adaptive_max_pool2d_ops.h>
#include <ATen/ops/adaptive_max_pool3d_backward_ops.h>
#include <ATen/ops/adaptive_max_pool3d_ops.h>
#include <ATen/ops/add_ops.h>
#include <ATen/ops/add_ops.h>
#include <ATen/ops/add_ops.h>
#include <ATen/ops/add_ops.h>
#include <ATen/ops/addbmm_ops.h>
#include <ATen/ops/addbmm_ops.h>
#include <ATen/ops/addcdiv_ops.h>
#include <ATen/ops/addcdiv_ops.h>
#include <ATen/ops/addcmul_ops.h>
#include <ATen/ops/addcmul_ops.h>
#include <ATen/ops/addmm_ops.h>
#include <ATen/ops/addmm_ops.h>
#include <ATen/ops/addmv_ops.h>
#include <ATen/ops/addmv_ops.h>
#include <ATen/ops/addr_ops.h>
#include <ATen/ops/addr_ops.h>
#include <ATen/ops/affine_grid_generator_ops.h>
#include <ATen/ops/alias_ops.h>
#include <ATen/ops/alias_copy_ops.h>
#include <ATen/ops/all_ops.h>
#include <ATen/ops/all_ops.h>
#include <ATen/ops/all_ops.h>
#include <ATen/ops/amax_ops.h>
#include <ATen/ops/amin_ops.h>
#include <ATen/ops/aminmax_ops.h>
#include <ATen/ops/angle_ops.h>
#include <ATen/ops/any_ops.h>
#include <ATen/ops/any_ops.h>
#include <ATen/ops/any_ops.h>
#include <ATen/ops/arange_ops.h>
#include <ATen/ops/arange_ops.h>
#include <ATen/ops/argmax_ops.h>
#include <ATen/ops/argmin_ops.h>
#include <ATen/ops/as_strided_ops.h>
#include <ATen/ops/as_strided_ops.h>
#include <ATen/ops/as_strided_copy_ops.h>
#include <ATen/ops/as_strided_scatter_ops.h>
#include <ATen/ops/asin_ops.h>
#include <ATen/ops/asin_ops.h>
#include <ATen/ops/asinh_ops.h>
#include <ATen/ops/asinh_ops.h>
#include <ATen/ops/atan2_ops.h>
#include <ATen/ops/atan2_ops.h>
#include <ATen/ops/atan_ops.h>
#include <ATen/ops/atan_ops.h>
#include <ATen/ops/atanh_ops.h>
#include <ATen/ops/atanh_ops.h>
#include <ATen/ops/avg_pool2d_backward_ops.h>
#include <ATen/ops/avg_pool2d_ops.h>
#include <ATen/ops/avg_pool3d_backward_ops.h>
#include <ATen/ops/avg_pool3d_ops.h>
#include <ATen/ops/baddbmm_ops.h>
#include <ATen/ops/baddbmm_ops.h>
#include <ATen/ops/bartlett_window_ops.h>
#include <ATen/ops/bartlett_window_ops.h>
#include <ATen/ops/batch_norm_backward_elemt_ops.h>
#include <ATen/ops/batch_norm_backward_reduce_ops.h>
#include <ATen/ops/batch_norm_elemt_ops.h>
#include <ATen/ops/batch_norm_gather_stats_ops.h>
#include <ATen/ops/batch_norm_gather_stats_with_counts_ops.h>
#include <ATen/ops/batch_norm_stats_ops.h>
#include <ATen/ops/batch_norm_update_stats_ops.h>
#include <ATen/ops/bernoulli_ops.h>
#include <ATen/ops/bernoulli_ops.h>
#include <ATen/ops/bernoulli_ops.h>
#include <ATen/ops/bernoulli_ops.h>
#include <ATen/ops/bernoulli_ops.h>
#include <ATen/ops/binary_cross_entropy_backward_ops.h>
#include <ATen/ops/binary_cross_entropy_ops.h>
#include <ATen/ops/binary_cross_entropy_with_logits_ops.h>
#include <ATen/ops/bincount_ops.h>
#include <ATen/ops/binomial_ops.h>
#include <ATen/ops/bitwise_and_ops.h>
#include <ATen/ops/bitwise_and_ops.h>
#include <ATen/ops/bitwise_and_ops.h>
#include <ATen/ops/bitwise_and_ops.h>
#include <ATen/ops/bitwise_and_ops.h>
#include <ATen/ops/bitwise_left_shift_ops.h>
#include <ATen/ops/bitwise_left_shift_ops.h>
#include <ATen/ops/bitwise_left_shift_ops.h>
#include <ATen/ops/bitwise_left_shift_ops.h>
#include <ATen/ops/bitwise_left_shift_ops.h>
#include <ATen/ops/bitwise_not_ops.h>
#include <ATen/ops/bitwise_not_ops.h>
#include <ATen/ops/bitwise_or_ops.h>
#include <ATen/ops/bitwise_or_ops.h>
#include <ATen/ops/bitwise_or_ops.h>
#include <ATen/ops/bitwise_or_ops.h>
#include <ATen/ops/bitwise_or_ops.h>
#include <ATen/ops/bitwise_right_shift_ops.h>
#include <ATen/ops/bitwise_right_shift_ops.h>
#include <ATen/ops/bitwise_right_shift_ops.h>
#include <ATen/ops/bitwise_right_shift_ops.h>
#include <ATen/ops/bitwise_right_shift_ops.h>
#include <ATen/ops/bitwise_xor_ops.h>
#include <ATen/ops/bitwise_xor_ops.h>
#include <ATen/ops/bitwise_xor_ops.h>
#include <ATen/ops/bitwise_xor_ops.h>
#include <ATen/ops/bitwise_xor_ops.h>
#include <ATen/ops/blackman_window_ops.h>
#include <ATen/ops/blackman_window_ops.h>
#include <ATen/ops/block_diag_ops.h>
#include <ATen/ops/bmm_ops.h>
#include <ATen/ops/bucketize_ops.h>
#include <ATen/ops/bucketize_ops.h>
#include <ATen/ops/cat_ops.h>
#include <ATen/ops/cauchy_ops.h>
#include <ATen/ops/cauchy_ops.h>
#include <ATen/ops/ccol_indices_ops.h>
#include <ATen/ops/ccol_indices_copy_ops.h>
#include <ATen/ops/ceil_ops.h>
#include <ATen/ops/ceil_ops.h>
#include <ATen/ops/celu_ops.h>
#include <ATen/ops/celu_ops.h>
#include <ATen/ops/channel_shuffle_ops.h>
#include <ATen/ops/cholesky_inverse_ops.h>
#include <ATen/ops/cholesky_ops.h>
#include <ATen/ops/cholesky_solve_ops.h>
#include <ATen/ops/chunk_ops.h>
#include <ATen/ops/clamp_ops.h>
#include <ATen/ops/clamp_ops.h>
#include <ATen/ops/clamp_max_ops.h>
#include <ATen/ops/clamp_max_ops.h>
#include <ATen/ops/clamp_max_ops.h>
#include <ATen/ops/clamp_max_ops.h>
#include <ATen/ops/clamp_min_ops.h>
#include <ATen/ops/clamp_min_ops.h>
#include <ATen/ops/clamp_min_ops.h>
#include <ATen/ops/clamp_min_ops.h>
#include <ATen/ops/clamp_ops.h>
#include <ATen/ops/clamp_ops.h>
#include <ATen/ops/clone_ops.h>
#include <ATen/ops/col2im_ops.h>
#include <ATen/ops/col_indices_ops.h>
#include <ATen/ops/col_indices_copy_ops.h>
#include <ATen/ops/complex_ops.h>
#include <ATen/ops/conj_physical_ops.h>
#include <ATen/ops/conj_physical_ops.h>
#include <ATen/ops/constant_pad_nd_ops.h>
#include <ATen/ops/conv_depthwise3d_ops.h>
#include <ATen/ops/conv_tbc_ops.h>
#include <ATen/ops/convolution_backward_ops.h>
#include <ATen/ops/convolution_backward_overrideable_ops.h>
#include <ATen/ops/convolution_ops.h>
#include <ATen/ops/convolution_overrideable_ops.h>
#include <ATen/ops/copy_ops.h>
#include <ATen/ops/copy_sparse_to_sparse_ops.h>
#include <ATen/ops/copy_sparse_to_sparse_ops.h>
#include <ATen/ops/copysign_ops.h>
#include <ATen/ops/copysign_ops.h>
#include <ATen/ops/copysign_ops.h>
#include <ATen/ops/copysign_ops.h>
#include <ATen/ops/cos_ops.h>
#include <ATen/ops/cos_ops.h>
#include <ATen/ops/cosh_ops.h>
#include <ATen/ops/cosh_ops.h>
#include <ATen/ops/count_nonzero_ops.h>
#include <ATen/ops/count_nonzero_ops.h>
#include <ATen/ops/crow_indices_ops.h>
#include <ATen/ops/crow_indices_copy_ops.h>
#include <ATen/ops/cudnn_affine_grid_generator_backward_ops.h>
#include <ATen/ops/cudnn_affine_grid_generator_ops.h>
#include <ATen/ops/cudnn_batch_norm_backward_ops.h>
#include <ATen/ops/cudnn_batch_norm_ops.h>
#include <ATen/ops/cudnn_convolution_add_relu_ops.h>
#include <ATen/ops/cudnn_convolution_ops.h>
#include <ATen/ops/cudnn_convolution_relu_ops.h>
#include <ATen/ops/cudnn_convolution_transpose_ops.h>
#include <ATen/ops/cudnn_grid_sampler_backward_ops.h>
#include <ATen/ops/cudnn_grid_sampler_ops.h>
#include <ATen/ops/cummax_ops.h>
#include <ATen/ops/cummin_ops.h>
#include <ATen/ops/cumprod_ops.h>
#include <ATen/ops/cumprod_ops.h>
#include <ATen/ops/cumsum_ops.h>
#include <ATen/ops/cumsum_ops.h>
#include <ATen/ops/deg2rad_ops.h>
#include <ATen/ops/deg2rad_ops.h>
#include <ATen/ops/dequantize_ops.h>
#include <ATen/ops/detach_copy_ops.h>
#include <ATen/ops/diag_embed_ops.h>
#include <ATen/ops/diagonal_ops.h>
#include <ATen/ops/diagonal_backward_ops.h>
#include <ATen/ops/diagonal_copy_ops.h>
#include <ATen/ops/diagonal_scatter_ops.h>
#include <ATen/ops/digamma_ops.h>
#include <ATen/ops/digamma_ops.h>
#include <ATen/ops/dist_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/dot_ops.h>
#include <ATen/ops/elu_ops.h>
#include <ATen/ops/elu_backward_ops.h>
#include <ATen/ops/elu_ops.h>
#include <ATen/ops/embedding_dense_backward_ops.h>
#include <ATen/ops/embedding_ops.h>
#include <ATen/ops/embedding_renorm_ops.h>
#include <ATen/ops/embedding_renorm_ops.h>
#include <ATen/ops/empty_like_ops.h>
#include <ATen/ops/empty_ops.h>
#include <ATen/ops/empty_permuted_ops.h>
#include <ATen/ops/empty_quantized_ops.h>
#include <ATen/ops/empty_strided_ops.h>
#include <ATen/ops/eq_ops.h>
#include <ATen/ops/eq_ops.h>
#include <ATen/ops/eq_ops.h>
#include <ATen/ops/eq_ops.h>
#include <ATen/ops/erf_ops.h>
#include <ATen/ops/erf_ops.h>
#include <ATen/ops/erfc_ops.h>
#include <ATen/ops/erfc_ops.h>
#include <ATen/ops/erfinv_ops.h>
#include <ATen/ops/erfinv_ops.h>
#include <ATen/ops/exp2_ops.h>
#include <ATen/ops/exp2_ops.h>
#include <ATen/ops/exp_ops.h>
#include <ATen/ops/exp_ops.h>
#include <ATen/ops/expand_ops.h>
#include <ATen/ops/expand_copy_ops.h>
#include <ATen/ops/expm1_ops.h>
#include <ATen/ops/expm1_ops.h>
#include <ATen/ops/exponential_ops.h>
#include <ATen/ops/exponential_ops.h>
#include <ATen/ops/eye_ops.h>
#include <ATen/ops/eye_ops.h>
#include <ATen/ops/fake_quantize_per_channel_affine_cachemask_ops.h>
#include <ATen/ops/fake_quantize_per_tensor_affine_cachemask_ops.h>
#include <ATen/ops/fft_fftfreq_ops.h>
#include <ATen/ops/fft_rfftfreq_ops.h>
#include <ATen/ops/fill_ops.h>
#include <ATen/ops/fill_ops.h>
#include <ATen/ops/fill_ops.h>
#include <ATen/ops/fill_ops.h>
#include <ATen/ops/flip_ops.h>
#include <ATen/ops/floor_ops.h>
#include <ATen/ops/floor_divide_ops.h>
#include <ATen/ops/floor_divide_ops.h>
#include <ATen/ops/floor_divide_ops.h>
#include <ATen/ops/floor_divide_ops.h>
#include <ATen/ops/floor_ops.h>
#include <ATen/ops/fmax_ops.h>
#include <ATen/ops/fmin_ops.h>
#include <ATen/ops/fmod_ops.h>
#include <ATen/ops/fmod_ops.h>
#include <ATen/ops/fmod_ops.h>
#include <ATen/ops/fmod_ops.h>
#include <ATen/ops/frac_ops.h>
#include <ATen/ops/frac_ops.h>
#include <ATen/ops/fractional_max_pool2d_backward_ops.h>
#include <ATen/ops/fractional_max_pool2d_ops.h>
#include <ATen/ops/fractional_max_pool3d_backward_ops.h>
#include <ATen/ops/fractional_max_pool3d_ops.h>
#include <ATen/ops/frexp_ops.h>
#include <ATen/ops/from_file_ops.h>
#include <ATen/ops/full_like_ops.h>
#include <ATen/ops/full_ops.h>
#include <ATen/ops/full_ops.h>
#include <ATen/ops/gather_ops.h>
#include <ATen/ops/gcd_ops.h>
#include <ATen/ops/gcd_ops.h>
#include <ATen/ops/ge_ops.h>
#include <ATen/ops/ge_ops.h>
#include <ATen/ops/ge_ops.h>
#include <ATen/ops/ge_ops.h>
#include <ATen/ops/gelu_ops.h>
#include <ATen/ops/gelu_backward_ops.h>
#include <ATen/ops/gelu_ops.h>
#include <ATen/ops/geometric_ops.h>
#include <ATen/ops/geometric_ops.h>
#include <ATen/ops/geqrf_ops.h>
#include <ATen/ops/glu_backward_jvp_ops.h>
#include <ATen/ops/glu_backward_ops.h>
#include <ATen/ops/glu_jvp_ops.h>
#include <ATen/ops/glu_ops.h>
#include <ATen/ops/grid_sampler_2d_backward_ops.h>
#include <ATen/ops/grid_sampler_2d_ops.h>
#include <ATen/ops/grid_sampler_3d_backward_ops.h>
#include <ATen/ops/grid_sampler_3d_ops.h>
#include <ATen/ops/gt_ops.h>
#include <ATen/ops/gt_ops.h>
#include <ATen/ops/gt_ops.h>
#include <ATen/ops/gt_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/hann_window_ops.h>
#include <ATen/ops/hann_window_ops.h>
#include <ATen/ops/hardshrink_backward_ops.h>
#include <ATen/ops/hardshrink_ops.h>
#include <ATen/ops/hardsigmoid_ops.h>
#include <ATen/ops/hardsigmoid_backward_ops.h>
#include <ATen/ops/hardsigmoid_ops.h>
#include <ATen/ops/hardswish_ops.h>
#include <ATen/ops/hardswish_backward_ops.h>
#include <ATen/ops/hardswish_ops.h>
#include <ATen/ops/hardtanh_ops.h>
#include <ATen/ops/hardtanh_backward_ops.h>
#include <ATen/ops/hardtanh_ops.h>
#include <ATen/ops/heaviside_ops.h>
#include <ATen/ops/heaviside_ops.h>
#include <ATen/ops/histc_ops.h>
#include <ATen/ops/histogram_ops.h>
#include <ATen/ops/histogram_ops.h>
#include <ATen/ops/hspmm_ops.h>
#include <ATen/ops/huber_loss_backward_ops.h>
#include <ATen/ops/huber_loss_ops.h>
#include <ATen/ops/hypot_ops.h>
#include <ATen/ops/hypot_ops.h>
#include <ATen/ops/i0_ops.h>
#include <ATen/ops/i0_ops.h>
#include <ATen/ops/igamma_ops.h>
#include <ATen/ops/igamma_ops.h>
#include <ATen/ops/igammac_ops.h>
#include <ATen/ops/igammac_ops.h>
#include <ATen/ops/im2col_ops.h>
#include <ATen/ops/index_add_ops.h>
#include <ATen/ops/index_add_ops.h>
#include <ATen/ops/index_copy_ops.h>
#include <ATen/ops/index_copy_ops.h>
#include <ATen/ops/index_fill_ops.h>
#include <ATen/ops/index_fill_ops.h>
#include <ATen/ops/index_fill_ops.h>
#include <ATen/ops/index_fill_ops.h>
#include <ATen/ops/index_ops.h>
#include <ATen/ops/index_put_ops.h>
#include <ATen/ops/index_put_ops.h>
#include <ATen/ops/index_reduce_ops.h>
#include <ATen/ops/index_reduce_ops.h>
#include <ATen/ops/index_select_ops.h>
#include <ATen/ops/indices_ops.h>
#include <ATen/ops/indices_copy_ops.h>
#include <ATen/ops/int_repr_ops.h>
#include <ATen/ops/isin_ops.h>
#include <ATen/ops/isin_ops.h>
#include <ATen/ops/isin_ops.h>
#include <ATen/ops/isinf_ops.h>
#include <ATen/ops/isnan_ops.h>
#include <ATen/ops/isneginf_ops.h>
#include <ATen/ops/isposinf_ops.h>
#include <ATen/ops/kaiser_window_ops.h>
#include <ATen/ops/kaiser_window_ops.h>
#include <ATen/ops/kaiser_window_ops.h>
#include <ATen/ops/kthvalue_ops.h>
#include <ATen/ops/lcm_ops.h>
#include <ATen/ops/lcm_ops.h>
#include <ATen/ops/le_ops.h>
#include <ATen/ops/le_ops.h>
#include <ATen/ops/le_ops.h>
#include <ATen/ops/le_ops.h>
#include <ATen/ops/leaky_relu_ops.h>
#include <ATen/ops/leaky_relu_backward_ops.h>
#include <ATen/ops/leaky_relu_ops.h>
#include <ATen/ops/lerp_ops.h>
#include <ATen/ops/lerp_ops.h>
#include <ATen/ops/lerp_ops.h>
#include <ATen/ops/lerp_ops.h>
#include <ATen/ops/lgamma_ops.h>
#include <ATen/ops/lgamma_ops.h>
#include <ATen/ops/lift_fresh_copy_ops.h>
#include <ATen/ops/lift_ops.h>
#include <ATen/ops/linalg_cholesky_ex_ops.h>
#include <ATen/ops/linalg_cross_ops.h>
#include <ATen/ops/linalg_eig_ops.h>
#include <ATen/ops/linalg_eigvals_ops.h>
#include <ATen/ops/linalg_householder_product_ops.h>
#include <ATen/ops/linalg_inv_ex_ops.h>
#include <ATen/ops/linalg_ldl_factor_ex_ops.h>
#include <ATen/ops/linalg_ldl_solve_ops.h>
#include <ATen/ops/linalg_lstsq_ops.h>
#include <ATen/ops/linalg_lu_factor_ex_ops.h>
#include <ATen/ops/linalg_lu_factor_ops.h>
#include <ATen/ops/linalg_lu_ops.h>
#include <ATen/ops/linalg_lu_solve_ops.h>
#include <ATen/ops/linalg_matrix_exp_ops.h>
#include <ATen/ops/linalg_pinv_ops.h>
#include <ATen/ops/linalg_qr_ops.h>
#include <ATen/ops/linalg_solve_triangular_ops.h>
#include <ATen/ops/linalg_vector_norm_ops.h>
#include <ATen/ops/linear_backward_ops.h>
#include <ATen/ops/linear_ops.h>
#include <ATen/ops/linspace_ops.h>
#include <ATen/ops/linspace_ops.h>
#include <ATen/ops/linspace_ops.h>
#include <ATen/ops/linspace_ops.h>
#include <ATen/ops/log10_ops.h>
#include <ATen/ops/log10_ops.h>
#include <ATen/ops/log1p_ops.h>
#include <ATen/ops/log1p_ops.h>
#include <ATen/ops/log2_ops.h>
#include <ATen/ops/log2_ops.h>
#include <ATen/ops/log_ops.h>
#include <ATen/ops/log_normal_ops.h>
#include <ATen/ops/log_normal_ops.h>
#include <ATen/ops/log_ops.h>
#include <ATen/ops/log_sigmoid_backward_ops.h>
#include <ATen/ops/log_sigmoid_forward_ops.h>
#include <ATen/ops/log_softmax_ops.h>
#include <ATen/ops/logaddexp2_ops.h>
#include <ATen/ops/logaddexp_ops.h>
#include <ATen/ops/logcumsumexp_ops.h>
#include <ATen/ops/logical_and_ops.h>
#include <ATen/ops/logical_and_ops.h>
#include <ATen/ops/logical_not_ops.h>
#include <ATen/ops/logical_not_ops.h>
#include <ATen/ops/logical_or_ops.h>
#include <ATen/ops/logical_or_ops.h>
#include <ATen/ops/logical_xor_ops.h>
#include <ATen/ops/logical_xor_ops.h>
#include <ATen/ops/logit_ops.h>
#include <ATen/ops/logit_backward_ops.h>
#include <ATen/ops/logit_ops.h>
#include <ATen/ops/logspace_ops.h>
#include <ATen/ops/logspace_ops.h>
#include <ATen/ops/logspace_ops.h>
#include <ATen/ops/logspace_ops.h>
#include <ATen/ops/logsumexp_ops.h>
#include <ATen/ops/lt_ops.h>
#include <ATen/ops/lt_ops.h>
#include <ATen/ops/lt_ops.h>
#include <ATen/ops/lt_ops.h>
#include <ATen/ops/lu_unpack_ops.h>
#include <ATen/ops/masked_fill_ops.h>
#include <ATen/ops/masked_fill_ops.h>
#include <ATen/ops/masked_fill_ops.h>
#include <ATen/ops/masked_fill_ops.h>
#include <ATen/ops/masked_scatter_ops.h>
#include <ATen/ops/masked_scatter_ops.h>
#include <ATen/ops/masked_select_ops.h>
#include <ATen/ops/matmul_backward_ops.h>
#include <ATen/ops/matmul_ops.h>
#include <ATen/ops/max_ops.h>
#include <ATen/ops/max_ops.h>
#include <ATen/ops/max_pool2d_backward_ops.h>
#include <ATen/ops/max_pool2d_with_indices_backward_ops.h>
#include <ATen/ops/max_pool2d_with_indices_ops.h>
#include <ATen/ops/max_pool3d_with_indices_backward_ops.h>
#include <ATen/ops/max_pool3d_with_indices_ops.h>
#include <ATen/ops/max_unpool2d_ops.h>
#include <ATen/ops/max_unpool3d_ops.h>
#include <ATen/ops/maximum_ops.h>
#include <ATen/ops/mean_ops.h>
#include <ATen/ops/mean_ops.h>
#include <ATen/ops/median_ops.h>
#include <ATen/ops/median_ops.h>
#include <ATen/ops/min_ops.h>
#include <ATen/ops/min_ops.h>
#include <ATen/ops/minimum_ops.h>
#include <ATen/ops/miopen_batch_norm_backward_ops.h>
#include <ATen/ops/miopen_batch_norm_ops.h>
#include <ATen/ops/miopen_convolution_ops.h>
#include <ATen/ops/miopen_convolution_transpose_ops.h>
#include <ATen/ops/miopen_depthwise_convolution_ops.h>
#include <ATen/ops/miopen_rnn_ops.h>
#include <ATen/ops/mish_ops.h>
#include <ATen/ops/mish_ops.h>
#include <ATen/ops/mkldnn_adaptive_avg_pool2d_backward_ops.h>
#include <ATen/ops/mkldnn_adaptive_avg_pool2d_ops.h>
#include <ATen/ops/mkldnn_convolution_ops.h>
#include <ATen/ops/mkldnn_linear_backward_input_ops.h>
#include <ATen/ops/mkldnn_linear_backward_ops.h>
#include <ATen/ops/mkldnn_linear_backward_weights_ops.h>
#include <ATen/ops/mkldnn_linear_ops.h>
#include <ATen/ops/mkldnn_max_pool2d_backward_ops.h>
#include <ATen/ops/mkldnn_max_pool2d_ops.h>
#include <ATen/ops/mkldnn_max_pool3d_backward_ops.h>
#include <ATen/ops/mkldnn_max_pool3d_ops.h>
#include <ATen/ops/mkldnn_reorder_conv2d_weight_ops.h>
#include <ATen/ops/mkldnn_reorder_conv3d_weight_ops.h>
#include <ATen/ops/mkldnn_rnn_layer_backward_ops.h>
#include <ATen/ops/mkldnn_rnn_layer_ops.h>
#include <ATen/ops/mm_ops.h>
#include <ATen/ops/mode_ops.h>
#include <ATen/ops/mps_convolution_backward_ops.h>
#include <ATen/ops/mps_convolution_transpose_backward_ops.h>
#include <ATen/ops/mse_loss_backward_ops.h>
#include <ATen/ops/mse_loss_ops.h>
#include <ATen/ops/mul_ops.h>
#include <ATen/ops/mul_ops.h>
#include <ATen/ops/mul_ops.h>
#include <ATen/ops/mul_ops.h>
#include <ATen/ops/multi_margin_loss_backward_ops.h>
#include <ATen/ops/multi_margin_loss_ops.h>
#include <ATen/ops/multilabel_margin_loss_backward_ops.h>
#include <ATen/ops/multilabel_margin_loss_forward_ops.h>
#include <ATen/ops/multinomial_ops.h>
#include <ATen/ops/mv_ops.h>
#include <ATen/ops/mvlgamma_ops.h>
#include <ATen/ops/mvlgamma_ops.h>
#include <ATen/ops/nan_to_num_ops.h>
#include <ATen/ops/nan_to_num_ops.h>
#include <ATen/ops/nanmedian_ops.h>
#include <ATen/ops/nanmedian_ops.h>
#include <ATen/ops/nansum_ops.h>
#include <ATen/ops/narrow_ops.h>
#include <ATen/ops/narrow_copy_ops.h>
#include <ATen/ops/native_batch_norm_backward_ops.h>
#include <ATen/ops/native_batch_norm_ops.h>
#include <ATen/ops/native_dropout_backward_ops.h>
#include <ATen/ops/native_dropout_ops.h>
#include <ATen/ops/native_group_norm_backward_ops.h>
#include <ATen/ops/native_group_norm_ops.h>
#include <ATen/ops/native_layer_norm_backward_ops.h>
#include <ATen/ops/native_layer_norm_ops.h>
#include <ATen/ops/native_norm_ops.h>
#include <ATen/ops/native_norm_ops.h>
#include <ATen/ops/ne_ops.h>
#include <ATen/ops/ne_ops.h>
#include <ATen/ops/ne_ops.h>
#include <ATen/ops/ne_ops.h>
#include <ATen/ops/neg_ops.h>
#include <ATen/ops/neg_ops.h>
#include <ATen/ops/new_empty_ops.h>
#include <ATen/ops/new_empty_strided_ops.h>
#include <ATen/ops/new_full_ops.h>
#include <ATen/ops/new_ones_ops.h>
#include <ATen/ops/new_zeros_ops.h>
#include <ATen/ops/nextafter_ops.h>
#include <ATen/ops/nextafter_ops.h>
#include <ATen/ops/nll_loss2d_backward_ops.h>
#include <ATen/ops/nll_loss2d_forward_ops.h>
#include <ATen/ops/nll_loss_backward_ops.h>
#include <ATen/ops/nll_loss_forward_ops.h>
#include <ATen/ops/nonzero_ops.h>
#include <ATen/ops/nonzero_static_ops.h>
#include <ATen/ops/norm_ops.h>
#include <ATen/ops/norm_ops.h>
#include <ATen/ops/norm_ops.h>
#include <ATen/ops/norm_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/ones_like_ops.h>
#include <ATen/ops/ones_ops.h>
#include <ATen/ops/ones_ops.h>
#include <ATen/ops/ormqr_ops.h>
#include <ATen/ops/permute_ops.h>
#include <ATen/ops/permute_copy_ops.h>
#include <ATen/ops/pixel_shuffle_ops.h>
#include <ATen/ops/pixel_unshuffle_ops.h>
#include <ATen/ops/poisson_ops.h>
#include <ATen/ops/polar_ops.h>
#include <ATen/ops/polygamma_ops.h>
#include <ATen/ops/polygamma_ops.h>
#include <ATen/ops/pow_ops.h>
#include <ATen/ops/pow_ops.h>
#include <ATen/ops/pow_ops.h>
#include <ATen/ops/pow_ops.h>
#include <ATen/ops/pow_ops.h>
#include <ATen/ops/prod_ops.h>
#include <ATen/ops/prod_ops.h>
#include <ATen/ops/put_ops.h>
#include <ATen/ops/put_ops.h>
#include <ATen/ops/q_per_channel_scales_ops.h>
#include <ATen/ops/q_per_channel_zero_points_ops.h>
#include <ATen/ops/quantize_per_channel_ops.h>
#include <ATen/ops/quantize_per_tensor_dynamic_ops.h>
#include <ATen/ops/quantize_per_tensor_ops.h>
#include <ATen/ops/quantize_per_tensor_ops.h>
#include <ATen/ops/quantized_batch_norm_ops.h>
#include <ATen/ops/quantized_max_pool1d_ops.h>
#include <ATen/ops/quantized_max_pool2d_ops.h>
#include <ATen/ops/quantized_max_pool3d_ops.h>
#include <ATen/ops/rad2deg_ops.h>
#include <ATen/ops/rad2deg_ops.h>
#include <ATen/ops/rand_like_ops.h>
#include <ATen/ops/rand_ops.h>
#include <ATen/ops/rand_ops.h>
#include <ATen/ops/rand_ops.h>
#include <ATen/ops/randint_like_ops.h>
#include <ATen/ops/randint_like_ops.h>
#include <ATen/ops/randint_ops.h>
#include <ATen/ops/randint_ops.h>
#include <ATen/ops/randint_ops.h>
#include <ATen/ops/randint_ops.h>
#include <ATen/ops/randn_like_ops.h>
#include <ATen/ops/randn_ops.h>
#include <ATen/ops/randn_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/randperm_ops.h>
#include <ATen/ops/randperm_ops.h>
#include <ATen/ops/range_ops.h>
#include <ATen/ops/range_ops.h>
#include <ATen/ops/reciprocal_ops.h>
#include <ATen/ops/reciprocal_ops.h>
#include <ATen/ops/reflection_pad1d_backward_ops.h>
#include <ATen/ops/reflection_pad1d_ops.h>
#include <ATen/ops/reflection_pad2d_backward_ops.h>
#include <ATen/ops/reflection_pad2d_ops.h>
#include <ATen/ops/reflection_pad3d_backward_ops.h>
#include <ATen/ops/reflection_pad3d_ops.h>
#include <ATen/ops/relu_ops.h>
#include <ATen/ops/relu_ops.h>
#include <ATen/ops/remainder_ops.h>
#include <ATen/ops/remainder_ops.h>
#include <ATen/ops/remainder_ops.h>
#include <ATen/ops/remainder_ops.h>
#include <ATen/ops/remainder_ops.h>
#include <ATen/ops/renorm_ops.h>
#include <ATen/ops/renorm_ops.h>
#include <ATen/ops/repeat_interleave_ops.h>
#include <ATen/ops/repeat_ops.h>
#include <ATen/ops/replication_pad1d_backward_ops.h>
#include <ATen/ops/replication_pad1d_ops.h>
#include <ATen/ops/replication_pad2d_backward_ops.h>
#include <ATen/ops/replication_pad2d_ops.h>
#include <ATen/ops/replication_pad3d_backward_ops.h>
#include <ATen/ops/replication_pad3d_ops.h>
#include <ATen/ops/resize_as_ops.h>
#include <ATen/ops/resize_as_sparse_ops.h>
#include <ATen/ops/resize_as_sparse_ops.h>
#include <ATen/ops/resize_ops.h>
#include <ATen/ops/roll_ops.h>
#include <ATen/ops/rot90_ops.h>
#include <ATen/ops/round_ops.h>
#include <ATen/ops/round_ops.h>
#include <ATen/ops/round_ops.h>
#include <ATen/ops/round_ops.h>
#include <ATen/ops/row_indices_ops.h>
#include <ATen/ops/row_indices_copy_ops.h>
#include <ATen/ops/rrelu_with_noise_ops.h>
#include <ATen/ops/rrelu_with_noise_backward_ops.h>
#include <ATen/ops/rrelu_with_noise_ops.h>
#include <ATen/ops/rsqrt_ops.h>
#include <ATen/ops/rsqrt_ops.h>
#include <ATen/ops/rsub_ops.h>
#include <ATen/ops/rsub_ops.h>
#include <ATen/ops/scalar_tensor_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_add_ops.h>
#include <ATen/ops/scatter_add_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_reduce_ops.h>
#include <ATen/ops/scatter_reduce_ops.h>
#include <ATen/ops/searchsorted_ops.h>
#include <ATen/ops/searchsorted_ops.h>
#include <ATen/ops/segment_reduce_ops.h>
#include <ATen/ops/select_ops.h>
#include <ATen/ops/select_backward_ops.h>
#include <ATen/ops/select_copy_ops.h>
#include <ATen/ops/select_scatter_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/sgn_ops.h>
#include <ATen/ops/sgn_ops.h>
#include <ATen/ops/sigmoid_ops.h>
#include <ATen/ops/sigmoid_backward_ops.h>
#include <ATen/ops/sigmoid_ops.h>
#include <ATen/ops/sign_ops.h>
#include <ATen/ops/sign_ops.h>
#include <ATen/ops/signbit_ops.h>
#include <ATen/ops/silu_ops.h>
#include <ATen/ops/silu_backward_ops.h>
#include <ATen/ops/silu_ops.h>
#include <ATen/ops/sin_ops.h>
#include <ATen/ops/sin_ops.h>
#include <ATen/ops/sinc_ops.h>
#include <ATen/ops/sinc_ops.h>
#include <ATen/ops/sinh_ops.h>
#include <ATen/ops/sinh_ops.h>
#include <ATen/ops/slice_ops.h>
#include <ATen/ops/slice_backward_ops.h>
#include <ATen/ops/slice_copy_ops.h>
#include <ATen/ops/slice_inverse_ops.h>
#include <ATen/ops/slice_scatter_ops.h>
#include <ATen/ops/slow_conv3d_forward_ops.h>
#include <ATen/ops/slow_conv_dilated2d_ops.h>
#include <ATen/ops/slow_conv_dilated3d_ops.h>
#include <ATen/ops/slow_conv_transpose2d_ops.h>
#include <ATen/ops/slow_conv_transpose3d_ops.h>
#include <ATen/ops/smooth_l1_loss_backward_ops.h>
#include <ATen/ops/smooth_l1_loss_ops.h>
#include <ATen/ops/soft_margin_loss_backward_ops.h>
#include <ATen/ops/soft_margin_loss_ops.h>
#include <ATen/ops/softmax_ops.h>
#include <ATen/ops/softplus_backward_ops.h>
#include <ATen/ops/softplus_ops.h>
#include <ATen/ops/softshrink_backward_ops.h>
#include <ATen/ops/softshrink_ops.h>
#include <ATen/ops/sort_ops.h>
#include <ATen/ops/sort_ops.h>
#include <ATen/ops/sparse_coo_tensor_ops.h>
#include <ATen/ops/sparse_mask_ops.h>
#include <ATen/ops/sparse_resize_ops.h>
#include <ATen/ops/sparse_resize_and_clear_ops.h>
#include <ATen/ops/sparse_resize_and_clear_ops.h>
#include <ATen/ops/sparse_resize_ops.h>
#include <ATen/ops/sparse_sampled_addmm_ops.h>
#include <ATen/ops/special_airy_ai_ops.h>
#include <ATen/ops/special_bessel_j0_ops.h>
#include <ATen/ops/special_bessel_j1_ops.h>
#include <ATen/ops/special_bessel_y0_ops.h>
#include <ATen/ops/special_bessel_y1_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_u_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_u_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_u_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_v_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_v_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_v_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_w_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_w_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_w_ops.h>
#include <ATen/ops/special_entr_ops.h>
#include <ATen/ops/special_erfcx_ops.h>
#include <ATen/ops/special_hermite_polynomial_h_ops.h>
#include <ATen/ops/special_hermite_polynomial_h_ops.h>
#include <ATen/ops/special_hermite_polynomial_h_ops.h>
#include <ATen/ops/special_hermite_polynomial_he_ops.h>
#include <ATen/ops/special_hermite_polynomial_he_ops.h>
#include <ATen/ops/special_hermite_polynomial_he_ops.h>
#include <ATen/ops/special_i0e_ops.h>
#include <ATen/ops/special_i1_ops.h>
#include <ATen/ops/special_i1e_ops.h>
#include <ATen/ops/special_laguerre_polynomial_l_ops.h>
#include <ATen/ops/special_laguerre_polynomial_l_ops.h>
#include <ATen/ops/special_laguerre_polynomial_l_ops.h>
#include <ATen/ops/special_legendre_polynomial_p_ops.h>
#include <ATen/ops/special_legendre_polynomial_p_ops.h>
#include <ATen/ops/special_legendre_polynomial_p_ops.h>
#include <ATen/ops/special_log_ndtr_ops.h>
#include <ATen/ops/special_modified_bessel_i0_ops.h>
#include <ATen/ops/special_modified_bessel_i1_ops.h>
#include <ATen/ops/special_modified_bessel_k0_ops.h>
#include <ATen/ops/special_modified_bessel_k1_ops.h>
#include <ATen/ops/special_ndtri_ops.h>
#include <ATen/ops/special_scaled_modified_bessel_k0_ops.h>
#include <ATen/ops/special_scaled_modified_bessel_k1_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_u_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_u_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_u_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_v_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_v_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_v_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_w_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_w_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_w_ops.h>
#include <ATen/ops/special_spherical_bessel_j0_ops.h>
#include <ATen/ops/special_xlog1py_ops.h>
#include <ATen/ops/special_xlog1py_ops.h>
#include <ATen/ops/special_xlog1py_ops.h>
#include <ATen/ops/special_zeta_ops.h>
#include <ATen/ops/special_zeta_ops.h>
#include <ATen/ops/special_zeta_ops.h>
#include <ATen/ops/split_ops.h>
#include <ATen/ops/split_with_sizes_ops.h>
#include <ATen/ops/sqrt_ops.h>
#include <ATen/ops/sqrt_ops.h>
#include <ATen/ops/squeeze_ops.h>
#include <ATen/ops/squeeze_ops.h>
#include <ATen/ops/squeeze_ops.h>
#include <ATen/ops/squeeze_ops.h>
#include <ATen/ops/squeeze_ops.h>
#include <ATen/ops/squeeze_ops.h>
#include <ATen/ops/squeeze_copy_ops.h>
#include <ATen/ops/squeeze_copy_ops.h>
#include <ATen/ops/squeeze_copy_ops.h>
#include <ATen/ops/sspaddmm_ops.h>
#include <ATen/ops/stack_ops.h>
#include <ATen/ops/std_mean_ops.h>
#include <ATen/ops/std_ops.h>
#include <ATen/ops/sub_ops.h>
#include <ATen/ops/sub_ops.h>
#include <ATen/ops/sub_ops.h>
#include <ATen/ops/sub_ops.h>
#include <ATen/ops/sum_ops.h>
#include <ATen/ops/sum_ops.h>
#include <ATen/ops/t_ops.h>
#include <ATen/ops/t_ops.h>
#include <ATen/ops/t_copy_ops.h>
#include <ATen/ops/take_ops.h>
#include <ATen/ops/tan_ops.h>
#include <ATen/ops/tan_ops.h>
#include <ATen/ops/tanh_ops.h>
#include <ATen/ops/tanh_backward_ops.h>
#include <ATen/ops/tanh_ops.h>
#include <ATen/ops/threshold_ops.h>
#include <ATen/ops/threshold_backward_ops.h>
#include <ATen/ops/threshold_ops.h>
#include <ATen/ops/to_mkldnn_ops.h>
#include <ATen/ops/to_padded_tensor_ops.h>
#include <ATen/ops/topk_ops.h>
#include <ATen/ops/trace_ops.h>
#include <ATen/ops/transpose_ops.h>
#include <ATen/ops/transpose_ops.h>
#include <ATen/ops/transpose_copy_ops.h>
#include <ATen/ops/triangular_solve_ops.h>
#include <ATen/ops/tril_ops.h>
#include <ATen/ops/tril_indices_ops.h>
#include <ATen/ops/tril_ops.h>
#include <ATen/ops/triu_ops.h>
#include <ATen/ops/triu_indices_ops.h>
#include <ATen/ops/triu_ops.h>
#include <ATen/ops/trunc_ops.h>
#include <ATen/ops/trunc_ops.h>
#include <ATen/ops/unbind_ops.h>
#include <ATen/ops/unfold_ops.h>
#include <ATen/ops/unfold_backward_ops.h>
#include <ATen/ops/unfold_copy_ops.h>
#include <ATen/ops/uniform_ops.h>
#include <ATen/ops/uniform_ops.h>
#include <ATen/ops/unique_consecutive_ops.h>
#include <ATen/ops/unique_dim_consecutive_ops.h>
#include <ATen/ops/unique_dim_ops.h>
#include <ATen/ops/unsqueeze_ops.h>
#include <ATen/ops/unsqueeze_ops.h>
#include <ATen/ops/unsqueeze_copy_ops.h>
#include <ATen/ops/upsample_bicubic2d_backward_ops.h>
#include <ATen/ops/upsample_bicubic2d_ops.h>
#include <ATen/ops/upsample_bilinear2d_backward_ops.h>
#include <ATen/ops/upsample_bilinear2d_ops.h>
#include <ATen/ops/upsample_linear1d_backward_ops.h>
#include <ATen/ops/upsample_linear1d_ops.h>
#include <ATen/ops/upsample_nearest1d_backward_ops.h>
#include <ATen/ops/upsample_nearest1d_ops.h>
#include <ATen/ops/upsample_nearest2d_backward_ops.h>
#include <ATen/ops/upsample_nearest2d_ops.h>
#include <ATen/ops/upsample_nearest3d_backward_ops.h>
#include <ATen/ops/upsample_nearest3d_ops.h>
#include <ATen/ops/upsample_trilinear3d_backward_ops.h>
#include <ATen/ops/upsample_trilinear3d_ops.h>
#include <ATen/ops/values_ops.h>
#include <ATen/ops/values_copy_ops.h>
#include <ATen/ops/var_mean_ops.h>
#include <ATen/ops/var_ops.h>
#include <ATen/ops/vdot_ops.h>
#include <ATen/ops/view_ops.h>
#include <ATen/ops/view_ops.h>
#include <ATen/ops/view_as_complex_ops.h>
#include <ATen/ops/view_as_complex_copy_ops.h>
#include <ATen/ops/view_as_real_ops.h>
#include <ATen/ops/view_as_real_copy_ops.h>
#include <ATen/ops/view_copy_ops.h>
#include <ATen/ops/view_copy_ops.h>
#include <ATen/ops/where_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/zero_ops.h>
#include <ATen/ops/zero_ops.h>
#include <ATen/ops/zeros_like_ops.h>
#include <ATen/ops/zeros_ops.h>
#include <ATen/ops/zeros_ops.h>
#endif

using namespace at;
using torch::autograd::CreationMeta;
using torch::autograd::as_view;
using torch::autograd::increment_version;

namespace torch {

namespace ADInplaceOrView {

namespace {
at::Tensor & __ilshift___Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::__ilshift___Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & __ilshift___Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::__ilshift___Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & __irshift___Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::__irshift___Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & __irshift___Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::__irshift___Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & __lshift___out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::__lshift___Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & __lshift___out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::__lshift___Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & __rshift___out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::__rshift___Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & __rshift___out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::__rshift___Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _adaptive_avg_pool2d_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_adaptive_avg_pool2d_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _adaptive_avg_pool2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_adaptive_avg_pool2d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _adaptive_avg_pool3d_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_adaptive_avg_pool3d_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _adaptive_avg_pool3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_adaptive_avg_pool3d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _add_relu__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_add_relu__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & _add_relu__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_add_relu__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & _add_relu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_add_relu_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _add_relu_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_add_relu_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _addmm_activation_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, bool use_gelu, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_addmm_activation_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mat1, mat2, beta, alpha, use_gelu, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> _aminmax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_aminmax_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
::std::tuple<at::Tensor &,at::Tensor &> _aminmax_out_dim_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_aminmax_dim_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & _amp_update_scale_(c10::DispatchKeySet ks, at::Tensor & self, at::Tensor & growth_tracker, const at::Tensor & found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_amp_update_scale_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);
  }
  increment_version(self);
  return self;
}
at::Tensor & _amp_update_scale_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & growth_tracker, const at::Tensor & found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_amp_update_scale_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _batch_norm_no_update_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, double momentum, double eps, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_batch_norm_no_update_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight, bias, running_mean, running_var, momentum, eps, out0, out1, out2, out3);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  increment_version(out3);
  return std::forward_as_tuple(out0, out1, out2, out3);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _batch_norm_with_update_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, at::Tensor & running_mean, at::Tensor & running_var, double momentum, double eps, at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd, at::Tensor & reserve) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_batch_norm_with_update_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight, bias, running_mean, running_var, momentum, eps, out, save_mean, save_invstd, reserve);
  }
  increment_version(out);
  increment_version(save_mean);
  increment_version(save_invstd);
  increment_version(reserve);
  return std::forward_as_tuple(out, save_mean, save_invstd, reserve);
}
at::Tensor & _cdist_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & x1, const at::Tensor & x2, double p, const at::Tensor & cdist, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_cdist_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad, x1, x2, p, cdist, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _cdist_forward_out_out(c10::DispatchKeySet ks, const at::Tensor & x1, const at::Tensor & x2, double p, ::std::optional<int64_t> compute_mode, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_cdist_forward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x1, x2, p, compute_mode, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _cholesky_solve_helper_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & A, bool upper, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_cholesky_solve_helper_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, A, upper, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _chunk_cat_out_out(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim, int64_t num_chunks, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_chunk_cat_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, tensors, dim, num_chunks, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _coalesce_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_coalesce_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _coalesced_(c10::DispatchKeySet ks, at::Tensor & self, bool coalesced) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_coalesced_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, coalesced);
  }
  increment_version(self);
  return self;
}
at::Tensor & _coalesced_out_out(c10::DispatchKeySet ks, const at::Tensor & self, bool coalesced, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_coalesced_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, coalesced, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _compute_linear_combination_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & coefficients, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_compute_linear_combination_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, coefficients, out);
  }
  increment_version(out);
  return out;
}
at::Tensor _conj(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::_conj::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (true ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::_ConjViewFunc>();
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::_conj_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & _conj_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_conj_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _conj_physical_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_conj_physical_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _conv_depthwise2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_conv_depthwise2d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, kernel_size, bias, stride, padding, dilation, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _convert_indices_from_coo_to_csr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t size, bool out_int32, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_convert_indices_from_coo_to_csr_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, out_int32, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _convert_indices_from_csr_to_coo_out_out(c10::DispatchKeySet ks, const at::Tensor & crow_indices, const at::Tensor & col_indices, bool out_int32, bool transpose, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_convert_indices_from_csr_to_coo_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, crow_indices, col_indices, out_int32, transpose, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _convolution_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_convolution_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _copy_from_and_resize_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & dst, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_copy_from_and_resize_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dst, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _copy_from_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & dst, bool non_blocking, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_copy_from_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dst, non_blocking, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _ctc_loss_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, const at::Tensor & neg_log_likelihood, const at::Tensor & log_alpha, int64_t blank, bool zero_infinity, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_ctc_loss_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> _ctc_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, bool zero_infinity, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_ctc_loss_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, log_probs, targets, input_lengths, target_lengths, blank, zero_infinity, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
::std::tuple<at::Tensor &,at::Tensor &> _ctc_loss_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & log_probs, const at::Tensor & targets, const at::Tensor & input_lengths, const at::Tensor & target_lengths, int64_t blank, bool zero_infinity, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_ctc_loss_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, log_probs, targets, input_lengths, target_lengths, blank, zero_infinity, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
::std::tuple<at::Tensor &,at::Tensor &> _cudnn_ctc_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_cudnn_ctc_loss_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, log_probs, targets, input_lengths, target_lengths, blank, deterministic, zero_infinity, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & _cudnn_init_dropout_state_out_out(c10::DispatchKeySet ks, double dropout, bool train, int64_t dropout_seed, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_cudnn_init_dropout_state_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, dropout, train, dropout_seed, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _cudnn_rnn_flatten_weight_out_out(c10::DispatchKeySet ks, at::TensorList weight_arr, int64_t weight_stride0, c10::SymInt input_size, int64_t mode, c10::SymInt hidden_size, c10::SymInt proj_size, int64_t num_layers, bool batch_first, bool bidirectional, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_cudnn_rnn_flatten_weight_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, weight_arr, weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _cudnn_rnn_out_out(c10::DispatchKeySet ks, const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const ::std::optional<at::Tensor> & weight_buf, const at::Tensor & hx, const ::std::optional<at::Tensor> & cx, int64_t mode, c10::SymInt hidden_size, c10::SymInt proj_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, c10::SymIntArrayRef batch_sizes, const ::std::optional<at::Tensor> & dropout_state, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3, at::Tensor & out4) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_cudnn_rnn_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, out0, out1, out2, out3, out4);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  increment_version(out3);
  increment_version(out4);
  return std::forward_as_tuple(out0, out1, out2, out3, out4);
}
at::Tensor & _dirichlet_grad_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & alpha, const at::Tensor & total, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_dirichlet_grad_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, alpha, total, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _efficientzerotensor_out_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_efficientzerotensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _embedding_bag_dense_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices, c10::SymInt num_weights, bool scale_grad_by_freq, int64_t mode, const ::std::optional<at::Tensor> & per_sample_weights, int64_t padding_idx, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_embedding_bag_dense_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad, indices, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _embedding_bag_forward_only_out_out(c10::DispatchKeySet ks, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const ::std::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_embedding_bag_forward_only_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx, out0, out1, out2, out3);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  increment_version(out3);
  return std::forward_as_tuple(out0, out1, out2, out3);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _embedding_bag_out_out(c10::DispatchKeySet ks, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const ::std::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_embedding_bag_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx, out0, out1, out2, out3);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  increment_version(out3);
  return std::forward_as_tuple(out0, out1, out2, out3);
}
at::Tensor & _embedding_bag_per_sample_weights_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, int64_t mode, int64_t padding_idx, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_embedding_bag_per_sample_weights_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad, weight, indices, offsets, offset2bag, mode, padding_idx, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _empty_affine_quantized_out_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, double scale, int64_t zero_point, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_empty_affine_quantized_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, size, scale, zero_point, memory_format, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _empty_per_channel_affine_quantized_out_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_empty_per_channel_affine_quantized_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, size, scales, zero_points, axis, memory_format, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _euclidean_dist_out_out(c10::DispatchKeySet ks, const at::Tensor & x1, const at::Tensor & x2, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_euclidean_dist_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x1, x2, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _fake_quantize_learnable_per_channel_affine_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_fake_quantize_learnable_per_channel_affine_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, scale, zero_point, axis, quant_min, quant_max, grad_factor, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _fake_quantize_learnable_per_tensor_affine_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t quant_min, int64_t quant_max, double grad_factor, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_fake_quantize_learnable_per_tensor_affine_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, scale, zero_point, quant_min, quant_max, grad_factor, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> _fake_quantize_per_tensor_affine_cachemask_tensor_qparams_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, const at::Tensor & fake_quant_enabled, int64_t quant_min, int64_t quant_max, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, scale, zero_point, fake_quant_enabled, quant_min, quant_max, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & _fft_c2c_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef dim, int64_t normalization, bool forward, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_fft_c2c_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, normalization, forward, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _fft_c2r_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, c10::SymInt last_dim_size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_fft_c2r_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, normalization, last_dim_size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _fft_r2c_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool onesided, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_fft_r2c_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, normalization, onesided, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _fill_mem_eff_dropout_mask_(c10::DispatchKeySet ks, at::Tensor & self, double dropout_p, int64_t seed, int64_t offset) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_fill_mem_eff_dropout_mask_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dropout_p, seed, offset);
  }
  increment_version(self);
  return self;
}
at::Tensor & _foobar_out_out(c10::DispatchKeySet ks, const at::Tensor & self, bool arg1, bool arg2, bool arg3, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_foobar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, arg1, arg2, arg3, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> _fused_dropout_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double p, ::std::optional<at::Generator> generator, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_fused_dropout_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, p, generator, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
::std::tuple<at::Tensor &,at::Tensor &> _fused_moving_avg_obs_fq_helper_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & observer_on, const at::Tensor & fake_quant_on, at::Tensor & running_min, at::Tensor & running_max, at::Tensor & scale, at::Tensor & zero_point, double averaging_const, int64_t quant_min, int64_t quant_max, int64_t ch_axis, bool per_row_fake_quant, bool symmetric_quant, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_fused_moving_avg_obs_fq_helper_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & _fw_primal_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t level, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_fw_primal_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, level, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _grid_sampler_2d_cpu_fallback_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_grid_sampler_2d_cpu_fallback_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, grid, interpolation_mode, padding_mode, align_corners, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _histogramdd_from_bin_cts_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_histogramdd_from_bin_cts_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, bins, range, weight, density, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _histogramdd_from_bin_tensors_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::TensorList bins, const ::std::optional<at::Tensor> & weight, bool density, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_histogramdd_from_bin_tensors_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, bins, weight, density, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _index_put_impl_(c10::DispatchKeySet ks, at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_index_put_impl_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, indices, values, accumulate, unsafe);
  }
  increment_version(self);
  return self;
}
at::Tensor & _index_put_impl_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_index_put_impl_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, indices, values, accumulate, unsafe, out);
  }
  increment_version(out);
  return out;
}
at::Tensor _indices(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::_indices::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  auto result = as_view(self, _tmp, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false);
  return result;
}
at::Tensor & _indices_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_indices_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _int_mm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_int_mm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mat2, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> _linalg_det_out_result(c10::DispatchKeySet ks, const at::Tensor & A, at::Tensor & result, at::Tensor & LU, at::Tensor & pivots) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_linalg_det_result::redispatch(ks & c10::after_ADInplaceOrView_keyset, A, result, LU, pivots);
  }
  increment_version(result);
  increment_version(LU);
  increment_version(pivots);
  return std::forward_as_tuple(result, LU, pivots);
}
::std::tuple<at::Tensor &,at::Tensor &> _linalg_eigh_out_eigenvalues(c10::DispatchKeySet ks, const at::Tensor & A, c10::string_view UPLO, bool compute_v, at::Tensor & eigenvalues, at::Tensor & eigenvectors) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_linalg_eigh_eigenvalues::redispatch(ks & c10::after_ADInplaceOrView_keyset, A, UPLO, compute_v, eigenvalues, eigenvectors);
  }
  increment_version(eigenvalues);
  increment_version(eigenvectors);
  return std::forward_as_tuple(eigenvalues, eigenvectors);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _linalg_slogdet_out_sign(c10::DispatchKeySet ks, const at::Tensor & A, at::Tensor & sign, at::Tensor & logabsdet, at::Tensor & LU, at::Tensor & pivots) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_linalg_slogdet_sign::redispatch(ks & c10::after_ADInplaceOrView_keyset, A, sign, logabsdet, LU, pivots);
  }
  increment_version(sign);
  increment_version(logabsdet);
  increment_version(LU);
  increment_version(pivots);
  return std::forward_as_tuple(sign, logabsdet, LU, pivots);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _linalg_solve_ex_out_result(c10::DispatchKeySet ks, const at::Tensor & A, const at::Tensor & B, bool left, bool check_errors, at::Tensor & result, at::Tensor & LU, at::Tensor & pivots, at::Tensor & info) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_linalg_solve_ex_result::redispatch(ks & c10::after_ADInplaceOrView_keyset, A, B, left, check_errors, result, LU, pivots, info);
  }
  increment_version(result);
  increment_version(LU);
  increment_version(pivots);
  increment_version(info);
  return std::forward_as_tuple(result, LU, pivots, info);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> _linalg_svd_out_U(c10::DispatchKeySet ks, const at::Tensor & A, bool full_matrices, bool compute_uv, ::std::optional<c10::string_view> driver, at::Tensor & U, at::Tensor & S, at::Tensor & Vh) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_linalg_svd_U::redispatch(ks & c10::after_ADInplaceOrView_keyset, A, full_matrices, compute_uv, driver, U, S, Vh);
  }
  increment_version(U);
  increment_version(S);
  increment_version(Vh);
  return std::forward_as_tuple(U, S, Vh);
}
at::Tensor & _log_softmax_backward_data_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_log_softmax_backward_data_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output, dim, input_dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _log_softmax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_log_softmax_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, half_to_float, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _logcumsumexp_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_logcumsumexp_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _lstm_mps_out_out(c10::DispatchKeySet ks, const at::Tensor & input, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3, at::Tensor & out4, at::Tensor & out5) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_lstm_mps_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first, out0, out1, out2, out3, out4, out5);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  increment_version(out3);
  increment_version(out4);
  increment_version(out5);
  return std::forward_as_tuple(out0, out1, out2, out3, out4, out5);
}
at::Tensor & _make_dual_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & primal, const at::Tensor & tangent, int64_t level, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_make_dual_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, primal, tangent, level, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _make_per_channel_quantized_tensor_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_make_per_channel_quantized_tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, scale, zero_point, axis, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _make_per_tensor_quantized_tensor_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double scale, int64_t zero_point, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_make_per_tensor_quantized_tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, scale, zero_point, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _masked_scale_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, double scale, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_masked_scale_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mask, scale, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _masked_softmax_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, const at::Tensor & mask, ::std::optional<int64_t> dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_masked_softmax_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output, mask, dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _masked_softmax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, ::std::optional<int64_t> dim, ::std::optional<int64_t> mask_type, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_masked_softmax_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mask, dim, mask_type, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _mkldnn_reshape_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef shape, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_mkldnn_reshape_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, shape, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _mkldnn_transpose_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim0, int64_t dim1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_mkldnn_transpose_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim0, dim1);
  }
  increment_version(self);
  return self;
}
at::Tensor & _mkldnn_transpose_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim0, int64_t dim1, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_mkldnn_transpose_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim0, dim1, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _mps_convolution_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_mps_convolution_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, bias, padding, stride, dilation, groups, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _mps_convolution_transpose_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_mps_convolution_transpose_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, padding, output_padding, stride, dilation, groups, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> _native_batch_norm_legit_no_training_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const at::Tensor & running_mean, const at::Tensor & running_var, double momentum, double eps, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_native_batch_norm_legit_no_training_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight, bias, running_mean, running_var, momentum, eps, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> _native_batch_norm_legit_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, at::Tensor & running_mean, at::Tensor & running_var, bool training, double momentum, double eps, at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_native_batch_norm_legit_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight, bias, running_mean, running_var, training, momentum, eps, out, save_mean, save_invstd);
  }
  increment_version(out);
  increment_version(save_mean);
  increment_version(save_invstd);
  return std::forward_as_tuple(out, save_mean, save_invstd);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> _native_batch_norm_legit_out_no_stats_out(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, bool training, double momentum, double eps, at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_native_batch_norm_legit_no_stats_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight, bias, training, momentum, eps, out, save_mean, save_invstd);
  }
  increment_version(out);
  increment_version(save_mean);
  increment_version(save_invstd);
  return std::forward_as_tuple(out, save_mean, save_invstd);
}
::std::tuple<at::Tensor &,at::Tensor &> _native_multi_head_attention_out_out(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t embed_dim, int64_t num_head, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, const ::std::optional<at::Tensor> & mask, bool need_weights, bool average_attn_weights, ::std::optional<int64_t> mask_type, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_native_multi_head_attention_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, query, key, value, embed_dim, num_head, qkv_weight, qkv_bias, proj_weight, proj_bias, mask, need_weights, average_attn_weights, mask_type, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
at::Tensor _neg_view(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::_neg_view::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (true ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::_NegViewViewFunc>();
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::_neg_view_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & _neg_view_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_neg_view_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _nested_from_padded_and_nested_example_out_out(c10::DispatchKeySet ks, const at::Tensor & padded, const at::Tensor & nt_example, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_nested_from_padded_and_nested_example_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, padded, nt_example, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _nested_from_padded_out_out(c10::DispatchKeySet ks, const at::Tensor & padded, const at::Tensor & cpu_nested_shape_example, bool fuse_transform_0213, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_nested_from_padded_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, padded, cpu_nested_shape_example, fuse_transform_0213, out);
  }
  increment_version(out);
  return out;
}
at::Tensor _nested_get_values(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::_nested_get_values::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (true ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::_NestedGetValuesViewFunc>();
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::_nested_get_values_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & _nested_get_values_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_nested_get_values_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _nested_tensor_from_mask_out_out(c10::DispatchKeySet ks, const at::Tensor & t, const at::Tensor & mask, bool mask_check, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_nested_tensor_from_mask_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, t, mask, mask_check, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _nested_tensor_from_tensor_list_out_out(c10::DispatchKeySet ks, at::TensorList list, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_nested_tensor_from_tensor_list_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, list, dtype, layout, device, pin_memory, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _nested_tensor_size_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_nested_tensor_size_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _nested_tensor_storage_offsets_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_nested_tensor_storage_offsets_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _nested_tensor_strides_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_nested_tensor_strides_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor _nested_view_from_buffer(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & nested_size, const at::Tensor & nested_strides, const at::Tensor & offsets) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::_nested_view_from_buffer::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, nested_size, nested_strides, offsets);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (true ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::_NestedViewFromBufferViewFunc>(nested_size, nested_strides, offsets);
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::_nested_view_from_buffer_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, nested_size, nested_strides, offsets);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & _nested_view_from_buffer_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & nested_size, const at::Tensor & nested_strides, const at::Tensor & offsets, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_nested_view_from_buffer_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, nested_size, nested_strides, offsets, out);
  }
  increment_version(out);
  return out;
}
at::Tensor _nested_view_from_jagged(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & offsets, const at::Tensor & dummy, const ::std::optional<at::Tensor> & lengths, int64_t ragged_idx, const ::std::optional<at::Tensor> & min_seqlen, const ::std::optional<at::Tensor> & max_seqlen) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::_nested_view_from_jagged::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, offsets, dummy, lengths, ragged_idx, min_seqlen, max_seqlen);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (true ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::_NestedViewFromJaggedViewFunc>(offsets, dummy, lengths, ragged_idx, min_seqlen, max_seqlen);
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::_nested_view_from_jagged_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, offsets, dummy, lengths, ragged_idx, min_seqlen, max_seqlen);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & _nested_view_from_jagged_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & offsets, const at::Tensor & dummy, const ::std::optional<at::Tensor> & lengths, int64_t ragged_idx, const ::std::optional<at::Tensor> & min_seqlen, const ::std::optional<at::Tensor> & max_seqlen, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_nested_view_from_jagged_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, offsets, dummy, lengths, ragged_idx, min_seqlen, max_seqlen, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _new_zeros_with_same_feature_meta_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, int64_t self_num_batch_dims, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_new_zeros_with_same_feature_meta_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, self_num_batch_dims, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _nnpack_spatial_convolution_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_nnpack_spatial_convolution_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight, bias, padding, stride, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> _pack_padded_sequence_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & lengths, bool batch_first, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_pack_padded_sequence_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, lengths, batch_first, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & _pdist_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & self, double p, const at::Tensor & pdist, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_pdist_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad, self, p, pdist, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _pdist_forward_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double p, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_pdist_forward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, p, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _pin_memory_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Device> device, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_pin_memory_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, device, out);
  }
  increment_version(out);
  return out;
}
at::Tensor _reshape_alias(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::_reshape_alias::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, stride);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    auto size_vec = size.vec();
    auto stride_vec = stride.vec();
    func = std::make_unique<torch::autograd::generated::_ReshapeAliasViewFunc>(size, stride);
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::_reshape_alias_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, size_vec, stride_vec);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & _reshape_alias_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_reshape_alias_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, stride, out);
  }
  increment_version(out);
  return out;
}
const at::Tensor & _resize_output_(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, at::Device device) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_resize_output_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, device);
  }
  increment_version(self);
  return self;
}
const at::Tensor & _resize_output_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, at::Device device, const at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_resize_output_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, device, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _sample_dirichlet_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_sample_dirichlet_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _scaled_mm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2, const at::Tensor & scale_a, const at::Tensor & scale_b, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & scale_result, ::std::optional<at::ScalarType> out_dtype, bool use_fast_accum, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_scaled_mm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _segment_reduce_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & output, const at::Tensor & data, c10::string_view reduce, const ::std::optional<at::Tensor> & lengths, const ::std::optional<at::Tensor> & offsets, int64_t axis, const ::std::optional<at::Scalar> & initial, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_segment_reduce_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad, output, data, reduce, lengths, offsets, axis, initial, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> _slow_conv2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_slow_conv2d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, weight, kernel_size, stride, padding, grad_input, grad_weight, grad_bias);
  }
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> _slow_conv2d_backward_out_output_mask_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_slow_conv2d_backward_output_mask_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, weight, kernel_size, stride, padding, output_mask, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
at::Tensor & _slow_conv2d_forward_out_output(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, at::Tensor & output) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_slow_conv2d_forward_output::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, kernel_size, bias, stride, padding, output);
  }
  increment_version(output);
  return output;
}
at::Tensor & _softmax_backward_data_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_softmax_backward_data_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output, dim, input_dtype, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & _softmax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_softmax_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, half_to_float, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _sparse_addmm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_sparse_addmm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mat1, mat2, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _sparse_broadcast_to_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_sparse_broadcast_to_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _sparse_coo_tensor_with_dims_and_tensors_out_out(c10::DispatchKeySet ks, int64_t sparse_dim, int64_t dense_dim, c10::SymIntArrayRef size, const at::Tensor & indices, const at::Tensor & values, ::std::optional<bool> is_coalesced, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_sparse_coo_tensor_with_dims_and_tensors_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, sparse_dim, dense_dim, size, indices, values, is_coalesced, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _sparse_coo_tensor_with_dims_out_out(c10::DispatchKeySet ks, int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_sparse_coo_tensor_with_dims_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, sparse_dim, dense_dim, size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _sparse_csr_prod_out_dim_dtype_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_sparse_csr_prod_dim_dtype_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _sparse_csr_sum_out_dim_dtype_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_sparse_csr_sum_dim_dtype_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _sparse_log_softmax_backward_data_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_sparse_log_softmax_backward_data_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output, dim, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _sparse_log_softmax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_sparse_log_softmax_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, half_to_float, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _sparse_mask_projection_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, bool accumulate_matches, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_sparse_mask_projection_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mask, accumulate_matches, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _sparse_softmax_backward_data_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_sparse_softmax_backward_data_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output, dim, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _sparse_softmax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_sparse_softmax_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, half_to_float, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _sparse_sparse_matmul_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_sparse_sparse_matmul_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _sparse_sum_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & self, at::IntArrayRef dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_sparse_sum_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad, self, dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _sparse_sum_out_dim_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_sparse_sum_dim_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _spdiags_out_out(c10::DispatchKeySet ks, const at::Tensor & diagonals, const at::Tensor & offsets, at::IntArrayRef shape, ::std::optional<at::Layout> layout, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_spdiags_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, diagonals, offsets, shape, layout, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _stack_out_out(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_stack_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, tensors, dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _standard_gamma_grad_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & output, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_standard_gamma_grad_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _standard_gamma_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_standard_gamma_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _test_autograd_multiple_dispatch_out_fullcoverage_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_test_autograd_multiple_dispatch_fullcoverage_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor _test_autograd_multiple_dispatch_view(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::_test_autograd_multiple_dispatch_view::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::_TestAutogradMultipleDispatchViewViewFunc>();
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::_test_autograd_multiple_dispatch_view_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & _test_autograd_multiple_dispatch_view_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_test_autograd_multiple_dispatch_view_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _test_functorch_fallback_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_test_functorch_fallback_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _test_optional_filled_intlist_out_out(c10::DispatchKeySet ks, const at::Tensor & values, at::OptionalIntArrayRef addends, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_test_optional_filled_intlist_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, values, addends, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _test_optional_floatlist_out_out(c10::DispatchKeySet ks, const at::Tensor & values, ::std::optional<at::ArrayRef<double>> addends, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_test_optional_floatlist_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, values, addends, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _test_optional_intlist_out_out(c10::DispatchKeySet ks, const at::Tensor & values, at::OptionalIntArrayRef addends, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_test_optional_intlist_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, values, addends, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _test_warn_in_autograd_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_test_warn_in_autograd_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _thnn_fused_gru_cell_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_hy, const at::Tensor & workspace, bool has_bias, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3, at::Tensor & out4) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_thnn_fused_gru_cell_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_hy, workspace, has_bias, out0, out1, out2, out3, out4);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  increment_version(out3);
  increment_version(out4);
  return std::forward_as_tuple(out0, out1, out2, out3, out4);
}
::std::tuple<at::Tensor &,at::Tensor &> _thnn_fused_gru_cell_out_out(c10::DispatchKeySet ks, const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & hx, const ::std::optional<at::Tensor> & input_bias, const ::std::optional<at::Tensor> & hidden_bias, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_thnn_fused_gru_cell_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input_gates, hidden_gates, hx, input_bias, hidden_bias, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> _thnn_fused_lstm_cell_backward_impl_out_out(c10::DispatchKeySet ks, const ::std::optional<at::Tensor> & grad_hy, const ::std::optional<at::Tensor> & grad_cy, const at::Tensor & cx, const at::Tensor & cy, const at::Tensor & workspace, bool has_bias, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_thnn_fused_lstm_cell_backward_impl_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_hy, grad_cy, cx, cy, workspace, has_bias, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> _thnn_fused_lstm_cell_out_out(c10::DispatchKeySet ks, const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & cx, const ::std::optional<at::Tensor> & input_bias, const ::std::optional<at::Tensor> & hidden_bias, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_thnn_fused_lstm_cell_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input_gates, hidden_gates, cx, input_bias, hidden_bias, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
at::Tensor & _to_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, bool non_blocking, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_to_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, non_blocking, memory_format, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _to_dense_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<bool> masked_grad, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_to_dense_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dtype, masked_grad, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _to_sparse_bsc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef blocksize, ::std::optional<int64_t> dense_dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_to_sparse_bsc_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, blocksize, dense_dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _to_sparse_bsr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef blocksize, ::std::optional<int64_t> dense_dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_to_sparse_bsr_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, blocksize, dense_dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _to_sparse_csc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<int64_t> dense_dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_to_sparse_csc_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dense_dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _to_sparse_csr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<int64_t> dense_dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_to_sparse_csr_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dense_dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _to_sparse_out_sparse_dim_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t sparse_dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_to_sparse_sparse_dim_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, sparse_dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _to_sparse_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Layout> layout, at::OptionalIntArrayRef blocksize, ::std::optional<int64_t> dense_dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_to_sparse_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, layout, blocksize, dense_dim, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> _transform_bias_rescale_qkv_out_out(c10::DispatchKeySet ks, const at::Tensor & qkv, const at::Tensor & qkv_bias, int64_t num_heads, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_transform_bias_rescale_qkv_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, qkv, qkv_bias, num_heads, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
at::Tensor & _transformer_encoder_layer_fwd_out_out(c10::DispatchKeySet ks, const at::Tensor & src, int64_t embed_dim, int64_t num_heads, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, bool use_gelu, bool norm_first, double eps, const at::Tensor & norm_weight_1, const at::Tensor & norm_bias_1, const at::Tensor & norm_weight_2, const at::Tensor & norm_bias_2, const at::Tensor & ffn_weight_1, const at::Tensor & ffn_bias_1, const at::Tensor & ffn_weight_2, const at::Tensor & ffn_bias_2, const ::std::optional<at::Tensor> & mask, ::std::optional<int64_t> mask_type, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_transformer_encoder_layer_fwd_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, src, embed_dim, num_heads, qkv_weight, qkv_bias, proj_weight, proj_bias, use_gelu, norm_first, eps, norm_weight_1, norm_bias_1, norm_weight_2, norm_bias_2, ffn_weight_1, ffn_bias_1, ffn_weight_2, ffn_bias_2, mask, mask_type, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _trilinear_out_out(c10::DispatchKeySet ks, const at::Tensor & i1, const at::Tensor & i2, const at::Tensor & i3, at::IntArrayRef expand1, at::IntArrayRef expand2, at::IntArrayRef expand3, at::IntArrayRef sumdim, int64_t unroll_dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_trilinear_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _triton_multi_head_attention_out_out(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t embed_dim, int64_t num_head, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, const ::std::optional<at::Tensor> & mask, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_triton_multi_head_attention_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, query, key, value, embed_dim, num_head, qkv_weight, qkv_bias, proj_weight, proj_bias, mask, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _triton_scaled_dot_attention_out_out(c10::DispatchKeySet ks, const at::Tensor & q, const at::Tensor & k, const at::Tensor & v, double dropout_p, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_triton_scaled_dot_attention_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, q, k, v, dropout_p, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> _unique2_out_out(c10::DispatchKeySet ks, const at::Tensor & self, bool sorted, bool return_inverse, bool return_counts, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_unique2_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, sorted, return_inverse, return_counts, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
::std::tuple<at::Tensor &,at::Tensor &> _unique_out_out(c10::DispatchKeySet ks, const at::Tensor & self, bool sorted, bool return_inverse, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_unique_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, sorted, return_inverse, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & _unsafe_view_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_unsafe_view_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _upsample_bicubic2d_aa_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_upsample_bicubic2d_aa_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & _upsample_bicubic2d_aa_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_upsample_bicubic2d_aa_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, align_corners, scales_h, scales_w, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _upsample_bilinear2d_aa_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_upsample_bilinear2d_aa_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & _upsample_bilinear2d_aa_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_upsample_bilinear2d_aa_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, align_corners, scales_h, scales_w, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _upsample_nearest_exact1d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, ::std::optional<double> scales, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_upsample_nearest_exact1d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output_size, input_size, scales, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & _upsample_nearest_exact1d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_upsample_nearest_exact1d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, scales, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _upsample_nearest_exact2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_upsample_nearest_exact2d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output_size, input_size, scales_h, scales_w, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & _upsample_nearest_exact2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_upsample_nearest_exact2d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, scales_h, scales_w, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _upsample_nearest_exact3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, ::std::optional<double> scales_d, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_upsample_nearest_exact3d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & _upsample_nearest_exact3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_d, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_upsample_nearest_exact3d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, scales_d, scales_h, scales_w, out);
  }
  increment_version(out);
  return out;
}
at::Tensor _values(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::_values::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  auto result = as_view(self, _tmp, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false);
  return result;
}
at::Tensor & _values_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_values_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> _weight_norm_interface_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_w, const at::Tensor & saved_v, const at::Tensor & saved_g, const at::Tensor & saved_norms, int64_t dim, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_weight_norm_interface_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_w, saved_v, saved_g, saved_norms, dim, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
::std::tuple<at::Tensor &,at::Tensor &> _weight_norm_interface_out_out(c10::DispatchKeySet ks, const at::Tensor & v, const at::Tensor & g, int64_t dim, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::_weight_norm_interface_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, v, g, dim, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & abs_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::abs_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & abs_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::abs_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & acos_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::acos_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & acos_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::acos_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & acosh_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::acosh_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & acosh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::acosh_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & adaptive_avg_pool2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::adaptive_avg_pool2d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & adaptive_avg_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::adaptive_avg_pool3d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & adaptive_avg_pool3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::adaptive_avg_pool3d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & adaptive_max_pool2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::adaptive_max_pool2d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, indices, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> adaptive_max_pool2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::adaptive_max_pool2d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, out, indices);
  }
  increment_version(out);
  increment_version(indices);
  return std::forward_as_tuple(out, indices);
}
at::Tensor & adaptive_max_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::adaptive_max_pool3d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, indices, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> adaptive_max_pool3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::adaptive_max_pool3d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, out, indices);
  }
  increment_version(out);
  increment_version(indices);
  return std::forward_as_tuple(out, indices);
}
at::Tensor & add__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::add__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & add__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::add__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & add_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::add_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & add_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::add_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & addbmm_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::addbmm_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, batch1, batch2, beta, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & addbmm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::addbmm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, batch1, batch2, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & addcdiv_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::addcdiv_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, tensor1, tensor2, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & addcdiv_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::addcdiv_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, tensor1, tensor2, value, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & addcmul_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::addcmul_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, tensor1, tensor2, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & addcmul_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::addcmul_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, tensor1, tensor2, value, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & addmm_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::addmm_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mat1, mat2, beta, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & addmm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::addmm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mat1, mat2, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & addmv_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::addmv_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mat, vec, beta, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & addmv_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::addmv_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mat, vec, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & addr_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::addr_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, vec1, vec2, beta, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & addr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::addr_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, vec1, vec2, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & affine_grid_generator_out_out(c10::DispatchKeySet ks, const at::Tensor & theta, c10::SymIntArrayRef size, bool align_corners, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::affine_grid_generator_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, theta, size, align_corners, out);
  }
  increment_version(out);
  return out;
}
at::Tensor alias(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::alias::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::AliasViewFunc>();
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::alias_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & alias_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::alias_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & all_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::all_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & all_out_dims_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::all_dims_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & all_out_all_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::all_all_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & amax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::amax_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & amin_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::amin_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> aminmax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<int64_t> dim, bool keepdim, at::Tensor & min, at::Tensor & max) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::aminmax_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, min, max);
  }
  increment_version(min);
  increment_version(max);
  return std::forward_as_tuple(min, max);
}
at::Tensor & angle_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::angle_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & any_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::any_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & any_out_dims_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::any_dims_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & any_out_all_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::any_all_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & arange_out_out(c10::DispatchKeySet ks, const at::Scalar & end, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::arange_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, end, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & arange_out_start_out(c10::DispatchKeySet ks, const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::arange_start_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, start, end, step, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & argmax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::argmax_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & argmin_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::argmin_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor as_strided(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<c10::SymInt> storage_offset) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::as_strided::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, stride, storage_offset);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    auto size_vec = size.vec();
    auto stride_vec = stride.vec();
    func = std::make_unique<torch::autograd::generated::AsStridedViewFunc>(size, stride, storage_offset);
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::as_strided_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, size_vec, stride_vec, storage_offset);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
const at::Tensor & as_strided_(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<c10::SymInt> storage_offset) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::as_strided_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, stride, storage_offset);
  }
  increment_version(self);
  return self;
}
at::Tensor & as_strided_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<c10::SymInt> storage_offset, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::as_strided_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, stride, storage_offset, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & as_strided_scatter_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & src, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<c10::SymInt> storage_offset, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::as_strided_scatter_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, src, size, stride, storage_offset, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & asin_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::asin_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & asin_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::asin_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & asinh_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::asinh_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & asinh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::asinh_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & atan2_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::atan2_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & atan2_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::atan2_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & atan_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::atan_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & atan_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::atan_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & atanh_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::atanh_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & atanh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::atanh_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & avg_pool2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, ::std::optional<int64_t> divisor_override, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::avg_pool2d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & avg_pool2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, ::std::optional<int64_t> divisor_override, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::avg_pool2d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & avg_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, ::std::optional<int64_t> divisor_override, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::avg_pool3d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & avg_pool3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, ::std::optional<int64_t> divisor_override, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::avg_pool3d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & baddbmm_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::baddbmm_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, batch1, batch2, beta, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & baddbmm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::baddbmm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, batch1, batch2, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bartlett_window_out_out(c10::DispatchKeySet ks, int64_t window_length, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bartlett_window_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, window_length, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bartlett_window_out_periodic_out(c10::DispatchKeySet ks, int64_t window_length, bool periodic, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bartlett_window_periodic_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, window_length, periodic, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & batch_norm_backward_elemt_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const ::std::optional<at::Tensor> & weight, const at::Tensor & sum_dy, const at::Tensor & sum_dy_xmu, const at::Tensor & count, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::batch_norm_backward_elemt_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> batch_norm_backward_reduce_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const ::std::optional<at::Tensor> & weight, bool input_g, bool weight_g, bool bias_g, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::batch_norm_backward_reduce_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g, out0, out1, out2, out3);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  increment_version(out3);
  return std::forward_as_tuple(out0, out1, out2, out3);
}
at::Tensor & batch_norm_elemt_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::batch_norm_elemt_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight, bias, mean, invstd, eps, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> batch_norm_gather_stats_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, double momentum, double eps, int64_t count, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::batch_norm_gather_stats_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, mean, invstd, running_mean, running_var, momentum, eps, count, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
::std::tuple<at::Tensor &,at::Tensor &> batch_norm_gather_stats_with_counts_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, double momentum, double eps, const at::Tensor & counts, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::batch_norm_gather_stats_with_counts_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, mean, invstd, running_mean, running_var, momentum, eps, counts, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
::std::tuple<at::Tensor &,at::Tensor &> batch_norm_stats_out_out(c10::DispatchKeySet ks, const at::Tensor & input, double eps, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::batch_norm_stats_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, eps, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
::std::tuple<at::Tensor &,at::Tensor &> batch_norm_update_stats_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, double momentum, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::batch_norm_update_stats_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, running_mean, running_var, momentum, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & bernoulli__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & p, ::std::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bernoulli__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, p, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & bernoulli__float(c10::DispatchKeySet ks, at::Tensor & self, double p, ::std::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bernoulli__float::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, p, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & bernoulli_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bernoulli_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bernoulli_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & p, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bernoulli_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, p, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bernoulli_out_float_out(c10::DispatchKeySet ks, const at::Tensor & self, double p, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bernoulli_float_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, p, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & binary_cross_entropy_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::binary_cross_entropy_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, target, weight, reduction, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & binary_cross_entropy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::binary_cross_entropy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, target, weight, reduction, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & binary_cross_entropy_with_logits_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & pos_weight, int64_t reduction, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::binary_cross_entropy_with_logits_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, target, weight, pos_weight, reduction, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bincount_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Tensor> & weights, int64_t minlength, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bincount_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weights, minlength, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & binomial_out_out(c10::DispatchKeySet ks, const at::Tensor & count, const at::Tensor & prob, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::binomial_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, count, prob, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_and__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_and__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & bitwise_and__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_and__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & bitwise_and_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_and_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_and_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_and_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_and_out_Scalar_Tensor_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_and_Scalar_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_left_shift__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_left_shift__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & bitwise_left_shift__Tensor_Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_left_shift__Tensor_Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & bitwise_left_shift_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_left_shift_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_left_shift_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_left_shift_Tensor_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_left_shift_out_Scalar_Tensor_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_left_shift_Scalar_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_not_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_not_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & bitwise_not_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_not_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_or__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_or__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & bitwise_or__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_or__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & bitwise_or_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_or_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_or_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_or_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_or_out_Scalar_Tensor_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_or_Scalar_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_right_shift__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_right_shift__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & bitwise_right_shift__Tensor_Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_right_shift__Tensor_Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & bitwise_right_shift_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_right_shift_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_right_shift_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_right_shift_Tensor_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_right_shift_out_Scalar_Tensor_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_right_shift_Scalar_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_xor__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_xor__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & bitwise_xor__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_xor__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & bitwise_xor_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_xor_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_xor_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_xor_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_xor_out_Scalar_Tensor_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bitwise_xor_Scalar_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & blackman_window_out_out(c10::DispatchKeySet ks, int64_t window_length, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::blackman_window_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, window_length, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & blackman_window_out_periodic_out(c10::DispatchKeySet ks, int64_t window_length, bool periodic, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::blackman_window_periodic_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, window_length, periodic, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & block_diag_out_out(c10::DispatchKeySet ks, at::TensorList tensors, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::block_diag_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, tensors, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bmm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bmm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mat2, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bucketize_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bucketize_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, boundaries, out_int32, right, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bucketize_out_Scalar_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & boundaries, bool out_int32, bool right, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::bucketize_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, boundaries, out_int32, right, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & cat_out_out(c10::DispatchKeySet ks, const at::ITensorListRef & tensors, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cat_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, tensors, dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & cauchy_(c10::DispatchKeySet ks, at::Tensor & self, double median, double sigma, ::std::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cauchy_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, median, sigma, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & cauchy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double median, double sigma, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cauchy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, median, sigma, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor ccol_indices(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::ccol_indices::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  auto result = as_view(self, _tmp, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false);
  return result;
}
at::Tensor & ccol_indices_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::ccol_indices_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & ceil_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::ceil_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & ceil_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::ceil_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & celu_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::celu_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & celu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::celu_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & channel_shuffle_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt groups, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::channel_shuffle_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, groups, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & cholesky_inverse_out_out(c10::DispatchKeySet ks, const at::Tensor & self, bool upper, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cholesky_inverse_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, upper, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & cholesky_out_out(c10::DispatchKeySet ks, const at::Tensor & self, bool upper, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cholesky_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, upper, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & cholesky_solve_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & input2, bool upper, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cholesky_solve_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, input2, upper, out);
  }
  increment_version(out);
  return out;
}
::std::vector<at::Tensor> chunk(c10::DispatchKeySet ks, const at::Tensor & self, int64_t chunks, int64_t dim) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::chunk::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, chunks, dim);
  })();
  for (auto view_idx : c10::irange(_tmp.size())) {
    std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
    std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
    if (false ||
        !self.unsafeGetTensorImpl()->support_as_strided() ||
        self.unsafeGetTensorImpl()->is_python_dispatch() ||
        c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
      func = std::make_unique<torch::autograd::generated::ChunkViewFunc>(chunks, dim, view_idx);
      rev_func = [=](const at::Tensor& input_view) {
        return at::functionalization::FunctionalInverses::chunk_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, view_idx, chunks, dim);
      };
    }
    as_view(/* base */ self, /* output */ _tmp[view_idx], /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::MULTI_OUTPUT_NODE : CreationMeta::NO_GRAD_MODE));
  }
  auto result = std::move(_tmp);
  return result;
}
at::Tensor & clamp_(c10::DispatchKeySet ks, at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::clamp_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, min, max);
  }
  increment_version(self);
  return self;
}
at::Tensor & clamp__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::clamp__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, min, max);
  }
  increment_version(self);
  return self;
}
at::Tensor & clamp_max_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & max) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::clamp_max_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, max);
  }
  increment_version(self);
  return self;
}
at::Tensor & clamp_max__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & max) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::clamp_max__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, max);
  }
  increment_version(self);
  return self;
}
at::Tensor & clamp_max_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & max, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::clamp_max_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, max, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & clamp_max_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & max, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::clamp_max_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, max, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & clamp_min_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & min) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::clamp_min_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, min);
  }
  increment_version(self);
  return self;
}
at::Tensor & clamp_min__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & min) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::clamp_min__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, min);
  }
  increment_version(self);
  return self;
}
at::Tensor & clamp_min_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & min, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::clamp_min_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, min, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & clamp_min_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & min, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::clamp_min_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, min, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & clamp_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::clamp_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, min, max, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & clamp_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::clamp_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, min, max, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & clone_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::clone_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, memory_format, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & col2im_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::col2im_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, kernel_size, dilation, padding, stride, out);
  }
  increment_version(out);
  return out;
}
at::Tensor col_indices(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::col_indices::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  auto result = as_view(self, _tmp, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false);
  return result;
}
at::Tensor & col_indices_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::col_indices_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & complex_out_out(c10::DispatchKeySet ks, const at::Tensor & real, const at::Tensor & imag, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::complex_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, real, imag, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & conj_physical_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::conj_physical_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & conj_physical_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::conj_physical_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & constant_pad_nd_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef pad, const at::Scalar & value, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::constant_pad_nd_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, pad, value, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & conv_depthwise3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::conv_depthwise3d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, kernel_size, bias, stride, padding, dilation, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & conv_tbc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const at::Tensor & bias, int64_t pad, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::conv_tbc_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, bias, pad, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> convolution_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::OptionalSymIntArrayRef bias_sizes, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::convolution_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> convolution_backward_overrideable_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::convolution_backward_overrideable_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
at::Tensor & convolution_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::convolution_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & convolution_overrideable_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::convolution_overrideable_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & src, bool non_blocking, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, src, non_blocking, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & copy_sparse_to_sparse_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & src, bool non_blocking) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::copy_sparse_to_sparse_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, src, non_blocking);
  }
  increment_version(self);
  return self;
}
at::Tensor & copy_sparse_to_sparse_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & src, bool non_blocking, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::copy_sparse_to_sparse_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, src, non_blocking, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & copysign__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::copysign__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & copysign__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::copysign__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & copysign_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::copysign_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & copysign_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::copysign_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & cos_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cos_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & cos_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cos_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & cosh_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cosh_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & cosh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cosh_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & count_nonzero_out_dim_IntList_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::count_nonzero_dim_IntList_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & count_nonzero_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<int64_t> dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::count_nonzero_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor crow_indices(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::crow_indices::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  auto result = as_view(self, _tmp, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false);
  return result;
}
at::Tensor & crow_indices_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::crow_indices_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & cudnn_affine_grid_generator_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cudnn_affine_grid_generator_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad, N, C, H, W, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & cudnn_affine_grid_generator_out_out(c10::DispatchKeySet ks, const at::Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cudnn_affine_grid_generator_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, theta, N, C, H, W, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> cudnn_batch_norm_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, const ::std::optional<at::Tensor> & save_mean, const ::std::optional<at::Tensor> & save_var, double epsilon, const at::Tensor & reserveSpace, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cudnn_batch_norm_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> cudnn_batch_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double exponential_average_factor, double epsilon, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cudnn_batch_norm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon, out0, out1, out2, out3);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  increment_version(out3);
  return std::forward_as_tuple(out0, out1, out2, out3);
}
at::Tensor & cudnn_convolution_add_relu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const at::Tensor & z, const ::std::optional<at::Scalar> & alpha, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, c10::SymInt groups, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cudnn_convolution_add_relu_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, z, alpha, bias, stride, padding, dilation, groups, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & cudnn_convolution_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, bool benchmark, bool deterministic, bool allow_tf32, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cudnn_convolution_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & cudnn_convolution_relu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, c10::SymInt groups, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cudnn_convolution_relu_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, bias, stride, padding, dilation, groups, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & cudnn_convolution_transpose_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, bool benchmark, bool deterministic, bool allow_tf32, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cudnn_convolution_transpose_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> cudnn_grid_sampler_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & grid, const at::Tensor & grad_output, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cudnn_grid_sampler_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, grid, grad_output, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & cudnn_grid_sampler_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & grid, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cudnn_grid_sampler_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, grid, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> cummax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & values, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cummax_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor &,at::Tensor &> cummin_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & values, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cummin_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
at::Tensor & cumprod_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cumprod_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, dtype);
  }
  increment_version(self);
  return self;
}
at::Tensor & cumprod_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cumprod_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & cumsum_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cumsum_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, dtype);
  }
  increment_version(self);
  return self;
}
at::Tensor & cumsum_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::cumsum_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & deg2rad_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::deg2rad_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & deg2rad_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::deg2rad_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & dequantize_out_self_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::dequantize_self_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & detach_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::detach_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & diag_embed_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::diag_embed_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, offset, dim1, dim2, out);
  }
  increment_version(out);
  return out;
}
at::Tensor diagonal(c10::DispatchKeySet ks, const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::diagonal::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, offset, dim1, dim2);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::DiagonalViewFunc>(offset, dim1, dim2);
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::diagonal_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, offset, dim1, dim2);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & diagonal_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef input_sizes, int64_t offset, int64_t dim1, int64_t dim2, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::diagonal_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, input_sizes, offset, dim1, dim2, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & diagonal_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::diagonal_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, offset, dim1, dim2, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & diagonal_scatter_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & src, int64_t offset, int64_t dim1, int64_t dim2, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::diagonal_scatter_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, src, offset, dim1, dim2, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & digamma_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::digamma_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & digamma_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::digamma_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & dist_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & p, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::dist_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, p, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & div__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::div__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & div__Tensor_mode(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::div__Tensor_mode::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, rounding_mode);
  }
  increment_version(self);
  return self;
}
at::Tensor & div__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::div__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & div__Scalar_mode(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other, ::std::optional<c10::string_view> rounding_mode) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::div__Scalar_mode::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, rounding_mode);
  }
  increment_version(self);
  return self;
}
at::Tensor & div_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::div_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & div_out_out_mode(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::div_out_mode::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, rounding_mode, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & div_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::div_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & div_out_Scalar_mode_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, ::std::optional<c10::string_view> rounding_mode, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::div_Scalar_mode_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, rounding_mode, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & dot_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & tensor, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::dot_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, tensor, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & elu_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::elu_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, alpha, scale, input_scale);
  }
  increment_version(self);
  return self;
}
at::Tensor & elu_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, bool is_result, const at::Tensor & self_or_result, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::elu_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, alpha, scale, input_scale, is_result, self_or_result, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & elu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::elu_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, alpha, scale, input_scale, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & embedding_dense_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & indices, c10::SymInt num_weights, c10::SymInt padding_idx, bool scale_grad_by_freq, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::embedding_dense_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & embedding_out_out(c10::DispatchKeySet ks, const at::Tensor & weight, const at::Tensor & indices, c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::embedding_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, weight, indices, padding_idx, scale_grad_by_freq, sparse, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & embedding_renorm_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & indices, double max_norm, double norm_type) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::embedding_renorm_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, indices, max_norm, norm_type);
  }
  increment_version(self);
  return self;
}
at::Tensor & embedding_renorm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & indices, double max_norm, double norm_type, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::embedding_renorm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, indices, max_norm, norm_type, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & empty_like_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::empty_like_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, memory_format, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & empty_out_names_out(c10::DispatchKeySet ks, at::IntArrayRef size, ::std::optional<at::DimnameList> names, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::empty_names_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, size, names, memory_format, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & empty_permuted_out_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, at::IntArrayRef physical_layout, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::empty_permuted_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, size, physical_layout, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & empty_quantized_out_out(c10::DispatchKeySet ks, at::IntArrayRef size, const at::Tensor & qtensor, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::empty_quantized_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, size, qtensor, memory_format, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & empty_strided_out_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::empty_strided_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, size, stride, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & eq__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::eq__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & eq__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::eq__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & eq_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::eq_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & eq_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::eq_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & erf_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::erf_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & erf_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::erf_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & erfc_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::erfc_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & erfc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::erfc_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & erfinv_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::erfinv_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & erfinv_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::erfinv_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & exp2_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::exp2_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & exp2_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::exp2_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & exp_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::exp_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & exp_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::exp_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor expand(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, bool implicit) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::expand::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, implicit);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    auto size_vec = size.vec();
    func = std::make_unique<torch::autograd::generated::ExpandViewFunc>(size, implicit);
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::expand_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, size_vec, implicit);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & expand_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, bool implicit, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::expand_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, implicit, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & expm1_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::expm1_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & expm1_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::expm1_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & exponential_(c10::DispatchKeySet ks, at::Tensor & self, double lambd, ::std::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::exponential_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, lambd, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & exponential_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double lambd, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::exponential_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, lambd, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & eye_out_out(c10::DispatchKeySet ks, c10::SymInt n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::eye_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & eye_out_m_out(c10::DispatchKeySet ks, c10::SymInt n, c10::SymInt m, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::eye_m_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, n, m, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> fake_quantize_per_channel_affine_cachemask_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::fake_quantize_per_channel_affine_cachemask_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, scale, zero_point, axis, quant_min, quant_max, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
::std::tuple<at::Tensor &,at::Tensor &> fake_quantize_per_tensor_affine_cachemask_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::fake_quantize_per_tensor_affine_cachemask_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, scale, zero_point, quant_min, quant_max, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & fft_fftfreq_out_out(c10::DispatchKeySet ks, int64_t n, double d, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::fft_fftfreq_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, n, d, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & fft_rfftfreq_out_out(c10::DispatchKeySet ks, int64_t n, double d, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::fft_rfftfreq_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, n, d, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & fill__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::fill__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & fill__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::fill__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & fill_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & value, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::fill_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, value, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & fill_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & value, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::fill_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, value, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & flip_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dims, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::flip_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dims, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & floor_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::floor_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & floor_divide__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::floor_divide__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & floor_divide__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::floor_divide__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & floor_divide_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::floor_divide_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & floor_divide_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::floor_divide_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & floor_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::floor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & fmax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::fmax_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & fmin_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::fmin_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & fmod__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::fmod__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & fmod__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::fmod__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & fmod_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::fmod_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & fmod_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::fmod_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & frac_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::frac_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & frac_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::frac_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & fractional_max_pool2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::fractional_max_pool2d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, kernel_size, output_size, indices, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> fractional_max_pool2d_out_output(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples, at::Tensor & output, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::fractional_max_pool2d_output::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, kernel_size, output_size, random_samples, output, indices);
  }
  increment_version(output);
  increment_version(indices);
  return std::forward_as_tuple(output, indices);
}
at::Tensor & fractional_max_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::fractional_max_pool3d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, kernel_size, output_size, indices, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> fractional_max_pool3d_out_output(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples, at::Tensor & output, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::fractional_max_pool3d_output::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, kernel_size, output_size, random_samples, output, indices);
  }
  increment_version(output);
  increment_version(indices);
  return std::forward_as_tuple(output, indices);
}
::std::tuple<at::Tensor &,at::Tensor &> frexp_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & mantissa, at::Tensor & exponent) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::frexp_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mantissa, exponent);
  }
  increment_version(mantissa);
  increment_version(exponent);
  return std::forward_as_tuple(mantissa, exponent);
}
at::Tensor & from_file_out_out(c10::DispatchKeySet ks, c10::string_view filename, ::std::optional<bool> shared, ::std::optional<int64_t> size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::from_file_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, filename, shared, size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & full_like_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & fill_value, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::full_like_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, fill_value, memory_format, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & full_out_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, const at::Scalar & fill_value, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::full_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, size, fill_value, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & full_out_names_out(c10::DispatchKeySet ks, at::IntArrayRef size, const at::Scalar & fill_value, ::std::optional<at::DimnameList> names, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::full_names_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, size, fill_value, names, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & gather_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::gather_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, sparse_grad, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & gcd_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::gcd_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & gcd_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::gcd_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & ge__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::ge__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & ge__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::ge__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & ge_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::ge_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & ge_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::ge_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & gelu_(c10::DispatchKeySet ks, at::Tensor & self, c10::string_view approximate) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::gelu_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, approximate);
  }
  increment_version(self);
  return self;
}
at::Tensor & gelu_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, c10::string_view approximate, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::gelu_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, approximate, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & gelu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::string_view approximate, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::gelu_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, approximate, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & geometric_(c10::DispatchKeySet ks, at::Tensor & self, double p, ::std::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::geometric_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, p, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & geometric_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double p, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::geometric_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, p, generator, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> geqrf_out_a(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & a, at::Tensor & tau) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::geqrf_a::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, a, tau);
  }
  increment_version(a);
  increment_version(tau);
  return std::forward_as_tuple(a, tau);
}
at::Tensor & glu_backward_jvp_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_x, const at::Tensor & grad_glu, const at::Tensor & x, const at::Tensor & dgrad_glu, const at::Tensor & dx, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::glu_backward_jvp_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_x, grad_glu, x, dgrad_glu, dx, dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & glu_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, int64_t dim, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::glu_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, dim, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & glu_jvp_out_out(c10::DispatchKeySet ks, const at::Tensor & glu, const at::Tensor & x, const at::Tensor & dx, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::glu_jvp_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, glu, x, dx, dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & glu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::glu_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> grid_sampler_2d_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, ::std::array<bool,2> output_mask, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::grid_sampler_2d_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & grid_sampler_2d_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::grid_sampler_2d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, grid, interpolation_mode, padding_mode, align_corners, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> grid_sampler_3d_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, ::std::array<bool,2> output_mask, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::grid_sampler_3d_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & grid_sampler_3d_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::grid_sampler_3d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, grid, interpolation_mode, padding_mode, align_corners, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & gt__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::gt__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & gt__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::gt__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & gt_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::gt_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & gt_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::gt_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & hamming_window_out_out(c10::DispatchKeySet ks, int64_t window_length, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hamming_window_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, window_length, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & hamming_window_out_periodic_out(c10::DispatchKeySet ks, int64_t window_length, bool periodic, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hamming_window_periodic_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, window_length, periodic, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & hamming_window_out_periodic_alpha_out(c10::DispatchKeySet ks, int64_t window_length, bool periodic, double alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hamming_window_periodic_alpha_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, window_length, periodic, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & hamming_window_out_periodic_alpha_beta_out(c10::DispatchKeySet ks, int64_t window_length, bool periodic, double alpha, double beta, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hamming_window_periodic_alpha_beta_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, window_length, periodic, alpha, beta, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & hann_window_out_out(c10::DispatchKeySet ks, int64_t window_length, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hann_window_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, window_length, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & hann_window_out_periodic_out(c10::DispatchKeySet ks, int64_t window_length, bool periodic, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hann_window_periodic_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, window_length, periodic, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & hardshrink_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hardshrink_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_out, self, lambd, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & hardshrink_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hardshrink_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, lambd, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & hardsigmoid_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hardsigmoid_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & hardsigmoid_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hardsigmoid_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & hardsigmoid_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hardsigmoid_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & hardswish_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hardswish_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & hardswish_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hardswish_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & hardswish_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hardswish_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & hardtanh_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hardtanh_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, min_val, max_val);
  }
  increment_version(self);
  return self;
}
at::Tensor & hardtanh_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hardtanh_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, min_val, max_val, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & hardtanh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hardtanh_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, min_val, max_val, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & heaviside_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & values) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::heaviside_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, values);
  }
  increment_version(self);
  return self;
}
at::Tensor & heaviside_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & values, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::heaviside_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, values, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & histc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::histc_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, bins, min, max, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> histogram_out_bins_tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & bins, const ::std::optional<at::Tensor> & weight, bool density, at::Tensor & hist, at::Tensor & bin_edges) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::histogram_bins_tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, bins, weight, density, hist, bin_edges);
  }
  increment_version(hist);
  increment_version(bin_edges);
  return std::forward_as_tuple(hist, bin_edges);
}
::std::tuple<at::Tensor &,at::Tensor &> histogram_out_bin_ct_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density, at::Tensor & hist, at::Tensor & bin_edges) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::histogram_bin_ct_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, bins, range, weight, density, hist, bin_edges);
  }
  increment_version(hist);
  increment_version(bin_edges);
  return std::forward_as_tuple(hist, bin_edges);
}
at::Tensor & hspmm_out_out(c10::DispatchKeySet ks, const at::Tensor & mat1, const at::Tensor & mat2, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hspmm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, mat1, mat2, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & huber_loss_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::huber_loss_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, target, reduction, delta, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & huber_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::huber_loss_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, target, reduction, delta, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & hypot_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hypot_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & hypot_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::hypot_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & i0_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::i0_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & i0_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::i0_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & igamma_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::igamma_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & igamma_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::igamma_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & igammac_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::igammac_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & igammac_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::igammac_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & im2col_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::im2col_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, kernel_size, dilation, padding, stride, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & index_add_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::index_add_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, source, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & index_add_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::index_add_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, source, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & index_copy_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::index_copy_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, source);
  }
  increment_version(self);
  return self;
}
at::Tensor & index_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::index_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, source, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & index_fill__int_Scalar(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::index_fill__int_Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & index_fill__int_Tensor(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::index_fill__int_Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & index_fill_out_int_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::index_fill_int_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, value, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & index_fill_out_int_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::index_fill_int_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, value, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & index_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::index_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, indices, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & index_put_(c10::DispatchKeySet ks, at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::index_put_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, indices, values, accumulate);
  }
  increment_version(self);
  return self;
}
at::Tensor & index_put_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::index_put_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, indices, values, accumulate, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & index_reduce_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, c10::string_view reduce, bool include_self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::index_reduce_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, source, reduce, include_self);
  }
  increment_version(self);
  return self;
}
at::Tensor & index_reduce_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, c10::string_view reduce, bool include_self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::index_reduce_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, source, reduce, include_self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & index_select_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::index_select_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, out);
  }
  increment_version(out);
  return out;
}
at::Tensor indices(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::indices::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  auto result = as_view(self, _tmp, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false);
  return result;
}
at::Tensor & indices_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::indices_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & int_repr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::int_repr_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & isin_out_Tensor_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & elements, const at::Tensor & test_elements, bool assume_unique, bool invert, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::isin_Tensor_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, elements, test_elements, assume_unique, invert, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & isin_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & elements, const at::Scalar & test_element, bool assume_unique, bool invert, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::isin_Tensor_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, elements, test_element, assume_unique, invert, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & isin_out_Scalar_Tensor_out(c10::DispatchKeySet ks, const at::Scalar & element, const at::Tensor & test_elements, bool assume_unique, bool invert, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::isin_Scalar_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, element, test_elements, assume_unique, invert, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & isinf_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::isinf_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & isnan_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::isnan_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & isneginf_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::isneginf_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & isposinf_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::isposinf_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & kaiser_window_out_out(c10::DispatchKeySet ks, int64_t window_length, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::kaiser_window_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, window_length, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & kaiser_window_out_periodic_out(c10::DispatchKeySet ks, int64_t window_length, bool periodic, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::kaiser_window_periodic_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, window_length, periodic, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & kaiser_window_out_beta_out(c10::DispatchKeySet ks, int64_t window_length, bool periodic, double beta, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::kaiser_window_beta_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, window_length, periodic, beta, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> kthvalue_out_values(c10::DispatchKeySet ks, const at::Tensor & self, int64_t k, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::kthvalue_values::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, k, dim, keepdim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
at::Tensor & lcm_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::lcm_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & lcm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::lcm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & le__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::le__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & le__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::le__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & le_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::le_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & le_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::le_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & leaky_relu_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & negative_slope) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::leaky_relu_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, negative_slope);
  }
  increment_version(self);
  return self;
}
at::Tensor & leaky_relu_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::leaky_relu_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, negative_slope, self_is_result, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & leaky_relu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & negative_slope, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::leaky_relu_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, negative_slope, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & lerp__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::lerp__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, end, weight);
  }
  increment_version(self);
  return self;
}
at::Tensor & lerp__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & end, const at::Tensor & weight) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::lerp__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, end, weight);
  }
  increment_version(self);
  return self;
}
at::Tensor & lerp_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::lerp_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, end, weight, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & lerp_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::lerp_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, end, weight, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & lgamma_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::lgamma_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & lgamma_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::lgamma_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & lift_fresh_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::lift_fresh_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & lift_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::lift_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> linalg_cholesky_ex_out_L(c10::DispatchKeySet ks, const at::Tensor & self, bool upper, bool check_errors, at::Tensor & L, at::Tensor & info) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linalg_cholesky_ex_L::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, upper, check_errors, L, info);
  }
  increment_version(L);
  increment_version(info);
  return std::forward_as_tuple(L, info);
}
at::Tensor & linalg_cross_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linalg_cross_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, dim, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> linalg_eig_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & eigenvalues, at::Tensor & eigenvectors) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linalg_eig_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, eigenvalues, eigenvectors);
  }
  increment_version(eigenvalues);
  increment_version(eigenvectors);
  return std::forward_as_tuple(eigenvalues, eigenvectors);
}
at::Tensor & linalg_eigvals_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linalg_eigvals_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & linalg_householder_product_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & tau, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linalg_householder_product_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, tau, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> linalg_inv_ex_out_inverse(c10::DispatchKeySet ks, const at::Tensor & A, bool check_errors, at::Tensor & inverse, at::Tensor & info) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linalg_inv_ex_inverse::redispatch(ks & c10::after_ADInplaceOrView_keyset, A, check_errors, inverse, info);
  }
  increment_version(inverse);
  increment_version(info);
  return std::forward_as_tuple(inverse, info);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> linalg_ldl_factor_ex_out_out(c10::DispatchKeySet ks, const at::Tensor & self, bool hermitian, bool check_errors, at::Tensor & LD, at::Tensor & pivots, at::Tensor & info) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linalg_ldl_factor_ex_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, hermitian, check_errors, LD, pivots, info);
  }
  increment_version(LD);
  increment_version(pivots);
  increment_version(info);
  return std::forward_as_tuple(LD, pivots, info);
}
at::Tensor & linalg_ldl_solve_out_out(c10::DispatchKeySet ks, const at::Tensor & LD, const at::Tensor & pivots, const at::Tensor & B, bool hermitian, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linalg_ldl_solve_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, LD, pivots, B, hermitian, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> linalg_lstsq_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & b, ::std::optional<double> rcond, ::std::optional<c10::string_view> driver, at::Tensor & solution, at::Tensor & residuals, at::Tensor & rank, at::Tensor & singular_values) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linalg_lstsq_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, b, rcond, driver, solution, residuals, rank, singular_values);
  }
  increment_version(solution);
  increment_version(residuals);
  increment_version(rank);
  increment_version(singular_values);
  return std::forward_as_tuple(solution, residuals, rank, singular_values);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> linalg_lu_factor_ex_out_out(c10::DispatchKeySet ks, const at::Tensor & A, bool pivot, bool check_errors, at::Tensor & LU, at::Tensor & pivots, at::Tensor & info) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linalg_lu_factor_ex_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, A, pivot, check_errors, LU, pivots, info);
  }
  increment_version(LU);
  increment_version(pivots);
  increment_version(info);
  return std::forward_as_tuple(LU, pivots, info);
}
::std::tuple<at::Tensor &,at::Tensor &> linalg_lu_factor_out_out(c10::DispatchKeySet ks, const at::Tensor & A, bool pivot, at::Tensor & LU, at::Tensor & pivots) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linalg_lu_factor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, A, pivot, LU, pivots);
  }
  increment_version(LU);
  increment_version(pivots);
  return std::forward_as_tuple(LU, pivots);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> linalg_lu_out_out(c10::DispatchKeySet ks, const at::Tensor & A, bool pivot, at::Tensor & P, at::Tensor & L, at::Tensor & U) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linalg_lu_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, A, pivot, P, L, U);
  }
  increment_version(P);
  increment_version(L);
  increment_version(U);
  return std::forward_as_tuple(P, L, U);
}
at::Tensor & linalg_lu_solve_out_out(c10::DispatchKeySet ks, const at::Tensor & LU, const at::Tensor & pivots, const at::Tensor & B, bool left, bool adjoint, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linalg_lu_solve_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, LU, pivots, B, left, adjoint, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & linalg_matrix_exp_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linalg_matrix_exp_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & linalg_pinv_out_atol_rtol_tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Tensor> & atol, const ::std::optional<at::Tensor> & rtol, bool hermitian, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linalg_pinv_atol_rtol_tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, atol, rtol, hermitian, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> linalg_qr_out_out(c10::DispatchKeySet ks, const at::Tensor & A, c10::string_view mode, at::Tensor & Q, at::Tensor & R) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linalg_qr_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, A, mode, Q, R);
  }
  increment_version(Q);
  increment_version(R);
  return std::forward_as_tuple(Q, R);
}
at::Tensor & linalg_solve_triangular_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & B, bool upper, bool left, bool unitriangular, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linalg_solve_triangular_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, B, upper, left, unitriangular, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & linalg_vector_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & ord, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linalg_vector_norm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, ord, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> linear_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linear_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, grad_output, weight, output_mask, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
at::Tensor & linear_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linear_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight, bias, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & linspace_out_out(c10::DispatchKeySet ks, const at::Scalar & start, const at::Scalar & end, int64_t steps, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linspace_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, start, end, steps, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & linspace_out_Tensor_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & start, const at::Tensor & end, int64_t steps, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linspace_Tensor_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, start, end, steps, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & linspace_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & start, const at::Scalar & end, int64_t steps, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linspace_Tensor_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, start, end, steps, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & linspace_out_Scalar_Tensor_out(c10::DispatchKeySet ks, const at::Scalar & start, const at::Tensor & end, int64_t steps, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::linspace_Scalar_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, start, end, steps, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & log10_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::log10_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & log10_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::log10_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & log1p_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::log1p_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & log1p_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::log1p_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & log2_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::log2_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & log2_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::log2_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & log_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::log_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & log_normal_(c10::DispatchKeySet ks, at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::log_normal_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mean, std, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & log_normal_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::log_normal_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mean, std, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & log_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::log_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & log_sigmoid_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::log_sigmoid_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, buffer, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> log_sigmoid_forward_out_output(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & output, at::Tensor & buffer) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::log_sigmoid_forward_output::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output, buffer);
  }
  increment_version(output);
  increment_version(buffer);
  return std::forward_as_tuple(output, buffer);
}
at::Tensor & log_softmax_out_int_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::log_softmax_int_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logaddexp2_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::logaddexp2_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logaddexp_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::logaddexp_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logcumsumexp_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::logcumsumexp_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logical_and_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::logical_and_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & logical_and_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::logical_and_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logical_not_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::logical_not_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & logical_not_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::logical_not_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logical_or_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::logical_or_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & logical_or_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::logical_or_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logical_xor_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::logical_xor_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & logical_xor_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::logical_xor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logit_(c10::DispatchKeySet ks, at::Tensor & self, ::std::optional<double> eps) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::logit_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, eps);
  }
  increment_version(self);
  return self;
}
at::Tensor & logit_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, ::std::optional<double> eps, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::logit_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, eps, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & logit_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<double> eps, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::logit_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, eps, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logspace_out_out(c10::DispatchKeySet ks, const at::Scalar & start, const at::Scalar & end, int64_t steps, double base, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::logspace_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, start, end, steps, base, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logspace_out_Tensor_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & start, const at::Tensor & end, int64_t steps, double base, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::logspace_Tensor_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, start, end, steps, base, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logspace_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & start, const at::Scalar & end, int64_t steps, double base, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::logspace_Tensor_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, start, end, steps, base, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logspace_out_Scalar_Tensor_out(c10::DispatchKeySet ks, const at::Scalar & start, const at::Tensor & end, int64_t steps, double base, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::logspace_Scalar_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, start, end, steps, base, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logsumexp_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::logsumexp_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & lt__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::lt__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & lt__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::lt__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & lt_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::lt_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & lt_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::lt_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> lu_unpack_out_out(c10::DispatchKeySet ks, const at::Tensor & LU_data, const at::Tensor & LU_pivots, bool unpack_data, bool unpack_pivots, at::Tensor & P, at::Tensor & L, at::Tensor & U) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::lu_unpack_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, LU_data, LU_pivots, unpack_data, unpack_pivots, P, L, U);
  }
  increment_version(P);
  increment_version(L);
  increment_version(U);
  return std::forward_as_tuple(P, L, U);
}
at::Tensor & masked_fill__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::masked_fill__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mask, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & masked_fill__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::masked_fill__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mask, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & masked_fill_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::masked_fill_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mask, value, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & masked_fill_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::masked_fill_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mask, value, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & masked_scatter_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & mask, const at::Tensor & source) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::masked_scatter_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mask, source);
  }
  increment_version(self);
  return self;
}
at::Tensor & masked_scatter_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, const at::Tensor & source, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::masked_scatter_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mask, source, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & masked_select_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::masked_select_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mask, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> matmul_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & self, const at::Tensor & other, ::std::array<bool,2> mask, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::matmul_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad, self, other, mask, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & matmul_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::matmul_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> max_out_dim_max(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & max, at::Tensor & max_values) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::max_dim_max::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, max, max_values);
  }
  increment_version(max);
  increment_version(max_values);
  return std::forward_as_tuple(max, max_values);
}
at::Tensor & max_out_unary_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::max_unary_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & max_pool2d_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::max_pool2d_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & max_pool2d_with_indices_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::max_pool2d_with_indices_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> max_pool2d_with_indices_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::max_pool2d_with_indices_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
  }
  increment_version(out);
  increment_version(indices);
  return std::forward_as_tuple(out, indices);
}
at::Tensor & max_pool3d_with_indices_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::max_pool3d_with_indices_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> max_pool3d_with_indices_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::max_pool3d_with_indices_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
  }
  increment_version(out);
  increment_version(indices);
  return std::forward_as_tuple(out, indices);
}
at::Tensor & max_unpool2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & indices, c10::SymIntArrayRef output_size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::max_unpool2d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, indices, output_size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & max_unpool3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & indices, c10::SymIntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::max_unpool3d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, indices, output_size, stride, padding, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & maximum_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::maximum_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & mean_out_dtype_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mean_dtype_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & mean_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mean_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> median_out_dim_values(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::median_dim_values::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
at::Tensor & median_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::median_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> min_out_dim_min(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & min, at::Tensor & min_indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::min_dim_min::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, min, min_indices);
  }
  increment_version(min);
  increment_version(min_indices);
  return std::forward_as_tuple(min, min_indices);
}
at::Tensor & min_out_unary_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::min_unary_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & minimum_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::minimum_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> miopen_batch_norm_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, const ::std::optional<at::Tensor> & save_mean, const ::std::optional<at::Tensor> & save_var, double epsilon, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::miopen_batch_norm_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> miopen_batch_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double exponential_average_factor, double epsilon, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::miopen_batch_norm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
at::Tensor & miopen_convolution_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, bool benchmark, bool deterministic, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::miopen_convolution_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & miopen_convolution_transpose_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, bool benchmark, bool deterministic, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::miopen_convolution_transpose_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & miopen_depthwise_convolution_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, bool benchmark, bool deterministic, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::miopen_depthwise_convolution_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> miopen_rnn_out_out(c10::DispatchKeySet ks, const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & hx, const ::std::optional<at::Tensor> & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const ::std::optional<at::Tensor> & dropout_state, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3, at::Tensor & out4) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::miopen_rnn_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, out0, out1, out2, out3, out4);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  increment_version(out3);
  increment_version(out4);
  return std::forward_as_tuple(out0, out1, out2, out3, out4);
}
at::Tensor & mish_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mish_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & mish_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mish_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & mkldnn_adaptive_avg_pool2d_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mkldnn_adaptive_avg_pool2d_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & mkldnn_adaptive_avg_pool2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mkldnn_adaptive_avg_pool2d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & mkldnn_convolution_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mkldnn_convolution_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, bias, padding, stride, dilation, groups, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & mkldnn_linear_backward_input_out_out(c10::DispatchKeySet ks, at::IntArrayRef input_size, const at::Tensor & grad_output, const at::Tensor & weight, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mkldnn_linear_backward_input_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input_size, grad_output, weight, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> mkldnn_linear_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mkldnn_linear_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, grad_output, weight, output_mask, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
::std::tuple<at::Tensor &,at::Tensor &> mkldnn_linear_backward_weights_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, bool bias_defined, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mkldnn_linear_backward_weights_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, input, weight, bias_defined, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & mkldnn_linear_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mkldnn_linear_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, bias, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & mkldnn_max_pool2d_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, const at::Tensor & input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mkldnn_max_pool2d_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & mkldnn_max_pool2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mkldnn_max_pool2d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, kernel_size, stride, padding, dilation, ceil_mode, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & mkldnn_max_pool3d_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, const at::Tensor & input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mkldnn_max_pool3d_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & mkldnn_max_pool3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mkldnn_max_pool3d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, kernel_size, stride, padding, dilation, ceil_mode, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & mkldnn_reorder_conv2d_weight_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, at::OptionalSymIntArrayRef input_size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mkldnn_reorder_conv2d_weight_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, padding, stride, dilation, groups, input_size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & mkldnn_reorder_conv3d_weight_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, at::OptionalSymIntArrayRef input_size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mkldnn_reorder_conv3d_weight_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, padding, stride, dilation, groups, input_size, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> mkldnn_rnn_layer_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight1, const at::Tensor & weight2, const at::Tensor & weight3, const at::Tensor & weight4, const at::Tensor & hx_, const at::Tensor & cx_tmp, const at::Tensor & output, const at::Tensor & hy_, const at::Tensor & cy_, const ::std::optional<at::Tensor> & grad_output, const ::std::optional<at::Tensor> & grad_hy, const ::std::optional<at::Tensor> & grad_cy, bool reverse, int64_t mode, int64_t hidden_size, int64_t num_layers, bool has_biases, bool train, bool bidirectional, at::IntArrayRef batch_sizes, bool batch_first, const at::Tensor & workspace, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3, at::Tensor & out4, at::Tensor & out5, at::Tensor & out6) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mkldnn_rnn_layer_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight1, weight2, weight3, weight4, hx_, cx_tmp, output, hy_, cy_, grad_output, grad_hy, grad_cy, reverse, mode, hidden_size, num_layers, has_biases, train, bidirectional, batch_sizes, batch_first, workspace, out0, out1, out2, out3, out4, out5, out6);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  increment_version(out3);
  increment_version(out4);
  increment_version(out5);
  increment_version(out6);
  return std::forward_as_tuple(out0, out1, out2, out3, out4, out5, out6);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> mkldnn_rnn_layer_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight0, const at::Tensor & weight1, const at::Tensor & weight2, const at::Tensor & weight3, const at::Tensor & hx_, const at::Tensor & cx_, bool reverse, at::IntArrayRef batch_sizes, int64_t mode, int64_t hidden_size, int64_t num_layers, bool has_biases, bool bidirectional, bool batch_first, bool train, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mkldnn_rnn_layer_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight0, weight1, weight2, weight3, hx_, cx_, reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train, out0, out1, out2, out3);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  increment_version(out3);
  return std::forward_as_tuple(out0, out1, out2, out3);
}
at::Tensor & mm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mat2, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> mode_out_values(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mode_values::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> mps_convolution_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mps_convolution_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, grad_output, weight, padding, stride, dilation, groups, output_mask, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
::std::tuple<at::Tensor &,at::Tensor &> mps_convolution_transpose_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, ::std::array<bool,2> output_mask, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mps_convolution_transpose_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, grad_output, weight, padding, output_padding, stride, dilation, groups, output_mask, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & mse_loss_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mse_loss_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, target, reduction, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & mse_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mse_loss_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, target, reduction, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & mul__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mul__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & mul__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mul__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & mul_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mul_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & mul_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mul_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & multi_margin_loss_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const ::std::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::multi_margin_loss_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, target, p, margin, weight, reduction, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & multi_margin_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const ::std::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::multi_margin_loss_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, target, p, margin, weight, reduction, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & multilabel_margin_loss_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, const at::Tensor & is_target, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::multilabel_margin_loss_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, target, reduction, is_target, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> multilabel_margin_loss_forward_out_output(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & output, at::Tensor & is_target) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::multilabel_margin_loss_forward_output::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, target, reduction, output, is_target);
  }
  increment_version(output);
  increment_version(is_target);
  return std::forward_as_tuple(output, is_target);
}
at::Tensor & multinomial_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t num_samples, bool replacement, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::multinomial_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, num_samples, replacement, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & mv_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & vec, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mv_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, vec, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & mvlgamma_(c10::DispatchKeySet ks, at::Tensor & self, int64_t p) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mvlgamma_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, p);
  }
  increment_version(self);
  return self;
}
at::Tensor & mvlgamma_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t p, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::mvlgamma_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, p, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & nan_to_num_(c10::DispatchKeySet ks, at::Tensor & self, ::std::optional<double> nan, ::std::optional<double> posinf, ::std::optional<double> neginf) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::nan_to_num_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, nan, posinf, neginf);
  }
  increment_version(self);
  return self;
}
at::Tensor & nan_to_num_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<double> nan, ::std::optional<double> posinf, ::std::optional<double> neginf, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::nan_to_num_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, nan, posinf, neginf, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> nanmedian_out_dim_values(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::nanmedian_dim_values::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
at::Tensor & nanmedian_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::nanmedian_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & nansum_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::nansum_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor narrow(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, c10::SymInt start, c10::SymInt length) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::narrow::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, start, length);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::NarrowViewFunc>(dim, start, length);
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::narrow_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, dim, start, length);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & narrow_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, c10::SymInt start, c10::SymInt length, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::narrow_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, start, length, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_batch_norm_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, const ::std::optional<at::Tensor> & save_mean, const ::std::optional<at::Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::native_batch_norm_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_batch_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double momentum, double eps, at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::native_batch_norm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight, bias, running_mean, running_var, training, momentum, eps, out, save_mean, save_invstd);
  }
  increment_version(out);
  increment_version(save_mean);
  increment_version(save_invstd);
  return std::forward_as_tuple(out, save_mean, save_invstd);
}
at::Tensor & native_dropout_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & mask, double scale, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::native_dropout_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, mask, scale, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> native_dropout_out_out(c10::DispatchKeySet ks, const at::Tensor & input, double p, ::std::optional<bool> train, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::native_dropout_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, p, train, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_group_norm_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & rstd, const ::std::optional<at::Tensor> & weight, c10::SymInt N, c10::SymInt C, c10::SymInt HxW, int64_t group, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::native_group_norm_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_group_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, c10::SymInt N, c10::SymInt C, c10::SymInt HxW, int64_t group, double eps, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::native_group_norm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight, bias, N, C, HxW, group, eps, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_layer_norm_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & input, c10::SymIntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::native_layer_norm_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_layer_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & input, c10::SymIntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::native_layer_norm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, normalized_shape, weight, bias, eps, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
at::Tensor & native_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & p, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::native_norm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, p, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & native_norm_out_ScalarOpt_dim_dtype_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::native_norm_ScalarOpt_dim_dtype_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, p, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & ne__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::ne__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & ne__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::ne__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & ne_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::ne_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & ne_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::ne_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & neg_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::neg_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & neg_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::neg_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & new_empty_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::new_empty_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & new_empty_strided_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::new_empty_strided_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, stride, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & new_full_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, const at::Scalar & fill_value, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::new_full_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, fill_value, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & new_ones_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::new_ones_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & new_zeros_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::new_zeros_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & nextafter_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::nextafter_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & nextafter_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::nextafter_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & nll_loss2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::nll_loss2d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> nll_loss2d_forward_out_output(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, at::Tensor & output, at::Tensor & total_weight) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::nll_loss2d_forward_output::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, target, weight, reduction, ignore_index, output, total_weight);
  }
  increment_version(output);
  increment_version(total_weight);
  return std::forward_as_tuple(output, total_weight);
}
at::Tensor & nll_loss_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::nll_loss_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> nll_loss_forward_out_output(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, at::Tensor & output, at::Tensor & total_weight) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::nll_loss_forward_output::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, target, weight, reduction, ignore_index, output, total_weight);
  }
  increment_version(output);
  increment_version(total_weight);
  return std::forward_as_tuple(output, total_weight);
}
at::Tensor & nonzero_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::nonzero_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & nonzero_static_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt size, int64_t fill_value, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::nonzero_static_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, fill_value, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & norm_out_dtype_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::norm_dtype_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, p, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & norm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::norm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, p, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & norm_out_ScalarOpt_dtype_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::ScalarType dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::norm_ScalarOpt_dtype_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, p, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & norm_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & p, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::norm_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, p, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & normal_(c10::DispatchKeySet ks, at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::normal_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mean, std, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & normal_out_Tensor_float_out(c10::DispatchKeySet ks, const at::Tensor & mean, double std, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::normal_Tensor_float_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, mean, std, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & normal_out_float_Tensor_out(c10::DispatchKeySet ks, double mean, const at::Tensor & std, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::normal_float_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, mean, std, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & normal_out_Tensor_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & mean, const at::Tensor & std, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::normal_Tensor_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, mean, std, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & normal_out_float_float_out(c10::DispatchKeySet ks, double mean, double std, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::normal_float_float_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, mean, std, size, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & normal_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::normal_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mean, std, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & ones_like_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::ones_like_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, memory_format, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & ones_out_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::ones_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & ones_out_names_out(c10::DispatchKeySet ks, at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::ones_names_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, size, names, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & ormqr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & input2, const at::Tensor & input3, bool left, bool transpose, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::ormqr_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, input2, input3, left, transpose, out);
  }
  increment_version(out);
  return out;
}
at::Tensor permute(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dims) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::permute::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dims);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    auto dims_vec = dims.vec();
    func = std::make_unique<torch::autograd::generated::PermuteViewFunc>(dims);
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::permute_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, dims_vec);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & permute_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dims, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::permute_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dims, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & pixel_shuffle_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t upscale_factor, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::pixel_shuffle_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, upscale_factor, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & pixel_unshuffle_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t downscale_factor, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::pixel_unshuffle_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, downscale_factor, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & poisson_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::poisson_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & polar_out_out(c10::DispatchKeySet ks, const at::Tensor & abs, const at::Tensor & angle, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::polar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, abs, angle, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & polygamma_(c10::DispatchKeySet ks, at::Tensor & self, int64_t n) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::polygamma_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, n);
  }
  increment_version(self);
  return self;
}
at::Tensor & polygamma_out_out(c10::DispatchKeySet ks, int64_t n, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::polygamma_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, n, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & pow__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & exponent) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::pow__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, exponent);
  }
  increment_version(self);
  return self;
}
at::Tensor & pow__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & exponent) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::pow__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, exponent);
  }
  increment_version(self);
  return self;
}
at::Tensor & pow_out_Tensor_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & exponent, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::pow_Tensor_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, exponent, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & pow_out_Scalar_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & exponent, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::pow_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, exponent, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & pow_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::pow_Tensor_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, exponent, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & prod_out_int_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::prod_int_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & prod_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::prod_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & put_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::put_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, index, source, accumulate);
  }
  increment_version(self);
  return self;
}
at::Tensor & put_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::put_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, index, source, accumulate, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & q_per_channel_scales_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::q_per_channel_scales_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & q_per_channel_zero_points_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::q_per_channel_zero_points_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & quantize_per_channel_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, at::ScalarType dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::quantize_per_channel_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, scales, zero_points, axis, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & quantize_per_tensor_dynamic_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::ScalarType dtype, bool reduce_range, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::quantize_per_tensor_dynamic_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dtype, reduce_range, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & quantize_per_tensor_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double scale, int64_t zero_point, at::ScalarType dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::quantize_per_tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, scale, zero_point, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & quantize_per_tensor_out_tensor_qparams_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, at::ScalarType dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::quantize_per_tensor_tensor_qparams_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, scale, zero_point, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & quantized_batch_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & var, double eps, double output_scale, int64_t output_zero_point, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::quantized_batch_norm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, input, weight, bias, mean, var, eps, output_scale, output_zero_point, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & quantized_max_pool1d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::quantized_max_pool1d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, kernel_size, stride, padding, dilation, ceil_mode, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & quantized_max_pool2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::quantized_max_pool2d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, kernel_size, stride, padding, dilation, ceil_mode, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & quantized_max_pool3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::quantized_max_pool3d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, kernel_size, stride, padding, dilation, ceil_mode, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & rad2deg_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::rad2deg_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & rad2deg_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::rad2deg_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & rand_like_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::rand_like_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, memory_format, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & rand_out_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::rand_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & rand_out_names_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::DimnameList> names, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::rand_names_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, size, names, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & rand_out_generator_with_names_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::rand_generator_with_names_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, size, generator, names, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & randint_like_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt high, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::randint_like_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, high, memory_format, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & randint_like_out_low_dtype_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt low, c10::SymInt high, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::randint_like_low_dtype_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, low, high, memory_format, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & randint_out_out(c10::DispatchKeySet ks, c10::SymInt high, c10::SymIntArrayRef size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::randint_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, high, size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & randint_out_generator_out(c10::DispatchKeySet ks, c10::SymInt high, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::randint_generator_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, high, size, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & randint_out_low_out(c10::DispatchKeySet ks, c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::randint_low_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, low, high, size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & randint_out_low_generator_out(c10::DispatchKeySet ks, c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::randint_low_generator_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, low, high, size, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & randn_like_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::randn_like_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, memory_format, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & randn_out_names_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::DimnameList> names, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::randn_names_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, size, names, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & randn_out_generator_with_names_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::randn_generator_with_names_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, size, generator, names, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & random__from(c10::DispatchKeySet ks, at::Tensor & self, int64_t from, ::std::optional<int64_t> to, ::std::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::random__from::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, from, to, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & random__to(c10::DispatchKeySet ks, at::Tensor & self, int64_t to, ::std::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::random__to::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, to, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & random_(c10::DispatchKeySet ks, at::Tensor & self, ::std::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::random_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & random_out_from_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t from, ::std::optional<int64_t> to, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::random_from_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, from, to, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & random_out_to_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t to, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::random_to_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, to, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & random_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::random_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & randperm_out_out(c10::DispatchKeySet ks, c10::SymInt n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::randperm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & randperm_out_generator_out(c10::DispatchKeySet ks, c10::SymInt n, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::randperm_generator_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, n, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & range_out_out_(c10::DispatchKeySet ks, const at::Scalar & start, const at::Scalar & end, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::range_out_::redispatch(ks & c10::after_ADInplaceOrView_keyset, start, end, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & range_out_out(c10::DispatchKeySet ks, const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::range_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, start, end, step, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & reciprocal_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::reciprocal_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & reciprocal_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::reciprocal_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & reflection_pad1d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::reflection_pad1d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, padding, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & reflection_pad1d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::reflection_pad1d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, padding, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & reflection_pad2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::reflection_pad2d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, padding, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & reflection_pad2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::reflection_pad2d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, padding, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & reflection_pad3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::reflection_pad3d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, padding, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & reflection_pad3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::reflection_pad3d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, padding, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & relu_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::relu_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & relu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::relu_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & remainder__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::remainder__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & remainder__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::remainder__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & remainder_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::remainder_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & remainder_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::remainder_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & remainder_out_Scalar_Tensor_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::remainder_Scalar_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & renorm_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::renorm_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, p, dim, maxnorm);
  }
  increment_version(self);
  return self;
}
at::Tensor & renorm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::renorm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, p, dim, maxnorm, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & repeat_interleave_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & repeats, ::std::optional<c10::SymInt> output_size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::repeat_interleave_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, repeats, output_size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & repeat_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef repeats, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::repeat_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, repeats, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & replication_pad1d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::replication_pad1d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, padding, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & replication_pad1d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::replication_pad1d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, padding, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & replication_pad2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::replication_pad2d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, padding, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & replication_pad2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::replication_pad2d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, padding, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & replication_pad3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::replication_pad3d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, padding, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & replication_pad3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::replication_pad3d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, padding, out);
  }
  increment_version(out);
  return out;
}
const at::Tensor & resize_as_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & the_template, ::std::optional<at::MemoryFormat> memory_format, const at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::resize_as_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, the_template, memory_format, out);
  }
  increment_version(out);
  return out;
}
const at::Tensor & resize_as_sparse_(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & the_template) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::resize_as_sparse_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, the_template);
  }
  increment_version(self);
  return self;
}
const at::Tensor & resize_as_sparse_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & the_template, const at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::resize_as_sparse_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, the_template, out);
  }
  increment_version(out);
  return out;
}
const at::Tensor & resize_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, ::std::optional<at::MemoryFormat> memory_format, const at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::resize_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, memory_format, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & roll_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef shifts, at::IntArrayRef dims, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::roll_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, shifts, dims, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & rot90_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t k, at::IntArrayRef dims, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::rot90_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, k, dims, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & round_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::round_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & round__decimals(c10::DispatchKeySet ks, at::Tensor & self, int64_t decimals) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::round__decimals::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, decimals);
  }
  increment_version(self);
  return self;
}
at::Tensor & round_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::round_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & round_out_decimals_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t decimals, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::round_decimals_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, decimals, out);
  }
  increment_version(out);
  return out;
}
at::Tensor row_indices(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::row_indices::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  auto result = as_view(self, _tmp, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false);
  return result;
}
at::Tensor & row_indices_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::row_indices_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & rrelu_with_noise_(c10::DispatchKeySet ks, at::Tensor & self, at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, ::std::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::rrelu_with_noise_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, noise, lower, upper, training, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & rrelu_with_noise_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, bool self_is_result, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::rrelu_with_noise_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, noise, lower, upper, training, self_is_result, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & rrelu_with_noise_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::rrelu_with_noise_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, noise, lower, upper, training, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & rsqrt_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::rsqrt_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & rsqrt_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::rsqrt_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & rsub_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::rsub_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & rsub_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::rsub_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & scalar_tensor_out_out(c10::DispatchKeySet ks, const at::Scalar & s, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::scalar_tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, s, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & scatter__src(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::scatter__src::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, src);
  }
  increment_version(self);
  return self;
}
at::Tensor & scatter__value(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::scatter__value::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & scatter__reduce(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::scatter__reduce::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, src, reduce);
  }
  increment_version(self);
  return self;
}
at::Tensor & scatter__value_reduce(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::scatter__value_reduce::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, value, reduce);
  }
  increment_version(self);
  return self;
}
at::Tensor & scatter_add_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::scatter_add_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, src);
  }
  increment_version(self);
  return self;
}
at::Tensor & scatter_add_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::scatter_add_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, src, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & scatter_out_src_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::scatter_src_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, src, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & scatter_out_value_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::scatter_value_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, value, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & scatter_out_reduce_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::scatter_reduce_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, src, reduce, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & scatter_out_value_reduce_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::scatter_value_reduce_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, value, reduce, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & scatter_reduce__two(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, bool include_self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::scatter_reduce__two::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, src, reduce, include_self);
  }
  increment_version(self);
  return self;
}
at::Tensor & scatter_reduce_out_two_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, bool include_self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::scatter_reduce_two_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, src, reduce, include_self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & searchsorted_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right, ::std::optional<c10::string_view> side, const ::std::optional<at::Tensor> & sorter, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::searchsorted_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, sorted_sequence, self, out_int32, right, side, sorter, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & searchsorted_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & sorted_sequence, const at::Scalar & self, bool out_int32, bool right, ::std::optional<c10::string_view> side, const ::std::optional<at::Tensor> & sorter, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::searchsorted_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, sorted_sequence, self, out_int32, right, side, sorter, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & segment_reduce_out_out(c10::DispatchKeySet ks, const at::Tensor & data, c10::string_view reduce, const ::std::optional<at::Tensor> & lengths, const ::std::optional<at::Tensor> & indices, const ::std::optional<at::Tensor> & offsets, int64_t axis, bool unsafe, const ::std::optional<at::Scalar> & initial, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::segment_reduce_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, data, reduce, lengths, indices, offsets, axis, unsafe, initial, out);
  }
  increment_version(out);
  return out;
}
at::Tensor select_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, c10::SymInt index) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::select_int::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::SelectIntViewFunc>(dim, index);
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::select_int_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, dim, index);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & select_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef input_sizes, int64_t dim, c10::SymInt index, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::select_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, input_sizes, dim, index, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & select_copy_out_int_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, c10::SymInt index, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::select_copy_int_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & select_scatter_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & src, int64_t dim, c10::SymInt index, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::select_scatter_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, src, dim, index, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & set__source_Storage(c10::DispatchKeySet ks, at::Tensor & self, at::Storage source) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::set__source_Storage::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, source);
  }
  increment_version(self);
  return self;
}
at::Tensor & set__source_Storage_storage_offset(c10::DispatchKeySet ks, at::Tensor & self, at::Storage source, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::set__source_Storage_storage_offset::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, source, storage_offset, size, stride);
  }
  increment_version(self);
  return self;
}
at::Tensor & set__source_Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & source) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::set__source_Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, source);
  }
  increment_version(self);
  return self;
}
at::Tensor & set_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::set_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & set_out_source_Storage_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Storage source, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::set_source_Storage_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, source, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & set_out_source_Storage_storage_offset_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Storage source, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::set_source_Storage_storage_offset_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, source, storage_offset, size, stride, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & set_out_source_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & source, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::set_source_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, source, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & set_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::set_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & sgn_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sgn_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & sgn_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sgn_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & sigmoid_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sigmoid_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & sigmoid_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sigmoid_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & sigmoid_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sigmoid_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & sign_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sign_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & sign_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sign_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & signbit_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::signbit_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & silu_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::silu_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & silu_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::silu_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & silu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::silu_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & sin_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sin_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & sin_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sin_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & sinc_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sinc_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & sinc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sinc_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & sinh_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sinh_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & sinh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sinh_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor slice_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::slice_Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, start, end, step);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::SliceTensorViewFunc>(dim, start, end, step);
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::slice_Tensor_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, dim, start, end, step);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & slice_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef input_sizes, int64_t dim, c10::SymInt start, c10::SymInt end, c10::SymInt step, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::slice_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, input_sizes, dim, start, end, step, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & slice_copy_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::slice_copy_Tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, start, end, step, out);
  }
  increment_version(out);
  return out;
}
at::Tensor slice_inverse(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & src, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::slice_inverse::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, src, dim, start, end, step);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::SliceInverseViewFunc>(src, dim, start, end, step);
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::slice_inverse_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, src, dim, start, end, step);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & slice_scatter_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & src, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::slice_scatter_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, src, dim, start, end, step, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & slow_conv3d_forward_out_output(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, at::Tensor & output) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::slow_conv3d_forward_output::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, kernel_size, bias, stride, padding, output);
  }
  increment_version(output);
  return output;
}
at::Tensor & slow_conv_dilated2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::slow_conv_dilated2d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, kernel_size, bias, stride, padding, dilation, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & slow_conv_dilated3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::slow_conv_dilated3d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, kernel_size, bias, stride, padding, dilation, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & slow_conv_transpose2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef dilation, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::slow_conv_transpose2d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, kernel_size, bias, stride, padding, output_padding, dilation, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & slow_conv_transpose3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef dilation, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::slow_conv_transpose3d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, weight, kernel_size, bias, stride, padding, output_padding, dilation, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & smooth_l1_loss_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::smooth_l1_loss_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, target, reduction, beta, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & smooth_l1_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::smooth_l1_loss_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, target, reduction, beta, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & soft_margin_loss_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::soft_margin_loss_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, target, reduction, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & soft_margin_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::soft_margin_loss_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, target, reduction, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & softmax_out_int_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::softmax_int_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & softplus_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::softplus_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, beta, threshold, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & softplus_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::softplus_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, beta, threshold, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & softshrink_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::softshrink_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, lambd, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & softshrink_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::softshrink_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, lambd, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> sort_out_values(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sort_values::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, descending, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor &,at::Tensor &> sort_out_values_stable(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<bool> stable, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sort_values_stable::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, stable, dim, descending, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
at::Tensor & sparse_coo_tensor_out_size_out(c10::DispatchKeySet ks, at::IntArrayRef size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sparse_coo_tensor_size_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & sparse_mask_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sparse_mask_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mask, out);
  }
  increment_version(out);
  return out;
}
const at::Tensor & sparse_resize_(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sparse_resize_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, sparse_dim, dense_dim);
  }
  increment_version(self);
  return self;
}
const at::Tensor & sparse_resize_and_clear_(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sparse_resize_and_clear_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, sparse_dim, dense_dim);
  }
  increment_version(self);
  return self;
}
const at::Tensor & sparse_resize_and_clear_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim, const at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sparse_resize_and_clear_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, sparse_dim, dense_dim, out);
  }
  increment_version(out);
  return out;
}
const at::Tensor & sparse_resize_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim, const at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sparse_resize_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, sparse_dim, dense_dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & sparse_sampled_addmm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sparse_sampled_addmm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mat1, mat2, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_airy_ai_out_out(c10::DispatchKeySet ks, const at::Tensor & x, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_airy_ai_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_bessel_j0_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_bessel_j0_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_bessel_j1_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_bessel_j1_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_bessel_y0_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_bessel_y0_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_bessel_y1_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_bessel_y1_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_chebyshev_polynomial_t_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_chebyshev_polynomial_t_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_chebyshev_polynomial_t_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_chebyshev_polynomial_t_x_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_chebyshev_polynomial_t_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_chebyshev_polynomial_t_n_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_chebyshev_polynomial_u_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_chebyshev_polynomial_u_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_chebyshev_polynomial_u_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_chebyshev_polynomial_u_x_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_chebyshev_polynomial_u_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_chebyshev_polynomial_u_n_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_chebyshev_polynomial_v_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_chebyshev_polynomial_v_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_chebyshev_polynomial_v_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_chebyshev_polynomial_v_x_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_chebyshev_polynomial_v_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_chebyshev_polynomial_v_n_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_chebyshev_polynomial_w_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_chebyshev_polynomial_w_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_chebyshev_polynomial_w_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_chebyshev_polynomial_w_x_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_chebyshev_polynomial_w_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_chebyshev_polynomial_w_n_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_entr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_entr_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_erfcx_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_erfcx_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_hermite_polynomial_h_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_hermite_polynomial_h_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_hermite_polynomial_h_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_hermite_polynomial_h_x_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_hermite_polynomial_h_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_hermite_polynomial_h_n_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_hermite_polynomial_he_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_hermite_polynomial_he_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_hermite_polynomial_he_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_hermite_polynomial_he_x_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_hermite_polynomial_he_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_hermite_polynomial_he_n_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_i0e_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_i0e_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_i1_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_i1_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_i1e_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_i1e_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_laguerre_polynomial_l_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_laguerre_polynomial_l_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_laguerre_polynomial_l_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_laguerre_polynomial_l_x_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_laguerre_polynomial_l_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_laguerre_polynomial_l_n_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_legendre_polynomial_p_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_legendre_polynomial_p_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_legendre_polynomial_p_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_legendre_polynomial_p_x_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_legendre_polynomial_p_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_legendre_polynomial_p_n_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_log_ndtr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_log_ndtr_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_modified_bessel_i0_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_modified_bessel_i0_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_modified_bessel_i1_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_modified_bessel_i1_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_modified_bessel_k0_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_modified_bessel_k0_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_modified_bessel_k1_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_modified_bessel_k1_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_ndtri_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_ndtri_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_scaled_modified_bessel_k0_out_out(c10::DispatchKeySet ks, const at::Tensor & x, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_scaled_modified_bessel_k0_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_scaled_modified_bessel_k1_out_out(c10::DispatchKeySet ks, const at::Tensor & x, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_scaled_modified_bessel_k1_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_t_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_shifted_chebyshev_polynomial_t_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_t_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_shifted_chebyshev_polynomial_t_x_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_t_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_shifted_chebyshev_polynomial_t_n_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_u_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_shifted_chebyshev_polynomial_u_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_u_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_shifted_chebyshev_polynomial_u_x_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_u_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_shifted_chebyshev_polynomial_u_n_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_v_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_shifted_chebyshev_polynomial_v_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_v_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_shifted_chebyshev_polynomial_v_x_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_v_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_shifted_chebyshev_polynomial_v_n_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_w_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_shifted_chebyshev_polynomial_w_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_w_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_shifted_chebyshev_polynomial_w_x_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_w_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_shifted_chebyshev_polynomial_w_n_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_spherical_bessel_j0_out_out(c10::DispatchKeySet ks, const at::Tensor & x, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_spherical_bessel_j0_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, x, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_xlog1py_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_xlog1py_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_xlog1py_out_self_scalar_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_xlog1py_self_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_xlog1py_out_other_scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_xlog1py_other_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_zeta_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_zeta_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_zeta_out_self_scalar_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_zeta_self_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_zeta_out_other_scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::special_zeta_other_scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
::std::vector<at::Tensor> split_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt split_size, int64_t dim) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::split_Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, split_size, dim);
  })();
  for (auto view_idx : c10::irange(_tmp.size())) {
    std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
    std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
    if (false ||
        !self.unsafeGetTensorImpl()->support_as_strided() ||
        self.unsafeGetTensorImpl()->is_python_dispatch() ||
        c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
      func = std::make_unique<torch::autograd::generated::SplitTensorViewFunc>(split_size, dim, view_idx);
      rev_func = [=](const at::Tensor& input_view) {
        return at::functionalization::FunctionalInverses::split_Tensor_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, view_idx, split_size, dim);
      };
    }
    as_view(/* base */ self, /* output */ _tmp[view_idx], /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::MULTI_OUTPUT_NODE : CreationMeta::NO_GRAD_MODE));
  }
  auto result = std::move(_tmp);
  return result;
}
::std::vector<at::Tensor> split_with_sizes(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef split_sizes, int64_t dim) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::split_with_sizes::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, split_sizes, dim);
  })();
  for (auto view_idx : c10::irange(_tmp.size())) {
    std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
    std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
    if (false ||
        !self.unsafeGetTensorImpl()->support_as_strided() ||
        self.unsafeGetTensorImpl()->is_python_dispatch() ||
        c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
      auto split_sizes_vec = split_sizes.vec();
      func = std::make_unique<torch::autograd::generated::SplitWithSizesViewFunc>(split_sizes, dim, view_idx);
      rev_func = [=](const at::Tensor& input_view) {
        return at::functionalization::FunctionalInverses::split_with_sizes_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, view_idx, split_sizes_vec, dim);
      };
    }
    as_view(/* base */ self, /* output */ _tmp[view_idx], /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::MULTI_OUTPUT_NODE : CreationMeta::NO_GRAD_MODE));
  }
  auto result = std::move(_tmp);
  return result;
}
at::Tensor & sqrt_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sqrt_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & sqrt_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sqrt_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor squeeze(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::squeeze::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::SqueezeViewFunc>();
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::squeeze_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor squeeze_dim(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::squeeze_dim::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::SqueezeDimViewFunc>(dim);
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::squeeze_dim_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, dim);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor squeeze_dims(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::squeeze_dims::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    auto dim_vec = dim.vec();
    func = std::make_unique<torch::autograd::generated::SqueezeDimsViewFunc>(dim);
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::squeeze_dims_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, dim_vec);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & squeeze_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::squeeze_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & squeeze__dim(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::squeeze__dim::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim);
  }
  increment_version(self);
  return self;
}
at::Tensor & squeeze__dims(c10::DispatchKeySet ks, at::Tensor & self, at::IntArrayRef dim) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::squeeze__dims::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim);
  }
  increment_version(self);
  return self;
}
at::Tensor & squeeze_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::squeeze_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & squeeze_copy_out_dim_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::squeeze_copy_dim_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & squeeze_copy_out_dims_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::squeeze_copy_dims_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & sspaddmm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sspaddmm_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, mat1, mat2, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & stack_out_out(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::stack_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, tensors, dim, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> std_mean_out_correction_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::std_mean_correction_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, correction, keepdim, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & std_out_correction_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::std_correction_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, correction, keepdim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & sub__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sub__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & sub__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sub__Scalar::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & sub_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sub_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & sub_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sub_Scalar_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & sum_out_IntList_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sum_IntList_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & sum_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::sum_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor t(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::t::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::TViewFunc>();
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::t_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & t_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::t_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & t_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::t_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & take_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & index, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::take_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, index, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & tan_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::tan_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & tan_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::tan_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & tanh_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::tanh_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & tanh_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::tanh_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & tanh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::tanh_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & threshold_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::threshold_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, threshold, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & threshold_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::threshold_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, threshold, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & threshold_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::threshold_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, threshold, value, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & to_mkldnn_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::to_mkldnn_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & to_padded_tensor_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double padding, at::OptionalSymIntArrayRef output_size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::to_padded_tensor_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, padding, output_size, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> topk_out_values(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt k, int64_t dim, bool largest, bool sorted, at::Tensor & values, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::topk_values::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, k, dim, largest, sorted, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
at::Tensor & trace_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::trace_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor transpose_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim0, int64_t dim1) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::transpose_int::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim0, dim1);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::TransposeIntViewFunc>(dim0, dim1);
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::transpose_int_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, dim0, dim1);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & transpose_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim0, int64_t dim1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::transpose_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim0, dim1);
  }
  increment_version(self);
  return self;
}
at::Tensor & transpose_copy_out_int_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim0, int64_t dim1, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::transpose_copy_int_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim0, dim1, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> triangular_solve_out_X(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & A, bool upper, bool transpose, bool unitriangular, at::Tensor & X, at::Tensor & M) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::triangular_solve_X::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, A, upper, transpose, unitriangular, X, M);
  }
  increment_version(X);
  increment_version(M);
  return std::forward_as_tuple(X, M);
}
at::Tensor & tril_(c10::DispatchKeySet ks, at::Tensor & self, int64_t diagonal) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::tril_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, diagonal);
  }
  increment_version(self);
  return self;
}
at::Tensor & tril_indices_out_out(c10::DispatchKeySet ks, int64_t row, int64_t col, int64_t offset, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::tril_indices_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, row, col, offset, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & tril_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::tril_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, diagonal, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & triu_(c10::DispatchKeySet ks, at::Tensor & self, int64_t diagonal) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::triu_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, diagonal);
  }
  increment_version(self);
  return self;
}
at::Tensor & triu_indices_out_out(c10::DispatchKeySet ks, int64_t row, int64_t col, int64_t offset, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::triu_indices_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, row, col, offset, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & triu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::triu_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, diagonal, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & trunc_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::trunc_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & trunc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::trunc_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
::std::vector<at::Tensor> unbind_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::unbind_int::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim);
  })();
  for (auto view_idx : c10::irange(_tmp.size())) {
    std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
    std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
    if (false ||
        !self.unsafeGetTensorImpl()->support_as_strided() ||
        self.unsafeGetTensorImpl()->is_python_dispatch() ||
        c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
      func = std::make_unique<torch::autograd::generated::UnbindIntViewFunc>(dim, view_idx);
      rev_func = [=](const at::Tensor& input_view) {
        return at::functionalization::FunctionalInverses::unbind_int_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, view_idx, dim);
      };
    }
    as_view(/* base */ self, /* output */ _tmp[view_idx], /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::MULTI_OUTPUT_NODE : CreationMeta::NO_GRAD_MODE));
  }
  auto result = std::move(_tmp);
  return result;
}
at::Tensor unfold(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dimension, int64_t size, int64_t step) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::unfold::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dimension, size, step);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::UnfoldViewFunc>(dimension, size, step);
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::unfold_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, dimension, size, step);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & unfold_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_in, c10::SymIntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::unfold_backward_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_in, input_sizes, dim, size, step, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & unfold_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dimension, int64_t size, int64_t step, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::unfold_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dimension, size, step, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & uniform_(c10::DispatchKeySet ks, at::Tensor & self, double from, double to, ::std::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::uniform_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, from, to, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & uniform_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double from, double to, ::std::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::uniform_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, from, to, generator, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> unique_consecutive_out_out(c10::DispatchKeySet ks, const at::Tensor & self, bool return_inverse, bool return_counts, ::std::optional<int64_t> dim, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::unique_consecutive_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, return_inverse, return_counts, dim, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> unique_dim_consecutive_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool return_inverse, bool return_counts, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::unique_dim_consecutive_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, return_inverse, return_counts, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> unique_dim_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::unique_dim_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, sorted, return_inverse, return_counts, out0, out1, out2);
  }
  increment_version(out0);
  increment_version(out1);
  increment_version(out2);
  return std::forward_as_tuple(out0, out1, out2);
}
at::Tensor unsqueeze(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::unsqueeze::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::UnsqueezeViewFunc>(dim);
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::unsqueeze_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, dim);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & unsqueeze_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::unsqueeze_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim);
  }
  increment_version(self);
  return self;
}
at::Tensor & unsqueeze_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::unsqueeze_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & upsample_bicubic2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::upsample_bicubic2d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & upsample_bicubic2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::upsample_bicubic2d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, align_corners, scales_h, scales_w, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & upsample_bilinear2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::upsample_bilinear2d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & upsample_bilinear2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::upsample_bilinear2d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, align_corners, scales_h, scales_w, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & upsample_linear1d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, bool align_corners, ::std::optional<double> scales, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::upsample_linear1d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output_size, input_size, align_corners, scales, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & upsample_linear1d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::upsample_linear1d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, align_corners, scales, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & upsample_nearest1d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, ::std::optional<double> scales, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::upsample_nearest1d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output_size, input_size, scales, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & upsample_nearest1d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::upsample_nearest1d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, scales, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & upsample_nearest2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::upsample_nearest2d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output_size, input_size, scales_h, scales_w, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & upsample_nearest2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::upsample_nearest2d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, scales_h, scales_w, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & upsample_nearest3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, ::std::optional<double> scales_d, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::upsample_nearest3d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & upsample_nearest3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_d, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::upsample_nearest3d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, scales_d, scales_h, scales_w, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & upsample_trilinear3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, bool align_corners, ::std::optional<double> scales_d, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::upsample_trilinear3d_backward_grad_input::redispatch(ks & c10::after_ADInplaceOrView_keyset, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & upsample_trilinear3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_d, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::upsample_trilinear3d_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, output_size, align_corners, scales_d, scales_h, scales_w, out);
  }
  increment_version(out);
  return out;
}
at::Tensor values(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::values::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::ValuesViewFunc>();
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::values_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & values_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::values_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> var_mean_out_correction_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim, at::Tensor & out0, at::Tensor & out1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::var_mean_correction_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, correction, keepdim, out0, out1);
  }
  increment_version(out0);
  increment_version(out1);
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & var_out_correction_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::var_correction_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dim, correction, keepdim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & vdot_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::vdot_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor view(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::view::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (false ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    auto size_vec = size.vec();
    func = std::make_unique<torch::autograd::generated::ViewViewFunc>(size);
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::view_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView, size_vec);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor view_dtype(c10::DispatchKeySet ks, const at::Tensor & self, at::ScalarType dtype) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::view_dtype::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dtype);
  })();
  auto result = as_view(self, _tmp, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false);
  return result;
}
at::Tensor view_as_complex(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::view_as_complex::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (true ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::ViewAsComplexViewFunc>();
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::view_as_complex_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & view_as_complex_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::view_as_complex_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor view_as_real(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::_ops::view_as_real::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  std::unique_ptr<torch::autograd::ViewFunc> func(nullptr);
  std::function<at::Tensor(const at::Tensor&)> rev_func=nullptr;
  if (true ||
      !self.unsafeGetTensorImpl()->support_as_strided() ||
      self.unsafeGetTensorImpl()->is_python_dispatch() ||
      c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    func = std::make_unique<torch::autograd::generated::ViewAsRealViewFunc>();
    rev_func = [=](const at::Tensor& input_view) {
      return at::functionalization::FunctionalInverses::view_as_real_inverse(self, input_view, at::functionalization::InverseReturnMode::AlwaysView);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ std::move(func), /* rev_view_func */ rev_func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & view_as_real_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::view_as_real_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & view_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::view_copy_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & view_copy_out_dtype_out(c10::DispatchKeySet ks, const at::Tensor & self, at::ScalarType dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::view_copy_dtype_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & where_out_self_out(c10::DispatchKeySet ks, const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::where_self_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, condition, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & xlogy__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::xlogy__Tensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & xlogy__Scalar_Other(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::xlogy__Scalar_Other::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & xlogy_out_OutTensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::xlogy_OutTensor::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & xlogy_out_OutScalar_Self(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::xlogy_OutScalar_Self::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & xlogy_out_OutScalar_Other(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::xlogy_OutScalar_Other::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & zero_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::zero_::redispatch(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & zero_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::zero_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & zeros_like_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::zeros_like_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, self, memory_format, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & zeros_out_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::zeros_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & zeros_out_names_out(c10::DispatchKeySet ks, at::IntArrayRef size, ::std::optional<at::DimnameList> names, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::_ops::zeros_names_out::redispatch(ks & c10::after_ADInplaceOrView_keyset, size, names, out);
  }
  increment_version(out);
  return out;
}
}  // namespace
}  // namespace ADInplaceOrView

namespace {

TORCH_LIBRARY_IMPL(aten, ADInplaceOrView, m) {
  m.impl("__ilshift__.Scalar",
         TORCH_FN(ADInplaceOrView::__ilshift___Scalar)
  );
  m.impl("__ilshift__.Tensor",
         TORCH_FN(ADInplaceOrView::__ilshift___Tensor)
  );
  m.impl("__irshift__.Scalar",
         TORCH_FN(ADInplaceOrView::__irshift___Scalar)
  );
  m.impl("__irshift__.Tensor",
         TORCH_FN(ADInplaceOrView::__irshift___Tensor)
  );
  m.impl("__lshift__.Scalar_out",
         TORCH_FN(ADInplaceOrView::__lshift___out_Scalar_out)
  );
  m.impl("__lshift__.Tensor_out",
         TORCH_FN(ADInplaceOrView::__lshift___out_Tensor_out)
  );
  m.impl("__rshift__.Scalar_out",
         TORCH_FN(ADInplaceOrView::__rshift___out_Scalar_out)
  );
  m.impl("__rshift__.Tensor_out",
         TORCH_FN(ADInplaceOrView::__rshift___out_Tensor_out)
  );
  m.impl("_adaptive_avg_pool2d_backward.out",
         TORCH_FN(ADInplaceOrView::_adaptive_avg_pool2d_backward_out_out)
  );
  m.impl("_adaptive_avg_pool2d.out",
         TORCH_FN(ADInplaceOrView::_adaptive_avg_pool2d_out_out)
  );
  m.impl("_adaptive_avg_pool3d_backward.out",
         TORCH_FN(ADInplaceOrView::_adaptive_avg_pool3d_backward_out_out)
  );
  m.impl("_adaptive_avg_pool3d.out",
         TORCH_FN(ADInplaceOrView::_adaptive_avg_pool3d_out_out)
  );
  m.impl("_add_relu_.Tensor",
         TORCH_FN(ADInplaceOrView::_add_relu__Tensor)
  );
  m.impl("_add_relu_.Scalar",
         TORCH_FN(ADInplaceOrView::_add_relu__Scalar)
  );
  m.impl("_add_relu.out",
         TORCH_FN(ADInplaceOrView::_add_relu_out_out)
  );
  m.impl("_add_relu.Scalar_out",
         TORCH_FN(ADInplaceOrView::_add_relu_out_Scalar_out)
  );
  m.impl("_addmm_activation.out",
         TORCH_FN(ADInplaceOrView::_addmm_activation_out_out)
  );
  m.impl("_aminmax.out",
         TORCH_FN(ADInplaceOrView::_aminmax_out_out)
  );
  m.impl("_aminmax.dim_out",
         TORCH_FN(ADInplaceOrView::_aminmax_out_dim_out)
  );
  m.impl("_amp_update_scale_",
         TORCH_FN(ADInplaceOrView::_amp_update_scale_)
  );
  m.impl("_amp_update_scale.out",
         TORCH_FN(ADInplaceOrView::_amp_update_scale_out_out)
  );
  m.impl("_batch_norm_no_update.out",
         TORCH_FN(ADInplaceOrView::_batch_norm_no_update_out_out)
  );
  m.impl("_batch_norm_with_update.out",
         TORCH_FN(ADInplaceOrView::_batch_norm_with_update_out_out)
  );
  m.impl("_cdist_backward.out",
         TORCH_FN(ADInplaceOrView::_cdist_backward_out_out)
  );
  m.impl("_cdist_forward.out",
         TORCH_FN(ADInplaceOrView::_cdist_forward_out_out)
  );
  m.impl("_cholesky_solve_helper.out",
         TORCH_FN(ADInplaceOrView::_cholesky_solve_helper_out_out)
  );
  m.impl("_chunk_cat.out",
         TORCH_FN(ADInplaceOrView::_chunk_cat_out_out)
  );
  m.impl("_coalesce.out",
         TORCH_FN(ADInplaceOrView::_coalesce_out_out)
  );
  m.impl("_coalesced_",
         TORCH_FN(ADInplaceOrView::_coalesced_)
  );
  m.impl("_coalesced.out",
         TORCH_FN(ADInplaceOrView::_coalesced_out_out)
  );
  m.impl("_compute_linear_combination.out",
         TORCH_FN(ADInplaceOrView::_compute_linear_combination_out_out)
  );
  m.impl("_conj",
         TORCH_FN(ADInplaceOrView::_conj)
  );
  m.impl("_conj_copy.out",
         TORCH_FN(ADInplaceOrView::_conj_copy_out_out)
  );
  m.impl("_conj_physical.out",
         TORCH_FN(ADInplaceOrView::_conj_physical_out_out)
  );
  m.impl("_conv_depthwise2d.out",
         TORCH_FN(ADInplaceOrView::_conv_depthwise2d_out_out)
  );
  m.impl("_convert_indices_from_coo_to_csr.out",
         TORCH_FN(ADInplaceOrView::_convert_indices_from_coo_to_csr_out_out)
  );
  m.impl("_convert_indices_from_csr_to_coo.out",
         TORCH_FN(ADInplaceOrView::_convert_indices_from_csr_to_coo_out_out)
  );
  m.impl("_convolution.out",
         TORCH_FN(ADInplaceOrView::_convolution_out_out)
  );
  m.impl("_copy_from_and_resize.out",
         TORCH_FN(ADInplaceOrView::_copy_from_and_resize_out_out)
  );
  m.impl("_copy_from.out",
         TORCH_FN(ADInplaceOrView::_copy_from_out_out)
  );
  m.impl("_ctc_loss_backward.out",
         TORCH_FN(ADInplaceOrView::_ctc_loss_backward_out_out)
  );
  m.impl("_ctc_loss.out",
         TORCH_FN(ADInplaceOrView::_ctc_loss_out_out)
  );
  m.impl("_ctc_loss.Tensor_out",
         TORCH_FN(ADInplaceOrView::_ctc_loss_out_Tensor_out)
  );
  m.impl("_cudnn_ctc_loss.out",
         TORCH_FN(ADInplaceOrView::_cudnn_ctc_loss_out_out)
  );
  m.impl("_cudnn_init_dropout_state.out",
         TORCH_FN(ADInplaceOrView::_cudnn_init_dropout_state_out_out)
  );
  m.impl("_cudnn_rnn_flatten_weight.out",
         TORCH_FN(ADInplaceOrView::_cudnn_rnn_flatten_weight_out_out)
  );
  m.impl("_cudnn_rnn.out",
         TORCH_FN(ADInplaceOrView::_cudnn_rnn_out_out)
  );
  m.impl("_dirichlet_grad.out",
         TORCH_FN(ADInplaceOrView::_dirichlet_grad_out_out)
  );
  m.impl("_efficientzerotensor.out",
         TORCH_FN(ADInplaceOrView::_efficientzerotensor_out_out)
  );
  m.impl("_embedding_bag_dense_backward.out",
         TORCH_FN(ADInplaceOrView::_embedding_bag_dense_backward_out_out)
  );
  m.impl("_embedding_bag_forward_only.out",
         TORCH_FN(ADInplaceOrView::_embedding_bag_forward_only_out_out)
  );
  m.impl("_embedding_bag.out",
         TORCH_FN(ADInplaceOrView::_embedding_bag_out_out)
  );
  m.impl("_embedding_bag_per_sample_weights_backward.out",
         TORCH_FN(ADInplaceOrView::_embedding_bag_per_sample_weights_backward_out_out)
  );
  m.impl("_empty_affine_quantized.out",
         TORCH_FN(ADInplaceOrView::_empty_affine_quantized_out_out)
  );
  m.impl("_empty_per_channel_affine_quantized.out",
         TORCH_FN(ADInplaceOrView::_empty_per_channel_affine_quantized_out_out)
  );
  m.impl("_euclidean_dist.out",
         TORCH_FN(ADInplaceOrView::_euclidean_dist_out_out)
  );
  m.impl("_fake_quantize_learnable_per_channel_affine.out",
         TORCH_FN(ADInplaceOrView::_fake_quantize_learnable_per_channel_affine_out_out)
  );
  m.impl("_fake_quantize_learnable_per_tensor_affine.out",
         TORCH_FN(ADInplaceOrView::_fake_quantize_learnable_per_tensor_affine_out_out)
  );
  m.impl("_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.out",
         TORCH_FN(ADInplaceOrView::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_out_out)
  );
  m.impl("_fft_c2c.out",
         TORCH_FN(ADInplaceOrView::_fft_c2c_out_out)
  );
  m.impl("_fft_c2r.out",
         TORCH_FN(ADInplaceOrView::_fft_c2r_out_out)
  );
  m.impl("_fft_r2c.out",
         TORCH_FN(ADInplaceOrView::_fft_r2c_out_out)
  );
  m.impl("_fill_mem_eff_dropout_mask_",
         TORCH_FN(ADInplaceOrView::_fill_mem_eff_dropout_mask_)
  );
  m.impl("_foobar.out",
         TORCH_FN(ADInplaceOrView::_foobar_out_out)
  );
  m.impl("_fused_dropout.out",
         TORCH_FN(ADInplaceOrView::_fused_dropout_out_out)
  );
  m.impl("_fused_moving_avg_obs_fq_helper.out",
         TORCH_FN(ADInplaceOrView::_fused_moving_avg_obs_fq_helper_out_out)
  );
  m.impl("_fw_primal_copy.out",
         TORCH_FN(ADInplaceOrView::_fw_primal_copy_out_out)
  );
  m.impl("_grid_sampler_2d_cpu_fallback.out",
         TORCH_FN(ADInplaceOrView::_grid_sampler_2d_cpu_fallback_out_out)
  );
  m.impl("_histogramdd_from_bin_cts.out",
         TORCH_FN(ADInplaceOrView::_histogramdd_from_bin_cts_out_out)
  );
  m.impl("_histogramdd_from_bin_tensors.out",
         TORCH_FN(ADInplaceOrView::_histogramdd_from_bin_tensors_out_out)
  );
  m.impl("_index_put_impl_",
         TORCH_FN(ADInplaceOrView::_index_put_impl_)
  );
  m.impl("_index_put_impl.out",
         TORCH_FN(ADInplaceOrView::_index_put_impl_out_out)
  );
  m.impl("_indices",
         TORCH_FN(ADInplaceOrView::_indices)
  );
  m.impl("_indices_copy.out",
         TORCH_FN(ADInplaceOrView::_indices_copy_out_out)
  );
  m.impl("_int_mm.out",
         TORCH_FN(ADInplaceOrView::_int_mm_out_out)
  );
  m.impl("_linalg_det.result",
         TORCH_FN(ADInplaceOrView::_linalg_det_out_result)
  );
  m.impl("_linalg_eigh.eigenvalues",
         TORCH_FN(ADInplaceOrView::_linalg_eigh_out_eigenvalues)
  );
  m.impl("_linalg_slogdet.sign",
         TORCH_FN(ADInplaceOrView::_linalg_slogdet_out_sign)
  );
  m.impl("_linalg_solve_ex.result",
         TORCH_FN(ADInplaceOrView::_linalg_solve_ex_out_result)
  );
  m.impl("_linalg_svd.U",
         TORCH_FN(ADInplaceOrView::_linalg_svd_out_U)
  );
  m.impl("_log_softmax_backward_data.out",
         TORCH_FN(ADInplaceOrView::_log_softmax_backward_data_out_out)
  );
  m.impl("_log_softmax.out",
         TORCH_FN(ADInplaceOrView::_log_softmax_out_out)
  );
  m.impl("_logcumsumexp.out",
         TORCH_FN(ADInplaceOrView::_logcumsumexp_out_out)
  );
  m.impl("_lstm_mps.out",
         TORCH_FN(ADInplaceOrView::_lstm_mps_out_out)
  );
  m.impl("_make_dual_copy.out",
         TORCH_FN(ADInplaceOrView::_make_dual_copy_out_out)
  );
  m.impl("_make_per_channel_quantized_tensor.out",
         TORCH_FN(ADInplaceOrView::_make_per_channel_quantized_tensor_out_out)
  );
  m.impl("_make_per_tensor_quantized_tensor.out",
         TORCH_FN(ADInplaceOrView::_make_per_tensor_quantized_tensor_out_out)
  );
  m.impl("_masked_scale.out",
         TORCH_FN(ADInplaceOrView::_masked_scale_out_out)
  );
  m.impl("_masked_softmax_backward.out",
         TORCH_FN(ADInplaceOrView::_masked_softmax_backward_out_out)
  );
  m.impl("_masked_softmax.out",
         TORCH_FN(ADInplaceOrView::_masked_softmax_out_out)
  );
  m.impl("_mkldnn_reshape.out",
         TORCH_FN(ADInplaceOrView::_mkldnn_reshape_out_out)
  );
  m.impl("_mkldnn_transpose_",
         TORCH_FN(ADInplaceOrView::_mkldnn_transpose_)
  );
  m.impl("_mkldnn_transpose.out",
         TORCH_FN(ADInplaceOrView::_mkldnn_transpose_out_out)
  );
  m.impl("_mps_convolution.out",
         TORCH_FN(ADInplaceOrView::_mps_convolution_out_out)
  );
  m.impl("_mps_convolution_transpose.out",
         TORCH_FN(ADInplaceOrView::_mps_convolution_transpose_out_out)
  );
  m.impl("_native_batch_norm_legit_no_training.out",
         TORCH_FN(ADInplaceOrView::_native_batch_norm_legit_no_training_out_out)
  );
  m.impl("_native_batch_norm_legit.out",
         TORCH_FN(ADInplaceOrView::_native_batch_norm_legit_out_out)
  );
  m.impl("_native_batch_norm_legit.no_stats_out",
         TORCH_FN(ADInplaceOrView::_native_batch_norm_legit_out_no_stats_out)
  );
  m.impl("_native_multi_head_attention.out",
         TORCH_FN(ADInplaceOrView::_native_multi_head_attention_out_out)
  );
  m.impl("_neg_view",
         TORCH_FN(ADInplaceOrView::_neg_view)
  );
  m.impl("_neg_view_copy.out",
         TORCH_FN(ADInplaceOrView::_neg_view_copy_out_out)
  );
  m.impl("_nested_from_padded_and_nested_example.out",
         TORCH_FN(ADInplaceOrView::_nested_from_padded_and_nested_example_out_out)
  );
  m.impl("_nested_from_padded.out",
         TORCH_FN(ADInplaceOrView::_nested_from_padded_out_out)
  );
  m.impl("_nested_get_values",
         TORCH_FN(ADInplaceOrView::_nested_get_values)
  );
  m.impl("_nested_get_values_copy.out",
         TORCH_FN(ADInplaceOrView::_nested_get_values_copy_out_out)
  );
  m.impl("_nested_tensor_from_mask.out",
         TORCH_FN(ADInplaceOrView::_nested_tensor_from_mask_out_out)
  );
  m.impl("_nested_tensor_from_tensor_list.out",
         TORCH_FN(ADInplaceOrView::_nested_tensor_from_tensor_list_out_out)
  );
  m.impl("_nested_tensor_size.out",
         TORCH_FN(ADInplaceOrView::_nested_tensor_size_out_out)
  );
  m.impl("_nested_tensor_storage_offsets.out",
         TORCH_FN(ADInplaceOrView::_nested_tensor_storage_offsets_out_out)
  );
  m.impl("_nested_tensor_strides.out",
         TORCH_FN(ADInplaceOrView::_nested_tensor_strides_out_out)
  );
  m.impl("_nested_view_from_buffer",
         TORCH_FN(ADInplaceOrView::_nested_view_from_buffer)
  );
  m.impl("_nested_view_from_buffer_copy.out",
         TORCH_FN(ADInplaceOrView::_nested_view_from_buffer_copy_out_out)
  );
  m.impl("_nested_view_from_jagged",
         TORCH_FN(ADInplaceOrView::_nested_view_from_jagged)
  );
  m.impl("_nested_view_from_jagged_copy.out",
         TORCH_FN(ADInplaceOrView::_nested_view_from_jagged_copy_out_out)
  );
  m.impl("_new_zeros_with_same_feature_meta.out",
         TORCH_FN(ADInplaceOrView::_new_zeros_with_same_feature_meta_out_out)
  );
  m.impl("_nnpack_spatial_convolution.out",
         TORCH_FN(ADInplaceOrView::_nnpack_spatial_convolution_out_out)
  );
  m.impl("_pack_padded_sequence.out",
         TORCH_FN(ADInplaceOrView::_pack_padded_sequence_out_out)
  );
  m.impl("_pdist_backward.out",
         TORCH_FN(ADInplaceOrView::_pdist_backward_out_out)
  );
  m.impl("_pdist_forward.out",
         TORCH_FN(ADInplaceOrView::_pdist_forward_out_out)
  );
  m.impl("_pin_memory.out",
         TORCH_FN(ADInplaceOrView::_pin_memory_out_out)
  );
  m.impl("_reshape_alias",
         TORCH_FN(ADInplaceOrView::_reshape_alias)
  );
  m.impl("_reshape_alias_copy.out",
         TORCH_FN(ADInplaceOrView::_reshape_alias_copy_out_out)
  );
  m.impl("_resize_output_",
         TORCH_FN(ADInplaceOrView::_resize_output_)
  );
  m.impl("_resize_output.out",
         TORCH_FN(ADInplaceOrView::_resize_output_out_out)
  );
  m.impl("_sample_dirichlet.out",
         TORCH_FN(ADInplaceOrView::_sample_dirichlet_out_out)
  );
  m.impl("_scaled_mm.out",
         TORCH_FN(ADInplaceOrView::_scaled_mm_out_out)
  );
  m.impl("_segment_reduce_backward.out",
         TORCH_FN(ADInplaceOrView::_segment_reduce_backward_out_out)
  );
  m.impl("_slow_conv2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::_slow_conv2d_backward_out_grad_input)
  );
  m.impl("_slow_conv2d_backward.output_mask_out",
         TORCH_FN(ADInplaceOrView::_slow_conv2d_backward_out_output_mask_out)
  );
  m.impl("_slow_conv2d_forward.output",
         TORCH_FN(ADInplaceOrView::_slow_conv2d_forward_out_output)
  );
  m.impl("_softmax_backward_data.out",
         TORCH_FN(ADInplaceOrView::_softmax_backward_data_out_out)
  );
  m.impl("_softmax.out",
         TORCH_FN(ADInplaceOrView::_softmax_out_out)
  );
  m.impl("_sparse_addmm.out",
         TORCH_FN(ADInplaceOrView::_sparse_addmm_out_out)
  );
  m.impl("_sparse_broadcast_to_copy.out",
         TORCH_FN(ADInplaceOrView::_sparse_broadcast_to_copy_out_out)
  );
  m.impl("_sparse_coo_tensor_with_dims_and_tensors.out",
         TORCH_FN(ADInplaceOrView::_sparse_coo_tensor_with_dims_and_tensors_out_out)
  );
  m.impl("_sparse_coo_tensor_with_dims.out",
         TORCH_FN(ADInplaceOrView::_sparse_coo_tensor_with_dims_out_out)
  );
  m.impl("_sparse_csr_prod.dim_dtype_out",
         TORCH_FN(ADInplaceOrView::_sparse_csr_prod_out_dim_dtype_out)
  );
  m.impl("_sparse_csr_sum.dim_dtype_out",
         TORCH_FN(ADInplaceOrView::_sparse_csr_sum_out_dim_dtype_out)
  );
  m.impl("_sparse_log_softmax_backward_data.out",
         TORCH_FN(ADInplaceOrView::_sparse_log_softmax_backward_data_out_out)
  );
  m.impl("_sparse_log_softmax.out",
         TORCH_FN(ADInplaceOrView::_sparse_log_softmax_out_out)
  );
  m.impl("_sparse_mask_projection.out",
         TORCH_FN(ADInplaceOrView::_sparse_mask_projection_out_out)
  );
  m.impl("_sparse_softmax_backward_data.out",
         TORCH_FN(ADInplaceOrView::_sparse_softmax_backward_data_out_out)
  );
  m.impl("_sparse_softmax.out",
         TORCH_FN(ADInplaceOrView::_sparse_softmax_out_out)
  );
  m.impl("_sparse_sparse_matmul.out",
         TORCH_FN(ADInplaceOrView::_sparse_sparse_matmul_out_out)
  );
  m.impl("_sparse_sum_backward.out",
         TORCH_FN(ADInplaceOrView::_sparse_sum_backward_out_out)
  );
  m.impl("_sparse_sum.dim_out",
         TORCH_FN(ADInplaceOrView::_sparse_sum_out_dim_out)
  );
  m.impl("_spdiags.out",
         TORCH_FN(ADInplaceOrView::_spdiags_out_out)
  );
  m.impl("_stack.out",
         TORCH_FN(ADInplaceOrView::_stack_out_out)
  );
  m.impl("_standard_gamma_grad.out",
         TORCH_FN(ADInplaceOrView::_standard_gamma_grad_out_out)
  );
  m.impl("_standard_gamma.out",
         TORCH_FN(ADInplaceOrView::_standard_gamma_out_out)
  );
  m.impl("_test_autograd_multiple_dispatch.fullcoverage_out",
         TORCH_FN(ADInplaceOrView::_test_autograd_multiple_dispatch_out_fullcoverage_out)
  );
  m.impl("_test_autograd_multiple_dispatch_view",
         TORCH_FN(ADInplaceOrView::_test_autograd_multiple_dispatch_view)
  );
  m.impl("_test_autograd_multiple_dispatch_view_copy.out",
         TORCH_FN(ADInplaceOrView::_test_autograd_multiple_dispatch_view_copy_out_out)
  );
  m.impl("_test_functorch_fallback.out",
         TORCH_FN(ADInplaceOrView::_test_functorch_fallback_out_out)
  );
  m.impl("_test_optional_filled_intlist.out",
         TORCH_FN(ADInplaceOrView::_test_optional_filled_intlist_out_out)
  );
  m.impl("_test_optional_floatlist.out",
         TORCH_FN(ADInplaceOrView::_test_optional_floatlist_out_out)
  );
  m.impl("_test_optional_intlist.out",
         TORCH_FN(ADInplaceOrView::_test_optional_intlist_out_out)
  );
  m.impl("_test_warn_in_autograd.out",
         TORCH_FN(ADInplaceOrView::_test_warn_in_autograd_out_out)
  );
  m.impl("_thnn_fused_gru_cell_backward.out",
         TORCH_FN(ADInplaceOrView::_thnn_fused_gru_cell_backward_out_out)
  );
  m.impl("_thnn_fused_gru_cell.out",
         TORCH_FN(ADInplaceOrView::_thnn_fused_gru_cell_out_out)
  );
  m.impl("_thnn_fused_lstm_cell_backward_impl.out",
         TORCH_FN(ADInplaceOrView::_thnn_fused_lstm_cell_backward_impl_out_out)
  );
  m.impl("_thnn_fused_lstm_cell.out",
         TORCH_FN(ADInplaceOrView::_thnn_fused_lstm_cell_out_out)
  );
  m.impl("_to_copy.out",
         TORCH_FN(ADInplaceOrView::_to_copy_out_out)
  );
  m.impl("_to_dense.out",
         TORCH_FN(ADInplaceOrView::_to_dense_out_out)
  );
  m.impl("_to_sparse_bsc.out",
         TORCH_FN(ADInplaceOrView::_to_sparse_bsc_out_out)
  );
  m.impl("_to_sparse_bsr.out",
         TORCH_FN(ADInplaceOrView::_to_sparse_bsr_out_out)
  );
  m.impl("_to_sparse_csc.out",
         TORCH_FN(ADInplaceOrView::_to_sparse_csc_out_out)
  );
  m.impl("_to_sparse_csr.out",
         TORCH_FN(ADInplaceOrView::_to_sparse_csr_out_out)
  );
  m.impl("_to_sparse.sparse_dim_out",
         TORCH_FN(ADInplaceOrView::_to_sparse_out_sparse_dim_out)
  );
  m.impl("_to_sparse.out",
         TORCH_FN(ADInplaceOrView::_to_sparse_out_out)
  );
  m.impl("_transform_bias_rescale_qkv.out",
         TORCH_FN(ADInplaceOrView::_transform_bias_rescale_qkv_out_out)
  );
  m.impl("_transformer_encoder_layer_fwd.out",
         TORCH_FN(ADInplaceOrView::_transformer_encoder_layer_fwd_out_out)
  );
  m.impl("_trilinear.out",
         TORCH_FN(ADInplaceOrView::_trilinear_out_out)
  );
  m.impl("_triton_multi_head_attention.out",
         TORCH_FN(ADInplaceOrView::_triton_multi_head_attention_out_out)
  );
  m.impl("_triton_scaled_dot_attention.out",
         TORCH_FN(ADInplaceOrView::_triton_scaled_dot_attention_out_out)
  );
  m.impl("_unique2.out",
         TORCH_FN(ADInplaceOrView::_unique2_out_out)
  );
  m.impl("_unique.out",
         TORCH_FN(ADInplaceOrView::_unique_out_out)
  );
  m.impl("_unsafe_view.out",
         TORCH_FN(ADInplaceOrView::_unsafe_view_out_out)
  );
  m.impl("_upsample_bicubic2d_aa_backward.grad_input",
         TORCH_FN(ADInplaceOrView::_upsample_bicubic2d_aa_backward_out_grad_input)
  );
  m.impl("_upsample_bicubic2d_aa.out",
         TORCH_FN(ADInplaceOrView::_upsample_bicubic2d_aa_out_out)
  );
  m.impl("_upsample_bilinear2d_aa_backward.grad_input",
         TORCH_FN(ADInplaceOrView::_upsample_bilinear2d_aa_backward_out_grad_input)
  );
  m.impl("_upsample_bilinear2d_aa.out",
         TORCH_FN(ADInplaceOrView::_upsample_bilinear2d_aa_out_out)
  );
  m.impl("_upsample_nearest_exact1d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::_upsample_nearest_exact1d_backward_out_grad_input)
  );
  m.impl("_upsample_nearest_exact1d.out",
         TORCH_FN(ADInplaceOrView::_upsample_nearest_exact1d_out_out)
  );
  m.impl("_upsample_nearest_exact2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::_upsample_nearest_exact2d_backward_out_grad_input)
  );
  m.impl("_upsample_nearest_exact2d.out",
         TORCH_FN(ADInplaceOrView::_upsample_nearest_exact2d_out_out)
  );
  m.impl("_upsample_nearest_exact3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::_upsample_nearest_exact3d_backward_out_grad_input)
  );
  m.impl("_upsample_nearest_exact3d.out",
         TORCH_FN(ADInplaceOrView::_upsample_nearest_exact3d_out_out)
  );
  m.impl("_values",
         TORCH_FN(ADInplaceOrView::_values)
  );
  m.impl("_values_copy.out",
         TORCH_FN(ADInplaceOrView::_values_copy_out_out)
  );
  m.impl("_weight_norm_interface_backward.out",
         TORCH_FN(ADInplaceOrView::_weight_norm_interface_backward_out_out)
  );
  m.impl("_weight_norm_interface.out",
         TORCH_FN(ADInplaceOrView::_weight_norm_interface_out_out)
  );
  m.impl("abs_",
         TORCH_FN(ADInplaceOrView::abs_)
  );
  m.impl("abs.out",
         TORCH_FN(ADInplaceOrView::abs_out_out)
  );
  m.impl("acos_",
         TORCH_FN(ADInplaceOrView::acos_)
  );
  m.impl("acos.out",
         TORCH_FN(ADInplaceOrView::acos_out_out)
  );
  m.impl("acosh_",
         TORCH_FN(ADInplaceOrView::acosh_)
  );
  m.impl("acosh.out",
         TORCH_FN(ADInplaceOrView::acosh_out_out)
  );
  m.impl("adaptive_avg_pool2d.out",
         TORCH_FN(ADInplaceOrView::adaptive_avg_pool2d_out_out)
  );
  m.impl("adaptive_avg_pool3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::adaptive_avg_pool3d_backward_out_grad_input)
  );
  m.impl("adaptive_avg_pool3d.out",
         TORCH_FN(ADInplaceOrView::adaptive_avg_pool3d_out_out)
  );
  m.impl("adaptive_max_pool2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::adaptive_max_pool2d_backward_out_grad_input)
  );
  m.impl("adaptive_max_pool2d.out",
         TORCH_FN(ADInplaceOrView::adaptive_max_pool2d_out_out)
  );
  m.impl("adaptive_max_pool3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::adaptive_max_pool3d_backward_out_grad_input)
  );
  m.impl("adaptive_max_pool3d.out",
         TORCH_FN(ADInplaceOrView::adaptive_max_pool3d_out_out)
  );
  m.impl("add_.Tensor",
         TORCH_FN(ADInplaceOrView::add__Tensor)
  );
  m.impl("add_.Scalar",
         TORCH_FN(ADInplaceOrView::add__Scalar)
  );
  m.impl("add.out",
         TORCH_FN(ADInplaceOrView::add_out_out)
  );
  m.impl("add.Scalar_out",
         TORCH_FN(ADInplaceOrView::add_out_Scalar_out)
  );
  m.impl("addbmm_",
         TORCH_FN(ADInplaceOrView::addbmm_)
  );
  m.impl("addbmm.out",
         TORCH_FN(ADInplaceOrView::addbmm_out_out)
  );
  m.impl("addcdiv_",
         TORCH_FN(ADInplaceOrView::addcdiv_)
  );
  m.impl("addcdiv.out",
         TORCH_FN(ADInplaceOrView::addcdiv_out_out)
  );
  m.impl("addcmul_",
         TORCH_FN(ADInplaceOrView::addcmul_)
  );
  m.impl("addcmul.out",
         TORCH_FN(ADInplaceOrView::addcmul_out_out)
  );
  m.impl("addmm_",
         TORCH_FN(ADInplaceOrView::addmm_)
  );
  m.impl("addmm.out",
         TORCH_FN(ADInplaceOrView::addmm_out_out)
  );
  m.impl("addmv_",
         TORCH_FN(ADInplaceOrView::addmv_)
  );
  m.impl("addmv.out",
         TORCH_FN(ADInplaceOrView::addmv_out_out)
  );
  m.impl("addr_",
         TORCH_FN(ADInplaceOrView::addr_)
  );
  m.impl("addr.out",
         TORCH_FN(ADInplaceOrView::addr_out_out)
  );
  m.impl("affine_grid_generator.out",
         TORCH_FN(ADInplaceOrView::affine_grid_generator_out_out)
  );
  m.impl("alias",
         TORCH_FN(ADInplaceOrView::alias)
  );
  m.impl("alias_copy.out",
         TORCH_FN(ADInplaceOrView::alias_copy_out_out)
  );
  m.impl("all.out",
         TORCH_FN(ADInplaceOrView::all_out_out)
  );
  m.impl("all.dims_out",
         TORCH_FN(ADInplaceOrView::all_out_dims_out)
  );
  m.impl("all.all_out",
         TORCH_FN(ADInplaceOrView::all_out_all_out)
  );
  m.impl("amax.out",
         TORCH_FN(ADInplaceOrView::amax_out_out)
  );
  m.impl("amin.out",
         TORCH_FN(ADInplaceOrView::amin_out_out)
  );
  m.impl("aminmax.out",
         TORCH_FN(ADInplaceOrView::aminmax_out_out)
  );
  m.impl("angle.out",
         TORCH_FN(ADInplaceOrView::angle_out_out)
  );
  m.impl("any.out",
         TORCH_FN(ADInplaceOrView::any_out_out)
  );
  m.impl("any.dims_out",
         TORCH_FN(ADInplaceOrView::any_out_dims_out)
  );
  m.impl("any.all_out",
         TORCH_FN(ADInplaceOrView::any_out_all_out)
  );
  m.impl("arange.out",
         TORCH_FN(ADInplaceOrView::arange_out_out)
  );
  m.impl("arange.start_out",
         TORCH_FN(ADInplaceOrView::arange_out_start_out)
  );
  m.impl("argmax.out",
         TORCH_FN(ADInplaceOrView::argmax_out_out)
  );
  m.impl("argmin.out",
         TORCH_FN(ADInplaceOrView::argmin_out_out)
  );
  m.impl("as_strided",
         TORCH_FN(ADInplaceOrView::as_strided)
  );
  m.impl("as_strided_",
         TORCH_FN(ADInplaceOrView::as_strided_)
  );
  m.impl("as_strided_copy.out",
         TORCH_FN(ADInplaceOrView::as_strided_copy_out_out)
  );
  m.impl("as_strided_scatter.out",
         TORCH_FN(ADInplaceOrView::as_strided_scatter_out_out)
  );
  m.impl("asin_",
         TORCH_FN(ADInplaceOrView::asin_)
  );
  m.impl("asin.out",
         TORCH_FN(ADInplaceOrView::asin_out_out)
  );
  m.impl("asinh_",
         TORCH_FN(ADInplaceOrView::asinh_)
  );
  m.impl("asinh.out",
         TORCH_FN(ADInplaceOrView::asinh_out_out)
  );
  m.impl("atan2_",
         TORCH_FN(ADInplaceOrView::atan2_)
  );
  m.impl("atan2.out",
         TORCH_FN(ADInplaceOrView::atan2_out_out)
  );
  m.impl("atan_",
         TORCH_FN(ADInplaceOrView::atan_)
  );
  m.impl("atan.out",
         TORCH_FN(ADInplaceOrView::atan_out_out)
  );
  m.impl("atanh_",
         TORCH_FN(ADInplaceOrView::atanh_)
  );
  m.impl("atanh.out",
         TORCH_FN(ADInplaceOrView::atanh_out_out)
  );
  m.impl("avg_pool2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::avg_pool2d_backward_out_grad_input)
  );
  m.impl("avg_pool2d.out",
         TORCH_FN(ADInplaceOrView::avg_pool2d_out_out)
  );
  m.impl("avg_pool3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::avg_pool3d_backward_out_grad_input)
  );
  m.impl("avg_pool3d.out",
         TORCH_FN(ADInplaceOrView::avg_pool3d_out_out)
  );
  m.impl("baddbmm_",
         TORCH_FN(ADInplaceOrView::baddbmm_)
  );
  m.impl("baddbmm.out",
         TORCH_FN(ADInplaceOrView::baddbmm_out_out)
  );
  m.impl("bartlett_window.out",
         TORCH_FN(ADInplaceOrView::bartlett_window_out_out)
  );
  m.impl("bartlett_window.periodic_out",
         TORCH_FN(ADInplaceOrView::bartlett_window_out_periodic_out)
  );
  m.impl("batch_norm_backward_elemt.out",
         TORCH_FN(ADInplaceOrView::batch_norm_backward_elemt_out_out)
  );
  m.impl("batch_norm_backward_reduce.out",
         TORCH_FN(ADInplaceOrView::batch_norm_backward_reduce_out_out)
  );
  m.impl("batch_norm_elemt.out",
         TORCH_FN(ADInplaceOrView::batch_norm_elemt_out_out)
  );
  m.impl("batch_norm_gather_stats.out",
         TORCH_FN(ADInplaceOrView::batch_norm_gather_stats_out_out)
  );
  m.impl("batch_norm_gather_stats_with_counts.out",
         TORCH_FN(ADInplaceOrView::batch_norm_gather_stats_with_counts_out_out)
  );
  m.impl("batch_norm_stats.out",
         TORCH_FN(ADInplaceOrView::batch_norm_stats_out_out)
  );
  m.impl("batch_norm_update_stats.out",
         TORCH_FN(ADInplaceOrView::batch_norm_update_stats_out_out)
  );
  m.impl("bernoulli_.Tensor",
         TORCH_FN(ADInplaceOrView::bernoulli__Tensor)
  );
  m.impl("bernoulli_.float",
         TORCH_FN(ADInplaceOrView::bernoulli__float)
  );
  m.impl("bernoulli.out",
         TORCH_FN(ADInplaceOrView::bernoulli_out_out)
  );
  m.impl("bernoulli.Tensor_out",
         TORCH_FN(ADInplaceOrView::bernoulli_out_Tensor_out)
  );
  m.impl("bernoulli.float_out",
         TORCH_FN(ADInplaceOrView::bernoulli_out_float_out)
  );
  m.impl("binary_cross_entropy_backward.grad_input",
         TORCH_FN(ADInplaceOrView::binary_cross_entropy_backward_out_grad_input)
  );
  m.impl("binary_cross_entropy.out",
         TORCH_FN(ADInplaceOrView::binary_cross_entropy_out_out)
  );
  m.impl("binary_cross_entropy_with_logits.out",
         TORCH_FN(ADInplaceOrView::binary_cross_entropy_with_logits_out_out)
  );
  m.impl("bincount.out",
         TORCH_FN(ADInplaceOrView::bincount_out_out)
  );
  m.impl("binomial.out",
         TORCH_FN(ADInplaceOrView::binomial_out_out)
  );
  m.impl("bitwise_and_.Scalar",
         TORCH_FN(ADInplaceOrView::bitwise_and__Scalar)
  );
  m.impl("bitwise_and_.Tensor",
         TORCH_FN(ADInplaceOrView::bitwise_and__Tensor)
  );
  m.impl("bitwise_and.Tensor_out",
         TORCH_FN(ADInplaceOrView::bitwise_and_out_Tensor_out)
  );
  m.impl("bitwise_and.Scalar_out",
         TORCH_FN(ADInplaceOrView::bitwise_and_out_Scalar_out)
  );
  m.impl("bitwise_and.Scalar_Tensor_out",
         TORCH_FN(ADInplaceOrView::bitwise_and_out_Scalar_Tensor_out)
  );
  m.impl("bitwise_left_shift_.Tensor",
         TORCH_FN(ADInplaceOrView::bitwise_left_shift__Tensor)
  );
  m.impl("bitwise_left_shift_.Tensor_Scalar",
         TORCH_FN(ADInplaceOrView::bitwise_left_shift__Tensor_Scalar)
  );
  m.impl("bitwise_left_shift.Tensor_out",
         TORCH_FN(ADInplaceOrView::bitwise_left_shift_out_Tensor_out)
  );
  m.impl("bitwise_left_shift.Tensor_Scalar_out",
         TORCH_FN(ADInplaceOrView::bitwise_left_shift_out_Tensor_Scalar_out)
  );
  m.impl("bitwise_left_shift.Scalar_Tensor_out",
         TORCH_FN(ADInplaceOrView::bitwise_left_shift_out_Scalar_Tensor_out)
  );
  m.impl("bitwise_not_",
         TORCH_FN(ADInplaceOrView::bitwise_not_)
  );
  m.impl("bitwise_not.out",
         TORCH_FN(ADInplaceOrView::bitwise_not_out_out)
  );
  m.impl("bitwise_or_.Scalar",
         TORCH_FN(ADInplaceOrView::bitwise_or__Scalar)
  );
  m.impl("bitwise_or_.Tensor",
         TORCH_FN(ADInplaceOrView::bitwise_or__Tensor)
  );
  m.impl("bitwise_or.Tensor_out",
         TORCH_FN(ADInplaceOrView::bitwise_or_out_Tensor_out)
  );
  m.impl("bitwise_or.Scalar_out",
         TORCH_FN(ADInplaceOrView::bitwise_or_out_Scalar_out)
  );
  m.impl("bitwise_or.Scalar_Tensor_out",
         TORCH_FN(ADInplaceOrView::bitwise_or_out_Scalar_Tensor_out)
  );
  m.impl("bitwise_right_shift_.Tensor",
         TORCH_FN(ADInplaceOrView::bitwise_right_shift__Tensor)
  );
  m.impl("bitwise_right_shift_.Tensor_Scalar",
         TORCH_FN(ADInplaceOrView::bitwise_right_shift__Tensor_Scalar)
  );
  m.impl("bitwise_right_shift.Tensor_out",
         TORCH_FN(ADInplaceOrView::bitwise_right_shift_out_Tensor_out)
  );
  m.impl("bitwise_right_shift.Tensor_Scalar_out",
         TORCH_FN(ADInplaceOrView::bitwise_right_shift_out_Tensor_Scalar_out)
  );
  m.impl("bitwise_right_shift.Scalar_Tensor_out",
         TORCH_FN(ADInplaceOrView::bitwise_right_shift_out_Scalar_Tensor_out)
  );
  m.impl("bitwise_xor_.Scalar",
         TORCH_FN(ADInplaceOrView::bitwise_xor__Scalar)
  );
  m.impl("bitwise_xor_.Tensor",
         TORCH_FN(ADInplaceOrView::bitwise_xor__Tensor)
  );
  m.impl("bitwise_xor.Tensor_out",
         TORCH_FN(ADInplaceOrView::bitwise_xor_out_Tensor_out)
  );
  m.impl("bitwise_xor.Scalar_out",
         TORCH_FN(ADInplaceOrView::bitwise_xor_out_Scalar_out)
  );
  m.impl("bitwise_xor.Scalar_Tensor_out",
         TORCH_FN(ADInplaceOrView::bitwise_xor_out_Scalar_Tensor_out)
  );
  m.impl("blackman_window.out",
         TORCH_FN(ADInplaceOrView::blackman_window_out_out)
  );
  m.impl("blackman_window.periodic_out",
         TORCH_FN(ADInplaceOrView::blackman_window_out_periodic_out)
  );
  m.impl("block_diag.out",
         TORCH_FN(ADInplaceOrView::block_diag_out_out)
  );
  m.impl("bmm.out",
         TORCH_FN(ADInplaceOrView::bmm_out_out)
  );
  m.impl("bucketize.Tensor_out",
         TORCH_FN(ADInplaceOrView::bucketize_out_Tensor_out)
  );
  m.impl("bucketize.Scalar_out",
         TORCH_FN(ADInplaceOrView::bucketize_out_Scalar_out)
  );
  m.impl("cat.out",
         TORCH_FN(ADInplaceOrView::cat_out_out)
  );
  m.impl("cauchy_",
         TORCH_FN(ADInplaceOrView::cauchy_)
  );
  m.impl("cauchy.out",
         TORCH_FN(ADInplaceOrView::cauchy_out_out)
  );
  m.impl("ccol_indices",
         TORCH_FN(ADInplaceOrView::ccol_indices)
  );
  m.impl("ccol_indices_copy.out",
         TORCH_FN(ADInplaceOrView::ccol_indices_copy_out_out)
  );
  m.impl("ceil_",
         TORCH_FN(ADInplaceOrView::ceil_)
  );
  m.impl("ceil.out",
         TORCH_FN(ADInplaceOrView::ceil_out_out)
  );
  m.impl("celu_",
         TORCH_FN(ADInplaceOrView::celu_)
  );
  m.impl("celu.out",
         TORCH_FN(ADInplaceOrView::celu_out_out)
  );
  m.impl("channel_shuffle.out",
         TORCH_FN(ADInplaceOrView::channel_shuffle_out_out)
  );
  m.impl("cholesky_inverse.out",
         TORCH_FN(ADInplaceOrView::cholesky_inverse_out_out)
  );
  m.impl("cholesky.out",
         TORCH_FN(ADInplaceOrView::cholesky_out_out)
  );
  m.impl("cholesky_solve.out",
         TORCH_FN(ADInplaceOrView::cholesky_solve_out_out)
  );
  m.impl("chunk",
         TORCH_FN(ADInplaceOrView::chunk)
  );
  m.impl("clamp_",
         TORCH_FN(ADInplaceOrView::clamp_)
  );
  m.impl("clamp_.Tensor",
         TORCH_FN(ADInplaceOrView::clamp__Tensor)
  );
  m.impl("clamp_max_",
         TORCH_FN(ADInplaceOrView::clamp_max_)
  );
  m.impl("clamp_max_.Tensor",
         TORCH_FN(ADInplaceOrView::clamp_max__Tensor)
  );
  m.impl("clamp_max.out",
         TORCH_FN(ADInplaceOrView::clamp_max_out_out)
  );
  m.impl("clamp_max.Tensor_out",
         TORCH_FN(ADInplaceOrView::clamp_max_out_Tensor_out)
  );
  m.impl("clamp_min_",
         TORCH_FN(ADInplaceOrView::clamp_min_)
  );
  m.impl("clamp_min_.Tensor",
         TORCH_FN(ADInplaceOrView::clamp_min__Tensor)
  );
  m.impl("clamp_min.out",
         TORCH_FN(ADInplaceOrView::clamp_min_out_out)
  );
  m.impl("clamp_min.Tensor_out",
         TORCH_FN(ADInplaceOrView::clamp_min_out_Tensor_out)
  );
  m.impl("clamp.out",
         TORCH_FN(ADInplaceOrView::clamp_out_out)
  );
  m.impl("clamp.Tensor_out",
         TORCH_FN(ADInplaceOrView::clamp_out_Tensor_out)
  );
  m.impl("clone.out",
         TORCH_FN(ADInplaceOrView::clone_out_out)
  );
  m.impl("col2im.out",
         TORCH_FN(ADInplaceOrView::col2im_out_out)
  );
  m.impl("col_indices",
         TORCH_FN(ADInplaceOrView::col_indices)
  );
  m.impl("col_indices_copy.out",
         TORCH_FN(ADInplaceOrView::col_indices_copy_out_out)
  );
  m.impl("complex.out",
         TORCH_FN(ADInplaceOrView::complex_out_out)
  );
  m.impl("conj_physical_",
         TORCH_FN(ADInplaceOrView::conj_physical_)
  );
  m.impl("conj_physical.out",
         TORCH_FN(ADInplaceOrView::conj_physical_out_out)
  );
  m.impl("constant_pad_nd.out",
         TORCH_FN(ADInplaceOrView::constant_pad_nd_out_out)
  );
  m.impl("conv_depthwise3d.out",
         TORCH_FN(ADInplaceOrView::conv_depthwise3d_out_out)
  );
  m.impl("conv_tbc.out",
         TORCH_FN(ADInplaceOrView::conv_tbc_out_out)
  );
  m.impl("convolution_backward.out",
         TORCH_FN(ADInplaceOrView::convolution_backward_out_out)
  );
  m.impl("convolution_backward_overrideable.out",
         TORCH_FN(ADInplaceOrView::convolution_backward_overrideable_out_out)
  );
  m.impl("convolution.out",
         TORCH_FN(ADInplaceOrView::convolution_out_out)
  );
  m.impl("convolution_overrideable.out",
         TORCH_FN(ADInplaceOrView::convolution_overrideable_out_out)
  );
  m.impl("copy.out",
         TORCH_FN(ADInplaceOrView::copy_out_out)
  );
  m.impl("copy_sparse_to_sparse_",
         TORCH_FN(ADInplaceOrView::copy_sparse_to_sparse_)
  );
  m.impl("copy_sparse_to_sparse.out",
         TORCH_FN(ADInplaceOrView::copy_sparse_to_sparse_out_out)
  );
  m.impl("copysign_.Tensor",
         TORCH_FN(ADInplaceOrView::copysign__Tensor)
  );
  m.impl("copysign_.Scalar",
         TORCH_FN(ADInplaceOrView::copysign__Scalar)
  );
  m.impl("copysign.out",
         TORCH_FN(ADInplaceOrView::copysign_out_out)
  );
  m.impl("copysign.Scalar_out",
         TORCH_FN(ADInplaceOrView::copysign_out_Scalar_out)
  );
  m.impl("cos_",
         TORCH_FN(ADInplaceOrView::cos_)
  );
  m.impl("cos.out",
         TORCH_FN(ADInplaceOrView::cos_out_out)
  );
  m.impl("cosh_",
         TORCH_FN(ADInplaceOrView::cosh_)
  );
  m.impl("cosh.out",
         TORCH_FN(ADInplaceOrView::cosh_out_out)
  );
  m.impl("count_nonzero.dim_IntList_out",
         TORCH_FN(ADInplaceOrView::count_nonzero_out_dim_IntList_out)
  );
  m.impl("count_nonzero.out",
         TORCH_FN(ADInplaceOrView::count_nonzero_out_out)
  );
  m.impl("crow_indices",
         TORCH_FN(ADInplaceOrView::crow_indices)
  );
  m.impl("crow_indices_copy.out",
         TORCH_FN(ADInplaceOrView::crow_indices_copy_out_out)
  );
  m.impl("cudnn_affine_grid_generator_backward.out",
         TORCH_FN(ADInplaceOrView::cudnn_affine_grid_generator_backward_out_out)
  );
  m.impl("cudnn_affine_grid_generator.out",
         TORCH_FN(ADInplaceOrView::cudnn_affine_grid_generator_out_out)
  );
  m.impl("cudnn_batch_norm_backward.out",
         TORCH_FN(ADInplaceOrView::cudnn_batch_norm_backward_out_out)
  );
  m.impl("cudnn_batch_norm.out",
         TORCH_FN(ADInplaceOrView::cudnn_batch_norm_out_out)
  );
  m.impl("cudnn_convolution_add_relu.out",
         TORCH_FN(ADInplaceOrView::cudnn_convolution_add_relu_out_out)
  );
  m.impl("cudnn_convolution.out",
         TORCH_FN(ADInplaceOrView::cudnn_convolution_out_out)
  );
  m.impl("cudnn_convolution_relu.out",
         TORCH_FN(ADInplaceOrView::cudnn_convolution_relu_out_out)
  );
  m.impl("cudnn_convolution_transpose.out",
         TORCH_FN(ADInplaceOrView::cudnn_convolution_transpose_out_out)
  );
  m.impl("cudnn_grid_sampler_backward.out",
         TORCH_FN(ADInplaceOrView::cudnn_grid_sampler_backward_out_out)
  );
  m.impl("cudnn_grid_sampler.out",
         TORCH_FN(ADInplaceOrView::cudnn_grid_sampler_out_out)
  );
  m.impl("cummax.out",
         TORCH_FN(ADInplaceOrView::cummax_out_out)
  );
  m.impl("cummin.out",
         TORCH_FN(ADInplaceOrView::cummin_out_out)
  );
  m.impl("cumprod_",
         TORCH_FN(ADInplaceOrView::cumprod_)
  );
  m.impl("cumprod.out",
         TORCH_FN(ADInplaceOrView::cumprod_out_out)
  );
  m.impl("cumsum_",
         TORCH_FN(ADInplaceOrView::cumsum_)
  );
  m.impl("cumsum.out",
         TORCH_FN(ADInplaceOrView::cumsum_out_out)
  );
  m.impl("deg2rad_",
         TORCH_FN(ADInplaceOrView::deg2rad_)
  );
  m.impl("deg2rad.out",
         TORCH_FN(ADInplaceOrView::deg2rad_out_out)
  );
  m.impl("dequantize.self_out",
         TORCH_FN(ADInplaceOrView::dequantize_out_self_out)
  );
  m.impl("detach_copy.out",
         TORCH_FN(ADInplaceOrView::detach_copy_out_out)
  );
  m.impl("diag_embed.out",
         TORCH_FN(ADInplaceOrView::diag_embed_out_out)
  );
  m.impl("diagonal",
         TORCH_FN(ADInplaceOrView::diagonal)
  );
  m.impl("diagonal_backward.out",
         TORCH_FN(ADInplaceOrView::diagonal_backward_out_out)
  );
  m.impl("diagonal_copy.out",
         TORCH_FN(ADInplaceOrView::diagonal_copy_out_out)
  );
  m.impl("diagonal_scatter.out",
         TORCH_FN(ADInplaceOrView::diagonal_scatter_out_out)
  );
  m.impl("digamma_",
         TORCH_FN(ADInplaceOrView::digamma_)
  );
  m.impl("digamma.out",
         TORCH_FN(ADInplaceOrView::digamma_out_out)
  );
  m.impl("dist.out",
         TORCH_FN(ADInplaceOrView::dist_out_out)
  );
  m.impl("div_.Tensor",
         TORCH_FN(ADInplaceOrView::div__Tensor)
  );
  m.impl("div_.Tensor_mode",
         TORCH_FN(ADInplaceOrView::div__Tensor_mode)
  );
  m.impl("div_.Scalar",
         TORCH_FN(ADInplaceOrView::div__Scalar)
  );
  m.impl("div_.Scalar_mode",
         TORCH_FN(ADInplaceOrView::div__Scalar_mode)
  );
  m.impl("div.out",
         TORCH_FN(ADInplaceOrView::div_out_out)
  );
  m.impl("div.out_mode",
         TORCH_FN(ADInplaceOrView::div_out_out_mode)
  );
  m.impl("div.Scalar_out",
         TORCH_FN(ADInplaceOrView::div_out_Scalar_out)
  );
  m.impl("div.Scalar_mode_out",
         TORCH_FN(ADInplaceOrView::div_out_Scalar_mode_out)
  );
  m.impl("dot.out",
         TORCH_FN(ADInplaceOrView::dot_out_out)
  );
  m.impl("elu_",
         TORCH_FN(ADInplaceOrView::elu_)
  );
  m.impl("elu_backward.grad_input",
         TORCH_FN(ADInplaceOrView::elu_backward_out_grad_input)
  );
  m.impl("elu.out",
         TORCH_FN(ADInplaceOrView::elu_out_out)
  );
  m.impl("embedding_dense_backward.out",
         TORCH_FN(ADInplaceOrView::embedding_dense_backward_out_out)
  );
  m.impl("embedding.out",
         TORCH_FN(ADInplaceOrView::embedding_out_out)
  );
  m.impl("embedding_renorm_",
         TORCH_FN(ADInplaceOrView::embedding_renorm_)
  );
  m.impl("embedding_renorm.out",
         TORCH_FN(ADInplaceOrView::embedding_renorm_out_out)
  );
  m.impl("empty_like.out",
         TORCH_FN(ADInplaceOrView::empty_like_out_out)
  );
  m.impl("empty.names_out",
         TORCH_FN(ADInplaceOrView::empty_out_names_out)
  );
  m.impl("empty_permuted.out",
         TORCH_FN(ADInplaceOrView::empty_permuted_out_out)
  );
  m.impl("empty_quantized.out",
         TORCH_FN(ADInplaceOrView::empty_quantized_out_out)
  );
  m.impl("empty_strided.out",
         TORCH_FN(ADInplaceOrView::empty_strided_out_out)
  );
  m.impl("eq_.Scalar",
         TORCH_FN(ADInplaceOrView::eq__Scalar)
  );
  m.impl("eq_.Tensor",
         TORCH_FN(ADInplaceOrView::eq__Tensor)
  );
  m.impl("eq.Scalar_out",
         TORCH_FN(ADInplaceOrView::eq_out_Scalar_out)
  );
  m.impl("eq.Tensor_out",
         TORCH_FN(ADInplaceOrView::eq_out_Tensor_out)
  );
  m.impl("erf_",
         TORCH_FN(ADInplaceOrView::erf_)
  );
  m.impl("erf.out",
         TORCH_FN(ADInplaceOrView::erf_out_out)
  );
  m.impl("erfc_",
         TORCH_FN(ADInplaceOrView::erfc_)
  );
  m.impl("erfc.out",
         TORCH_FN(ADInplaceOrView::erfc_out_out)
  );
  m.impl("erfinv_",
         TORCH_FN(ADInplaceOrView::erfinv_)
  );
  m.impl("erfinv.out",
         TORCH_FN(ADInplaceOrView::erfinv_out_out)
  );
  m.impl("exp2_",
         TORCH_FN(ADInplaceOrView::exp2_)
  );
  m.impl("exp2.out",
         TORCH_FN(ADInplaceOrView::exp2_out_out)
  );
  m.impl("exp_",
         TORCH_FN(ADInplaceOrView::exp_)
  );
  m.impl("exp.out",
         TORCH_FN(ADInplaceOrView::exp_out_out)
  );
  m.impl("expand",
         TORCH_FN(ADInplaceOrView::expand)
  );
  m.impl("expand_copy.out",
         TORCH_FN(ADInplaceOrView::expand_copy_out_out)
  );
  m.impl("expm1_",
         TORCH_FN(ADInplaceOrView::expm1_)
  );
  m.impl("expm1.out",
         TORCH_FN(ADInplaceOrView::expm1_out_out)
  );
  m.impl("exponential_",
         TORCH_FN(ADInplaceOrView::exponential_)
  );
  m.impl("exponential.out",
         TORCH_FN(ADInplaceOrView::exponential_out_out)
  );
  m.impl("eye.out",
         TORCH_FN(ADInplaceOrView::eye_out_out)
  );
  m.impl("eye.m_out",
         TORCH_FN(ADInplaceOrView::eye_out_m_out)
  );
  m.impl("fake_quantize_per_channel_affine_cachemask.out",
         TORCH_FN(ADInplaceOrView::fake_quantize_per_channel_affine_cachemask_out_out)
  );
  m.impl("fake_quantize_per_tensor_affine_cachemask.out",
         TORCH_FN(ADInplaceOrView::fake_quantize_per_tensor_affine_cachemask_out_out)
  );
  m.impl("fft_fftfreq.out",
         TORCH_FN(ADInplaceOrView::fft_fftfreq_out_out)
  );
  m.impl("fft_rfftfreq.out",
         TORCH_FN(ADInplaceOrView::fft_rfftfreq_out_out)
  );
  m.impl("fill_.Scalar",
         TORCH_FN(ADInplaceOrView::fill__Scalar)
  );
  m.impl("fill_.Tensor",
         TORCH_FN(ADInplaceOrView::fill__Tensor)
  );
  m.impl("fill.Scalar_out",
         TORCH_FN(ADInplaceOrView::fill_out_Scalar_out)
  );
  m.impl("fill.Tensor_out",
         TORCH_FN(ADInplaceOrView::fill_out_Tensor_out)
  );
  m.impl("flip.out",
         TORCH_FN(ADInplaceOrView::flip_out_out)
  );
  m.impl("floor_",
         TORCH_FN(ADInplaceOrView::floor_)
  );
  m.impl("floor_divide_.Tensor",
         TORCH_FN(ADInplaceOrView::floor_divide__Tensor)
  );
  m.impl("floor_divide_.Scalar",
         TORCH_FN(ADInplaceOrView::floor_divide__Scalar)
  );
  m.impl("floor_divide.out",
         TORCH_FN(ADInplaceOrView::floor_divide_out_out)
  );
  m.impl("floor_divide.Scalar_out",
         TORCH_FN(ADInplaceOrView::floor_divide_out_Scalar_out)
  );
  m.impl("floor.out",
         TORCH_FN(ADInplaceOrView::floor_out_out)
  );
  m.impl("fmax.out",
         TORCH_FN(ADInplaceOrView::fmax_out_out)
  );
  m.impl("fmin.out",
         TORCH_FN(ADInplaceOrView::fmin_out_out)
  );
  m.impl("fmod_.Scalar",
         TORCH_FN(ADInplaceOrView::fmod__Scalar)
  );
  m.impl("fmod_.Tensor",
         TORCH_FN(ADInplaceOrView::fmod__Tensor)
  );
  m.impl("fmod.Scalar_out",
         TORCH_FN(ADInplaceOrView::fmod_out_Scalar_out)
  );
  m.impl("fmod.Tensor_out",
         TORCH_FN(ADInplaceOrView::fmod_out_Tensor_out)
  );
  m.impl("frac_",
         TORCH_FN(ADInplaceOrView::frac_)
  );
  m.impl("frac.out",
         TORCH_FN(ADInplaceOrView::frac_out_out)
  );
  m.impl("fractional_max_pool2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::fractional_max_pool2d_backward_out_grad_input)
  );
  m.impl("fractional_max_pool2d.output",
         TORCH_FN(ADInplaceOrView::fractional_max_pool2d_out_output)
  );
  m.impl("fractional_max_pool3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::fractional_max_pool3d_backward_out_grad_input)
  );
  m.impl("fractional_max_pool3d.output",
         TORCH_FN(ADInplaceOrView::fractional_max_pool3d_out_output)
  );
  m.impl("frexp.Tensor_out",
         TORCH_FN(ADInplaceOrView::frexp_out_Tensor_out)
  );
  m.impl("from_file.out",
         TORCH_FN(ADInplaceOrView::from_file_out_out)
  );
  m.impl("full_like.out",
         TORCH_FN(ADInplaceOrView::full_like_out_out)
  );
  m.impl("full.out",
         TORCH_FN(ADInplaceOrView::full_out_out)
  );
  m.impl("full.names_out",
         TORCH_FN(ADInplaceOrView::full_out_names_out)
  );
  m.impl("gather.out",
         TORCH_FN(ADInplaceOrView::gather_out_out)
  );
  m.impl("gcd_",
         TORCH_FN(ADInplaceOrView::gcd_)
  );
  m.impl("gcd.out",
         TORCH_FN(ADInplaceOrView::gcd_out_out)
  );
  m.impl("ge_.Scalar",
         TORCH_FN(ADInplaceOrView::ge__Scalar)
  );
  m.impl("ge_.Tensor",
         TORCH_FN(ADInplaceOrView::ge__Tensor)
  );
  m.impl("ge.Scalar_out",
         TORCH_FN(ADInplaceOrView::ge_out_Scalar_out)
  );
  m.impl("ge.Tensor_out",
         TORCH_FN(ADInplaceOrView::ge_out_Tensor_out)
  );
  m.impl("gelu_",
         TORCH_FN(ADInplaceOrView::gelu_)
  );
  m.impl("gelu_backward.grad_input",
         TORCH_FN(ADInplaceOrView::gelu_backward_out_grad_input)
  );
  m.impl("gelu.out",
         TORCH_FN(ADInplaceOrView::gelu_out_out)
  );
  m.impl("geometric_",
         TORCH_FN(ADInplaceOrView::geometric_)
  );
  m.impl("geometric.out",
         TORCH_FN(ADInplaceOrView::geometric_out_out)
  );
  m.impl("geqrf.a",
         TORCH_FN(ADInplaceOrView::geqrf_out_a)
  );
  m.impl("glu_backward_jvp.out",
         TORCH_FN(ADInplaceOrView::glu_backward_jvp_out_out)
  );
  m.impl("glu_backward.grad_input",
         TORCH_FN(ADInplaceOrView::glu_backward_out_grad_input)
  );
  m.impl("glu_jvp.out",
         TORCH_FN(ADInplaceOrView::glu_jvp_out_out)
  );
  m.impl("glu.out",
         TORCH_FN(ADInplaceOrView::glu_out_out)
  );
  m.impl("grid_sampler_2d_backward.out",
         TORCH_FN(ADInplaceOrView::grid_sampler_2d_backward_out_out)
  );
  m.impl("grid_sampler_2d.out",
         TORCH_FN(ADInplaceOrView::grid_sampler_2d_out_out)
  );
  m.impl("grid_sampler_3d_backward.out",
         TORCH_FN(ADInplaceOrView::grid_sampler_3d_backward_out_out)
  );
  m.impl("grid_sampler_3d.out",
         TORCH_FN(ADInplaceOrView::grid_sampler_3d_out_out)
  );
  m.impl("gt_.Scalar",
         TORCH_FN(ADInplaceOrView::gt__Scalar)
  );
  m.impl("gt_.Tensor",
         TORCH_FN(ADInplaceOrView::gt__Tensor)
  );
  m.impl("gt.Scalar_out",
         TORCH_FN(ADInplaceOrView::gt_out_Scalar_out)
  );
  m.impl("gt.Tensor_out",
         TORCH_FN(ADInplaceOrView::gt_out_Tensor_out)
  );
  m.impl("hamming_window.out",
         TORCH_FN(ADInplaceOrView::hamming_window_out_out)
  );
  m.impl("hamming_window.periodic_out",
         TORCH_FN(ADInplaceOrView::hamming_window_out_periodic_out)
  );
  m.impl("hamming_window.periodic_alpha_out",
         TORCH_FN(ADInplaceOrView::hamming_window_out_periodic_alpha_out)
  );
  m.impl("hamming_window.periodic_alpha_beta_out",
         TORCH_FN(ADInplaceOrView::hamming_window_out_periodic_alpha_beta_out)
  );
  m.impl("hann_window.out",
         TORCH_FN(ADInplaceOrView::hann_window_out_out)
  );
  m.impl("hann_window.periodic_out",
         TORCH_FN(ADInplaceOrView::hann_window_out_periodic_out)
  );
  m.impl("hardshrink_backward.grad_input",
         TORCH_FN(ADInplaceOrView::hardshrink_backward_out_grad_input)
  );
  m.impl("hardshrink.out",
         TORCH_FN(ADInplaceOrView::hardshrink_out_out)
  );
  m.impl("hardsigmoid_",
         TORCH_FN(ADInplaceOrView::hardsigmoid_)
  );
  m.impl("hardsigmoid_backward.grad_input",
         TORCH_FN(ADInplaceOrView::hardsigmoid_backward_out_grad_input)
  );
  m.impl("hardsigmoid.out",
         TORCH_FN(ADInplaceOrView::hardsigmoid_out_out)
  );
  m.impl("hardswish_",
         TORCH_FN(ADInplaceOrView::hardswish_)
  );
  m.impl("hardswish_backward.out",
         TORCH_FN(ADInplaceOrView::hardswish_backward_out_out)
  );
  m.impl("hardswish.out",
         TORCH_FN(ADInplaceOrView::hardswish_out_out)
  );
  m.impl("hardtanh_",
         TORCH_FN(ADInplaceOrView::hardtanh_)
  );
  m.impl("hardtanh_backward.grad_input",
         TORCH_FN(ADInplaceOrView::hardtanh_backward_out_grad_input)
  );
  m.impl("hardtanh.out",
         TORCH_FN(ADInplaceOrView::hardtanh_out_out)
  );
  m.impl("heaviside_",
         TORCH_FN(ADInplaceOrView::heaviside_)
  );
  m.impl("heaviside.out",
         TORCH_FN(ADInplaceOrView::heaviside_out_out)
  );
  m.impl("histc.out",
         TORCH_FN(ADInplaceOrView::histc_out_out)
  );
  m.impl("histogram.bins_tensor_out",
         TORCH_FN(ADInplaceOrView::histogram_out_bins_tensor_out)
  );
  m.impl("histogram.bin_ct_out",
         TORCH_FN(ADInplaceOrView::histogram_out_bin_ct_out)
  );
  m.impl("hspmm.out",
         TORCH_FN(ADInplaceOrView::hspmm_out_out)
  );
  m.impl("huber_loss_backward.out",
         TORCH_FN(ADInplaceOrView::huber_loss_backward_out_out)
  );
  m.impl("huber_loss.out",
         TORCH_FN(ADInplaceOrView::huber_loss_out_out)
  );
  m.impl("hypot_",
         TORCH_FN(ADInplaceOrView::hypot_)
  );
  m.impl("hypot.out",
         TORCH_FN(ADInplaceOrView::hypot_out_out)
  );
  m.impl("i0_",
         TORCH_FN(ADInplaceOrView::i0_)
  );
  m.impl("i0.out",
         TORCH_FN(ADInplaceOrView::i0_out_out)
  );
  m.impl("igamma_",
         TORCH_FN(ADInplaceOrView::igamma_)
  );
  m.impl("igamma.out",
         TORCH_FN(ADInplaceOrView::igamma_out_out)
  );
  m.impl("igammac_",
         TORCH_FN(ADInplaceOrView::igammac_)
  );
  m.impl("igammac.out",
         TORCH_FN(ADInplaceOrView::igammac_out_out)
  );
  m.impl("im2col.out",
         TORCH_FN(ADInplaceOrView::im2col_out_out)
  );
  m.impl("index_add_",
         TORCH_FN(ADInplaceOrView::index_add_)
  );
  m.impl("index_add.out",
         TORCH_FN(ADInplaceOrView::index_add_out_out)
  );
  m.impl("index_copy_",
         TORCH_FN(ADInplaceOrView::index_copy_)
  );
  m.impl("index_copy.out",
         TORCH_FN(ADInplaceOrView::index_copy_out_out)
  );
  m.impl("index_fill_.int_Scalar",
         TORCH_FN(ADInplaceOrView::index_fill__int_Scalar)
  );
  m.impl("index_fill_.int_Tensor",
         TORCH_FN(ADInplaceOrView::index_fill__int_Tensor)
  );
  m.impl("index_fill.int_Scalar_out",
         TORCH_FN(ADInplaceOrView::index_fill_out_int_Scalar_out)
  );
  m.impl("index_fill.int_Tensor_out",
         TORCH_FN(ADInplaceOrView::index_fill_out_int_Tensor_out)
  );
  m.impl("index.Tensor_out",
         TORCH_FN(ADInplaceOrView::index_out_Tensor_out)
  );
  m.impl("index_put_",
         TORCH_FN(ADInplaceOrView::index_put_)
  );
  m.impl("index_put.out",
         TORCH_FN(ADInplaceOrView::index_put_out_out)
  );
  m.impl("index_reduce_",
         TORCH_FN(ADInplaceOrView::index_reduce_)
  );
  m.impl("index_reduce.out",
         TORCH_FN(ADInplaceOrView::index_reduce_out_out)
  );
  m.impl("index_select.out",
         TORCH_FN(ADInplaceOrView::index_select_out_out)
  );
  m.impl("indices",
         TORCH_FN(ADInplaceOrView::indices)
  );
  m.impl("indices_copy.out",
         TORCH_FN(ADInplaceOrView::indices_copy_out_out)
  );
  m.impl("int_repr.out",
         TORCH_FN(ADInplaceOrView::int_repr_out_out)
  );
  m.impl("isin.Tensor_Tensor_out",
         TORCH_FN(ADInplaceOrView::isin_out_Tensor_Tensor_out)
  );
  m.impl("isin.Tensor_Scalar_out",
         TORCH_FN(ADInplaceOrView::isin_out_Tensor_Scalar_out)
  );
  m.impl("isin.Scalar_Tensor_out",
         TORCH_FN(ADInplaceOrView::isin_out_Scalar_Tensor_out)
  );
  m.impl("isinf.out",
         TORCH_FN(ADInplaceOrView::isinf_out_out)
  );
  m.impl("isnan.out",
         TORCH_FN(ADInplaceOrView::isnan_out_out)
  );
  m.impl("isneginf.out",
         TORCH_FN(ADInplaceOrView::isneginf_out_out)
  );
  m.impl("isposinf.out",
         TORCH_FN(ADInplaceOrView::isposinf_out_out)
  );
  m.impl("kaiser_window.out",
         TORCH_FN(ADInplaceOrView::kaiser_window_out_out)
  );
  m.impl("kaiser_window.periodic_out",
         TORCH_FN(ADInplaceOrView::kaiser_window_out_periodic_out)
  );
  m.impl("kaiser_window.beta_out",
         TORCH_FN(ADInplaceOrView::kaiser_window_out_beta_out)
  );
  m.impl("kthvalue.values",
         TORCH_FN(ADInplaceOrView::kthvalue_out_values)
  );
  m.impl("lcm_",
         TORCH_FN(ADInplaceOrView::lcm_)
  );
  m.impl("lcm.out",
         TORCH_FN(ADInplaceOrView::lcm_out_out)
  );
  m.impl("le_.Scalar",
         TORCH_FN(ADInplaceOrView::le__Scalar)
  );
  m.impl("le_.Tensor",
         TORCH_FN(ADInplaceOrView::le__Tensor)
  );
  m.impl("le.Scalar_out",
         TORCH_FN(ADInplaceOrView::le_out_Scalar_out)
  );
  m.impl("le.Tensor_out",
         TORCH_FN(ADInplaceOrView::le_out_Tensor_out)
  );
  m.impl("leaky_relu_",
         TORCH_FN(ADInplaceOrView::leaky_relu_)
  );
  m.impl("leaky_relu_backward.grad_input",
         TORCH_FN(ADInplaceOrView::leaky_relu_backward_out_grad_input)
  );
  m.impl("leaky_relu.out",
         TORCH_FN(ADInplaceOrView::leaky_relu_out_out)
  );
  m.impl("lerp_.Scalar",
         TORCH_FN(ADInplaceOrView::lerp__Scalar)
  );
  m.impl("lerp_.Tensor",
         TORCH_FN(ADInplaceOrView::lerp__Tensor)
  );
  m.impl("lerp.Scalar_out",
         TORCH_FN(ADInplaceOrView::lerp_out_Scalar_out)
  );
  m.impl("lerp.Tensor_out",
         TORCH_FN(ADInplaceOrView::lerp_out_Tensor_out)
  );
  m.impl("lgamma_",
         TORCH_FN(ADInplaceOrView::lgamma_)
  );
  m.impl("lgamma.out",
         TORCH_FN(ADInplaceOrView::lgamma_out_out)
  );
  m.impl("lift_fresh_copy.out",
         TORCH_FN(ADInplaceOrView::lift_fresh_copy_out_out)
  );
  m.impl("lift.out",
         TORCH_FN(ADInplaceOrView::lift_out_out)
  );
  m.impl("linalg_cholesky_ex.L",
         TORCH_FN(ADInplaceOrView::linalg_cholesky_ex_out_L)
  );
  m.impl("linalg_cross.out",
         TORCH_FN(ADInplaceOrView::linalg_cross_out_out)
  );
  m.impl("linalg_eig.out",
         TORCH_FN(ADInplaceOrView::linalg_eig_out_out)
  );
  m.impl("linalg_eigvals.out",
         TORCH_FN(ADInplaceOrView::linalg_eigvals_out_out)
  );
  m.impl("linalg_householder_product.out",
         TORCH_FN(ADInplaceOrView::linalg_householder_product_out_out)
  );
  m.impl("linalg_inv_ex.inverse",
         TORCH_FN(ADInplaceOrView::linalg_inv_ex_out_inverse)
  );
  m.impl("linalg_ldl_factor_ex.out",
         TORCH_FN(ADInplaceOrView::linalg_ldl_factor_ex_out_out)
  );
  m.impl("linalg_ldl_solve.out",
         TORCH_FN(ADInplaceOrView::linalg_ldl_solve_out_out)
  );
  m.impl("linalg_lstsq.out",
         TORCH_FN(ADInplaceOrView::linalg_lstsq_out_out)
  );
  m.impl("linalg_lu_factor_ex.out",
         TORCH_FN(ADInplaceOrView::linalg_lu_factor_ex_out_out)
  );
  m.impl("linalg_lu_factor.out",
         TORCH_FN(ADInplaceOrView::linalg_lu_factor_out_out)
  );
  m.impl("linalg_lu.out",
         TORCH_FN(ADInplaceOrView::linalg_lu_out_out)
  );
  m.impl("linalg_lu_solve.out",
         TORCH_FN(ADInplaceOrView::linalg_lu_solve_out_out)
  );
  m.impl("linalg_matrix_exp.out",
         TORCH_FN(ADInplaceOrView::linalg_matrix_exp_out_out)
  );
  m.impl("linalg_pinv.atol_rtol_tensor_out",
         TORCH_FN(ADInplaceOrView::linalg_pinv_out_atol_rtol_tensor_out)
  );
  m.impl("linalg_qr.out",
         TORCH_FN(ADInplaceOrView::linalg_qr_out_out)
  );
  m.impl("linalg_solve_triangular.out",
         TORCH_FN(ADInplaceOrView::linalg_solve_triangular_out_out)
  );
  m.impl("linalg_vector_norm.out",
         TORCH_FN(ADInplaceOrView::linalg_vector_norm_out_out)
  );
  m.impl("linear_backward.out",
         TORCH_FN(ADInplaceOrView::linear_backward_out_out)
  );
  m.impl("linear.out",
         TORCH_FN(ADInplaceOrView::linear_out_out)
  );
  m.impl("linspace.out",
         TORCH_FN(ADInplaceOrView::linspace_out_out)
  );
  m.impl("linspace.Tensor_Tensor_out",
         TORCH_FN(ADInplaceOrView::linspace_out_Tensor_Tensor_out)
  );
  m.impl("linspace.Tensor_Scalar_out",
         TORCH_FN(ADInplaceOrView::linspace_out_Tensor_Scalar_out)
  );
  m.impl("linspace.Scalar_Tensor_out",
         TORCH_FN(ADInplaceOrView::linspace_out_Scalar_Tensor_out)
  );
  m.impl("log10_",
         TORCH_FN(ADInplaceOrView::log10_)
  );
  m.impl("log10.out",
         TORCH_FN(ADInplaceOrView::log10_out_out)
  );
  m.impl("log1p_",
         TORCH_FN(ADInplaceOrView::log1p_)
  );
  m.impl("log1p.out",
         TORCH_FN(ADInplaceOrView::log1p_out_out)
  );
  m.impl("log2_",
         TORCH_FN(ADInplaceOrView::log2_)
  );
  m.impl("log2.out",
         TORCH_FN(ADInplaceOrView::log2_out_out)
  );
  m.impl("log_",
         TORCH_FN(ADInplaceOrView::log_)
  );
  m.impl("log_normal_",
         TORCH_FN(ADInplaceOrView::log_normal_)
  );
  m.impl("log_normal.out",
         TORCH_FN(ADInplaceOrView::log_normal_out_out)
  );
  m.impl("log.out",
         TORCH_FN(ADInplaceOrView::log_out_out)
  );
  m.impl("log_sigmoid_backward.grad_input",
         TORCH_FN(ADInplaceOrView::log_sigmoid_backward_out_grad_input)
  );
  m.impl("log_sigmoid_forward.output",
         TORCH_FN(ADInplaceOrView::log_sigmoid_forward_out_output)
  );
  m.impl("log_softmax.int_out",
         TORCH_FN(ADInplaceOrView::log_softmax_out_int_out)
  );
  m.impl("logaddexp2.out",
         TORCH_FN(ADInplaceOrView::logaddexp2_out_out)
  );
  m.impl("logaddexp.out",
         TORCH_FN(ADInplaceOrView::logaddexp_out_out)
  );
  m.impl("logcumsumexp.out",
         TORCH_FN(ADInplaceOrView::logcumsumexp_out_out)
  );
  m.impl("logical_and_",
         TORCH_FN(ADInplaceOrView::logical_and_)
  );
  m.impl("logical_and.out",
         TORCH_FN(ADInplaceOrView::logical_and_out_out)
  );
  m.impl("logical_not_",
         TORCH_FN(ADInplaceOrView::logical_not_)
  );
  m.impl("logical_not.out",
         TORCH_FN(ADInplaceOrView::logical_not_out_out)
  );
  m.impl("logical_or_",
         TORCH_FN(ADInplaceOrView::logical_or_)
  );
  m.impl("logical_or.out",
         TORCH_FN(ADInplaceOrView::logical_or_out_out)
  );
  m.impl("logical_xor_",
         TORCH_FN(ADInplaceOrView::logical_xor_)
  );
  m.impl("logical_xor.out",
         TORCH_FN(ADInplaceOrView::logical_xor_out_out)
  );
  m.impl("logit_",
         TORCH_FN(ADInplaceOrView::logit_)
  );
  m.impl("logit_backward.grad_input",
         TORCH_FN(ADInplaceOrView::logit_backward_out_grad_input)
  );
  m.impl("logit.out",
         TORCH_FN(ADInplaceOrView::logit_out_out)
  );
  m.impl("logspace.out",
         TORCH_FN(ADInplaceOrView::logspace_out_out)
  );
  m.impl("logspace.Tensor_Tensor_out",
         TORCH_FN(ADInplaceOrView::logspace_out_Tensor_Tensor_out)
  );
  m.impl("logspace.Tensor_Scalar_out",
         TORCH_FN(ADInplaceOrView::logspace_out_Tensor_Scalar_out)
  );
  m.impl("logspace.Scalar_Tensor_out",
         TORCH_FN(ADInplaceOrView::logspace_out_Scalar_Tensor_out)
  );
  m.impl("logsumexp.out",
         TORCH_FN(ADInplaceOrView::logsumexp_out_out)
  );
  m.impl("lt_.Scalar",
         TORCH_FN(ADInplaceOrView::lt__Scalar)
  );
  m.impl("lt_.Tensor",
         TORCH_FN(ADInplaceOrView::lt__Tensor)
  );
  m.impl("lt.Scalar_out",
         TORCH_FN(ADInplaceOrView::lt_out_Scalar_out)
  );
  m.impl("lt.Tensor_out",
         TORCH_FN(ADInplaceOrView::lt_out_Tensor_out)
  );
  m.impl("lu_unpack.out",
         TORCH_FN(ADInplaceOrView::lu_unpack_out_out)
  );
  m.impl("masked_fill_.Scalar",
         TORCH_FN(ADInplaceOrView::masked_fill__Scalar)
  );
  m.impl("masked_fill_.Tensor",
         TORCH_FN(ADInplaceOrView::masked_fill__Tensor)
  );
  m.impl("masked_fill.Scalar_out",
         TORCH_FN(ADInplaceOrView::masked_fill_out_Scalar_out)
  );
  m.impl("masked_fill.Tensor_out",
         TORCH_FN(ADInplaceOrView::masked_fill_out_Tensor_out)
  );
  m.impl("masked_scatter_",
         TORCH_FN(ADInplaceOrView::masked_scatter_)
  );
  m.impl("masked_scatter.out",
         TORCH_FN(ADInplaceOrView::masked_scatter_out_out)
  );
  m.impl("masked_select.out",
         TORCH_FN(ADInplaceOrView::masked_select_out_out)
  );
  m.impl("matmul_backward.out",
         TORCH_FN(ADInplaceOrView::matmul_backward_out_out)
  );
  m.impl("matmul.out",
         TORCH_FN(ADInplaceOrView::matmul_out_out)
  );
  m.impl("max.dim_max",
         TORCH_FN(ADInplaceOrView::max_out_dim_max)
  );
  m.impl("max.unary_out",
         TORCH_FN(ADInplaceOrView::max_out_unary_out)
  );
  m.impl("max_pool2d_backward.out",
         TORCH_FN(ADInplaceOrView::max_pool2d_backward_out_out)
  );
  m.impl("max_pool2d_with_indices_backward.grad_input",
         TORCH_FN(ADInplaceOrView::max_pool2d_with_indices_backward_out_grad_input)
  );
  m.impl("max_pool2d_with_indices.out",
         TORCH_FN(ADInplaceOrView::max_pool2d_with_indices_out_out)
  );
  m.impl("max_pool3d_with_indices_backward.grad_input",
         TORCH_FN(ADInplaceOrView::max_pool3d_with_indices_backward_out_grad_input)
  );
  m.impl("max_pool3d_with_indices.out",
         TORCH_FN(ADInplaceOrView::max_pool3d_with_indices_out_out)
  );
  m.impl("max_unpool2d.out",
         TORCH_FN(ADInplaceOrView::max_unpool2d_out_out)
  );
  m.impl("max_unpool3d.out",
         TORCH_FN(ADInplaceOrView::max_unpool3d_out_out)
  );
  m.impl("maximum.out",
         TORCH_FN(ADInplaceOrView::maximum_out_out)
  );
  m.impl("mean.dtype_out",
         TORCH_FN(ADInplaceOrView::mean_out_dtype_out)
  );
  m.impl("mean.out",
         TORCH_FN(ADInplaceOrView::mean_out_out)
  );
  m.impl("median.dim_values",
         TORCH_FN(ADInplaceOrView::median_out_dim_values)
  );
  m.impl("median.out",
         TORCH_FN(ADInplaceOrView::median_out_out)
  );
  m.impl("min.dim_min",
         TORCH_FN(ADInplaceOrView::min_out_dim_min)
  );
  m.impl("min.unary_out",
         TORCH_FN(ADInplaceOrView::min_out_unary_out)
  );
  m.impl("minimum.out",
         TORCH_FN(ADInplaceOrView::minimum_out_out)
  );
  m.impl("miopen_batch_norm_backward.out",
         TORCH_FN(ADInplaceOrView::miopen_batch_norm_backward_out_out)
  );
  m.impl("miopen_batch_norm.out",
         TORCH_FN(ADInplaceOrView::miopen_batch_norm_out_out)
  );
  m.impl("miopen_convolution.out",
         TORCH_FN(ADInplaceOrView::miopen_convolution_out_out)
  );
  m.impl("miopen_convolution_transpose.out",
         TORCH_FN(ADInplaceOrView::miopen_convolution_transpose_out_out)
  );
  m.impl("miopen_depthwise_convolution.out",
         TORCH_FN(ADInplaceOrView::miopen_depthwise_convolution_out_out)
  );
  m.impl("miopen_rnn.out",
         TORCH_FN(ADInplaceOrView::miopen_rnn_out_out)
  );
  m.impl("mish_",
         TORCH_FN(ADInplaceOrView::mish_)
  );
  m.impl("mish.out",
         TORCH_FN(ADInplaceOrView::mish_out_out)
  );
  m.impl("mkldnn_adaptive_avg_pool2d_backward.out",
         TORCH_FN(ADInplaceOrView::mkldnn_adaptive_avg_pool2d_backward_out_out)
  );
  m.impl("mkldnn_adaptive_avg_pool2d.out",
         TORCH_FN(ADInplaceOrView::mkldnn_adaptive_avg_pool2d_out_out)
  );
  m.impl("mkldnn_convolution.out",
         TORCH_FN(ADInplaceOrView::mkldnn_convolution_out_out)
  );
  m.impl("mkldnn_linear_backward_input.out",
         TORCH_FN(ADInplaceOrView::mkldnn_linear_backward_input_out_out)
  );
  m.impl("mkldnn_linear_backward.out",
         TORCH_FN(ADInplaceOrView::mkldnn_linear_backward_out_out)
  );
  m.impl("mkldnn_linear_backward_weights.out",
         TORCH_FN(ADInplaceOrView::mkldnn_linear_backward_weights_out_out)
  );
  m.impl("mkldnn_linear.out",
         TORCH_FN(ADInplaceOrView::mkldnn_linear_out_out)
  );
  m.impl("mkldnn_max_pool2d_backward.out",
         TORCH_FN(ADInplaceOrView::mkldnn_max_pool2d_backward_out_out)
  );
  m.impl("mkldnn_max_pool2d.out",
         TORCH_FN(ADInplaceOrView::mkldnn_max_pool2d_out_out)
  );
  m.impl("mkldnn_max_pool3d_backward.out",
         TORCH_FN(ADInplaceOrView::mkldnn_max_pool3d_backward_out_out)
  );
  m.impl("mkldnn_max_pool3d.out",
         TORCH_FN(ADInplaceOrView::mkldnn_max_pool3d_out_out)
  );
  m.impl("mkldnn_reorder_conv2d_weight.out",
         TORCH_FN(ADInplaceOrView::mkldnn_reorder_conv2d_weight_out_out)
  );
  m.impl("mkldnn_reorder_conv3d_weight.out",
         TORCH_FN(ADInplaceOrView::mkldnn_reorder_conv3d_weight_out_out)
  );
  m.impl("mkldnn_rnn_layer_backward.out",
         TORCH_FN(ADInplaceOrView::mkldnn_rnn_layer_backward_out_out)
  );
  m.impl("mkldnn_rnn_layer.out",
         TORCH_FN(ADInplaceOrView::mkldnn_rnn_layer_out_out)
  );
  m.impl("mm.out",
         TORCH_FN(ADInplaceOrView::mm_out_out)
  );
  m.impl("mode.values",
         TORCH_FN(ADInplaceOrView::mode_out_values)
  );
  m.impl("mps_convolution_backward.out",
         TORCH_FN(ADInplaceOrView::mps_convolution_backward_out_out)
  );
  m.impl("mps_convolution_transpose_backward.out",
         TORCH_FN(ADInplaceOrView::mps_convolution_transpose_backward_out_out)
  );
  m.impl("mse_loss_backward.grad_input",
         TORCH_FN(ADInplaceOrView::mse_loss_backward_out_grad_input)
  );
  m.impl("mse_loss.out",
         TORCH_FN(ADInplaceOrView::mse_loss_out_out)
  );
  m.impl("mul_.Tensor",
         TORCH_FN(ADInplaceOrView::mul__Tensor)
  );
  m.impl("mul_.Scalar",
         TORCH_FN(ADInplaceOrView::mul__Scalar)
  );
  m.impl("mul.out",
         TORCH_FN(ADInplaceOrView::mul_out_out)
  );
  m.impl("mul.Scalar_out",
         TORCH_FN(ADInplaceOrView::mul_out_Scalar_out)
  );
  m.impl("multi_margin_loss_backward.grad_input",
         TORCH_FN(ADInplaceOrView::multi_margin_loss_backward_out_grad_input)
  );
  m.impl("multi_margin_loss.out",
         TORCH_FN(ADInplaceOrView::multi_margin_loss_out_out)
  );
  m.impl("multilabel_margin_loss_backward.grad_input",
         TORCH_FN(ADInplaceOrView::multilabel_margin_loss_backward_out_grad_input)
  );
  m.impl("multilabel_margin_loss_forward.output",
         TORCH_FN(ADInplaceOrView::multilabel_margin_loss_forward_out_output)
  );
  m.impl("multinomial.out",
         TORCH_FN(ADInplaceOrView::multinomial_out_out)
  );
  m.impl("mv.out",
         TORCH_FN(ADInplaceOrView::mv_out_out)
  );
  m.impl("mvlgamma_",
         TORCH_FN(ADInplaceOrView::mvlgamma_)
  );
  m.impl("mvlgamma.out",
         TORCH_FN(ADInplaceOrView::mvlgamma_out_out)
  );
  m.impl("nan_to_num_",
         TORCH_FN(ADInplaceOrView::nan_to_num_)
  );
  m.impl("nan_to_num.out",
         TORCH_FN(ADInplaceOrView::nan_to_num_out_out)
  );
  m.impl("nanmedian.dim_values",
         TORCH_FN(ADInplaceOrView::nanmedian_out_dim_values)
  );
  m.impl("nanmedian.out",
         TORCH_FN(ADInplaceOrView::nanmedian_out_out)
  );
  m.impl("nansum.out",
         TORCH_FN(ADInplaceOrView::nansum_out_out)
  );
  m.impl("narrow",
         TORCH_FN(ADInplaceOrView::narrow)
  );
  m.impl("narrow_copy.out",
         TORCH_FN(ADInplaceOrView::narrow_copy_out_out)
  );
  m.impl("native_batch_norm_backward.out",
         TORCH_FN(ADInplaceOrView::native_batch_norm_backward_out_out)
  );
  m.impl("native_batch_norm.out",
         TORCH_FN(ADInplaceOrView::native_batch_norm_out_out)
  );
  m.impl("native_dropout_backward.out",
         TORCH_FN(ADInplaceOrView::native_dropout_backward_out_out)
  );
  m.impl("native_dropout.out",
         TORCH_FN(ADInplaceOrView::native_dropout_out_out)
  );
  m.impl("native_group_norm_backward.out",
         TORCH_FN(ADInplaceOrView::native_group_norm_backward_out_out)
  );
  m.impl("native_group_norm.out",
         TORCH_FN(ADInplaceOrView::native_group_norm_out_out)
  );
  m.impl("native_layer_norm_backward.out",
         TORCH_FN(ADInplaceOrView::native_layer_norm_backward_out_out)
  );
  m.impl("native_layer_norm.out",
         TORCH_FN(ADInplaceOrView::native_layer_norm_out_out)
  );
  m.impl("native_norm.out",
         TORCH_FN(ADInplaceOrView::native_norm_out_out)
  );
  m.impl("native_norm.ScalarOpt_dim_dtype_out",
         TORCH_FN(ADInplaceOrView::native_norm_out_ScalarOpt_dim_dtype_out)
  );
  m.impl("ne_.Scalar",
         TORCH_FN(ADInplaceOrView::ne__Scalar)
  );
  m.impl("ne_.Tensor",
         TORCH_FN(ADInplaceOrView::ne__Tensor)
  );
  m.impl("ne.Scalar_out",
         TORCH_FN(ADInplaceOrView::ne_out_Scalar_out)
  );
  m.impl("ne.Tensor_out",
         TORCH_FN(ADInplaceOrView::ne_out_Tensor_out)
  );
  m.impl("neg_",
         TORCH_FN(ADInplaceOrView::neg_)
  );
  m.impl("neg.out",
         TORCH_FN(ADInplaceOrView::neg_out_out)
  );
  m.impl("new_empty.out",
         TORCH_FN(ADInplaceOrView::new_empty_out_out)
  );
  m.impl("new_empty_strided.out",
         TORCH_FN(ADInplaceOrView::new_empty_strided_out_out)
  );
  m.impl("new_full.out",
         TORCH_FN(ADInplaceOrView::new_full_out_out)
  );
  m.impl("new_ones.out",
         TORCH_FN(ADInplaceOrView::new_ones_out_out)
  );
  m.impl("new_zeros.out",
         TORCH_FN(ADInplaceOrView::new_zeros_out_out)
  );
  m.impl("nextafter_",
         TORCH_FN(ADInplaceOrView::nextafter_)
  );
  m.impl("nextafter.out",
         TORCH_FN(ADInplaceOrView::nextafter_out_out)
  );
  m.impl("nll_loss2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::nll_loss2d_backward_out_grad_input)
  );
  m.impl("nll_loss2d_forward.output",
         TORCH_FN(ADInplaceOrView::nll_loss2d_forward_out_output)
  );
  m.impl("nll_loss_backward.grad_input",
         TORCH_FN(ADInplaceOrView::nll_loss_backward_out_grad_input)
  );
  m.impl("nll_loss_forward.output",
         TORCH_FN(ADInplaceOrView::nll_loss_forward_out_output)
  );
  m.impl("nonzero.out",
         TORCH_FN(ADInplaceOrView::nonzero_out_out)
  );
  m.impl("nonzero_static.out",
         TORCH_FN(ADInplaceOrView::nonzero_static_out_out)
  );
  m.impl("norm.dtype_out",
         TORCH_FN(ADInplaceOrView::norm_out_dtype_out)
  );
  m.impl("norm.out",
         TORCH_FN(ADInplaceOrView::norm_out_out)
  );
  m.impl("norm.ScalarOpt_dtype_out",
         TORCH_FN(ADInplaceOrView::norm_out_ScalarOpt_dtype_out)
  );
  m.impl("norm.Scalar_out",
         TORCH_FN(ADInplaceOrView::norm_out_Scalar_out)
  );
  m.impl("normal_",
         TORCH_FN(ADInplaceOrView::normal_)
  );
  m.impl("normal.Tensor_float_out",
         TORCH_FN(ADInplaceOrView::normal_out_Tensor_float_out)
  );
  m.impl("normal.float_Tensor_out",
         TORCH_FN(ADInplaceOrView::normal_out_float_Tensor_out)
  );
  m.impl("normal.Tensor_Tensor_out",
         TORCH_FN(ADInplaceOrView::normal_out_Tensor_Tensor_out)
  );
  m.impl("normal.float_float_out",
         TORCH_FN(ADInplaceOrView::normal_out_float_float_out)
  );
  m.impl("normal.out",
         TORCH_FN(ADInplaceOrView::normal_out_out)
  );
  m.impl("ones_like.out",
         TORCH_FN(ADInplaceOrView::ones_like_out_out)
  );
  m.impl("ones.out",
         TORCH_FN(ADInplaceOrView::ones_out_out)
  );
  m.impl("ones.names_out",
         TORCH_FN(ADInplaceOrView::ones_out_names_out)
  );
  m.impl("ormqr.out",
         TORCH_FN(ADInplaceOrView::ormqr_out_out)
  );
  m.impl("permute",
         TORCH_FN(ADInplaceOrView::permute)
  );
  m.impl("permute_copy.out",
         TORCH_FN(ADInplaceOrView::permute_copy_out_out)
  );
  m.impl("pixel_shuffle.out",
         TORCH_FN(ADInplaceOrView::pixel_shuffle_out_out)
  );
  m.impl("pixel_unshuffle.out",
         TORCH_FN(ADInplaceOrView::pixel_unshuffle_out_out)
  );
  m.impl("poisson.out",
         TORCH_FN(ADInplaceOrView::poisson_out_out)
  );
  m.impl("polar.out",
         TORCH_FN(ADInplaceOrView::polar_out_out)
  );
  m.impl("polygamma_",
         TORCH_FN(ADInplaceOrView::polygamma_)
  );
  m.impl("polygamma.out",
         TORCH_FN(ADInplaceOrView::polygamma_out_out)
  );
  m.impl("pow_.Scalar",
         TORCH_FN(ADInplaceOrView::pow__Scalar)
  );
  m.impl("pow_.Tensor",
         TORCH_FN(ADInplaceOrView::pow__Tensor)
  );
  m.impl("pow.Tensor_Tensor_out",
         TORCH_FN(ADInplaceOrView::pow_out_Tensor_Tensor_out)
  );
  m.impl("pow.Scalar_out",
         TORCH_FN(ADInplaceOrView::pow_out_Scalar_out)
  );
  m.impl("pow.Tensor_Scalar_out",
         TORCH_FN(ADInplaceOrView::pow_out_Tensor_Scalar_out)
  );
  m.impl("prod.int_out",
         TORCH_FN(ADInplaceOrView::prod_out_int_out)
  );
  m.impl("prod.out",
         TORCH_FN(ADInplaceOrView::prod_out_out)
  );
  m.impl("put_",
         TORCH_FN(ADInplaceOrView::put_)
  );
  m.impl("put.out",
         TORCH_FN(ADInplaceOrView::put_out_out)
  );
  m.impl("q_per_channel_scales.out",
         TORCH_FN(ADInplaceOrView::q_per_channel_scales_out_out)
  );
  m.impl("q_per_channel_zero_points.out",
         TORCH_FN(ADInplaceOrView::q_per_channel_zero_points_out_out)
  );
  m.impl("quantize_per_channel.out",
         TORCH_FN(ADInplaceOrView::quantize_per_channel_out_out)
  );
  m.impl("quantize_per_tensor_dynamic.out",
         TORCH_FN(ADInplaceOrView::quantize_per_tensor_dynamic_out_out)
  );
  m.impl("quantize_per_tensor.out",
         TORCH_FN(ADInplaceOrView::quantize_per_tensor_out_out)
  );
  m.impl("quantize_per_tensor.tensor_qparams_out",
         TORCH_FN(ADInplaceOrView::quantize_per_tensor_out_tensor_qparams_out)
  );
  m.impl("quantized_batch_norm.out",
         TORCH_FN(ADInplaceOrView::quantized_batch_norm_out_out)
  );
  m.impl("quantized_max_pool1d.out",
         TORCH_FN(ADInplaceOrView::quantized_max_pool1d_out_out)
  );
  m.impl("quantized_max_pool2d.out",
         TORCH_FN(ADInplaceOrView::quantized_max_pool2d_out_out)
  );
  m.impl("quantized_max_pool3d.out",
         TORCH_FN(ADInplaceOrView::quantized_max_pool3d_out_out)
  );
  m.impl("rad2deg_",
         TORCH_FN(ADInplaceOrView::rad2deg_)
  );
  m.impl("rad2deg.out",
         TORCH_FN(ADInplaceOrView::rad2deg_out_out)
  );
  m.impl("rand_like.out",
         TORCH_FN(ADInplaceOrView::rand_like_out_out)
  );
  m.impl("rand.out",
         TORCH_FN(ADInplaceOrView::rand_out_out)
  );
  m.impl("rand.names_out",
         TORCH_FN(ADInplaceOrView::rand_out_names_out)
  );
  m.impl("rand.generator_with_names_out",
         TORCH_FN(ADInplaceOrView::rand_out_generator_with_names_out)
  );
  m.impl("randint_like.out",
         TORCH_FN(ADInplaceOrView::randint_like_out_out)
  );
  m.impl("randint_like.low_dtype_out",
         TORCH_FN(ADInplaceOrView::randint_like_out_low_dtype_out)
  );
  m.impl("randint.out",
         TORCH_FN(ADInplaceOrView::randint_out_out)
  );
  m.impl("randint.generator_out",
         TORCH_FN(ADInplaceOrView::randint_out_generator_out)
  );
  m.impl("randint.low_out",
         TORCH_FN(ADInplaceOrView::randint_out_low_out)
  );
  m.impl("randint.low_generator_out",
         TORCH_FN(ADInplaceOrView::randint_out_low_generator_out)
  );
  m.impl("randn_like.out",
         TORCH_FN(ADInplaceOrView::randn_like_out_out)
  );
  m.impl("randn.names_out",
         TORCH_FN(ADInplaceOrView::randn_out_names_out)
  );
  m.impl("randn.generator_with_names_out",
         TORCH_FN(ADInplaceOrView::randn_out_generator_with_names_out)
  );
  m.impl("random_.from",
         TORCH_FN(ADInplaceOrView::random__from)
  );
  m.impl("random_.to",
         TORCH_FN(ADInplaceOrView::random__to)
  );
  m.impl("random_",
         TORCH_FN(ADInplaceOrView::random_)
  );
  m.impl("random.from_out",
         TORCH_FN(ADInplaceOrView::random_out_from_out)
  );
  m.impl("random.to_out",
         TORCH_FN(ADInplaceOrView::random_out_to_out)
  );
  m.impl("random.out",
         TORCH_FN(ADInplaceOrView::random_out_out)
  );
  m.impl("randperm.out",
         TORCH_FN(ADInplaceOrView::randperm_out_out)
  );
  m.impl("randperm.generator_out",
         TORCH_FN(ADInplaceOrView::randperm_out_generator_out)
  );
  m.impl("range.out_",
         TORCH_FN(ADInplaceOrView::range_out_out_)
  );
  m.impl("range.out",
         TORCH_FN(ADInplaceOrView::range_out_out)
  );
  m.impl("reciprocal_",
         TORCH_FN(ADInplaceOrView::reciprocal_)
  );
  m.impl("reciprocal.out",
         TORCH_FN(ADInplaceOrView::reciprocal_out_out)
  );
  m.impl("reflection_pad1d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::reflection_pad1d_backward_out_grad_input)
  );
  m.impl("reflection_pad1d.out",
         TORCH_FN(ADInplaceOrView::reflection_pad1d_out_out)
  );
  m.impl("reflection_pad2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::reflection_pad2d_backward_out_grad_input)
  );
  m.impl("reflection_pad2d.out",
         TORCH_FN(ADInplaceOrView::reflection_pad2d_out_out)
  );
  m.impl("reflection_pad3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::reflection_pad3d_backward_out_grad_input)
  );
  m.impl("reflection_pad3d.out",
         TORCH_FN(ADInplaceOrView::reflection_pad3d_out_out)
  );
  m.impl("relu_",
         TORCH_FN(ADInplaceOrView::relu_)
  );
  m.impl("relu.out",
         TORCH_FN(ADInplaceOrView::relu_out_out)
  );
  m.impl("remainder_.Scalar",
         TORCH_FN(ADInplaceOrView::remainder__Scalar)
  );
  m.impl("remainder_.Tensor",
         TORCH_FN(ADInplaceOrView::remainder__Tensor)
  );
  m.impl("remainder.Scalar_out",
         TORCH_FN(ADInplaceOrView::remainder_out_Scalar_out)
  );
  m.impl("remainder.Tensor_out",
         TORCH_FN(ADInplaceOrView::remainder_out_Tensor_out)
  );
  m.impl("remainder.Scalar_Tensor_out",
         TORCH_FN(ADInplaceOrView::remainder_out_Scalar_Tensor_out)
  );
  m.impl("renorm_",
         TORCH_FN(ADInplaceOrView::renorm_)
  );
  m.impl("renorm.out",
         TORCH_FN(ADInplaceOrView::renorm_out_out)
  );
  m.impl("repeat_interleave.Tensor_out",
         TORCH_FN(ADInplaceOrView::repeat_interleave_out_Tensor_out)
  );
  m.impl("repeat.out",
         TORCH_FN(ADInplaceOrView::repeat_out_out)
  );
  m.impl("replication_pad1d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::replication_pad1d_backward_out_grad_input)
  );
  m.impl("replication_pad1d.out",
         TORCH_FN(ADInplaceOrView::replication_pad1d_out_out)
  );
  m.impl("replication_pad2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::replication_pad2d_backward_out_grad_input)
  );
  m.impl("replication_pad2d.out",
         TORCH_FN(ADInplaceOrView::replication_pad2d_out_out)
  );
  m.impl("replication_pad3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::replication_pad3d_backward_out_grad_input)
  );
  m.impl("replication_pad3d.out",
         TORCH_FN(ADInplaceOrView::replication_pad3d_out_out)
  );
  m.impl("resize_as.out",
         TORCH_FN(ADInplaceOrView::resize_as_out_out)
  );
  m.impl("resize_as_sparse_",
         TORCH_FN(ADInplaceOrView::resize_as_sparse_)
  );
  m.impl("resize_as_sparse.out",
         TORCH_FN(ADInplaceOrView::resize_as_sparse_out_out)
  );
  m.impl("resize.out",
         TORCH_FN(ADInplaceOrView::resize_out_out)
  );
  m.impl("roll.out",
         TORCH_FN(ADInplaceOrView::roll_out_out)
  );
  m.impl("rot90.out",
         TORCH_FN(ADInplaceOrView::rot90_out_out)
  );
  m.impl("round_",
         TORCH_FN(ADInplaceOrView::round_)
  );
  m.impl("round_.decimals",
         TORCH_FN(ADInplaceOrView::round__decimals)
  );
  m.impl("round.out",
         TORCH_FN(ADInplaceOrView::round_out_out)
  );
  m.impl("round.decimals_out",
         TORCH_FN(ADInplaceOrView::round_out_decimals_out)
  );
  m.impl("row_indices",
         TORCH_FN(ADInplaceOrView::row_indices)
  );
  m.impl("row_indices_copy.out",
         TORCH_FN(ADInplaceOrView::row_indices_copy_out_out)
  );
  m.impl("rrelu_with_noise_",
         TORCH_FN(ADInplaceOrView::rrelu_with_noise_)
  );
  m.impl("rrelu_with_noise_backward.out",
         TORCH_FN(ADInplaceOrView::rrelu_with_noise_backward_out_out)
  );
  m.impl("rrelu_with_noise.out",
         TORCH_FN(ADInplaceOrView::rrelu_with_noise_out_out)
  );
  m.impl("rsqrt_",
         TORCH_FN(ADInplaceOrView::rsqrt_)
  );
  m.impl("rsqrt.out",
         TORCH_FN(ADInplaceOrView::rsqrt_out_out)
  );
  m.impl("rsub.Tensor_out",
         TORCH_FN(ADInplaceOrView::rsub_out_Tensor_out)
  );
  m.impl("rsub.Scalar_out",
         TORCH_FN(ADInplaceOrView::rsub_out_Scalar_out)
  );
  m.impl("scalar_tensor.out",
         TORCH_FN(ADInplaceOrView::scalar_tensor_out_out)
  );
  m.impl("scatter_.src",
         TORCH_FN(ADInplaceOrView::scatter__src)
  );
  m.impl("scatter_.value",
         TORCH_FN(ADInplaceOrView::scatter__value)
  );
  m.impl("scatter_.reduce",
         TORCH_FN(ADInplaceOrView::scatter__reduce)
  );
  m.impl("scatter_.value_reduce",
         TORCH_FN(ADInplaceOrView::scatter__value_reduce)
  );
  m.impl("scatter_add_",
         TORCH_FN(ADInplaceOrView::scatter_add_)
  );
  m.impl("scatter_add.out",
         TORCH_FN(ADInplaceOrView::scatter_add_out_out)
  );
  m.impl("scatter.src_out",
         TORCH_FN(ADInplaceOrView::scatter_out_src_out)
  );
  m.impl("scatter.value_out",
         TORCH_FN(ADInplaceOrView::scatter_out_value_out)
  );
  m.impl("scatter.reduce_out",
         TORCH_FN(ADInplaceOrView::scatter_out_reduce_out)
  );
  m.impl("scatter.value_reduce_out",
         TORCH_FN(ADInplaceOrView::scatter_out_value_reduce_out)
  );
  m.impl("scatter_reduce_.two",
         TORCH_FN(ADInplaceOrView::scatter_reduce__two)
  );
  m.impl("scatter_reduce.two_out",
         TORCH_FN(ADInplaceOrView::scatter_reduce_out_two_out)
  );
  m.impl("searchsorted.Tensor_out",
         TORCH_FN(ADInplaceOrView::searchsorted_out_Tensor_out)
  );
  m.impl("searchsorted.Scalar_out",
         TORCH_FN(ADInplaceOrView::searchsorted_out_Scalar_out)
  );
  m.impl("segment_reduce.out",
         TORCH_FN(ADInplaceOrView::segment_reduce_out_out)
  );
  m.impl("select.int",
         TORCH_FN(ADInplaceOrView::select_int)
  );
  m.impl("select_backward.out",
         TORCH_FN(ADInplaceOrView::select_backward_out_out)
  );
  m.impl("select_copy.int_out",
         TORCH_FN(ADInplaceOrView::select_copy_out_int_out)
  );
  m.impl("select_scatter.out",
         TORCH_FN(ADInplaceOrView::select_scatter_out_out)
  );
  m.impl("set_.source_Storage",
         TORCH_FN(ADInplaceOrView::set__source_Storage)
  );
  m.impl("set_.source_Storage_storage_offset",
         TORCH_FN(ADInplaceOrView::set__source_Storage_storage_offset)
  );
  m.impl("set_.source_Tensor",
         TORCH_FN(ADInplaceOrView::set__source_Tensor)
  );
  m.impl("set_",
         TORCH_FN(ADInplaceOrView::set_)
  );
  m.impl("set.source_Storage_out",
         TORCH_FN(ADInplaceOrView::set_out_source_Storage_out)
  );
  m.impl("set.source_Storage_storage_offset_out",
         TORCH_FN(ADInplaceOrView::set_out_source_Storage_storage_offset_out)
  );
  m.impl("set.source_Tensor_out",
         TORCH_FN(ADInplaceOrView::set_out_source_Tensor_out)
  );
  m.impl("set.out",
         TORCH_FN(ADInplaceOrView::set_out_out)
  );
  m.impl("sgn_",
         TORCH_FN(ADInplaceOrView::sgn_)
  );
  m.impl("sgn.out",
         TORCH_FN(ADInplaceOrView::sgn_out_out)
  );
  m.impl("sigmoid_",
         TORCH_FN(ADInplaceOrView::sigmoid_)
  );
  m.impl("sigmoid_backward.grad_input",
         TORCH_FN(ADInplaceOrView::sigmoid_backward_out_grad_input)
  );
  m.impl("sigmoid.out",
         TORCH_FN(ADInplaceOrView::sigmoid_out_out)
  );
  m.impl("sign_",
         TORCH_FN(ADInplaceOrView::sign_)
  );
  m.impl("sign.out",
         TORCH_FN(ADInplaceOrView::sign_out_out)
  );
  m.impl("signbit.out",
         TORCH_FN(ADInplaceOrView::signbit_out_out)
  );
  m.impl("silu_",
         TORCH_FN(ADInplaceOrView::silu_)
  );
  m.impl("silu_backward.grad_input",
         TORCH_FN(ADInplaceOrView::silu_backward_out_grad_input)
  );
  m.impl("silu.out",
         TORCH_FN(ADInplaceOrView::silu_out_out)
  );
  m.impl("sin_",
         TORCH_FN(ADInplaceOrView::sin_)
  );
  m.impl("sin.out",
         TORCH_FN(ADInplaceOrView::sin_out_out)
  );
  m.impl("sinc_",
         TORCH_FN(ADInplaceOrView::sinc_)
  );
  m.impl("sinc.out",
         TORCH_FN(ADInplaceOrView::sinc_out_out)
  );
  m.impl("sinh_",
         TORCH_FN(ADInplaceOrView::sinh_)
  );
  m.impl("sinh.out",
         TORCH_FN(ADInplaceOrView::sinh_out_out)
  );
  m.impl("slice.Tensor",
         TORCH_FN(ADInplaceOrView::slice_Tensor)
  );
  m.impl("slice_backward.out",
         TORCH_FN(ADInplaceOrView::slice_backward_out_out)
  );
  m.impl("slice_copy.Tensor_out",
         TORCH_FN(ADInplaceOrView::slice_copy_out_Tensor_out)
  );
  m.impl("slice_inverse",
         TORCH_FN(ADInplaceOrView::slice_inverse)
  );
  m.impl("slice_scatter.out",
         TORCH_FN(ADInplaceOrView::slice_scatter_out_out)
  );
  m.impl("slow_conv3d_forward.output",
         TORCH_FN(ADInplaceOrView::slow_conv3d_forward_out_output)
  );
  m.impl("slow_conv_dilated2d.out",
         TORCH_FN(ADInplaceOrView::slow_conv_dilated2d_out_out)
  );
  m.impl("slow_conv_dilated3d.out",
         TORCH_FN(ADInplaceOrView::slow_conv_dilated3d_out_out)
  );
  m.impl("slow_conv_transpose2d.out",
         TORCH_FN(ADInplaceOrView::slow_conv_transpose2d_out_out)
  );
  m.impl("slow_conv_transpose3d.out",
         TORCH_FN(ADInplaceOrView::slow_conv_transpose3d_out_out)
  );
  m.impl("smooth_l1_loss_backward.grad_input",
         TORCH_FN(ADInplaceOrView::smooth_l1_loss_backward_out_grad_input)
  );
  m.impl("smooth_l1_loss.out",
         TORCH_FN(ADInplaceOrView::smooth_l1_loss_out_out)
  );
  m.impl("soft_margin_loss_backward.grad_input",
         TORCH_FN(ADInplaceOrView::soft_margin_loss_backward_out_grad_input)
  );
  m.impl("soft_margin_loss.out",
         TORCH_FN(ADInplaceOrView::soft_margin_loss_out_out)
  );
  m.impl("softmax.int_out",
         TORCH_FN(ADInplaceOrView::softmax_out_int_out)
  );
  m.impl("softplus_backward.grad_input",
         TORCH_FN(ADInplaceOrView::softplus_backward_out_grad_input)
  );
  m.impl("softplus.out",
         TORCH_FN(ADInplaceOrView::softplus_out_out)
  );
  m.impl("softshrink_backward.grad_input",
         TORCH_FN(ADInplaceOrView::softshrink_backward_out_grad_input)
  );
  m.impl("softshrink.out",
         TORCH_FN(ADInplaceOrView::softshrink_out_out)
  );
  m.impl("sort.values",
         TORCH_FN(ADInplaceOrView::sort_out_values)
  );
  m.impl("sort.values_stable",
         TORCH_FN(ADInplaceOrView::sort_out_values_stable)
  );
  m.impl("sparse_coo_tensor.size_out",
         TORCH_FN(ADInplaceOrView::sparse_coo_tensor_out_size_out)
  );
  m.impl("sparse_mask.out",
         TORCH_FN(ADInplaceOrView::sparse_mask_out_out)
  );
  m.impl("sparse_resize_",
         TORCH_FN(ADInplaceOrView::sparse_resize_)
  );
  m.impl("sparse_resize_and_clear_",
         TORCH_FN(ADInplaceOrView::sparse_resize_and_clear_)
  );
  m.impl("sparse_resize_and_clear.out",
         TORCH_FN(ADInplaceOrView::sparse_resize_and_clear_out_out)
  );
  m.impl("sparse_resize.out",
         TORCH_FN(ADInplaceOrView::sparse_resize_out_out)
  );
  m.impl("sparse_sampled_addmm.out",
         TORCH_FN(ADInplaceOrView::sparse_sampled_addmm_out_out)
  );
  m.impl("special_airy_ai.out",
         TORCH_FN(ADInplaceOrView::special_airy_ai_out_out)
  );
  m.impl("special_bessel_j0.out",
         TORCH_FN(ADInplaceOrView::special_bessel_j0_out_out)
  );
  m.impl("special_bessel_j1.out",
         TORCH_FN(ADInplaceOrView::special_bessel_j1_out_out)
  );
  m.impl("special_bessel_y0.out",
         TORCH_FN(ADInplaceOrView::special_bessel_y0_out_out)
  );
  m.impl("special_bessel_y1.out",
         TORCH_FN(ADInplaceOrView::special_bessel_y1_out_out)
  );
  m.impl("special_chebyshev_polynomial_t.out",
         TORCH_FN(ADInplaceOrView::special_chebyshev_polynomial_t_out_out)
  );
  m.impl("special_chebyshev_polynomial_t.x_scalar_out",
         TORCH_FN(ADInplaceOrView::special_chebyshev_polynomial_t_out_x_scalar_out)
  );
  m.impl("special_chebyshev_polynomial_t.n_scalar_out",
         TORCH_FN(ADInplaceOrView::special_chebyshev_polynomial_t_out_n_scalar_out)
  );
  m.impl("special_chebyshev_polynomial_u.out",
         TORCH_FN(ADInplaceOrView::special_chebyshev_polynomial_u_out_out)
  );
  m.impl("special_chebyshev_polynomial_u.x_scalar_out",
         TORCH_FN(ADInplaceOrView::special_chebyshev_polynomial_u_out_x_scalar_out)
  );
  m.impl("special_chebyshev_polynomial_u.n_scalar_out",
         TORCH_FN(ADInplaceOrView::special_chebyshev_polynomial_u_out_n_scalar_out)
  );
  m.impl("special_chebyshev_polynomial_v.out",
         TORCH_FN(ADInplaceOrView::special_chebyshev_polynomial_v_out_out)
  );
  m.impl("special_chebyshev_polynomial_v.x_scalar_out",
         TORCH_FN(ADInplaceOrView::special_chebyshev_polynomial_v_out_x_scalar_out)
  );
  m.impl("special_chebyshev_polynomial_v.n_scalar_out",
         TORCH_FN(ADInplaceOrView::special_chebyshev_polynomial_v_out_n_scalar_out)
  );
  m.impl("special_chebyshev_polynomial_w.out",
         TORCH_FN(ADInplaceOrView::special_chebyshev_polynomial_w_out_out)
  );
  m.impl("special_chebyshev_polynomial_w.x_scalar_out",
         TORCH_FN(ADInplaceOrView::special_chebyshev_polynomial_w_out_x_scalar_out)
  );
  m.impl("special_chebyshev_polynomial_w.n_scalar_out",
         TORCH_FN(ADInplaceOrView::special_chebyshev_polynomial_w_out_n_scalar_out)
  );
  m.impl("special_entr.out",
         TORCH_FN(ADInplaceOrView::special_entr_out_out)
  );
  m.impl("special_erfcx.out",
         TORCH_FN(ADInplaceOrView::special_erfcx_out_out)
  );
  m.impl("special_hermite_polynomial_h.out",
         TORCH_FN(ADInplaceOrView::special_hermite_polynomial_h_out_out)
  );
  m.impl("special_hermite_polynomial_h.x_scalar_out",
         TORCH_FN(ADInplaceOrView::special_hermite_polynomial_h_out_x_scalar_out)
  );
  m.impl("special_hermite_polynomial_h.n_scalar_out",
         TORCH_FN(ADInplaceOrView::special_hermite_polynomial_h_out_n_scalar_out)
  );
  m.impl("special_hermite_polynomial_he.out",
         TORCH_FN(ADInplaceOrView::special_hermite_polynomial_he_out_out)
  );
  m.impl("special_hermite_polynomial_he.x_scalar_out",
         TORCH_FN(ADInplaceOrView::special_hermite_polynomial_he_out_x_scalar_out)
  );
  m.impl("special_hermite_polynomial_he.n_scalar_out",
         TORCH_FN(ADInplaceOrView::special_hermite_polynomial_he_out_n_scalar_out)
  );
  m.impl("special_i0e.out",
         TORCH_FN(ADInplaceOrView::special_i0e_out_out)
  );
  m.impl("special_i1.out",
         TORCH_FN(ADInplaceOrView::special_i1_out_out)
  );
  m.impl("special_i1e.out",
         TORCH_FN(ADInplaceOrView::special_i1e_out_out)
  );
  m.impl("special_laguerre_polynomial_l.out",
         TORCH_FN(ADInplaceOrView::special_laguerre_polynomial_l_out_out)
  );
  m.impl("special_laguerre_polynomial_l.x_scalar_out",
         TORCH_FN(ADInplaceOrView::special_laguerre_polynomial_l_out_x_scalar_out)
  );
  m.impl("special_laguerre_polynomial_l.n_scalar_out",
         TORCH_FN(ADInplaceOrView::special_laguerre_polynomial_l_out_n_scalar_out)
  );
  m.impl("special_legendre_polynomial_p.out",
         TORCH_FN(ADInplaceOrView::special_legendre_polynomial_p_out_out)
  );
  m.impl("special_legendre_polynomial_p.x_scalar_out",
         TORCH_FN(ADInplaceOrView::special_legendre_polynomial_p_out_x_scalar_out)
  );
  m.impl("special_legendre_polynomial_p.n_scalar_out",
         TORCH_FN(ADInplaceOrView::special_legendre_polynomial_p_out_n_scalar_out)
  );
  m.impl("special_log_ndtr.out",
         TORCH_FN(ADInplaceOrView::special_log_ndtr_out_out)
  );
  m.impl("special_modified_bessel_i0.out",
         TORCH_FN(ADInplaceOrView::special_modified_bessel_i0_out_out)
  );
  m.impl("special_modified_bessel_i1.out",
         TORCH_FN(ADInplaceOrView::special_modified_bessel_i1_out_out)
  );
  m.impl("special_modified_bessel_k0.out",
         TORCH_FN(ADInplaceOrView::special_modified_bessel_k0_out_out)
  );
  m.impl("special_modified_bessel_k1.out",
         TORCH_FN(ADInplaceOrView::special_modified_bessel_k1_out_out)
  );
  m.impl("special_ndtri.out",
         TORCH_FN(ADInplaceOrView::special_ndtri_out_out)
  );
  m.impl("special_scaled_modified_bessel_k0.out",
         TORCH_FN(ADInplaceOrView::special_scaled_modified_bessel_k0_out_out)
  );
  m.impl("special_scaled_modified_bessel_k1.out",
         TORCH_FN(ADInplaceOrView::special_scaled_modified_bessel_k1_out_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_t.out",
         TORCH_FN(ADInplaceOrView::special_shifted_chebyshev_polynomial_t_out_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_t.x_scalar_out",
         TORCH_FN(ADInplaceOrView::special_shifted_chebyshev_polynomial_t_out_x_scalar_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_t.n_scalar_out",
         TORCH_FN(ADInplaceOrView::special_shifted_chebyshev_polynomial_t_out_n_scalar_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_u.out",
         TORCH_FN(ADInplaceOrView::special_shifted_chebyshev_polynomial_u_out_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_u.x_scalar_out",
         TORCH_FN(ADInplaceOrView::special_shifted_chebyshev_polynomial_u_out_x_scalar_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_u.n_scalar_out",
         TORCH_FN(ADInplaceOrView::special_shifted_chebyshev_polynomial_u_out_n_scalar_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_v.out",
         TORCH_FN(ADInplaceOrView::special_shifted_chebyshev_polynomial_v_out_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_v.x_scalar_out",
         TORCH_FN(ADInplaceOrView::special_shifted_chebyshev_polynomial_v_out_x_scalar_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_v.n_scalar_out",
         TORCH_FN(ADInplaceOrView::special_shifted_chebyshev_polynomial_v_out_n_scalar_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_w.out",
         TORCH_FN(ADInplaceOrView::special_shifted_chebyshev_polynomial_w_out_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_w.x_scalar_out",
         TORCH_FN(ADInplaceOrView::special_shifted_chebyshev_polynomial_w_out_x_scalar_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_w.n_scalar_out",
         TORCH_FN(ADInplaceOrView::special_shifted_chebyshev_polynomial_w_out_n_scalar_out)
  );
  m.impl("special_spherical_bessel_j0.out",
         TORCH_FN(ADInplaceOrView::special_spherical_bessel_j0_out_out)
  );
  m.impl("special_xlog1py.out",
         TORCH_FN(ADInplaceOrView::special_xlog1py_out_out)
  );
  m.impl("special_xlog1py.self_scalar_out",
         TORCH_FN(ADInplaceOrView::special_xlog1py_out_self_scalar_out)
  );
  m.impl("special_xlog1py.other_scalar_out",
         TORCH_FN(ADInplaceOrView::special_xlog1py_out_other_scalar_out)
  );
  m.impl("special_zeta.out",
         TORCH_FN(ADInplaceOrView::special_zeta_out_out)
  );
  m.impl("special_zeta.self_scalar_out",
         TORCH_FN(ADInplaceOrView::special_zeta_out_self_scalar_out)
  );
  m.impl("special_zeta.other_scalar_out",
         TORCH_FN(ADInplaceOrView::special_zeta_out_other_scalar_out)
  );
  m.impl("split.Tensor",
         TORCH_FN(ADInplaceOrView::split_Tensor)
  );
  m.impl("split_with_sizes",
         TORCH_FN(ADInplaceOrView::split_with_sizes)
  );
  m.impl("sqrt_",
         TORCH_FN(ADInplaceOrView::sqrt_)
  );
  m.impl("sqrt.out",
         TORCH_FN(ADInplaceOrView::sqrt_out_out)
  );
  m.impl("squeeze",
         TORCH_FN(ADInplaceOrView::squeeze)
  );
  m.impl("squeeze.dim",
         TORCH_FN(ADInplaceOrView::squeeze_dim)
  );
  m.impl("squeeze.dims",
         TORCH_FN(ADInplaceOrView::squeeze_dims)
  );
  m.impl("squeeze_",
         TORCH_FN(ADInplaceOrView::squeeze_)
  );
  m.impl("squeeze_.dim",
         TORCH_FN(ADInplaceOrView::squeeze__dim)
  );
  m.impl("squeeze_.dims",
         TORCH_FN(ADInplaceOrView::squeeze__dims)
  );
  m.impl("squeeze_copy.out",
         TORCH_FN(ADInplaceOrView::squeeze_copy_out_out)
  );
  m.impl("squeeze_copy.dim_out",
         TORCH_FN(ADInplaceOrView::squeeze_copy_out_dim_out)
  );
  m.impl("squeeze_copy.dims_out",
         TORCH_FN(ADInplaceOrView::squeeze_copy_out_dims_out)
  );
  m.impl("sspaddmm.out",
         TORCH_FN(ADInplaceOrView::sspaddmm_out_out)
  );
  m.impl("stack.out",
         TORCH_FN(ADInplaceOrView::stack_out_out)
  );
  m.impl("std_mean.correction_out",
         TORCH_FN(ADInplaceOrView::std_mean_out_correction_out)
  );
  m.impl("std.correction_out",
         TORCH_FN(ADInplaceOrView::std_out_correction_out)
  );
  m.impl("sub_.Tensor",
         TORCH_FN(ADInplaceOrView::sub__Tensor)
  );
  m.impl("sub_.Scalar",
         TORCH_FN(ADInplaceOrView::sub__Scalar)
  );
  m.impl("sub.out",
         TORCH_FN(ADInplaceOrView::sub_out_out)
  );
  m.impl("sub.Scalar_out",
         TORCH_FN(ADInplaceOrView::sub_out_Scalar_out)
  );
  m.impl("sum.IntList_out",
         TORCH_FN(ADInplaceOrView::sum_out_IntList_out)
  );
  m.impl("sum.out",
         TORCH_FN(ADInplaceOrView::sum_out_out)
  );
  m.impl("t",
         TORCH_FN(ADInplaceOrView::t)
  );
  m.impl("t_",
         TORCH_FN(ADInplaceOrView::t_)
  );
  m.impl("t_copy.out",
         TORCH_FN(ADInplaceOrView::t_copy_out_out)
  );
  m.impl("take.out",
         TORCH_FN(ADInplaceOrView::take_out_out)
  );
  m.impl("tan_",
         TORCH_FN(ADInplaceOrView::tan_)
  );
  m.impl("tan.out",
         TORCH_FN(ADInplaceOrView::tan_out_out)
  );
  m.impl("tanh_",
         TORCH_FN(ADInplaceOrView::tanh_)
  );
  m.impl("tanh_backward.grad_input",
         TORCH_FN(ADInplaceOrView::tanh_backward_out_grad_input)
  );
  m.impl("tanh.out",
         TORCH_FN(ADInplaceOrView::tanh_out_out)
  );
  m.impl("threshold_",
         TORCH_FN(ADInplaceOrView::threshold_)
  );
  m.impl("threshold_backward.grad_input",
         TORCH_FN(ADInplaceOrView::threshold_backward_out_grad_input)
  );
  m.impl("threshold.out",
         TORCH_FN(ADInplaceOrView::threshold_out_out)
  );
  m.impl("to_mkldnn.out",
         TORCH_FN(ADInplaceOrView::to_mkldnn_out_out)
  );
  m.impl("to_padded_tensor.out",
         TORCH_FN(ADInplaceOrView::to_padded_tensor_out_out)
  );
  m.impl("topk.values",
         TORCH_FN(ADInplaceOrView::topk_out_values)
  );
  m.impl("trace.out",
         TORCH_FN(ADInplaceOrView::trace_out_out)
  );
  m.impl("transpose.int",
         TORCH_FN(ADInplaceOrView::transpose_int)
  );
  m.impl("transpose_",
         TORCH_FN(ADInplaceOrView::transpose_)
  );
  m.impl("transpose_copy.int_out",
         TORCH_FN(ADInplaceOrView::transpose_copy_out_int_out)
  );
  m.impl("triangular_solve.X",
         TORCH_FN(ADInplaceOrView::triangular_solve_out_X)
  );
  m.impl("tril_",
         TORCH_FN(ADInplaceOrView::tril_)
  );
  m.impl("tril_indices.out",
         TORCH_FN(ADInplaceOrView::tril_indices_out_out)
  );
  m.impl("tril.out",
         TORCH_FN(ADInplaceOrView::tril_out_out)
  );
  m.impl("triu_",
         TORCH_FN(ADInplaceOrView::triu_)
  );
  m.impl("triu_indices.out",
         TORCH_FN(ADInplaceOrView::triu_indices_out_out)
  );
  m.impl("triu.out",
         TORCH_FN(ADInplaceOrView::triu_out_out)
  );
  m.impl("trunc_",
         TORCH_FN(ADInplaceOrView::trunc_)
  );
  m.impl("trunc.out",
         TORCH_FN(ADInplaceOrView::trunc_out_out)
  );
  m.impl("unbind.int",
         TORCH_FN(ADInplaceOrView::unbind_int)
  );
  m.impl("unfold",
         TORCH_FN(ADInplaceOrView::unfold)
  );
  m.impl("unfold_backward.out",
         TORCH_FN(ADInplaceOrView::unfold_backward_out_out)
  );
  m.impl("unfold_copy.out",
         TORCH_FN(ADInplaceOrView::unfold_copy_out_out)
  );
  m.impl("uniform_",
         TORCH_FN(ADInplaceOrView::uniform_)
  );
  m.impl("uniform.out",
         TORCH_FN(ADInplaceOrView::uniform_out_out)
  );
  m.impl("unique_consecutive.out",
         TORCH_FN(ADInplaceOrView::unique_consecutive_out_out)
  );
  m.impl("unique_dim_consecutive.out",
         TORCH_FN(ADInplaceOrView::unique_dim_consecutive_out_out)
  );
  m.impl("unique_dim.out",
         TORCH_FN(ADInplaceOrView::unique_dim_out_out)
  );
  m.impl("unsqueeze",
         TORCH_FN(ADInplaceOrView::unsqueeze)
  );
  m.impl("unsqueeze_",
         TORCH_FN(ADInplaceOrView::unsqueeze_)
  );
  m.impl("unsqueeze_copy.out",
         TORCH_FN(ADInplaceOrView::unsqueeze_copy_out_out)
  );
  m.impl("upsample_bicubic2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::upsample_bicubic2d_backward_out_grad_input)
  );
  m.impl("upsample_bicubic2d.out",
         TORCH_FN(ADInplaceOrView::upsample_bicubic2d_out_out)
  );
  m.impl("upsample_bilinear2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::upsample_bilinear2d_backward_out_grad_input)
  );
  m.impl("upsample_bilinear2d.out",
         TORCH_FN(ADInplaceOrView::upsample_bilinear2d_out_out)
  );
  m.impl("upsample_linear1d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::upsample_linear1d_backward_out_grad_input)
  );
  m.impl("upsample_linear1d.out",
         TORCH_FN(ADInplaceOrView::upsample_linear1d_out_out)
  );
  m.impl("upsample_nearest1d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::upsample_nearest1d_backward_out_grad_input)
  );
  m.impl("upsample_nearest1d.out",
         TORCH_FN(ADInplaceOrView::upsample_nearest1d_out_out)
  );
  m.impl("upsample_nearest2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::upsample_nearest2d_backward_out_grad_input)
  );
  m.impl("upsample_nearest2d.out",
         TORCH_FN(ADInplaceOrView::upsample_nearest2d_out_out)
  );
  m.impl("upsample_nearest3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::upsample_nearest3d_backward_out_grad_input)
  );
  m.impl("upsample_nearest3d.out",
         TORCH_FN(ADInplaceOrView::upsample_nearest3d_out_out)
  );
  m.impl("upsample_trilinear3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::upsample_trilinear3d_backward_out_grad_input)
  );
  m.impl("upsample_trilinear3d.out",
         TORCH_FN(ADInplaceOrView::upsample_trilinear3d_out_out)
  );
  m.impl("values",
         TORCH_FN(ADInplaceOrView::values)
  );
  m.impl("values_copy.out",
         TORCH_FN(ADInplaceOrView::values_copy_out_out)
  );
  m.impl("var_mean.correction_out",
         TORCH_FN(ADInplaceOrView::var_mean_out_correction_out)
  );
  m.impl("var.correction_out",
         TORCH_FN(ADInplaceOrView::var_out_correction_out)
  );
  m.impl("vdot.out",
         TORCH_FN(ADInplaceOrView::vdot_out_out)
  );
  m.impl("view",
         TORCH_FN(ADInplaceOrView::view)
  );
  m.impl("view.dtype",
         TORCH_FN(ADInplaceOrView::view_dtype)
  );
  m.impl("view_as_complex",
         TORCH_FN(ADInplaceOrView::view_as_complex)
  );
  m.impl("view_as_complex_copy.out",
         TORCH_FN(ADInplaceOrView::view_as_complex_copy_out_out)
  );
  m.impl("view_as_real",
         TORCH_FN(ADInplaceOrView::view_as_real)
  );
  m.impl("view_as_real_copy.out",
         TORCH_FN(ADInplaceOrView::view_as_real_copy_out_out)
  );
  m.impl("view_copy.out",
         TORCH_FN(ADInplaceOrView::view_copy_out_out)
  );
  m.impl("view_copy.dtype_out",
         TORCH_FN(ADInplaceOrView::view_copy_out_dtype_out)
  );
  m.impl("where.self_out",
         TORCH_FN(ADInplaceOrView::where_out_self_out)
  );
  m.impl("xlogy_.Tensor",
         TORCH_FN(ADInplaceOrView::xlogy__Tensor)
  );
  m.impl("xlogy_.Scalar_Other",
         TORCH_FN(ADInplaceOrView::xlogy__Scalar_Other)
  );
  m.impl("xlogy.OutTensor",
         TORCH_FN(ADInplaceOrView::xlogy_out_OutTensor)
  );
  m.impl("xlogy.OutScalar_Self",
         TORCH_FN(ADInplaceOrView::xlogy_out_OutScalar_Self)
  );
  m.impl("xlogy.OutScalar_Other",
         TORCH_FN(ADInplaceOrView::xlogy_out_OutScalar_Other)
  );
  m.impl("zero_",
         TORCH_FN(ADInplaceOrView::zero_)
  );
  m.impl("zero.out",
         TORCH_FN(ADInplaceOrView::zero_out_out)
  );
  m.impl("zeros_like.out",
         TORCH_FN(ADInplaceOrView::zeros_like_out_out)
  );
  m.impl("zeros.out",
         TORCH_FN(ADInplaceOrView::zeros_out_out)
  );
  m.impl("zeros.names_out",
         TORCH_FN(ADInplaceOrView::zeros_out_names_out)
  );;
}

}  // namespace
} // namespace torch
