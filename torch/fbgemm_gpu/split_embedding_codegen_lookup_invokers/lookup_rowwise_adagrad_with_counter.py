#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .lookup_args import *


def invoke(
    common_args: CommonArgs,
    optimizer_args: OptimizerArgs,
    momentum1: Momentum,
    prev_iter: Momentum,
    row_counter: Momentum,
    iter: int,
    max_counter: float,
) -> torch.Tensor:
    if (common_args.host_weights.numel() > 0):
        return torch.ops.fbgemm.split_embedding_codegen_lookup_rowwise_adagrad_with_counter_function_cpu(
            # common_args
            host_weights=common_args.host_weights,
            weights_placements=common_args.weights_placements,
            weights_offsets=common_args.weights_offsets,
            D_offsets=common_args.D_offsets,
            total_D=common_args.total_D,
            max_D=common_args.max_D,
            hash_size_cumsum=common_args.hash_size_cumsum,
            total_hash_size_bits=common_args.total_hash_size_bits,
            indices=common_args.indices,
            offsets=common_args.offsets,
            pooling_mode=common_args.pooling_mode,
            indice_weights=common_args.indice_weights,
            feature_requires_grad=common_args.feature_requires_grad,
            # optimizer_args
            gradient_clipping = optimizer_args.gradient_clipping,
            max_gradient=optimizer_args.max_gradient,
            stochastic_rounding=optimizer_args.stochastic_rounding,
            learning_rate=optimizer_args.learning_rate,
            eps=optimizer_args.eps,
            weight_decay=optimizer_args.weight_decay,
            weight_decay_mode=optimizer_args.weight_decay_mode,
            counter_halflife=optimizer_args.counter_halflife,
            adjustment_iter=optimizer_args.adjustment_iter,
            adjustment_ub=optimizer_args.adjustment_ub,
            learning_rate_mode=optimizer_args.learning_rate_mode,
            grad_sum_decay=optimizer_args.grad_sum_decay,
            tail_id_threshold=optimizer_args.tail_id_threshold,
            is_tail_id_thresh_ratio=optimizer_args.is_tail_id_thresh_ratio,
            # momentum1
            momentum1_host=momentum1.host,
            momentum1_offsets=momentum1.offsets,
            momentum1_placements=momentum1.placements,
            # momentum2
            # prev_iter
            prev_iter_host=prev_iter.host,
            prev_iter_offsets=prev_iter.offsets,
            prev_iter_placements=prev_iter.placements,
            # row_counter
            row_counter_host=row_counter.host,
            row_counter_offsets=row_counter.offsets,
            row_counter_placements=row_counter.placements,
            # iter
            iter=iter,
            # max counter
            max_counter=max_counter,
        )
    vbe_metadata = common_args.vbe_metadata
    return torch.ops.fbgemm.split_embedding_codegen_lookup_rowwise_adagrad_with_counter_function(
        # common_args
        placeholder_autograd_tensor=common_args.placeholder_autograd_tensor,
        dev_weights=common_args.dev_weights,
        uvm_weights=common_args.uvm_weights,
        lxu_cache_weights=common_args.lxu_cache_weights,
        weights_placements=common_args.weights_placements,
        weights_offsets=common_args.weights_offsets,
        D_offsets=common_args.D_offsets,
        total_D=common_args.total_D,
        max_D=common_args.max_D,
        hash_size_cumsum=common_args.hash_size_cumsum,
        total_hash_size_bits=common_args.total_hash_size_bits,
        indices=common_args.indices,
        offsets=common_args.offsets,
        pooling_mode=common_args.pooling_mode,
        indice_weights=common_args.indice_weights,
        feature_requires_grad=common_args.feature_requires_grad,
        lxu_cache_locations=common_args.lxu_cache_locations,
        # VBE metadata
        B_offsets=vbe_metadata.B_offsets,
        vbe_output_offsets_feature_rank=vbe_metadata.output_offsets_feature_rank,
        vbe_B_offsets_rank_per_feature=vbe_metadata.B_offsets_rank_per_feature,
        max_B=vbe_metadata.max_B,
        max_B_feature_rank=vbe_metadata.max_B_feature_rank,
        vbe_output_size=vbe_metadata.output_size,
        # optimizer_args
        gradient_clipping = optimizer_args.gradient_clipping,
        max_gradient=optimizer_args.max_gradient,
        stochastic_rounding=optimizer_args.stochastic_rounding, # if optimizer == none
        learning_rate=optimizer_args.learning_rate,
        eps=optimizer_args.eps,
        weight_decay=optimizer_args.weight_decay,
        weight_decay_mode=optimizer_args.weight_decay_mode,
        counter_halflife=optimizer_args.counter_halflife,
        adjustment_iter=optimizer_args.adjustment_iter,
        adjustment_ub=optimizer_args.adjustment_ub,
        learning_rate_mode=optimizer_args.learning_rate_mode,
        grad_sum_decay=optimizer_args.grad_sum_decay,
        tail_id_threshold=optimizer_args.tail_id_threshold,
        is_tail_id_thresh_ratio=optimizer_args.is_tail_id_thresh_ratio,
        # momentum1
        momentum1_dev=momentum1.dev,
        momentum1_uvm=momentum1.uvm,
        momentum1_offsets=momentum1.offsets,
        momentum1_placements=momentum1.placements,
        # momentum2
        # prev_iter
        prev_iter_dev=prev_iter.dev,
        prev_iter_uvm=prev_iter.uvm,
        prev_iter_offsets=prev_iter.offsets,
        prev_iter_placements=prev_iter.placements,
        # row_counter
        row_counter_dev=row_counter.dev,
        row_counter_uvm=row_counter.uvm,
        row_counter_offsets=row_counter.offsets,
        row_counter_placements=row_counter.placements,
        # iter
        iter=iter,
        # max counter
        max_counter=max_counter,
        # total_unique_indices
        output_dtype=common_args.output_dtype,
        is_experimental=common_args.is_experimental,
    )