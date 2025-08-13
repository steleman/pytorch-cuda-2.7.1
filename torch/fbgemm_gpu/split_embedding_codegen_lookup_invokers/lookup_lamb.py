#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import warnings
from .lookup_args import *


def invoke(
    common_args: CommonArgs,
    optimizer_args: OptimizerArgs,
    momentum1: Momentum,
    momentum2: Momentum,
    iter: int,
) -> torch.Tensor:
    # By design, the warning only shows up once
    warnings.warn(
        f"""\033[93m
        [FBGEMM_GPU] NOTE: The training optimizer 'lamb' is marked as
        EXPERIMENTAL and thus not optimized, in order to reduce code compilation
        times and build sizes!
        \033[0m"""
    )
    vbe_metadata = common_args.vbe_metadata
    return torch.ops.fbgemm.split_embedding_codegen_lookup_lamb_function(
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
        beta1=optimizer_args.beta1,
        beta2=optimizer_args.beta2,
        weight_decay=optimizer_args.weight_decay,
        # momentum1
        momentum1_dev=momentum1.dev,
        momentum1_uvm=momentum1.uvm,
        momentum1_offsets=momentum1.offsets,
        momentum1_placements=momentum1.placements,
        # momentum2
        momentum2_dev=momentum2.dev,
        momentum2_uvm=momentum2.uvm,
        momentum2_offsets=momentum2.offsets,
        momentum2_placements=momentum2.placements,
        # prev_iter
        # row_counter
        # iter
        iter=iter,
        # max counter
        # total_unique_indices
        output_dtype=common_args.output_dtype,
        is_experimental=common_args.is_experimental,
    )