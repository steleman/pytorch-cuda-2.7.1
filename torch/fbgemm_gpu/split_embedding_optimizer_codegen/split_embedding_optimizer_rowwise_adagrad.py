#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .optimizer_args import *
from typing import Optional, List, Tuple
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    construct_split_state,
    WeightDecayMode,
    EmbeddingLocation,
    ComputeDevice,
    apply_split_helper,
)
from torch.optim.optimizer import Optimizer
import logging


# Currently the template is implemented specifically for rowwise_adagrad.  It
# might or might not be applicable for other optimizers
# TODO: Add support for other optimizers
class SplitEmbeddingRowwiseAdagrad(Optimizer):
    def __init__(
        self,
        params: SplitEmbeddingOptimizerParams,
        embedding_args: SplitEmbeddingArgs,
        embedding_specs: List[Tuple[int, int, EmbeddingLocation, ComputeDevice]],
        feature_table_map: Optional[List[int]] = None,
        stochastic_rounding: bool = True,
        learning_rate: float = 0.01,
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        weight_decay_mode: WeightDecayMode = WeightDecayMode.NONE,
    ) -> None:

        # TODO: Add arg checkers
        defaults = dict(
            learning_rate=learning_rate,
            eps=eps,
            weight_decay=weight_decay,
            weight_decay_mode=weight_decay_mode,
        )

        super().__init__([params.weights_dev], defaults)

        # Assume rowwise_adagrad for now
        for group in self.param_groups:
            for p in group["params"]:
                if p is params.weights_dev:
                    state = self.state[p]
                    # Get device
                    device = p.device
                    assert p.is_cuda, "SplitEmbeddingRowwiseAdagrad only support GPU tensors"
                    # Dummy tensors for UVM and LXU cache
                    # TODO: support UVM and LXU cache
                    state["dummy_weights_uvm"] = torch.empty(
                            0,
                            device=device,
                            dtype=params.weights_dev.dtype)
                    state["dummy_lxu_cache_weights"] = torch.zeros(
                            0,
                            0,
                            device=device,
                            dtype=params.weights_dev.dtype)

                    
                    
                    
                    rowwise = True
                    
                    
                    # Move this to the global state (if possible)
                    # Construct momentum1 state
                    momentum1_state = construct_split_state(
                            embedding_specs,
                            rowwise=rowwise,
                            cacheable=False)

                    def set_state(name: str, tensor: torch.Tensor) -> None:
                        state[name] = tensor

                    apply_split_helper(
                        persistent_state_fn=set_state,
                        set_attr_fn=set_state,
                        current_device=device,
                        use_cpu=False,
                        feature_table_map=feature_table_map,
                        split=momentum1_state,
                        prefix="momentum1",
                        dtype=torch.float32,
                    )
                    
                else:
                    raise NotImplementedError(
                            "SplitEmbeddingRowwiseAdagrad only supports weights_dev update")

        self.params = params
        self.embedding_args = embedding_args
        self.stochastic_rounding = stochastic_rounding
        self.learning_rate = learning_rate
        self.eps = eps
        self.weight_decay = weight_decay
        self.weight_decay_mode = weight_decay_mode


    def step(self, closure=None) -> torch.Tensor:
        for group in self.param_groups:
            for p in group["params"]:
                if p is self.params.weights_dev:
                    state = self.state[p]
                    sparse_grad_dev_weights = p.grad
                    # Dummy tensors for UVM and LXU cache
                    dummy_weights_uvm = state["dummy_weights_uvm"]
                    dummy_lxu_cache_weights = state["dummy_lxu_cache_weights"]
                    
                    # momentum1 state
                    momentum1_dev = state["momentum1_dev"]
                    momentum1_uvm = state["momentum1_uvm"]
                    momentum1_offsets = state["momentum1_offsets"]
                    momentum1_placements = state["momentum1_placements"]
                    
                    assert sparse_grad_dev_weights.is_sparse, "sparse_grad_dev_weights must be sparse"
                else:
                    raise NotImplementedError(
                            "SplitEmbeddingRowwiseAdagrad only supports weights_dev update")

        # Call rowwise_adagrad
        torch.ops.fbgemm.split_embedding_rowwise_adagrad_update(
                dev_weights=self.params.weights_dev,
                uvm_weights=dummy_weights_uvm,
                lxu_cache_weights=dummy_lxu_cache_weights,
                grad_dev_weights=sparse_grad_dev_weights._values(),
                grad_dev_indices=sparse_grad_dev_weights._indices(),
                weights_placement=self.embedding_args.weights_placements,
                weights_offsets=self.embedding_args.weights_offsets,
                max_D=self.embedding_args.max_D,
                stochastic_rounding=self.stochastic_rounding,
                learning_rate=self.learning_rate,
                eps=self.eps,
                weight_decay=self.weight_decay,
                weight_decay_mode=self.weight_decay_mode,
                
                # momentum1
                momentum1_dev=momentum1_dev,
                momentum1_uvm=momentum1_uvm,
                momentum1_placements=momentum1_placements,
                momentum1_offsets=momentum1_offsets,
                
        )