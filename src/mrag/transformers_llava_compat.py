"""Compatibility shims so LLaVA-NeXT works with newer ``transformers``.

LLaVA language_model files historically import helpers from ``transformers.modeling_utils``.
In recent ``transformers`` (e.g. 5.5+), many of these live only in ``pytorch_utils``, and some
(``find_pruneable_heads_and_indices``, ``prune_conv1d_layer``, ``prune_layer``) were removed
from ``pytorch_utils`` entirely — we re-attach them to ``modeling_utils`` before any ``llava`` import.

Call :func:`ensure_modeling_utils_chunking_compat` (alias of :func:`patch_transformers_modeling_utils_for_llava`)
**before** any ``llava`` import.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Optional, Union

import torch
from torch import nn


def _apply_chunking_to_forward_impl(
    forward_fn: Callable[..., Any],
    chunk_size: int,
    chunk_dim: int,
    *input_tensors: Any,
) -> Any:
    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"
    num_args = len(inspect.signature(forward_fn).parameters)
    if num_args != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args} arguments, but only {len(input_tensors)} input tensors are given"
        )
    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )
        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )
        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        output_chunks = tuple(forward_fn(*c) for c in zip(*input_tensors_chunks))
        return torch.cat(output_chunks, dim=chunk_dim)
    return forward_fn(*input_tensors)


def _find_pruneable_heads_and_indices(
    heads: list[int],
    n_heads: int,
    head_size: int,
    already_pruned_heads: set[int],
) -> tuple[set[int], torch.LongTensor]:
    """Logic aligned with historical ``transformers.pytorch_utils`` (v4.40 era)."""
    mask = torch.ones(n_heads, head_size)
    heads_set = set(heads) - already_pruned_heads
    for head in heads_set:
        head_adj = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head_adj] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask), device=mask.device)[mask].long()
    return heads_set, index


def _prune_linear_layer_impl(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    index = index.to(layer.weight.device)
    w = layer.weight.index_select(dim, index).detach().clone()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.detach().clone()
        else:
            b = layer.bias[index].detach().clone()
    else:
        b = None
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(w.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())  # type: ignore[union-attr]
        new_layer.bias.requires_grad = True
    return new_layer


def _prune_conv1d_layer_impl(conv1d_cls: type, layer: Any, index: torch.LongTensor, dim: int = 1) -> Any:
    index = index.to(layer.weight.device)
    w = layer.weight.index_select(dim, index).detach().clone()
    if dim == 0:
        b = layer.bias.detach().clone()
    else:
        b = layer.bias[index].detach().clone()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = conv1d_cls(new_size[1], new_size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(w.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer


def _prune_layer_impl(conv1d_cls: type, layer: Union[nn.Linear, Any], index: torch.LongTensor, dim: Optional[int] = None):
    if isinstance(layer, nn.Linear):
        return _prune_linear_layer_impl(layer, index, dim=0 if dim is None else dim)
    if isinstance(layer, conv1d_cls):
        return _prune_conv1d_layer_impl(conv1d_cls, layer, index, dim=1 if dim is None else dim)
    raise ValueError(f"Can't prune layer of class {layer.__class__}")


def patch_transformers_modeling_utils_for_llava() -> None:
    import transformers.modeling_utils as mu
    import transformers.pytorch_utils as pu

    # Still present in modern pytorch_utils — copy onto modeling_utils if missing.
    for name in (
        "Conv1D",
        "prune_linear_layer",
        "apply_chunking_to_forward",
        "id_tensor_storage",
        "meshgrid",
    ):
        if getattr(mu, name, None) is None:
            obj = getattr(pu, name, None)
            if obj is not None:
                setattr(mu, name, obj)

    if getattr(mu, "apply_chunking_to_forward", None) is None:
        setattr(mu, "apply_chunking_to_forward", _apply_chunking_to_forward_impl)

    if getattr(mu, "prune_linear_layer", None) is None:
        setattr(mu, "prune_linear_layer", _prune_linear_layer_impl)

    conv1d_cls = getattr(mu, "Conv1D", None)
    if conv1d_cls is None:
        raise RuntimeError("transformers_llava_compat: Conv1D missing from modeling_utils and pytorch_utils")

    if getattr(mu, "find_pruneable_heads_and_indices", None) is None:
        setattr(mu, "find_pruneable_heads_and_indices", _find_pruneable_heads_and_indices)

    if getattr(mu, "prune_conv1d_layer", None) is None:
        setattr(
            mu,
            "prune_conv1d_layer",
            lambda layer, index, dim=1: _prune_conv1d_layer_impl(conv1d_cls, layer, index, dim),
        )

    if getattr(mu, "prune_layer", None) is None:
        setattr(
            mu,
            "prune_layer",
            lambda layer, index, dim=None: _prune_layer_impl(conv1d_cls, layer, index, dim),
        )


def ensure_modeling_utils_chunking_compat() -> None:
    """Backward-compatible name used by benchmark entrypoints."""
    patch_transformers_modeling_utils_for_llava()
