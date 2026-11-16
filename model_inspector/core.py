"""Turning any `torch.nn.Module` into a procedural one. Depends on torch only."""

import contextlib

import torch

from .tensor import ProceduralTensor, _mix


@contextlib.contextmanager
def meta_parameters():
    """Inside this context, modules create their parameters on the meta device (no memory) and their buffers as
    usual, so buffers computed at init (e.g. RoPE frequencies) keep their real values."""
    register = torch.nn.Module.register_parameter

    def on_meta(module, name, param):
        if param is not None and not isinstance(param, ProceduralTensor):
            param = torch.nn.Parameter(param.to("meta"), requires_grad=param.requires_grad)
        register(module, name, param)

    torch.nn.Module.register_parameter = on_meta
    try:
        yield
    finally:
        torch.nn.Module.register_parameter = register


def make_procedural(module: torch.nn.Module, seed: int = 0, std: float = 0.02, device="cpu") -> torch.nn.Module:
    """Replaces every parameter of `module` by a `ProceduralTensor` of the same shape and dtype, in place, and
    returns `module`. Matrices get `randn * std`, 1-D weights (norms) ones and biases zeros. Parameters shared by
    several modules (tied weights) stay shared. `device` is a device, or a callable `name -> device`."""
    new = {}
    for index, (name, param) in enumerate(module.named_parameters()):
        leaf = name.rsplit(".", 1)[-1]
        kind = "randn" if param.dim() >= 2 else ("zeros" if leaf == "bias" else "ones")
        dev = device(name) if callable(device) else device
        new[id(param)] = torch.nn.Parameter(
            ProceduralTensor(param.shape, param.dtype, _mix(seed, index), kind, std, dev), requires_grad=False
        )
    for sub in module.modules():
        for leaf, param in list(sub._parameters.items()):
            if param is not None:
                sub._parameters[leaf] = new[id(param)]
    return module


def is_procedural(module: torch.nn.Module) -> bool:
    return any(isinstance(p, ProceduralTensor) for p in module.parameters())
