"""Run and inspect any model without its weights: procedural tensors generated on the fly, for torch modules."""

from .core import is_procedural, make_procedural, meta_parameters
from .tensor import ProceduralTensor

__all__ = ["ProceduralTensor", "dummy_model", "is_procedural", "make_procedural", "meta_parameters"]
__version__ = "0.1.0"


def dummy_model(*args, **kwargs):
    from .hf import dummy_model

    return dummy_model(*args, **kwargs)


dummy_model.__doc__ = "See `model_inspector.hf.dummy_model` (needs transformers)."
