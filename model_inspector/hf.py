"""Hugging Face models from their config alone. Needs `transformers` (`pip install model_inspector[hf]`)."""

import contextlib

from .core import make_procedural, meta_parameters


def dummy_model(name_or_config, seed: int = 0, device="cpu", auto_class=None, **config_kwargs):
    """The model of a Hub repo (or a local path, or a config), with procedural weights: only `config.json` is
    downloaded, the module tree, names, shapes and dtypes are the real ones and forward / generate work, in a few MB
    of memory whatever the model size. A given (config, seed) always gives the same model."""
    import transformers

    config = name_or_config
    if not isinstance(config, transformers.PretrainedConfig):
        config = transformers.AutoConfig.from_pretrained(name_or_config, **config_kwargs)
    # Dummy weights are never quantized: the model is built in its unquantized dtype
    if getattr(config, "quantization_config", None) is not None:
        del config.quantization_config
    if auto_class is None:
        mapping = transformers.models.auto.modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        auto_class = transformers.AutoModelForCausalLM if config.model_type in mapping else transformers.AutoModel

    with meta_parameters():
        model = auto_class.from_config(config)
    model.tie_weights()  # on meta first, so tied parameters share one procedural tensor
    make_procedural(model, seed, getattr(config, "initializer_range", None) or 0.02, device)
    model.to(device if not callable(device) else "cpu").eval()  # buffers; parameters keep their recipe
    # Fused MoE experts: the per-expert path generates only the experts tokens are routed to
    if hasattr(model, "set_experts_implementation"):
        with contextlib.suppress(Exception):
            model.set_experts_implementation("eager")
    return model
