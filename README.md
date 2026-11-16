# model_inspector

Run and inspect any model without its weights. Parameters are replaced by procedural tensors: a shape, a dtype, a
device and a seed, with values generated only when an op reads them, and dropped right after. A model keeps its real
module tree, names, shapes and dtypes, and forward and generate work, in about a GB of memory whatever its size.

```bash
pip install "model_inspector[hf] @ git+https://github.com/nov3o/model_inspector"
```

```python
from model_inspector import dummy_model

model = dummy_model("Qwen/Qwen3-235B-A22B")   # downloads config.json only
print(model)                                  # the real architecture
model.generate(input_ids, max_new_tokens=8)   # runs (slowly: weights are regenerated as they are read)
```

Measured on one CPU core (Jean Zay login node): Qwen3-235B-A22B builds in 15 s with a peak RSS of 0.9 GB;
Qwen3-30B-A3B builds in 15 s and generates in 1.2 GB. The same weights in bf16 would take 470 GB and 61 GB.

Useful for testing loading, sharding, quantization, hooks or export code on models too large to allocate, and for
looking at an architecture from its config alone.

## Any torch module

The core depends on torch only, and works on any `nn.Module`:

```python
import torch
from model_inspector import make_procedural, meta_parameters

with meta_parameters():          # parameters on the meta device, buffers as usual
    model = MyModel(huge_config)
make_procedural(model, seed=0)   # matrices randn * std, norms ones, biases zeros; tied weights stay tied
```

`ProceduralTensor` is a `torch.Tensor` subclass. Each block of 16 rows of a matrix has its own seed, so `w[i]` and
`w[a:b]` generate only the rows they need; `x @ W.T` and embedding lookups go chunk by chunk, so a whole matrix is
never in memory; fused MoE experts are generated per expert. A given (model, seed) is always the same model. The
weights are read-only: in-place ops raise.

## Layout

| file | depends on | what |
|---|---|---|
| `model_inspector/tensor.py` | torch | `ProceduralTensor` |
| `model_inspector/core.py` | torch | `meta_parameters`, `make_procedural` |
| `model_inspector/hf.py` | transformers | `dummy_model(repo or config)` |

No compiled code: values come from `torch.randn` with a seeded generator, so generation runs at torch's speed on any
device.

## Tests

```bash
python3 -m unittest discover tests
```

## License

MIT
