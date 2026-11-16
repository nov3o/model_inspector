import unittest

import torch

from model_inspector import ProceduralTensor, is_procedural, make_procedural, meta_parameters


def mlp():
    return torch.nn.Sequential(
        torch.nn.Embedding(1000, 64), torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.LayerNorm(256),
        torch.nn.Linear(256, 1000),
    )  # fmt: skip


class TorchOnlyTest(unittest.TestCase):
    def test_no_memory_and_same_structure(self):
        with meta_parameters():
            m = mlp()
        self.assertTrue(all(p.is_meta for p in m.parameters()))
        make_procedural(m)
        self.assertTrue(is_procedural(m))
        self.assertEqual([n for n, _ in m.named_parameters()], [n for n, _ in mlp().named_parameters()])

    def test_forward_is_deterministic(self):
        x = torch.randint(0, 1000, (2, 7))
        a, b = make_procedural(mlp(), seed=1), make_procedural(mlp(), seed=1)
        c = make_procedural(mlp(), seed=2)
        self.assertEqual(a(x).shape, (2, 7, 1000))
        torch.testing.assert_close(a(x), b(x))
        self.assertFalse(torch.allclose(a(x), c(x)))

    def test_norms_and_biases(self):
        m = make_procedural(mlp())
        torch.testing.assert_close(m[3].weight.materialize(), torch.ones(256))
        torch.testing.assert_close(m[1].bias.materialize(), torch.zeros(256))

    def test_slices_matmul_embedding_match_the_full_tensor(self):
        w = ProceduralTensor((300, 40), torch.float32, seed=7)
        full = w.materialize()
        torch.testing.assert_close(w[5], full[5])
        torch.testing.assert_close(w[17:90], full[17:90])
        x = torch.randn(3, 40)
        torch.testing.assert_close(x @ w.t(), x @ full.t())
        idx = torch.tensor([[0, 299, 17]])
        torch.testing.assert_close(torch.nn.functional.embedding(idx, w), full[idx])

    def test_experts_slice(self):
        w = ProceduralTensor((8, 16, 32), torch.float32, seed=3)
        torch.testing.assert_close(w[5], w.materialize()[5])

    def test_tied_weights_stay_tied(self):
        m = mlp()
        m[4].weight = torch.nn.Parameter(torch.empty(1000, 64))
        m[0].weight = m[4].weight
        make_procedural(m)
        self.assertIs(m[0].weight, m[4].weight)

    def test_read_only(self):
        m = make_procedural(mlp())
        with self.assertRaises(RuntimeError):
            m[1].weight.add_(1)


try:
    import transformers  # noqa: F401
except ImportError:
    transformers = None


@unittest.skipIf(transformers is None, "needs transformers")
class HuggingFaceTest(unittest.TestCase):
    def tiny(self, cls, **kw):
        base = dict(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2)  # fmt: skip
        return cls(**{**base, **kw})

    def test_llama_matches_real_structure_and_generates(self):
        from transformers import AutoModelForCausalLM, LlamaConfig

        from model_inspector import dummy_model

        config = self.tiny(LlamaConfig)
        real = AutoModelForCausalLM.from_config(config)
        dummy = dummy_model(config, seed=0)
        self.assertEqual(
            [(n, p.shape, p.dtype) for n, p in real.named_parameters()],
            [(n, p.shape, p.dtype) for n, p in dummy.named_parameters()],
        )
        for (n, b), (_, rb) in zip(dummy.named_buffers(), real.named_buffers()):
            torch.testing.assert_close(b, rb, msg=n)  # e.g. RoPE frequencies are real
        ids = torch.tensor([[1, 2, 3, 4]])
        out1 = dummy.generate(ids, max_new_tokens=5, do_sample=False)
        out2 = dummy_model(config, seed=0).generate(ids, max_new_tokens=5, do_sample=False)
        self.assertEqual(out1.shape, (1, 9))
        torch.testing.assert_close(out1, out2)

    def test_moe(self):
        from transformers import Qwen3MoeConfig

        from model_inspector import dummy_model

        config = self.tiny(Qwen3MoeConfig, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32)
        logits = dummy_model(config)(torch.tensor([[5, 6, 7]])).logits
        self.assertEqual(logits.shape, (1, 3, 300))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
