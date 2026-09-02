"""fake-model-weights 单测(纯 stdlib;pytest)。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_model_weights import (
    cli,
)
from fake_model_weights import layer_plan as lp
from fake_model_weights import reduce as rd
from fake_model_weights import resolve as rv
from fake_model_weights import weights as wd

DATA = Path(__file__).resolve().parent / "data"


def _load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


class ResolveTest(unittest.TestCase):
    def test_split_prefix(self):
        self.assertEqual(rv._split_prefix("hf:owner/repo"), ("hf", "owner/repo"))
        self.assertEqual(rv._split_prefix("ms:owner/repo"), ("ms", "owner/repo"))
        self.assertEqual(rv._split_prefix("owner/repo"), (None, "owner/repo"))

    def test_local_detection_and_resolve(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "config.json").write_text("{}")
            self.assertTrue(rv._is_local_model(p))
            out = rv.resolve_model(d)
            self.assertEqual(out["source"], "local")
            self.assertEqual(out["dir"], str(p.resolve()))

    def test_local_missing_config_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                rv.resolve_model(d, source="local")

    def test_known_official_repos(self):
        self.assertIn("deepseek-v4", rv.KNOWN_OFFICIAL_REPOS)
        self.assertEqual(
            rv.KNOWN_OFFICIAL_REPOS["deepseek-v4"], "deepseek-ai/DeepSeek-V4-Flash-0731"
        )


class LayerPlanTest(unittest.TestCase):
    def test_dsv4_pattern(self):
        cfg = _load("deepseek-v4.json")
        kinds = lp.layer_types("deepseek-v4", cfg)
        self.assertEqual(
            kinds[:8],
            [
                "full",
                "full",
                "csa_c4",
                "csa_c128",
                "csa_c4",
                "csa_c128",
                "csa_c4",
                "csa_c128",
            ],
        )
        self.assertEqual(
            lp.type_string("deepseek-v4", cfg, 8),
            "full,full,csa_c4,csa_c128,csa_c4,csa_c128,csa_c4,csa_c128",
        )

    def test_k3_pattern(self):
        cfg = _load("kimi-k3.json")
        kinds = lp.layer_types("kimi-k3", cfg)
        self.assertEqual(len(kinds), 93)
        self.assertEqual(kinds[0], "mla")  # 0 层默认 mla
        self.assertEqual(kinds[1:4], ["kda", "kda", "kda"])
        self.assertEqual(kinds[4], "mla")  # full_attn_layers 含 4
        self.assertEqual(
            lp.type_string("kimi-k3", cfg, 8), "mla,kda,kda,kda,mla,kda,kda,kda"
        )

    def test_glm_pattern(self):
        cfg = _load("glm-5.3.json")
        kinds = lp.layer_types("glm-5.3", cfg)
        self.assertEqual(len(kinds), 45)
        self.assertEqual(kinds[:3], ["kda", "kda", "kda"])
        self.assertEqual(kinds[3], "dsa")  # deepseek_sparse_attention
        self.assertEqual(
            lp.type_string("glm-5.3", cfg, 8), "kda,kda,kda,dsa,kda,kda,kda,dsa"
        )

    def test_dsv4_kv_groups(self):
        cfg = _load("deepseek-v4.json")
        plan = lp.layer_plan("deepseek-v4", cfg)
        names = {g["name"] for g in plan["kv_groups"]}
        self.assertGreaterEqual(names, {"full", "csa_c4", "csa_c128", "swa", "indexer"})
        by = {g["name"]: g for g in plan["kv_groups"]}
        self.assertEqual(by["indexer"]["kind"], lp.KIND_SIDECAR)
        self.assertEqual(by["swa"]["sliding_window"], 128)

    def test_qwen36_pattern(self):
        cfg = _load("qwen3.6-27b.json")
        self.assertEqual(cli._detect_model_key(cfg), "qwen3_5")
        kinds = lp.layer_types("qwen3_5", cfg)
        self.assertEqual(len(kinds), 64)
        # linear_attention x3 + full_attention x1 循环
        self.assertEqual(
            kinds[:8], ["kda", "kda", "kda", "full", "kda", "kda", "kda", "full"]
        )
        self.assertEqual(
            lp.type_string("qwen3_5", cfg, 8),
            "kda,kda,kda,full,kda,kda,kda,full",
        )
        names = {g["name"] for g in lp.kv_group_plan("qwen3_5", cfg)}
        self.assertGreaterEqual(names, {"full", "kda"})

    def test_generic_all_full(self):
        cfg = {
            "model_type": "llama",
            "num_hidden_layers": 4,
            "hidden_size": 64,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "intermediate_size": 128,
        }
        self.assertEqual(
            lp.layer_types("generic", cfg), ["full", "full", "full", "full"]
        )


class ReduceTest(unittest.TestCase):
    def test_layer_truncation_preserves_prefix(self):
        cfg = _load("deepseek-v4.json")
        reduced = rd.reduce_config("deepseek-v4", cfg, 8)
        self.assertEqual(reduced["num_hidden_layers"], 8)
        self.assertEqual(reduced["compress_ratios"], cfg["compress_ratios"][:8])

    def test_kv_shape_snapshot_preserved(self):
        for key, mk in (
            ("deepseek-v4.json", "deepseek-v4"),
            ("kimi-k3.json", "kimi-k3"),
            ("glm-5.3.json", "glm-5.3"),
        ):
            with self.subTest(model=key):
                cfg = _load(key)
                reduced = rd.reduce_config(mk, cfg, 8)
                self.assertEqual(
                    rd.kv_shape_snapshot(mk, cfg), rd.kv_shape_snapshot(mk, reduced)
                )

    def test_quantization_and_dspark_stripped(self):
        cfg = _load("deepseek-v4.json")
        self.assertIn("quantization_config", cfg)
        reduced = rd.reduce_config("deepseek-v4", cfg, 8)
        self.assertNotIn("quantization_config", reduced)
        self.assertNotIn("dspark_target_layer_ids", reduced)

    def test_ffn_clamp(self):
        cfg = _load("deepseek-v4.json")
        reduced = rd.reduce_config("deepseek-v4", cfg, 8)
        self.assertLessEqual(
            reduced["num_experts_per_tok"], max(1, reduced["n_routed_experts"])
        )


class WeightsTest(unittest.TestCase):
    def setUp(self):
        self.cfg = _load("deepseek-v4.json")
        self.plan = lp.layer_plan("deepseek-v4", self.cfg)
        self.reduced = rd.reduce_config("deepseek-v4", self.cfg, 4)

    def test_manifest_nonempty_and_shape_sane(self):
        m = wd.build_manifest(self.plan, self.reduced)
        self.assertTrue(m)
        for item in m:
            self.assertGreater(item["shape"][0], 0)
            self.assertIn(item["dtype"], ("F32", "F16", "BF16", "I8", "U8"))

    def test_no_ffn_manifest_excludes_mlp(self):
        full = wd.build_manifest(self.plan, self.reduced)
        no_ffn = wd.build_manifest(self.plan, self.reduced, no_ffn=True)
        self.assertLess(len(no_ffn), len(full))
        self.assertTrue(any("mlp" in it["name"] for it in full))
        self.assertFalse(any("mlp" in it["name"] for it in no_ffn))
        # attention/KV 相关张量仍然齐全
        self.assertTrue(any("self_attn" in it["name"] for it in no_ffn))

    def test_write_and_readback_stdlib(self):
        with tempfile.TemporaryDirectory() as d:
            m = wd.build_manifest(self.plan, self.reduced)
            files = wd.write_safetensors(m, Path(d), seed=7)
            self.assertTrue(files)
            idx = json.loads((Path(d) / "model.safetensors.index.json").read_text())
            self.assertEqual(len(idx["weight_map"]), len(m))
            for fname in set(idx["weight_map"].values()):
                h, start = wd.read_safetensors_header(Path(d) / fname)
                self.assertEqual(h["__metadata__"]["format"], "pt")
                for meta in h.values():
                    if meta == h.get("__metadata__"):
                        continue
                    b, e = meta["data_offsets"]
                    self.assertEqual(b % 8, 0)
                    self.assertLess(b, e)
            report = wd.verify_safetensors(Path(d))
            self.assertEqual(report["tensors"], len(m))


class CliTest(unittest.TestCase):
    def test_local_config_only(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            src.mkdir()
            reduced = rd.reduce_config("deepseek-v4", _load("deepseek-v4.json"), 8)
            (src / "config.json").write_text(json.dumps(reduced))
            out = Path(d) / "out"
            code = cli.main([str(src), "--layers", "8", "--out", str(out)])
            self.assertEqual(code, 0)
            self.assertTrue((out / "config.json").exists())
            self.assertTrue((out / "layer_plan.json").exists())
            plan = json.loads((out / "layer_plan.json").read_text())
            self.assertEqual(plan["layers"], 8)
            self.assertIn("kv_groups", plan)

    def test_local_with_safetensors_weights(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            src.mkdir()
            reduced = rd.reduce_config("deepseek-v4", _load("deepseek-v4.json"), 4)
            (src / "config.json").write_text(json.dumps(reduced))
            out = Path(d) / "out"
            code = cli.main(
                [
                    str(src),
                    "--layers",
                    "4",
                    "--out",
                    str(out),
                    "--weights",
                    "safetensors",
                    "--seed",
                    "1",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((out / "model.safetensors.index.json").exists())
            idx = json.loads((out / "model.safetensors.index.json").read_text())
            self.assertTrue(idx["weight_map"])


if __name__ == "__main__":
    unittest.main()
