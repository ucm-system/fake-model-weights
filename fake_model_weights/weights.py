"""假权重生成: 按架构生成 (参数名, shape, dtype) 清单 + 流式写 safetensors。

两条路径:
 - ``use_torch=True``: 用 torch 创建随机张量并落盘(最标准);
 - 否则纯 stdlib: 按 dtype bytesize 直接写确定性随机字节(不依赖 torch)。

safetensors 布局: 8 字节小端 header 长度 + JSON header + 数据区;
每个张量 data_offsets 以数据区起点计, 并 8 字节对齐。
"""

from __future__ import annotations

import json
import random
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .layer_plan import layer_types, text_config

# safetensors dtype -> (名字, itemsize)
_DTYPES = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I8": 1,
    "U8": 1,
}
_HEADER_BYTE = 8


def _v(cfg: Dict[str, Any], name: str, default: Any = None) -> Any:
    return cfg.get(name, default)


def _default_dtype(cfg: Dict[str, Any]) -> str:
    td = str(_v(cfg, "torch_dtype", "bfloat16")).lower()
    return {"float32": "F32", "float16": "F16", "bfloat16": "BF16"}.get(td, "BF16")


def build_manifest(
    plan: Dict[str, Any], reduced_config: Dict[str, Any], no_ffn: bool = False
) -> List[Dict[str, Any]]:
    """按 layer_plan + 缩减 config 生成 (name, shape, dtype) 清单。

    自定义架构(full/mla/csa/kda/dsa)的参数名遵循 HF 风格启发式清单;
    llama/qwen 等标准架构使用标准命名,可直接被 vllm --load-format safetensors 读取。
    ``no_ffn=True`` 时跳过 MLP/专家权重,只保留 attention/KV 相关张量
    (纯 attention 层;结构验证用)。
    """
    cfg = text_config(reduced_config)
    model_key = plan["model_key"]
    hidden = int(_v(cfg, "hidden_size", 4096))
    n_heads = int(_v(cfg, "num_attention_heads", 0) or 0)
    n_kv = int(_v(cfg, "num_key_value_heads", n_heads) or n_heads)
    head_dim = int(
        _v(
            cfg,
            "head_dim",
            _v(
                cfg,
                "qk_head_dim",
                _v(_v(cfg, "linear_attn_config", {}), "head_dim", 0) or 0,
            )
            or 0,
        )
    )
    q_lora = int(_v(cfg, "q_lora_rank", 0) or 0)
    kv_lora = int(_v(cfg, "kv_lora_rank", 0) or 0)
    rope_dim = int(_v(cfg, "qk_rope_head_dim", 0) or 0)
    inter = int(_v(cfg, "intermediate_size", 0) or 0)
    moe_inter = int(_v(cfg, "moe_intermediate_size", 0) or inter)
    n_routed = int(_v(cfg, "n_routed_experts", _v(cfg, "num_experts", 0) or 0) or 0)
    n_shared = int(
        _v(cfg, "n_shared_experts", _v(cfg, "num_shared_experts", 1) or 1) or 1
    )
    vocab = int(_v(cfg, "vocab_size", 0) or 0)
    tie = bool(_v(cfg, "tie_word_embeddings", False))
    exp_dtype = {"fp8": "F16", "bf16": "BF16", "float32": "F32"}.get(
        str(_v(cfg, "expert_dtype", "")).lower(), None
    )
    dt = exp_dtype or _default_dtype(cfg)

    manifest: List[Dict[str, Any]] = []

    def add(name: str, *shape: int, dtype: Optional[str] = None) -> None:
        manifest.append(
            {"name": name, "shape": [int(s) for s in shape], "dtype": dtype or dt}
        )

    prefix = "language_model.model" if model_key == "deepseek-v4" else "model"
    L = lambda name: f"{prefix}.{name}"  # noqa: E731

    if vocab and hidden:
        add(L("embed_tokens.weight"), vocab, hidden)
    if vocab and hidden and not tie:
        add(L("lm_head.weight"), vocab, hidden)

    entries = plan.get("layer_plans") or plan.get("layer_plan") or []
    for e in entries:
        i = int(e["index"])
        kind = e["type"]
        p = e.get("params") or {}
        hd = int(p.get("head_dim") or head_dim)
        add(L(f"layers.{i}.input_layernorm.weight"), hidden)

        if kind in ("full", "swa", "dsa") or kind == "mla" and not q_lora:
            if kind == "mla":
                add(L(f"layers.{i}.self_attn.q_proj.weight"), n_kv * hd, hidden)
                add(
                    L(f"layers.{i}.self_attn.kv_a_proj_with_mqa.weight"),
                    (kv_lora or hd) + rope_dim,
                    hidden,
                )
                add(L(f"layers.{i}.self_attn.o_proj.weight"), hidden, n_kv * hd)
            else:
                add(L(f"layers.{i}.self_attn.q_proj.weight"), n_heads * hd, hidden)
                add(L(f"layers.{i}.self_attn.k_proj.weight"), n_kv * hd, hidden)
                add(L(f"layers.{i}.self_attn.v_proj.weight"), n_kv * hd, hidden)
                add(L(f"layers.{i}.self_attn.o_proj.weight"), hidden, n_heads * hd)
        elif kind == "mla":  # q_lora 压缩注意力
            add(L(f"layers.{i}.self_attn.q_proj.weight"), q_lora, hidden)
            add(
                L(f"layers.{i}.self_attn.kv_a_proj_with_mqa.weight"),
                (kv_lora or hd) + rope_dim,
                hidden,
            )
            add(
                L(f"layers.{i}.self_attn.o_proj.weight"), hidden, (kv_lora or hd) * n_kv
            )
        elif kind in ("csa_c4", "csa_c128"):
            add(L(f"layers.{i}.self_attn.q_proj.weight"), n_heads * hd, hidden)
            add(L(f"layers.{i}.self_attn.o_proj.weight"), hidden, n_heads * hd)
        elif kind == "kda":
            add(L(f"layers.{i}.linear_attn.lambda_proj.weight"), hidden, hidden)
            add(L(f"layers.{i}.linear_attn.q_proj.weight"), hidden, hidden)
            add(L(f"layers.{i}.linear_attn.k_proj.weight"), hidden, hidden)
            add(L(f"layers.{i}.linear_attn.v_proj.weight"), hidden, hidden)
            add(L(f"layers.{i}.linear_attn.o_proj.weight"), hidden, hidden)

        # MLP(MoE 或 dense);--no-ffn 时跳过(纯 attention 层)。
        if not no_ffn and n_routed and moe_inter:
            add(L(f"layers.{i}.mlp.gate.weight"), n_routed, hidden)
            add(
                L(f"layers.{i}.mlp.experts.gate_up_proj.weight"),
                n_routed,
                2 * moe_inter,
                hidden,
            )
            add(
                L(f"layers.{i}.mlp.experts.down_proj.weight"),
                n_routed,
                moe_inter,
                hidden,
            )
            if n_shared:
                add(
                    L(f"layers.{i}.mlp.shared_experts.gate_up_proj.weight"),
                    n_shared * 2 * moe_inter,
                    hidden,
                )
                add(
                    L(f"layers.{i}.mlp.shared_experts.down_proj.weight"),
                    n_shared * moe_inter,
                    hidden,
                )
        elif not no_ffn:
            inter_ = inter or moe_inter or 256
            add(L(f"layers.{i}.mlp.gate_proj.weight"), inter_, hidden)
            add(L(f"layers.{i}.mlp.up_proj.weight"), inter_, hidden)
            add(L(f"layers.{i}.mlp.down_proj.weight"), hidden, inter_)

        add(L(f"layers.{i}.post_attention_layernorm.weight"), hidden)

    if hidden:
        add(L("norm.weight"), hidden)
    return manifest


# ------------------------------- 写 safetensors -----------------------------


def _write_tensor_bytes(f, writer, nbytes: int) -> bytes:
    if writer is not None:
        f.write(writer(nbytes))
        return b""
    return f.read(nbytes)


def write_safetensors(
    manifest: List[Dict[str, Any]],
    out_dir: Path,
    seed: int = 0,
    use_torch: Optional[bool] = None,
    shard_gb: float = 2.0,
    prefix: str = "model",
) -> List[str]:
    """把 manifest 流式写入 ``<prefix>-NNNNN-of-NNNNN.safetensors``(+索引)。

    safetensors 布局: 文件 = [8B header 长度][JSON header][数据区],header 先行;
    data_offsets 相对数据区起点,begin/end 均 8 字节对齐。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    max_bytes = int(shard_gb * (1 << 30))

    def torch_ok() -> bool:
        if use_torch is None:
            return False
        try:
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    with_torch = torch_ok()

    def stream_tensor(fh, item: Dict[str, Any]) -> None:
        nbytes = _size_of(item)
        if with_torch:
            import torch

            torch.manual_seed(rng.randrange(0, 2**31))
            t = torch.empty(
                [int(s) for s in item["shape"]], dtype=_torch_dtype(item["dtype"])
            )
            t.normal_()
            fh.write(t.numpy().tobytes())
        else:
            left = nbytes
            while left:
                chunk = min(left, 1 << 20)
                fh.write(rng.randbytes(chunk))
                left -= chunk

    # 1) 按字节预算切分片(不拆单个张量)。
    shards: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_bytes = 0
    for item in manifest:
        sz = _size_of(item)
        if cur and cur_bytes + sz > max_bytes:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(item)
        cur_bytes += (sz + 7) & ~7
    if cur:
        shards.append(cur)

    # 2) 写每个分片: header 先行,再顺序写数据。
    files: List[str] = []
    weight_map: Dict[str, str] = {}
    n_shards = len(shards)
    for i, items in enumerate(shards):
        header: Dict[str, Any] = {"__metadata__": {"format": "pt"}}
        offset = 0
        for item in items:
            size = _size_of(item)
            begin, end = offset, offset + size
            header[item["name"]] = {
                "dtype": item["dtype"],
                "shape": [int(s) for s in item["shape"]],
                "data_offsets": [begin, end],
            }
            offset = end + ((-end) % 8)
        raw = json.dumps(header, separators=(",", ":")).encode()
        assert len(raw) < 2**63
        fname = f"{prefix}-{i:05d}-of-{n_shards:05d}.safetensors"
        with open(out_dir / fname, "wb") as fh:
            fh.write(struct.pack("<Q", len(raw)))
            fh.write(raw)
            for item in items:
                stream_tensor(fh, item)
        files.append(fname)
        for item in items:
            weight_map[item["name"]] = fname

    # 3) safetensors index。
    (out_dir / f"{prefix}.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"format": "pt"},
                "weight_map": dict(sorted(weight_map.items())),
            },
            indent=2,
        )
    )
    return files


def _size_of(item: Dict[str, Any]) -> int:
    n = 1
    for s in item["shape"]:
        n *= int(s)
    return n * _DTYPES[item["dtype"]]


def _torch_dtype(dtype: str):
    import torch

    return {
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I8": torch.int8,
        "U8": torch.uint8,
    }[dtype]


def read_safetensors_header(path: Path) -> Tuple[Dict[str, Any], int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def verify_safetensors(
    out_dir: Path, use_torch: Optional[bool] = None
) -> Dict[str, Any]:
    """读回所有分片 header + 数据长度校验;可选 torch 加载校验。"""
    idx = json.loads((out_dir / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    total = 0
    for fname in sorted(set(wm.values())):
        h, data_start = read_safetensors_header(out_dir / fname)
        total += (out_dir / fname).stat().st_size
        for k, meta in h.items():
            if k == "__metadata__":
                continue
            b, e = meta["data_offsets"]
            assert e > b and e - b > 0
    if use_torch:
        import torch
        from safetensors.torch import load_file

        for fname in sorted(set(wm.values())):
            tensors = load_file(str(out_dir / fname))
            assert set(tensors) == {k for k in wm if wm[k] == fname}
    return {"tensors": len(wm), "files": sorted(set(wm.values()))}


__all__ = [
    "build_manifest",
    "write_safetensors",
    "read_safetensors_header",
    "verify_safetensors",
]
