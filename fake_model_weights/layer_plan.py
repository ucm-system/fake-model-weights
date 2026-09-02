"""层分布解析 + 注意力组规划(模型/引擎视角,与缓存系统解耦)。

按官方 config 把每一层归为注意力/状态种类(full / mla / csa_c4 / csa_c128 /
swa / kda / dsa …),并推导"注意力组"(与 vllm 的 KVCacheConfig 分组同构):
组名、层索引、块大小、压缩比/窗口/索引器等引擎参数。

KV 形状字段纪律: hidden/heads/head_dim/lora ranks/sliding_window/
compress_ratios/indexer 等只读不改,保证 vllm 的 KVCacheConfig 分组与真实
模型一致。

注意: chain/snapshot/sidecar、seed、storage_block_size 是 **UCM 缓存系统的
规格表语义**,不属于模型描述;如需这些行,请显式使用 ``views_ucm.py``
(``--ucm-view``),本模块保持中立。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ------------------------------ 层分类 -------------------------------------


def text_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("text_config") or cfg


def _v(cfg: Dict[str, Any], name: str, default: Any = None) -> Any:
    return cfg.get(name, default)


def layer_types(model_key: str, cfg: Dict[str, Any]) -> List[str]:
    """逐层注意力种类(按官方 config 字段)。未知架构一律 full。"""
    c = text_config(cfg)
    if model_key == "deepseek-v4" or "compress_ratios" in c:
        ratios = c.get("compress_ratios") or []
        kinds: List[str] = []
        for r in ratios:
            r = int(r)
            kinds.append(
                "full"
                if r == 0
                else ("csa_c4" if r == 4 else ("csa_c128" if r == 128 else "full"))
            )
        return kinds
    lac = _v(c, "linear_attn_config") or {}
    n = int(_v(c, "num_hidden_layers", 0) or 0)

    # Kimi K3: full_attn_layers/kda_layers 集合(第 0 层默认 mla)。
    if model_key == "kimi-k3":
        full_idx = {int(i) for i in (lac.get("full_attn_layers") or [])}
        kda_idx = {int(i) for i in (lac.get("kda_layers") or [])}
        return [
            "mla" if i in full_idx else ("kda" if i in kda_idx else "mla")
            for i in range(n)
        ]

    # GLM-5.3 / 通用: layer_types 逐层字符串。
    if "layer_types" in c and isinstance(c["layer_types"], list):
        kinds = []
        for t in c["layer_types"]:
            t = str(t).lower()
            if "linear" in t or "kda" in t or "mamba" in t or "gated" in t:
                kinds.append("kda")
            elif "dsa" in t or "sparse" in t:
                kinds.append("dsa")
            elif "mla" in t:
                kinds.append("mla")
            elif "full" in t or "attention" in t or "attn" in t or "dense" in t:
                kinds.append("full")
            else:
                kinds.append("full")
        return kinds

    # 兜底: 只有 full_attn_layers(其余 kda);完全没有线性注意力配置则为全 full。
    if not lac:
        return ["full"] * n
    full_idx = {int(i) for i in (lac.get("full_attn_layers") or [])}
    return ["mla" if i in full_idx else "kda" for i in range(n)]


def type_string(model_key: str, cfg: Dict[str, Any], n: Optional[int] = None) -> str:
    kinds = layer_types(model_key, cfg)
    if n is not None:
        kinds = kinds[:n]
    return ",".join(kinds)


# ------------------------------ 注意力组规划 --------------------------------


def _estimate_per_token_bytes(c: Dict[str, Any], kind: str) -> int:
    hidden = int(_v(c, "hidden_size", 0) or 0)
    n_kv = int(_v(c, "num_key_value_heads", _v(c, "num_attention_heads", 0) or 0) or 0)
    head = int(_v(c, "head_dim", 0) or 0)
    if kind in ("mla", "csa_c4", "csa_c128", "full", "swa", "dsa") and n_kv and head:
        return 2 * n_kv * head * 2  # K+V,bf16
    if kind == "kda":
        return 2 * int(_v(c, "qk_head_dim", _v(c, "head_dim", 0) or 0)) * 2
    return 0


def kv_group_plan(model_key: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """引擎/模型视角的注意力组(与 vllm KVCacheConfig 分组同构)。

    每组只含引擎事实: 组名(类型)、层索引、块大小、压缩比/窗口/索引器等参数。
    不含任何缓存系统的规格表语义(chain/snapshot/sidecar、seed、存储块等,
    见 ``views_ucm``)。
    """
    c = text_config(cfg)
    kinds = layer_types(model_key, cfg)
    groups: Dict[str, Dict[str, Any]] = {}
    lac = _v(c, "linear_attn_config") or {}

    for i, kind in enumerate(kinds):
        g = groups.setdefault(
            kind,
            {
                "name": kind,
                "block_size": 128,
                "layers": [],
            },
        )
        g["layers"].append(i)
        if kind == "csa_c4":
            g["compress_ratio"] = 4
        elif kind == "csa_c128":
            g["compress_ratio"] = 128
        elif kind == "swa":
            g["block_size"] = int(_v(c, "sliding_window", 128) or 128)
            g["sliding_window"] = g["block_size"]

    # 稀疏索引器缓存(indexer.k_cache 是引擎真实存在的 KV 缓存张量)。
    idx_layers = [i for i, k in enumerate(kinds) if k in ("csa_c4", "csa_c128", "dsa")]
    if idx_layers:
        groups["indexer"] = {
            "name": "indexer",
            "block_size": 128,
            "layers": idx_layers,
            "index_topk": int(_v(c, "index_topk", _v(lac, "index_topk", 512) or 512)),
            "params": {
                "index_n_heads": _v(c, "index_n_heads", _v(lac, "index_n_heads", 64)),
                "index_head_dim": _v(
                    c, "index_head_dim", _v(lac, "index_head_dim", 128)
                ),
            },
        }

    # K3: MLA+KDA 共用字节页池(引擎分层事实)。
    if model_key == "kimi-k3":
        for g in groups.values():
            if g["name"] in ("mla", "kda"):
                g["shared_pool"] = "k3_mixed_pool"

    # DSV4: 每层都有滑动窗口分支(swa_cache),窗口组覆盖全部层。
    if model_key == "deepseek-v4" and _v(c, "sliding_window"):
        groups["swa"] = {
            "name": "swa",
            "block_size": int(_v(c, "sliding_window", 128) or 128),
            "layers": list(range(len(kinds))),
            "sliding_window": int(_v(c, "sliding_window", 128) or 128),
        }

    return list(groups.values())


def layer_plan(
    model_key: str, cfg: Dict[str, Any], n: Optional[int] = None
) -> Dict[str, Any]:
    """组装 layer_plan 文档(dict, 可 JSON 序列化)。"""
    kinds = layer_types(model_key, cfg)
    if n is not None:
        kinds = kinds[:n]
    entries = [{"index": i, "type": k, "params": {}} for i, k in enumerate(kinds)]
    return {
        "model_key": model_key,
        "layers": len(kinds),
        "type_string": ",".join(kinds),
        "layer_plan": entries,
        "attention_groups": kv_group_plan(model_key, cfg),
    }


__all__ = [
    "text_config",
    "layer_types",
    "type_string",
    "kv_group_plan",
    "layer_plan",
]
