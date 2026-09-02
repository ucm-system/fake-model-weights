"""UCM 规格表投影视图（可选，与核心工具解耦）。

核心 ``fake-model-weights`` 只描述**模型/引擎视角**的层结构与注意力组；
chain / snapshot / sidecar、独立种子、storage_block_size 是 **UCM 缓存系统**
的规格表语义，不属于模型描述——本模块以"显式投影"方式提供，默认不输出，
仅在 ``--ucm-view``（或直接调用本函数）时给出。

这样保证了工具本身与 UCM 完全解耦：任何缓存系统（vLLM 原生 / LMCache /
Mooncake / UCM …）都可以基于中立的 ``attention_groups`` 自行映射。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .layer_plan import text_config

# UCM 规格表语义的数据分类（缓存系统视角）。
KIND_CHAIN = "chain"
KIND_SNAPSHOT = "snapshot"
KIND_SIDECAR = "sidecar"
KIND_NONE = "none"


def _v(cfg: Dict[str, Any], name: str, default: Any = None) -> Any:
    return cfg.get(name, default)


def ucm_spec_table_view(
    model_key: str,
    cfg: Dict[str, Any],
    groups: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """把中立的注意力组投影为 UCM 规格表行。

    映射规则（启发式，可按需要调整）：
      - kda → snapshot（快照/状态数据，位置键）；
      - indexer → sidecar（索引器跟随源组，不参与命中投票）；
      - 其余（full/mla/csa_*/swa/dsa） → chain；
      - storage_block_size：CSA 压缩组按 block/compress_ratio 计算；
      - seed：稳定派生名（缓存系统实现通常用独立种子做组间哈希隔离）。
    """
    c = text_config(cfg)
    rows: List[Dict[str, Any]] = []
    for g in groups:
        name = g["name"]
        if name == "kda":
            kind = KIND_SNAPSHOT
        elif name == "indexer":
            kind = KIND_SIDECAR
        else:
            kind = KIND_CHAIN
        storage: Optional[int] = None
        ratio = g.get("compress_ratio")
        bs = int(g.get("block_size") or 128)
        if ratio:
            storage = max(1, bs // int(ratio))
        row: Dict[str, Any] = {
            "group_name": name,
            "kind": kind,
            "block_size": bs,
            "storage_block_size": storage,
            "seed": f"S_{name}" if name not in ("indexer",) else "S_indexer",
        }
        if g.get("shared_pool"):
            row["shared_pool"] = g["shared_pool"]
        if g.get("index_topk") is not None:
            row["index_topk"] = g["index_topk"]
        rows.append(row)
    return rows


__all__ = [
    "KIND_CHAIN",
    "KIND_SNAPSHOT",
    "KIND_SIDECAR",
    "KIND_NONE",
    "ucm_spec_table_view",
]
