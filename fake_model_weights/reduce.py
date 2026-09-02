"""配置缩减: 砍层数/缩 MoE/词表,但 KV 相关形状一字不动。"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

from .layer_plan import text_config

# KV 形状字段: 只读不改(减层可以,减 KV 形状不行)。
KV_SHAPE_KEYS = (
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "qk_head_dim",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "kv_lora_rank",
    "q_lora_rank",
    "o_lora_rank",
    "compress_ratio",
    "compress_ratios",
    "compress_rope_theta",
    "sliding_window",
    "max_window_layers",
    "index_n_heads",
    "index_head_dim",
    "index_topk",
    "index_kpool",
    "rotary_dim",
    "headwise_attn_output_gate",
    "partial_rotary_factor",
    # Qwen3.5/GDN 线性注意力参数
    "attn_output_gate",
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_num_key_heads",
    "linear_value_head_dim",
    "linear_num_value_heads",
)

# 可安全缩小的非 KV 字段(默认缩小,省 dummy 权重显存)。
FFN_SHRINK: Dict[str, Dict[str, int]] = {
    "deepseek-v4": {"moe_intermediate_size": 64, "n_routed_experts": 4},
    "kimi-k3": {"moe_intermediate_size": 64, "num_experts": 4},
    "glm-5.3": {"moe_intermediate_size": 64, "n_routed_experts": 4},
    "qwen3_5": {"moe_intermediate_size": 64, "n_routed_experts": 4},
}
FFN_SHRINK["generic"] = {"intermediate_size": 64}


def _original_layer_count(model_key: str, cfg: Dict[str, Any]) -> int:
    if model_key == "deepseek-v4":
        return int(cfg.get("num_hidden_layers") or 0)
    return int(text_config(cfg).get("num_hidden_layers") or 0)


def _apply_ffn_shrink(cfg: Dict[str, Any], targets: Dict[str, int]) -> None:
    for field, target in targets.items():
        if field in cfg and isinstance(cfg[field], int):
            cfg[field] = target


def _clamp_ffn_consistency(c: Dict[str, Any], model_key: str) -> None:
    pairs = {
        "deepseek-v4": ("num_experts_per_tok", "n_routed_experts"),
        "kimi-k3": ("num_experts_per_token", "num_experts"),
        "glm-5.3": ("num_experts_per_tok", "n_routed_experts"),
        "qwen3_5": ("num_experts_per_tok", "n_routed_experts"),
    }
    per_tok, routed = pairs.get(model_key, (None, None))
    cfg = c if model_key == "deepseek-v4" else text_config(c)
    if per_tok in cfg and routed in cfg:
        cfg[per_tok] = min(int(cfg[per_tok]), max(1, int(cfg[routed])))


def reduce_config(
    model_key: str,
    cfg: Dict[str, Any],
    n_layers: int,
    shrink_ffn: bool = True,
    shrink_vocab: int = 0,
    drop_vision: bool = False,
) -> Dict[str, Any]:
    """把官方 config 砍到前 ``n_layers`` 层,保持层类型模式与 KV 形状。"""
    if n_layers < 1:
        raise ValueError(f"--layers 必须 >=1,实际 {n_layers}")
    c = copy.deepcopy(cfg)
    # 假模型只验 KV 结构: 剥离官方量化配置(避免 --load-format dummy 走(假)量化
    # 权重路径,在 Ascend NPU 上会报 aclnnInplaceCopy 561103)。
    for scope in (c, c.get("text_config") or {}, c.get("vision_config") or {}):
        if isinstance(scope, dict):
            scope.pop("quantization_config", None)
            scope.pop("quant_method", None)
    tc = text_config(c)
    n_orig = _original_layer_count(model_key, c)

    if model_key == "deepseek-v4" or "compress_ratios" in c:
        if n_layers < n_orig:
            c["num_hidden_layers"] = n_layers
            ratios = list(c.get("compress_ratios") or [])
            c["compress_ratios"] = ratios[:n_layers]
            c.pop("dspark_target_layer_ids", None)
            c.pop("dspark_noise_token_id", None)
            c["num_hash_layers"] = 0
        if shrink_ffn:
            _apply_ffn_shrink(c, FFN_SHRINK["deepseek-v4"])
        if shrink_vocab:
            c["vocab_size"] = shrink_vocab
    else:
        if n_layers < n_orig:
            tc["num_hidden_layers"] = n_layers
        lac = tc.get("linear_attn_config") or {}
        if isinstance(lac, dict):
            for key in (
                "full_attn_layers",
                "kda_layers",
                "layer_types",
                "mlp_layer_types",
                "indexer_types",
                "dense_attn_layers",
            ):
                if lac.get(key) is not None:
                    if all(isinstance(x, int) for x in lac[key]):
                        lac[key] = [int(x) for x in lac[key] if int(x) < n_layers]
                    else:
                        lac[key] = list(lac[key])[:n_layers]
        for key in ("layer_types",):
            if tc.get(key) is not None:
                tc[key] = list(tc[key])[:n_layers]
        if n_layers < int(tc.get("first_k_dense_replace") or 0):
            tc["first_k_dense_replace"] = n_layers
        if shrink_ffn:
            _apply_ffn_shrink(tc, FFN_SHRINK.get(model_key, FFN_SHRINK["generic"]))
        if shrink_vocab:
            tc["vocab_size"] = shrink_vocab
        if drop_vision:
            c.pop("vision_config", None)

    _clamp_ffn_consistency(c, model_key)
    return c


def kv_shape_snapshot(model_key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """提取全部 KV 形状**标量**字段(用于缩减前后一致性断言)。

    逐层分布列表(compress_ratios / layer_types / full_attn_layers …)不属于
    "形状",缩减时按层截断属预期行为,由专门的 prefix 测试校验。
    """
    c = text_config(cfg)
    snap = {
        k: c.get(k) for k in KV_SHAPE_KEYS if k in c and not isinstance(c.get(k), list)
    }
    lac = c.get("linear_attn_config") or {}
    if isinstance(lac, dict):
        for k in KV_SHAPE_KEYS:
            if k in lac and k not in snap and not isinstance(lac[k], list):
                snap[k] = lac[k]
    return snap


__all__ = ["KV_SHAPE_KEYS", "FFN_SHRINK", "reduce_config", "kv_shape_snapshot"]
