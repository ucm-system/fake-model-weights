"""模型来源解析: HF 模型名 / ModelScope 模型名 / 本地路径 -> config+tokenizer。"""

from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

# 目标仓库的官方 HF 仓库 ID(用作默认/文档,并作为 --fetch 的候选)。
KNOWN_OFFICIAL_REPOS = {
    "deepseek-v4": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "kimi-k3": "moonshotai/Kimi-K3",
    "glm-5.3": "zai-org/GLM-5.3-Flash",
    "qwen3_5": "Qwen/Qwen3.6-27B",
}

# 同步拉取时仅需的模型描述文件(权重文件不拉)。
_CONFIG_LIKE = ("*.json", "tokenizer*", "*.model", "*.txt", "special_tokens_map*")


def resolve_model(
    model: str,
    source: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, object]:
    """把 ``model`` 解析成本地模型目录,返回 {dir, source, fetched_via, note}。

    支持的 ``model`` 写法:
      - HF 仓库名 ``owner/repo``(自动或 ``hf:owner/repo``)
      - ModelScope 仓库名 ``owner/repo`` 或 ``ms:owner/repo``
      - 本地已存在模型目录/路径(含 config.json)
    ``source`` 可强制: hf / ms / local。
    """
    p = Path(model).expanduser()
    if source == "local" or (source in (None, "auto") and _is_local_model(p)):
        cfg = p / "config.json"
        if not cfg.exists():
            raise FileNotFoundError(f"本地模型目录缺少 config.json: {p}")
        return {
            "dir": str(p.resolve()),
            "source": "local",
            "fetched_via": None,
            "note": "本地路径",
        }

    prefix, repo_id = _split_prefix(model)
    if source is None:
        source = prefix or "auto"

    if source == "hf" or (source == "auto" and prefix != "ms"):
        return _from_hf(repo_id, cache_dir)
    if source == "ms" or (source == "auto"):
        return _from_ms(repo_id, cache_dir)
    raise ValueError(f"未知 source={source}")


def _is_local_model(p: Path) -> bool:
    return (
        p.exists()
        and (p / "config.json").exists()
        or (p.suffix == ".json" and p.exists())
    )


def _split_prefix(model: str) -> tuple[Optional[str], str]:
    if model.startswith("hf:"):
        return "hf", model[3:]
    if model.startswith("ms:"):
        return "ms", model[3:]
    return None, model


def _from_hf(repo_id: str, cache_dir: Optional[str]) -> Dict[str, object]:
    tried: List[str] = []
    try:
        from huggingface_hub import snapshot_download

        d = snapshot_download(
            repo_id=repo_id,
            allow_patterns=list(_CONFIG_LIKE),
            cache_dir=cache_dir,
        )
        return {
            "dir": d,
            "source": "hf",
            "fetched_via": "huggingface.co",
            "note": "huggingface_hub",
        }
    except Exception as e:
        tried.append(f"huggingface.co: {type(e).__name__}")
    # 兜底: 直连抓 config.json(hf-mirror 镜像)。
    for base in ("https://hf-mirror.com", "https://huggingface.co"):
        try:
            d = _raw_config_dir(repo_id, base)
            tried.append(f"{base}: ok")
            return {
                "dir": d,
                "source": "hf",
                "fetched_via": base,
                "note": "raw config only",
            }
        except Exception as e:
            tried.append(f"{base}: {type(e).__name__}")
    raise RuntimeError(f"HF 拉取失败(尝试: {tried})。可 pip install huggingface_hub")


def _raw_config_dir(repo_id: str, base: str) -> str:
    """只抓 config.json + tokenizer 相关小文件到临时目录(纯 stdlib)。"""
    import tempfile

    url = f"{base}/{repo_id}/resolve/main/config.json"
    with urllib.request.urlopen(url, timeout=60) as r:
        raw = r.read()
    d = Path(tempfile.mkdtemp(prefix="fmw-hf-"))
    (d / "config.json").write_bytes(raw)
    json.loads(raw)  # 校验
    return str(d)


def _from_ms(repo_id: str, cache_dir: Optional[str]) -> Dict[str, object]:
    try:
        from modelscope import snapshot_download

        d = snapshot_download(
            repo_id,
            cache_dir=cache_dir,
            allow_file_pattern=[
                "*.json",
                "tokenizer*",
                "*.model",
                "*.txt",
            ],
        )
        return {
            "dir": d,
            "source": "ms",
            "fetched_via": "modelscope.cn",
            "note": "modelscope",
        }
    except ImportError as e:
        raise RuntimeError("需要 modelscope 才能从魔搭拉取: pip install modelscope") from e


def stage_outputs(src_dir: str, out_dir: Path) -> None:
    """把 config/tokenizer 等模型描述文件复制到输出目录。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(src_dir)
    copied: List[str] = []
    for f in src.iterdir():
        if f.is_file() and (
            f.name.startswith(("config", "tokenizer", "special_tokens_map"))
            or f.suffix in (".model", ".txt", ".json")
        ):
            shutil.copy2(f, out_dir / f.name)
            copied.append(f.name)
    if not (out_dir / "config.json").exists():
        raise FileNotFoundError(f"来源 {src_dir} 没有 config.json")


__all__ = [
    "KNOWN_OFFICIAL_REPOS",
    "resolve_model",
    "stage_outputs",
]
