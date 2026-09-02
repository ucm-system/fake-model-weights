"""命令行入口: fake-model-weights。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .layer_plan import layer_plan, type_string
from .reduce import kv_shape_snapshot, reduce_config
from .resolve import resolve_model, stage_outputs
from .weights import build_manifest, verify_safetensors, write_safetensors


def _layer_plan_file(plan: dict) -> dict:
    return plan


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fake-model-weights",
        description="按官方注意力类型/层分布生成假模型与假权重(验证 KV 缓存架构用)。",
        epilog=(
            "MODEL 可以是 HF 仓库名(owner/repo)[hf: 前缀]、ModelScope 仓库名"
            "[ms: 前缀],或本地模型目录;\n"
            "用法示例见 README.md。输出可用 vllm --load-format dummy(只出 config)"
            "或 --load-format safetensors(生成真权重文件)。"
        ),
    )
    p.add_argument("model", help="HF 仓库名 / ModelScope 仓库名 / 本地模型目录")
    p.add_argument(
        "--source",
        choices=["auto", "hf", "ms", "local"],
        default="auto",
        help="强制来源(默认 auto 自动识别)",
    )
    p.add_argument("--cache-dir", default=None, help="HF/modelscope 下载缓存目录")
    p.add_argument("--layers", type=int, default=8, help="保留前 N 层(默认 8)")
    p.add_argument("--keep-ffn", action="store_true", help="默认会缩小 MoE/FFN,本开关保持原样")
    p.add_argument(
        "--shrink-vocab", type=int, default=0, help="把词表缩到该值(>0 时,省 dummy 权重显存;默认不缩)"
    )
    p.add_argument("--drop-vision", action="store_true", help="多模态模型去掉 vision_config")
    p.add_argument(
        "--no-ffn",
        action="store_true",
        help="权重清单只含 attention/KV 相关张量,不含 MLP/专家权重"
        "(纯 attention 层;结构验证用;vllm 主路径仍建议 --load-format dummy)",
    )
    p.add_argument("--out", default=None, help="输出目录(默认 fake-model-<out>/ 于当前目录)")
    p.add_argument(
        "--weights",
        choices=["none", "safetensors"],
        default="none",
        help="是否生成真 safetensors 权重文件(默认只出 config)",
    )
    p.add_argument("--seed", type=int, default=0, help="随机权重种子")
    p.add_argument("--shard-gb", type=float, default=2.0, help="safetensors 分片大小(GB)")
    p.add_argument("--torch", action="store_true", help="用 torch 生成权重(需已安装 torch)")
    p.add_argument(
        "--verify", action="store_true", help="写出后读回校验(建议配合 --weights safetensors)"
    )
    p.add_argument("--list-layers", action="store_true", help="只打印前 N 层层类型,不生成文件")
    p.add_argument("--json", action="store_true", help="结果以 JSON 打印(配合 --list-layers)")
    p.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # 1) 解析来源
    src = resolve_model(args.model, source=args.source, cache_dir=args.cache_dir)

    # 2) 读 config
    cfg_path = Path(src["dir"]) / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    model_key = _detect_model_key(cfg)

    # 3) 层计划
    plan = layer_plan(model_key, cfg, n=None)
    if args.list_layers:
        if args.json:
            print(
                json.dumps(
                    {
                        "model": args.model,
                        "model_key": model_key,
                        "source": src,
                        "layers": plan,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(f"model_key={model_key}  layers={plan['layers']}")
            print("type_string:", type_string(model_key, cfg, args.layers))
            for g in plan["kv_groups"]:
                print(
                    f"  group {g['name']:<10} kind={g['kind']:<8} "
                    f"block={g.get('block_size')} layers={g['layers'][:6]}{'...' if len(g['layers'])>6 else ''}"
                )
        return 0

    # 4) 缩减
    reduced = reduce_config(
        model_key,
        cfg,
        args.layers,
        shrink_ffn=not args.keep_ffn,
        shrink_vocab=args.shrink_vocab,
        drop_vision=args.drop_vision,
    )

    # 5) 输出目录 + 拷贝模型描述文件
    out = Path(args.out) if args.out else Path.cwd() / "fake-model"
    out.mkdir(parents=True, exist_ok=True)
    stage_outputs(src["dir"], out)

    (out / "config.json").write_text(json.dumps(reduced, ensure_ascii=False, indent=2))
    (out / "official_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2)
    )
    plan["layers"] = args.layers
    plan["type_string"] = type_string(model_key, cfg, args.layers)
    plan["layer_plan"] = plan["layer_plan"][: args.layers]
    (out / "layer_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))

    # 6) 可选: 生成 safetensors 权重
    files: List[str] = []
    if args.weights == "safetensors":
        manifest = build_manifest(plan, reduced, no_ffn=args.no_ffn)
        files = write_safetensors(
            manifest,
            out,
            seed=args.seed,
            use_torch=args.torch or None,
            shard_gb=args.shard_gb,
        )
        if args.verify or args.torch:
            if args.torch:
                try:
                    report = verify_safetensors(out, use_torch=True)
                    print(f"verify: {report}")
                except Exception as e:  # noqa: BLE001
                    print(f"verify(failed): {type(e).__name__}: {e}", file=sys.stderr)
            else:
                report = verify_safetensors(out)
                print(f"verify: {report}")

    snap_ok = kv_shape_snapshot(model_key, cfg) == kv_shape_snapshot(model_key, reduced)
    summary = {
        "model": args.model,
        "model_key": model_key,
        "source": src,
        "layers": args.layers,
        "type_string": plan["type_string"],
        "kv_groups": [g["name"] for g in plan["kv_groups"]],
        "kv_shape_preserved": snap_ok,
        "out": str(out.resolve()),
        "weights": len(files) if files else None,
        "hint": (
            f"vllm serve {out.resolve()} --load-format "
            + ("dummy " if not files else "safetensors ")
            + "--enable-prefix-caching --no-disable-hybrid-kv-cache-manager"
        ),
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        for k, v in summary.items():
            print(f"{k}: {v}")
    return 0


def _detect_model_key(cfg: dict) -> str:
    arches = [str(a).lower() for a in (cfg.get("architectures") or [])]
    if any("deepseek" in a for a in arches) or "compress_ratios" in cfg:
        return "deepseek-v4"
    if any("kimi" in a or "k3" in a for a in arches):
        return "kimi-k3"
    if any("glm" in a for a in arches) or any("bailing" in a for a in arches):
        return "glm-5.3"
    if any("qwen3_5" in a for a in arches) or mtype.startswith("qwen3_5"):
        return "qwen3_5"
    return "generic"


if __name__ == "__main__":
    raise SystemExit(main())
