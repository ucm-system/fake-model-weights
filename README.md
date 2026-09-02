# fake-model-weights

按官方模型的 **注意力类型与层分布** 生成“假模型 / 假权重”的命令行工具，
用于在单张 64GB NPU 上验证 KV 缓存架构（UCM：规格表 / 双原语 / 命中裁决）对
真实混合注意力模型的适配——不需要下载动辄几十~上千 GB 的真实权重。

支持三类输入来源，全部一条命令：

```bash
# 从 HuggingFace 模型名
fake-model-weights deepseek-ai/DeepSeek-V4-Flash-0731 --layers 8 --out ./dsv4-8l
# 从 ModelScope 模型名
fake-model-weights ms:Qwen/Qwen2.5-0.5B-Instruct --layers 8 --out ./qwen-8l
# 本地模型目录
fake-model-weights /data/models/GLM-5.3-Flash --layers 8 --out ./glm-8l
```

---

## 为什么需要“假权重”

目标模型（DeepSeek-V4-Flash / Kimi K3 / GLM-5.3-Flash 一类的 2026 混合注意力模型）
权重体量巨大（例：DSV4-Flash-0731 ≈ **240–250B 参数**，Q8 ≈ 162GB），单张 910B
(64GB) 装不下，无法在本地直接验证缓存架构对“真实 KV 分组”的适配。

本工具只保留模型的：
- **层分布**（每一层的注意力种类：full / mla / csa_c4 / csa_c128 / swa / kda / dsa）
- **KV 相关形状**（hidden_size、heads、head_dim、lora ranks、sliding_window、
  compress_ratios、indexer 参数……**一字不改**）

然后把层数砍到 `N` 层（默认 8）、把 MoE/FFN/词表缩小，得到：
1. **结构真实**：vllm/vllm-ascend 解析出的 `KVCacheConfig` 分组与真实模型同构
   （实测 DSV4 假模型解析出 full / csa_c4 / csa_c128 / swa / indexer + compressor
   state 等真实分组）；
2. **权重极小**：8 层 dummy 权重仅数个 GB，单卡可跑。

## 两种使用姿势

| 姿势 | 命令 | 适合 |
|---|---|---|
| **只出 config**（推荐） | 默认（`--weights none`） | 配合 vllm `--load-format dummy` 加载随机权重，无需真实权重文件 |
| **出真 safetensors** | `--weights safetensors [--torch]` | 权重文件可被 `--load-format safetensors` 读取 |

## 安装

```bash
pip install .            # 安装 CLI: fake-model-weights
# 或免安装: python -m fake_model_weights ...
```

可选依赖（缺失时自动降级/提示）：

| extra | 功能 |
|---|---|
| `hf` | 从 HuggingFace 精确拉取（`pip install huggingface_hub`） |
| `ms` | 从 ModelScope 拉取（`pip install modelscope`） |
| `weight` | 用 torch 生成权重/校验（`pip install torch safetensors`） |

```bash
pip install .[all]
```

## 用法

```
fake-model-weights MODEL \
    [--source auto|hf|ms|local] [--cache-dir DIR] \
    [--layers N] [--keep-ffn] [--shrink-vocab N] [--drop-vision] \
    [--out DIR] [--weights none|safetensors] [--seed S] [--shard-gb GB] \
    [--torch] [--verify] [--list-layers] [--json]
```

- `MODEL`：HF 仓库名（`owner/repo` 或 `hf:owner/repo`）、ModelScope 仓库名
  （`owner/repo` 或 `ms:owner/repo`）、本地模型目录（含 `config.json`）。
  `--source` 可强制来源；`auto` 时本地路径优先，否则默认 HF。
- `--layers N`：保留前 N 层（默认 8）。**只砍层数，KV 形状不动。**
- `--list-layers`：只打印层计划（研究模式），不落文件。
- `--weights safetensors`：生成随机权重文件（分片 + index.json）。
- `--no-ffn`：权重清单只含 attention/KV 相关张量，**不含 MLP/专家权重**
  （“纯 attention 层”）；用于只验 KV 结构的场景（vllm 主路径仍建议
  `--load-format dummy`）。
- `--torch`：用 torch 生成（正态分布，shape/dtype 标准）；缺 torch 时用
  纯 stdlib 确定性随机字节。
- `--verify`：写出后读回校验（offset 对齐 / 张量数 / 可选 torch 加载）。

## 输出

| 文件 | 说明 |
|---|---|
| `config.json` | 缩减后的模型配置（KV 形状原样，量化配置已剥离） |
| `official_config.json` | 官方原始 config（来源见 `layer_plan.json`） |
| `layer_plan.json` | 逐层注意力种类 + KV 组规划（chain/snapshot/sidecar、种子、块大小等） |
| `model-NNNNN-of-NNNNN.safetensors` + `model.safetensors.index.json` | 随机权重（`--weights safetensors` 时） |

生成的目录可直接给 vllm：

```bash
vllm serve ./dsv4-8l --load-format dummy \
    --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
    --kv-transfer-config '...'          # 示例见下文
```

## 已内置的官方层分布（真 config 快照）

| model_key（自动识别） | 来源仓库 | 层数 | 前 8 层层类型 |
|---|---|---|---|
| `deepseek-v4` | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 43 | `full,full,csa_c4,csa_c128,csa_c4,csa_c128,csa_c4,csa_c128` |
| `kimi-k3` | [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | 93 | `mla,kda,kda,kda,mla,kda,kda,kda` |
| `glm-5.3` | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 45 | `kda,kda,kda,dsa,kda,kda,kda,dsa` |
| `qwen3_5` | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) 等 | 64 | `kda,kda,kda,full,kda,kda,kda,full` |

自动化识别 `model_key`：架构名含 DeepSeek / `compress_ratios` → `deepseek-v4`；
Kimi/K3 → `kimi-k3`；GLM/Bailing → `glm-5.3`；其余 → `generic`（全层按 full 处理）。
必要字段的解析见 `official_configs/`（仓库内缓存，校验 sha256 见各文件来源）。

派生 KV 组（UCM 规格表视角，`layer_plan.json` 的 `kv_groups`）：

| 组 | kind | 说明 |
|---|---|---|
| `full` / `csa_c4` / `csa_c128` | chain | 全注意力 / 压缩注意力（C4=4x、C128=128x） |
| `swa` | chain | 滑动窗口（DSV4 每层分支，window=128） |
| `mla` | chain | 潜压缩注意力（K3，与 kda 共用字节页池） |
| `kda` | snapshot | 记忆编辑/线性注意力状态（位置键快照数据） |
| `dsa` | chain | 稀疏注意力（GLM） |
| `indexer` | sidecar | 稀疏索引器（跟随源组，不参与命中投票） |

## 与缓存系统解耦（v0.3.0）

本工具**只描述模型/引擎视角**的结构：

- 逐层注意力种类（full / mla / csa_c4 / csa_c128 / swa / kda / dsa）；
- 注意力组（`attention_groups`：组名、层索引、块大小、压缩比/窗口/索引器参数，
  与 vLLM 的 `KVCacheConfig` 分组同构）；
- KV 相关形状（`config.json` 原样保留）。

**不含任何缓存系统的规格表语义**——“chain / snapshot / sidecar”分类、`seed`、
`storage_block_size` 是 UCM（或其他缓存系统）自己叠加的数据语义，不属于模型
描述。如需 UCM 视角的规格表行，显式开启投影视图：

```bash
fake-model-weights <model> --layers 8 --out ./out --ucm-view
# layer_plan.json 里会多出 "ucm_spec_table": [{group_name, kind, seed, ...}]
```

任何缓存系统（vLLM 原生 / LMCache / Mooncake / UCM …）都可以基于中立的
`attention_groups` 自行映射，工具本身不绑定实现。

## 与 vllm-ascend 对接的注意点（实战踩坑记录）

1. **量化配置需剥离**：官方 config 常带 `quantization_config`（如 DSV4 的 FP8），
   配合 `--load-format dummy` 会走（假）量化权重路径，在 Ascend NPU 上报
   `aclnnInplaceCopy 561103`——工具默认剥离，无需手动处理。
2. **tokenizer**：假模型目录需要 tokenizer 文件（可从官方仓库/同系模型复制）。
3. **KV 形状纪律**：KV 相关字段必须原样，否则 `KVCacheConfig` 分组与真实模型
   不一致（本工具强制约束，并有测试兜底）。
4. **v0.26 DSV4 已知引擎坑**：`cache_config.block_size` 会被压到最小组（C4
   状态块），导致 `_dsv4_block_sizes()[2/8] KeyError`——属 vllm-ascend 引擎
   缺陷（专用 DSV4 镜像/更新版本已修复），与假模型方法无关。
5. **自定义架构参数名**：`--weights safetensors` 的权重名按 HF 风格启发式清单
   生成（`model.layers.N.self_attn.*`）。标准架构（llama/qwen2/phi…）可直接
   被 vllm 读取；DeepSeekV4/Kimi/GLM 这类自定义架构以 `--verify` 与 vllm 日志
   为准（可能需按 vllm 的 weight_map 微调命名）。

## 开发

```bash
pip install -e .[test]
pytest                    # tests/ 下 17 条单测(纯 stdlib, 不联网)
```

目录结构：

```
fake_model_weights/
  cli.py        命令行入口
  resolve.py    HF / ModelScope / 本地路径 解析与(可选)拉取
  layer_plan.py 层分类 + KV 组规划
  reduce.py     配置缩减(砍层/缩 FFN/词表; KV 形状纪律)
  weights.py    按架构生成 safetensors(流式分片 + index + 校验)
tests/data/     官方 config 快照(deepseek-v4 / kimi-k3 / glm-5.3)
```

## License

MIT（见 `LICENSE`）。

> 本工具来自 [ucm-system](https://github.com/ucm-system) 的 UCM 缓存架构工作；
> 相关设计见 [unified-cache-management](https://github.com/ModelEngine-Group/unified-cache-management)。