#
# MIT License
#
# Copyright (c) 2026 ucm-system
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""fake-model-weights: 按官方注意力类型/层分布生成"假权重"的工具。

动机: 数十~上千亿参数的大模型(DeepSeek-V4-Flash / Kimi K3 / GLM-5.3-Flash
等)单张 64GB NPU 装不下,无法在本地直接验证缓存架构对"真实 KV 分组"的适配。
本工具只保留模型的**层分布与 KV 相关形状**(hidden/heads/head_dim/lora
ranks/sliding_window/compress_ratios/indexer… 一字不改),把层数砍到 N 层、
把 MoE/FFN/词表缩小,从而生成一个:
 1. 结构真实(与官方模型同构的 KVCacheConfig 分组) 2. 权重极小 的"假模型"。

两种使用姿势:
 - 只出 config(推荐): 生成缩减版 config.json + layer_plan.json,配合
   vllm 的 ``--load-format dummy`` 加载随机权重(无需真实权重文件);
 - 出真 safetensors: ``--weights safetensors`` 按架构生成带正确参数名的
   随机权重文件(model.safetensors + index.json),可被 ``--load-format
   safetensors`` 读取(自定义架构参数名按启发式清单,以 ``--verify``/日志为准)。
"""

__version__ = "0.2.0"
