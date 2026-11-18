# PyTorch MPS 低侵入优化实验

本轮沿用 docs/10-mps-performance.md 的已获试听认可基线，保留原环境、完整歌曲与所有中间产物。所有数字均为局部实测；没有优化后整曲成绩，也没有新的整曲听感验收。转谱、分离、混音、模型权重、上下文和生成步数保持不变。

## 隔离环境与依赖

- envs/yue2-torch215：Torch 2.10.0 → 2.15.0，Transformers 保持 4.57.6。
- envs/seed-vc-torch215：Torch 2.8.0 → 2.15.0、torchaudio 2.8.0 → 2.11.1，Transformers 保持 4.46.3。
- 固定输入及哈希锁位于 configs/macos/candidates/；正式环境锁不变。其他已有运行依赖保持固定，详见 dependency-diff.json。

torchaudio 新保存接口要求 TorchCodec；0.17.0 可安装，但本机只有 imageio 提供的 FFmpeg 可执行文件，缺少 TorchCodec 所需的共享 FFmpeg 库，实际保存失败。候选兼容补丁 patches/candidates/seed-vc-soundfile.patch 使用已经安装的 SoundFile 保存 FLOAT WAV。单声道/立体声、超出 ±1 的浮点样本与旧保存结果逐样本一致，无裁剪，不需要更改系统 FFmpeg。补丁不改模型或声码器。

## YuE2：依赖与调用修改分开测量

05_upgrade-yue 运行时尚未应用任何本轮候选源码补丁。相同真实歌曲尺寸的合成 BF16 激活，Q=[5123,16,128]、K/V=[13004,8,128]：

| 候选 | 已预热 attention 秒数 | 含义 |
| --- | ---: | --- |
| 原 Torch 2.10 / 256 分块 | 0.55257 | 前轮已有证据 |
| Torch 2.15 / 原调用不变 | 0.05990 | 仅升级依赖，约 9.2 倍 |
| Torch 2.15 / 原生 GQA、256 分块 | 0.05813 | 相对新版原调用收益很小 |
| Torch 2.15 / 原生 GQA、整块 | 0.05258 | 仅算子层约再省 12% |

抽样 query 的 CPU FP32 参考使用相同的量化后输入；三种新版路径最大绝对误差均小于 0.00015。另检查 1–8 query、非连续布局、方形因果和矩形显式 mask，均通过 atol=0.001, rtol=0.02 的逐元素检查。这个容差是算子检查标准，不是歌曲质量标准。前轮异常的短尾块在新版消失；完整真实模型的影响仍不能仅凭合成激活推断。

| 真实权重固定 token 回放 | 原 2.10 | 仅升级 2.15 | 再使用原生 AR GQA |
| --- | ---: | ---: | ---: |
| 2759 上下文，8 次前向 | 0.767 s | 0.377 s | 0.297 s |
| 7864 上下文，8 次前向 | 1.342 s | 0.581 s | 0.363 s |

这些是一次预热后的局部配对，不是吞吐统计分布。两处前缀 logits 在旧→新运行时之间相对 RMS 差异约 0.93% / 0.73%，top-20 集合相同；在新版展开 K/V→原生 GQA 之间逐值一致。没有保存前轮全部逐 token logits，因此不宣称整段生成逐值一致。原生 GQA 候选补丁只改已有 AR sdpa，Torch < 2.15 保留旧 MPS 展开分支，CPU/CUDA 行为不变。

真实整曲尺寸的单次 NAR velocity（并非完整 32 步合成）在新版原调用下为 3.435 s、原生整块为 3.180 s；两者均为首次调用。输出相对 RMS 差异 1.197%，最大绝对差异 0.203125。**不采用 NAR 整块改动**：额外收益较小，存在迭代累计数值差异且没有整曲质量证据。保留现有 NAR 分块与求解方法。

## 候选复现与恢复

在仓库根目录使用原生 arm64 Python 3.11；以下命令只创建候选环境，不切换正式环境：

```bash
uv venv --python envs/yue2/bin/python envs/yue2-torch215
uv pip sync --require-hashes --python envs/yue2-torch215/bin/python configs/macos/candidates/yue2-torch215.txt
uv pip install --no-deps --python envs/yue2-torch215/bin/python -e third_party/YuE
uv venv --python envs/seed-vc/bin/python envs/seed-vc-torch215
uv pip sync --require-hashes --python envs/seed-vc-torch215/bin/python configs/macos/candidates/seed-vc-torch215.txt
```

实验补丁通过原有固定 commit 校验入口显式应用，普通安装不会应用：

```bash
envs/control/bin/python scripts/apply_compat_patches.py --candidate yue2-native-ar-gqa --candidate seed-vc-soundfile
```

补丁重复应用会识别已应用状态；目标 commit 不匹配时拒绝。试验结束后只撤回这两份候选，不能撤掉此前 MPS 和采样修复：

```bash
git -C third_party/YuE apply --reverse ../../patches/candidates/yue2-native-ar-gqa.patch
git -C third_party/seed-vc apply --reverse ../../patches/candidates/seed-vc-soundfile.patch
```

诊断选择解释器只使用 profile_mps.py --python，不新增正式推理参数。仍使用原累计账本：

```bash
envs/control/bin/python scripts/diagnostics/profile_mps.py upgrade-yue --python envs/yue2-torch215/bin/python --run runs/20261117_165850_tonight_awake_folk_f9176f --output runs/probes/mps-performance
```

该命令用于复现“仅升级依赖”，应在候选 AR 补丁未应用时运行。不要直接连续执行所有探针；入口强制 300 秒冷却、单 worker 60 秒截止和总预算 300 秒。更换环境不重置预算。native-yue 为诊断调用替换，保留独立数值报告；seed 保存一个真实 DiT 输入，upgrade-seed 在同账本中读取它，分别比较旧 FP32→新 FP32、新 FP32→新 FP16，均不执行整曲换声。

源码与包版本继续由原有运行记录及缓存身份处理。实验补丁会改变源码哈希；不要为了查看旧音频而执行 --resume，直接打开旧交付物。

## Seed-VC：固定输入与精度对照

08_seed 在旧环境保存了实际第一步的全部输入与输出；09_upgrade-seed 在候选环境使用相同张量。总共执行 5 次 DiT 前向（原输入首次调用，加固定输入 FP32/FP16 各一次预热与一次测量），没有执行 30 步转换。实际输入、参数、输出均记录为 MPS，隐式 CPU 算子回退关闭。

| 检查 | 旧 Torch 2.8 FP32 | 新 Torch 2.15 FP32 | 新 Torch 2.15 FP16 |
| --- | ---: | ---: | ---: |
| 模型加载 | 4.665 s | 37.676 s | 同一已加载模型 |
| 第一次 DiT 前向 | 1.446 s | 1.019 s | 不作为独立稳态比较 |
| 相同输入预热后前向 | 未测 | 0.897 s | 0.469 s |
| 20 秒源 mel 的 BigVGAN 解码 | 9.024 s | 8.465 s | 没有半精度化 BigVGAN |

旧→新 FP32 输出相对 RMS 差异 4.14e-7，最大绝对差异 0.000203；新 FP32→新 FP16 相对 RMS 差异 0.000526，最大绝对差异 0.2201。输出均有限。两个对照分开记录，没有把依赖与精度收益混合归因。FP16 的预热耗时为 1.145 s，另行列出，未计入 0.469 s。

首次加载增加约 33 秒。原因尚未隔离，不能把它当作确定的缓存/编译成本忽略。BigVGAN 的约 6% 差异仅来自一次未独立预热的测量，不能据此认定稳定收益；旧环境没有预热后单步记录，也不能拿旧冷调用与新热调用声称准确的依赖加速比。

**FP16 保留为候选，不默认启用。** 尚未验证完整 30 步累计误差、转换音高、块间衔接或听感。源 mel 声码器探针不等于换声质量验收。本轮不调整 BigVGAN、CFG、求解方法或迭代步数。

## 整曲外推与范围退出

本轮不使用唯一一次整曲机会。以下只是解释剩余差距的粗略计算，**估算，未做优化后整曲实测**：

- 未修改的准备、转谱、分离、混音合计已约 259.4 秒（4.32 分钟）。
- 若把原生 AR 的短/长上下文局部速度乘以 5122 次，前向约 190–232 秒；采样、预填充、加载和长时间热降频尚未完整计入。
- 新版原有分块 NAR 的首次 velocity 为 3.435 秒；机械乘以 64 次约 220 秒，另有预填充。这不是完整求解的已测耗时。
- 新版 Seed FP32 单步乘以 30×11 块约 296 秒；还需声码器、特征、加载。FP16 仅从单步线性外推，最多约少 141 秒，不能视作已验证整曲收益。

连同 VAE 和声码器等成本，这组数值指向约 20 分钟的量级，不能支持已达到几分钟的结论。该外推忽略了各块长短、状态变化、首次编译和持续热状态等差异，不是置信区间或性能承诺。尤其不能把 NAR 单个 attention 的 9.2 倍收益套到整个 YuE2 或端到端。

保留依赖升级、AR 原生 GQA 和局部 FP16 的独立证据；排除本轮 NAR 整块改动。没有扩大到 MLX、Metal、KV cache/声码器重写或其他阶段。

## 最终状态与验证

本轮全部候选检查完成后，已撤回临时应用的两份候选源码补丁；正式环境锁、默认精度及旧音频保持原样。候选环境、可重复应用的版本化补丁和报告予以保留，**没有将缺少整曲质量证据的候选设为默认**。标准安装不启用 --candidate；采用候选需要显式选择它。

10_verify-yue 实际应用 AR 候选补丁，在同一候选预填充产生的 KV 上，将 8 次固定 token decode 与固定上游原函数对照：2759/7864 两个上下文的最终 logits 均逐值相同。这里验证的是相同 KV 条件下的局部 decode；前缀 logits 一致性另由 05/07 报告支持，不能据此推导所有缓存张量或整曲随机生成均逐值相同。该 worker 内的实际 MPS attention 3 项回归通过，覆盖 1–8 query、7864 keys、非连续布局、因果预填充和矩形可见 key 掩码。

CPU 验证包括：控制环境 31 项通过、7 项需要 YuE 环境的测试跳过；新旧 YuE 环境 attention 各 3 项通过；新版采样 4 项通过；上游模型/缓存/NAR 34 项通过、1 项原始发布源码检查因未安装该源码跳过。目标 commit 错误拒绝、补丁重复应用与撤回验证通过。独立只读审查已完成，并修复了诊断调用数、FP16 对照参考和数值失败状态的问题。CUDA 实机回归未执行。

累计账本从此前 56.09 秒增加到 **218.77 秒**，本轮新增 **162.68 秒**，均包含导入、加载及 CPU 部分，不冒充精确 GPU 活跃时间。最长 worker 56.42 秒，所有新增组间冷却超过 300 秒。没有重跑整曲，也没有逐阶段重跑完整歌曲。旧 final.wav SHA256 仍为 3ee0f331f461a5adebbc77001831ebf94ea4f405954ef686bc2ad08f954cb557。

机器可读结果及完整账本快照见 [optimization.json](../configs/macos/optimization.json)，原始日志、张量、数值比较在 runs/probes/mps-performance/05_upgrade-yue 至 10_verify-yue。

剩余候选诊断的复现形式如下；现有账本中已保存 Seed 输入，不要再次生成一份基线输入。执行仍受原有预算与冷却限制，不应批量运行：

```bash
# 使用新版依赖、临时应用所需候选补丁之后：
envs/control/bin/python scripts/diagnostics/profile_mps.py verify-yue --python envs/yue2-torch215/bin/python --run runs/20261117_165850_tonight_awake_folk_f9176f --output runs/probes/mps-performance
envs/control/bin/python scripts/diagnostics/profile_mps.py upgrade-seed --python envs/seed-vc-torch215/bin/python --run runs/20261117_165850_tonight_awake_folk_f9176f --output runs/probes/mps-performance
```

新版依赖的机制背景参考 [PyTorch 2.15 MPS attention 源码](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/mps/operations/Attention.mm)；保存接口依赖参考 [TorchCodec 安装说明](https://github.com/pytorch/torchcodec#installing-torchcodec)。采用与否以上述本机实测和质量门槛为准。



