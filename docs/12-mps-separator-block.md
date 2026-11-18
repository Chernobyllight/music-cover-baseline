# M5 分离单块实验及两个 CPU 候选

机器为无风扇 M5，24 GiB。基线为 20261117_165850_tonight_awake_folk_f9176f，整曲成功阶段合计 2925.37 秒，分离 215.38 秒。上一轮候选组合约 20 分钟的数字来自外推。两轮均未重新生成整曲。

## 结果与取舍

| 候选 | 一首歌可省时间 | 证据与取舍 |
| --- | ---: | --- |
| BS-RoFormer 升级运行时 | 约 50.4 秒 | 同块首次 7.829 → 5.979 秒，第二次 7.376 → 5.429 秒，降幅 26.4%。加载并传入输入 0.656 → 0.804 秒。通过本轮 20% 筛选，保留为候选环境。 |
| YuE2 语义输出投影 | 33–40 秒，等比例情景 | 仅 CPU 数值通过。该情景取决于输出层在 MPS 的实际耗时份额，暂不计入整曲估算。 |
| Seed-VC 循环 | 耗时收益待测 | 小型 solver 末态一致，单块约少保留 36.5 MiB 旧状态。正式推理仍使用旧循环。 |

分离外推按一次首次调用与 25 次重复调用计算，扣除新增的 0.148 秒加载，预计从旧分离阶段 215.38 秒降至约 165 秒。以此前粗略的 20 分钟为锚，仅替换分离耗时后约 **19 分钟**。测量只有一个输入块与一次热调用，没有测全曲持续温度和音质。相对 5 分钟参照仍差约 14 分钟。

两套运行各记录 12 次时间轴和 12 次频带轴原生 SDPA 调用，Q/K 形状分别为 [62,8,801,64] 与 [801,8,62,64]，dtype 为 FP32，设备为 MPS。两个输出均为有限的双声道 FLOAT WAV。人声和伴奏相对 RMS 误差分别为 5.00e-6 和 7.77e-6，最大绝对误差均低于 9.8e-6，两端样本最大差低于 6.3e-7。两条原环境输出对交付 PCM16 的最大差均为一个量化单位。试听片段在原账本的 11_separator 和 12_separator 目录，各包含 vocals.wav 与 instrumental.wav。人工听感待验收。

两次 worker 分别计费 17.10 和 14.25 秒，中间冷却 348.15 秒，累计 250.12 / 300 秒。机器可读结果、数值明细和脚本哈希见 [optimization-round2.json](../configs/macos/optimization-round2.json)。本轮控制环境测试 31 项通过，7 项依赖独立 YuE 环境的测试跳过。

## 复现

在仓库根目录执行。原运行目录、冻结的音频输入与同一累计账本是复现所需的本地数据。下列 GPU 命令受原账本额度及 300 秒冷却约束，本机已有 A/B 结果可直接比较。

```bash
run=runs/20261117_165850_tonight_awake_folk_f9176f
ledger=runs/probes/mps-performance
export PATH="$PWD/envs/control/bin:$PATH"
uv venv --python envs/separator/bin/python envs/separator-torch215
uv pip sync --require-hashes --python envs/separator-torch215/bin/python configs/macos/candidates/separator-torch215.txt
envs/separator/bin/python scripts/diagnostics/profile_separator_block.py "$run" "$ledger"
envs/separator-torch215/bin/python scripts/diagnostics/profile_separator_block.py "$run" "$ledger"
envs/control/bin/python scripts/diagnostics/profile_mps.py separator --run "$run" --output "$ledger" --deadline 30
# 第二个 worker 需等同一账本冷却完成
envs/control/bin/python scripts/diagnostics/profile_mps.py separator --run "$run" --output "$ledger" --deadline 45 --python envs/separator-torch215/bin/python
envs/control/bin/python scripts/diagnostics/compare_separator_blocks.py "$ledger/11_separator" "$ledger/12_separator" "$run"
envs/yue2-torch215/bin/python scripts/diagnostics/check_mps_candidates_cpu.py yue
envs/seed-vc-torch215/bin/python scripts/diagnostics/check_mps_candidates_cpu.py seed
```

正式入口使用原 separator 环境。试验通过诊断入口的解释器参数选择候选，恢复时选择原解释器即可。Seed 循环补丁只在临时目录对固定 commit 应用并做 CPU 筛选。显式应用后可用 `git -C third_party/seed-vc apply --reverse ../../patches/candidates/seed-vc-loop.patch` 撤回。YuE2 输出投影仅在 CPU 实验脚本中实现。

## 方法

两套环境使用 audio-separator 0.30.2、同一 checkpoint、模型 YAML 及 FP32。运行时组合是 Torch / torchaudio / torchvision 2.8.0 / 2.8.0 / 0.23.0 与 2.15.0 / 2.11.1 / 0.29.2。真正导入 onnx2torch，加载 CPU 权重并通过候选环境的依赖一致性检查。两次独立调用上游 prepare_mix 和全曲 normalize，再从第 1,764,000 个样本起取 352,800 个立体声样本。这个 40–48 秒块在基线的完整内部块中人声 RMS 最高（0.1302）。两套环境的输入 SHA256 完全相同。

实际调用加载后的 MDXC 模型。同步 MPS 后分别记录加载并传入输入、第一次和第二次前向。输出 FLOAT WAV 使用拼接前人声及输入减人声所得伴奏。旧整曲交付为 PCM16，额外比较同一位置以及左右边界样本。末块在上游从歌曲结尾反向取满 8 秒，并与前块加窗叠加，单个内部块的检查不覆盖候选末块。

## CPU 候选

**YuE2 语义输出投影。** 原 head 的 BF16 权重为 184704 × 2048。结束符与 codec 区间连续，允许 32769 行。仅在语义阶段投影这些行，将结果放回原维度再调用既有 distribution 和 multinomial，可保留原有排序、概率和 RNG 维度。ABC 使用原路径。在两组固定隐藏向量、CFG 1/1.5、惩罚历史、最短 token 数前后和两种历史精度模式下，允许行 logits 和全词表概率完全一致，固定 RNG 的 token 与状态一致。另用与 EOS 权重对齐的向量验证 greedy 在最短长度前阻止结束、之后选中结束。相同运算量口径下，输出层占 AR 稠密线性层乘加约 21.16%，其中可省 82.26%，相当于总稠密乘加的 17.41%。按上一轮 190–232 秒 AR 前向且假设耗时与稠密乘加完全成比例，情景估算为 33–40 秒；attention、内存流量和 MPS 内核可能使实际结果明显不同。

**Seed-VC 扩散循环。** 用原账本保存的第一个 DiT 输入核对 CFG 拼接后条件形状与空分支。每个同尺寸块的 30 步可少写 536,354,304 字节 cat 输出和 268,177,152 字节 zeros_like 临时值，最后一步可少保留 38,307,840 字节旧状态。这些是张量大小推算，并非实测峰值或秒数。取固定上游 solver 和候选补丁，逐值比较小型确定性 estimator 的 30 步末态，覆盖 CFG 关闭/开启、prompt 条件清零开关及最终 prompt 区域。四种情形均通过。真实 DiT 的长程数值、MPS 耗时和听感仍待验证。
