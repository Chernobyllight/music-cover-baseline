# M5 上 Seed-VC 完整 30 步与 YuE2 语义投影的局部实验

平台为无风扇 M5、24 GB。使用已有 Torch 2.15.0 候选环境与原诊断账本。已验收的整曲实测为 48.76 分钟；此前组合候选的约 19 分钟来自外推。以下实验复用该歌曲的真实输入，不计入整曲实测。

## Seed-VC

在旧记录的 20 秒源人声范围内，以 pYIN 有声比例与 RMS 的乘积选择 176 帧乐句，对应源音频 17.485–19.528 秒。使用第一个实际 DiT 输入保存的正条件、style、源噪声与完整 898 帧参考 mel，保留参考部分并抽取相同源帧区间。参考原录音为 10.435 秒。两路复用同一份真实 DiT 权重和 BigVGAN；CPU 校验了所有可学习 DiT 参数，配置派生的位置 buffer 按上游加载行为处理。

| 项目 | FP32 | 局部 FP16 |
| --- | ---: | ---: |
| 30 步求解（秒） | 9.309 | 5.721 |
| 相同 FP32 声码器解码（秒） | 0.982 | 0.870 |
| 采样率、样本数 | 44,100 Hz，90,112 | 44,100 Hz，90,112 |
| 峰值 | 0.997558 | 0.996913 |

DiT 和声码器一次共享加载及设备准备共 2.314 秒。两路均使用上游 Euler 求解、CFG 0.7、同一网格和初态，求解状态与声码器输入为 FP32，只有 DiT 前向切换 autocast。两路共用模型实例，FP32 先行；计时含首次调用和轨迹记录，分别在两端同步 MPS。声码器精度相同，解码时间差不归因于 DiT 优化。

两路 30 步输出和音频均为有限值。第 1、10、20、30 步源乐句 mel 相对 RMS 误差分别为 0.0194%、0.0309%、0.0250%、0.0317%，末步最大绝对误差 0.03671。每步在下一次 CFG stack 读取状态，第 30 步取 solver 返回值。pYIN 按 10 ms 采样、音高分辨率 10 音分；205 帧中 135 帧共同有声，有声判定分歧为 0，绝对音高差中位数及 P95 均为 0，最大为 10 音分。辅音、音色及齿音由人工 A/B 试听验收。该局部音频没有跨块衔接。

局部 30 步节省 3.588 秒，求解耗时降低 38.5%。将该比例套用到先前估计的 296 秒整曲 DiT 计算，对应约 114 秒。较长块的旧单步数据乘 330 步得到约 141 秒。两数采用不同长度与计量口径，均为情景估算，不能相加。正式精度仍为 FP32。

## YuE2 声学阶段

现有真实整曲尺寸单次首次 velocity 为 3.435 秒；一首 32 步 midpoint 调用 64 次，量级约 220 秒。以下算量来自 28 层、5123 个 query、13004 个 key、2048 hidden 和 6144 intermediate。MAC 表示一次乘加。对激活的读写、归一化和位置编码未折算为 MAC。

| 每层主要工作 | 已知成本 | 计时边界 |
| --- | --- | --- |
| Q/K/V 加输出投影 | 64.46 G MAC/层/次 | 尚无独立耗时 |
| 门控 MLP 三次线性变换 | 193.39 G MAC/层/次 | 尚无独立耗时 |
| 非因果 attention，QK 与 AV | 272.87 G MAC/层/次；0.05990 秒/层 | 真实尺寸合成激活、原 256 分块，独立预热。包含上层 K/V 展开；乘 28 约 1.68 秒/次，和真实首次 velocity 的计时口径不同 |
| 拼接 AR/NAR 的 K、V | 按 BF16 输出尺寸共 50.80 MiB/层/次，整首约 88.90 GiB 累计写入 | 无独立耗时；行数含可见 AR 与 NAR，上述为数据量而非驻留内存 |

要从 NAR 省 1 或 2 分钟，每次 velocity 需实省约 0.94 或 1.88 秒。已测整块 attention 路径将首次真实 velocity 从 3.435 降到 3.180 秒，线性外推约 16 秒，输出相对 RMS 差 1.20%。现有记录没有能解释两分钟净收益的单项改动。只凭 MLP 的 MAC 或拼接字节数都无法推导实际秒数。没有启动新的 NAR 探针。

## 复现与试听

在仓库根目录运行，先查看原账本的剩余额度。GPU 命令记录实际实验调用，累计余额不足时入口会拒绝。两组至少冷却 300 秒。

```bash
ledger=runs/probes/mps-performance
run=runs/20261117_165850_tonight_awake_folk_f9176f
envs/seed-vc-torch215/bin/python scripts/diagnostics/profile_seed_phrase.py prepare "$ledger"
envs/control/bin/python scripts/diagnostics/profile_mps.py seed-phrase --run "$run" --output "$ledger" --deadline 38
envs/seed-vc-torch215/bin/python scripts/diagnostics/profile_seed_phrase.py compare "$ledger/13_seed-phrase"
envs/control/bin/python scripts/diagnostics/profile_mps.py yue-head --run "$run" --output "$ledger" --deadline 11
```

[FP32 乐句](../runs/probes/mps-performance/13_seed-phrase/fp32.wav) · [FP16 乐句](../runs/probes/mps-performance/13_seed-phrase/fp16.wav) · [源人声](../runs/probes/mps-performance/seed-phrase-source.wav)。诊断只在独立子进程中选择候选精度及投影，正常推理使用原配置，无需撤回上游补丁。

## YuE2 语义阶段与取舍

真实 BF16 输出层权重，固定合成 hidden，沿用真实语义 token 历史及采样配置。两种投影先各预热，再以原版、候选、候选、原版顺序测量，每批 8 次。允许的 32,769 行连续权重直接取 view，候选在原词表上恢复 logits 后执行原采样，CPU 抽样、标量读取与下一 token 的 MPS 传输也计入净时间。计时区间前后同步 MPS。

| 每 token 平均 | 全词表 | 允许行投影 |
| --- | ---: | ---: |
| 纯投影 | 6.04 ms | 1.09 ms |
| 含恢复、采样和 token 回传 | 9.39 ms | 4.87 ms |

三个检查用例的允许位置 logits 与全词表概率逐值一致，固定随机 token、RNG 状态和贪心结果一致。对齐 EOS 的 hidden 在最短长度之前正确屏蔽 EOS，达到长度后两路 greedy 均选择 EOS。局部净省约 4.53 ms/token，乘以 5,122 次约 **23.2 秒**。以此替代前轮 33–40 秒的理论估计。测量含固定 hidden、单分支和连续调用，真实模型权重流、长上下文与热降频可能改变实际收益。候选保留用于语义生成，正式全曲入口未切换。

从此前约 19 分钟的候选组合出发，若 Seed-VC 两种外推情景及 YuE2 的 23 秒均能兑现，对应约 **16–17 分钟**。人工试听与整曲衔接仍待验收。整曲重跑 0 次。两次 worker 用时 20.50、1.61 秒，间隔 345.55 秒，共享账本累计约 272.22 / 300 秒。控制环境测试 31 项通过、7 项隔离环境测试跳过。原始计时、全部 30 步误差与账本条目见 [机器可读结果](../configs/macos/optimization-round3.json)。
