# 原 main 流水线的 M5 优化组合整曲实测

2026-11-18，无风扇 M5、24 GB 统一内存。完整命令从启动、五环境预检到交付 WAV **995.66 秒（16 分 35.66 秒）**。按项目交付命令另行编码 320 kb/s MP3 用时 1.21 秒，两次命令的实际执行时间合计 **996.87 秒**。CPU 验收另计。六个成功阶段的计时之和为 986.97 秒，历史成功阶段合计为 2925.37 秒，同口径缩短 66.26%，提速 2.96 倍。距 600 秒目标仍差 395.66 秒（按生成命令）。

| 阶段 | 原 M5 基线，秒 | 本次，秒 | 本次 MPS driver 峰值，GiB |
| --- | ---: | ---: | ---: |
| 准备 | 0.35 | 0.38 | — |
| SheetSage2 | 39.72 | 43.23 | 15.94 |
| YuE2 | 1978.73 | 494.98 | 9.66 |
| BS-RoFormer | 215.38 | 159.15 | 2.88 |
| Seed-VC | 687.26 | 285.21 | 4.40 |
| 混音 | 3.94 | 4.02 | — |

YuE2 中 AR 为 211.02 秒、32 步 midpoint 声学合成为 259.08 秒、VAE 为 16.35 秒。Seed-VC 完成 11 块、每块 30 步。YuE2 与 Seed-VC 合计 780.19 秒，约占阶段耗时的 79%。本次为一次整曲运行，无法据此确定持续负载下的耗时分布或各补丁的独立贡献。

## 配置与验收

复用原曲、歌词、参考文件和起点、风格、种子 42、移调 0 和 16 GiB 预算。原始素材路径和准备阶段的输出哈希经核对相同。新目录中每个阶段均为第一次执行。控制与转谱环境使用 Torch 2.8；YuE2、分离、Seed-VC 均使用已锁定的 Torch 2.15。四个模型阶段都记录了真实 MPS 前向，禁止隐式回退。

隔离副本使用原生 AR GQA、单 CFG 分支的语义允许行投影、现有 NAR 分块、广播非因果 key mask、等头数免复制和 DiT 局部 FP16。Seed-VC 的状态、时间网格及 BigVGAN 为 FP32。DiT 的 FP16 输出参与 CFG 和增量计算，再与 FP32 状态相加。Seed-VC 中间文件用 soundfile 的 FLOAT 写出。两份新补丁通过既有的固定 revision 可选补丁入口管理。主工作树的默认推理设置仍使用基线配置。

新旧 ABC 谱面逐字节相同。新旧语义序列分别为 5119 与 5121 个内容 token，首次不同在第 52 个。新音频长 204.7587 秒，旧音频长 204.8387 秒，两版均自然结束。全链路组合有多项同时生效，不能从这次对比定位造成序列差异的单一原因。MPS driver 峰值按 0.2 秒采样，最高为转谱的 15.94 GiB。

交付 WAV 为 48 kHz、双声道、24 bit PCM，共 9,828,416 帧。成品与中间音轨数值有限；按 1 秒窗口统计，成品没有低于 −80 dBFS RMS 的窗口。估算真峰值 −1.10 dBTP。首、中、末三个窗口均测得 10 ms 人声延迟，混音已校正 480 个样本，换声原始时长差 −5.125 ms。MP3 经 FFmpeg 完整解码，采样率、声道和帧数均与 WAV 一致。旧成品 WAV 哈希复核通过。用户已试听本次新版整曲，反馈听感没有问题。

新 MP3、WAV 和分轨试听页分别保存在本地 `runs/m5-full-candidate/runs/20261118_133412_tonight_awake_folk_optimized_56881f/deliverables/` 下的 `final.mp3`、`final.wav` 与 `listen.html`。旧 MP3 位于本地 `runs/20261117_165850_tonight_awake_folk_f9176f/deliverables/final.mp3`。[机器可读核验](../configs/macos/optimized-full-song.json)

## 以后运行

当前项目已提供可从锁文件重建的优化入口。安装、运行和恢复步骤见 [Mac 使用说明](09-macos.md)。

## 本次实际命令

工作目录为本机仓库下的 `runs/m5-full-candidate`，其中保留了实际执行时的两个源码副本和五个环境的链接。YuE2 的 editable 安装原指向主工作树，运行时显式优先导入本副本的源码。下列命令记录实际运行方式；回看和复核现有成品无需重跑生成。

```bash
cd runs/m5-full-candidate
env PYTHONPATH="$PWD/third_party/YuE/src" HF_HUB_OFFLINE=1 HF_HOME="$PWD/cache/huggingface" MPLCONFIGDIR=/tmp/music-mpl-cache PYTHONUNBUFFERED=1 /usr/bin/time -p bash scripts/run_baseline.sh --demo --device mps --memory-budget-gib 16 --transpose 0 --name tonight_awake_folk_optimized
```

在新的隔离源码副本上，候选补丁的选择如下。主工作树与旧成品保留，可直接切回原入口。整曲记录中的源码哈希对应隔离副本；归档的 FP16 补丁与当前可选补丁仅有一行注释措辞不同。

```bash
envs/control/bin/python scripts/apply_compat_patches.py --candidate yue2-native-ar-gqa --candidate yue2-mps-semantic-head --candidate seed-vc-mps-dit-fp16 --candidate seed-vc-equal-heads --candidate seed-vc-key-mask --candidate seed-vc-soundfile
```

控制环境 33 项测试通过，7 项 YuE 专用测试在候选环境的 CPU 上另行运行并全部通过。本次仅实测该 M5，CUDA 实机回归未执行。
