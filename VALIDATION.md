# 实际验证记录

## 2026-11-18 优化版 M5 整曲

该组合在 M5、24 GB 上完成《今晚不眠》生成。命令到 WAV 用时 995.66 秒，MP3 另用 1.21 秒。WAV 和 MP3 完整性、对齐和峰值检查通过。用户已试听新版整曲，反馈听感没有问题。旧版与优化版的阶段合计分别为 2925.37 秒、986.97 秒。[完整计时与试听](docs/15-mps-full-song.md) · [机器可读验收](configs/macos/optimized-full-song.json)

## 2026-11-17：M5 24 GB / macOS MPS

完整运行：`runs/20261117_165850_tonight_awake_folk_f9176f`。输入为 203.48 秒演示歌曲，默认不移调；最终输出为 204.838667 秒。主仓库基于 `90cb742b7160a2479b75d4a141d82e8fb3026a78`，上游固定 commit 及其实际补丁源码哈希、逐阶段设备与内存证据保存在 `configs/macos/validation.json` 和运行目录。安装、运行与恢复方法见 [Mac 文档](docs/09-macos.md)。

执行命令：`bash scripts/run_baseline.sh --demo --device mps --memory-budget-gib 16`。所有核心模型在 MPS 执行，`PYTORCH_ENABLE_MPS_FALLBACK=0`；实际学习层前向 hook 校验参数和输入设备，转谱、生成、分离、换声均有运行记录。YuE2 使用 torch-eager、BF16 AR/NAR、FP32 VAE，32 步声学合成；没有量化、缩短整曲或降低生产生成长度。

| 阶段 | 成功阶段耗时秒，含加载 |
| --- | ---: |
| 输入准备 | 0.35 |
| SheetSage2 / MERT 整曲转谱 | 39.72 |
| YuE2 换风格 | 1978.73 |
| BS-RoFormer 分离 | 215.38 |
| Seed-VC 换声 | 687.26 |
| 对齐混音 | 3.94 |
| 合计 | 2925.37（48.76 分钟） |

以上为成功阶段合计，不包含下载、首次编译和失败/中断的尝试。首次整曲 NAR 遇到 16 GiB MPS 内存限制；启用上游闲置 AR 权重暂存，并在阶段切换时释放空闲 MPS 缓存后，保持 32 步 NAR 完成整曲。最终 YuE2 采样峰值为 tensor 8.36 GiB、driver 16.21 GiB，进程峰值 RSS 0.87 GiB；这些指标不能相加作为统一内存总用量，16 GiB allocator 预算也不是 driver 或系统内存硬上限。采样间隔为 0.2 秒，可能漏掉短暂峰值。

输出为 48 kHz、立体声、PCM 24 bit，共 9,832,256 帧。所有音频检查通过，没有非有限值、空音频或生成截断；峰值工程估计为 −1.10 dBTP。换声时长差约 −3.85 ms，三个对齐窗口为 0 / 10 / 10 ms，应用 −480 样本平移。WAV SHA256：`3ee0f331f461a5adebbc77001831ebf94ea4f405954ef686bc2ad08f954cb557`。另用 FFmpeg 导出 320 kbps MP3，完整解码通过，时长与 WAV 一致。

恢复运行复用全部六个阶段。26 项单元测试通过，覆盖原有音频、对齐、谱面、长度和恢复逻辑，以及 GPU 选择、禁止回退、设备切换恢复、源码变化缓存失效、损坏的 YuE2 检查点和语义/latent 对应关系。188 个模型资产按原始 manifest 的 SHA256 校验通过，五个独立环境均通过导入及真实 MPS 矩阵计算预检。

默认关闭移调已通过整曲验证；自动及显式移调另用同一首歌 30–50 秒片段做端到端检查。自动模式选择 0 半音，9 阶段全部成功，耗时 213.84 秒，输出 20.918667 秒；这是算法对当前参考的实际选择，不代表自动模式一定需要升调。

显式 `--transpose 1` 的 9 阶段也全部成功，耗时 213.56 秒，输出 20.918667 秒。62 个人声音符逐事件精确升高 1 个半音，调号 F → Gb，时值、拍号和速度不变；RMVPE 前后检查均在 MPS。运行目录为 `runs/20261117_181859_mps_short_manual_1e7a32`，自动模式对照为 `runs/20261117_181433_mps_short_auto_913236`。两次重新生成音频的共同有声帧中，换声前/后的音高差中位数为 +1.07 / +1.14 半音；未经时间规整的逐帧统计中，仅 24.39% / 22.92% 落在目标升调量 ±50 cents 内，因此不能据此宣称音频逐音符精确移调。谱面转换已验证，生成音频对旋律的遵从仍需试听。

人工听感、歌词识别质量和本次改动后的 CUDA 实机回归尚未验收。历史 NVIDIA 演示输出为 204.718667 秒，本次 MPS 为 204.838667 秒，两者均无生成截断；这只是整曲结构/时长对照，不是听感等价或跨设备数值一致的证明。不得将当前 M5 实测外推为所有 Mac 内存档位已验证。

试听入口：[MPS 整曲对照](runs/20261117_165850_tonight_awake_folk_f9176f/deliverables/listen.html)、[WAV](runs/20261117_165850_tonight_awake_folk_f9176f/deliverables/final.wav)、[MP3](runs/20261117_165850_tonight_awake_folk_f9176f/deliverables/final.mp3)。文件位于本地，未随 Git 上传。

## 2026-10-22：孙燕姿参考音色《落叶归根》升调

完整运行：`runs/20261022_022204_luo_ye_gui_gen_sun_yanzi_pitch_adapted_15e6f1`；便捷入口：`runs/sun_yanzi_transposed_verified`。输入、曲风、seed 42、30 步换声和 16 秒参考均与旧版 `20261021_112422_luo_ye_gui_gen_rock_jazz_sun_yanzi_d0d44c` 一致；新增 `--transpose auto`。

自动选调基于分离后的原曲人声与参考歌声，选择 +3 个半音（Ab → B）。12 项测试通过，覆盖跨小节连音、跨八度临时变音、调号、和弦及转位、变拍、音域越界、低音匹配与高音约束，以及原有音频和缓存测试。测试命令：`PYTHONPATH=src envs/control/bin/python -m unittest discover -s tests -v`。

实际 9 阶段全部成功，耗时合计 262.84 秒；整曲 307.278667 秒、48 kHz、立体声、PCM 24 bit，峰值工程估计 −1.10 dBTP，生成无截断。最终 WAV SHA256：`d7ec38201b8830b7e38f033c97712384eba6c571cb6a5c28cfd0b1b9f838607b`。MP3 已完整解码检查。

原始转谱与旧版哈希一致；移调后逐事件检查了 372 个人声音符、99 个器乐音符，全部升 3 个半音，起点、时值、拍号和速度不变。原谱、移调谱和选调依据均随结果交付。

RMVPE 实测换声前全曲中位音高 236.75 → 282.63 Hz（约 +3.07 个半音）；最终人声 30–60 秒中位音高 165.21 → 206.28 Hz。独立 pYIN 的 40–56 秒检查为 164.74 → 205.17 Hz。新版换声前后共同有声帧中绝对音高差中位数 10.01 cents，99.19% 不超过 100 cents。新旧重新演唱的时长不同，统计差异不能解释为每个音符精确移调量。

音色自然度和粗糙程度尚未人工试听验收。前奏 0–30 秒换声前没有达到分析门限的有声帧，换声后约 2 秒，提示可能仍有额外声响；此项未通过升调修复。当前 16 秒参考只支持样本音区估计，不代表歌手完整音域。

验收入口：[新旧试听对照](runs/sun_yanzi_transposed_verified/deliverables/pitch_comparison/listen.html)、[详细报告](runs/sun_yanzi_transposed_verified/deliverables/report.md)。独立复核脚本为 `scripts/verify_transposition.py`；结果及窗口统计保存在该运行的 `deliverables/pitch_comparison/`。

## 2026-10-21：首次完整演示记录

验证日期：2026-10-21。成功运行目录：`runs/20261021_100959_tonight_awake_folk_df5b52`，便捷入口：`runs/demo_verified`。

## 端到端执行

执行命令：`bash scripts/run_baseline.sh --demo`。这次运行的六个阶段均从输入开始执行成功，没有复用之前试运行的模型结果。其后另行执行 `--resume` 检查缓存恢复。

| 阶段 | 耗时秒，包含进程启动与模型加载 |
| --- | ---: |
| 输入准备 | 1.34 |
| 整首旋律转录 | 26.83 |
| YuE2 换风格 | 82.69 |
| 人声伴奏分离 | 18.25 |
| 整首参考音色转换 | 27.82 |
| 对齐与混音 | 10.20 |
| 合计 | 167.13 |

输入为 203.48 秒的官方 AI 生成歌曲及配套歌词；输出为 204.718667 秒。目标参考为公开 LibriSpeech 语音，不是用户目标人物。

最终音频：48 kHz、2 声道、PCM 24 bit，9,826,496 帧。文件 SHA256：`b6361d76669a643b7f83cfa39391416a6a072afc91cf3cbe03362559730c6588`。

YuE2 的 `abc` 和 `semantic` 截断标记均为 false。换声输出相对分离人声短 11.56 ms，混音时补齐尾部；三个对齐窗口估计延迟为 10 / 0 / 0 ms，中位数为 0，没有施加额外固定平移，也没有对整首进行拉伸。

最终峰值的 4 倍过采样工程估计为 −1.10 dBTP，无越界采样或非有限数值。所有音频可解码且非空。具体数值在 `deliverables/checks.json`；这不是认证级真峰值测量。

## 额外检查

六项针对歌词重复时间戳、重采样时间点、固定延迟恢复、混音峰值和时长、缓存损坏重建、失败状态处理的测试均通过。测试命令：`PYTHONPATH=src envs/control/bin/python -m pytest tests -q`。

20 秒换声探针也已执行，产物在 `runs/probes/seed_20s/`，输出 19.992381 秒。Seed-VC 的有效参数已核对，未发现缺失或尺寸不匹配的可训练参数；上游日志中的跳过项属于未使用的输出分支和位置缓存，详情在 `configs/seed_checkpoint_audit.json`。

整首音高、ASR 和音色诊断由 `scripts/verify_result.py` 执行，结果在 `deliverables/verification.json`。音色诊断使用了换声模型也在使用的 CAMPPlus，不能作为独立测评。

实测音高绝对差中位数为 8.56 cents；共同有声帧中 91.91% 的差异不超过 100 cents，7.44% 超过 600 cents，后者需要试听复核，也可能包含音高估计的倍频误差。CAMPPlus 与参考录音的平均余弦相似度由 0.077 提升到 0.585。

**歌词识别没有通过质量验收。** Whisper-small 相对输入歌词的原始字符错误率为换声前 83.80%、换声后 62.28%。识别文本存在漏识别、同音字和简繁体混用，因此不能将该比例等同于实际漏词比例；应听 `restyled.flac` 和 `final.wav` 逐段核对。报告已将高 ASR 误差和音高异常列为待试听项，没有将这些结果包装为听感合格。

## 实测环境

所有环境为 Python 3.11.13。YuE2 使用 Torch 2.10.0+cu128 / Transformers 4.57.6；转录使用 Torch 2.8.0+cu128 / Transformers 4.45.2；换声使用 Torch 2.8.0+cu128 / Transformers 4.46.3；分离使用 audio-separator 0.30.2 / Torch 2.8.0+cu128。

SheetSage2 使用当前固定版本的 `default` preset，由 FFmpeg 8.0 命令行解码；未采用要求 FFmpeg 6 共享库的 `paper` preset。分离使用 PyTorch CUDA；日志里的 ONNX CPU 提示不代表 BS-RoFormer 在 CPU 执行。

精确环境和上游 commit 在 `configs/environments.lock.json`，各环境局部安装清单为 `configs/*.installed.txt`；模型 revision 和下载后计算的 SHA256 在相应 lock 文件中。上游仓库没有修改。

## 验收边界

已验证整首技术链路、输出格式、时长和对齐，以及缓存恢复。没有进行人工听感验收，不能据此宣称目标音色、自然度和曲风达到生产要求。此处也尚未验证用户目标人物的参考录音或商业歌曲输入。

建议按 `USAGE.md` 中的试听顺序，重点听主副歌、高音、间奏前后及结尾。之前的失败安装和适配试运行保留在 `cache/` 和其他 `runs/` 目录，当前成功结果以 `runs/demo_verified` 为准。
