# 音乐换风格与参考音色转换 baseline

**逐步上手：**[孙燕姿＋秋意浓＋Bossa Nova / Samba Jazz，00–11 完整命令与结果路径](docs/11_modular_walkthrough.md)。示例输出为 `runs/lhd_test_modular`；每步均说明使用哪个上游产物、应查看哪些文件。版本变化见 [CHANGELOG.md](CHANGELOG.md)。

**模型准备：**[模型权重清单与准备方法](docs/12_model_weights.md)，集中列出 13 组模型、阶段对应关系、下载命令、文件路径、固定版本和校验方法。

项目实现了：**原曲 → SheetSage2 旋律转录 → YuE2 换风格 → BS-RoFormer 人声分离 → Seed-VC 参考音色转换 → 对齐混音 → 成品分轨、MIDI 和专业评审包**。使用通用预训练权重，不训练目标人物模型。

当前版本 **v0.2**（分支 `v0.2`），基于 `v0.1` 完成模块化，支持可选的双向音区分析与谱面移调（默认 auto 候选 -6～+6），默认输出 WAV、MP3 和评审包。《秋意浓》Synthwave / Dance-pop 与《喜欢你》Bossa Nova / Samba Jazz 均已完成整首推理；后者在 2026-10-23 通过 122 个评审文件的独立检查。目标风格和参考音色不等于听感验收结论。技术路线、模型分工和限制见 [Baseline 技术报告](docs/07-baseline-report.md)。

此仓库保存项目代码、测试、文档、模型版本及环境记录；YuE 和 Seed-VC 上游代码以固定 commit 的 Git 子模块管理。权重、运行环境、歌曲、歌词、参考录音和生成音频保留在本地。克隆及恢复说明见 [仓库与本地资源说明](docs/08-repository.md)。文档中的 `runs/...` 试听链接仅在拥有本地运行产物时可用。

模块化首轮验证已完成：33 项测试通过；孙燕姿《秋意浓》Synthwave / Dance-pop 的 12 个模块完整执行，最终 WAV/MP3 与 v0.1 哈希一致；单独重跑 MIDI 通过，再次续跑全部复用。157 个评审文件及所有阶段产物通过独立检查，听感质量仍需人工验收。详见 [验证记录](VALIDATION.md)。

双向选调 v2 已通过 45 项测试及完整 12 阶段验证：现有《秋意浓》实际采用 -2 半音（Am → Gm），生成音频对齐后的实测中位变化约 -2.14 半音。新的评审交付包含 12 条音轨、10 个主 MIDI 和 167 个校验文件；听感仍需人工验收。

新增双向选调的配置、推荐/实际值、MIDI 与音频核验见 [双向选调说明](docs/10_bidirectional_pitch.md)。

模块化入口、阶段输入输出和重跑方式见 [模块化说明](docs/09_modular_pipeline.md)。12 个 `scripts/00_prepare`～`scripts/11_delivery` 子目录均可独立运行；默认完整执行保留原有使用方式。

逐模块操作示例（孙燕姿＋秋意浓＋Synthwave / Dance-pop）：

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"

# 00：读取原曲、歌词、参考录音，创建运行并保存配置
bash scripts/00_prepare/run.sh \
  --config configs/tasks/qiu_yi_nong_synthwave_sun_yanzi.yaml \
  --run runs/qiu_yi_nong_modular_test

# 01：使用同一运行中 00 产出的 source.wav 转写原曲乐谱
bash scripts/01_source_score/run.sh --run runs/qiu_yi_nong_modular_test

# 从 02 继续：读取 01 的 score.abc 和 00 的音频，直到最终交付
bash scripts/run_baseline.sh --run runs/qiu_yi_nong_modular_test --from 02_pitch_adaptation
```

例如，01 读取 `runs/qiu_yi_nong_modular_test/stages/00_prepare/attempt_001/artifacts/source.wav`，输出 `runs/qiu_yi_nong_modular_test/stages/01_source_score/attempt_001/artifacts/score.abc`；随后 02 使用该乐谱及 00 的原曲和参考片段。`attempt_001` 是首次尝试的示例，实际使用的成功 attempt 由程序从运行记录解析。所有模块沿用同一个 `--run`；只有初始化时需要提供任务配置。各子目录 README 均列出本模块的完整命令、具体输入路径和输出文件。

先阅读 [使用说明](USAGE.md)：只需要准备原曲、歌词、目标人物参考录音，并给出目标风格。

v0.1 上传前复核（2026-10-23）：22 项测试和完整环境预检通过。《喜欢你》现有结果包含约 269.84 秒成品、12 条评审音轨、10 个主 MIDI 文件；技术检查通过，人工听音与乐谱校对待完成。首次演示和后续验证的日期、耗时、哈希及质量问题见 [验证记录](VALIDATION.md)。旧运行及便捷链接已清理，不作为当前下载入口。

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
bash scripts/run_baseline.sh --demo
```

结果在命令最后打印的 `runs/<运行编号>/deliverables/`。每次运行默认同时输出最终混音 `final.wav`（48 kHz、立体声、24 bit）和 `final.mp3`（192 kbps，便于下载和分享）。打开 `listen.html` 可对照试听并下载。每次运行均保存参数、输入及输出哈希、阶段日志和检查报告。

现在默认还生成 `deliverables/review/` 专业评审包：转换后的主唱／背景人声、鼓、贝斯、吉他、钢琴、其他乐器、残留与残差音轨；成品重新识别的 MIDI、音符 CSV、和弦／调性／拍点／段落、模型激活和人工评分表。入口为 `review/index.html`，MIDI 可整体下载为 `review/midi_bundle.zip`。所有评审音轨统一为 48 kHz 立体声 32-bit float、同起点同长度。自动估计不等于真实分轨或人工乐谱；鼓 MIDI 是频带起音候选，必须校对。

新增阶段默认启用；只需要原来的基础交付时加 `--no-review`。新增环境和模型安装、轨道相加规则、验证命令见 [使用说明](USAGE.md)。

| 位置 | 内容 |
| --- | --- |
| `USAGE.md` | 数据准备、执行命令、结果验收、失败恢复 |
| `scripts/run_baseline.sh` | 统一入口，无需手动切换环境 |
| `src/music_cover/modules/` | 12 个逻辑阶段，各自声明依赖和产物 |
| `src/music_cover/backends/` | 各模型/评估/打包的独立进程适配 |
| `src/music_cover/engine.py` | 依赖校验、分阶段缓存、运行锁、原子发布 |
| `scripts/setup/` | 下载、环境记录、数据准备，原脚本路径保留兼容入口 |
| `scripts/stage_runner.py` | 原后端命令的兼容入口 |
| `data/raw/` | 原曲、歌词、参考音频 |
| `data/demo_sources.json` | 演示素材来源与哈希 |
| `configs/` | 演示配置、模型 revision、本地资产哈希和实测环境版本 |
| `models/`、`envs/`、`third_party/` | 已下载权重、环境、上游代码 |
| `runs/` | 独立运行记录和交付音频 |
| `docs/` | 01–06 为设计存档，07 为当前技术报告，08 为仓库恢复说明，09 为模块化执行契约，10 为双向选调，11 为逐步实操，12 为模型权重指南；当前操作以 USAGE.md 为准 |

演示采用官方 AI 生成中文歌曲《今晚不眠》及配套歌词，参考音色采用公开 LibriSpeech 语音。它验证链路能否执行；实际目标人物的相似度需要替换参考录音后试听。

RVC / so-vits-svc 未进入当前实现，因为用户只有参考录音，没有相应目标音色模型。当前换声后端为零样本 Seed-VC。

当前部署每次使用一张 H200 串行推理，YuE2 显存预算 80 GiB。虚拟环境复用了 `/opt/conda` 的基础 CUDA 包；环境快照记录的是已配置服务器，不能直接当作跨机器安装清单。完整评审流程的耗时和磁盘占用高于历史基础流程。音频生成质量和目标音色相似度须单独验收，自动检查不会替代听感判断。
