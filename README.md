# 音乐换风格与参考音色转换 baseline

项目实现了：**原曲 → SheetSage2 旋律转录 → YuE2 换风格 → BS-RoFormer 人声分离 → Seed-VC 参考音色转换 → 对齐和混音**。使用通用预训练权重，不训练目标人物模型。

当前支持可选的音区分析与谱面移调，已完成《落叶归根》爵士摇滚版及《秋意浓》乡村版的整首推理。技术路线、模型分工和已知限制见 [Baseline 技术报告](docs/07-baseline-report.md)。

此仓库保存项目代码、测试、文档、模型版本及环境记录；YuE 和 Seed-VC 上游代码以固定 commit 的 Git 子模块管理。权重、运行环境、歌曲、歌词、参考录音和生成音频保留在本地。克隆及恢复说明见 [仓库与本地资源说明](docs/08-repository.md)。文档中的 `runs/...` 试听链接仅在拥有本地运行产物时可用。

先阅读 [使用说明](USAGE.md)：只需要准备原曲、歌词、目标人物参考录音，并给出目标风格。

已于 2026-10-21 完成整首端到端运行：输出 204.72 秒、48 kHz、立体声、24 bit WAV。六个阶段共约 167 秒，不含首次安装、下载和附加评价。直接打开 [已验证演示试听页](runs/demo_verified/deliverables/listen.html) 或 [最终音频](runs/demo_verified/deliverables/final.wav)。实测详情见 [验证记录](VALIDATION.md)。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/run_baseline.sh --demo
```

结果在命令最后打印的 `runs/<运行编号>/deliverables/`。打开 `listen.html` 对照试听，`final.wav` 是最终混音。每次运行均保存参数、输入及输出哈希、阶段日志和检查报告。

| 位置 | 内容 |
| --- | --- |
| `USAGE.md` | 数据准备、执行命令、结果验收、失败恢复 |
| `scripts/run_baseline.sh` | 统一入口，无需手动切换环境 |
| `src/music_cover/` | 调度、音频检查、对齐和混音 |
| `scripts/stage_runner.py` | 各模型的独立进程适配 |
| `data/raw/` | 原曲、歌词、参考音频 |
| `data/demo_sources.json` | 演示素材来源与哈希 |
| `configs/` | 演示配置、模型 revision、本地资产哈希和实测环境版本 |
| `models/`、`envs/`、`third_party/` | 已下载权重、环境、上游代码 |
| `runs/` | 独立运行记录和交付音频 |
| `docs/` | 01–06 为设计存档，07 为当前技术报告，08 为仓库恢复说明；当前操作以 USAGE.md 为准 |

演示采用官方 AI 生成中文歌曲《今晚不眠》及配套歌词，参考音色采用公开 LibriSpeech 语音。它验证链路能否执行；实际目标人物的相似度需要替换参考录音后试听。

RVC / so-vits-svc 未进入当前实现，因为用户只有参考录音，没有相应目标音色模型。当前换声后端为零样本 Seed-VC。

本机配置为单张 H200 串行推理，YuE2 显存预算 80 GiB。虚拟环境复用了 /opt/conda 的基础 CUDA 包；没有修改共享基础环境或上游仓库代码。音频生成质量和目标音色相似度须单独验收，自动检查不会替代听感判断。
