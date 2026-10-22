> 设计存档：本文件记录实施前的方案，部分环境、文件名和步骤已调整。当前使用方法以 [USAGE.md](../USAGE.md) 为准；实测版本见 `configs/environments.lock.json`，实测结果见运行目录的 `deliverables/report.md`。

资料核对日期为 2026-10-21。以下链接用于说明上游能力和调用方式；main 分支会变化，后续实验以本地锁定的 commit、模型 revision 和依赖清单为准。

| 官方资料 | 支持的设计内容 |
| --- | --- |
| [YuE2 仓库](https://github.com/multimodal-art-projection/YuE) | 模型定位、48 kHz 立体声输出、Python 3.12 与部署起点 |
| [YuE2 Cover 指南](https://github.com/multimodal-art-projection/YuE/blob/main/docs/covers.md) | SheetSage2 转录、melody-only、ABC 条件、cot=melody、歌词与段落对齐 |
| [SheetSage2 模型卡](https://huggingface.co/m-a-p/SheetSage2) | 转录模型、依赖及 MERT2 编码器 |
| [YuE2 生成入口](https://github.com/multimodal-art-projection/YuE/blob/main/examples/generate.py) | request 与 ABC 输入、生成输出的实际接口 |
| [Seed-VC 仓库](https://github.com/Plachtaa/seed-vc) | 零样本参考音频换声、44.1 kHz 歌声模型和参数 |
| [Seed-VC 推理代码](https://github.com/Plachtaa/seed-vc/blob/main/inference.py) | F0 分支默认模型、附属权重、单声道读取、分块、参考截断和输出 |
| [Seed-VC 歌声权重](https://huggingface.co/Plachta/Seed-VC/blob/main/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth) | 当前 F0 分支使用的权重资产 |
| [Seed-VC 依赖文件](https://github.com/Plachtaa/seed-vc/blob/main/requirements.txt) | 安装前需处理的 PyTorch 版本与索引声明 |
| [Audio Separator](https://github.com/nomadkaraoke/python-audio-separator) | 现成 CLI、BS-RoFormer 模型支持与文件输出 |
| [分离输出实现](https://github.com/nomadkaraoke/python-audio-separator/blob/main/audio_separator/separator/common_separator.py) | 输出采样率、归一化和音频写出行为 |
| [分离默认参数](https://github.com/nomadkaraoke/python-audio-separator/blob/main/audio_separator/separator/separator.py) | 默认采样率、峰值门限、写出后端 |
| [分离峰值缩放](https://github.com/nomadkaraoke/python-audio-separator/blob/main/audio_separator/separator/uvr_lib_v5/spec_utils.py) | 超出门限时的逐轨缩放行为 |
| [RVC CLI 文档](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/cli.md) | 目标模型、说话人和检索索引的推理条件 |
| [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) | 模型、配置和目标说话人的推理要求 |
| [歌曲宝样本页](https://www.gequbao.com/music/6762) | 歌曲下载入口与歌词页面；具体文件质量需下载后检查 |

本项目作出的工程选择包括：单卡串行、独立环境、固定文件接口、先做短段换声探针、以生成歌曲为混音时间轴、保留 A／B／C 对照，以及按内容哈希恢复阶段。这些是项目设计，不是上游保证。

仍需通过首轮实际运行确定的事项如下。

| 待验证项 | 验证方式 | 对方案的影响 |
| --- | --- | --- |
| 目标人物参考录音能否支持自然歌声 | 20–30 秒探针与参考独听对比 | 决定零样本换声是否达到团队可接受水平 |
| 原曲转录的旋律准确性 | ABC 检查、旋律试听、原曲对照 | 决定是否需要人工修谱 |
| YuE2 对风格与歌曲内容的平衡 | 固定谱词和 seed，听目标风格输出 | 决定风格描述及 melody/full 对照是否必要 |
| 分离的主唱残留和电平变化 | 人声／伴奏独听，A／B 重混对照 | 决定是否更换分离配置或增加去混响 |
| Seed-VC 的长音频对齐 | 短段、整曲及三个时间位置的对齐检查 | 决定是否需要处理额外延迟或分块 |
| 依赖与 CUDA 兼容 | 四个独立环境分别最小推理 | 决定最终 environments.lock 内容 |
| 实际吞吐和显存 | 每阶段记录加载、推理和峰值 | 决定后续批量和并行方案 |

只有通用预训练模型和参考录音的前提下，RVC 与 so-vits-svc 不能按其常规路径直接产生任意目标人物音色。因此首版采用 Seed-VC；若零样本质量不足，下一步应明确比较其他零样本歌声后端、改进参考素材，或另立目标音色训练实验，不能把训练悄悄加入“免训练”结果。
