> 设计存档：本文件记录实施前的方案，部分环境、文件名和步骤已调整。当前使用方法以 [USAGE.md](../USAGE.md) 为准；实测版本见 `configs/environments.lock.json`，实测结果见运行目录的 `deliverables/report.md`。

首版按 Linux、单张 NVIDIA GPU、各阶段串行执行部署。当前工作机资源充足，优先解决环境兼容和模型完整性，不引入多卡切分、量化或服务化。

2026-10-21 对本机做了只读检查，结果如下。GPU 空闲量和磁盘空间是当时快照，不代表后续运行的资源预留。

| 项目 | 检查结果 | 对实施的影响 |
| --- | --- | --- |
| 工作根目录 | /data/liuhongda/music-cover-baseline | 项目文件和后续产物使用独立子目录 |
| GPU | 8 张 NVIDIA H200；每卡 143771 MiB 显存 | 首版选择一张卡即可，不默认占用全部设备 |
| NVIDIA 驱动 | 565.57.01 | 在各环境中验证 PyTorch CUDA 实际可用 |
| 当前 Python | /opt/conda/bin/python3，3.11.13 | 不直接改动共享 base 环境 |
| 当前 FFmpeg | /opt/conda/bin/ffmpeg，8.0 | SheetSage2 另备其文档要求的 6.1 及共享库 |
| 环境工具 | conda、uv 均在 PATH | 可为每个后端创建独立环境 |
| CUDA 编译器 | 当前 PATH 未找到 nvcc | 不等同于没有 CUDA；优先验证预编译推理依赖 |
| /data 可用空间 | df 显示约 1.5 TiB 可用，共享文件系统 | 下载前重新检查可用空间和项目配额 |

YuE2 官方给出的起点是 Python 3.12、支持 BF16 的 NVIDIA GPU 和 24 GB 显存。当前硬件规格高于这个显存起点，但端到端运行时长和峰值占用仍由首个样本测量。[YuE2 官方安装说明](https://github.com/multimodal-art-projection/YuE#quick-start)

计划建立四个模型环境，调度器使用独立的轻量 Python 环境或最初的手工调度，不将所有模型依赖合并。

| 环境位置 | Python 与依赖起点 | 首个检查 |
| --- | --- | --- |
| envs/sheetsage2 | Python 3.11；官方方案为 torch/torchaudio 2.8.0、cu126、FFmpeg 6.1 共享库 | 短音频解码、编码器加载、生成可解析 ABC |
| envs/yue2 | Python 3.12；采用选定上游版本的依赖 | 加载模型并运行其原始旋律示例 |
| envs/separator | Python 3.11 作为候选，安装前核对所选包版本要求 | 固定 BS-RoFormer 权重，输出双轨并核对增益 |
| envs/seed-vc | Python 3.10；采用相互兼容的 torch/torchaudio/torchvision 版本 | 44.1 kHz F0 分支完成短段转换 |

SheetSage2 的 FFmpeg 要求包括共享库，不应只检查命令行程序存在。需要在该环境中实际验证解码模块加载。系统现有 FFmpeg 8.0 保留，采用环境局部路径解决版本差异。[Cover 环境文档](https://github.com/multimodal-art-projection/YuE/blob/main/docs/covers.md)

Seed-VC 当前 requirements 同时出现 nightly cu126 的 torch 声明和 torch 2.4.0、torchaudio 2.4.0、torchvision 0.19.0 的固定声明。实施时先解析实际依赖；以稳定配套版本为候选，验证 H200 上的短段推理后再冻结环境。不能把一次不明确的依赖解析结果当作可复现安装配方，也不为此修改其他项目环境。[当前 requirements](https://github.com/Plachtaa/seed-vc/blob/main/requirements.txt)

模型准备按以下清单执行。仓库的 branch 名只用于发现，正式实验需记录实际 commit 和模型 revision。

| 模块 | 计划使用的模型与资产 | 需要一起记录 |
| --- | --- | --- |
| 转录 | m-a-p/SheetSage2、其引用的 MERT-v2-FullSong 编码器 | 代码 revision、模型配置、全部实际下载文件 |
| 生成 | m-a-p/YuE2-3B 及 pipeline 引用的解码等资产 | 生成模型与解码资产身份、配置、推理代码 commit |
| 分离 | model_bs_roformer_ep_317_sdr_12.9755.ckpt | 权重、对应配置、audio-separator 版本 |
| 换声主模型 | DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth | config_dit_mel_seed_uvit_whisper_base_f0_44k.yml |
| 换声依赖 | Whisper 编码器、CAMPPlus、RMVPE、配置指定的 BigVGAN | 实际模型仓库 revision 与本地哈希 |

Seed-VC 上述主权重由当前 `inference.py` 的 F0 分支选择，且文件实际存在于官方模型仓库。文件名中的 v2 不应被误认为必须改用普通语音的 `inference_v2.py`。[推理代码](https://github.com/Plachtaa/seed-vc/blob/main/inference.py)、[歌声权重](https://huggingface.co/Plachta/Seed-VC/blob/main/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth)

实施时生成两个锁定清单，而不是只保存一个 requirements 文件。

| 未来文件 | 保存内容 |
| --- | --- |
| configs/environments.lock.json | Python、安装包版本、CUDA runtime、驱动、关键动态库、上游 commit、安装来源 |
| configs/models.lock.json | 模型 ID、revision、配置文件、下载位置、下载后 SHA256、关联上游版本 |

权重下载完成后才计算本地文件哈希，不将网页显示的信息冒充本地校验结果。SheetSage2 涉及远程模型实现，应同时固定模型代码 revision；只固定权重文件仍不足以描述实际推理实现。

优先将明确管理的模型放在 models，将共享缓存放在 cache。部分上游拥有自己的相对缓存路径，例如当前 Seed-VC 在代码中设置 `./checkpoints/hf_cache`；首版记录这些真实位置，不能假设一个环境变量就控制了所有下载。模型资产清单允许引用这些路径。[Seed-VC 缓存设置](https://github.com/Plachtaa/seed-vc/blob/main/inference.py)

安装与模型下载完成后的 preflight 必须同时验证：路径可写、磁盘空间、GPU 可见、模型与配置齐全、导入成功、短音频解码、一次最小推理，以及最终音频可读。模型下载时间和预热时间单独记录，不计入稳定推理吞吐。

现阶段没有安装环境、克隆上游仓库或下载权重。上述环境组合是部署方案；正式锁定清单要在对应环境的最小推理通过后生成。
