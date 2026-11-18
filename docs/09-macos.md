# Apple Silicon / MPS

M5、24 GB 统一内存，原生 arm64 Python 3.11。优化版端到端生成 WAV 用时 **16 分 36 秒**，用户已试听验收。模型前向使用 MPS。已验收音频与详细计时见 [整曲报告](15-mps-full-song.md)。

## 推荐配置

在仓库根目录执行。首次安装后按下方资产恢复命令准备权重和演示素材。

```bash
bash scripts/setup_macos.sh --optimized
bash envs/mps-optimized/scripts/run_baseline.sh --device mps --preflight
bash envs/mps-optimized/scripts/run_baseline.sh --demo --device mps --memory-budget-gib 16
```

已有五个运行环境及权重时，可用 `envs/control/bin/python scripts/prepare_macos_optimized.py` 直接组装入口。YuE2、分离和 Seed-VC 使用各自锁定的 Torch 2.15 环境；转谱和控制仍使用原环境。安装脚本通过固定 revision 的补丁清单组装隔离源码工作区。工作区链接到项目共用的环境、配置、权重和 `runs/`，成品直接写入项目的 `runs/<运行编号>/deliverables/`。源码或补丁更新后，重新组装会选用新副本。

自己的歌曲在优化入口沿用 `--song`、`--lyrics`、`--reference`、`--style` 等参数。恢复优化版运行时重复使用优化入口和 `--resume`。此前的整曲实测目录 `runs/m5-full-candidate` 是运行记录，供回看与核对。

## 基线环境与实测兼容性

| 模型 | MPS 有效精度 | 验证范围 |
| --- | --- | --- |
| SheetSage2 / MERT | FP32 | 20 秒和整曲转谱 |
| YuE2 AR / NAR | BF16，torch-eager | 短谱面和整曲，32 步 NAR，未截断 |
| YuE2 VAE | FP32 | 整曲分块解码 |
| BS-RoFormer | FP32 | 短样本和整曲分离 |
| Seed-VC | Whisper FP16，其余实测学习层 FP32 | 短样本和整曲换声 |
| RMVPE | FP32 | 短样本音高与可选移调链路 |

实际环境为 Python 3.11.17；YuE2 使用 Torch 2.10.0 / Transformers 4.57.6，转谱使用 Torch 2.8.0 / Transformers 4.45.2，换声使用 Torch 2.8.0 / Transformers 4.46.3，分离使用 Torch 2.8.0 / audio-separator 0.30.2。完整安装清单和哈希锁文件保存在 `configs/macos/`。

## 基线安装与两套配置共用的资产恢复

先安装 Git 和 uv。需要原 48.8 分钟基线环境时，执行下方第一条。已安装优化版时，可直接执行后续的资产恢复命令。[uv 安装说明](https://docs.astral.sh/uv/)

```bash
bash scripts/setup_macos.sh
export PATH="$PWD/envs/control/bin:$PATH"
export HF_HUB_DISABLE_XET=1
envs/control/bin/python scripts/download_models.py
envs/control/bin/python scripts/download_voice_models.py
envs/control/bin/python scripts/download_separator_model.py
envs/control/bin/python scripts/prepare_demo.py
envs/control/bin/python scripts/verify_assets.py
```

五个环境分别由 `configs/macos/*.txt` 锁定依赖与文件哈希；`.in` 是直接依赖来源。安装脚本可以重复执行，FFmpeg 使用 control 环境内的独立二进制，不修改系统安装。网络中断后重跑下载命令可继续恢复；Xet 可按网络情况启用，但不应关闭 HTTPS 校验。

`patches/seed-vc-mps.patch` 仅修复 MPS 不支持的 float64 音高输入，CUDA 路径保持原有 dtype。`scripts/apply_compat_patches.py` 校验固定 commit 后应用补丁，已应用时不重复修改。上游源码的实际哈希计入缓存身份；不要依赖手工修改模型缓存来维持兼容。

`patches/yue2-mps-memory.patch` 在模型阶段切换及闲置 AR 权重暂存后释放空闲 MPS 缓存。Mac 使用上游已有的 `offload_ar`：仅暂存声学合成阶段不参与计算的权重，核心前向仍在 MPS。YuE2 的语义 token 与 latent 分别保存检查点；后续阶段失败后，同配置、同源码且哈希完整的检查点可在下一次尝试复用。

`patches/yue2-mps-sampling.patch` 消除 CPU 抽样后多余的 GPU 标量读回，不改变采样规则或精度。性能排查复用已有整曲，未重跑；实测范围、瓶颈与尚未解决的 attention 数值线索见 [性能报告](10-mps-performance.md)。

## 基线运行与音频导出

```bash
bash scripts/run_baseline.sh --demo --device mps --memory-budget-gib 16
bash scripts/run_baseline.sh --resume runs/具体运行目录 --device mps
```

主入口交付 `final.wav`。需要 MP3 时，使用已安装的 FFmpeg 导出并验证：

```bash
envs/control/bin/ffmpeg -i runs/具体运行目录/deliverables/final.wav -codec:a libmp3lame -b:a 320k runs/具体运行目录/deliverables/final.mp3
envs/control/bin/ffmpeg -v error -i runs/具体运行目录/deliverables/final.mp3 -f null -
```

自己的歌曲沿用 `USAGE.md` 的 `--song`、`--lyrics`、`--reference` 和 `--style`。可选 `--transpose auto` 或指定 `-12..12` 个半音。首次运行下载与加载耗时不计为稳定推理速度。

`--device auto` 优先 CUDA，其次 MPS；都不可用时失败。`--gpu` 仅用于 CUDA，不能与显式 `--device mps` 同时指定。旧 CUDA 运行可通过 `--resume ... --device mps` 迁移，旧设备的模型阶段结果不会直接复用。

MPS 默认目标预算 16 GiB，CUDA 默认保持 80 GiB。MPS 实际限制取请求值与设备推荐工作集的较小值；两者均记录。这个限制约束 MPS allocator，不是整个进程 RSS 或系统统一内存的硬上限。每阶段独立进程，记录进程峰值 RSS、采样得到的 MPS tensor/driver 峰值与实际 learned-layer 前向调用。

运行器将 `PYTORCH_ENABLE_MPS_FALLBACK` 设为 `0`。直接调用阶段脚本时也必须关闭它。MPS 阶段保留 24 小时故障超时，CUDA 保持原有 3 小时；这不是性能承诺。错误、显存不足和生成截断均不会被当作成功。

## 验证与定位

```bash
PYTHONPATH=src envs/control/bin/python -m unittest discover -s tests -v
PYTORCH_ENABLE_MPS_FALLBACK=0 envs/yue2/bin/python scripts/diagnostics/probe_yue_mps.py --output runs/probes/yue-new
envs/control/bin/python scripts/capture_environment.py
```

YuE2 探针使用固定短谱面，分别保存语义 token、NAR latent 和解码音频；探针预算不修改生产链路的整曲长度策略。`runtime.json` 提供真实设备和内存证据；主流程另保留 `process.log`、命令、状态、输入输出哈希、检查报告和试听页。失败后检查对应阶段的 `failure.json` 与日志，再使用 `--resume`。

环境与资产快照写入 `configs/macos/`，原始 NVIDIA 快照保留。跨设备随机生成不要求字节相同；整曲完整性、有限数值、时长、对齐、输出格式与试听需要分别验收。尚未实测的 CUDA 回归、Intel Mac 和其他内存档位不能宣称已支持。

## 性能记录

按实验顺序可查阅[瓶颈排查](10-mps-performance.md)、[低侵入候选](11-mps-low-intrusion-optimization.md)、[分离对照](12-mps-separator-block.md)、[换声与语义投影](13-mps-seed-yue.md)、[完整前向拆分](14-mps-ten-minute-target.md)及[优化版整曲验收](15-mps-full-song.md)。探针脚本集中在 `scripts/diagnostics/`，计算账本和原始数据保存在 `runs/probes/`。
