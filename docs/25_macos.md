# 在 Apple Silicon 上运行 v1.2 的 00–06

在 M5、24 GB 统一内存的 Mac 上，使用原生 arm64 Python 3.11 和 PyTorch MPS，从原曲、歌词和参考录音生成 WAV 与 MP3。本次覆盖 SheetSage2、可选移调、YuE2、BS-RoFormer、Seed-VC 零样本换声及混音。模块、阶段编号和任务 YAML 沿用 v1.2。

## 首次安装

在仓库根目录准备 Git、uv 以及带 ffprobe 的 FFmpeg。使用 Homebrew 时可执行 `brew install uv ffmpeg`。安装和模型下载会使用本地磁盘缓存。

```bash
bash scripts/setup/macos.sh
envs/control/bin/python scripts/download_models.py
envs/control/bin/python scripts/download_voice_models.py
envs/control/bin/python scripts/download_separator_model.py
envs/control/bin/python scripts/prepare_demo.py
envs/control/bin/python scripts/setup/verify_macos_assets.py
```

控制和转谱使用各自的 Torch 2.8 环境，YuE2、分离和换声分别使用锁定的 Torch 2.15 环境。安装器从固定提交组装本地 YuE 和 Seed-VC 源码并应用已验证补丁。源码、环境和模型就绪后，可用下面的预检检查运行依赖及 MPS 计算。更新仓库后，重新运行安装命令以同步依赖并组装当前版本。依赖已就绪时，也可以用 `envs/control/bin/python scripts/setup/prepare_macos_runtime.py` 重新组装源码。

## 生成成品与恢复

```bash
runner=scripts/run_baseline.sh
bash "$runner" --demo --device mps --until 06_mix --preflight
bash "$runner" --demo --device mps --memory-budget-gib 16 --until 06_mix --run runs/mac-demo
```

用自己的素材时以 `--song`、`--lyrics`、`--reference` 和 `--style` 替换 `--demo`，也可以用 `--config` 传入任务 YAML。仓库内的素材和任务可用 `data/`、`configs/` 开头的相对路径，其他本地文件用绝对路径。Mac 任务 YAML 的 `modules.05_voice_conversion` 需省略或使用以下配置。

```yaml
modules:
  05_voice_conversion:
    backend: seed_vc
    mode: zero_shot
```

默认原调，`--transpose 2` 指定升两个半音，`--transpose auto` 执行 v1.2 的音域判定。自动模式会在参考录音或音域证据不足时保留原调。换声使用 30 步、YuE2 声学合成使用 32 步。

`06_mix` 的成功 attempt 中有 `artifacts/final.wav` 与 `artifacts/final.mp3`。可从 `stage_state.json` 的 `stages.06_mix.outputs` 获取本次输出路径。执行到 06 时整体状态为 `partial`，每个已完成模块的状态为 `succeeded`。07–11 属于后续评审与交付流程。

```bash
bash "$runner" --run runs/mac-demo --status
bash "$runner" --resume runs/mac-demo --device mps --until 06_mix
```

每个后端在独立进程执行。阶段指纹包括输入、参数、源码和运行环境。YuE2 在同一阶段的失败 attempt 间还可复用身份与哈希都匹配的语义 token 和 latent。查看 `attempt_*/process.log`、`command.json`、`config.json` 以及模型产物中的 `runtime.json`，可以核对实际解释器、设备、内存峰值，以及学习层的前向设备、参数 dtype 和调用次数。局部 autocast 精度由下文对应的补丁设置说明。

## Mac 优化入口的推理配置

YuE2 的 BF16 非因果声学 attention 会合并最多 20 个完整的 256-query 块，末尾不足 256 个 query 的部分仍独立计算。显式分块设置和因果预填充沿用原有路径。

优化入口在 BS-RoFormer 的时间与频率 Transformer 中使用 MPS FP16 autocast，输出恢复 FP32。Seed-VC 的 BigVGAN 对 Conv1d 和 ConvTranspose1d 使用相同的局部精度，滤波和激活沿用原精度。权重、分离块长、YuE2 的 32 步和 Seed-VC 的 30 步沿用原配置。

同一台 M5 上加速前与最新版本的整曲对照见 [验收记录](../VALIDATION.md)。

## 设备与环境

`--device auto` 优先 CUDA，其次 MPS。显式 `--gpu` 指定 CUDA 设备编号。MPS 默认内存目标为 16 GiB，allocator 的实际限制还受设备推荐工作集影响。Mac 入口关闭 PyTorch 隐式 CPU 回退。

五套环境的安装锁均位于 `configs/macos/`。可用 `envs/control/bin/python scripts/capture_environment.py --skip-assets` 将实际运行的环境快照写入本地 `cache/macos-optimized-snapshot/`。
