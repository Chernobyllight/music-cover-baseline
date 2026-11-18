# 在 Apple Silicon 上运行 v1.2 的 00–06

在 M5、24 GB 统一内存的 Mac 上，使用原生 arm64 Python 3.11 和 PyTorch MPS，从原曲、歌词和参考录音生成 WAV 与 MP3。本次覆盖 SheetSage2、可选移调、YuE2、BS-RoFormer、Seed-VC 零样本换声及混音。模块、阶段编号和任务 YAML 沿用 v1.2。

## 首次安装

在仓库根目录准备 Git、uv 以及带 ffprobe 的 FFmpeg。使用 Homebrew 时可执行 `brew install uv ffmpeg`。安装和模型下载会使用本地磁盘缓存。

```bash
bash scripts/setup_macos.sh --optimized
envs/control/bin/python scripts/download_models.py
envs/control/bin/python scripts/download_voice_models.py
envs/control/bin/python scripts/download_separator_model.py
envs/control/bin/python scripts/prepare_demo.py
envs/control/bin/python scripts/verify_assets.py
```

控制和转谱使用各自的 Torch 2.8 环境，YuE2、分离和换声分别使用锁定的 Torch 2.15 环境。安装器从固定提交组装本地 YuE 和 Seed-VC 源码并应用已验证补丁。源码、环境和模型就绪后，可用下面的预检检查运行依赖及 MPS 计算。已有这些环境时，执行 `envs/control/bin/python scripts/prepare_macos_optimized.py` 即可按当前代码重新组装优化入口。

## 生成成品与恢复

```bash
runner=envs/mps-optimized/scripts/run_baseline.sh
bash "$runner" --demo --device mps --until 06_mix --preflight
bash "$runner" --demo --device mps --memory-budget-gib 16 --until 06_mix --run runs/mac-demo
```

用自己的素材时以 `--song`、`--lyrics`、`--reference` 和 `--style` 替换 `--demo`，或传入 v1.2 的任务 YAML。默认原调，`--transpose 2` 指定升两个半音，`--transpose auto` 执行 v1.2 的音域判定。自动模式会在参考录音或音域证据不足时保留原调。换声使用 30 步、YuE2 声学合成使用 32 步。

`06_mix` 的成功 attempt 中有 `artifacts/final.wav` 与 `artifacts/final.mp3`。可从 `stage_state.json` 的 `stages.06_mix.outputs` 获取本次输出路径。执行到 06 时整体状态为 `partial`，每个已完成模块的状态为 `succeeded`。07–11 属于后续评审与交付流程。

```bash
bash "$runner" --run runs/mac-demo --status
bash "$runner" --resume runs/mac-demo --device mps --until 06_mix
```

每个后端在独立进程执行。阶段指纹包括输入、参数、源码和运行环境。YuE2 在同一阶段的失败 attempt 间还可复用身份与哈希都匹配的语义 token 和 latent。查看 `attempt_*/process.log`、`command.json`、`config.json` 以及模型产物中的 `runtime.json`，可以核对实际解释器、设备、学习层前向精度和内存峰值。

## 环境和历史对照

`--device auto` 优先 CUDA，其次 MPS。显式 `--gpu` 指定 CUDA 设备编号。MPS 默认内存目标为 16 GiB，allocator 的实际限制还受设备推荐工作集影响。Mac 入口关闭 PyTorch 隐式 CPU 回退。

需要复现初版 MPS 环境时，使用 `bash scripts/setup_macos.sh`，通过 `scripts/run_baseline.sh` 运行相同的 00–06 命令。分环境输入和安装锁在 `configs/macos/`，优化版的三份环境锁在 `configs/macos/candidates/`。可用 `envs/control/bin/python scripts/capture_environment.py --optimized --skip-assets` 将优化环境快照写入 `cache/macos-optimized-snapshot/`。

原 main 实现的 16 分 35.66 秒为生成 WAV 的历史成绩；当时的 MP3 另行编码。v1.2 的 06 模块在同一次执行中生成这两种文件。[原实现实测](15-mps-full-song.md) · [v1.2 验收记录](../VALIDATION.md)
