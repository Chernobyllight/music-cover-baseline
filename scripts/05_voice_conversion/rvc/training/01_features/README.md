# RVC 01_features

输入：00 准备的 train 音频。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/05_voice_conversion/rvc/training/01_features/run.sh \
  --config configs/voice_training/rvc/shan_yi_chun_base_20261103.yaml \
  --run runs/voice_training/shan_yi_chun_rvc_base_20261103
```

输出：派生数据目录 features/*.npy、audio16/*.wav、train.txt、features.json；供 02 和 03 使用。

过程记录：`runs/voice_training/shan_yi_chun_rvc_base_20261103/stages/01_features/attempt_*/`；派生数据目录：`data/voice_datasets/shan_yi_chun/rvc/base_portable_20261103/`。命令用于新版本的首次运行；已有 02 检查点须显式 `--resume`，已有导出版本不得覆盖。详细前置条件见 [完整指南](../../../../../docs/20_rvc.md)。

本入口和子进程使用 `/opt/conda/bin/python`，自动配置 base PATH。相对参数以项目根目录解析，换工作目录时可使用本脚本的绝对路径启动。训练 `state.json` 的 output 相对训练 run；实际解释器记录在 attempt 的 `command.json`。

`train.txt` 的四个文件路径相对派生数据集目录；`features.json` 标记 `filelist_path_base: dataset`。02 会在自己的 attempt 中生成供上游读取的 `runtime_train.txt`，不会修改已校验的规范清单。
