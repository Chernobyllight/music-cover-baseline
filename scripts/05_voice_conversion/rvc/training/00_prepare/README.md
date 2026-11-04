# RVC 00_prepare

输入：原 dataset/manifest.json，保留 train/validation/test 歌曲组划分，排除 prompt。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/05_voice_conversion/rvc/training/00_prepare/run.sh \
  --config configs/voice_training/rvc/shan_yi_chun_base_20261103.yaml \
  --run runs/voice_training/shan_yi_chun_rvc_base_20261103
```

输出：派生数据目录 manifest.json 与 48 kHz FLOAT WAV；训练 run 的 model_config.json；供 01 使用。

过程记录：`runs/voice_training/shan_yi_chun_rvc_base_20261103/stages/00_prepare/attempt_*/`；派生数据目录：`data/voice_datasets/shan_yi_chun/rvc/base_portable_20261103/`。命令用于新版本的首次运行；已有 02 检查点须显式 `--resume`，已有导出版本不得覆盖。详细前置条件见 [完整指南](../../../../../docs/20_rvc.md)。

本入口和子进程使用 `/opt/conda/bin/python`，自动配置 base PATH。相对参数以项目根目录解析，换工作目录时可使用本脚本的绝对路径启动。训练 `state.json` 的 output 相对训练 run；实际解释器记录在 attempt 的 `command.json`。

本例 source_dataset 为 `data/voice_training/shan_yi_chun/manual_20261102/dataset`，须先按 Seed-VC 数据准备流程生成，保留原歌曲组划分。
