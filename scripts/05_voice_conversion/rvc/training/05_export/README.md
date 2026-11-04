# RVC 05_export

输入：04 已完成的最终模型验收（可以未通过，导出时标记待复核）与 02/03 文件。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/05_voice_conversion/rvc/training/05_export/run.sh \
  --config configs/voice_training/rvc/shan_yi_chun_base_20261103.yaml \
  --run runs/voice_training/shan_yi_chun_rvc_base_20261103
```

输出：models/voices/shan_yi_chun/rvc/base_portable_20261103/voice.json 及配套模型、配置、索引、验收文件；供正式 05 音色转换使用。

过程记录：`runs/voice_training/shan_yi_chun_rvc_base_20261103/stages/05_export/attempt_*/`；派生数据目录：`data/voice_datasets/shan_yi_chun/rvc/base_portable_20261103/`。命令用于新版本的首次运行；已有 02 检查点须显式 `--resume`，已有导出版本不得覆盖。详细前置条件见 [完整指南](../../../../../docs/20_rvc.md)。

本入口和子进程使用 `/opt/conda/bin/python`，自动配置 base PATH。相对参数以项目根目录解析，换工作目录时可使用本脚本的绝对路径启动。训练 `state.json` 的 output 相对训练 run；实际解释器记录在 attempt 的 `command.json`。
