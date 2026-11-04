# RVC 03_index

输入：01 的训练集内容特征，不读验证和测试特征。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/05_voice_conversion/rvc/training/03_index/run.sh \
  --config configs/voice_training/rvc/shan_yi_chun_base_20261103.yaml \
  --run runs/voice_training/shan_yi_chun_rvc_base_20261103
```

输出：训练 run 的 added.index 与 index_manifest.json；供 04 和 05 使用。

过程记录：`runs/voice_training/shan_yi_chun_rvc_base_20261103/stages/03_index/attempt_*/`；派生数据目录：`data/voice_datasets/shan_yi_chun/rvc/base_portable_20261103/`。命令用于新版本的首次运行；已有 02 检查点须显式 `--resume`，已有导出版本不得覆盖。详细前置条件见 [完整指南](../../../../../docs/20_rvc.md)。

本入口和子进程使用 `/opt/conda/bin/python`，自动配置 base PATH。相对参数以项目根目录解析，换工作目录时可使用本脚本的绝对路径启动。训练 `state.json` 的 output 相对训练 run；实际解释器记录在 attempt 的 `command.json`。
