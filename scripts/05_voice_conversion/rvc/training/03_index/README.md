# RVC 03_index

输入：01 的训练集内容特征，不读验证和测试特征。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/05_voice_conversion/rvc/training/03_index/run.sh \
  --config configs/voice_training/rvc/huang_qi_shan.yaml \
  --run runs/voice_training/huang_qi_shan_rvc_v001_20261029
```

输出：训练 run 的 added.index 与 index_manifest.json；供 04 和 05 使用。

过程记录：`runs/voice_training/huang_qi_shan_rvc_v001_20261029/stages/03_index/attempt_*/`；派生数据目录：`data/voice_datasets/huang_qi_shan/rvc/v001_20261029/`。命令用于新版本的首次运行；已有 02 检查点须显式 `--resume`，已有导出版本不得覆盖。详细前置条件见 [完整指南](../../../../../docs/20_rvc.md)。
