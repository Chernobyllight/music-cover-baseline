# RVC 05_export

输入：04 已通过的最终模型验收与 02/03 文件。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/05_voice_conversion/rvc/training/05_export/run.sh \
  --config configs/voice_training/rvc/huang_qi_shan.yaml \
  --run runs/voice_training/huang_qi_shan_rvc_v001_20261029
```

输出：models/voices/huang_qi_shan/rvc/v001_20261029/voice.json 及配套模型、配置、索引、验收文件；供正式 05 音色转换使用。

过程记录：`runs/voice_training/huang_qi_shan_rvc_v001_20261029/stages/05_export/attempt_*/`；派生数据目录：`data/voice_datasets/huang_qi_shan/rvc/v001_20261029/`。命令用于新版本的首次运行；已有 02 检查点须显式 `--resume`，已有导出版本不得覆盖。详细前置条件见 [完整指南](../../../../../docs/20_rvc.md)。
