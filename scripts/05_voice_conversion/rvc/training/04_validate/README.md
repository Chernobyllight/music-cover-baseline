# RVC 04_validate

输入：02 的配置最终 4000 步权重与 03 的索引，使用 validation 参考。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/05_voice_conversion/rvc/training/04_validate/run.sh \
  --config configs/voice_training/rvc/huang_qi_shan.yaml \
  --run runs/voice_training/huang_qi_shan_rvc_v001_20261029
```

输出：训练 run 的 validation.json 及当前 attempt 下的转换音频和指标；供 05 导出验收，不比较多个模型。

过程记录：`runs/voice_training/huang_qi_shan_rvc_v001_20261029/stages/04_validate/attempt_*/`；派生数据目录：`data/voice_datasets/huang_qi_shan/rvc/v001_20261029/`。命令用于新版本的首次运行；已有 02 检查点须显式 `--resume`，已有导出版本不得覆盖。详细前置条件见 [完整指南](../../../../../docs/20_rvc.md)。
