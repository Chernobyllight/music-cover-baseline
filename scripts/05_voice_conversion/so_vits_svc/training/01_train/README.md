# 01_train

读取 00 的训练清单、特征和模型配置，更新 G/D 并保存可恢复检查点。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/05_voice_conversion/so_vits_svc/training/01_train/run.sh \
  --config configs/voice_training/so_vits_svc/yang_kun.yaml \
  --run runs/voice_training/yang_kun_so_vits_svc_v001_20261028
```

本阶段记录：`/data/liuhongda/music-cover-baseline/runs/voice_training/yang_kun_so_vits_svc_v001_20261028/stages/01_train/attempt_*/`。具体产物、前后依赖和续训示例见 [完整指南](../../../../../docs/18_so_vits_svc.md)。
