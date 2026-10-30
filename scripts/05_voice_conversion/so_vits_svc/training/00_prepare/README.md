# 00_prepare

读取杨坤源 dataset manifest，保留歌曲组划分；生成 PCM16、ContentVec、F0/UV 和频谱。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/05_voice_conversion/so_vits_svc/training/00_prepare/run.sh \
  --config configs/voice_training/so_vits_svc/yang_kun.yaml \
  --run runs/voice_training/yang_kun_so_vits_svc_v001_20261028
```

本阶段记录：`/data/liuhongda/music-cover-baseline/runs/voice_training/yang_kun_so_vits_svc_v001_20261028/stages/00_prepare/attempt_*/`。具体产物、前后依赖和续训示例见 [完整指南](../../../../../docs/18_so_vits_svc.md)。
