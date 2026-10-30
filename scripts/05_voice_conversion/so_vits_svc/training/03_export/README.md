# 03_export

读取 02 的合格检查点选择记录，导出不可覆盖的 SVC 歌手版本。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/05_voice_conversion/so_vits_svc/training/03_export/run.sh \
  --config configs/voice_training/so_vits_svc/yang_kun.yaml \
  --run runs/voice_training/yang_kun_so_vits_svc_v001_20261028
```

本阶段记录：`/data/liuhongda/music-cover-baseline/runs/voice_training/yang_kun_so_vits_svc_v001_20261028/stages/03_export/attempt_*/`。具体产物、前后依赖和续训示例见 [完整指南](../../../../../docs/18_so_vits_svc.md)。
