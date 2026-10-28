# 目标音色训练流程

按 `00_prepare` → `01_train` → `02_validate` → `03_export` 执行，各文件夹 README 都含具体命令和中间产物。它们是歌曲主流程之外的训练链路。

首次命令示例：

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
bash scripts/05_voice_conversion/finetuned/training/00_prepare/run.sh \
  --config configs/voice_training/sun_yanzi_seed_vc_smoke.yaml \
  --run runs/voice_training/sun_yanzi_seed_vc_smoke_001
```

读取目标歌手原始录音，生成 `data/voice_datasets/sun_yanzi/smoke_v001/manifest.json` 与训练切片供 01 使用。训练完成后通过 `finetuned/run.sh --voice-profile .../voice.json` 接回歌曲主流程05。

[完整手册](../../../../docs/13_voice_adaptation.md)
