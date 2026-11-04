# 00_prepare

读取新训练 YAML 指定的源 dataset manifest，保留歌曲组划分；生成 PCM16、ContentVec、F0/UV 和频谱。先用 [create_config](../create_config/README.md) 生成自己的配置。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/05_voice_conversion/so_vits_svc/training/00_prepare/run.sh \
  --config configs/voice_training/so_vits_svc/shan_yi_chun_manual_20261104.yaml \
  --run runs/voice_training/shan_yi_chun_svc_manual_20261104
```

本阶段记录：`/data/liuhongda/music-cover-baseline/runs/voice_training/shan_yi_chun_svc_manual_20261104/stages/00_prepare/attempt_*/`。输出专用数据目录由 YAML 的 `dataset_dir` 指定，供 01_train 使用。

新配置默认显式设置 `audio.peak_policy: attenuate_if_needed`、`peak_limit: 0.99`，对 FLOAT 峰值 >= 1 的片段先降低整体增益再写 PCM16，保留原音频并在 manifest 每个 clip 的 `audio_preparation` 中记录原峰值和增益。无 audio 字段的旧配置仍对超限音频报错。具体前后依赖见 [完整手动流程](../../../../../docs/svc_config_scripts.md)。
