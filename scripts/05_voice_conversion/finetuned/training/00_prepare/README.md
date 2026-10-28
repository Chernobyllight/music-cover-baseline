# 训练 00_prepare · 数据准备

## 具体示例

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
bash scripts/05_voice_conversion/finetuned/training/00_prepare/run.sh \
  --config configs/voice_training/sun_yanzi_seed_vc_smoke.yaml \
  --run runs/voice_training/sun_yanzi_seed_vc_smoke_001
```

读取：目标歌手原始录音和配置中的 sources；当前示例是 data/raw/references/sun_yanzi/reference.wav。

结果：data/voice_datasets/sun_yanzi/smoke_v001/manifest.json 和 train/*.wav。各阶段记录在 `runs/voice_training/sun_yanzi_seed_vc_smoke_001/stages/00_prepare/attempt_*/`；查看 `state.json` 确认最新成功 attempt。必须按 00→01→02→03 执行，后续阶段不会暗中补跑训练。

本例只有 16 秒素材，属于集成测试，不能作为泛化效果结论。目录已训练过时请用新版本、新 run，或在 01 显式加 `--resume` 并增加累计 `--max-steps`。已导出的版本不允许覆盖。

完整参数和前后串联示例见 [音色微调手册](../../../../../docs/13_voice_adaptation.md)。
