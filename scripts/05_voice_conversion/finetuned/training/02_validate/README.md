# 训练 02_validate · 通用与微调模型对照

运行前须按 [音色手册第 4 节](../../../../../docs/13_voice_adaptation.md) 新建配置中的主流程 source_run 并完成 00–04；`runs/lhd_test_modular` 旧数据已清理。

## 具体示例

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
bash scripts/05_voice_conversion/finetuned/training/02_validate/run.sh \
  --config configs/voice_training/sun_yanzi_seed_vc_smoke.yaml \
  --run runs/voice_training/sun_yanzi_seed_vc_smoke_001
```

读取：01 的 checkpoint、00 的数据划分和参考片段，以及 runs/lhd_test_modular 中主流程04的 vocals.wav / instrumental.wav。

结果：本阶段 artifacts/validation.json、metrics.csv、listening.html、listening/、review_scores.csv。各阶段记录在 `runs/voice_training/sun_yanzi_seed_vc_smoke_001/stages/02_validate/attempt_*/`；查看 `state.json` 确认最新成功 attempt。必须按 00→01→02→03 执行，后续阶段不会暗中补跑训练。

本例只有 16 秒素材，属于集成测试，不能作为泛化效果结论。目录已训练过时请用新版本、新 run，或在 01 显式加 `--resume` 并增加累计 `--max-steps`。已导出的版本不允许覆盖。

完整参数和前后串联示例见 [音色微调手册](../../../../../docs/13_voice_adaptation.md)。
