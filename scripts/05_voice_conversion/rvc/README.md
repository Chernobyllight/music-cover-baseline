# RVC 音色转换

RVC v2 / RMVPE / 48 kHz / 训练集检索索引。模型训练与推理独立；本入口不训练。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/05_voice_conversion/rvc/run.sh \
  --run runs/my_lv_zhou_huang_qi_shan_rvc \
  --config configs/voice_inference/rvc/huang_qi_shan.yaml --gpu 0
```

输入：同一 run 04 的人声；先完成 00–04，或使用 `scripts/fork_voice_run.py` 复用现有编曲。模型：`models/voices/huang_qi_shan/rvc/v001_20261029/voice.json`，训练配置最终 4000 步。输出：新 run 的 `stages/05_voice_conversion/attempt_*/artifacts/converted.wav`，交给 06。另有 `candidates.json` 和各候选原生 `converted_native_48k.wav`。完整例子和训练顺序见 [RVC 指南](../../../docs/20_rvc.md)。
