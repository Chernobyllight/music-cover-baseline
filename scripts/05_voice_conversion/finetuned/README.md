# 05 · finetuned

本例的 `runs/lhd_test_modular` 是可新建的运行名，旧数据已清理。先按 [三候选指南](../../../docs/15_voice_candidates.md) 完成 00–04（使用同一个 --run），或运行下面的准备命令；不能直接使用已删除的上游。

加载已经训练并导出的目标音色模型；该脚本只推理。

## 具体示例

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
bash scripts/run_baseline.sh \
  --config configs/tasks/qiu_yi_nong_bossa_nova_sun_yanzi.yaml \
  --run runs/lhd_test_modular --until 04_vocal_separation

bash scripts/05_voice_conversion/finetuned/run.sh \
  --run runs/lhd_test_modular \
  --voice-profile models/voices/sun_yanzi/seed_vc/smoke_v001/voice.json
```

本例通过 `/data` 入口运行；它指向 `/mnt/afs_f/liuhongda/music-cover-baseline`。

读取同一运行目录中 04 的 `vocals.wav` 和 00 的 `reference.wav`。这两个阶段及其上游必须成功且缓存有效。实际 attempt 自动从 `stage_state.json` 获取。

输出 `runs/lhd_test_modular/stages/05_voice_conversion/attempt_*/artifacts/converted.wav` 和 `conversion.json`，供 06 混音使用。参数变化或 --force 重跑创建新 attempt，未变化时复用有效缓存；不覆盖通用模型。05 保持音高不变。

详细训练、对照验证、权重要求和后续命令见 [音色微调手册](../../../docs/13_voice_adaptation.md)。

## 微调推理的三个候选

微调入口默认生成三个完整候选（seed=42 时为 42、43、44），使用同一目标音色模型、源人声、参考片段和推理参数。05 保存干声与逐项指标，06 分别混音，11 交付三个 WAV/MP3。仅技术检查合格的候选参与 ECAPA 排名，推荐版本进入分轨、MIDI 和最终交付入口；零样本入口仍为单次推理。

具体命令与中间文件路径见[三个候选与推荐结果使用说明](../../../docs/15_voice_candidates.md)。
