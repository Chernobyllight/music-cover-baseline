# 05 · zero_shot

本例的 `runs/lhd_test_modular` 是可新建的运行名，旧数据已清理。先按 [三候选指南](../../../docs/15_voice_candidates.md) 完成 00–04（使用同一个 --run），或运行下面的准备命令；不能直接使用已删除的上游。

使用通用 Seed-VC，保留原有 training-free 转换接口。

## 具体示例

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
bash scripts/run_baseline.sh \
  --config configs/tasks/qiu_yi_nong_bossa_nova_sun_yanzi.yaml \
  --run runs/lhd_test_modular --until 04_vocal_separation

bash scripts/05_voice_conversion/zero_shot/run.sh \
  --run runs/lhd_test_modular
```

本例通过 `/data` 入口运行；它指向 `/mnt/afs_f/liuhongda/music-cover-baseline`。

读取同一运行目录中 04 的 `vocals.wav` 和 00 的 `reference.wav`。这两个阶段及其上游必须成功且缓存有效。实际 attempt 自动从 `stage_state.json` 获取。

输出 `runs/lhd_test_modular/stages/05_voice_conversion/attempt_*/artifacts/converted.wav` 和 `conversion.json`，供 06 混音使用。参数变化或 --force 重跑创建新 attempt，未变化时复用有效缓存；不覆盖通用模型。05 保持音高不变。

详细训练、对照验证、权重要求和后续命令见 [音色微调手册](../../../docs/13_voice_adaptation.md)。
