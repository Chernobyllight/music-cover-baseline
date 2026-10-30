# SVC 歌手模型推理

使用已有 base 环境；模型必须先完成独立训练和导出。完整步骤见 [SVC 指南](../../../docs/18_so_vits_svc.md)。

输入：本 run 的 04 阶段 vocals，以及指定 SVC voice.json；00 的参考音频仅用于指标和音区分析。

```bash
cd /data/liuhongda/music-cover-baseline
# 新输出目录需尚不存在；从保留的验证结果复用00–04。
/opt/conda/bin/python scripts/fork_voice_run.py \
  --source-run runs/lhd_yang_kun_svc_optimized_validation_20261028 \
  --run runs/my_yang_kun_svc_optimized

bash scripts/05_voice_conversion/so_vits_svc/run.sh \
  --run runs/my_yang_kun_svc_optimized \
  --config configs/voice_inference/so_vits_svc/yang_kun_optimized.yaml \
  --gpu 0
```

上面的命令会从已保留的验证 run 创建新的输出目录。输出在该 run 的 `stages/05_voice_conversion/attempt_*/artifacts/`，含 converted.wav、conversion.json、candidates.json，供 06_mix 使用。

## 推荐杨坤优化配置

另建对照run后执行 `bash scripts/05_voice_conversion/so_vits_svc/run.sh --run runs/my_yang_kun_svc_optimized --config configs/voice_inference/so_vits_svc/yang_kun_optimized.yaml --gpu 0`。完整创建run命令、训练数据和对照结果见 [优化指南](../../../docs/19_svc_optimization.md)；研究工具见 [experiments](experiments/README.md)。原版profile和默认参数不变。
