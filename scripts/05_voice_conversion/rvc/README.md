# RVC 音色转换

RVC v2 / RMVPE / 48 kHz / 训练集检索索引。模型训练与推理独立；本入口不训练。

直接使用现有 `/opt/conda/bin/python`，包含主控制器和推理子进程。相对路径按项目根目录解析，可从任意工作目录通过脚本绝对路径启动。当前单依纯训练入口和中间产物见[训练步骤](training/00_prepare/README.md)与[RVC 指南](../../../docs/20_rvc.md)。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/05_voice_conversion/rvc/run.sh \
  --run runs/lhd_lv_zhou_shan_yi_chun_rvc_base_20261103 \
  --config configs/voice_inference/rvc/shan_yi_chun_base_20261103.yaml --gpu 0
```

输入：同一 run 04 的人声；先完成 00–04，或通过 `scripts/05_voice_conversion/fork_run.sh --source-run runs/lhd_lv_zhou_shan_yi_chun_modular_test --run runs/lhd_lv_zhou_shan_yi_chun_rvc_base_20261103` 建立独立运行。模型：`models/voices/shan_yi_chun/rvc/base_portable_20261103/voice.json`，须先完成本次训练、验证和导出。本例使用最终 4000 步模型，1 个候选，index_rate=0.5。输出：新 run 的 `stages/05_voice_conversion/attempt_*/artifacts/converted.wav`，交给 06。另有 `candidates.json` 和候选原生 `converted_native_48k.wav`。实际解释器记录在候选 `conversion.json` 的 runtime 中。
