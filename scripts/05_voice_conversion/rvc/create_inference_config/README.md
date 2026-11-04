# RVC 推理配置生成

第 14 步；固定 base `/opt/conda/bin/python`，相对路径按项目根目录解析。配置生成器可加 `--plan` 只检查预览，已有输出不覆盖。

前置：你已依次完成 RVC 训练 00–05，以下 `voice.json` 是第 13 步导出的产物。

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/rvc/create_inference_config/run.sh \
  --voice-profile models/voices/shan_yi_chun/rvc/manual_20261103/voice.json \
  --candidate-count 1 --seed 42
```

输出 `/data/liuhongda/music-cover-baseline/configs/voice_inference/rvc/shan_yi_chun_manual_20261103.yaml`。先校验模型、索引、配置、验证报告的哈希和后端，再按 `RVC 默认值 < voice.json 推荐参数 < 显式命令行参数` 的优先级生成完整推理配置。训练配置与推理配置均显式写入参数，推理引擎本身行为不变。

支持 `--index-rate 0.75` 等覆盖；多个候选用 `--candidate-count 3`，默认种子从 `--seed` 连续生成，也可用 `--candidate-seeds 42 43 44` 显式指定。配置中不会产生跨后端遗留参数。

第 15 步由你执行：

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/rvc/run.sh \
  --run runs/lhd_lv_zhou_shan_yi_chun_rvc_manual_20261103 \
  --config configs/voice_inference/rvc/shan_yi_chun_manual_20261103.yaml --gpu 0
```

此时才加载你训练的模型生成音频。配置生成器没有代替执行前面的任何实验节点。


完整四步见[配置脚本说明](../../../../docs/rvc_config_scripts.md)。
