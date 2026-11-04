# 生成新的 SVC 训练 YAML

输入是用户刚制作的数据集（含 manifest）和完成 00/04 的歌曲 run。自动读取当前 00 参考片段并校验其与数据集一致。

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/so_vits_svc/training/create_config/run.sh \
  --voice-id shan_yi_chun --version manual_20261104 \
  --source-dataset data/voice_training/shan_yi_chun/svc_manual_20261104/dataset \
  --source-run runs/lhd_lv_zhou_shan_yi_chun_svc_manual_20261104 \
  --validation-windows 36:12 156:12 \
  --max-steps 4000 --save-every 1000 --batch-size 6 \
  --learning-rate 0.0001 --seed 42 --gpu 0 \
  --peak-policy attenuate_if_needed --peak-limit 0.99
```

输出 `/data/liuhongda/music-cover-baseline/configs/voice_training/so_vits_svc/shan_yi_chun_manual_20261104.yaml`，供后续 SVC `00_prepare` 使用。`--output`、`--dataset-dir`、`--export-dir` 可修改输出路径。只写新 YAML，不训练或导出模型。

追加 `--plan` 只校验和预览，拒绝覆盖已有配置、目标数据集或模型目录。完整步骤见 [SVC 手动验证](../../../../../docs/svc_config_scripts.md)。

新配置的峰值策略只在后续 SVC 00 生成 PCM16 副本时生效：绝对峰值 >= 1 才降低到 0.99，逐段记录增益，保留原始 FLOAT 数据。设 `--peak-policy error` 可严格拒绝超限片段。生成配置时不会处理或改写音频。
