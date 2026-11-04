# 生成新的 SVC 推理 YAML

输入是用户运行 SVC `03_export` 后得到的真实 `voice.json`。

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/so_vits_svc/create_inference_config/run.sh \
  --voice-profile models/voices/shan_yi_chun/so_vits_svc/manual_20261104/voice.json \
  --seed 42 --f0-predictor rmvpe --noise-scale 0.4 \
  --chunk-seconds 20 --overlap-seconds 0.5 --pad-seconds 0.5
```

输出 `/data/liuhongda/music-cover-baseline/configs/voice_inference/so_vits_svc/shan_yi_chun_manual_20261104.yaml`，供整曲推理的 `--config` 使用。检查模型文件哈希，固定一个候选，不训练、不推理。

支持 `--output`、只读 `--plan`，拒绝覆盖已有配置。未显式传入的值取 profile 的 `recommended_settings`，再回退到 SVC 默认值。上例与标准训练验证入口参数一致。完整实例见 [SVC 手动验证](../../../../docs/svc_config_scripts.md)。
