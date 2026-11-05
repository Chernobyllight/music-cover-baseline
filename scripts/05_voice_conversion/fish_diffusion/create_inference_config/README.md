# Fish Diffusion · create_inference_config

由导出模型生成新的推理 YAML。

全部使用 `/opt/conda/bin/python`。脚本自行定位项目根目录，不依赖当前目录或 VC_* 变量。

输入：04_export 的 voice.json，以及显式推理参数。无需重新训练。

输出：configs/voice_inference/fish_diffusion/shan_yi_chun_manual_v001.yaml；常规示例为一个候选。sampler_interval 是扩散采样间隔，不是训练步数或 Seed-VC steps。

## 具体使用示例

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/fish_diffusion/create_inference_config/run.sh \
  --voice-profile models/voices/shan_yi_chun/fish_diffusion/manual_v001/voice.json \
  --sampler-interval 10 --noise-predictor plms --seed 42
```

[完整部署、训练和验证说明](../../../../docs/22_fish_diffusion.md)。
