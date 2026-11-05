# Fish Diffusion · preflight

检查现有 base 环境、权重完整性和真实模型加载。

全部使用 `/opt/conda/bin/python`。脚本自行定位项目根目录，不依赖当前目录或 VC_* 变量。

输入：公共权重、环境和上游源码。--model-load 会在选定 GPU 加载模型，但不训练。

输出：终端报告；可显式传 --output runs/fish_diffusion_setup_20261105/preflight.json 保存。

## 具体使用示例

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/fish_diffusion/preflight/run.sh --scope training --model-load --gpu 0
```

[完整部署、训练和验证说明](../../../../docs/22_fish_diffusion.md)。
