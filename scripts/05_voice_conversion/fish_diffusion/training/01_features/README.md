# Fish Diffusion · training/01_features

提取 ContentVec 256 维第 9 层投影、Parselmouth F0 和 128 维 Mel。

全部使用 `/opt/conda/bin/python`。脚本自行定位项目根目录，不依赖当前目录或 VC_* 变量。

输入：00_prepare 的数据清单、ContentVec 和 NSF-HiFiGAN。仅 train 做音高增强，validation/test 不增强。

输出：独立数据目录 features/*.npz、*.json；本阶段 artifacts/features.json 记录哈希、帧数、划分和增强。有效缓存可续用，参数变化要求新的数据目录。

## 具体使用示例

```bash
bash scripts/05_voice_conversion/fish_diffusion/training/01_features/run.sh \
  --config configs/voice_training/fish_diffusion/shan_yi_chun_manual_v001.yaml \
  --run runs/voice_training/shan_yi_chun_fish_manual_v001
```

查看本阶段中间结果：`/data/liuhongda/music-cover-baseline/runs/voice_training/shan_yi_chun_fish_manual_v001/stages/01_features/attempt_*/artifacts/`。以训练 run 的 `state.json` 为准。上游阶段必须显式执行成功，不会暗中补跑。

[完整部署、训练和验证说明](../../../../../docs/22_fish_diffusion.md)。

只读已成功 00 的数据，训练集增强，验证和测试不增强。

[从原始录音开始的手动实操](../../../../../docs/24_fish_diffusion_manual.md)。
