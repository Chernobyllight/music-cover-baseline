# Fish Diffusion · 05 推理

只执行主流程 05 音色转换。

全部使用 `/opt/conda/bin/python`。脚本自行定位项目根目录，不依赖当前目录或 VC_* 变量。

输入：当前歌曲 run 的 00/04 成功产物、Fish 推理 YAML 和 voice.json。00 参考音频仅用于评估，歌手音色来自训练权重；05 额外移调固定为 0。

输出：歌曲 run/stages/05_voice_conversion/attempt_*/artifacts/converted.wav、conversion.json、candidates.json/csv/html。供已有 06_mix 使用。

## 具体使用示例

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/fish_diffusion/run.sh \
  --run runs/lhd_lv_zhou_shan_yi_chun_fish_manual_v001 \
  --config configs/voice_inference/fish_diffusion/shan_yi_chun_manual_v001.yaml --gpu 0
```

首次需先运行 00–04，或用已有 `fork_run.sh` 创建独立运行。完成后混音：

```bash
bash /data/liuhongda/music-cover-baseline/scripts/06_mix/run.sh \
  --run runs/lhd_lv_zhou_shan_yi_chun_fish_manual_v001 \
  --config configs/voice_inference/fish_diffusion/shan_yi_chun_manual_v001.yaml --gpu 0
```

[完整部署、训练和验证说明](../../../docs/22_fish_diffusion.md)。

从原始录音亲自创建配置、训练和混音见 [手动实操](../../../docs/24_fish_diffusion_manual.md)；训练参数依据见 [优化实验](../../../docs/23_fish_diffusion_optimization.md)。最终跨模型报告可选，06 完成即有成品。
