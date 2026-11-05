# Fish Diffusion · training/02_train

微调扩散声学模型和歌手嵌入，冻结内容编码器及声码器。

全部使用 `/opt/conda/bin/python`。脚本自行定位项目根目录，不依赖当前目录或 VC_* 变量。

输入：01_features 的特征与官方预训练权重。使用上游 DiffSinger/WaveNet 和 noise prediction loss，项目控制 AdamW、调度、保存与续训。

输出：训练 run/checkpoints/step_XXXXXXXX.ckpt；阶段 artifacts/losses.csv 和 training.json。继续训练显式加 --resume --max-steps 2000。默认使用配置最终步数，不自动选多个检查点。

## 具体使用示例

```bash
bash scripts/05_voice_conversion/fish_diffusion/training/02_train/run.sh \
  --config configs/voice_training/fish_diffusion/shan_yi_chun_manual_v001.yaml \
  --run runs/voice_training/shan_yi_chun_fish_manual_v001
```

查看本阶段中间结果：`/data/liuhongda/music-cover-baseline/runs/voice_training/shan_yi_chun_fish_manual_v001/stages/02_train/attempt_*/artifacts/`。以训练 run 的 `state.json` 为准。上游阶段必须显式执行成功，不会暗中补跑。

[完整部署、训练和验证说明](../../../../../docs/22_fish_diffusion.md)。

默认使用本次配置的 5000 步；中断可追加 `--resume`。更新声学主干和人物 embedding，内容编码器/声码器冻结。

[从原始录音开始的手动实操](../../../../../docs/24_fish_diffusion_manual.md)。
