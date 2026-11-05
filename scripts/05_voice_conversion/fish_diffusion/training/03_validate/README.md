# Fish Diffusion · training/03_validate

对独立歌手录音和源歌曲片段做换声与指标评估。

全部使用 `/opt/conda/bin/python`。脚本自行定位项目根目录，不依赖当前目录或 VC_* 变量。

输入：02_train 的最终模型、配置中 validation windows 和独立验证录音。可用 --checkpoint 指定同一训练数据身份的模型；--evaluation-split test 做测试集报告。

输出：artifacts/validation.json、metrics.csv、listening.html、各 probe 的 converted.wav。质量未通过或评估失败仍保留音频；只在推理无法产生结果时失败。

## 具体使用示例

```bash
bash scripts/05_voice_conversion/fish_diffusion/training/03_validate/run.sh \
  --config configs/voice_training/fish_diffusion/shan_yi_chun_manual_v001.yaml \
  --run runs/voice_training/shan_yi_chun_fish_manual_v001 \
  --checkpoint runs/voice_training/shan_yi_chun_fish_manual_v001/checkpoints/step_00005000.ckpt
```

查看本阶段中间结果：`/data/liuhongda/music-cover-baseline/runs/voice_training/shan_yi_chun_fish_manual_v001/stages/03_validate/attempt_*/artifacts/`。以训练 run 的 `state.json` 为准。上游阶段必须显式执行成功，不会暗中补跑。

[完整部署、训练和验证说明](../../../../../docs/22_fish_diffusion.md)。

默认最终步，也支持 --checkpoint。当前最多两个留出录音探针加歌曲窗口，不是全测试集遍历。evaluation-split=test 的报告不能用于导出；需重新跑 validation。

[从原始录音开始的手动实操](../../../../../docs/24_fish_diffusion_manual.md)。
