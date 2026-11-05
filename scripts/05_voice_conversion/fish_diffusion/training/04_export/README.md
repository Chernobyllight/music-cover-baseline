# Fish Diffusion · training/04_export

导出可移植的歌手模型版本。

全部使用 `/opt/conda/bin/python`。脚本自行定位项目根目录，不依赖当前目录或 VC_* 变量。

输入：03_validate 的 validation 报告和哈希匹配的权重。test 报告不用于模型选择/导出。

输出：models/voices/shan_yi_chun/fish_diffusion/manual_v001/voice.json 及 model.ckpt、config.json、validation.json。质量拒绝时导出带 review_required 标记；已有版本不覆盖。

## 具体使用示例

```bash
bash scripts/05_voice_conversion/fish_diffusion/training/04_export/run.sh \
  --config configs/voice_training/fish_diffusion/shan_yi_chun_manual_v001.yaml \
  --run runs/voice_training/shan_yi_chun_fish_manual_v001
```

查看本阶段中间结果：`/data/liuhongda/music-cover-baseline/runs/voice_training/shan_yi_chun_fish_manual_v001/stages/04_export/attempt_*/artifacts/`。以训练 run 的 `state.json` 为准。上游阶段必须显式执行成功，不会暗中补跑。

[完整部署、训练和验证说明](../../../../../docs/22_fish_diffusion.md)。

读取 03 的 validation 报告及哈希匹配权重；目标导出目录不得预先存在。质量未通过时保留音频并导出待复核标记。

[从原始录音开始的手动实操](../../../../../docs/24_fish_diffusion_manual.md)。
