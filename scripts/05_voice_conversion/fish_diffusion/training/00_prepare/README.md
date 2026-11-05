# Fish Diffusion · training/00_prepare

建立 Fish 独立数据副本，保留按原始录音划分的 train/validation/test。

全部使用 `/opt/conda/bin/python`。脚本自行定位项目根目录，不依赖当前目录或 VC_* 变量。

输入：create_config 创建的 YAML、显式指定的歌手数据。该阶段是 Fish 内部准备，不修改主流程 scripts/00_prepare。

输出：data/voice_datasets/shan_yi_chun/fish_diffusion/manual_v001/manifest.json 和 train/validation/test WAV；不提取特征。

## 具体使用示例

```bash
bash scripts/05_voice_conversion/fish_diffusion/training/00_prepare/run.sh \
  --config configs/voice_training/fish_diffusion/shan_yi_chun_manual_v001.yaml \
  --run runs/voice_training/shan_yi_chun_fish_manual_v001
```

查看本阶段中间结果：`/data/liuhongda/music-cover-baseline/runs/voice_training/shan_yi_chun_fish_manual_v001/stages/00_prepare/attempt_*/artifacts/`。以训练 run 的 `state.json` 为准。上游阶段必须显式执行成功，不会暗中补跑。

[完整部署、训练和验证说明](../../../../../docs/22_fish_diffusion.md)。

先完成 create_config；`--config` 只读取已有 YAML，不负责创建。报 FileNotFoundError 时补配置生成，不重做已成功的录音分离。

[从原始录音开始的手动实操](../../../../../docs/24_fish_diffusion_manual.md)。
