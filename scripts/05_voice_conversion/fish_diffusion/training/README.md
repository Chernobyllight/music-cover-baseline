# Fish Diffusion 独立训练

显式执行 `create_config → 00_prepare → 01_features → 02_train → 03_validate → 04_export`。每个目录 README 含命令、输入和中间产物路径。这些编号属于 Fish 的训练链路，不修改歌曲主流程 00–04。

- [create_config](./create_config/README.md)
- [00_prepare](./00_prepare/README.md)
- [01_features](./01_features/README.md)
- [02_train](./02_train/README.md)
- [03_validate](./03_validate/README.md)
- [04_export](./04_export/README.md)

从原始录音开始的完整实例及每步路径见 [13 步手册](../../../../docs/24_fish_diffusion_manual.md)。以下是第一个训练节点的调用，前提是 create_config 已经成功创建新 YAML：

```bash
bash scripts/05_voice_conversion/fish_diffusion/training/00_prepare/run.sh \
  --config configs/voice_training/fish_diffusion/shan_yi_chun_manual_v001.yaml \
  --run runs/voice_training/shan_yi_chun_fish_manual_v001
```

输入来自本次分离的 recordings；输出 `data/voice_datasets/shan_yi_chun/fish_diffusion/manual_v001/manifest.json` 和 train/validation/test WAV。模型训练与歌曲运行使用不同 run，所有阶段使用 base `/opt/conda/bin/python`。新运行按 5000 步示例完成预热/衰减，具体配置可以调整，不要求用户逐个筛选检查点。
