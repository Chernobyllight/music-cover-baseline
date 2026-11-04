# 独立 SVC 训练

依次执行 00_prepare、01_train、02_validate、03_export。此流程使用已有 base，不会调用 Seed-VC 训练。

需要亲自生成新配置时，先运行 [create_config](create_config/README.md)，不要复制下面的旧实验 YAML。单依纯的数据准备与配置生成命令见 [SVC 手动流程](../../../../docs/svc_config_scripts.md)。模型结构 JSON 由 00 根据官方模板新生成。

第一步实例：

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/05_voice_conversion/so_vits_svc/training/00_prepare/run.sh \
  --config configs/voice_training/so_vits_svc/yang_kun.yaml \
  --run runs/voice_training/yang_kun_so_vits_svc_v001_20261028
```

输入是 `data/raw/references/yang_kun/dataset/manifest.json`；00 生成 SVC 专用数据后供 01 使用。每个子目录 README 有下一步具体命令。详见 [完整指南](../../../../docs/18_so_vits_svc.md)。
