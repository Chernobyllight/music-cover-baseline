# 人声数据配置生成

第 5 步；固定 base `/opt/conda/bin/python`，相对路径按项目根目录解析。配置生成器可加 `--plan` 只检查预览，已有输出不覆盖。

前置：第 4 步已完成新分离，且第 3 步的歌曲运行已建立。下面的 recordings 必须是你本次 `scripts/voice_data/separate/run.sh` 产生的目录。

```bash
bash /data/liuhongda/music-cover-baseline/scripts/voice_data/create_config/run.sh \
  --voice-id shan_yi_chun \
  --version rvc_manual_20261103 \
  --recordings-dir data/voice_training/shan_yi_chun/rvc_manual_20261103/recordings \
  --source-run runs/lhd_lv_zhou_shan_yi_chun_rvc_manual_20261103 \
  --validation-ids 1698244260 1842315470 \
  --test-ids 1843079986 1864010428 \
  --segment-seconds 8 \
  --output configs/voice_training/rvc/shan_yi_chun_manual_20261103_data.yaml
```

检查分离批次状态、lead.wav 哈希及来源、集合编号、重复素材跨集合泄漏和当前 00 的 reference 产物。余下有效素材为 train，不读取目录中任意残留文件。不加载任何 Seed-VC 权重。

输出 `/data/liuhongda/music-cover-baseline/configs/voice_training/rvc/shan_yi_chun_manual_20261103_data.yaml`，标记 `purpose: dataset_preparation_only`，不包含模型权重和训练参数。默认数据集目标为 `/data/liuhongda/music-cover-baseline/data/voice_training/shan_yi_chun/rvc_manual_20261103/dataset`。省略 `--output` 时配置默认位于 `configs/voice_data/<voice-id>_<version>.yaml`。

第 6 步仍由你执行现有数据准备入口，它只进行解码、裁剪与划分：

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/finetuned/training/00_prepare/run.sh \
  --config configs/voice_training/rvc/shan_yi_chun_manual_20261103_data.yaml \
  --run runs/voice_data/shan_yi_chun_rvc_manual_20261103
```

输出新数据集的 `train/`、`validation/`、`test/` 和 `manifest.json`。这个数据配置不能传给 Seed-VC 训练阶段；原有 Seed-VC 训练配置格式和入口继续支持。


完整四步见[配置脚本说明](../../../docs/rvc_config_scripts.md)。
