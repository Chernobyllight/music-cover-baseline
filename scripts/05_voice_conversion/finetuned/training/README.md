# 目标音色训练流程

按 `00_prepare` → `01_train` → `02_validate` → `03_export` 执行，各文件夹 README 都含具体命令和中间产物。它们是歌曲主流程之外的训练链路。

如果手中是带伴奏的原始录音，先运行[参考人声分离脚本](../../../voice_data/separate/README.md)，再运行[训练配置生成脚本](create_config/README.md)，最后执行这里的 00_prepare 切片。两个准备脚本分别对应手动验证流程的第 4、5 步，均使用 base Python，不依赖 VC_* 变量，不复制旧训练配置或旧切片。配置生成器从新分离的 lead.wav 清单和命令参数构建 YAML；已有干净目标人声时仍可直接编写符合格式的训练 YAML。

第 5 步完整示例（第 4 步分离批次及歌曲 00/04 必须已成功）：

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/finetuned/training/create_config/run.sh \
  --voice-id shan_yi_chun --version manual_20261102 \
  --recordings-dir data/voice_training/shan_yi_chun/manual_20261102/recordings \
  --source-run runs/lhd_lv_zhou_shan_yi_chun_modular_test \
  --validation-ids 1698244260 1842315470 \
  --test-ids 1843079986 1864010428 \
  --validation-windows 36:12 156:12 --max-steps 1000
```

新配置位于 `/data/liuhongda/music-cover-baseline/configs/voice_training/shan_yi_chun_seedvc_manual_20261102.yaml`，之后通过 00_prepare 的 `--config` 传入；训练 run 使用 `runs/voice_training/shan_yi_chun_seedvc_manual_20261102`。预览加 `--plan`，不会写文件。数据集和模型导出路径根据 voice-id/version 推导，也可显式覆盖。

以下为原有 smoke 示例入口，完整从原始素材开始的单依纯流程见上方第 4、5 步文档：

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
bash scripts/05_voice_conversion/finetuned/training/00_prepare/run.sh \
  --config configs/voice_training/sun_yanzi_seed_vc_smoke.yaml \
  --run runs/voice_training/sun_yanzi_seed_vc_smoke_001
```

读取目标歌手原始录音，生成 `data/voice_datasets/sun_yanzi/smoke_v001/manifest.json` 与训练切片供 01 使用。训练完成后通过 `finetuned/run.sh --voice-profile .../voice.json` 接回歌曲主流程05。

[完整手册](../../../../docs/13_voice_adaptation.md)

训练、验证和导出均使用 `/opt/conda/bin/python`。配置相对路径以项目根目录解析，运行状态中的 output 相对训练 run。单依纯 + 绿洲的正式 1000 步例子见[完整手册第 7 节](../../../../docs/13_voice_adaptation.md#7-单依纯--绿洲base-环境训练与单版本混音)。
