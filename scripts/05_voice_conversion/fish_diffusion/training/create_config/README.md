# Fish Diffusion · 创建训练配置

从显式参数和本次分离清单生成新 YAML；全部使用 base Python。以下命令从项目根目录执行，相对参数路径按项目根目录解析。示例 `manual_v001` 必须是新版本，完整前置步骤见 [手动实操](../../../../../docs/24_fish_diffusion_manual.md)。

| 参数 | 输入/输出 | 前置与创建时机 |
|---|---|---|
| --recordings-dir | 已有输入 | voice_data/separate 完成的 manifest.json 与 lead.wav |
| --source-run | 已有输入 | 歌曲 00–04 成功运行，供验证，不是训练 run |
| --output | 新输出 | 本命令创建 YAML；父目录可存在，文件不能已存在 |
| --dataset-dir | 新输出位置 | 写入 YAML，后续训练 00/01 创建切片和特征，不提前 mkdir |
| --export-dir | 新输出位置 | 写入 YAML，后续训练 04 创建人物包，不提前 mkdir |

## 使用实例

```bash
bash scripts/05_voice_conversion/fish_diffusion/training/create_config/run.sh \
  --voice-id shan_yi_chun --version manual_v001 \
  --recordings-dir data/voice_training/shan_yi_chun/fish_manual_v001/recordings \
  --source-run runs/lhd_lv_zhou_shan_yi_chun_fish_manual_v001 \
  --validation-ids 1698244260 1842315470 --test-ids 1843079986 1864010428 \
  --validation-windows 36:12 156:12 \
  --dataset-dir data/voice_datasets/shan_yi_chun/fish_diffusion/manual_v001 \
  --export-dir models/voices/shan_yi_chun/fish_diffusion/manual_v001 \
  --output configs/voice_training/fish_diffusion/shan_yi_chun_manual_v001.yaml \
  --segment-seconds 8 --max-steps 5000 --save-every 1000 \
  --batch-size 4 --crop-frames 384 --warmup-steps 1000 --decay-steps 5000 \
  --learning-rate 0.0004 --start-learning-rate 0.00001 --final-learning-rate 0.00002 \
  --pitch-shift-probability 1.5 --precision bfloat16 --seed 42 --gpu 0
```

输出 `configs/voice_training/fish_diffusion/shan_yi_chun_manual_v001.yaml`，看到 `Created ...yaml` 再执行训练 00。`--plan` 只检查，不创建文件。省略三个输出参数时根据 voice-id/version 推导相同的默认位置。

验证/测试编号须真实存在且不交叉；其他录音归 train，参考所在录音在分离时排除。窗口是 START:DURATION 秒，按歌曲实际演唱调整。5000 步示例完成 1000 步预热与衰减；短流程测试仍可显式配置，生成器会提醒截断调度。

另一种输入 `--source-dataset` 与 recordings-dir 互斥，显式复用已有 train/validation/test manifest 和音频；亲自从原录音构建时采用上面的 recordings-dir。两者都不读取旧训练 YAML。
