# Fish Diffusion 训练优化：单依纯 × 绿洲

本文是优化实验及历史计数；最新由使用者从原始录音重新生成配置、训练和混音的步骤见 [手动实操](24_fish_diffusion_manual.md)。新手动数据为 96 段，本文实验数据为 95 段，分数与计数不可混用。

本轮保持现有 00–04 的人声、伴奏、曲风与参考不变，只比较 Fish 的训练配置。原 1000 步模型和歌曲结果保留，Seed-VC、SVC、RVC 不重新训练。全部使用 base `/opt/conda/bin/python`，没有增加依赖或新解释器。

## 数据、筛选和结论边界

训练集为 95 段、713.30 秒，来自 24 个原始录音组；验证和测试分别为 8 段、约 60 秒、2 个录音组。训练来源：`data/voice_training/shan_yi_chun/svc_manual_20261104/dataset`。没有新增音频，只有训练集使用音高增强。

固定筛选使用 4 段验证自重建和绿洲的 `36:12`、`156:12` 两个转换窗口。筛选分数为两类 ECAPA 平均值的等权平均；同时报告 log10 Mel 平均绝对误差。测试集在固定检查点后才进行评估，未用于调参。ECAPA 自重建分数、跨歌曲音色分数和整曲参考分数的比较对象不同，不能混用。

| 配置 | 步数 | 验证自相似度 ↑ | 绿洲窗口对独立参考 ↑ | Mel 误差 ↓ |
|---|---:|---:|---:|---:|
| 原版：1000 步预热后即停止 | 1000 | 0.5274 | 0.4790 | 0.7784 |
| 原增强、完整学习率过程 | 3000 | 0.6352 | 0.5331 | 0.6541 |
| **原增强、完整学习率过程，本轮选中** | **5000** | **0.6770** | **0.5422** | **0.6119** |
| 降低峰值学习率至 5e-5 | 3000 | 0.6440 | 0.5321 | 0.7046 |
| 降低峰值学习率至 5e-5 | 5000 | 0.6654 | 0.5521 | 0.6403 |
| 原增强继续训练，学习率 2e-5 | 10000 | 0.6664 | 0.5373 | 0.6686 |
| 无音高增强、batch=8、crop=512、峰值 1e-4 | 5000 | 0.6712 | 0.5230 | 0.5914 |
| 无音高增强、batch=8、crop=512、峰值 1e-4 | 10000 | 0.6780 | 0.5367 | 0.5618 |

5000 步原增强方案按预先固定的综合分数选中；几个新方案的分差较小，不表示统计上显著优于其他方案。无增强方案的频谱误差更低，但本轮音色筛选分数没有超过选中方案。不能把这些差异单独归因于某一个参数，也不能据此声称 Fish 超过 Seed-VC 或人工听感已合格。

原 1000 步停在预热终点，没有执行学习率衰减。配置生成器现在会显示 `training_advisories`，提醒预热内结束或提前截断衰减的配置；显式的短流程测试仍然允许。

## 从头复现选中方案

以下使用新的 `quality_5000` 版本名，不复用旧训练 YAML 或旧训练 run。源数据和源歌曲 run 是用户明确指定的输入。示例目录已存在时更换版本名和下游路径。

```bash
cd /data/liuhongda/music-cover-baseline

bash scripts/05_voice_conversion/fish_diffusion/training/create_config/run.sh \
  --voice-id shan_yi_chun --version quality_5000 \
  --source-dataset data/voice_training/shan_yi_chun/svc_manual_20261104/dataset \
  --source-run runs/lhd_lv_zhou_shan_yi_chun_modular_test \
  --validation-windows 36:12 156:12 \
  --max-steps 5000 --save-every 1000 \
  --batch-size 4 --crop-frames 384 \
  --warmup-steps 1000 --decay-steps 5000 \
  --learning-rate 0.0004 --start-learning-rate 0.00001 --final-learning-rate 0.00002 \
  --pitch-shift-probability 1.5 --seed 42 --gpu 0
```

输出：`configs/voice_training/fish_diffusion/shan_yi_chun_quality_5000.yaml`。所有数据和模型路径保存为项目相对路径；项目根目录由脚本自动定位。首次部署的公共权重见 [Fish 部署说明](22_fish_diffusion.md)。

依次运行准备、特征、训练、验证、导出，各阶段读取前一阶段清单：

```bash
bash scripts/05_voice_conversion/fish_diffusion/training/00_prepare/run.sh \
  --config configs/voice_training/fish_diffusion/shan_yi_chun_quality_5000.yaml \
  --run runs/voice_training/shan_yi_chun_fish_quality_5000

bash scripts/05_voice_conversion/fish_diffusion/training/01_features/run.sh \
  --config configs/voice_training/fish_diffusion/shan_yi_chun_quality_5000.yaml \
  --run runs/voice_training/shan_yi_chun_fish_quality_5000

bash scripts/05_voice_conversion/fish_diffusion/training/02_train/run.sh \
  --config configs/voice_training/fish_diffusion/shan_yi_chun_quality_5000.yaml \
  --run runs/voice_training/shan_yi_chun_fish_quality_5000

bash scripts/05_voice_conversion/fish_diffusion/training/03_validate/run.sh \
  --config configs/voice_training/fish_diffusion/shan_yi_chun_quality_5000.yaml \
  --run runs/voice_training/shan_yi_chun_fish_quality_5000

bash scripts/05_voice_conversion/fish_diffusion/training/04_export/run.sh \
  --config configs/voice_training/fish_diffusion/shan_yi_chun_quality_5000.yaml \
  --run runs/voice_training/shan_yi_chun_fish_quality_5000
```

| 阶段 | 主要输出 |
|---|---|
| 准备 | `data/voice_datasets/shan_yi_chun/fish_diffusion/quality_5000/manifest.json` 和分组 WAV |
| 特征 | 同目录 `features/`；阶段清单中的哈希和增强参数 |
| 训练 | `runs/voice_training/shan_yi_chun_fish_quality_5000/checkpoints/step_00005000.ckpt` |
| 验证 | 训练 run 内 `stages/03_validate/attempt_*/artifacts/validation.json` 和试听 WAV |
| 导出 | `models/voices/shan_yi_chun/fish_diffusion/quality_5000/voice.json` 和 `model.ckpt` |

质量标记不阻止保留和导出音频；技术检查通过不等于听感合格。选择其他训练步数时可显式向 03_validate 传 `--checkpoint`，没有固定步数限制。

创建推理配置并生成完整混音：

```bash
bash scripts/05_voice_conversion/fish_diffusion/create_inference_config/run.sh \
  --voice-profile models/voices/shan_yi_chun/fish_diffusion/quality_5000/voice.json \
  --output configs/voice_inference/fish_diffusion/shan_yi_chun_quality_5000.yaml \
  --sampler-interval 10 --noise-predictor plms --seed 42

bash scripts/05_voice_conversion/fish_diffusion/validate/run.sh \
  --source-run runs/lhd_lv_zhou_shan_yi_chun_modular_test \
  --run runs/lhd_lv_zhou_shan_yi_chun_fish_quality_5000 \
  --config configs/voice_inference/fish_diffusion/shan_yi_chun_quality_5000.yaml \
  --gpu 0
```

完整音频位于新歌曲 run 的 `stages/06_mix/attempt_001/artifacts/final.wav` 和 `final.mp3`。源 run 的 00–04 不重新生成。推理保持输入音高，PLMS 采样间隔 10，8 秒分块、0.2 秒重叠、0.5 秒上下文。

## 本轮实际保存的文件

- 筛选与测试：`runs/fish_optimization_20261105/screening.json`、`selection.json`、`heldout_test/report.json`。
- 选中检查点：`runs/voice_training/shan_yi_chun_fish_opt_full_20261105/checkpoints/step_00005000.ckpt`。该训练 run 还保存了未选中的 10000 步继续训练结果，导出明确选择 5000 步。
- 训练配置：`configs/voice_training/fish_diffusion/shan_yi_chun_opt_full_20261105.yaml`。本轮控制变量实验显式复用了首轮的特征缓存；上面的从头复现命令会创建新缓存。
- 导出音色：`models/voices/shan_yi_chun/fish_diffusion/opt_full_20261105/voice.json`。
- 新歌曲 run：`runs/lhd_lv_zhou_shan_yi_chun_fish_optimized_20261105`。
- 推理配置：`configs/voice_inference/fish_diffusion/shan_yi_chun_optimized_20261105.yaml`。

保存检查点、纯人声和最终混音，不自动删除旧模型或其它路线的数据。

## 独立测试结果

选择固定后，对此前未用于筛选的全部 8 段测试录音进行自重建：

| 模型 | 平均 ECAPA 自相似度 ↑ | 平均 log10 Mel 误差 ↓ |
|---|---:|---:|
| 原 1000 步 | 0.5087 | 0.6151 |
| 选中 5000 步 | 0.7156 | 0.4666 |

频谱误差相对下降约 24.1%。这是同一目标歌手录音的重建诊断，不等于自然度评分；8 段测试来自 2 个录音组，不能据此泛化到所有歌手和曲风。最终歌曲的定量对照位于歌曲 run 的 `fish_validation/comparison.json`。

## 同一份 04 输入的最终歌曲对照

| 版本 | 整曲 ECAPA ↑ | 音高误差 ≤ 1 半音 |
|---|---:|---:|
| 原 Fish 1000 步 | 0.5581 | 97.95% |
| 优化后 Fish 5000 步 | 0.7663 | 97.42% |
| 已有 Seed-VC 零样本 | 0.8115 | 99.49% |

音色指标改善，但音高一致性略降，不能描述为所有指标全面提升。Seed-VC 同时将该评估参考用于推理，Fish 不使用它作为条件，因此此表是工程诊断，不是公平的独立总体模型排名。新混音时长 257.399 秒，WAV/MP3 均已生成；试听才能确认自然度和个人偏好。

本轮 255 项回归测试通过；配置、源码和文档链接检查通过。未安装新依赖、未新建解释器，旧模型与旧结果保留。
