# 人物模型训练、导出与主流程衔接

仅在需要新训练 Seed-VC/SVC/RVC 人物模型时阅读。已有完整导出包可直接按 [主文件](../SKILL.md) 运行 05。训练独立于歌曲主流程，不会在执行 05 时隐式开始。

所有命令从项目根目录执行，训练脚本使用 `/opt/conda/bin/python`。`my_voice`、`v001` 是示例人物和新版本标识；每次实验使用新配置、数据目录、训练 run 和导出目录。公共权重可复用，不把别人已生成的实验文件当成自己的生成步骤。

## 1. 前置数据与独立目录

先完成主流程 `runs/my_cover_v001` 的 00–04，供训练验证使用。训练用目标人物的多份录音；04 的源人声不是目标人物训练集，也不生成单曲伪标签。

示例原始素材布局：

```text
data/raw/references/my_voice/
├── reference.wav                  # 独立参考片段，供主流程 00
└── sources/
    ├── train_01/original.wav      # 以下均为使用者自己的真实录音
    ├── train_02/original.wav
    ├── validation_01/original.wav
    └── test_01/original.wav
```

上面的编号是演示命名；传参使用自己真实素材编号。train、validation、test 必须按独立录音/歌曲分组，不能把同一首歌切几段分散到不同集合。参考片段所在歌曲也应单独留出。先试听并排除多人对唱、严重混响、明显削顶或伴奏泄漏的录音；自动校验不等于身份与音质鉴定。

## 2. 从原始素材重新分离主唱

先按 [weights.md](weights.md) 准备 BS-RoFormer 和 Karaoke 资源。默认两级分离，不训练模型：

```bash
bash scripts/voice_data/separate/run.sh \
  --input-dir data/raw/references/my_voice/sources \
  --pattern '*/original.wav' \
  --output-dir data/voice_training/my_voice/recordings_v001 --gpu 0 --plan
bash scripts/voice_data/separate/run.sh \
  --input-dir data/raw/references/my_voice/sources \
  --pattern '*/original.wav' \
  --output-dir data/voice_training/my_voice/recordings_v001 --gpu 0
```

预期输出 `recordings_v001/manifest.json`，以及每个编号下的 `lead.wav`、`vocals.wav`、`instrumental.wav`、`backing.wav`、`separation.json`。整批成功并试听后，进入下方所选路线。这里尚未产生训练切片。需要续跑使用 `--resume`，改变所选素材集合用新目录。

## 3A. Seed-VC：从分离结果生成配置并训练

先准备 Seed-VC 公共权重、OpenVoiceV2、`se_db.pt` 和 ECAPA。命令中的 `36:12 156:12` 指在 04 人声时间轴上取两个 12 秒验证窗；**仅在歌曲超过 168 秒且这两处有清晰演唱时使用**，否则听取本次 04 人声后更换。窗口不是通用常量。

```bash
bash scripts/05_voice_conversion/finetuned/training/create_config/run.sh \
  --voice-id my_voice --version v001 \
  --recordings-dir data/voice_training/my_voice/recordings_v001 \
  --source-run runs/my_cover_v001 \
  --validation-ids validation_01 --test-ids test_01 \
  --validation-windows 36:12 156:12 \
  --segment-seconds 8 --max-steps 1000 --save-every 500 \
  --batch-size 1 --learning-rate 0.00001 --seed 42 \
  --validation-steps 30 --validation-seed 42 --gpu 0

bash scripts/05_voice_conversion/finetuned/training/00_prepare/run.sh \
  --config configs/voice_training/my_voice_seedvc_v001.yaml --run runs/voice_training/my_voice_seedvc_v001
bash scripts/05_voice_conversion/finetuned/training/01_train/run.sh \
  --config configs/voice_training/my_voice_seedvc_v001.yaml --run runs/voice_training/my_voice_seedvc_v001
bash scripts/05_voice_conversion/finetuned/training/02_validate/run.sh \
  --config configs/voice_training/my_voice_seedvc_v001.yaml --run runs/voice_training/my_voice_seedvc_v001 \
  --checkpoint runs/voice_training/my_voice_seedvc_v001/checkpoints/step_001000.pth
bash scripts/05_voice_conversion/finetuned/training/03_export/run.sh \
  --config configs/voice_training/my_voice_seedvc_v001.yaml --run runs/voice_training/my_voice_seedvc_v001
```

配置生成器从参数和本次分离 manifest 构建 YAML；追加 `--plan` 可先检查，不写文件。1000 是本示例步数，不是所有歌手的最佳步数。显式验证该步权重，避免使用者逐个选择检查点。

| 步骤 | 新输出 |
|---|---|
| create_config | `configs/voice_training/my_voice_seedvc_v001.yaml` |
| 训练 00 | `data/voice_training/my_voice/v001/dataset/manifest.json`、train/validation/test 切片 |
| 训练 01 | 训练 run 下 `checkpoints/step_001000.pth`、训练日志 |
| 训练 02 | 当前验证 attempt 的 `artifacts/validation.json`、指标和试听材料 |
| 训练 03 | `models/voices/my_voice/seed_vc/v001/voice.json` 及整个导出包 |

然后回到主文件执行 Seed-VC finetuned 05，再执行 06。默认三候选使用同一个人物模型，变化是推理种子，不是重新训练三次。

## 3B. SVC/RVC：先从本次分离结果生成公共切片

此公共步骤只准备数据，不加载 Seed-VC 模型；之所以调用 finetuned 的训练 00，是因为当前项目复用同一个数据整理器。

```bash
bash scripts/voice_data/create_config/run.sh \
  --voice-id my_voice --version prepared_v001 \
  --recordings-dir data/voice_training/my_voice/recordings_v001 \
  --source-run runs/my_cover_v001 \
  --validation-ids validation_01 --test-ids test_01 --segment-seconds 8 \
  --output configs/voice_data/my_voice_prepared_v001.yaml
bash scripts/05_voice_conversion/finetuned/training/00_prepare/run.sh \
  --config configs/voice_data/my_voice_prepared_v001.yaml \
  --run runs/voice_data/my_voice_prepared_v001
```

输出为 `data/voice_training/my_voice/prepared_v001/dataset/manifest.json` 和三个集合的切片。该数据配置带 `purpose: dataset_preparation_only`，不能直接交给 Seed-VC 训练 01。下一步 `--source-dataset` 指向这个 dataset，不能指向上面的 recordings。

### SVC 训练、验证与导出

```bash
/opt/conda/bin/python scripts/setup_svc.py --download --training
/opt/conda/bin/python scripts/setup/setup_voice_training.py --scope evaluation
bash scripts/05_voice_conversion/so_vits_svc/preflight/run.sh --scope training --gpu 0

bash scripts/05_voice_conversion/so_vits_svc/training/create_config/run.sh \
  --voice-id my_voice --version v001 \
  --source-dataset data/voice_training/my_voice/prepared_v001/dataset \
  --source-run runs/my_cover_v001 --validation-windows 36:12 156:12 \
  --max-steps 4000 --save-every 1000 --batch-size 6 \
  --learning-rate 0.0001 --seed 42 --gpu 0 \
  --peak-policy attenuate_if_needed --peak-limit 0.99

bash scripts/05_voice_conversion/so_vits_svc/training/00_prepare/run.sh \
  --config configs/voice_training/so_vits_svc/my_voice_v001.yaml --run runs/voice_training/my_voice_svc_v001
bash scripts/05_voice_conversion/so_vits_svc/training/01_train/run.sh \
  --config configs/voice_training/so_vits_svc/my_voice_v001.yaml --run runs/voice_training/my_voice_svc_v001
bash scripts/05_voice_conversion/so_vits_svc/training/02_validate/run.sh \
  --config configs/voice_training/so_vits_svc/my_voice_v001.yaml --run runs/voice_training/my_voice_svc_v001 \
  --checkpoint runs/voice_training/my_voice_svc_v001/checkpoints/step_00004000.pth
bash scripts/05_voice_conversion/so_vits_svc/training/03_export/run.sh \
  --config configs/voice_training/so_vits_svc/my_voice_v001.yaml --run runs/voice_training/my_voice_svc_v001
```

同样必须按本次歌曲调整验证窗口。模型为 ContentVec 768l12 + RMVPE + 标准 NSF-HiFiGAN；先用当前通用路线，不自动启动历史 hubertsoft、扩散或增强器实验。

输出数据在 `data/voice_datasets/my_voice/so_vits_svc/v001/`；模型结构 `model_config.json` 由训练 00 新生成；导出在 `models/voices/my_voice/so_vits_svc/v001/`。`attenuate_if_needed` 只在转 PCM16 时对超限峰值降低整体增益，不修改原录音、不移调。

### RVC 训练、索引、验证与导出

```bash
/opt/conda/bin/python scripts/setup_rvc.py --download --training
/opt/conda/bin/python scripts/setup/setup_voice_training.py --scope evaluation
bash scripts/05_voice_conversion/rvc/preflight/run.sh --scope training --gpu 0

bash scripts/05_voice_conversion/rvc/training/create_config/run.sh \
  --voice-id my_voice --version v001 \
  --source-dataset data/voice_training/my_voice/prepared_v001/dataset \
  --source-run runs/my_cover_v001 --validation-windows 36:12 156:12 \
  --max-steps 4000 --save-every 1000 --batch-size 6 --learning-rate 0.0001 \
  --input-crop-frames 400 --seed 42 --f0-predictor rmvpe \
  --index-rate 0.5 --protect 0.33 \
  --chunk-seconds 8 --overlap-seconds 0.2 --pad-seconds 0.5 --gpu 0

bash scripts/05_voice_conversion/rvc/training/00_prepare/run.sh \
  --config configs/voice_training/rvc/my_voice_v001.yaml --run runs/voice_training/my_voice_rvc_v001
bash scripts/05_voice_conversion/rvc/training/01_features/run.sh \
  --config configs/voice_training/rvc/my_voice_v001.yaml --run runs/voice_training/my_voice_rvc_v001
bash scripts/05_voice_conversion/rvc/training/02_train/run.sh \
  --config configs/voice_training/rvc/my_voice_v001.yaml --run runs/voice_training/my_voice_rvc_v001
bash scripts/05_voice_conversion/rvc/training/03_index/run.sh \
  --config configs/voice_training/rvc/my_voice_v001.yaml --run runs/voice_training/my_voice_rvc_v001
bash scripts/05_voice_conversion/rvc/training/04_validate/run.sh \
  --config configs/voice_training/rvc/my_voice_v001.yaml --run runs/voice_training/my_voice_rvc_v001
bash scripts/05_voice_conversion/rvc/training/05_export/run.sh \
  --config configs/voice_training/rvc/my_voice_v001.yaml --run runs/voice_training/my_voice_rvc_v001
```

RVC 采用当前 v2/F0/48k 路线。默认验证配置最终步数的 `checkpoints/model_00004000.pth`，可以显式传入自己需要的 `--checkpoint`；不是硬性限制只能用 4000 步。检索索引只由训练集内容特征建立。

输出数据在 `data/voice_datasets/my_voice/rvc/v001/`；训练 run 下有检查点、`added.index`、`index_manifest.json` 和验证报告；导出在 `models/voices/my_voice/rvc/v001/`。4000 是当前示例预算，不承诺最佳听感。

## 4. 回到歌曲主流程 05、06

使用 [SKILL.md 的 05 命令](../SKILL.md#05-的替代命令四选一) 为本次导出的模型新建推理配置并转换整曲，再执行 06。训练 run 的 `state.json` 和歌曲 run 的 `stage_state.json` 是两套状态，不能混用。

质量验收和执行成功分开记录：验收不合格仍可导出供试听，但 profile 标记 `quality_review_required`，05/06 保留音频并报告待复核。文件损坏、来源不匹配或模型根本未生成不是质量拒绝，不能跳过这些结构检查。

导出包已存在时换新版本；不要覆盖其它人的模型。源 run 仍被训练配置/验证引用时不要删除。更深入的格式与实现分别见仓库 [Seed-VC](../../docs/13_voice_adaptation.md)、[SVC](../../docs/18_so_vits_svc.md)、[RVC](../../docs/20_rvc.md) 文档；历史具体 run 编号只作背景，不作为新部署输入。
