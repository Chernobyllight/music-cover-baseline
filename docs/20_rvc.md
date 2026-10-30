# RVC：黄绮珊歌手训练与独立音色转换

RVC 是 05 的额外后端，`backend=rvc`、`mode=finetuned`。Seed-VC zero-shot、Seed-VC finetuned、so-vits-svc 的入口继续保留。RVC 推理不会启动训练，也不把参考音频当作音色条件；参考音频仍供 02 音区分析和评估使用。

工程根目录 `/data/liuhongda/music-cover-baseline` 与 `/mnt/afs_f/liuhongda/music-cover-baseline` 为同一共享工程。下文命令在根目录执行。RVC 接入已纳入 v1.0 版本，固定上游版本见仓库恢复说明。

## 模型与环境

- 上游：[RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/tree/81eed5e8f68b6bed1789f682fe78cdd324495afc)，固定提交 `81eed5e8f68b6bed1789f682fe78cdd324495afc`。
- RVC v2：HuBERT/ContentVec 第 12 层 768 维、RMVPE、48 kHz 带 F0 的 NSF 生成器。
- 训练更新完整生成器和判别器；内容编码器、RMVPE 冻结。使用上游模型、数据整理类及 Mel、KL、对抗、特征匹配四项损失。训练包装器按实际优化步数保存，不用 epoch 冒充 step。
- 基于现有 `/opt/conda/bin/python`。主控制器不导入 RVC 重型依赖；候选评估使用现有 `envs/seed-vc`。不使用 WebUI、实时变声、VST、fairseq，不安装上游整套 requirements。
- 本地适配器处理上游 `infer/tools/train` 命名空间隔离；固定关闭 CUDA Graph；固定训练/推理共享音频归一化与 F0 处理。保留原始无声 F0=0，使保护参数和 NSF 清浊音判定使用真实 RMVPE 标记，未沿用该提交的全段 F0 插值。此差异在配置中记录为 `f0_policy=unvoiced_zero`。

权重由 `configs/rvc_assets.lock.json` 固定 URL、revision、大小和 SHA256：

| 位置 | 作用 |
|---|---|
| `models/RVC/hubert_base/config.json`、`preprocessor_config.json`、`pytorch_model.bin` | 768 维内容特征编码器 |
| `models/RVC/rmvpe.pt` | 音高与清浊音提取 |
| `models/RVC/pretrained_v2/f0G48k.pth` | 训练用预训练生成器 |
| `models/RVC/pretrained_v2/f0D48k.pth` | 训练用预训练判别器 |
| `models/voices/huang_qi_shan/rvc/v001_20261029/` | 配置指定的最终步数歌手权重、索引及模型档案 |

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
git submodule update --init third_party/rvc
/opt/conda/bin/python scripts/setup_rvc.py --training
# 新机器缺少权重时才使用，要求可以连接固定模型下载地址：
/opt/conda/bin/python scripts/setup_rvc.py --training --download
```

不需要初始化 RVC 内部的实时 VST 子模块。现有 SVC/RMVPE 文件不会被新权重覆盖。`rvc-base.installed.txt` 是当前环境快照；`requirements-rvc-base.txt` 是已用到的核心依赖清单，不是允许覆盖工作环境的安装指令。

## 数据与训练阶段

数据只来自 `/data/liuhongda/music-cover-baseline/data/raw/references/huang_qi_shan/`。直接读取其中已整理的 `dataset/manifest.json`；训练 52 段 416 秒，验证 8 段 64 秒，测试 5 段 40 秒，另有 20 秒 prompt 被排除。保留歌曲组隔离与文件哈希检查。素材是已筛选的人声估计，含有损来源和可能的分离残留，不等于录音棚纯干声。

RVC 派生数据保存在 `data/voice_datasets/huang_qi_shan/rvc/v001_20261029/`，不覆盖原数据。48 kHz FLOAT WAV 用于生成器训练；16 kHz 音频用于内容和 F0 提取。特征和检索索引只取训练集。当前小数据集采用精确 `IndexFlatL2`、8 近邻距离加权，避免近似索引参数成为额外变量；不是把目标歌手音频剪片拼接。

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"

bash scripts/05_voice_conversion/rvc/training/00_prepare/run.sh \
  --config configs/voice_training/rvc/huang_qi_shan.yaml \
  --run runs/voice_training/huang_qi_shan_rvc_v001_20261029
bash scripts/05_voice_conversion/rvc/training/01_features/run.sh \
  --config configs/voice_training/rvc/huang_qi_shan.yaml \
  --run runs/voice_training/huang_qi_shan_rvc_v001_20261029
bash scripts/05_voice_conversion/rvc/training/02_train/run.sh \
  --config configs/voice_training/rvc/huang_qi_shan.yaml \
  --run runs/voice_training/huang_qi_shan_rvc_v001_20261029
bash scripts/05_voice_conversion/rvc/training/03_index/run.sh \
  --config configs/voice_training/rvc/huang_qi_shan.yaml \
  --run runs/voice_training/huang_qi_shan_rvc_v001_20261029
bash scripts/05_voice_conversion/rvc/training/04_validate/run.sh \
  --config configs/voice_training/rvc/huang_qi_shan.yaml \
  --run runs/voice_training/huang_qi_shan_rvc_v001_20261029
bash scripts/05_voice_conversion/rvc/training/05_export/run.sh \
  --config configs/voice_training/rvc/huang_qi_shan.yaml \
  --run runs/voice_training/huang_qi_shan_rvc_v001_20261029
```

已有训练检查点时必须在 02 命令末尾加 `--resume`，并用 `--max-steps` 指定大于已恢复步数的新目标；如 `--resume --max-steps 4000`。原有导出版本不能覆盖，重新训练请使用新版本配置、dataset_dir、训练 run 和 export_dir。03 只依赖 01，因此可在开始训练前构建索引。

| 阶段 | 检查位置（相对训练 run） | 下游用途 |
|---|---|---|
| 00 | `stages/00_prepare/attempt_*/artifacts/dataset.json`；派生数据目录 `manifest.json` | 锁定输入划分、音频及身份 |
| 01 | `stages/01_features/attempt_*/artifacts/features.json`；派生数据目录 `features/`、`audio16/`、`train.txt` | 02 训练，03 建索引 |
| 02 | `checkpoints/step_*.pth`、`model_*.pth`；`stages/02_train/attempt_*/artifacts/training.json`、`losses.jsonl` | 前者含优化器/RNG供续训，后者为严格加载的推理参数 |
| 03 | `added.index`、`index_manifest.json` | 训练集索引、特征来源、向量数及校验值 |
| 04 | `validation.json`、`stages/04_validate/attempt_*/artifacts/` | 配置最终步数权重的验证参考、转换音频及验收指标 |
| 05 | `stages/05_export/attempt_*/artifacts/export.json` | 导出目录中的 `voice.json` 供正式 05 使用 |

训练配置为4000次实际更新、LR=1e-4、batch=6（最后一批可更小）、FP32/TF32，4秒输入随机裁剪，上游声码器损失窗口17280样本约0.36秒。**直接使用本次配置的4000步模型，固定index_rate=0.5、protect=0.33，不搜索检查点或推理参数。** 4000是本工程明确配置的训练目标，不声称是RVC上游的通用最佳步数。04在《穿越曙光》两个20秒窗口上验收；其他检查点仅用于记录、恢复。验证失败会明确停止，不静默改用另一个模型。需要验证其他权重时，在04命令中传入 `--checkpoint runs/voice_training/huang_qi_shan_rvc_v001_20261029/checkpoints/model_00002000.pth` 即可；接口不限制只能使用4000步。测试参考不进入训练、索引或候选选择。

## 与现有 Seed-VC 对齐生成

三首歌的原有曲风分别为：《穿越曙光》电影配乐式管弦灵魂乐、《Last Dance》复古 Funk/Soul、《绿洲》古巴 Mambo/大乐队爵士。复用原 run 的全部 00–04 产物，保持同一源人声、伴奏、歌词、乐谱和已应用音高；RVC 内部始终偏移0半音。RVC 的48k原生输出会另存，转换到44.1k后的 `converted.wav` 交给现有06。

下面是复用《绿洲》的独立例子。目标目录应尚不存在；源 run 必须一直保留，因为 fork 引用其上游文件并逐次校验哈希。

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
/opt/conda/bin/python scripts/fork_voice_run.py \
  --source-run runs/lhd_lv_zhou_mambo_big_band_huang_qi_shan_seedvc_20261029 \
  --run runs/my_lv_zhou_huang_qi_shan_rvc
bash scripts/05_voice_conversion/rvc/run.sh \
  --run runs/my_lv_zhou_huang_qi_shan_rvc \
  --config configs/voice_inference/rvc/huang_qi_shan.yaml \
  --gpu 0
bash scripts/run_baseline.sh \
  --run runs/my_lv_zhou_huang_qi_shan_rvc --from 06_mix --gpu 0
```

推理 YAML 显式指定 voice_profile、检索率、protect、分块及候选种子；**只指定 voice.json 不会自动应用推荐参数**。RVC 支持1至8个显式种子候选，当前完整对照与Seed微调一样提供42/43/44三个候选；每首歌按同一技术门槛与prompt诊断分数选择交付入口，独立测试参考另行评估推荐结果，不能用测试分数反选候选。

结果在新 run 的：

- `stages/05_voice_conversion/attempt_*/artifacts/converted.wav`、`conversion.json`、`candidates.json`；候选目录另有 `converted_native_48k.wav`。
- `deliverables/final.mp3`、`final.wav`、`review/`：复用06–11完成混音、分轨、MIDI与交付检查。

独立ECAPA在转换干人声上计算，使用原黄绮珊 dataset/test 的5段参考，双方用同一版本指标实现。报告F0误差≤1半音比例、有效帧覆盖、对齐、音频完整性，并提供相同伴奏的最终MP3。ECAPA不是身份概率或听感质量分；不能由它单独判断沙哑、气声、齿音和电音感。未重新加入已撤回的歌词检测流程。

## 显式 0.75 检索预设

项目默认 `huang_qi_shan.yaml` 保持 index_rate=0.5、protect=0.33。`huang_qi_shan_index075.yaml` 使用官方单次 WebUI 默认的 index_rate=0.75，其余推理与候选配置不变。官方批量 WebUI 的检索默认值为1.0，并未自动应用到本项目。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/05_voice_conversion/rvc/run.sh \
  --run runs/my_chuan_yue_shu_guang_rvc \
  --config configs/voice_inference/rvc/huang_qi_shan_index075.yaml --gpu 0
bash scripts/06_mix/run.sh --run runs/my_chuan_yue_shu_guang_rvc --gpu 0
```

前置是同 run 的 00–04 已完成；公平对照应先 fork 为新运行。已保留的三个 0.75 运行名含 `rvc_index075_20261029`，目前完成到06，最终 MP3在 `stages/06_mix/attempt_001/artifacts/final.mp3`，不能声称它们已完成11。

训练模型、数据和检索索引保持不变；增加检索比例不保证音色相似度或自然度单调提高。当前正式检查入口见 [交付指南](21_delivery.md)。
