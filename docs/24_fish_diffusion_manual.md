# Fish Diffusion 手动验证：单依纯＋绿洲

按脚本逐步创建训练配置、数据、特征、模型和推理配置，再完成 05、06。仅复用公共底模、原始素材和已成功的歌曲 00–04，不复用历史训练 YAML、切片或人物模型。训练与推理均使用 `/opt/conda/bin/python`。

## 示例名称、已有输入与输出位置

下面示例版本统一为 **manual_v001**，与已经完成的 `manual_20261105` 分开；若示例输出已存在，请整体改用新的版本/目录。不要先 mkdir 创建 dataset-dir/export-dir；脚本会拒绝覆盖已有目标。

本服务器根目录为 `/data/liuhongda/music-cover-baseline`，实际解析到 `/mnt/afs_f/liuhongda/music-cover-baseline`。以下从该根目录运行；各脚本的相对参数路径也按根目录解析，不依赖 VC_*。新机器先按 [Skill](../skill/SKILL.md)部署并完成自己的 00–04，将歌手名、素材编号、窗口和 source-run 替换为真实输入；本服务器录音/模型不随 Git 提供。

| 内容 | 项目内路径 | 是否必须事先存在 |
|---|---|---|
| 原始录音 | `data/raw/references/shan_yi_chun/sources/*/original.m4a` | 是，当前 30 份 |
| 原歌曲运行 | `runs/lhd_lv_zhou_shan_yi_chun_modular_test` | 是，00–04 全部成功 |
| 本次歌曲运行 | `runs/lhd_lv_zhou_shan_yi_chun_fish_manual_v001` | 第 2 步创建；引用原 00–04，必须保留原 run |
| recordings-dir | `data/voice_training/shan_yi_chun/fish_manual_v001/recordings` | 第 3 步创建；第 4 步前须成功 |
| 训练 YAML/output | `configs/voice_training/fish_diffusion/shan_yi_chun_manual_v001.yaml` | 第 4 步创建；第 5 步前必须存在 |
| dataset-dir | `data/voice_datasets/shan_yi_chun/fish_diffusion/manual_v001` | 第 5 步切片，第 6 步生成 features |
| 训练 run | `runs/voice_training/shan_yi_chun_fish_manual_v001` | 第 5 步建立状态，随后存日志和检查点 |
| export-dir | `models/voices/shan_yi_chun/fish_diffusion/manual_v001` | 第 9 步创建人物模型包 |
| 推理 YAML | `configs/voice_inference/fish_diffusion/shan_yi_chun_manual_v001.yaml` | 第 10 步创建 |

`source-run` 是提供源人声/伴奏和评估参考的**歌曲 run**，不是训练 run。04 源人声仅用于验证转换，不会加入目标人物训练集。所需组件、公共权重路径和下载链接见 [部署说明](22_fish_diffusion.md#环境与公共权重部署)；本手册没有部署其他模型的额外训练依赖。

## 1. base 与模型预检

输入：已部署的 Fish、ContentVec、NSF-HiFiGAN、ECAPA/RMVPE。原始录音分离另需 BS-RoFormer 和 Karaoke 模型及 YAML。

```bash
cd /data/liuhongda/music-cover-baseline
source /opt/conda/etc/profile.d/conda.sh
conda activate base
export PATH="/opt/conda/bin:$PATH"
bash scripts/05_voice_conversion/fish_diffusion/preflight/run.sh \
  --scope training --model-load --gpu 0
```

预期：检查与模型加载通过，python 为 `/opt/conda/bin/python`；不训练人物模型。缺依赖/权重先按部署说明补齐，再继续。

## 2. 建立独立歌曲 run

```bash
bash scripts/05_voice_conversion/fork_run.sh \
  --source-run runs/lhd_lv_zhou_shan_yi_chun_modular_test \
  --run runs/lhd_lv_zhou_shan_yi_chun_fish_manual_v001
```

输入：原 run 的成功 00–04。输出：新 run 的 request.json、config.resolved.json、stage_state.json、shared_upstream.json。不会重新生成音乐；00–04 文件仍在原 run，不能删掉源目录。

## 3. 重新分离目标人物录音

```bash
bash scripts/voice_data/separate/run.sh \
  --input-dir data/raw/references/shan_yi_chun/sources \
  --pattern '*/original.m4a' \
  --output-dir data/voice_training/shan_yi_chun/fish_manual_v001/recordings \
  --exclude-ids 1787247852 6779157198 --gpu 0
```

输入：原始录音；BS-RoFormer→Karaoke 两级分离。`1787247852` 是单独参考《续写》，`6779157198` 是此前未筛出合格训练窗口的《你的珍藏》。这些编号仅适用于本数据，不是程序写死规则。

预期：28 份成功、2 份排除，recordings/manifest.json 的 status=succeeded。每个编号下有 lead.wav、vocals.wav、instrumental.wav、backing.wav、separation.json；训练使用 **lead.wav**。试听检查主唱及分离残留。中断后原命令加 `--resume`，改变素材集合改用新输出目录。当前原素材是有损试听，不是录音棚干声。

## 4. 创建本次训练 YAML

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

**输入**是已有 recordings-dir 和 source-run；**本步输出只有 YAML**。dataset-dir/export-dir 仅写进配置，由后续阶段创建。voice-id 是人物标识，version 是用户命名的版本；省略三个输出参数时会按这两个标识生成上述默认路径。

终端必须出现 `Created .../shan_yi_chun_manual_v001.yaml` 才进入第 5 步。`--plan` 只预览、不创建；`--config` 也不会替你生成 YAML。

验证/测试各留两份整录音，剩余 24 份用于训练。36:12、156:12 是 04 人声第 36/156 秒起各 12 秒，换歌时按实际演唱选择，不能超出时长。segment=8 秒，训练 crop=384 帧约 4.46 秒；probability=1.5 是训练增强次数规则，不是推理移调。5000 步完成预热和衰减，是当前质量方案示例，不保证新训练与旧实验逐位相同。

## 5. Fish 训练 00_prepare：生成切片

```bash
bash scripts/05_voice_conversion/fish_diffusion/training/00_prepare/run.sh \
  --config configs/voice_training/fish_diffusion/shan_yi_chun_manual_v001.yaml \
  --run runs/voice_training/shan_yi_chun_fish_manual_v001
```

读取第 4 步 YAML 和第 3 步 lead.wav。输出 dataset-dir 的 manifest.json、train/validation/test WAV：44.1 kHz、单声道、FLOAT、最长 8 秒，少于 1 秒的尾段舍弃。训练 run 的 `stages/00_prepare/attempt_001/artifacts/` 有 manifest.json 和 dataset.json。这不是主流程的 00，不调用主流程准备。

若报 YAML 的 FileNotFoundError，先补第 4 步，不重做已成功分离。服务器 `/data/...` 解析为 `/mnt/afs_f/...` 本身不表示路径错误；先看缺失的是哪个文件。

## 6. Fish 训练 01_features：内容、F0、Mel

```bash
bash scripts/05_voice_conversion/fish_diffusion/training/01_features/run.sh \
  --config configs/voice_training/fish_diffusion/shan_yi_chun_manual_v001.yaml \
  --run runs/voice_training/shan_yi_chun_fish_manual_v001
```

读取第 5 步切片，使用 ContentVec、Parselmouth 和 NSF-HiFiGAN 配套 Mel 处理。输出 dataset-dir/features 的 npz/json，以及训练 run 的 `stages/01_features/attempt_001/artifacts/features.json`。仅 train 增强，验证/测试不增强。

## 7. Fish 训练 02_train：5000 步

```bash
bash scripts/05_voice_conversion/fish_diffusion/training/02_train/run.sh \
  --config configs/voice_training/fish_diffusion/shan_yi_chun_manual_v001.yaml \
  --run runs/voice_training/shan_yi_chun_fish_manual_v001
```

读取特征和公共底模，更新声学主干与新歌手 embedding，冻结内容编码器和声码器。输出训练 run 的 checkpoints/step_00001000.ckpt 至 step_00005000.ckpt；当前训练 attempt 的 artifacts/training.json、losses.csv，attempt 下的 command.json、process.log。检查 runtime.python 为 `/opt/conda/bin/python`。中断后同命令加 `--resume`；数据/身份改变改用新 run。

## 8. Fish 训练 03_validate：验证指定 5000 步

```bash
bash scripts/05_voice_conversion/fish_diffusion/training/03_validate/run.sh \
  --config configs/voice_training/fish_diffusion/shan_yi_chun_manual_v001.yaml \
  --run runs/voice_training/shan_yi_chun_fish_manual_v001 \
  --checkpoint runs/voice_training/shan_yi_chun_fish_manual_v001/checkpoints/step_00005000.ckpt
```

输入：新检查点、留出验证录音、两个歌曲窗口。输出训练 run 的 `stages/03_validate/attempt_001/artifacts/validation.json`、metrics.csv、listening.html，以及 held_out_1、held_out_2、song_window_01、song_window_02 子目录的 converted.wav。

当前是两个留出录音探针和两个歌曲窗口，不是全部 test 切片评估。核对 ECAPA、音高保持、evaluation_status，再试听音色/发音/电音感。评估失败会留下原因与音频；技术通过不等于听感通过。

## 9. Fish 训练 04_export：导出人物包

```bash
bash scripts/05_voice_conversion/fish_diffusion/training/04_export/run.sh \
  --config configs/voice_training/fish_diffusion/shan_yi_chun_manual_v001.yaml \
  --run runs/voice_training/shan_yi_chun_fish_manual_v001
```

读取匹配的 validation 报告和权重，输出 export-dir 的 voice.json、model.ckpt、config.json、validation.json。检查 voice_id=shan_yi_chun、backend=fish_diffusion、selected_step=5000、reference_required=false。完整目录一起保留；质量未通过时仍导出带待复核标记，已有版本不可覆盖。

## 10. 创建推理 YAML

```bash
bash scripts/05_voice_conversion/fish_diffusion/create_inference_config/run.sh \
  --voice-profile models/voices/shan_yi_chun/fish_diffusion/manual_v001/voice.json \
  --output configs/voice_inference/fish_diffusion/shan_yi_chun_manual_v001.yaml \
  --sampler-interval 10 --noise-predictor plms \
  --chunk-seconds 8 --overlap-seconds 0.2 --pad-seconds 0.5 --seed 42
```

输入：第 9 步真实导出包。输出上述新推理 YAML；不会生成模型。PLMS 间隔 10，8 秒分块、0.2 秒重叠、0.5 秒上下文；保持输入音高，不把主流程参考当推理条件。

## 11. 主流程 05：整曲换声

```bash
bash scripts/05_voice_conversion/fish_diffusion/run.sh \
  --run runs/lhd_lv_zhou_shan_yi_chun_fish_manual_v001 \
  --config configs/voice_inference/fish_diffusion/shan_yi_chun_manual_v001.yaml --gpu 0
```

输入：04 的 vocals.wav、新人物模型及推理配置。输出歌曲 run 的 `stages/05_voice_conversion/attempt_001/artifacts/converted.wav`、conversion.json、candidates.json/csv/html。converted.wav 是尚未混伴奏的完整转换人声；当前只有一个候选。参考仅用于评估，05 额外移调为 0。

## 12. 主流程 06：混音成品

```bash
bash scripts/06_mix/run.sh \
  --run runs/lhd_lv_zhou_shan_yi_chun_fish_manual_v001 \
  --config configs/voice_inference/fish_diffusion/shan_yi_chun_manual_v001.yaml --gpu 0
```

输入：05 converted.wav、04 instrumental.wav 和时间轴。无需新增模型。最终音频：

```text
/data/liuhongda/music-cover-baseline/runs/lhd_lv_zhou_shan_yi_chun_fish_manual_v001/stages/06_mix/attempt_001/artifacts/final.wav
/data/liuhongda/music-cover-baseline/runs/lhd_lv_zhou_shan_yi_chun_fish_manual_v001/stages/06_mix/attempt_001/artifacts/final.mp3
```

同目录 mix.json 记录混音；只到 06 不会创建顶层 deliverables（主流程 11 的职责）。完成这里已经得到最终歌曲，不需要下一步才能使用。

## 13. 可选：整曲定量对比

```bash
bash scripts/05_voice_conversion/fish_diffusion/validate/run.sh \
  --run runs/lhd_lv_zhou_shan_yi_chun_fish_manual_v001 \
  --config configs/voice_inference/fish_diffusion/shan_yi_chun_manual_v001.yaml \
  --compare-run runs/lhd_lv_zhou_shan_yi_chun_modular_test --gpu 0
```

对照 run 必须已经完成 05/06，且 04 人声和伴奏与本次相同。本服务器对照为已有 Seed-VC 零样本；其他用户换自己的结果，或省略 compare-run 只评估当前结果。

输出歌曲 run 的 fish_validation/comparison.json、index.html。入口会检查并调用 05/06，参数与产物有效时复用缓存；不要求重算就不加 --force。目标已经在第 2 步创建，因此此处不传 --source-run。跳过第 13 步不影响模型、05 人声、06 WAV/MP3，也保留第 8/11 步已有指标。

同参考对比不是独立总体模型排名：Seed-VC 也以该参考作推理提示，Fish 不使用它。技术指标不代表人工自然度评价。

## 已完成的手动验证版本（2026-11-05）

实际版本是 `manual_20261105`，不是上述新示例 `manual_v001`。用户已经从原始录音完成训练与 05→06，随后只读核对记录：

| 项目 | 实际结果 |
|---|---|
| 训练数据 | 96 段、719.273 秒、24 个录音组 |
| validation/test | 各 8 段、59.951 秒、2 个录音组 |
| 训练/推理 Python | `/opt/conda/bin/python` |
| 检查点 | `runs/voice_training/shan_yi_chun_fish_manual_20261105/checkpoints/step_00005000.ckpt` |
| 导出档案 | `models/voices/shan_yi_chun/fish_diffusion/manual_20261105/voice.json` |
| 歌曲 run | `runs/lhd_lv_zhou_shan_yi_chun_fish_manual_20261105` |
| 05 同参考 ECAPA | 0.7619887590 |
| 对齐后音高误差 ≤1 半音比例 | 98.1959% |
| 混音时长 | 257.3986875 秒，48 kHz 双声道 WAV/MP3 |

实际最终文件：`/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_lv_zhou_shan_yi_chun_fish_manual_20261105/stages/06_mix/attempt_001/artifacts/final.mp3`；SHA256 `5e199ae0943e827dd4df7bac2cafe9db6690f7600ef48c1dc12451332a496699`。上述指标来自 05 的 candidates/candidate_001/diagnostics/metrics.json，不冒称已运行可选第 13 步或全部独立测试集。

训练数据与上一轮优化实验（95 段）不同，这次重新分离/切片得到 96 段，不能混用两轮计数和分数。以后重跑以训练 state.json、歌曲 stage_state.json 指向的成功 attempt 为准。数据和结果是服务器资产，不是 Git 自带文件；本文记录技术完成，未宣称人工听感已通过。
