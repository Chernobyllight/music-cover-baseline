# so-vits-svc 4.1：歌手微调与转换

工程根目录：`/data/liuhongda/music-cover-baseline`（本机解析到 `/mnt/afs_f/liuhongda/music-cover-baseline`）。

新增 `backend=so_vits_svc, mode=finetuned`；原来的 Seed-VC zero-shot / finetuned 入口继续保留。SVC 不支持参考音频零样本转换。SVC 训练不使用源歌曲伪标签，只使用目标歌手的训练录音；模型可用于多首歌曲。

## 环境与权重

本次按用户要求使用已有 `(base)`：`/opt/conda/bin/python`。保留 Python 3.11、PyTorch 2.8、CUDA 12.8、NumPy 2.2.6、Transformers 4.57.3；不新建 SVC Conda 环境。新增依赖见 `configs/requirements-svc-base.txt`，实际依赖快照见 `configs/svc-base.installed.txt`。指标计算继续使用原有 `envs/seed-vc`，其它模块继续使用原有环境。

ContentVec 通过现有 Transformers 的 HuBERT 实现加载同一份权重，逐项检查 211 个推理参数；仅丢弃训练用的标签嵌入和最终投影。无需安装旧 fairseq/hydra，也无需降级 base。适配代码位于 `src/music_cover/backends/svc_contentvec.py`。

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
git submodule update --init third_party/so-vits-svc
# 新机器先检查已有核心版本，再补装这几个依赖；不要覆盖当前其他环境。
/opt/conda/bin/python -m pip install -r configs/requirements-svc-base.txt
/opt/conda/bin/python scripts/setup_svc.py --download --training
```

| 权重 | 工程位置 | 来源 / 用途 |
|---|---|---|
| ContentVec | `models/SoVitsSVC/contentvec.pt` | `lj1995/VoiceConversionWebUI`，768 维第 12 层内容特征 |
| SVC G/D 底模 | `models/SoVitsSVC/G_0.pth`、`D_0.pth` | `Sucial/so-vits-svc4.1-pretrain_model/vec768l12`，训练初始化；该底模页面标注 CC-BY-NC-SA-4.0 |
| RMVPE | `models/SoVitsSVC/rmvpe.pt` | `yxlllc/RMVPE` 的 `230917/rmvpe.zip` 中 `model.pt`；不能直接替换成现有 Seed-VC 同名权重 |
| 训练后的歌手 G | `models/voices/yang_kun/so_vits_svc/v001_20261028/model.pth` | 本工程训练并验证后导出 |

下载 URL、revision、大小和 SHA256 统一记录于 `configs/so_vits_svc_assets.lock.json`。上游固定提交 `730930d337d171479eadf305f96cbed4bb393e77`，保持源码不修改。底模仅允许重新初始化 `emb_g.weight`；所有其他参数必须完整匹配，加载失败不会偷偷改成随机训练。

基础接入使用标准 VITS + NSF-HiFiGAN，不启用浅层扩散、检索、增强器或音色融合，不需要这些可选模块的权重。

> 2026-10-29 已清理旧训练 run；下节是原 v001 的训练流程示例，已导出 v001 模型仍保留。新训练请为配置中的 version、dataset_dir、export_dir 和 --run 使用新名称。当前推荐的 45 段数据及 4000+2000 步配方见 [优化指南](19_svc_optimization.md)。

## 训练：四个独立阶段

示例配置：`/data/liuhongda/music-cover-baseline/configs/voice_training/so_vits_svc/yang_kun.yaml`。

输入是已有的杨坤数据集：

- `/data/liuhongda/music-cover-baseline/data/raw/references/yang_kun/dataset/manifest.json`
- train：41 段 / 328 秒；validation：6 段 / 48 秒；test：5 段 / 40 秒。
- 沿用歌曲组划分，prompt 不参与训练。数据来自分离人声，仍需人工听审，不能将自动技术检查当成录音棚干声质量认证。

以下命令依次执行。训练与转换均不会隐式启动另一个流程。

```bash
cd /data/liuhongda/music-cover-baseline

bash scripts/05_voice_conversion/so_vits_svc/training/00_prepare/run.sh \
  --config configs/voice_training/so_vits_svc/yang_kun.yaml \
  --run runs/voice_training/yang_kun_so_vits_svc_v001_20261028

bash scripts/05_voice_conversion/so_vits_svc/training/01_train/run.sh \
  --config configs/voice_training/so_vits_svc/yang_kun.yaml \
  --run runs/voice_training/yang_kun_so_vits_svc_v001_20261028

bash scripts/05_voice_conversion/so_vits_svc/training/02_validate/run.sh \
  --config configs/voice_training/so_vits_svc/yang_kun.yaml \
  --run runs/voice_training/yang_kun_so_vits_svc_v001_20261028 \
  --checkpoint runs/voice_training/yang_kun_so_vits_svc_v001_20261028/checkpoints/step_00001000.pth

bash scripts/05_voice_conversion/so_vits_svc/training/03_export/run.sh \
  --config configs/voice_training/so_vits_svc/yang_kun.yaml \
  --run runs/voice_training/yang_kun_so_vits_svc_v001_20261028
```

| 阶段 | 使用上一步什么数据 | 中间结果 |
|---|---|---|
| 00_prepare | 原始 dataset manifest 和各 split WAV | `data/voice_datasets/yang_kun/so_vits_svc/v001_20261028/`：PCM16、训练特征、F0/UV、频谱、filelists、带哈希 manifest |
| 01_train | 00 的 train.txt、特征和 model_config.json | `runs/voice_training/yang_kun_so_vits_svc_v001_20261028/checkpoints/step_*.pth`：G/D、优化器、随机状态、数据游标 |
| 02_validate | 01 检查点；固定 04 源人声窗口；独立 validation 参考 | `stages/02_validate/attempt_*/artifacts/validation.json`、各窗口 WAV、ECAPA/F0/对齐指标 |
| 03_export | 02 选出的合格检查点 | `models/voices/yang_kun/so_vits_svc/v001_20261028/voice.json` 和推理权重 |

上表 stage 路径均位于训练 run 下。用 `state.json` 查每阶段最近一次成功/失败状态与实际输出位置。

`--max-steps` 表示累计训练步数。已有检查点必须显式 `--resume`，会恢复 G/D、优化器、随机状态、采样顺序及游标；修改训练参数、数据或训练实现必须新建 run。导出版本不可覆盖。

短训练与恢复示例（请用新 run / 新版本，不要覆盖上面的验证产物）：第一次加 `--max-steps 2`，第二次加 `--resume --max-steps 1000`。训练只使用指定 GPU，不自动占用所有 GPU。

02 可多次传 `--checkpoint`，在验证集上按技术合格后平均 ECAPA 选择；无合格候选则拒绝导出。测试集仅审计已经在 validation 上选好的单个检查点：在 02 命令追加 `--evaluation-split test`。先导出验证选中的模型再做测试审计；03 不接受以 test 报告重新选模。

## 转换：复用同一份 00–04，公平比较模型

可复用同一份 00–04 的已保留验证 run：

`/data/liuhongda/music-cover-baseline/runs/lhd_yang_kun_svc_optimized_validation_20261028`

创建 SVC 对照 run，不重新生成 03 的歌曲：

```bash
cd /data/liuhongda/music-cover-baseline
/opt/conda/bin/python scripts/fork_voice_run.py \
  --source-run runs/lhd_yang_kun_svc_optimized_validation_20261028 \
  --run runs/lhd_bu_bai_de_ying_xiong_celtic_epic_yang_kun_svc_20261028

bash scripts/05_voice_conversion/so_vits_svc/run.sh \
  --run runs/lhd_bu_bai_de_ying_xiong_celtic_epic_yang_kun_svc_20261028 \
  --voice-profile models/voices/yang_kun/so_vits_svc/v001_20261028/voice.json

bash scripts/run_baseline.sh \
  --run runs/lhd_bu_bai_de_ying_xiong_celtic_epic_yang_kun_svc_20261028 \
  --from 06_mix
```

`fork_voice_run.py` 校验并引用源 run 的 00–04 文件，不复制或改写源 run；继续运行时仍会由管线校验上游 fingerprint。需保留源 run，后续才能重跑依赖阶段。最终交付音频是独立文件。

新 run 中查看：

- `stages/05_voice_conversion/attempt_*/artifacts/converted.wav`：目标音色干声。
- 同目录 `conversion.json`：模型、原始输入哈希、切片与 F0 参数、时间和显存。
- 同目录 `candidates.json`：技术检查与 ECAPA，当前 SVC 一个候选。
- `stages/06_mix/attempt_*/artifacts/final.mp3`：最终混音。
- `stages/10_evaluation/attempt_*/artifacts/evaluation.json`：完整定量报告。
- `deliverables/final.mp3`：完成 11 后的最终交付入口。

通用入口等价参数为 `--voice-backend so_vits_svc --voice-mode finetuned --voice-candidates 1 --voice-profile ...`。首次跑完整歌曲仍需正常提供素材、歌词和风格或 task config。

## 推理约束和评估口径

- 02 负责移调，05 的 `semitones=0`、自动 F0 预测关闭，保留输入演唱音高。
- 参考音频用于音区分析、ECAPA，不是 SVC 推理条件；报告明确标注。
- SVC 不使用 Seed-VC 的 diffusion steps；记录为不适用。训练 max_steps 与推理 steps 是不同参数。
- 默认 20 秒片段、0.5 秒上下文与重叠淡入淡出；保留静音位置和总样本数；记录峰值限制增益。
- 随机种子显式传入核心 infer；分片 seed 为主 seed 加分片编号。首轮只开放一个 SVC 候选，Seed-VC 三候选功能保留。
- 先检查 F0、对齐、时长、削波、静音，再使用 ECAPA；不放宽现有技术检查阈值。
- ECAPA 不代表身份百分比或主观音质。新增训练只验证这条技术路线；泛化、气息、沙哑质感和咬字需要更多歌曲及人工试听。

## 杨坤优化版与独立实验

原v001仍保留。新增 `optimized_v002_20261028`，使用清理后的7.47分钟素材和4000+2000步训练，推荐以 `configs/voice_inference/so_vits_svc/yang_kun_optimized.yaml` 显式启用。标准适配器还支持严格匹配的vec256l9/256维架构，原vec768l12默认行为不变。实验性的HuBERT、Whisper、扩散等位于独立experiments目录，不会自动用于普通pipeline。具体证据、局限与完整命令见 [优化实验](19_svc_optimization.md)。
