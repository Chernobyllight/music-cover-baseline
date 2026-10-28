# 音色微调接入：实际验证记录

历史验证记录：数值和路径描述当日实验；后续按用户要求清理过运行数据，不能假定下列目录全部仍存在。当前保留结果和可复制命令见 [v0.3 实例](17_v03_walkthrough.md)。

验证日期：2026-10-26。服务器：`10.210.8.11:7830`。项目目录：`/mnt/afs_f/liuhongda/music-cover-baseline`。

这次验证使用孙燕姿现有 16 秒参考片段、秋意浓、upbeat bossa nova / samba jazz。目的是验证训练、续训、对照评估、导出和歌曲流水线接入；16 秒素材不足以证明对目标歌手音色的泛化能力。

## 数据与运行位置

| 内容 | 路径（相对于项目根目录） |
| --- | --- |
| 训练配置 | `configs/voice_training/sun_yanzi_seed_vc_smoke.yaml` |
| 数据清单与两个 8 秒训练切片 | `data/voice_datasets/sun_yanzi/smoke_v001/` |
| 训练与对照验证运行 | `runs/voice_training/sun_yanzi_seed_vc_smoke_001/` |
| 00–04 原始歌曲产物来源 | `runs/lhd_test_modular/` |
| 独立整曲验证运行 | `runs/lhd_voice_finetune_validation/` |

整曲验证复用了原运行 00–04 的真实状态和音频；控制器重新核验代码、参数、运行依赖、产物哈希和成功清单。新 05–11 产物写到独立目录。原运行的状态与交付文件保持不动。`reused_upstream.json` 记录了来源。

旧运行使用 `/data/liuhongda/music-cover-baseline` 入口，整曲验证沿用该别名以匹配旧版缓存中的模块路径。它和 `/mnt/afs_f/...` 指向同一共享目录。

## 已验证的训练和兼容性

- 64 项自动化测试通过，包括原有功能、双模式默认行为、DDP 参数名兼容、数据划分泄漏、权重/验证清单篡改、不可覆盖导出及 transpose=0 时独立音色评估的配置逻辑。
- 真实 GPU 训练 0→20 步，再从 checkpoint 恢复到 40 步；恢复网络、AdamW 状态、学习率调度器、迭代步和随机数状态。
- CFM 有 186,337,664 个可训练参数，length regulator 有 10,039,296 个。两次训练均检测到 218 个 CFM 参数张量和 21 个 length regulator 参数张量的抽样值变化。
- Whisper、CAMPPlus、RMVPE、OpenVoiceV2、BigVGAN 冻结。通用 Seed-VC 权重没有被覆盖。
- 第 20 步 loss=0.528432，第 40 步 loss=0.500807。它们是随机训练批次的单步数值，不代表单调下降或主观音质改善；完整记录见 `loss.jsonl`。
- 新零样本脚本完成了整曲转换；随后旧 `scripts/05_voice_conversion/run.sh` 成功校验并复用该结果。
- 第一次加载训练权重时，严格校验发现 DDP 的 `module.` 前缀差异并拒绝继续；修复后完整覆盖全部主模型参数。失败 attempt 保留，不能当作成功训练记录。

训练成功记录在：

```text
runs/voice_training/sun_yanzi_seed_vc_smoke_001/stages/01_train/attempt_002/artifacts/training.json
runs/voice_training/sun_yanzi_seed_vc_smoke_001/stages/01_train/attempt_003/artifacts/training.json
```

本次使用 PyTorch 2.8.0+cu128、Torchaudio 2.8.0+cu128、Transformers 4.46.3；独立评估添加 SpeechBrain 1.0.3、HyperPyYAML 1.2.2 及固定 ruamel 版本，未升级原有 PyTorch/Transformers。实际版本记录在整曲验证运行的 `voice_environment.json`。

## 同输入 A/B 对照结果

固定 steps=30、seed=42，同一源片段、目标参考和音高设置。参考和训练片段来自同一16秒录音，因此没有独立音色测试集。

| 源歌曲窗口 | 通用模型 | 20步模型 | 40步模型 | 40步增量 | 音高保持 |
| --- | ---: | ---: | ---: | ---: | --- |
| 40–52 秒 | 0.569456 | 0.564787 | 0.590239 | +0.020783 | verified |
| 120–132 秒 | 0.627122 | 0.623209 | 0.639423 | +0.012301 | verified |

20步平均增量 -0.004291；40步平均增量 +0.016542，验证选择 `step_000040.pth`。这只是该提示与两个片段上的 ECAPA 诊断值，不能当作音色相似百分比或主观质量结论。

验证包：

```text
runs/voice_training/sun_yanzi_seed_vc_smoke_001/stages/02_validate/attempt_001/artifacts/
├── validation.json
├── metrics.csv
├── listening.html
├── listening/001…004/        # 4组随机 A/B，含干声、相同伴奏混音、源与参考
├── listening_key.json       # 听完再看身份表
└── review_scores.csv
```

## 导出音色版本

已导出 `models/voices/sun_yanzi/seed_vc/smoke_v001/`：

- `model.pth`：785,729,855 字节，只包含网络参数，不携带优化器。
- 模型 SHA256：`fab72a3549b05ef42b78f7f363cf754fc80ea95fd325271e4f602ed295153aa2`。
- `voice.json` 绑定模型、配置与验证报告哈希，并明确记录 `integration_only`、`reference_required: true` 和待人工评审状态。
- `config.yml` 与 `validation.json` 一同保存，通用模型仍保留原文件。

可通过 `scripts/05_voice_conversion/finetuned/run.sh --voice-profile models/voices/sun_yanzi/seed_vc/smoke_v001/voice.json --run runs/lhd_voice_finetune_validation` 读取这套模型。详见 [使用手册](13_voice_adaptation.md)。

## 整曲转换接入

`runs/lhd_voice_finetune_validation/stages/05_voice_conversion/` 中保留了两次真实整曲转换：

| attempt | 模式 | 使用的模型 | 用时 |
| --- | --- | --- | ---: |
| `attempt_001` | zero_shot | 原有通用权重 | 78.1 秒 |
| `attempt_002` | finetuned | 已导出的 `sun_yanzi/smoke_v001` | 86.5 秒 |

两次 `conversion.json` 都记录 `training_performed: false`、steps=30、seed=42、半音偏移=0。微调版本的转换记录绑定上述导出模型哈希。原有零样本兼容入口复用检查也成功。

混音输出为 48 kHz 双声道、397.638667 秒。转换人声相对原人声时长差为 -0.0085625 秒；跨窗口对齐一致，施加额外延迟为 0。WAV 工程真峰值估计 -1.10 dBTP，MP3 为 libmp3lame 192 kbps。

- 成品 WAV SHA256：`13aa523b77d6b8282fd6e9aa79820df9eadbdfa66f88ec8c1ac2987f80871a8f`。
- 成品 MP3 SHA256：`4c3c8010e3e33c6bc4791ba6336a7fb056430888fb9a7fe4eb17b0dba7ed21de`。

整曲分轨（07）已实际重新执行，用时 185.2 秒。输入绑定本次微调成品的 WAV 哈希，各分轨统一保留 48 kHz、397.638667 秒和同一时间原点。结果清单位于 `runs/lhd_voice_finetune_validation/stages/07_stem_separation/attempt_001/artifacts/stems.json`；分轨文件的存在和可解码不代表分离质量已经通过人工评审。

## 总谱、MIDI 与整曲评估

08 总谱重新转录完成（63.7 秒），09 各轨 MIDI 重新生成完成（88.3 秒），10 评估完成（105.9 秒）。主唱、和声、总人声、贝斯、吉他、钢琴、其他乐器和鼓的输出以 `stages/09_stem_midi/attempt_001/artifacts/midi.json` 实际清单为准；这些是估计转录结果，仍需专业人士审阅。

新增整曲音色评估读取真实的 04 源人声、06 已对齐转换人声和00参考，实际加载了 `finetuned` 模式记录：

- ECAPA 相似度：0.667989（同一推理参考，仅为诊断）。这与上面的12秒片段不是同一评估样本，不能直接相减当作增益。
- F0 检查：`verified`，975个配对时间段，源有效音高覆盖率约98.48%，中位音高差约+0.0117半音。
- 原有移调后旋律检查：`pitch_adherence.status = verified`。
- `subjective_quality` 仍待人工评审；自动检查通过不表示音色自然度、歌词或乐器转录准确度已经达标。

查看：

```text
runs/lhd_voice_finetune_validation/stages/10_evaluation/attempt_001/artifacts/evaluation.json
runs/lhd_voice_finetune_validation/stages/10_evaluation/attempt_001/artifacts/voice_conversion_evaluation.json
runs/lhd_voice_finetune_validation/stages/10_evaluation/attempt_001/artifacts/voice_diagnostics/
```

本次实际 MIDI 清单有10项：`lead_vocals`、`backing_vocals`、`vocals`、`bass`、`guitar`、`piano`、`other`、`drums_onsets`、`arrangement_estimate`、`harmony`。其中鼓是起音事件估计，和声与合奏同样需要人工检查，不能当作精确原始编曲。

## 最终交付校验

`stages/11_delivery/attempt_001/artifacts/delivery_validation.json` 的真实校验结果：

- `technical_validation: passed`，检查174个载荷文件。
- 12条音频轨，其中10条为叶子轨；轨道重建最大绝对误差约 `5.57e-8`。
- 10项 MIDI 均非空，`empty_midi_tracks: []`。
- `subjective_quality: awaiting_human_review`，仍需听音和专业评审。

最终交付入口：

```text
/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_voice_finetune_validation/deliverables/final.wav
/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_voice_finetune_validation/deliverables/final.mp3
/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_voice_finetune_validation/deliverables/listen.html
/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_voice_finetune_validation/deliverables/review/index.html
/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_voice_finetune_validation/reviews/review_scores.csv
```

工程流程通过不等于完成正式音色训练。正式实验需要增加独立目标歌手录音，按录音分组划分训练、验证、测试和提示集合，按相同流程重新训练，再做盲听及歌词/旋律检查。

最终 `stage_state.json` 为 `status: completed`，00–11全为 succeeded；其中00–04复用原始成功产物，05–11实际重新执行。11交付阶段用时202.9秒。
