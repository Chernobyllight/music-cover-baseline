# 零样本与微调三候选：命令和结果路径

只保留两个方案：零样本 `zero_shot`、目标音色通用微调 `finetuned`。单曲伪标签训练入口及实现已删除。以前生成的音频与导出模型不会因升级代码被删除，但单曲专用 schema-2 profile 不再被推理接口接受。

## 1. 先准备 00–04 的共同输入

本例为孙燕姿参考音色＋《秋意浓》＋upbeat bossa nova and samba jazz。新建运行：

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
bash scripts/run_baseline.sh \
  --config configs/tasks/qiu_yi_nong_bossa_nova_sun_yanzi.yaml \
  --run runs/lhd_candidates_example --until 04_vocal_separation
```

共同输入是本次运行的：

- `stages/00_prepare/attempt_001/artifacts/reference.wav`：截取的目标音色参考。
- `stages/04_vocal_separation/attempt_001/artifacts/vocals.wav`：03 新曲经 04 分离出的源人声。
- `stages/04_vocal_separation/attempt_001/artifacts/instrumental.wav`：所有候选共用的伴奏。

以上文件均在 `/data/liuhongda/music-cover-baseline/runs/lhd_candidates_example/` 下。实际成功 attempt 由 `stage_state.json` 决定，重跑后可能不是 001。02 决定旋律移调；05 保持 F0，不再额外移调。

## 2. 选择一个音色转换入口

零样本：原始通用权重，生成一个版本；无需音色训练模型或 05 候选评估依赖。

```bash
bash scripts/05_voice_conversion/zero_shot/run.sh \
  --run runs/lhd_candidates_example
```

微调：使用已导出的目标音色模型，默认生成三个不同随机种子的版本，不重新训练。

```bash
bash scripts/05_voice_conversion/finetuned/run.sh \
  --run runs/lhd_candidates_example \
  --voice-profile models/voices/sun_yanzi/seed_vc/smoke_v001/voice.json \
  --voice-candidates 3 --voice-candidate-seeds 42 43 44
```

两个入口选一个执行；在同一个运行切换会创建新的 05 attempt，不覆盖旧 attempt，后续交付要继续重跑。模型训练/导出仍使用 [音色微调手册](13_voice_adaptation.md) 的独立 `finetuned/training/` 脚本，训练出的音色模型可用于其它歌曲。`smoke_v001` 是现有短素材集成验证模型，不代表正式音色质量。

依赖：finetuned 的三个候选需要已配置的 Seed-VC 环境、ECAPA 和 RMVPE；见 [权重指南](12_model_weights.md)。只推理不需要 OpenVoice 训练资产。候选数可设 1–8；显式种子数必须一致且不重复。未指定种子时从主 seed 顺次递增，超出 uint32 时循环。

05 首次输出目录：

`/data/liuhongda/music-cover-baseline/runs/lhd_candidates_example/stages/05_voice_conversion/attempt_001/artifacts/`

| 文件 | 内容 |
| --- | --- |
| `candidates/candidate_001/converted.wav` | seed 42 的完整转换干声 |
| `candidates/candidate_002/converted.wav` | seed 43 的完整转换干声 |
| `candidates/candidate_003/converted.wav` | seed 44 的完整转换干声 |
| `candidates/candidate_001/diagnostics/metrics.json` | 此候选 ECAPA、对齐、F0、时长、峰值、RMS 等 |
| `candidates.json`、`candidates.csv`、`candidates.html` | 三个候选的指标、合格状态与推荐；HTML 可试听 |
| `converted.wav`、`conversion.json` | 推荐干声及其来源，供 06 继续使用 |

## 3. 混音、分轨、MIDI、评价和交付

```bash
# 06 把三个版本分别与同一个伴奏混音。
bash scripts/06_mix/run.sh --run runs/lhd_candidates_example
# 07–11 以推荐版本继续，最终包同时保留三个混音版本。
bash scripts/run_baseline.sh --run runs/lhd_candidates_example --from 07_stem_separation
```

06 结果在 `/data/liuhongda/music-cover-baseline/runs/lhd_candidates_example/stages/06_mix/attempt_001/artifacts/`：

- `candidates/candidate_001/final.mp3`、`candidate_002/final.mp3`、`candidate_003/final.mp3`：三个完整混音，同时有对应 WAV 和 `mix.json`。
- `candidates.html`：同伴奏试听和 05 干声指标对照。
- 根目录 `final.wav` / `final.mp3`：推荐候选的精确副本；后续分轨和 MIDI 只处理这一版。

10 的 `evaluation.json` 增加 `voice_candidates`，即使 `voice_check: false` 也保留 05 已计算的全部候选指标。若开启 `voice_check`，另评估推荐人声经 06 对齐后的结果，音频处理不同，分数可能略有差异。

最终入口为 `/data/liuhongda/music-cover-baseline/runs/lhd_candidates_example/deliverables/`：

| 路径 | 用途 |
| --- | --- |
| `listen.html` | 原曲、生成曲、最终曲与候选页入口 |
| `candidates.html`、`candidates.json` | 全部候选的相似度、技术检查、推荐和试听 |
| `candidates/candidate_001/final.mp3` | 第 1 个完整候选；002、003 同理，各有 WAV |
| `final.mp3` / `final.wav` | 推荐版本，可直接交给用户 |
| `review/index.html`、`review/midi_bundle.zip` | 推荐版本的分轨、MIDI 和专业评审包 |
| `evaluation.json` | 整曲检查及所有候选指标 |

首次部署的实际验证使用 `runs/lhd_voice_candidates_validation`，具体值另记验证报告。

## 4. 推荐规则与失败处理

三个候选仅改变推理随机种子；源人声、目标参考、微调模型、steps 和 F0 设置完全一致。先检查：ECAPA 有限、音高状态 verified、时间对齐 consistent、时长差≤0.25秒、峰值≤1.001、RMS≥0.0001。合格者按 ECAPA 降序选择；完全同分按较小 seed，再按候选 ID 排序。

候选失败会显示失败原因，不冒充成功。没有合格者时 05 停止并保存诊断，不发布推荐结果；可检查日志/参考数据后显式重跑。非推荐候选若无法混音，也会记录 `mix_status: failed`，不会阻止已合格推荐版本的交付。

ECAPA 测的是 05 转换后干声与同一目标参考的余弦相似度，不是整曲混音，也不是身份百分比。参考与推理提示相同，属于运行诊断；不能据此声称独立测试集泛化。混音使用同一伴奏和算法，各候选独立按原人声 RMS 配平并限峰，具体增益写入 `mix.json`；人工仍需判断歌词、发音、自然度和音乐性。
