# 使用说明（v1.0）

以下命令在已配置服务器执行。工程根目录为 `/data/liuhongda/music-cover-baseline`，与 `/mnt/afs_f/liuhongda/music-cover-baseline` 指向同一工程。新机器先按[环境说明](docs/03-environment.md)和[模型权重指南](docs/12_model_weights.md)恢复依赖。

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
```

## 输入、默认值和运行目录

| 输入 | 本例使用的文件 |
|---|---|
| 源歌曲 | `data/raw/songs/chuan_yue_shu_guang_tv.mp3` |
| 歌词 | `data/raw/lyrics/chuan_yue_shu_guang_tv.txt` |
| 参考人声 | `data/raw/references/huang_qi_shan/reference.wav` |
| 完整任务配置 | `configs/tasks/chuan_yue_shu_guang_huang_qi_shan_seedvc_20261029.yaml` |

本例配置为黄绮珊 Seed-VC finetuned、管弦灵魂乐、参考前 20 秒、auto 选调、3 个候选。通用入口的默认值是 Seed-VC zero_shot、原调 `transpose=0`、参考前 15 秒、30 个 Seed-VC 推理步、seed=42、完整评审开启。任务文件与命令行可覆盖这些默认值；不能把本例参数当成全局默认值。

参考录音须事先整理清晰，不会在 00 自动去伴奏。歌词使用 UTF-8 TXT/LRC，演唱顺序应与原曲一致。源素材不包含在 Git 中。

首次使用新的 `--run`；同名目录再次执行会按输入、参数、代码和产物哈希检查缓存。`--force` 只强制重跑选中的阶段；旧 attempt 保留。不要手动改写已经成功的 `artifacts/`。

## 一次执行完整流程

```bash
bash scripts/run_baseline.sh \
  --config configs/tasks/chuan_yue_shu_guang_huang_qi_shan_seedvc_20261029.yaml \
  --preflight
bash scripts/run_baseline.sh \
  --config configs/tasks/chuan_yue_shu_guang_huang_qi_shan_seedvc_20261029.yaml \
  --run runs/my_chuan_yue_shu_guang_seedvc --gpu 0
```

预检不创建歌曲运行，也不启动训练。`RESULTS (completed)` 表示已发布交付目录；`RESULTS (partial)` 表示只完成选定的前半段。

## 依次执行 00–11

下面每一条命令使用同一个运行目录；首次 00 提供配置，后续读取已保存的配置。

```bash
bash scripts/00_prepare/run.sh --run runs/my_chuan_yue_shu_guang_seedvc \
  --config configs/tasks/chuan_yue_shu_guang_huang_qi_shan_seedvc_20261029.yaml --gpu 0
bash scripts/01_source_score/run.sh --run runs/my_chuan_yue_shu_guang_seedvc
bash scripts/02_pitch_adaptation/run.sh --run runs/my_chuan_yue_shu_guang_seedvc
bash scripts/03_style_generation/run.sh --run runs/my_chuan_yue_shu_guang_seedvc
bash scripts/04_vocal_separation/run.sh --run runs/my_chuan_yue_shu_guang_seedvc
bash scripts/05_voice_conversion/finetuned/run.sh --run runs/my_chuan_yue_shu_guang_seedvc
bash scripts/06_mix/run.sh --run runs/my_chuan_yue_shu_guang_seedvc
bash scripts/07_stem_separation/run.sh --run runs/my_chuan_yue_shu_guang_seedvc
bash scripts/08_final_score/run.sh --run runs/my_chuan_yue_shu_guang_seedvc
bash scripts/09_stem_midi/run.sh --run runs/my_chuan_yue_shu_guang_seedvc
bash scripts/10_evaluation/run.sh --run runs/my_chuan_yue_shu_guang_seedvc
bash scripts/11_delivery/run.sh --run runs/my_chuan_yue_shu_guang_seedvc
```

下表的输出前缀统一为 `runs/my_chuan_yue_shu_guang_seedvc/stages/`，例如 `00_prepare/attempt_001/artifacts/source.wav`。`attempt_001` 是首次尝试示例；重跑后查看 `stage_state.json` 的实际路径。

| 阶段 | 读取什么 | 在本阶段 artifacts 查看什么 |
|---|---|---|
| 00_prepare | 原曲、歌词、参考录音 | `source.wav`、`reference.wav`、`lyrics.txt` |
| 01_source_score | 00 的 source | `score.abc`、`events.json`、MIDI |
| 02_pitch_adaptation | 01 的 score；00 的音频 | `transposition.json`、`score_original.abc`、`score_transposed.abc`；启用分析时另有音区报告与 `score_transposed.mid` |
| 03_style_generation | 02 的 score；00 的歌词；任务曲风 | `generated/audio.flac`、`generation_budget.json` |
| 04_vocal_separation | 03 的生成歌曲 | `vocals.wav`、`instrumental.wav` |
| 05_voice_conversion | 04 的人声；模型/参考输入 | `converted.wav`、`conversion.json`；微调另有 `candidates.json` |
| 06_mix | 03 时长；04 人声/伴奏；05 转换人声 | `final.wav`、`final.mp3`、`mix.json`、候选混音 |
| 07_stem_separation | 06 的混音组件 | 主唱、背景人声、乐器估计及 `stems.json` |
| 08_final_score | 06 的成品 | 成品 ABC、MIDI、节拍、和弦等 |
| 09_stem_midi | 07 的各轨；08 的和弦 | `midi/`、`notes/`、`midi.json` |
| 10_evaluation | 上游音频/分轨/MIDI及音高记录 | `evaluation.json`、可选音高/音色诊断 |
| 11_delivery | 全部选定产物和检查结果 | `listen.html`、`final.mp3`、`review/` 等；通过后发布为顶层 deliverables |

每个阶段同级的 `inputs.json`、`config.json`、`command.json`、`process.log`、`checks.json`、`outputs.json` 保存来源与执行证据。单阶段不自动补跑缺失或失效的上游。

## 只生成音频、续跑和重跑

```bash
# 只做到混音，直接试听 06 的 final.mp3
bash scripts/run_baseline.sh --run runs/my_chuan_yue_shu_guang_seedvc --until 06_mix
# 继续完成分轨、MIDI、评估和交付
bash scripts/run_baseline.sh --run runs/my_chuan_yue_shu_guang_seedvc --from 07_stem_separation
# 只重新转各轨 MIDI，再更新下游交付
bash scripts/09_stem_midi/run.sh --run runs/my_chuan_yue_shu_guang_seedvc --force
bash scripts/run_baseline.sh --run runs/my_chuan_yue_shu_guang_seedvc --from 10_evaluation
# 检查运行状态
bash scripts/run_baseline.sh --run runs/my_chuan_yue_shu_guang_seedvc --status
```

如果新任务不需要分轨和 MIDI，可在首次完整命令加 `--no-review`；仍执行 10–11，并交付 WAV/MP3。当前工程没有自动歌词检测、歌词门禁或静音修复。

## 四条音色入口

| 路线 | 独立入口（位于 scripts/05_voice_conversion） | 使用条件 |
|---|---|---|
| Seed-VC 零样本 | `zero_shot/run.sh` | 通用预训练权重 + 参考片段，1 次推理 |
| Seed-VC 微调 | `finetuned/run.sh` | 已导出 Seed-VC voice.json + 参考片段，默认 3 候选 |
| so-vits-svc | `so_vits_svc/run.sh` | 已导出 SVC voice.json，1 候选 |
| RVC | `rvc/run.sh` | 已导出 RVC voice.json 与训练集索引，默认 3 候选 |

SVC/RVC 推理不把参考人声作为音色条件；主流程仍需要参考文件供 02 音区分析和评估。训练是单独操作，不会由 05 隐式启动。

公平比较时先引用同一份已完成的 00–04。下面源运行在本服务器保留；克隆源码后需要自行生成上游。

```bash
/opt/conda/bin/python scripts/fork_voice_run.py \
  --source-run runs/lhd_chuan_yue_shu_guang_orchestral_soul_huang_qi_shan_seedvc_20261029 \
  --run runs/my_chuan_yue_shu_guang_rvc
bash scripts/05_voice_conversion/rvc/run.sh \
  --run runs/my_chuan_yue_shu_guang_rvc \
  --config configs/voice_inference/rvc/huang_qi_shan.yaml --gpu 0
bash scripts/run_baseline.sh --run runs/my_chuan_yue_shu_guang_rvc --from 06_mix --gpu 0
```

RVC 项目预设为 `index_rate=0.5, protect=0.33`。官方单次推理检索比例 0.75 另有显式预设 `configs/voice_inference/rvc/huang_qi_shan_index075.yaml`，可替换上面的 `--config`；不修改现有模型档案或全局默认值。详见 [RVC 指南](docs/20_rvc.md)。

Seed-VC 零样本的同编曲对照：

```bash
/opt/conda/bin/python scripts/fork_voice_run.py \
  --source-run runs/lhd_chuan_yue_shu_guang_orchestral_soul_huang_qi_shan_seedvc_20261029 \
  --run runs/my_chuan_yue_shu_guang_zero_shot
bash scripts/05_voice_conversion/zero_shot/run.sh --run runs/my_chuan_yue_shu_guang_zero_shot --gpu 0
bash scripts/run_baseline.sh --run runs/my_chuan_yue_shu_guang_zero_shot --from 06_mix --gpu 0
```

杨坤 SVC 优化版例子：

```bash
/opt/conda/bin/python scripts/fork_voice_run.py \
  --source-run runs/lhd_yang_kun_svc_optimized_validation_20261028 \
  --run runs/my_yang_kun_svc
bash scripts/05_voice_conversion/so_vits_svc/run.sh \
  --run runs/my_yang_kun_svc \
  --config configs/voice_inference/so_vits_svc/yang_kun_optimized.yaml --gpu 0
bash scripts/run_baseline.sh --run runs/my_yang_kun_svc --from 06_mix --gpu 0
```

fork 引用原来的文件并校验哈希，必须保留源 run。切换后端时自动清除旧后端的私有参数、旧 voice_profile 和候选种子，新的参数应通过配置显式提供。显式传入不兼容参数仍会报错。

## 训练和选调

训练路径与中间产物详见 [Seed-VC 手册](docs/13_voice_adaptation.md)、[SVC 手册](docs/18_so_vits_svc.md)、[RVC 手册](docs/20_rvc.md)。模型版本、训练 run 和派生数据目录分别命名，已有导出版本不可覆盖。训练步数与 Seed-VC 的推理 `--steps` 无关；SVC/RVC 不使用扩散推理步数。

`--transpose auto` 默认搜索 -6～+6 半音，只有 `reference_kind: singing` 等条件满足才自动采用非零建议；未知/讲话参考可能推荐非零但实际保持 0。`--transpose -2` 明确降两半音。02 调整乐谱，03 据此生成，05 保留新歌 F0，不重复移调。详见 [双向选调说明](docs/10_bidirectional_pitch.md)。

## 最终文件与验收

完整交付在 `runs/<运行目录>/deliverables/`：

- `final.wav`：48 kHz 立体声 24-bit；`final.mp3`：48 kHz 立体声 192 kbps。
- `listen.html`、`original.wav`、`reference.wav`、`restyled.flac`、`vocals_before.wav`、`vocals_after.wav`：试听与过程对照。
- 微调候选由 `candidates.json` / `candidates.html` 说明；分轨和 MIDI 对应推荐候选。
- `review/audio/`、`review/midi/`、`review/notes/`、`review/index.html`：专业评审资料。
- `report.md`、`checks.json`、`review/manifest.json`：技术检查与来源。

分轨、和声和 MIDI 是估计；音轨相加只使用 `stems.json` 的叶轨，不重复叠加父轨。MIDI 的 120 BPM 是保存绝对时间的刻度，不声称原曲速度为 120 BPM。ECAPA、F0 和自动检查不能替代听感与乐谱校对。

```bash
PYTHONPATH=src envs/separator/bin/python scripts/verify_review.py \
  runs/my_chuan_yue_shu_guang_seedvc --output cache/my_delivery_check.json
```

该命令检查成品与所有阶段文件的哈希、音频、MIDI、打包和链接。不要把检查报告写进已发布的 artifacts。完整交付清单与数据保留规则见 [交付指南](docs/21_delivery.md)。
