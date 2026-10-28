# v0.3 分模块实操：孙燕姿音色＋秋意浓＋Bossa Nova / Samba Jazz

这套零样本命令与当前 v0.3 兼容。`runs/lhd_test_modular` 是示例输出名，其 2026-10-23 旧数据已清理；首次从 00 新建即可，文末历史指标不表示目录仍存在。微调单候选的当前完整实例见 [杨坤＋不败的英雄](17_v03_walkthrough.md)。

这是一份按 **00 → 11** 顺序操作的使用手册。每一步都提供可复制命令、上游输入、结果路径和检查重点。先执行一个模块，检查中间结果，再进入下一步。

本文使用已经配置好的 7830 服务器。仓库克隆本身不包含歌曲、参考录音、模型或 Python 环境；新机器先按 [仓库恢复说明](08-repository.md) 准备资源。关于参数、其他歌曲和评审说明，见 [USAGE.md](../USAGE.md)。

模型名称、对应阶段、权重文件、下载命令和校验方法集中见[模型权重清单与准备方法](12_model_weights.md)。

## 1. 本次示例的材料与参数

| 项目 | 本次设置 |
| --- | --- |
| 项目根目录 | `/mnt/afs_f/liuhongda/music-cover-baseline` |
| 原曲 | `/mnt/afs_f/liuhongda/music-cover-baseline/data/raw/songs/qiu_yi_nong.mp3` |
| 歌词 | `/mnt/afs_f/liuhongda/music-cover-baseline/data/raw/lyrics/qiu_yi_nong.txt` |
| 参考音色 | `/mnt/afs_f/liuhongda/music-cover-baseline/data/raw/references/sun_yanzi/reference.wav`，取 0–16 秒歌声 |
| 曲风 | upbeat bossa nova and samba jazz；国语演唱，尼龙弦吉他、贝斯、刷鼓、轻打击乐、长笛和电钢琴等 |
| 自动选调 | `auto`，默认候选 -6～+6 半音，参考明确标记为 `singing` |
| 换声步数 / 随机种子 / GPU | 30 / 42 / 0 |
| 分轨与 MIDI | `--review` 开启 |
| 运行目录 | `/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_test_modular` |

`--run` 决定实际输出目录；`--name` 是任务名称。**lhd_test_modular 位于 runs 内。** 本文所有模块必须使用同一个 `--run`。

## 2. 先初始化当前终端

先从本地连接服务器；已经处于该服务器终端时跳过这一条：

```bash
ssh root@10.210.8.11 -p 7830
```

然后在服务器终端执行：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate base
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
LHD_RUN="/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_test_modular"
```

`(base)` 是交互终端环境；各模块仍自动使用项目的隔离环境。`PATH` 保证找到 FFmpeg。新开终端后请重新执行上面的初始化，尤其是 `LHD_RUN` 变量。

本文按首次成功尝试 `attempt_001` 展示路径。已有运行会校验并复用有效结果；加 `--force` 或修复失败后重跑可能出现 `attempt_002`。实际路径以 `stage_state.json` 为准，查看方法在文末。已有此目录又想做独立实验时，先把 `LHD_RUN` 改成新的 `runs/名称`，例如 `runs/lhd_test_modular_02`。

## 3. 按顺序执行 00–11

每步看到 `[done]` 或 `[reuse]` 且命令正常结束后再继续。独立模块会校验上游产物，不会自动补跑缺失或失效的上游。

### 00_prepare：准备原曲、歌词和音色参考

第一次执行时保存完整请求，后续模块沿用这些参数：

```bash
bash scripts/00_prepare/run.sh \
  --run "$LHD_RUN" \
  --config configs/tasks/pitch_singing_auto.yaml \
  --song data/raw/songs/qiu_yi_nong.mp3 \
  --lyrics data/raw/lyrics/qiu_yi_nong.txt \
  --reference data/raw/references/sun_yanzi/reference.wav \
  --reference-start 0 \
  --reference-duration 16 \
  --transpose auto \
  --style "Mandarin Chinese, upbeat bossa nova and samba jazz, swaying Latin groove, syncopated nylon-string guitar, warm upright bass, brushed drums, shakers and gentle hand percussion, airy flute phrases, soft electric piano, playful jazz chord voicings, clear expressive female lead vocal, delicate female backing harmonies in choruses, bright intimate acoustic production" \
  --name lhd_test_modular \
  --steps 30 \
  --seed 42 \
  --gpu 0 \
  --review
```

读取输入：本节材料表中的原曲、歌词、参考音频。`pitch_singing_auto.yaml` 只覆盖选调设置，明确参考是歌声，不会把曲风改成 Synthwave。

结果目录：

```text
/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_test_modular/stages/00_prepare/attempt_001/artifacts/
```

| 产物 | 检查内容 |
| --- | --- |
| `source.wav` | 完整原曲，48 kHz 立体声 |
| `reference.wav` | 本次实际使用的 16 秒参考歌声，44.1 kHz 单声道 |
| `lyrics.txt` | 清理后的歌词，核对演唱顺序 |
| `metadata.json` | 音频长度、格式、输入哈希等 |

运行根目录还保存 `request.json` 和 `config.resolved.json`，可核对 style、transpose、GPU 等参数。

### 01_source_score：原曲转谱

```bash
bash scripts/01_source_score/run.sh --run "$LHD_RUN"
```

读取输入：**00 的 source.wav**。

本模块结果目录：

```text
/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_test_modular/stages/01_source_score/attempt_001/artifacts/
```

01 原曲乐谱将供 02 移调。此时尚未生成新曲。

| 产物（相对上述目录） | 查看内容 |
| --- | --- |
| `score.abc` | 原曲旋律谱 |
| `melody_vocal.mid / melody_instrumental.mid` | 原曲人声与器乐旋律 MIDI |
| `melody_vocal.lab` | 逐音起止秒数和音高，供 02 核对八度 |
| `key.lab / chord.lab / beat.lab` | 原曲调性、和弦、拍点 |
| `events.json` | 结构化事件 |

### 02_pitch_adaptation：分析音区并双向移调

```bash
bash scripts/02_pitch_adaptation/run.sh --run "$LHD_RUN"
```

读取输入：**01 的 score.abc、melody_vocal.lab，以及 00 的 source.wav、reference.wav**。

本模块结果目录：

```text
/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_test_modular/stages/02_pitch_adaptation/attempt_001/artifacts/
```

先查看 pitch_report.html 和 transposition.json，再比较原谱与移调谱。相同素材在本次实测中选择了 -2 半音（Am → Gm）；新实验以实际报告为准。

| 产物（相对上述目录） | 查看内容 |
| --- | --- |
| `pitch_report.html` | 音高分布图、全部候选评分与选择原因 |
| `transposition.json` | recommended_semitones 是推荐值，applied_semitones 是实际采用值；包含可信度和逐音检查 |
| `pitch_candidates.csv` | 默认 -6～+6 的 13 个候选评分 |
| `score_original.abc / score_transposed.abc` | 移调前后乐谱；03 使用 score_transposed.abc |
| `score_transposed.mid` | 移调后的输入旋律 MIDI，保持原谱时间轴，不是成品转谱 |
| `score_audio_consistency.json` | 谱面与原曲 F0 的音高/八度差检查 |
| `source_separation/vocals.wav` | 分离出的原曲人声 |
| `source_f0.npz / reference_f0.npz` | 原曲与参考 F0 原始数据 |

### 03_style_generation：按指定风格生成完整新曲

```bash
bash scripts/03_style_generation/run.sh --run "$LHD_RUN"
```

读取输入：**02 的 score_transposed.abc、00 的 lyrics.txt，以及保存的 style / seed**。

本模块结果目录：

```text
/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_test_modular/stages/03_style_generation/attempt_001/artifacts/
```

试听 generated/audio.flac，检查歌词、旋律、曲风、段落和结尾。这个 FLAC 已是新生成的完整歌曲，包含人声与伴奏；其人声由 YuE2 生成，05 才使用孙燕姿参考做音色转换。

| 产物（相对上述目录） | 查看内容 |
| --- | --- |
| `generated/audio.flac` | 完整换风格歌曲 |
| `generation_request.json` | 实际风格、歌词、随机种子与旋律条件 |
| `generation_budget.json` | 按乐谱长度安排的生成预算 |

### 04_vocal_separation：分离新歌曲的人声和伴奏

```bash
bash scripts/04_vocal_separation/run.sh --run "$LHD_RUN"
```

读取输入：**03 的 generated/audio.flac**。

本模块结果目录：

```text
/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_test_modular/stages/04_vocal_separation/attempt_001/artifacts/
```

分别试听两轨，检查人声完整性、串音、混响与伴奏损伤。

| 产物（相对上述目录） | 查看内容 |
| --- | --- |
| `vocals.wav` | 换声前的人声，供 05 转换 |
| `instrumental.wav` | 新曲伴奏，供 06 混音及 07 乐器分离 |
| `separation_input.wav` | 本阶段实际分离输入 |

### 05_voice_conversion：转换为孙燕姿参考音色

```bash
bash scripts/05_voice_conversion/run.sh --run "$LHD_RUN"
```

读取输入：**04 的 vocals.wav，以及 00 的 reference.wav**。

本模块结果目录：

```text
/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_test_modular/stages/05_voice_conversion/attempt_001/artifacts/
```

对照 04 人声、00 参考和 converted.wav，判断音色、咬字、颤音、长音、破音及杂音。本阶段额外半音偏移为 0，保持生成歌曲的音高。

| 产物（相对上述目录） | 查看内容 |
| --- | --- |
| `converted.wav` | 转换后的人声，还未做 06 的固定延迟校正 |
| `source_mono.wav` | 换声模型使用的单声道输入 |

### 06_mix：对齐混音并输出 WAV / MP3

```bash
bash scripts/06_mix/run.sh --run "$LHD_RUN"
```

读取输入：**03 的新曲、04 的人声与伴奏、05 的 converted.wav**。

本模块结果目录：

```text
/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_test_modular/stages/06_mix/attempt_001/artifacts/
```

这一步完成后已经可以试听完整成品。检查人声是否与伴奏错位、音量是否合适，并与 reconstruction.wav 对照分离和换声带来的变化。

| 产物（相对上述目录） | 查看内容 |
| --- | --- |
| `final.wav / final.mp3` | 最终混音：48 kHz 立体声，WAV 24 bit；默认 MP3 192 kbps |
| `converted_aligned.wav` | 校正固定延迟后的人声 |
| `reconstruction.wav` | 分离后原人声直接重混的对照 |
| `mix.json` | 对齐、增益、峰值、长度和哈希 |
| `vocal_post_gain.wav / instrumental_post_gain.wav / mix_residual.wav` | 实际入混音的各部分及残差 |

### 07_stem_separation：拆分成品的主唱、和声和乐器

```bash
bash scripts/07_stem_separation/run.sh --run "$LHD_RUN"
```

读取输入：**06 的 final.wav、converted_aligned.wav、mix.json，以及 04 的 instrumental.wav**。

本模块结果目录：

```text
/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_test_modular/stages/07_stem_separation/attempt_001/artifacts/
```

逐轨试听目标声部是否保留、是否串音。12 条 WAV 包含父轨、子轨和残差；相加重建时只使用 stems.json 的 10 条 leaf_sum_roles，避免重复加入父轨。背景人声轨可能含混响，不能直接认定为独立和声。

| 产物（相对上述目录） | 查看内容 |
| --- | --- |
| `audio/lead_vocals.wav / audio/backing_vocals.wav` | 主唱与背景人声估计 |
| `audio/drums.wav / audio/bass.wav` | 鼓与贝斯 |
| `audio/guitar.wav / audio/piano.wav / audio/other.wav` | 吉他、钢琴与其他乐器估计 |
| `audio/vocals.wav / audio/instrumental.wav` | 总人声和总伴奏父轨 |
| `audio/instrumental_vocal_leak.wav / audio/instrumental_residual.wav / audio/mix_residual.wav` | 伴奏人声泄漏、分离残差和混音残差 |
| `stems.json / stems_recombined.wav` | 轨道来源与相加规则、重建对照 |

### 08_final_score：重新识别成品的乐谱和和弦

```bash
bash scripts/08_final_score/run.sh --run "$LHD_RUN"
```

读取输入：**06 的 final.wav**。

本模块结果目录：

```text
/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_test_modular/stages/08_final_score/attempt_001/artifacts/
```

这是成品时间轴上的转谱。01 是原曲谱，02 是调整后输入谱，08 是成品重新识别的谱，三者用途不同。生成可能改变时长，不能直接用相同秒数比较音乐位置。

| 产物（相对上述目录） | 查看内容 |
| --- | --- |
| `score.abc / melody_vocal.mid / melody_instrumental.mid` | 成品的旋律谱与 MIDI |
| `chords.mid / chord.lab / key.lab` | 成品和弦与调性估计，chords.mid 供 09 使用 |
| `beat.lab / downbeat.lab / structure.lab` | 拍点、强拍和段落估计 |
| `transcription_summary.json` | 转谱输出说明与清单 |

### 09_stem_midi：将成品分轨转换为 MIDI

```bash
bash scripts/09_stem_midi/run.sh --run "$LHD_RUN"
```

读取输入：**07 的各声部音频与 stems.json，以及 08 的 chords.mid**。

本模块结果目录：

```text
/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_test_modular/stages/09_stem_midi/attempt_001/artifacts/
```

在 DAW 中把 MIDI 与相应分轨一起检查音高、起止时间、漏音和多余音符。主 MIDI 用固定 120 BPM 表达绝对秒时间，不是声称歌曲实际为 120 BPM；鼓为起音候选，harmony 为和弦指南，均需校对。MIDI 不包含孙燕姿音色。

| 产物（相对上述目录） | 查看内容 |
| --- | --- |
| `midi/lead_vocals.mid / midi/backing_vocals.mid / midi/vocals.mid` | 主唱、背景人声和总人声 MIDI |
| `midi/bass.mid / midi/guitar.mid / midi/piano.mid / midi/other.mid` | 各乐器转谱 |
| `midi/drums_onsets.mid / midi/harmony.mid` | 鼓起音候选与和弦指南 |
| `midi/arrangement_estimate.mid` | 合并编配估计，不重复加入父轨 |
| `notes/ / raw_midi/ / activations/` | 音符 CSV、原始 MIDI、模型激活 |
| `midi.json` | 参数、音符数量和转谱结果状态 |

### 10_evaluation：核验音频、移调、分轨和 MIDI

```bash
bash scripts/10_evaluation/run.sh --run "$LHD_RUN"
```

读取输入：**00 的原曲、02 的移调计划与源 F0、04 的换声前人声、06 的成品与对齐人声，以及 07–09 评审产物**。

本模块结果目录：

```text
/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_test_modular/stages/10_evaluation/attempt_001/artifacts/
```

先检查 technical_validation、pitch_adherence.status 和各项细节。verified 表示当前工程规则通过，不能代替音色、曲风、歌词和自然度的人工评判。

| 产物（相对上述目录） | 查看内容 |
| --- | --- |
| `evaluation.json` | 音频特征、混音、分轨重建、MIDI 与音高核验汇总 |
| `pitch_adherence.json` | 生成实际移调量、换声前后音高保持，以及 needs_review / unassessable 原因 |
| `pitch_profiles.json` | 后端音区统计，供辅助对照 |
| `before_f0.npz / after_f0.npz` | 生成与换声后 F0 数据 |

### 11_delivery：汇总并发布最终交付

```bash
bash scripts/11_delivery/run.sh --run "$LHD_RUN"
```

读取输入：**此前成功阶段中需要交付的音频、乐谱、分轨、MIDI、配置和评估报告**。

本模块结果目录：

```text
/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_test_modular/stages/11_delivery/attempt_001/artifacts/
```

所有交付检查通过后，运行根目录的 deliverables 指向本次成功结果。人工评分请写入运行根目录 reviews 的副本，保持 stages/attempt/artifacts 内的缓存产物不变。

| 产物（相对上述目录） | 查看内容 |
| --- | --- |
| `listen.html / final.wav / final.mp3` | 最终试听入口及完整成品 |
| `pitch_report.html / transposition.json / pitch_adherence.json` | 选调依据和音频实测 |
| `score_transposed.mid` | 02 的移调后输入旋律 MIDI |
| `review/index.html / review/audio/ / review/midi/` | 成品分轨和 MIDI 评审 |
| `review/midi_bundle.zip` | MIDI、音符标注和选调证据的便携包 |
| `evaluation.json / delivery_validation.json / config.resolved.json` | 评估、交付检查和实际配置 |

最终统一查看目录：

```text
/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_test_modular/deliverables/
```

人工评分文件：

```text
/mnt/afs_f/liuhongda/music-cover-baseline/runs/lhd_test_modular/reviews/review_scores.csv
```

## 4. 同一示例的简短命令

完整参数也保存为 `configs/tasks/qiu_yi_nong_bossa_nova_sun_yanzi.yaml`。它与上面的 00 长命令对应；不必再输入长风格描述：

```bash
bash scripts/00_prepare/run.sh \
  --config configs/tasks/qiu_yi_nong_bossa_nova_sun_yanzi.yaml \
  --run "$LHD_RUN"
```

如果希望一次性执行 00–11：

```bash
bash scripts/run_baseline.sh \
  --config configs/tasks/qiu_yi_nong_bossa_nova_sun_yanzi.yaml \
  --run "$LHD_RUN"
```

如果已经逐步执行到 02，从 03 继续全部后续步骤：

```bash
bash scripts/run_baseline.sh --run "$LHD_RUN" --from 03_style_generation
```

上述命令默认复用有效缓存；使用已有目录时不会默认重生成同一份有效结果。

## 5. 进度、实际 attempt 和重跑

查看完整运行状态：

```bash
bash scripts/run_baseline.sh --run "$LHD_RUN" --status
```

打印每个模块当前记录的状态和实际结果路径：

```bash
envs/control/bin/python - "$LHD_RUN" <<'PYCODE'
import json
from pathlib import Path
import sys
state = json.loads((Path(sys.argv[1]) / 'stage_state.json').read_text())
for name, item in state['stages'].items():
    print(name, item['status'], Path(item['attempt']) / 'artifacts', sep='	')
PYCODE
```

记录为 failed / running 的 attempt 不是可交付的成功结果。每个 attempt 旁边有 `inputs.json`、`config.json`、`command.json`、`process.log`、`checks.json`；成功后有 `outputs.json`，可核对输入路径、哈希、参数与输出。

例如观察 03 首次尝试的日志：

```bash
tail -f "$LHD_RUN/stages/03_style_generation/attempt_001/process.log"
```

按 Ctrl+C 结束 `tail` 只会停止看日志，不会停止另一个终端正在运行的生成任务。

只强制重跑 09，然后重新评估与打包：

```bash
bash scripts/09_stem_midi/run.sh --run "$LHD_RUN" --force
bash scripts/run_baseline.sh --run "$LHD_RUN" --from 10_evaluation
```

修复失败原因后，可直接重发该模块命令。若修改的是上游参数/代码，需要从受影响的最早阶段继续，下游会按依赖重新执行。不要手工修改缓存目录中的音频、JSON 或 MIDI，否则哈希检查会令缓存失效。

只查看执行计划，不启动模型也不创建运行目录：

```bash
bash scripts/run_baseline.sh \
  --config configs/tasks/qiu_yi_nong_bossa_nova_sun_yanzi.yaml \
  --run "$LHD_RUN" --plan
```

## 6. 这次实测如何理解

2026-10-23，服务器的 `runs/lhd_test_modular` 已完成 00–11：成品 397.638667 秒，12 条评审音轨、10 个主 MIDI、167 个评审文件通过交付检查。

| 项目 | 本次实测 | 解读 |
| --- | --- | --- |
| 推荐 / 实际移调 | -2 / -2 半音 | 原谱 Am → Gm，参考音区判断为 medium |
| 生成歌声音高变化中位数 | -2.2533 半音 | 当前验收规则通过，接近要求的 -2 |
| 对齐片段在目标 ±1 半音内 | 84.96% | 不是逐音音准正确率，DTW 和 F0 估计均可能有误差 |
| 换声前后变化中位数 | +0.0127 半音 | 整体音高基本保持 |
| 换声前后差异在 ±1 半音内 | 99.80% | 通过音高保持检查 |
| 主观质量 | awaiting_human_review | 音色、曲风、自然度、歌词完整性仍待人工试听 |

`source_coverage=1.0` 主要来自 DTW 全序列对齐，不代表歌词和旋律保留了 100%。分轨相加误差小不代表分离干净；MIDI 非空不代表音符准确；同为约 152 BPM 的自动估计也不能判断曲风是否变化。完整核验记录见 [VALIDATION.md](../VALIDATION.md)。

优先试听：00 原曲 → 03 新曲 → 04 换声前人声 → 05 换声后人声 → 06 最终混音，然后逐轨核对 07 音频和 09 MIDI。这样可以区分生成、换声、混音和转谱各阶段的问题。
