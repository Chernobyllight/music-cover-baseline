# 使用说明（v0.2）

项目位置：`/mnt/afs_f/liuhongda/music-cover-baseline`（现有 `/data/liuhongda/music-cover-baseline` 为同一共享目录的入口）。已有的环境、权重和演示数据可重复使用，无需训练目标人物音色模型。

**按模块一步步运行：**见 [00–11 完整实操手册](docs/11_modular_walkthrough.md)，使用“孙燕姿＋秋意浓＋upbeat bossa nova and samba jazz”，包含长命令、简短配置、完整输出路径及检查重点。

**模型权重：**见[完整清单与准备方法](docs/12_model_weights.md)，包含主模型与附属模型、下载步骤、实际路径、版本及校验命令。

现有服务器可直接打开 `runs/20261022_111548_xi_huan_ni_bossa_nova_sun_yanzi_auto_review_a714d5/deliverables/listen.html` 验收《喜欢你》结果；具体记录见 [VALIDATION.md](VALIDATION.md)。该目录是服务器本地数据，不随 Git 克隆。早期演示目录及便捷链接已清理。历史基础流程约 3 分钟的耗时不适用于当前完整评审流程。

实际链路：原曲 → SheetSage2 提取旋律 → YuE2 按指定风格重生成 → BS-RoFormer 分离 → Seed-VC 参考音色转换 → 对齐、混音 → 成品分轨、重新转谱、专业评审包。历史耗时不包含新增的评审阶段。

## 模块化执行（v0.2）

完整阶段表、依赖图、配置覆盖与产物契约见 [模块化说明](docs/09_modular_pipeline.md)。原有长命令仍可完整执行；每个阶段也可从对应的 `scripts/NN_名称/run.sh` 独立执行。

```bash
export PATH="/opt/conda/bin:$PATH"
bash scripts/run_baseline.sh --config configs/tasks/qiu_yi_nong_synthwave_sun_yanzi.yaml

# 先做到混音，再从成品分轨继续
bash scripts/run_baseline.sh --config configs/tasks/qiu_yi_nong_synthwave_sun_yanzi.yaml \
  --run runs/my_modular_test --until 06_mix
bash scripts/run_baseline.sh --run runs/my_modular_test --from 07_stem_separation

# 只重跑 MIDI；省略 --force 时可复用本阶段缓存
bash scripts/09_stem_midi/run.sh --run runs/my_modular_test --force
bash scripts/run_baseline.sh --run runs/my_modular_test --from 10_evaluation
```

新产物位于 `runs/<run>/stages/<module>/attempt_NNN/artifacts/`；输入、配置、命令、日志、检查及成功清单保存在 attempt 同级。单阶段不会悄悄补跑失效的上游；错误会指出需从哪一步开始。完整交付通过后 `deliverables` 指向 11 的成功产物；人工评分请写在 `runs/<run>/reviews/`。

历史 v0.1 运行目录继续只读保留，验证器兼容旧格式；`--resume` 只用于新的模块化目录。下文历史案例中的旧阶段编号只描述当时的运行，不应直接用于新模块命令。

## 你需要准备什么

| 材料 | 放置位置 | 要求 |
| --- | --- | --- |
| 原曲 | `data/raw/songs/my_song.mp3` | WAV、FLAC、MP3 均可；首轮建议一首中文独唱歌曲 |
| 歌词 | `data/raw/lyrics/my_song.txt` | UTF-8；内容和演唱顺序与原曲一致；可用带时间戳的 LRC |
| 目标人物参考录音 | `data/raw/references/my_voice.wav` | 建议 10–20 秒、单人、清晰、尽量无伴奏、混响和噪声；说话或清唱均可 |
| 目标风格 | 命令中的 `--style` | 描述曲风、乐器、演唱和编曲，例如 acoustic folk ballad, warm acoustic guitar, soft piano |

默认取参考录音的前 15 秒，最长允许 25 秒。可用 `--reference-start 20 --reference-duration 15` 改为从第 20 秒取 15 秒。参考录音不会自动分离伴奏，请先准备干净的录音。只有参考录音也能执行；相似度和自然度需要听结果判断。

## 先复现内置演示

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/run_baseline.sh --demo
```

演示输入是 3 分 23 秒的官方 AI 生成中文歌曲《今晚不眠》，歌词与歌曲来自同一份官方示例；目标音色是公开 LibriSpeech 语音样本。素材来源和 SHA256 见 `data/demo_sources.json`。该演示用于验证流程，尚未使用你的目标人物录音。

## 跑自己的歌曲

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/run_baseline.sh \
  --song data/raw/songs/my_song.mp3 \
  --lyrics data/raw/lyrics/my_song.txt \
  --reference data/raw/references/my_voice.wav \
  --style "Mandarin Chinese, acoustic folk ballad, warm acoustic guitar, soft piano, intimate clear solo vocal" \
  --name my_song_folk \
  --gpu 0
```

不需要手动激活环境。默认使用 GPU 0，各模型依次加载。相对输入路径按项目根目录解释；使用绝对路径也可以。

入口使用项目内固定的 Python 环境；FFmpeg / FFprobe 仍需在 `PATH` 中。若登录后提示 `ffmpeg is not available`，先执行 `export PATH="/opt/conda/bin:$PATH"`，再用 `bash scripts/run_baseline.sh --preflight` 检查。如果希望终端显示 `(base)`，可执行 `source /opt/conda/etc/profile.d/conda.sh` 和 `conda activate base`；这不会替代项目各模型的隔离环境。

当前脚本为本机 H200 配置，YuE2 的显存预算设为 80 GiB。不要直接将这份配置套在 24 GB 显卡上；小显存部署需要调整预算并重新验证。

换风格是根据旋律和歌词重新生成音乐，时长、演唱细节和编曲可能变化，不能当作原音频的严格逐帧编辑。换音色和最终混音则保持新生成歌曲的时间轴。

YuE2 默认的 9000 个语义 token 约对应 6 分钟。当前脚本会按乐谱时长自动提高生成上限，并预留结尾余量；例如《秋意浓》使用 12000 token 上限（约 8 分钟容量），模型仍可提前正常结束，不会强制生成 8 分钟。实际设置保存在 `stages/03_style_generation/attempt_NNN/artifacts/generation_budget.json` 及原生生成配置中。若仍出现 `semantic: True`，该次输出属于截断结果，不应作为完整歌曲验收；脚本保留此检查。

## 按参考歌声音区自动升调或降调

使用已标记为歌声参考的任务配置：

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
bash scripts/run_baseline.sh \
  --config configs/tasks/qiu_yi_nong_synthwave_sun_yanzi.yaml \
  --run runs/qiu_yi_nong_pitch_v2
```

`auto` 在 -6～+6 半音中比较分布距离、移调幅度和参考稳定性，可以升、降或保持原调。推荐值与实际采用值分别记录。人工 `--transpose -2` 固定降 2 半音；`--transpose 3` 固定升 3 半音；支持 -12～12。默认及 `--transpose 0` 关闭分析、原谱直通。

只有确认歌唱参考的 `reference_kind: singing` 才自动采用非零建议。上述任务已配置；其他歌曲可在原有命令中加 `--config configs/tasks/pitch_singing_auto.yaml`，它只覆盖 transpose 和 02 模块选调参数。讲话或未知参考只提供临时候选并保留原调。更长/多段音区参考、候选范围和人工确认音域见 [02 README](scripts/02_pitch_adaptation/README.md)；不要把短参考的 P95 当作歌手上限。

流程为：原曲人声与参考 F0 → 双向选择 k → 全谱统一移调 → YuE2 生成歌声与伴奏 → Seed-VC 保持音高换声 → 10 实测核验。02 同时检查原谱和原曲音频的八度差，不自动施加隐藏的八度修正。

| 交付文件 | 内容 |
| --- | --- |
| `pitch_report.html` / `pitch_candidates.csv` | 分布图、所有候选评分及决策原因 |
| `transposition.json` | recommended_semitones、applied_semitones、可信度、参考质量和逐音检查 |
| `score_original.abc` / `score_transposed.abc` / `score_transposed.mid` | 原谱与统一移调后的谱、输入旋律 MIDI；不同于成品转谱 MIDI |
| `score_audio_consistency.json` | 原谱与原曲 F0 的八度/音高检查 |
| `pitch_adherence.json` | 生成实际移调量、换声前后保持程度及需要复核的问题 |
| `pitch_input.json` / `pitch_output.json` | 后端原有口径统计，供兼容对照；选调实际采用的统计在 transposition.json |

技术流程成功不等于 YuE2 逐音遵循或听感通过。详细算法、阈值及已知限制见 [双向选调说明](docs/10_bidirectional_pitch.md)。历史 v1 升调/原调结果保留在 [VALIDATION.md](VALIDATION.md)，不会由新算法改写。

## 粤语示例：《喜欢你》巴萨诺瓦／桑巴爵士

现有服务器已准备 `xi_huan_ni_beyond.mp3`、测试用 `.txt` 歌词、原始 `.lrc` 和来源记录。Git 不包含这些媒体或歌词，需要自行准备同名文件或替换输入路径。`--style` 使用 `Cantonese` 指定粤语；这属于生成目标，实际发音和歌词忠实度仍需验收。

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
bash scripts/run_baseline.sh \
  --song data/raw/songs/xi_huan_ni_beyond.mp3 \
  --lyrics data/raw/lyrics/xi_huan_ni_beyond.txt \
  --reference data/raw/references/sun_yanzi/reference.wav \
  --reference-start 0 --reference-duration 16 \
  --config configs/tasks/pitch_singing_auto.yaml \
  --transpose auto \
  --style "Cantonese, upbeat bossa nova and samba jazz, swaying Latin groove, syncopated nylon-string guitar, warm upright bass, brushed drums, shakers and gentle hand percussion, airy flute phrases, soft electric piano, playful jazz chord voicings, clear expressive female lead vocal, delicate female backing harmonies in choruses, bright intimate acoustic production" \
  --name xi_huan_ni_bossa_nova_sun_yanzi_auto_review \
  --steps 30 --seed 42 --gpu 0 --review
```

评审资料见下文；整首生成和评审可能占用数 GB 磁盘空间。已有结果无需再次生成，直接对其运行 `scripts/verify_review.py` 即可检查文件完整性。

## 在哪里验收

命令开始时打印 `RUN: ...`，成功后打印 `RESULTS (completed): <运行目录>`；只执行部分模块时显示 `RESULTS (partial)`。完整交付位于该运行目录的 `deliverables/`。未指定 `--run` 时自动创建新目录，指定已有 `--run` 则校验并复用有效缓存。

打开该目录的 `listen.html`，依次播放原曲、换风格、最终混音，并与参考录音比较。也可以直接播放 WAV/FLAC/MP3 文件。最终混音默认同时输出 WAV 和 MP3，无需额外参数；试听页提供两种格式的播放和下载入口。

| 文件 | 验收内容 |
| --- | --- |
| `original.wav` | 输入原曲 |
| `reference.wav` | 本次实际使用的目标音色片段 |
| `restyled.flac` | YuE2 换风格结果；先确认歌词、旋律、曲风 |
| `reconstruction.wav` | 分离后原人声重新贴回的对照；检查分离损伤 |
| `final.wav` | 最终换音色并混入伴奏的结果，48 kHz、立体声、24 bit WAV |
| `final.mp3` | 同一最终混音的 48 kHz、立体声、192 kbps MP3，便于下载和分享 |
| `vocals_before.wav` / `vocals_after.wav` | 单独比较换声前后的歌声 |
| `instrumental.wav` | 分离出的伴奏 |
| `report.md` / `checks.json` | 耗时、时长、对齐、峰值等自动检查记录 |

优先听开头、主歌、副歌、高音、间奏前后、结尾，确认没有漏词、乱词、破音、明显人声残留和伴奏错位。自动检查通过表示技术流程完成，不等于听感达标。

MP3 由最终 WAV 使用 FFmpeg 的 `libmp3lame` 编码器生成。启动前会检查编码器是否可用；`checks.json` 的 `final_mp3` 项保存时长、采样率、声道、码率和 SHA256，阶段记录也包含 MP3 文件哈希。既有运行目录不会自动补写 MP3。

## 成品分轨与 MIDI 专业评审包

默认开启，无需额外参数；`--no-review` 关闭新增阶段。`--review` 可显式开启。旧运行的原曲 MIDI 不能代替成品 MIDI。

| 位置（相对 deliverables/review） | 内容 |
| --- | --- |
| `index.html` | 分轨试听、MIDI 下载、说明和评分表入口 |
| `audio/vocals.wav`、`audio/instrumental.wav` | 已应用最终混音增益的总人声、总伴奏父轨 |
| `audio/lead_vocals.wav`、`audio/backing_vocals.wav` | Mel-Band RoFormer karaoke 对转换后人声的主唱／背景人声估计 |
| `audio/drums.wav`、`bass.wav`、`guitar.wav`、`piano.wav`、`other.wav` | Demucs htdemucs_6s 对成品伴奏的分离估计；other 可含合成器、弦乐等 |
| `audio/instrumental_vocal_leak.wav` | 伴奏中被模型判成人声的残留，不能视为转换后的目标人声 |
| `audio/instrumental_residual.wav`、`audio/mix_residual.wav` | 分离重建残差、混音量化残差；并非乐器 |
| `midi/*.mid`、`notes/*.csv` | 各声部成品音频重新转谱，含起止秒数、音高、力度和模型激活值 |
| `midi/harmony.mid` | SheetSage2 从成品识别的和弦指南；不是实际乐器的完整演奏谱 |
| `midi/drums_onsets.mid` | 鼓的低／中／高频起音候选；GM 36／38／42 为粗略试听映射，不是可靠鼓种识别 |
| `midi/arrangement_estimate.mid` | 主唱、背景人声和各乐器的合并估计，不重复加入父轨人声或和弦指南 |
| `final_transcription/` | 成品完整转谱：MIDI、ABC（如可生成）、和弦、调性、拍点、段落、原始事件 |
| `source_transcription/` | 输入原曲转谱对照；与成品时间线、调性可能不同 |
| `raw_midi/`、`activations/` | Basic Pitch 原始 MIDI 和压缩后的 note／onset／contour 激活 |
| `stems.json`、`midi.json`、`manifest.json` | 文件来源、参数、状态、校验和、重建误差与限制 |
| `audio_comparison.json` | 原曲／成品响度、峰值、速度和频谱描述；不能当作风格或音色评分 |
| `review_scores.csv`、`excerpts/` | 人工评审空表、固定时间的 20 秒试听片段 |
| `midi_bundle.zip` | MIDI、音符和分析标注的便携包，不含大型音轨或模型激活 |

评审音轨均为 48 kHz、立体声、32-bit float、相同采样点长度和起点，不进行单轨响度归一化。相加还原最终混音时，**只使用 `stems.json` 的 `leaf_sum_roles`**；不要再加入父轨 vocals／instrumental。残差保留了总和，但小重建误差不代表串音少或分离准确。

“背景人声”可能包含主唱混响、残留或伪影；钢琴／吉他轨存在文件不代表编曲真的包含该乐器。低能量和空 MIDI 会明确标记，不添加假音符填充。所有转谱为估计，需人工检查音高、节奏、漏音和多余音符。MIDI 没有参考歌手的音色。

主 MIDI 用 120 BPM 固定时间刻度保存绝对秒数，**不是声称歌曲速度为 120 BPM**，也没有进行节拍量化。成品估计拍点另见 `final_transcription/beat.lab`。主 MIDI 不合并可能冲突的复音弯音，原始输出另存 `raw_midi/`。

验证交付文件（使用已安装 MIDI 依赖的 separator 环境）：

```bash
cd /data/liuhongda/music-cover-baseline
envs/separator/bin/python scripts/verify_review.py /绝对路径/runs/某次运行 \
  --output /绝对路径/runs/某次运行/delivery_verification.json
PYTHONPATH=src envs/separator/bin/python -m unittest discover -s tests -v
```

验证包括 SHA256、音轨时长／格式／数值、叶轨相加误差、MIDI 合法性与时间范围、压缩包和试听链接完整性。它不替代专业听音和乐谱校对。

新机器上的评审环境安装（复用已配置的 separator 环境与 CUDA 包，不降级 Torch）：

```bash
envs/separator/bin/python -m pip install --no-deps -r configs/review-requirements.txt
envs/separator/bin/python scripts/setup_review_assets.py
bash scripts/run_baseline.sh --preflight
```

首次需下载额外模型；`setup_review_assets.py --local-assets /目录` 可使用同名本地下载文件。模型 URL、SHA256 和实际包版本写入 `configs/review_models.lock.json`，运行前校验。该脚本记录下载后的哈希，并只对 Demucs 权重检查上游文件名中的短哈希；迁移时可与仓库保存的完整哈希比较。GPU 分离与 CPU ONNX 转谱的耗时、磁盘占用均高于旧基础流程。Basic Pitch 使用 ONNX CPU；提示未安装 TensorFlow、CoreML 或 TFLite 不影响此配置。

## 失败重跑、只换参考人物

保留命令打印的运行目录，失败后查看对应阶段的 `process.log`。修复问题后：

```bash
bash scripts/run_baseline.sh --resume /绝对路径/runs/某次运行
```

更换参考录音后按依赖复用有效结果（启用 auto 时，02 会重新分析，03 及后续也可能重跑）：

```bash
bash scripts/run_baseline.sh \
  --resume /绝对路径/runs/某次运行 \
  --reference /绝对路径/new_reference.wav
```

输入、参数、适配代码和产物哈希一致的已完成阶段会复用；受影响阶段会重新执行。各次成功交付保存在对应的 `stages/11_delivery/attempt_NNN/artifacts/`，`deliverables` 只在新交付通过后切换指向；若接管的是旧实体交付目录，才会移入 `deliverables_history/`。失败时不会把旧音频标成新结果。

## 可选的附加自动验证

```bash
CUDA_VISIBLE_DEVICES=0 USE_TF=0 USE_FLAX=0 \
  envs/seed-vc/bin/python scripts/verify_result.py /绝对路径/runs/某次运行
```

结果写入 `deliverables/verification.json`，包括换声前后音高差异、Whisper 歌词识别结果及 CAMPPlus 音色相似度诊断。CAMPPlus 同时被换声模型使用，因此该项不是独立评价；ASR 和音高估计也有自身误差，不用这些分数替代试听。

## 文件位置

代码在 `src/music_cover/` 和 `scripts/`，上游仓库在 `third_party/`，权重在 `models/`，隔离环境在 `envs/`，下载缓存和安装日志在 `cache/`。模型 revision 清单在 `configs/`。

本机的虚拟环境复用了 `/opt/conda` 的基础 CUDA 包，不能直接复制 `envs/` 到另一台机器使用。迁移时应按环境记录重新安装，并验证 GPU 推理。旧 `docs/` 中的方案文件保留为设计记录；当前操作以本说明和实际运行报告为准。
