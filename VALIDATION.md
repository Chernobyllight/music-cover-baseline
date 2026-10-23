# 实际验证记录


## 2026-10-23：v0.2 Bossa Nova / Samba Jazz 使用示例

发布前复核：对 `runs/lhd_test_modular` 的独立交付检查通过；45 项回归测试和完整环境预检通过；实操手册的 22 个 Bash 代码块通过语法检查，12 个独立模块命令的 `--plan` 均对应正确阶段；19 个示例绝对路径和 105 个本地 Markdown 链接存在；79 个项目源码文件与代码锁一致。新增 Bossa Nova 任务配置与本次完整运行保存的请求和模块设置一致。提交候选仅含源码、文档、配置、测试、演示来源元数据及两个固定子模块，不含媒体、模型、运行目录或临时认证文件。

运行 `runs/lhd_test_modular` 使用孙燕姿 0–16 秒歌声参考、《秋意浓》及 upbeat bossa nova and samba jazz 风格，seed=42、30 步、GPU 0、review=true。00–11 全部 succeeded，运行 completed；可复制命令和逐阶段路径见 [实操手册](docs/11_modular_walkthrough.md)。

- 实际选调 -2 半音，Am → Gm，可信度 medium。生成音频对齐后的音高变化中位数 -2.253262 半音，目标 ±1 半音内比例 84.9638%；换声变化中位数 +0.012660 半音，±1 半音比例 99.7955%，两项均通过当前工程规则。
- 最终长度 397.638667 秒，48 kHz 立体声 WAV / 192 kbps MP3；12 条音轨、10 个主 MIDI 均已生成，没有空主 MIDI。
- 交付检查通过，167 个评审文件，叶轨相加重建最大绝对误差约 `5.01957e-8`。音色、风格、歌词完整性、演唱自然度与逐音转谱准确性仍为 awaiting_human_review。
- 原曲/成品 LUFS 为 -14.74 / -17.51，成品真峰值 -1.10 dBTP。自动速度估计均约 152 BPM；这些描述性数值不构成曲风或音色相似度评分。DTW source_coverage=1.0 不等于歌词/旋律 100% 保留。


## 2026-10-23：双向选调 v2 与降调版完整验证

新运行：`runs/20261023_pitch_v2_qiu_yi_nong_sun_yanzi`。使用现有《秋意浓》、孙燕姿 0–16 秒歌声参考、Synthwave / Dance-pop 风格，seed=42、30 步、GPU 0、review=true。00–11 全部成功；最终长度 393.318667 秒，WAV 与 MP3 均交付。历史运行只用于读取对照数据，没有改写其产物。

### 真实数据与已知量对照

| 数据 | 统计推荐 | 实际采用 | 结论 |
| --- | ---: | ---: | --- |
| 《秋意浓》＋孙燕姿歌声，新跑 00–02 | -2 | -2 | medium；Am → Gm；方向重采样支持率 0.71875 |
| 《喜欢你》＋同一参考，历史原始 F0 回放 | -1 | 0 | low；方向支持率 0.5390625，不满足默认 0.60，保留原调并标记 ambiguous_reference |
| 两首歌源 F0 的人工参考 -3 / 0 / +3 | 分别 -3 / 0 / +3 | 与已知量全部一致 | 6 个对照通过；这是人工 F0 对照，不是新增真实演唱 |

《秋意浓》参考有效有声 11.43 秒，4 个可用时间块。精确幅度仍有不确定性：±1 半音稳定率约 0.4922，报告保留所有候选和替代值；中等可信度不代表已测出完整歌唱音域。默认评分从原调约 2.4863 改善到 -2 的约 1.3027。

原谱与移调谱逐事件检查了 258 个人声音符、127 个器乐音符：全部精确下降 2 半音，起点、时值、速度和节拍不变，调号及和弦同步变化。移调 MIDI 另用 pretty_midi 解析核对音符及绝对时间，通过；原生 MIDI 导出不依赖 control 环境安装 mido。

### 音频实测

10 的正式 `pitch_adherence.json` 标记 verified：

- 原曲→新生成歌声：相对轮廓 DTW 后，1104 个源有声片段的中位移调量 **-2.14053 半音**；相对要求 -2 的 ±1 半音比例 75.634%，差值的中位绝对偏差 0.35362 半音。
- 换声前→延迟校正后的换声人声：1146 个共同时段，中位变化 **+0.01344 半音**，99.738% 的比较值在 ±1 半音内。没有额外施加隐藏移调。
- 真实反例检查：旧原调版实测 -0.22381，按原调判定通过，故意按 -2 验收则 needs_review；把新降调版故意按 0 验收也 needs_review。
- 独立 pYIN 在原曲、生成歌声、换声歌声各取 55–67 秒和 155–167 秒，与各自 RMVPE 同时间戳比较。六组音高差绝对值中位数约 0.086–0.188 半音，±1 半音比例约 88.9%–97.9%。这只是同一音频的双方法抽查，不是按相同秒数对齐原曲和成品。

原谱与原曲 F0 的约 +12 半音差异仍会报告为 octave_offset_detected；没有据此擅自再降一八度。生成后实测支持相对 -2 的控制生效，但不保证所有音符、歌词和结构完全正确。

### 完整交付与代码检查

交付校验通过：12 条评审音轨、10 个主 MIDI，均无空主 MIDI；167 个评审文件哈希、ZIP 内容、HTML 链接检查通过，叶轨相加重建最大绝对误差约 `1.5232e-7`。02 的报告、CSV、ABC、输入谱 MIDI、八度检查，以及 10 的音高核验同时进入主交付、`review/pitch_adaptation` 和 MIDI ZIP。

45 项回归测试全部通过（原 33 项 + 新 12 项）。覆盖升/降/0、讲话与未知参考、参考不足或不稳定、手动覆盖、安静歌声、非有限值、孤立八度尖峰、参数范围、多个参考、八度识别、错误移调检测和 MIDI 事件/时间往返。完整测试应使用已有 separator 环境；control 环境不含评审测试所需 mido。另完成 66 个 Python 文件语法、79 个代码锁文件哈希、23 个文档 Bash 示例语法检查。

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
PATH="/opt/conda/bin:$PATH" PYTHONPATH=src envs/separator/bin/python -m unittest discover -s tests -v
```

可复核证据：新运行 `deliverables/pitch_report.html`、`pitch_adherence.json`、`delivery_validation.json`；回放位于 `runs/20261023_pitch_v2_replay_qiu/selection_replay.json` 和 `runs/20261023_pitch_v2_replay_xi/selection_replay.json`；额外原谱/MIDI、反例、pYIN 检查及测试日志保存在 `cache/pitch_v2_backup_20261023/`。新算法与阈值见 [双向选调说明](docs/10_bidirectional_pitch.md)。

完整续跑再次验证：00–11 共 12 阶段全部复用，未新建 attempt，最终状态仍为 completed，发布链接指向本次已验证交付。日志见 `cache/pitch_v2_backup_20261023/resume.log`。

自动核验不替代专业试听。短参考的推荐幅度、孙燕姿音色相似度、合成自然度和成品转谱准确性仍需人工评审。


## 2026-10-23：模块化重构整首验证

基于 v0.1（`83819b5`）拆为 12 个独立模块，当前服务器工作分支 `codex/modular-baseline`。没有使用旧运行的生成缓存替代本次推理：新建 `runs/20261023_modular_qiu_yi_nong_sun_yanzi_validation`，使用孙燕姿歌声 0–16 秒、完整《秋意浓》、Synthwave / Dance-pop 风格、auto 移调、seed 42、30 步、GPU 0。

实际验证分三次完成：先通过 `scripts/00_prepare/run.sh` 独立准备输入；再从 01 跑到 `--until 06_mix`，状态为 partial；最后 `--from 07_stem_separation` 复用已验证的 00–06，完成成品分轨、成品转谱、MIDI、评估与交付。所有 12 个模块均成功。

| 检查 | 实测结果 |
| --- | --- |
| 单元与集成测试 | 33 项通过，覆盖损坏拒绝、参数/源码/契约变更、失败重试、锁、原子发布、保留人工评分、零移调直通等 |
| 完整交付 | 396.998667 秒，48 kHz 立体声，24-bit WAV / 192 kbps MP3 |
| 模块契约 | 12 个模块均保留 inputs/config/command/process.log/checks/outputs 清单 |
| 独立评审文件检查 | 157 个文件；所有模块输出哈希也通过 |
| 分轨 / MIDI | 12 条 WAV（10 条叶轨可相加）、10 个主 MIDI；没有空主 MIDI |
| 重建最大绝对误差 | `1.4761553757125512e-07`，包含残差，不代表分离准确率 |
| MIDI 主唱 / 合并编曲音符数 | 795 / 10240 |
| 单模块入口 | 09 子目录入口通过 --force 独立重新转写 MIDI，生成 attempt_002；之前的生成、换声和分轨均复用 |
| 再次完整续跑 | 12 个模块全部 `[reuse]`，没有新建任何 attempt |
| 历史兼容 | 旧《喜欢你》122 个评审文件通过新版验证器；旧运行未改写 |
| 基础交付分支 | 在无 mido 的 control 环境中通过 `review=false` 验证；音频单测也覆盖基础交付 |
| 源码锁 | 75 个 Python / Shell 文件与当前锁和交付 provenance 对应 |

原曲准备数据、乐谱、YuE2 音频、分离人声与伴奏、最终 WAV / MP3 等与 v0.1 对照。13 个主要产物中 11 个文件哈希完全相同；另外两份浮点 WAV 容器字节不同，但解码后所有采样值完全相同，最大误差 0。最终 WAV SHA256 为 `2fae3f11a08649180c70b18b981cb4fc1b7bf20ab743ce0239eb6d5f342a1843`，MP3 为 `56f569b81ffe4f6c7c39a574da1f9579491a64d398320300a114180726838247`。本次重构保持默认音频结果。

验证命令：

```bash
export PATH="/opt/conda/bin:$PATH" PYTHONPATH=src
envs/separator/bin/python -m unittest discover -s tests -v
envs/separator/bin/python scripts/verify_review.py \
  runs/20261023_modular_qiu_yi_nong_sun_yanzi_validation \
  --output cache/modularization_backup_20261023/modular_verification.json
```

证据保存在 `cache/modularization_backup_20261023/`：`tests_final.log`、`baseline_comparison.json`、`modular_verification.json`、`legacy_verification.json`、`final_report.json`。源码修改前备份为同目录 `code_before.tar.gz`。各次执行记录在运行目录的 `executions/`。

自动检查确认可执行、文件完整、时轴和数值有效，不代替专业听评。此前 v0.1 的歌词识别异常与听感待验收结论仍然适用；本次没有重新做 Whisper/CAMPPlus 诊断，也没有宣称提升音乐质量。模块接口和使用方式见 [模块化说明](docs/09_modular_pipeline.md)。

## 2026-10-23：v0.1 上传前复核

本次复核使用服务器当前代码和已经完成的结果，没有重新运行音乐生成。`PYTHONPATH=src envs/separator/bin/python -m unittest discover -s tests -v` 的 **22 项测试全部通过**；`bash scripts/run_baseline.sh --preflight` 通过，包括新增评审模型的文件大小、SHA256 及依赖导入检查。

测试覆盖歌词时间戳、长曲生成预算、音频对齐与重采样、WAV / MP3 输出和失败保护、缓存损坏恢复、谱面移调、混音后分轨重建，以及 MIDI 的绝对时间、重复音、静音、鼓通道和非法音符拒绝。

对用户已生成的《喜欢你》Bossa Nova / Samba Jazz 版本执行独立验证：

```bash
envs/separator/bin/python scripts/verify_review.py \
  runs/20261022_111548_xi_huan_ni_bossa_nova_sun_yanzi_auto_review_a714d5 \
  --output cache/publish_v0.1_20261023/xi_huan_ni_verification.json
```

输入为 BEYOND《喜欢你》，歌词提示语言为粤语，参考为孙燕姿歌声前 16 秒；`seed=42`、30 步、GPU 0、`--transpose auto --review`。自动选择为 0 半音（C → C）。所有生成及评审阶段已完成，结果如下：

| 检查项 | 实测结果 |
| --- | --- |
| 成品时长 | 269.838667 秒 |
| WAV / MP3 | 48 kHz、立体声；24-bit WAV / 192 kbps MP3 |
| 评审文件 | 122 个文件通过哈希与完整性检查 |
| 评审音轨 | 12 条 48 kHz 立体声 FLOAT 音轨；其中 10 条为可相加的叶轨 |
| 重建最大绝对误差 / RMS 误差 | `1.1428164725657552e-7` / `5.2050908815840604e-9` |
| 主 MIDI 文件 | 10 个，音符数和时间范围通过检查，没有空轨 |
| MIDI 示例音符数 | 主唱 584、背景人声 492、合并编曲 9607 |
| 最终 WAV SHA256 | `f989d6c930c7efc395a730d826fdd13970ae7e32092639e3c0475416805a7b3e` |
| 最终 MP3 SHA256 | `e0872bee0e4e054da094026fcbffbb736afc144b51b5966bd5557eeb418edba5` |

还检查了 ZIP 内容、页面链接、阶段完成状态和音轨数值。重建误差包含残差轨，不能解释为分离准确率。MIDI 是音频转录估计，不是原生制作工程；背景人声、乐器分配、节奏和音高仍需人工校对。本次未对《喜欢你》运行额外 Whisper / CAMPPlus 诊断，也未进行人工听感验收，不能据此确认粤语发音、歌词忠实度或参考音色相似度达标。

## 2026-10-22：成品分轨与 MIDI 的整首验证

完整运行：`runs/20261022_101113_qiu_yi_nong_synthwave_dance_sun_yanzi_review_910315`。输入《秋意浓》，参考孙燕姿歌声 0～16 秒，目标 Synthwave / Dance-pop，`seed=42`、30 步、`--transpose auto --review`；自动选择 0 半音（Am → Am）。

结果为 396.998667 秒、48 kHz 立体声 24-bit WAV，以及 192 kbps MP3。主流程加评审共 13 个阶段完成。新增的 `07_review_stems`、`08_final_transcribe`、`09_review_midi`、`10_review_package` 分别生成转换后分轨、成品完整转谱、各声部 MIDI 和专业评审包。

独立验证通过 122 个评审文件，含 12 条音轨、10 个主 MIDI 文件，合并编曲 10240 个音符。叶轨与残差相加后的最大绝对误差为 `1.4761553757125512e-7`，RMS 误差为 `6.22090207119758e-9`。音轨总和一致不意味着分离没有串音；吉他和钢琴等低能量轨仍可能主要是残留。

最终 WAV SHA256：`2fae3f11a08649180c70b18b981cb4fc1b7bf20ab743ce0239eb6d5f342a1843`；MP3 SHA256：`56f569b81ffe4f6c7c39a574da1f9579491a64d398320300a114180726838247`。完整交付约 2.9 GiB，音频和 MIDI 保留在服务器，不上传 Git。

附加诊断中，换声前后共同有声帧的绝对音高差中位数为 8.35 cents，99.27% 在 100 cents 内。CAMPPlus 与参考的余弦相似度由 0.343 变为 0.785，但该编码器也由 Seed-VC 使用，不是独立音色评价。

**歌词识别仍需重点复核。** Whisper 的原始 CER 为换声前 255.14%、换声后 449.07%，并标记 `high_asr_error_requires_listening`。识别中的插入和重复可使 CER 超过 100%；该值不能当作实际漏词比例，也不能据此认定发音或歌词已达标。人工听音与专业乐谱校对仍待完成。

## 历史记录说明

以下记录对应早期代码与环境，不含当前全部评审阶段。所列旧运行、`demo_verified` 和 `sun_yanzi_transposed_verified` 便捷链接已经清理；编号、哈希和当时的测量结果仅作为历史记录，不能当作现存下载入口。当前结果以以上完整运行和实际服务器文件为准。

## 2026-10-22：孙燕姿参考音色《落叶归根》升调

完整运行：`runs/20261022_022204_luo_ye_gui_gen_sun_yanzi_pitch_adapted_15e6f1`；便捷入口：`runs/sun_yanzi_transposed_verified`。输入、曲风、seed 42、30 步换声和 16 秒参考均与旧版 `20261021_112422_luo_ye_gui_gen_rock_jazz_sun_yanzi_d0d44c` 一致；新增 `--transpose auto`。

自动选调基于分离后的原曲人声与参考歌声，选择 +3 个半音（Ab → B）。12 项测试通过，覆盖跨小节连音、跨八度临时变音、调号、和弦及转位、变拍、音域越界、低音匹配与高音约束，以及原有音频和缓存测试。测试命令：`PYTHONPATH=src envs/control/bin/python -m unittest discover -s tests -v`。

实际 9 阶段全部成功，耗时合计 262.84 秒；整曲 307.278667 秒、48 kHz、立体声、PCM 24 bit，峰值工程估计 −1.10 dBTP，生成无截断。最终 WAV SHA256：`d7ec38201b8830b7e38f033c97712384eba6c571cb6a5c28cfd0b1b9f838607b`。MP3 已完整解码检查。

原始转谱与旧版哈希一致；移调后逐事件检查了 372 个人声音符、99 个器乐音符，全部升 3 个半音，起点、时值、拍号和速度不变。原谱、移调谱和选调依据均随结果交付。

RMVPE 实测换声前全曲中位音高 236.75 → 282.63 Hz（约 +3.07 个半音）；最终人声 30–60 秒中位音高 165.21 → 206.28 Hz。独立 pYIN 的 40–56 秒检查为 164.74 → 205.17 Hz。新版换声前后共同有声帧中绝对音高差中位数 10.01 cents，99.19% 不超过 100 cents。新旧重新演唱的时长不同，统计差异不能解释为每个音符精确移调量。

音色自然度和粗糙程度尚未人工试听验收。前奏 0–30 秒换声前没有达到分析门限的有声帧，换声后约 2 秒，提示可能仍有额外声响；此项未通过升调修复。当前 16 秒参考只支持样本音区估计，不代表歌手完整音域。

验收入口：新旧试听对照（历史路径：`runs/sun_yanzi_transposed_verified/deliverables/pitch_comparison/listen.html`）、详细报告（历史路径：`runs/sun_yanzi_transposed_verified/deliverables/report.md`）。独立复核脚本为 `scripts/verify_transposition.py`；结果及窗口统计保存在该运行的 `deliverables/pitch_comparison/`。

## 2026-10-21：首次完整演示记录

验证日期：2026-10-21。成功运行目录：`runs/20261021_100959_tonight_awake_folk_df5b52`，便捷入口：`runs/demo_verified`。

## 端到端执行

执行命令：`bash scripts/run_baseline.sh --demo`。这次运行的六个阶段均从输入开始执行成功，没有复用之前试运行的模型结果。其后另行执行 `--resume` 检查缓存恢复。

| 阶段 | 耗时秒，包含进程启动与模型加载 |
| --- | ---: |
| 输入准备 | 1.34 |
| 整首旋律转录 | 26.83 |
| YuE2 换风格 | 82.69 |
| 人声伴奏分离 | 18.25 |
| 整首参考音色转换 | 27.82 |
| 对齐与混音 | 10.20 |
| 合计 | 167.13 |

输入为 203.48 秒的官方 AI 生成歌曲及配套歌词；输出为 204.718667 秒。目标参考为公开 LibriSpeech 语音，不是用户目标人物。

最终音频：48 kHz、2 声道、PCM 24 bit，9,826,496 帧。文件 SHA256：`b6361d76669a643b7f83cfa39391416a6a072afc91cf3cbe03362559730c6588`。

YuE2 的 `abc` 和 `semantic` 截断标记均为 false。换声输出相对分离人声短 11.56 ms，混音时补齐尾部；三个对齐窗口估计延迟为 10 / 0 / 0 ms，中位数为 0，没有施加额外固定平移，也没有对整首进行拉伸。

最终峰值的 4 倍过采样工程估计为 −1.10 dBTP，无越界采样或非有限数值。所有音频可解码且非空。具体数值在 `deliverables/checks.json`；这不是认证级真峰值测量。

## 额外检查

六项针对歌词重复时间戳、重采样时间点、固定延迟恢复、混音峰值和时长、缓存损坏重建、失败状态处理的测试均通过。测试命令：`PYTHONPATH=src envs/control/bin/python -m pytest tests -q`。

20 秒换声探针也已执行，产物在 `runs/probes/seed_20s/`，输出 19.992381 秒。Seed-VC 的有效参数已核对，未发现缺失或尺寸不匹配的可训练参数；上游日志中的跳过项属于未使用的输出分支和位置缓存，详情在 `configs/seed_checkpoint_audit.json`。

整首音高、ASR 和音色诊断由 `scripts/verify_result.py` 执行，结果在 `deliverables/verification.json`。音色诊断使用了换声模型也在使用的 CAMPPlus，不能作为独立测评。

实测音高绝对差中位数为 8.56 cents；共同有声帧中 91.91% 的差异不超过 100 cents，7.44% 超过 600 cents，后者需要试听复核，也可能包含音高估计的倍频误差。CAMPPlus 与参考录音的平均余弦相似度由 0.077 提升到 0.585。

**歌词识别没有通过质量验收。** Whisper-small 相对输入歌词的原始字符错误率为换声前 83.80%、换声后 62.28%。识别文本存在漏识别、同音字和简繁体混用，因此不能将该比例等同于实际漏词比例；应听 `restyled.flac` 和 `final.wav` 逐段核对。报告已将高 ASR 误差和音高异常列为待试听项，没有将这些结果包装为听感合格。

## 实测环境

所有环境为 Python 3.11.13。YuE2 使用 Torch 2.10.0+cu128 / Transformers 4.57.6；转录使用 Torch 2.8.0+cu128 / Transformers 4.45.2；换声使用 Torch 2.8.0+cu128 / Transformers 4.46.3；分离使用 audio-separator 0.30.2 / Torch 2.8.0+cu128。

SheetSage2 使用当前固定版本的 `default` preset，由 FFmpeg 8.0 命令行解码；未采用要求 FFmpeg 6 共享库的 `paper` preset。分离使用 PyTorch CUDA；日志里的 ONNX CPU 提示不代表 BS-RoFormer 在 CPU 执行。

精确环境和上游 commit 在 `configs/environments.lock.json`，各环境局部安装清单为 `configs/*.installed.txt`；模型 revision 和下载后计算的 SHA256 在相应 lock 文件中。上游仓库没有修改。

## 验收边界

首次演示验证了整首技术链路、输出格式、时长和对齐，以及缓存恢复。当时没有进行人工听感验收，也未验证用户目标人物的参考录音或商业歌曲输入；后续歌曲验证见本页前文。所有技术记录均不能据此宣称目标音色、自然度和曲风达到生产要求。

建议按 `USAGE.md` 中的试听顺序，重点听主副歌、高音、间奏前后及结尾。首次演示曾通过 `runs/demo_verified` 验收；该旧入口已清理，现有结果和上传前检查见本页前文。
