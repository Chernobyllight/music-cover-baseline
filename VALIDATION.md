# 实际验证记录

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
