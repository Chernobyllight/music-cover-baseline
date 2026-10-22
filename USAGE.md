# 使用说明

项目位置：`/data/liuhongda/music-cover-baseline`。已有的环境、权重和演示数据可重复使用，无需训练目标人物音色模型。

已有完整结果可直接验收：[试听页](runs/demo_verified/deliverables/listen.html)、[最终音频](runs/demo_verified/deliverables/final.wav)、[运行报告](runs/demo_verified/deliverables/report.md)。完整推理约 3 分钟（本机 H200 实测；不含安装、下载和附加评价），不必先重跑才能听。

实际链路：原曲 → SheetSage2 提取旋律 → YuE2 按指定风格重生成 → BS-RoFormer 分离 → Seed-VC 参考音色转换 → 对齐、混音。

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

当前脚本为本机 H200 配置，YuE2 的显存预算设为 80 GiB。不要直接将这份配置套在 24 GB 显卡上；小显存部署需要调整预算并重新验证。

换风格是根据旋律和歌词重新生成音乐，时长、演唱细节和编曲可能变化，不能当作原音频的严格逐帧编辑。换音色和最终混音则保持新生成歌曲的时间轴。

YuE2 默认的 9000 个语义 token 约对应 6 分钟。当前脚本会按乐谱时长自动提高生成上限，并预留结尾余量；例如《秋意浓》使用 12000 token 上限（约 8 分钟容量），模型仍可提前正常结束，不会强制生成 8 分钟。实际设置保存在 `02_cover/attempt_*/generation_budget.json` 及原生生成配置中。若仍出现 `semantic: True`，该次输出属于截断结果，不应作为完整歌曲验收；脚本保留此检查。

## 按参考歌声音区升调：孙燕姿《落叶归根》

已生成一版完整结果，便捷入口：`runs/sun_yanzi_transposed_verified/deliverables/`。
直接听 [最终 MP3](runs/sun_yanzi_transposed_verified/deliverables/final.mp3)、[最终 WAV](runs/sun_yanzi_transposed_verified/deliverables/final.wav)，或打开 [新旧试听对照](runs/sun_yanzi_transposed_verified/deliverables/pitch_comparison/listen.html)。本次完整流程约 4 分 23 秒，输出约 5 分 7 秒。开头主歌实测音高已上移，粗糙度和高音自然度仍需试听。

在原有命令中加 `--transpose auto` 即可。该模式需要清晰的**歌唱**参考片段；不要用说话录音估计歌唱音区。当前孙燕姿参考已是分离出的歌声，无需重新下载。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/run_baseline.sh \
  --song data/raw/songs/luo_ye_gui_gen.mp3 \
  --lyrics data/raw/lyrics/luo_ye_gui_gen.txt \
  --reference data/raw/references/sun_yanzi/reference.wav \
  --reference-duration 16 \
  --transpose auto \
  --style "Mandarin Chinese, energetic modern jazz rock fusion, driving rock drums, groovy electric bass, punchy electric guitars, syncopated rhythms, jazzy piano chords, brass accents, clear solo female lead vocal" \
  --name luo_ye_gui_gen_sun_yanzi_pitch_adapted \
  --gpu 0
```

流程为：分离原曲人声并测音高 + 测参考歌声音区 → 选升调量 → 对原曲乐谱所有旋律声部统一移调 → YuE2 重新生成同调的歌声与伴奏 → 分离 → Seed-VC 换声 → 混音 → 再测生成/换声后的音高。换声阶段不额外移调。

自动模式在 0～6 个半音中选取折中值，优先改善低音区，同时惩罚过高的全曲高音。它使用参考片段的统计音区，不等于识别歌手完整音域；高音余量是实验参数，仍需试听。当前素材的自动选择为 **+3 个半音，Ab → B**。可用 `--transpose 3` 固定升 3 个半音，或 `--transpose 5` 另做对照；手动支持 -12～12。不加此参数与 `--transpose 0` 均执行原有流程。

每次运行仍输出新的 `runs/.../deliverables/`，首先听 `final.wav` 的第一段主歌及高音段。额外文件：

| 文件 | 内容 |
| --- | --- |
| `transposition.json` | 选择的半音数、候选方案、谱面逐音检查 |
| `score_original.abc` / `score_transposed.abc` | 移调前后谱子 |
| `pitch_input.json` | 原曲人声及参考歌声的实测音高 |
| `pitch_output.json` | 新生成及换声后歌声的实测音高，含每 30 秒统计 |

谱面移调通过检查不代表 YuE2 每个音都严格遵循；实际音高和音色自然度需要分别验收。当前改动不处理无歌声处的换声噪音，也不保证升调本身解决全部“粗糙”问题。

如需对下一次新运行制作同样的对照页，生成完成后执行：

```bash
envs/seed-vc/bin/python scripts/verify_transposition.py \
  runs/20261021_112422_luo_ye_gui_gen_rock_jazz_sun_yanzi_d0d44c \
  /绝对路径/runs/新的升调运行目录 --gpu 0
```

脚本检查相同原曲、参考片段、曲风、种子和换声步数，输出 `deliverables/pitch_comparison/listen.html`、片段及音高诊断。对照目录必须尚不存在，以保留已有验收记录。

## 在哪里验收

命令开始时打印 `RUN: ...`，成功后打印 `RESULTS: .../deliverables`。每次新运行使用新的目录。

打开该目录的 `listen.html`，依次播放原曲、换风格、最终混音，并与参考录音比较。也可以直接播放 WAV/FLAC 文件。

| 文件 | 验收内容 |
| --- | --- |
| `original.wav` | 输入原曲 |
| `reference.wav` | 本次实际使用的目标音色片段 |
| `restyled.flac` | YuE2 换风格结果；先确认歌词、旋律、曲风 |
| `reconstruction.wav` | 分离后原人声重新贴回的对照；检查分离损伤 |
| `final.wav` | 最终换音色并混入伴奏的结果，48 kHz、立体声、24 bit WAV |
| `vocals_before.wav` / `vocals_after.wav` | 单独比较换声前后的歌声 |
| `instrumental.wav` | 分离出的伴奏 |
| `report.md` / `checks.json` | 耗时、时长、对齐、峰值等自动检查记录 |

优先听开头、主歌、副歌、高音、间奏前后、结尾，确认没有漏词、乱词、破音、明显人声残留和伴奏错位。自动检查通过表示技术流程完成，不等于听感达标。

## 失败重跑、只换参考人物

保留命令打印的运行目录，失败后查看对应阶段的 `process.log`。修复问题后：

```bash
bash scripts/run_baseline.sh --resume /绝对路径/runs/某次运行
```

更换参考录音并复用已有的旋律、换风格和分离结果：

```bash
bash scripts/run_baseline.sh \
  --resume /绝对路径/runs/某次运行 \
  --reference /绝对路径/new_reference.wav
```

输入、参数、适配代码和产物哈希一致的已完成阶段会复用；受影响阶段会重新执行。旧交付目录保存在 `deliverables_history/`，失败时不会把旧音频标成新结果。

## 可选的附加自动验证

```bash
CUDA_VISIBLE_DEVICES=0 USE_TF=0 USE_FLAX=0 \
  envs/seed-vc/bin/python scripts/verify_result.py /绝对路径/runs/某次运行
```

结果写入 `deliverables/verification.json`，包括换声前后音高差异、Whisper 歌词识别结果及 CAMPPlus 音色相似度诊断。CAMPPlus 同时被换声模型使用，因此该项不是独立评价；ASR 和音高估计也有自身误差，不用这些分数替代试听。

## 文件位置

代码在 `src/music_cover/` 和 `scripts/`，上游仓库在 `third_party/`，权重在 `models/`，隔离环境在 `envs/`，下载缓存和安装日志在 `cache/`。模型 revision 清单在 `configs/`。

本机的虚拟环境复用了 `/opt/conda` 的基础 CUDA 包，不能直接复制 `envs/` 到另一台机器使用。迁移时应按环境记录重新安装，并验证 GPU 推理。旧 `docs/` 中的方案文件保留为设计记录；当前操作以本说明和实际运行报告为准。
