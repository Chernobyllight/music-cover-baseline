# 参考人声素材分离

输入原始录音，输出估计总人声、主唱和对应剩余音轨。此入口独立于歌曲主流程 00–11，不训练音色模型，也不创建训练 YAML、数据集切片或修改已有主流程状态。Seed-VC、SVC、RVC、Fish 的训练数据准备都可使用这些音轨，但各自的数据准备入口仍需另行执行。

## 4．从原始音频重新分离单依纯人声

输入是 `/data/liuhongda/music-cover-baseline/data/raw/references/shan_yi_chun/sources/<素材编号>/original.m4a`。默认两级分离：BS-RoFormer 将原始音乐分成总人声和伴奏，Karaoke Mel-Band RoFormer 再将总人声分成估计主唱和剩余部分。保持此前长命令的模型及参数：44.1 kHz WAV，normalization=1.0，amplification=0.0，batch=1，一级 overlap=8，二级 overlap=4，pitch_shift=0。

在 6969 服务器执行（不依赖之前设置的 VC_* 变量，也不要求当前终端位于项目目录）：

```bash
bash /data/liuhongda/music-cover-baseline/scripts/voice_data/separate/run.sh \
  --input-dir data/raw/references/shan_yi_chun/sources \
  --pattern '*/original.m4a' \
  --output-dir data/voice_training/shan_yi_chun/manual_20261102/recordings \
  --exclude-ids 1787247852 6779157198 \
  --gpu 0
```

`1787247852` 是单独保留作参考的《续写》；`6779157198` 是此前没有筛出合格训练窗口的《你的珍藏》。排除规则由本次命令传入，没有写死在代码中。当前 30 份素材中应处理 28 份、排除 2 份。先预览时在同一命令末尾追加 `--plan`：检查匹配文件、排除编号、权重、依赖和 GPU，打印清单，不创建输出、不加载分离模型。

脚本直接使用已有 `/opt/conda/bin/python`，不创建或切换到其他解释器。相对路径以项目根目录解析；也接受绝对路径。`--pattern` 加引号，避免 Shell 提前展开。匹配规则只读取原始录音，不会误读 sources 下旧的 stems。

## 输出及下一步

```text
/data/liuhongda/music-cover-baseline/data/voice_training/shan_yi_chun/manual_20261102/recordings/
├── 1632874979/
│   ├── vocals.wav
│   ├── instrumental.wav
│   ├── lead.wav
│   ├── backing.wav
│   ├── separation.json
│   ├── command.json
│   └── process.log
├── 其他素材编号/
├── manifest.json
└── process.log
```

终端打印每份素材的开始、完成或失败及日志路径。整批摘要见 `manifest.json`，成功时 `status=succeeded`，命令退出码为 0；任一失败时退出码非零。子进程详细日志在对应素材目录；尚未成功的工作目录位于 `.work/`，不会生成顶层 `<素材编号>/lead.wav`。最终目录还保存实际执行命令、工作请求和阶段进度，命令中的临时目录为当时执行位置。

检查摘要：

```bash
/opt/conda/bin/python -m json.tool \
  /data/liuhongda/music-cover-baseline/data/voice_training/shan_yi_chun/manual_20261102/recordings/manifest.json
```

新分离的 `lead.wav` 供下一步生成训练配置时使用。例如：

`/data/liuhongda/music-cover-baseline/data/voice_training/shan_yi_chun/manual_20261102/recordings/1632874979/lead.wav`

下一步运行[第 5 步训练配置生成脚本](../../05_voice_conversion/finetuned/training/create_config/README.md)：传入本次 recordings 目录、验证/测试素材编号及歌曲 run，从空白生成 YAML。随后运行[训练 00_prepare](../../05_voice_conversion/finetuned/training/00_prepare/README.md)，其输入是这份 YAML，其中 sources 指向本次 lead.wav。这里的 `recordings/manifest.json` 是分离记录，不是后续 `dataset/manifest.json`。务必确认整批成功后再生成训练清单，尤其是强制重算失败时不能把保留的旧结果当成本次成功产物。

输出会检查可解码性、有限数值、采样率、与原始音频的时长差（不超过 0.25 秒），并拒绝近乎静音的人声。它不证明身份、主唱纯净度或听感质量。原始素材是有损试听，lead/backing 均为模型估计；训练前需要试听 lead.wav。backing.wav 可能包含混响与泄漏。

## 断点续跑与重新计算

中断或部分失败后，重跑原命令并追加 `--resume`。只有输入音频、模型权重、本地模型 YAML、实现、关键参数和所有音轨哈希都匹配时才跳过成功结果；失败且没有已发布结果的素材重新处理。

如果现有结果缺失、被改动，或输入/模型/参数变了，`--resume` 会报告校验失败，不会假装成功。用 `--force` 明确重新计算：新结果通过检查后，将旧素材目录移到 `.history/` 再发布新结果，不删除旧音频。`--resume` 和 `--force` 互斥。

更改素材集合、匹配规则造成的选择结果或排除列表时，如果选中素材集合改变，应使用新的 output-dir，避免训练程序混入残留素材。不会覆盖不相关文件。不要同时在同一 output-dir 启动两个批次；脚本会加锁拒绝并发写入。

## 参数

| 参数 | 默认值/含义 |
|---|---|
| `--input-dir` | 必填，原始音频目录 |
| `--pattern` | `*/original.m4a`；平铺 WAV 可传 `*.wav` |
| `--output-dir` | 必填，只存放本次分离结果 |
| `--exclude-ids` | 可选；子目录素材编号，平铺文件使用文件名去掉扩展名；无匹配编号会报错 |
| `--gpu` | `0`，直接选择 GPU 编号 |
| `--mode` | `lead`：两级分离；`vocals`：仅总人声与伴奏 |
| `--model-dir` | `models/separation` |
| `--vocals-model` | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` |
| `--lead-model` | `mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` |
| `--lead-overlap` | `4` |
| `--timeout-seconds` | 每份素材 `10800` 秒 |
| `--plan` | 只检查并展示，不写文件、不加载模型 |
| `--resume` | 校验后跳过已完成结果，继续未完成素材 |
| `--force` | 重新计算所选素材，旧结果保留在 `.history/` |

默认权重分别位于 `/data/liuhongda/music-cover-baseline/models/separation/model_bs_roformer_ep_317_sdr_12.9755.ckpt` 和 `/data/liuhongda/music-cover-baseline/models/separation/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt`；需要配套模型 YAML。覆盖模型参数时，使用 audio-separator 支持的、输出 Vocals/Instrumental 的模型，二级模型必须将主唱作为 Vocals 输出。

## 公共数据配置入口

为 RVC 等路线准备全新数据集，可在分离成功后使用[公共数据配置生成脚本](../create_config/README.md)。它不要求 Seed-VC 权重，只生成供现有数据准备 00 使用的 YAML；后续另行生成 RVC 训练配置。

Fish 可直接以本脚本的新 `recordings/manifest.json` 和 `lead.wav` 作为 `training/create_config --recordings-dir` 输入，无需先运行 Seed-VC 的数据配置或准备入口。分离完成后仍须显式生成 Fish 训练 YAML，见 [Fish 手动步骤](../../../docs/24_fish_diffusion_manual.md)。
