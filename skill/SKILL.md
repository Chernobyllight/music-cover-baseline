---
name: music-cover-deploy
description: 在 music-cover-baseline 仓库部署固定版本的模型权重与运行依赖，准备用户自己的歌曲、歌词和目标音色，使用已有 Shell 脚本运行 00 准备及 01–06 音乐生成流程；支持 Seed-VC 零样本、Seed-VC 微调、so-vits-svc、RVC 和 Fish Diffusion 歌手微调，并定位各阶段产物与部署故障。
---

# Music Cover 部署与音乐生成

让使用者从本仓库和自己的素材出发，得到可以试听的曲风、音色转换结果 `final.wav` / `final.mp3`。采用仓库现有脚本，不另写一套推理流程。主范围是 01–06；首次任务必须先做 00。07–11 的分轨、MIDI 和交付打包不是本 Skill 的默认任务。

本 Skill 与 v1.2 源码配套，接口核对日期为 2026-11-05；包含 Fish 独立训练和 05→06 推理。仓库外单独复制本文件不能代替仓库代码、子模块、模型权重和素材。

## 先确定工作位置和路线

1. 找到同时含 `scripts/run_baseline.sh`、`src/music_cover/`、`configs/pipeline.yaml` 的仓库根目录。下文所有命令从该目录执行；`data/`、`models/`、`configs/`、`runs/` 都相对它解析。不要把原服务器用户名、挂载点或旧运行目录带给新用户。
2. 核对代码版本与脚本 `--help`；以当前源码、模型锁文件为准。历史文档中的演示歌手和旧 run 不是部署依赖。不能假设 GitHub 旧分支含本工作树后来增加的脚本。
3. 根据需求选一条 05 路线。只有用户要求训练时才启动训练；已有合格格式的导出包可直接部署。没有人物模型时，可先用 Seed-VC 零样本验证公共主流程，不能把它冒充微调结果。

| 05 路线 | 音色条件 | 必要准备 |
|---|---|---|
| Seed-VC zero_shot | 目标人物参考人声 | 公共 Seed-VC 及配套权重；无需人物训练 |
| Seed-VC finetuned | 人物专用模型 + 参考人声 | 本项目导出的 `voice.json` 及完整目录，或独立完成训练 |
| so-vits-svc | 人物专用 SVC 模型 | SVC 导出目录和公共编码器/F0 权重 |
| RVC | 人物专用 RVC 模型 + 检索索引 | RVC 导出目录、索引和公共编码器/F0 权重 |
| Fish Diffusion | 人物专用扩散声学模型 | Fish 导出目录、ContentVec、NSF-HiFiGAN；无参考音频零样本接口 |

SVC/RVC/Fish 不用参考片段作为推理音色提示，但主流程仍需要该人物的参考人声用于 02 音区分析及质量评估。`voice.json` 是模型档案，不是训练 YAML；不要用一个裸 `.pth` 代替整个导出包。

## 部署顺序

1. 新机器先读 [运行环境与迁移](references/deployment.md)，检查 GPU、解释器、子模块、FFmpeg，再准备依赖。当前代码 **02–06 模型后端使用 `/opt/conda/bin/python`，01 使用 `envs/sheetsage2/bin/python`，部分 Shell 入口还经过 `envs/control/bin/python`**。不要声称所有节点已经只用 base，也不要为 02–06 另建解释器。
2. 按 [模型权重与下载命令](references/weights.md) 下载所选路线需要的资源，保留固定 revision 和校验记录。原始歌曲、歌手参考及人物微调模型不随 Git 发布。
3. 要从录音新训练人物模型时，读 [人物模型训练与导出](references/training.md)，从本次素材生成全新配置和数据，再训练、验证、导出。不要复用作者的历史实验 YAML、切片或 checkpoint。Fish 的独立入口和参数见该参考的 Fish 小节；有完整数据布局、每步输入输出需求时再读 [Fish 手动实操](../docs/24_fish_diffusion_manual.md)，将其中服务器素材编号替换为用户真实录音编号。
4. 完成下面的输入准备、00–06 和结果检查。只读预检通过不等于真实推理通过；最终交付要有实际音频和当前运行证据。

## 输入准备：使用自己的三个文件

以下为统一示例命名，不是仓库自带数据。执行前放入真实文件，可修改文件名；不要创建空文件凑齐输入。

| 文件 | 内容与用途 |
|---|---|
| `data/raw/songs/my_song.mp3` | 完整源歌曲，供 00 解码、01 转谱和 02 音高分析 |
| `data/raw/lyrics/my_song.txt` | UTF-8 歌词，按实际演唱顺序整理；供 03 使用 |
| `data/raw/references/my_voice/reference.wav` | 清晰、单人、尽量少伴奏的目标人声；本例从 0 秒截取 20 秒 |

参考截取时长支持 1–25 秒，起点和时长应落在有声区间。00 不会自动去除参考伴奏；需要从歌曲分离参考时，使用 [录音分离脚本](../scripts/voice_data/separate/README.md)。自动选调的 `reference_kind: singing` 只适用于确认是演唱的参考，讲话/未知素材不要伪标为 singing。

从空白创建本次任务配置，首次应选择未使用的文件和运行目录：

```bash
mkdir -p configs/tasks
test ! -e configs/tasks/my_cover_v001.yaml && (
  set -o noclobber
  cat > configs/tasks/my_cover_v001.yaml <<'YAML'
request:
  name: my_cover_v001
  song: data/raw/songs/my_song.mp3
  lyrics: data/raw/lyrics/my_song.txt
  reference: data/raw/references/my_voice/reference.wav
  reference_start: 0
  reference_duration: 20
  transpose: auto
  style: "upbeat bossa nova and samba jazz, syncopated nylon-string guitar, warm upright bass, piano, lively Brazilian percussion, expressive solo lead vocal"
  seed: 42
  steps: 30
  gpu: "0"
  review: false
modules:
  02_pitch_adaptation:
    reference_kind: singing
    auto_min_semitones: -6
    auto_max_semitones: 6
  05_voice_conversion:
    backend: seed_vc
    mode: zero_shot
YAML
)
```

曲风由使用者决定；示例仅给出一组可用描述。`steps: 30` 是 Seed-VC 推理步数，既不是训练步数，也不是 YuE2 的生成步数。02 默认在 -6～+6 中推荐；可能合理地选择 0，查看报告的“推荐”和“实际采用”，不能保证每首歌都会移调。03 的上下文及长度预算由后端根据谱面计算。

## 主流程：先 00，再依次 01–06

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate base
export PATH="/opt/conda/bin:$PATH"

# 只预检到 06，避免要求范围外的评审环境和模型。
bash scripts/run_baseline.sh --config configs/tasks/my_cover_v001.yaml --until 06_mix --preflight

bash scripts/00_prepare/run.sh --config configs/tasks/my_cover_v001.yaml --run runs/my_cover_v001 --gpu 0
bash scripts/01_source_score/run.sh --run runs/my_cover_v001 --gpu 0
bash scripts/02_pitch_adaptation/run.sh --run runs/my_cover_v001 --gpu 0
bash scripts/03_style_generation/run.sh --run runs/my_cover_v001 --gpu 0
bash scripts/04_vocal_separation/run.sh --run runs/my_cover_v001 --gpu 0

# 本例选择零样本；若选微调、SVC、RVC 或 Fish，替换本条，见下一节。
bash scripts/05_voice_conversion/zero_shot/run.sh --run runs/my_cover_v001 --gpu 0
bash scripts/06_mix/run.sh --run runs/my_cover_v001 --gpu 0
```

所有阶段使用同一个 run。后续节点读取 00 保存的配置；不要把 05 的推理 YAML 当成首次 00 的完整任务配置。独立节点不会自动补跑缺失或失效的上游。

也可在输入和路线已确定后，一次执行到混音：

```bash
bash scripts/run_baseline.sh --config configs/tasks/my_cover_v001.yaml --run runs/my_cover_v001 --until 06_mix --gpu 0
```

上条配置是 zero_shot；微调的一次执行方式须在任务配置中写好对应 `backend`、`mode: finetuned` 和 `voice_profile`，不要在已切换后端的 run 上重新传入零样本配置。

## 05 的使用路线：含零样本共五条，按需选择

以下 profile 必须已经部署/训练导出，不能靠配置生成命令凭空得到。每条路线先在实际 05 推理 `run.sh` 命令末尾加 `--preflight` 检查，再正式运行；配置生成入口不接受该参数。完成任一路线后再执行 06。

**Seed-VC 微调**：`my_voice/v001` 是使用者本次导出的版本；默认生成 3 个候选，明确指定种子：

```bash
bash scripts/05_voice_conversion/finetuned/run.sh \
  --run runs/my_cover_v001 \
  --voice-profile models/voices/my_voice/seed_vc/v001/voice.json \
  --voice-candidates 3 --voice-candidate-seeds 42 43 44 --gpu 0
```

**SVC**：先生成本次推理配置，再运行。参数采用当前通用入口默认值，已优化 profile 可以不显式覆盖这些参数：

```bash
bash scripts/05_voice_conversion/so_vits_svc/create_inference_config/run.sh \
  --voice-profile models/voices/my_voice/so_vits_svc/v001/voice.json \
  --seed 42 --f0-predictor rmvpe --noise-scale 0.4 \
  --chunk-seconds 20 --overlap-seconds 0.5 --pad-seconds 0.5 \
  --output configs/voice_inference/so_vits_svc/my_voice_v001.yaml
bash scripts/05_voice_conversion/so_vits_svc/run.sh \
  --run runs/my_cover_v001 \
  --config configs/voice_inference/so_vits_svc/my_voice_v001.yaml --gpu 0
```

**RVC**：创建配置时明确候选数；示例使用本工程的 `index_rate=0.5`，不是宣称这是上游所有入口的默认值：

```bash
bash scripts/05_voice_conversion/rvc/create_inference_config/run.sh \
  --voice-profile models/voices/my_voice/rvc/v001/voice.json \
  --candidate-count 1 --seed 42 --index-rate 0.5 --protect 0.33 \
  --output configs/voice_inference/rvc/my_voice_v001.yaml
bash scripts/05_voice_conversion/rvc/run.sh \
  --run runs/my_cover_v001 \
  --config configs/voice_inference/rvc/my_voice_v001.yaml --gpu 0
```

**Fish Diffusion**：先有歌手导出包，再创建新推理 YAML；参考只用于评估，05 不额外移调。参数采用已验证的 PLMS 方案：

```bash
bash scripts/05_voice_conversion/fish_diffusion/create_inference_config/run.sh \
  --voice-profile models/voices/my_voice/fish_diffusion/v001/voice.json \
  --output configs/voice_inference/fish_diffusion/my_voice_v001.yaml \
  --sampler-interval 10 --noise-predictor plms \
  --chunk-seconds 8 --overlap-seconds 0.2 --pad-seconds 0.5 --seed 42
bash scripts/05_voice_conversion/fish_diffusion/run.sh \
  --run runs/my_cover_v001 \
  --config configs/voice_inference/fish_diffusion/my_voice_v001.yaml --gpu 0
bash scripts/06_mix/run.sh --run runs/my_cover_v001 \
  --config configs/voice_inference/fish_diffusion/my_voice_v001.yaml --gpu 0
```

Fish 公共预训练模型不是可直接按任意歌手 ID 推理的零样本人物库。`validate/run.sh` 的整曲对比为可选报告，05→06 已产生最终歌曲；不为了生成 MP3 强制用户比较其他模型。

对比多条路线时，在 00–04 成功后用 `bash scripts/05_voice_conversion/fork_run.sh --source-run runs/my_cover_v001 --run runs/my_cover_rvc_v001` 建立新 run，再对该新 run 执行 05、06。fork 引用源 run 的文件，必须保留源 run；不要复制残缺状态或删掉源 run。

## 各阶段看哪里

下表前缀为 `runs/my_cover_v001/stages/`。`attempt_001` 只表示首次尝试；实际使用 `stage_state.json` 中当前成功的 attempt，不靠最大编号猜测，不把失败尝试当成果。

| 节点 | 读取上游 | 首次成功时的主要产物路径 |
|---|---|---|
| 00_prepare | 源歌、歌词、参考 | `00_prepare/attempt_001/artifacts/source.wav`、`reference.wav`、`lyrics.txt` |
| 01_source_score | 00/source.wav | `01_source_score/attempt_001/artifacts/score.abc`、`events.json` |
| 02_pitch_adaptation | 01 乐谱、00 音频 | `02_pitch_adaptation/attempt_001/artifacts/transposition.json`、`score_original.abc`、`score_transposed.abc` |
| 03_style_generation | 02 调整后乐谱、00 歌词、style | `03_style_generation/attempt_001/artifacts/generated/audio.flac`、同级 `generation_budget.json` |
| 04_vocal_separation | 03/audio.flac | `04_vocal_separation/attempt_001/artifacts/vocals.wav`、`instrumental.wav` |
| 05_voice_conversion | 04 人声、选定人物/公共模型 | `05_voice_conversion/attempt_001/artifacts/converted.wav`、`conversion.json`；微调另有 `candidates.json` / `candidates.html` |
| 06_mix | 04 伴奏、人声时间轴，05 转换人声 | `06_mix/attempt_001/artifacts/final.wav`、`final.mp3`、`mix.json` |

03 的 FLAC 是换曲风后的新编曲，尚未经过目标音色转换。最终试听 06；只做到 06 时没有顶层 `deliverables/`，那是 11 的产物。

## 运行检查、重跑与交付

```bash
bash scripts/run_baseline.sh --run runs/my_cover_v001 --status
# 仅需要重新生成 05 时，在所选的 05 命令末尾追加 --force；之后重新执行 06。
# 例如零样本：
bash scripts/05_voice_conversion/zero_shot/run.sh --run runs/my_cover_v001 --gpu 0 --force
bash scripts/06_mix/run.sh --run runs/my_cover_v001 --gpu 0
```

- 同一命令重跑会校验输入、代码、参数和产物哈希后复用缓存；`--force` 新建 attempt，旧文件保留。不要手工改写成功产物以绕过校验。
- 看当前 attempt 的 `command.json` 确认实际解释器，`process.log` 查模型错误，`checks.json` / `outputs.json` 查产物与哈希。配置在 run 的 `request.json` 和 `config.resolved.json`。
- 验收未通过仍保留可解码转换音频和混音；区分 `selected_id`（默认输出）与 `recommended_id`（合格推荐）。没有合格候选时报告待复核，不能声称最高 ECAPA 必然最好听。模型加载失败、损坏输入等执行错误仍须修复。
- 新部署至少实际完成一个用户授权的 00–06 任务，检查 WAV/MP3 可完整解码，试听 04/05/06 对照。仅运行 `--help`、`--plan` 或 `--preflight` 时，如实称为静态检查/预检。
- 向用户报告所用代码版本、路线、模型版本、实际音高、成功 run/attempt、最终音频路径、候选指标和待复核项。不要声称已训练/推理而只完成了环境准备。

常见部署错误和迁移边界见 [deployment.md](references/deployment.md#故障定位)。
