# 11_delivery · 打包并发布交付结果

汇总各模块已验证的产物，生成试听页、音频及专业评审包，校验成功后发布 deliverables。

## 使用示例

以下示例统一使用“孙燕姿参考音色＋秋意浓＋Synthwave / Dance-pop”，运行目录为 `runs/qiu_yi_nong_modular_test`。各模块使用同一个 `--run`，串联同一首歌曲的中间产物。

前置条件：同一个运行目录中，下列直接依赖及其上游已成功完成，且产物和参数仍有效：[00_prepare](../00_prepare/README.md)、[01_source_score](../01_source_score/README.md)、[02_pitch_adaptation](../02_pitch_adaptation/README.md)、[03_style_generation](../03_style_generation/README.md)、[04_vocal_separation](../04_vocal_separation/README.md)、[06_mix](../06_mix/README.md)、[10_evaluation](../10_evaluation/README.md)、[07_stem_separation](../07_stem_separation/README.md)、[08_final_score](../08_final_score/README.md)、[09_stem_midi](../09_stem_midi/README.md)。

本例沿用 00 创建的运行目录，自动读取其中已保存的请求与模块配置。只执行当前模块，不自动补跑上游。

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"

bash scripts/11_delivery/run.sh \
  --run runs/qiu_yi_nong_modular_test
```

05 的转换后人声经 06 对齐后进入交付；11 直接读取 06 的对齐结果。--no-review 时不依赖 07–09，也不创建 review 评审包。发布前的任何失败都会保留之前已发布的交付结果。

## 读取哪些输入

以下路径相对于 `runs/qiu_yi_nong_modular_test/stages/`。以各模块首次成功尝试 `attempt_001` 为例；重跑后，程序按运行状态和成功清单自动读取匹配的 attempt，无需手动传文件路径。

| 上游模块 | 中间产物路径 | 本模块用途 |
| --- | --- | --- |
| `00_prepare` | `00_prepare/attempt_001/artifacts/` | 全部登记产物；包含原曲、参考片段、歌词 |
| `01_source_score` | `01_source_score/attempt_001/artifacts/` | 全部登记产物；包含原曲乐谱及转谱数据 |
| `02_pitch_adaptation` | `02_pitch_adaptation/attempt_001/artifacts/` | 全部登记产物；包含移调乐谱和计划 |
| `03_style_generation` | `03_style_generation/attempt_001/artifacts/` | 全部登记产物；包含生成歌曲及生成请求 |
| `04_vocal_separation` | `04_vocal_separation/attempt_001/artifacts/` | 全部登记产物；包含换声前人声及伴奏 |
| `06_mix` | `06_mix/attempt_001/artifacts/` | 全部登记产物；包含最终 WAV/MP3、对齐人声、增益及残差 |
| `07_stem_separation` | `07_stem_separation/attempt_001/artifacts/` | review=true 时纳入全部分轨产物 |
| `08_final_score` | `08_final_score/attempt_001/artifacts/` | review=true 时纳入全部成品乐谱产物 |
| `09_stem_midi` | `09_stem_midi/attempt_001/artifacts/` | review=true 时纳入全部 MIDI 产物 |
| `10_evaluation` | `10_evaluation/attempt_001/artifacts/` | 全部登记产物；包含技术评估和可选的 F0 数据 |

## 本模块输出

首次执行的输出目录：`runs/qiu_yi_nong_modular_test/stages/11_delivery/attempt_001/artifacts/`。

| 文件 / 目录 | 内容及下游用途 |
| --- | --- |
| `final.wav` | 最终 WAV |
| `final.mp3` | 最终 MP3 |
| `listen.html` | 试听入口 |
| `report.md` | 运行报告 |
| `evaluation.json` | 技术评估副本 |
| `config.resolved.json` | 本次请求和模块配置 |
| `review/index.html` | review=true 时的专业评审入口 |
| `review/midi_bundle.zip` | review=true 时的 MIDI 下载包 |
| `delivery_validation.json` | 交付包检查结果 |

全部交付校验通过后，`runs/qiu_yi_nong_modular_test/deliverables/` 指向本次成功的 artifacts；可从中读取音频和页面。人工评分请填写 `runs/qiu_yi_nong_modular_test/reviews/review_scores.csv`，这是发布时创建的可编辑副本，后续重发不会覆盖已填写内容。

## 查看记录与重跑

每次尝试在 artifacts 同级保存 `inputs.json`（本次实际读取的完整路径、生产模块与哈希）、`config.json`、`command.json`、`process.log`、`checks.json` 和成功后的 `outputs.json`。有多次尝试时，以运行目录的 `stage_state.json` 及其中记录的 attempt 为准。

重复执行示例命令会复用本模块已有的有效结果；在命令末尾加 `--force` 可只重跑本模块并创建新 attempt。上游缺失或失效时，按报错提示从相应上游模块开始执行。

业务实现：`src/music_cover/modules/delivery.py`。默认模块配置：`configs/modules/11_delivery.yaml`。已有运行保留初始化时的配置；调整已有运行请通过 `--config` 显式传入任务配置覆盖。

完整依赖、缓存及失败恢复说明见 [模块化说明](../../docs/09_modular_pipeline.md)。

启用移调分析时，交付还包含 pitch_report.html、pitch_candidates.csv、score_audio_consistency.json、score_transposed.mid；启用音高核验时包含 pitch_adherence.json。相同证据会进入 review/pitch_adaptation 和 midi_bundle.zip。此处 MIDI 是 02 的输入旋律谱；review/midi 中的 MIDI 才是成品音频重新转谱，二者不能混用。

另一个完整实例：[孙燕姿＋秋意浓＋Bossa Nova / Samba Jazz，输出 runs/lhd_test_modular](../../docs/11_modular_walkthrough.md)。该手册包含从 00 到 11 的每条命令、上下游路径和评估解读。

## 微调三个候选

当 05 采用 finetuned 三候选时，本阶段也保留三个版本；根目录 final 为推荐版本。具体例子：

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
bash scripts/11_delivery/run.sh --run runs/lhd_candidates_example
```

需要同一运行的前置模块已完成。读取的候选来自 06 的 candidates.json 及三个候选混音。本模块输出在 `runs/lhd_candidates_example/stages/11_delivery/attempt_001/artifacts/`，查看 `candidates.html`、`candidates.json` 和 `candidates/candidate_001/final.mp3`（002、003 同理）。实际 attempt 由 stage_state.json 决定。分轨与 MIDI 对应推荐版本。完整顺序见[三候选指南](../../docs/15_voice_candidates.md)。
