# 05_voice_conversion · 转换目标音色

默认及本页原有命令为 Seed-VC；新增 SVC 歌手模型路线见文末及 [独立入口](so_vits_svc/README.md)。
用 Seed-VC 将生成歌曲的人声转换为参考录音的目标音色。默认保留零样本模式，并新增独立的微调模型入口。

- 零样本：`bash scripts/05_voice_conversion/zero_shot/run.sh --run runs/lhd_test_modular`。
- 微调推理：`bash scripts/05_voice_conversion/finetuned/run.sh --run runs/lhd_test_modular --voice-profile models/voices/sun_yanzi/seed_vc/smoke_v001/voice.json`。
- 训练、验证、导出：见 [音色微调手册](../../docs/13_voice_adaptation.md)。微调推理入口不会触发训练。

## 使用示例

以下示例统一使用“孙燕姿参考音色＋秋意浓＋Synthwave / Dance-pop”，运行目录为 `runs/qiu_yi_nong_modular_test`。各模块使用同一个 `--run`，串联同一首歌曲的中间产物。

前置条件：同一个运行目录中，下列直接依赖及其上游已成功完成，且产物和参数仍有效：[00_prepare](../00_prepare/README.md)、[04_vocal_separation](../04_vocal_separation/README.md)。

本例沿用 00 创建的运行目录，自动读取其中已保存的请求与模块配置。只执行当前模块，不自动补跑上游。

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"

bash scripts/05_voice_conversion/run.sh \
  --run runs/qiu_yi_nong_modular_test
```

示例继承 steps=30、seed=42。旋律移调已由 02 处理；本模块保持 Seed-VC 半音偏移为 0，关闭其自动 F0 移调。输出在 06 中再与伴奏对齐。

## 读取哪些输入

以下路径相对于 `runs/qiu_yi_nong_modular_test/stages/`。以各模块首次成功尝试 `attempt_001` 为例；重跑后，程序按运行状态和成功清单自动读取匹配的 attempt，无需手动传文件路径。

| 上游模块 | 中间产物路径 | 本模块用途 |
| --- | --- | --- |
| `04_vocal_separation` | `04_vocal_separation/attempt_001/artifacts/vocals.wav` | 待转换的人声 |
| `00_prepare` | `00_prepare/attempt_001/artifacts/reference.wav` | 孙燕姿参考片段 |

## 本模块输出

首次执行的输出目录：`runs/qiu_yi_nong_modular_test/stages/05_voice_conversion/attempt_001/artifacts/`。

| 文件 / 目录 | 内容及下游用途 |
| --- | --- |
| `conversion.json` | 模式、音色版本、模型与输入哈希、推理参数 |
| `converted.wav` | 转换后人声；供 06 对齐和混音 |
| `source_mono.wav` | 实际送入换声模型的 44.1 kHz 单声道人声 |

## 查看记录与重跑

每次尝试在 artifacts 同级保存 `inputs.json`（本次实际读取的完整路径、生产模块与哈希）、`config.json`、`command.json`、`process.log`、`checks.json` 和成功后的 `outputs.json`。有多次尝试时，以运行目录的 `stage_state.json` 及其中记录的 attempt 为准。

重复执行示例命令会复用本模块已有的有效结果；在命令末尾加 `--force` 可只重跑本模块并创建新 attempt。上游缺失或失效时，按报错提示从相应上游模块开始执行。

本模块完成后，如需从这里继续到最终交付，在同一项目根目录执行：

```bash
bash scripts/run_baseline.sh --run runs/qiu_yi_nong_modular_test --from 05_voice_conversion
```

总入口先校验并复用有效的本模块及上游结果，再执行后续阶段。

业务实现：`src/music_cover/modules/voice_conversion.py`。默认模块配置：`configs/modules/05_voice_conversion.yaml`。已有运行保留初始化时的配置；调整已有运行请通过 `--config` 显式传入任务配置覆盖。

完整依赖、缓存及失败恢复说明见 [模块化说明](../../docs/09_modular_pipeline.md)。

另一个完整实例：[孙燕姿＋秋意浓＋Bossa Nova / Samba Jazz，输出 runs/lhd_test_modular](../../docs/11_modular_walkthrough.md)。该手册包含从 00 到 11 的每条命令、上下游路径和评估解读。

## 微调推理的三个候选

微调入口默认生成三个完整候选（seed=42 时为 42、43、44），使用同一目标音色模型、源人声、参考片段和推理参数。05 保存干声与逐项指标，06 分别混音，11 交付三个 WAV/MP3。仅技术检查合格的候选参与 ECAPA 排名，推荐版本进入分轨、MIDI 和最终交付入口；零样本入口仍为单次推理。

具体命令与中间文件路径见[三个候选与推荐结果使用说明](../../docs/15_voice_candidates.md)。


## 新增 so-vits-svc 4.1 后端

支持独立歌手微调与 05 音色转换，复用已有 base 环境；Seed-VC 零样本与微调入口继续保留。SVC 目前一个候选，不支持零样本。训练数据、权重清单、完整示例及中间产物见 [SVC 接入指南](../../docs/18_so_vits_svc.md)。

## RVC 已训练歌手模型

读取同一 run 的 04 分离人声，写入 05 的 `converted.wav`、候选指标和模型哈希，供 06 混音使用。

```bash
cd /data/liuhongda/music-cover-baseline
bash scripts/05_voice_conversion/rvc/run.sh \
  --run runs/my_lv_zhou_huang_qi_shan_rvc \
  --config configs/voice_inference/rvc/huang_qi_shan.yaml --gpu 0
```

新 run 应先运行 00–04 或按 [RVC 指南](../../docs/20_rvc.md)从现有 Seed-VC run 创建独立副本。模型训练入口位于 `rvc/training/00_prepare` 至 `05_export`，固定使用训练配置的最终步数模型，不要求用户比较检查点。
