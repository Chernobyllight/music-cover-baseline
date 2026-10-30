> 历史验证/实例记录：其中部分运行已按此前要求清理。当前可执行示例见 [USAGE](../USAGE.md)，当前保留与验收范围见 [交付指南](21_delivery.md)。不要把历史路径当作现有下载入口。

# 音色方案收敛验证记录（2026-10-27）

历史验证记录：数值和路径描述当日实验；后续按用户要求清理过运行数据，不能假定下列目录全部仍存在。当前命令与保留结果见 [使用说明](../USAGE.md) 和 [验证记录](../VALIDATION.md)；[v0.3 实例](17_v03_walkthrough.md)保留为历史实操。

删除单曲伪标签适配实现，保留零样本和目标音色通用微调。微调推理默认三个候选，以同一模型和共同输入仅改变推理种子，技术合格后按 ECAPA 选择推荐版本。

## 自动测试

在已有的 separator 环境执行（control 环境未安装评审测试所需 mido）：

```bash
cd /data/liuhongda/music-cover-baseline
PATH="/opt/conda/bin:$PATH" PYTHONPATH=src \
  envs/separator/bin/python -m unittest discover -s tests -v
```

78 项通过，包括种子约束、零样本无候选评估依赖、旧单曲 profile 拒绝、音高/时长/峰值/静音门槛、非合格高 ECAPA 不入选、无合格候选、真实 WAV/MP3 三候选混音、推荐音频哈希一致、候选交付和路径检查，以及现有模块、音高、MIDI 和音色训练回归。更新文档的 Bash 示例通过 bash -n 检查。

## 实际 GPU 验证配置

继承 `runs/lhd_voice_finetune_validation/config.resolved.json` 的原任务：孙燕姿 0–16 秒参考、《秋意浓》、upbeat bossa nova and samba jazz、30 步、主 seed=42、transpose=auto（02 原结果 -2 半音）。不重训模型，不重生成 03 风格音频。

00–04 成功 attempt 复制到新的验证运行，并保留原输入/代码/运行环境指纹与输出清单；新流程逐一校验后复用。这里验证的是从相同生成曲出发的音色模块改造，不能宣称重新生成了原曲转谱和风格生成结果。

- 微调运行：`/data/liuhongda/music-cover-baseline/runs/lhd_voice_candidates_validation`。
- 零样本对照：`/data/liuhongda/music-cover-baseline/runs/lhd_zero_shot_convergence_validation`，复用上述同一 00–04，使用 GPU 1、review=false、voice_check=false，验证基础单结果路线。其它歌曲参数相同。
- 微调 profile：`models/voices/sun_yanzi/seed_vc/smoke_v001/voice.json`，短素材集成模型。

微调 05 实测：

| 候选 | seed | ECAPA ↑ | F0 差中位数（半音） | 误差≤1半音 | 技术检查 | 推荐 |
| --- | --- | --- | --- | --- | --- | --- |
| candidate_001 | 42 | 0.66784066 | +0.01142 | 99.8973% | 通过 | |
| candidate_002 | 43 | 0.68439817 | +0.01916 | 99.4888% | 通过 | 是 |
| candidate_003 | 44 | 0.67681444 | +0.01666 | 99.2828% | 通过 | |

三条原始转换干声均为 397.6301 秒。评分在 05 原始干声上计算，与相同提示参考比较，仅用于运行诊断。推荐以 ECAPA 为排序指标，因此并非选择 F0 误差最小的版本；三者均先通过音高及时间完整性门槛。

零样本同口径补测 ECAPA=0.63457453，音高检查通过，误差≤1半音比例 99.6933%。核验源人声及参考片段 SHA256 与三个微调候选完全一致；评分对象同为原始转换干声。

固定 seed=42 的微调相对零样本提升 0.03326613；三候选推荐版相对零样本提升 0.04982364，其中含候选选择收益。此处使用同一提示录音、短素材 smoke 模型，只是本次运行诊断，不证明独立数据集上的泛化改善，也不替代人工听音。

## 数据清理

已删除已淘汰的单曲伪标签训练检查点 26 个，及其 7 个 latest 符号链接，释放 54.4048 GiB。清理目录限于旧 benchmark 的六个 song_training case 和旧 song_adaptation 测试。所有历史成品、对比报告、原始素材、8 份已导出的模型 profile 及关联文件均保留；通用音色训练检查点也保留，可继续其原训练流程。

精确清单：`cache/maintenance/20261027_runs_cleanup.json`；导出模型独立性/哈希核验：`cache/maintenance/20261027_preserved_models.json`。清理前的代码快照为 `cache/maintenance/voice_convergence_20261027/code_before.tar.gz`，本地也保存副本。

## 完成状态与交付核验

微调运行 12 个阶段均 succeeded，运行状态 completed；00–04 校验复用，05–11 实际重新执行。零样本运行基础 9 个阶段均 succeeded，状态 completed，跳过 07–09 专业评审。

推荐版生成了 177 个 review 文件（包括成品分轨、各声部 MIDI、乐谱和评审材料）。三个候选均有完整 WAV/MP3；交付根目录 final 的 WAV/MP3 哈希与 candidate_002 精确一致。10 的候选指标与最终 candidates.json 完全一致。已核验三个候选使用相同源/参考/模型哈希，steps=30、半音偏移=0、不重新训练。

推荐 MP3：

`/data/liuhongda/music-cover-baseline/runs/lhd_voice_candidates_validation/deliverables/final.mp3`

全部候选：同目录 `candidates/candidate_001/final.mp3`、`candidate_002/final.mp3`、`candidate_003/final.mp3`。试听与定量入口：`candidates.html`；MIDI ZIP：`review/midi_bundle.zip`。

验证证据：`cache/maintenance/20261027_voice_convergence_validation.json`；零样本同口径补测：`cache/maintenance/20261027_zero_voice_metrics/voice_conversion_evaluation.json`。历史六组比较链接仍完整。人工听感验收待完成。
