# 音乐曲风与音色转换 baseline

从原曲、歌词和参考人声生成新编曲，支持 Seed-VC 零样本/微调、so-vits-svc、RVC 和 Fish Diffusion 歌手模型，输出最终 WAV、MP3、分轨、MIDI 和评审资料。

当前源码版本为 **v1.2**，新增 Fish Diffusion 歌手微调、5000 步训练方案及完整手动验证指南；保留 Seed-VC 零样本/微调、SVC 和 RVC。部署请获取 v1.2 分支及其固定上游版本。交付范围与验收证据见 [交付指南](docs/21_delivery.md) 和 [验证记录](VALIDATION.md)。

## English overview

Modular music restyling and voice conversion with YuE2, Seed-VC, so-vits-svc, RVC and Fish Diffusion, featuring reproducible training, staged inference, audio review and deployment guidance.

Version 1.2 adds Fish Diffusion singer fine-tuning with a validated 5,000-step training recipe, shell-based configuration tools, and an end-to-end manual guide from raw recordings to converted vocals and final mixes. Seed-VC zero-shot and fine-tuned conversion, so-vits-svc, and RVC remain available. Generated audio remains available for review even when quality checks do not pass. Models and media are provisioned separately; fresh-machine deployment still requires environment validation.

## 从哪里开始

| 需要做什么 | 文档 |
|---|---|
| 新使用者部署并运行 00–06 | [部署 Skill](skill/SKILL.md) |
| 直接生成、分步检查中间文件 | [使用说明](USAGE.md) |
| 理解整个流程与模块职责 | [当前架构](docs/01-architecture.md)、[模块化说明](docs/09_modular_pipeline.md) |
| 准备环境与模型权重 | [环境说明](docs/03-environment.md)、[权重清单](docs/12_model_weights.md) |
| 训练歌手模型 | [Seed-VC](docs/13_voice_adaptation.md)、[SVC](docs/18_so_vits_svc.md)、[RVC](docs/20_rvc.md)、[Fish Diffusion](docs/22_fish_diffusion.md) |
| 从原始录音亲自验证 Fish 训练→05→06 | [手动实操与输入输出](docs/24_fish_diffusion_manual.md)、[训练优化依据](docs/23_fish_diffusion_optimization.md) |
| 候选推荐与指标含义 | [候选说明](docs/15_voice_candidates.md)、[评估说明](docs/05-evaluation.md) |
| 交付、回归验证、数据保留、迁移 | [交付指南](docs/21_delivery.md)、[仓库说明](docs/08-repository.md) |

## 当前流程

```text
00 准备音频和歌词 → 01 原曲转谱 → 02 分析音区并移调乐谱
→ 03 YuE2 生成新曲风 → 04 分离新曲的人声与伴奏
→ 05 Seed-VC / SVC / RVC / Fish Diffusion 换声 → 06 对齐混音（WAV + MP3）
→ 07 成品分轨 → 08 成品转谱 → 09 各轨 MIDI
→ 10 技术评估 → 11 发布交付目录
```

05 共四种模型后端、五条使用路线：Seed-VC 零样本/微调，以及 SVC、RVC、Fish 的歌手模型。只有 Seed-VC 提供本工程的参考音频零样本路线。训练位于 05 对应后端的 `training/`，与每首歌的推理流程独立。`--no-review` 只跳过 07–09，仍执行 10–11。

## 已配置服务器的完整示例

本机工程 `/data/liuhongda/music-cover-baseline` 解析到 `/mnt/afs_f/liuhongda/music-cover-baseline`。下面用现有黄绮珊 Seed-VC 微调模型，生成《穿越曙光》管弦灵魂乐；输入路径、歌词与参考片段均写在任务配置中。

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
bash scripts/run_baseline.sh \
  --config configs/tasks/chuan_yue_shu_guang_huang_qi_shan_seedvc_20261029.yaml \
  --preflight
bash scripts/run_baseline.sh \
  --config configs/tasks/chuan_yue_shu_guang_huang_qi_shan_seedvc_20261029.yaml \
  --run runs/my_chuan_yue_shu_guang_seedvc --gpu 0
```

最终音频：`runs/my_chuan_yue_shu_guang_seedvc/deliverables/final.mp3`。每阶段中间文件在 `stages/<阶段>/attempt_NNN/artifacts/`，当前有效 attempt 以 `stage_state.json` 为准。具体逐阶段命令见 [USAGE](USAGE.md)。

## 项目检查

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
/opt/conda/bin/python scripts/check_project.py --output cache/source_check.json
PYTHONPATH=src envs/separator/bin/python -m unittest discover -s tests -v
```

检查语法、配置、文档链接和测试不等于 GPU 推理通过；完整验收另检查真实生成和交付文件。模型、训练数据及运行结果保存在服务器，不随源码包或 Git 提交分发。下载与迁移依赖各模型锁文件及实际环境快照，不能把现有 `envs/` 直接复制到另一台机器。

曲风生成可能改动演唱细节，分轨与 MIDI 均为估计。ECAPA 不代表身份百分比或主观音质；技术完成后的音乐仍需人工试听和乐谱校对。

Fish 的完整手动步骤见 [实操手册](docs/24_fish_diffusion_manual.md)：先分离目标人物录音，再显式创建训练 YAML，依次准备数据、提取特征、训练、验证、导出，最后创建推理配置并运行 05 和 06。只做到 06 已得到最终 WAV/MP3；整曲对比报告为可选步骤。
