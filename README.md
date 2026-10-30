# 音乐曲风与音色转换 baseline

从原曲、歌词和参考人声生成新编曲，支持 Seed-VC 零样本/微调、so-vits-svc 和 RVC 歌手模型，输出最终 WAV、MP3、分轨、MIDI 和评审资料。

当前版本 **v1.0**（分支 `v1.0`），包含 Seed-VC 零样本/微调、SVC、RVC、候选推荐和工程整理。交付范围与验收证据见 [交付指南](docs/21_delivery.md) 和 [验证记录](VALIDATION.md)。

## 从哪里开始

| 需要做什么 | 文档 |
|---|---|
| 直接生成、分步检查中间文件 | [使用说明](USAGE.md) |
| 理解整个流程与模块职责 | [当前架构](docs/01-architecture.md)、[模块化说明](docs/09_modular_pipeline.md) |
| 准备环境与模型权重 | [环境说明](docs/03-environment.md)、[权重清单](docs/12_model_weights.md) |
| 训练歌手模型 | [Seed-VC](docs/13_voice_adaptation.md)、[SVC](docs/18_so_vits_svc.md)、[RVC](docs/20_rvc.md) |
| 候选推荐与指标含义 | [候选说明](docs/15_voice_candidates.md)、[评估说明](docs/05-evaluation.md) |
| 交付、回归验证、数据保留、迁移 | [交付指南](docs/21_delivery.md)、[仓库说明](docs/08-repository.md) |

## 当前流程

```text
00 准备音频和歌词 → 01 原曲转谱 → 02 分析音区并移调乐谱
→ 03 YuE2 生成新曲风 → 04 分离新曲的人声与伴奏
→ 05 Seed-VC / SVC / RVC 换声 → 06 对齐混音（WAV + MP3）
→ 07 成品分轨 → 08 成品转谱 → 09 各轨 MIDI
→ 10 技术评估 → 11 发布交付目录
```

训练位于 05 对应后端的 `training/`，与每首歌的推理流程独立。`--no-review` 只跳过 07–09，仍执行 10–11。

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
