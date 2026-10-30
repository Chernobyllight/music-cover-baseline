# 交付、验证与数据保留

本指南对应 v1.0，包含 Seed-VC 零样本/微调、SVC、RVC 和 00–11 模块流程。源码版本由 `VERSION` 和 Git commit 标识；验收记录见 VALIDATION.md，模型/数据仍在服务器。

## 可执行入口

| 范围 | 入口 |
|---|---|
| 完整流程或指定模块 | `bash scripts/run_baseline.sh`、`scripts/00_prepare` 至 `11_delivery` |
| Seed-VC 零样本 / 微调 | `scripts/05_voice_conversion/zero_shot/run.sh`、`finetuned/run.sh` |
| SVC / RVC | `scripts/05_voice_conversion/so_vits_svc/run.sh`、`rvc/run.sh` |
| 独立训练 | 各后端的 `training/`；Seed/SVC 四阶段，RVC 六阶段 |
| 复用编曲 | `scripts/fork_voice_run.py` |
| 源码与文档检查 | `scripts/check_project.py` |
| 已完成运行验收 | `scripts/verify_review.py` |

训练和推理互相独立；Zero-shot 不需要歌手微调权重。正式支持的配置见 [USAGE](../USAGE.md)。历史 SVC 消融脚本位于 `scripts/05_voice_conversion/so_vits_svc/experiments/`，不是普通推理依赖；旧 ASR 诊断隔离在 `scripts/legacy/`。

## 交付验收命令

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
/opt/conda/bin/python scripts/check_project.py --output cache/source_check.json
PYTHONPATH=src envs/separator/bin/python -m unittest discover -s tests -v
bash scripts/run_baseline.sh --preflight
bash scripts/05_voice_conversion/rvc/run.sh \
  --config configs/voice_inference/rvc/huang_qi_shan.yaml --preflight
bash scripts/05_voice_conversion/so_vits_svc/run.sh \
  --config configs/voice_inference/so_vits_svc/yang_kun_optimized.yaml --preflight
```

预检仅确认所需资源与导入。真实验收还需一次完整 00–11、各音色路线的实际音频生成、训练入口与权重完整性检查，并执行：

```bash
PYTHONPATH=src envs/separator/bin/python scripts/verify_review.py \
  runs/project_delivery_validation_20261029/rvc_full \
  --output cache/project_cleanup_20261029/full_delivery_check.json
```

上面的完整验证运行由此次整理生成，最终路径为 `runs/project_delivery_validation_20261029/rvc_full/deliverables/final.mp3`。检查报告写在 cache，不修改已经发布的 artifacts。此次具体通过项与限制在 [VALIDATION](../VALIDATION.md) 的工程整理章节记录。

## 数据依赖和保留规则

| 目录/数据 | 处理方式 |
|---|---|
| `data/raw/` 原曲、歌词、参考素材与来源记录 | 保留，不覆盖原始字节 |
| `data/voice_datasets/` 有效训练集与分组清单 | 保留；重建时使用新版本目录 |
| `models/` 底模与已导出歌手权重 | 保留，按锁文件/profile 校验 |
| `runs/voice_training/` 原训练记录和检查点 | 保留，支持复核与续训 |
| 成功运行的当前 stage_state、各阶段 artifacts、deliverables | 保留；被其他 run 引用的上游不能单独删除 |
| `runs/svc_optimization_yang_kun_20261028/` | 保留选定 SVC 模型的训练与冻结输入、最终结果 |
| `cache/svc_optimization/python_deps/` | 是实验运行依赖，不能按普通缓存清空 |
| 下载失败/已合并分片、过期自测副本、操作系统元数据 | 经依赖核对后可清理，记录路径与字节数 |
| 历史文档 | 移入 docs/archive 或明确标注历史，不冒充当前状态 |

本机原有九组黄绮珊对照（Seed-VC、RVC 0.5、RVC 0.75 各三首）继续保留。RVC 的 fork 使用 Seed-VC 的 00–04 文件；即使其最终 MP3 已复制到桌面，也不能据此删除源 run。

当前没有提供“删除整个 runs/cache”的命令。2026-10-29 清理使用明确清单，完成后核验受保护权重和既有结果，修改前源码另有快照。v1.0 发布代码可按 Git commit 恢复；任何回退前都应备份未提交的本地改动和运行资产。

## 源码、运行资产与迁移

源码交付应包含 src/scripts/tests/configs/docs、顶层说明、四个子模块 commit、检查报告与 SHA256。权重、歌曲、训练集、环境及生成音频不装入源码包；其路径、来源与校验方法见 [权重指南](12_model_weights.md) 和 [仓库说明](08-repository.md)。

现有服务器验收通过不等于任意新机器可直接运行。尚未在空白机器独立验证全量安装；迁移必须重建各环境，处理模型配置中的绝对路径，恢复数据后再做短推理和完整验收。
