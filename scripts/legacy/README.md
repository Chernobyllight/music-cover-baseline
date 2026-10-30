# v0.1 历史诊断

`verify_result_v01.py` 是旧 ASR/CAMPPlus 诊断，只供已有 v0.1 格式运行复核。它会写入旧式交付目录；检测到模块化 `stage_state.json` 时立即退出，不加载模型、不写文件。当前 00–11 不调用此目录。

历史目录的用法（须自行提供保留的 v0.1 运行）：

```bash
cd /data/liuhongda/music-cover-baseline
PYTHONPATH=src envs/seed-vc/bin/python scripts/legacy/verify_result_v01.py \
  runs/legacy_v01 --skip-asr
```

当前模块化运行请使用只读验证：

```bash
cd /data/liuhongda/music-cover-baseline
PYTHONPATH=src envs/separator/bin/python scripts/verify_review.py \
  runs/lhd_chuan_yue_shu_guang_orchestral_soul_huang_qi_shan_rvc_20261029 \
  --output cache/delivery_check.json
```

输入是 11 发布的 `deliverables/` 和运行清单，检查结果写入 `cache/`。旧 `scripts/verify_result.py` 仅保留兼容转发；正式交付流程见[交付与维护指南](../../docs/21_delivery.md)。
