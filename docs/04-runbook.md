# 当前运行与故障处理

完整命令和输入输出路径统一维护在 [USAGE](../USAGE.md)，交付验收见 [交付指南](21_delivery.md)。

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
bash scripts/run_baseline.sh \
  --config configs/tasks/chuan_yue_shu_guang_huang_qi_shan_seedvc_20261029.yaml \
  --run runs/my_chuan_yue_shu_guang_seedvc --gpu 0
```

| 情况 | 处理 |
|---|---|
| 缺少 FFmpeg/FFprobe | 将已安装的 FFmpeg 目录加入 PATH，再运行 --preflight |
| 缺模型或哈希不匹配 | 查看所选后端的锁文件与权重指南，恢复原版本后重试 |
| 上游缓存失效 | 从错误指出的最早失效模块使用 --from；不要改清单伪装有效 |
| 阶段报错 | 查看对应 attempt/process.log、failure.json；修复原因后同 run 继续 |
| 已有导出版本 | 使用新的模型版本和 export_dir；不覆盖现有 voice.json |
| 只想换音色 | fork 已完成的 00–04，再运行 05–11；保留被引用的源 run |
| 想检查已交付结果 | 使用 scripts/verify_review.py；报告写到 cache |

旧 v0.1 诊断保留在 `scripts/legacy/`，不对当前模块化结果执行 ASR 写入。最初操作计划见 [历史 runbook](archive/04-runbook_before_cleanup_20261029.md)。
