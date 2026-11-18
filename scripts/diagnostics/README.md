# MPS 诊断

在仓库根目录执行，使用对应的锁定 Python 环境。日常运行入口见 [Mac 使用说明](../../docs/09-macos.md)。

- `probe_yue_mps.py` 验证 YuE2 短谱面、声学合成与解码。
- `profile_mps.py` 保留旧 `main` 性能实验的受限探针。复现需要原始运行快照及其旧阶段产物，Round 4 还需要原始计时账本。历史命令和结果见 [性能报告](../../docs/14-mps-ten-minute-target.md)。
- `compare_separator_blocks.py`、`check_mps_candidates_cpu.py` 和 `profile_seed_phrase.py` 可对既有输出运行 CPU 侧算子及音频比较。

原始运行快照和 `runs/probes/mps-performance/budget.json` 都是本地实验产物。账本保留实际执行时的路径与计费历史。使用 v1.2 时，可通过 Mac 使用说明里的流水线入口生成新 run，并在各阶段的 attempt 中查看日志与运行记录。
