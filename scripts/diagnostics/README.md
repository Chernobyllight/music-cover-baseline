# MPS 诊断

在仓库根目录执行，使用对应的锁定 Python 环境。日常运行入口见 [Mac 使用说明](../../docs/09-macos.md)。

- `probe_yue_mps.py` 验证 YuE2 短谱面、声学合成与解码。
- `profile_mps.py` 启动受限性能探针，复用既有输入和中间产物。各对照组的复现命令见 [性能报告](../../docs/14-mps-ten-minute-target.md)。
- `compare_separator_blocks.py`、`check_mps_candidates_cpu.py` 和 `profile_seed_phrase.py` 可对既有输出运行 CPU 侧算子及音频比较。

既有计时账本在 `runs/probes/mps-performance/budget.json`。保留原有记录和计费历史。账本中历史命令保留实际执行时的路径。
