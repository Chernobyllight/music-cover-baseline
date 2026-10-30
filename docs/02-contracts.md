# 当前输入输出契约

任务 YAML/JSON 顶层使用 `request` 和 `modules`。`request` 包含 song、lyrics、reference、style、name、seed、steps、gpu、reference_start、reference_duration、transpose、review；未知字段会报错。配置与 CLI 的具体用法见 [USAGE](../USAGE.md)。

示例：`configs/tasks/chuan_yue_shu_guang_huang_qi_shan_seedvc_20261029.yaml`。各模块默认值在 `configs/modules/`，同时由 `modules/*.py` 检查。

| 记录 | 实际作用 |
|---|---|
| `request.json` | 本次解析后的请求，输入使用绝对路径 |
| `config.resolved.json` | 请求与各模块的实际配置 |
| `stage_state.json` | schema_version=2；各阶段当前状态、attempt、指纹和产物 |
| `stages/<模块>/attempt_NNN/inputs.json` | 输入绝对路径、SHA256、字节数及生产阶段 |
| `config.json` | 实际参数与缓存 provenance |
| `command.json` / `process.log` | 命令、后端环境、日志与返回状态 |
| `checks.json` / `outputs.json` | 阶段检查与成功产物清单 |
| `manifest.json` | 整体完成状态与阶段汇总 |
| `deliverables` | 11 通过检查后发布的当前交付入口 |

缓存同时核对输入、参数、相关实现、环境/模型资源和产物哈希。更改代码可能使相关阶段失效，旧 attempt 与已发布音频仍保留。只执行一个模块时不自动补跑失效上游；从更早的 `--from` 恢复。

## 音频与时间线

- 00 原曲与原曲转谱属于 original 时间线；参考片段独立。
- 02 调整谱面；03 生成后可能改变时长和演唱细节。
- 04–11 使用 generated 时间线。05 保持源人声总长度和 F0；06 验证对齐并统一输出。
- 最终 WAV 为 48 kHz、立体声、24-bit；MP3 为 192 kbps。评审分轨为同长 48 kHz 立体声浮点音频。
- RVC 保存原生 48 kHz 人声，另输出 44.1 kHz converted.wav 交给 06。

`voice.json` schema_version=1，保存 backend、voice_id、version、权重/配置/验证记录的相对路径与哈希。RVC 另含训练集索引与来源清单。加载时检查对应后端、架构和文件身份；不会启动训练或自动替换不匹配权重。

最初的规划字段（例如 score_override、vc_probe）未必已实现，见 [历史契约](archive/02-contracts_before_cleanup_20261029.md)，不要直接传给当前 CLI。
