> 设计存档：本文件记录实施前的方案，部分环境、文件名和步骤已调整。当前使用方法以 [USAGE.md](../USAGE.md) 为准；实测版本见 `configs/environments.lock.json`，实测结果见运行目录的 `deliverables/report.md`。

接口设计以“音频内容、时间位置和处理来源都可追溯”为原则。下面的 JSON 文件名和字段是未来实现的约定，本轮仅记录定义，不创建可执行配置。

一次任务的 request 至少包括以下信息。示例值用于设计说明，不代表已有素材。

| 字段 | 类型与例子 | 约束 |
| --- | --- | --- |
| schema_version | 整数，首版为 1 | 不支持的版本直接拒绝 |
| mode | cover 或 vc_probe | 正式歌曲与模块探针分开 |
| song_id | 稳定字符串，如 song_001 | 不直接使用任意歌曲名作为目录路径 |
| source_audio | 项目内路径或登记过的本地路径 | 必须可解码；首次登记保存文件 SHA256 |
| lyrics_file | 文本文件路径 | 正式 cover 必填；保存编码和内容哈希 |
| reference_audio | 参考录音路径 | 必填，与 source_audio 的用途不同 |
| reference_segment | start_s、end_s | 0 ≤ start_s < end_s ≤ 原文件时长 |
| target_style | 非空字符串 | 原样保存；首版不自动改写用户风格 |
| score_override | 可选 ABC 路径 | 有效时使用指定谱，仍进行结构检查 |
| yue2 | cot、seed 及经验证的上游参数 | 首版 cot=melody；未支持的参数拒绝传递 |
| separation | 模型身份、采样率和增益策略 | 参数随版本固定 |
| conversion | 后端及其参数 | 首版为 Seed-VC 歌声分支 |
| mix | 输出采样率、增益和延迟校正 | 首版输出 48000 Hz 立体声 |
| device | 选择的物理 GPU、精度 | resolved 中记录实际设备和执行精度 |

项目配置与上游配置分开。适配器将歌词文本、ABC 文件和风格描述转换成固定版本上游实际接受的格式；不得把项目自定义字段直接传给模型。

raw 目录保留原始字节。prepared 目录按“原文件哈希＋预处理参数＋处理工具版本”区分素材；不能把重采样、裁剪或去混响后的文件覆盖回 raw。每个 prepared 产物记录父文件、裁剪区间、重采样算法、声道转换和增益。

音频资产统一记录下列 metadata。

| 字段 | 定义 |
| --- | --- |
| artifact_id、role | 唯一产物标识及 source、reference、cover、vocal、instrumental 等用途 |
| path、sha256 | 实际文件位置及内容哈希 |
| sample_rate、channels、num_frames | 解码后的采样率、声道数、帧数 |
| duration_s | num_frames / sample_rate，保留足够精度 |
| timeline_id | original、reference、probe 或某次 cover 的唯一时间轴 |
| start_sample、timeline_sample_rate | 在对应时间轴中的整数起点及该时间轴的采样率 |
| parent_artifact_ids | 直接上游产物 |
| processing | 裁剪、声道混合、重采样、增益和延迟处理记录 |

`num_frames` 表示所有声道共有的时间帧数；立体声不会将时长乘二。不同采样率之间不能直接比较采样点个数，应先换算为秒或共同时间轴上的采样点。

各阶段的音频约定如下。

| 产物 | 采样率与声道 | 时间约束 |
| --- | --- | --- |
| 原曲准备文件 | 项目中保存 48 kHz 立体声工作副本；上游按自身需要读取 | 保留原曲全长和原始起点 |
| 参考片段 | 保留清晰工作副本，Seed-VC 入口转为所需格式 | 独立 reference 时间轴；允许裁剪首尾无用静音 |
| YuE2 输出 | 上游为 48 kHz 立体声 | 建立新 cover 时间轴，起点为 0 |
| 分离人声与伴奏 | 记录分离模型实际输出，不假设天然为 48 kHz | 必须保持相同 cover 起点；记录帧数差异 |
| Seed-VC 输入人声 | 44.1 kHz 单声道 | 保留 cover 时间位置，不能删除中间停顿 |
| Seed-VC 输出 | 44.1 kHz 单声道 | 与输入近似等长，但必须实测延迟及局部漂移 |
| 最终混音与直接重混 | 48 kHz 立体声 | 同一 cover 时间轴，默认长度等于选定的 cover 主文件 |

YuE2 的采样率来自[官方说明](https://github.com/multimodal-art-projection/YuE)；Seed-VC 的实际读取、分块和输出行为以[固定版本推理代码](https://github.com/Plachtaa/seed-vc/blob/main/inference.py)为准。其余格式选择是本项目的工程约定。

分离适配器需要显式处理工具默认值。当前 audio-separator 默认 sample_rate=44100、normalization_threshold=0.9、amplification_threshold=0.0、use_soundfile=False；输出路径会检查并可能缩放每条音轨的峰值。[默认参数](https://github.com/nomadkaraoke/python-audio-separator/blob/main/audio_separator/separator/separator.py)、[峰值缩放实现](https://github.com/nomadkaraoke/python-audio-separator/blob/main/audio_separator/separator/uvr_lib_v5/spec_utils.py)

首版候选设置为：核对所选模型配置后使用 44100 Hz，normalization_threshold=1.0、amplification_threshold=0.0、use_soundfile=True，输入提供浮点 WAV 工作副本，写出后检查实际 subtype。阈值 1.0 仍可能缩放超出该峰值的轨道，不能当作彻底关闭归一化。若无法取得准确缩放倍率，应在分离记录中标明，必要时统一降低分离输入电平重跑；不能按轨猜测增益来制造“完美重混”。

立体声人声转单声道前检查相位和声道能量。首版可使用固定的左右声道平均策略，但如出现明显相消，应更换声道策略并记录；不能用转换后音量变小来误判模型失败。

中间音频优先使用浮点 WAV，最终归档使用 24-bit PCM WAV。工具输出 FLAC 可以直接保留，必要的工作副本另存；不在中间阶段反复进行 MP3 等有损编码。不得默认对人声、伴奏分别做不留记录的峰值归一化。

混音采用固定的电平关系：输出左、右声道分别等于该声道伴奏乘以伴奏增益，再加转换人声乘以人声增益。首版将单声道人声等量送入左右声道，明确不额外施加未知的 DAW 声像衰减。最终限制峰值时只使用一次记录过的整体增益或限制器。

对齐记录至少包括：输入与输出长度、换声前后起音位置、估计固定延迟、开头／中间／结尾三个检查区间、采用的校正方式和剩余偏差。声线改变后不依赖原始波形相关性作为唯一判据，可结合能量包络、音节起点和听感。

首版不自动时间拉伸整首人声。只允许对已确认的固定延迟和尾部帧取整误差做校正；发现随时间增加的漂移时，标记需要检查分块与长度处理。若以后允许保留额外尾响，必须显式改变 mix 输出长度策略。

阶段状态使用 pending、running、needs_review、succeeded、failed、skipped。外部进程退出码为 0 后仍需检查产物存在、可解码、数值有限、声道和时长合理；检查通过才提交该阶段产物。阶段标记 succeeded 表示技术检查通过，听感评价另存为 pass、fail 或 not_reviewed。

旋律和生成内容检查可使运行停在 needs_review；记录的是音乐质量判断，不是每个文件操作都要额外申请权限。首个样本由操作人员核对，后续是否减少人工检查由实验结果决定。

断点恢复遵循以下约定。

1. 缓存键包含输入内容哈希、该阶段有效参数、模型权重与配置身份、代码版本、关键环境版本，以及影响结果的 seed、精度和设备类别。
2. 只复用哈希和技术校验均通过的产物；仅发现同名文件不能判为成功。
3. 需要人工复核的结果保留 needs_review 状态，恢复时不自动变成通过。
4. 先写到本次 attempt 下的临时文件，校验后在同一文件系统内原子重命名，再更新 manifest；完整日志保留。
5. 同一 run 只允许一个写入进程。进程异常结束后检查 PID 和 attempt 状态，不能盲目删除锁或覆盖成功结果。
6. 同一随机种子不承诺跨 GPU、依赖版本和推理实现逐采样点一致；复现以已保存产物和环境身份共同描述。

上游调用失败时保存错误、日志和耗时，不静默改精度、模型、采样率或风格参数。重试参数变化必须进入新 attempt 的 resolved 配置。
