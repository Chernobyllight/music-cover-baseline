# 当前运行环境与恢复边界

本服务器使用现有 `/opt/conda` 和项目 `envs/`。当前实际环境快照见 `configs/environments.lock.json`，各模型资源见 [权重清单](12_model_weights.md)。

共享工程文件不代表不同服务器的 base 包也共享。切换服务器后需要在实际运行的那台机器执行预检；
输入和中间产物使用[可迁移路径](09_modular_pipeline.md#路径与跨机器续跑)，无需固定挂载别名。

| 环境 | 主要用途 |
|---|---|
| `envs/control/bin/python` | 总入口主控制器、音频准备；06 独立脚本使用 base |
| `envs/sheetsage2/bin/python` | 01/08 转谱 |
| `envs/separator/bin/python` | 07 成品分轨、09 MIDI、评审打包与单元测试 |
| `/opt/conda/bin/python` | 06 独立脚本的混音、03 YuE2 曲风生成、02/04 的 BS-RoFormer 分离、02/10 的独立 RMVPE 提取，以及 Seed-VC、SVC、RVC 训练/推理和 ECAPA 评估；不另建环境 |

当前在 Linux、Python 3.11、NVIDIA H200 上验证。不同环境的 Torch/NumPy 版本并不相同；使用对应环境快照，不在工作环境中安装上游整套 requirements。YuE2 配置的显存预算为 80 GiB，不能宣称已验证 24 GB GPU。

## 已配置机器检查

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
bash scripts/run_baseline.sh --preflight
bash scripts/05_voice_conversion/rvc/run.sh \
  --config configs/voice_inference/rvc/huang_qi_shan.yaml --preflight
bash scripts/05_voice_conversion/so_vits_svc/run.sh \
  --config configs/voice_inference/so_vits_svc/yang_kun_optimized.yaml --preflight
```

检查 FFmpeg、FFprobe、MP3 编码器、所选后端包和所需资产。预检不等同于真实 GPU 推理验收。

## BS-RoFormer 与 RMVPE 直接使用 base

`separate`、`pitch` 后端操作的预检和实际子进程均使用 `/opt/conda/bin/python`，
由 `src/music_cover/common/resources.py` 统一解析，实际路径记录在每次尝试的 `command.json`。
02/04 分离与 02/10 基频提取不需要激活或新建项目虚拟环境。其它后端仍使用上表对应的既有环境。

在现有 base 中补充分离依赖，保留已有 PyTorch、NumPy、SciPy 和 Transformers：

```bash
cd /data/liuhongda/music-cover-baseline
/opt/conda/bin/python -m pip install --no-deps -r configs/requirements-pitch-base.txt
export PATH="/opt/conda/bin:$PATH"
bash scripts/02_pitch_adaptation/run.sh --preflight
```

该依赖清单用于本服务器已配置的 base，不是空白环境的完整安装配方。
`audio-separator==0.30.2` 的上游元数据声明 NumPy < 2；本项目保留 base 的 NumPy 2.2.6，
仅验证这里使用的 BS-RoFormer 分离路径，不代表该包所有模型都兼容 NumPy 2。
独立 RMVPE 同样用真实音频验证。运行快照位于 `configs/pitch-base.installed.txt`，参与这两个后端的缓存身份。

2026-10-30 用单依纯参考与完整《绿洲》重跑 02：两个后端的实际解释器均为 `/opt/conda/bin/python`，
13 个产物检查通过，推荐与采用均为 -4 半音，移调 ABC 与旧环境结果逐字节一致。
原曲 F0 最大绝对差约 0.000184 Hz，参考 F0 一致。128 项回归测试及主流程、SVC、RVC 预检通过。
这次只增加了依赖，没有替换 base 已有包；旧 attempt 保留。

此最小安装未包含该分离包其它架构使用的 diffq、julius、onnx、onnx2torch，
所以全局 `pip check` 仍会报告这些声明依赖和 NumPy 约束；不能把所选后端的运行通过理解为全局依赖检查无告警。

## YuE2 直接使用 base

03 的 `cover` 操作与预检均由 `src/music_cover/common/resources.py` 解析到
`/opt/conda/bin/python`，不再依赖 `envs/yue2/bin/python`。安装当前锁定的 YuE 子模块即可：

```bash
cd /data/liuhongda/music-cover-baseline
/opt/conda/bin/python -m pip install --no-deps --no-build-isolation -e third_party/YuE
export PATH="/opt/conda/bin:$PATH"
bash scripts/03_style_generation/run.sh --preflight
bash scripts/03_style_generation/run.sh \
  --run /data/liuhongda/music-cover-baseline/runs/lhd_lv_zhou_shan_yi_chun_modular_test --gpu 0
```

这里复用已经配置的 base，不是空白环境的安装配方。仅添加 `yue2-infer==0.1.6`，
保留 PyTorch 2.8.0+cu128、Transformers 4.57.3、NumPy 2.2.6 与其它模型依赖。
上游包元数据固定了 PyTorch 2.10.0 等较新版本，因此使用 `--no-deps` 避免全局升级；
全局 `pip check` 会报告与其声明版本不同，实际兼容范围以本项目验证的生成路径为准。
`configs/yue-base.installed.txt` 保存实际依赖并参与 03 缓存指纹；
旧 `configs/yue2.installed.txt` 是历史快照，不参与当前 03 的运行和缓存身份。
两份 YuE2 模型权重、歌词、ABC 乐谱和生成参数沿用现有配置，权重无需重新下载。

2026-10-30 已用 `runs/lhd_lv_zhou_shan_yi_chun_modular_test` 完整验证 03，
复用 00–02 产物（-4 半音 ABC）、Afro-Cuban mambo / Latin big-band jazz 风格与 seed=42。
`attempt_001/command.json` 的实际解释器为 `/opt/conda/bin/python`；生成耗时约 158.5 秒，
输出 257.4 秒、48 kHz 双声道 FLAC，12 个产物检查通过，ABC 与 semantic 均未截断。
实际执行使用 CUDA Graph / Flash Attention，未修改上游生成算法或模型权重。
129 项回归测试、主流程/SVC/RVC 预检与源代码/文档链接检查通过。
这验证了本服务器完整生成路径可运行，音质仍需人工试听；不是全新环境复现验收。

## Seed-VC 与音色评估直接使用 base

05 零样本、微调候选推理、Seed-VC 训练入口，以及 ECAPA/RMVPE 音色评估统一使用
`/opt/conda/bin/python`。预检和实际子进程由 `src/music_cover/common/resources.py` 统一解析；
不要求激活、安装或调用 `envs/seed-vc`。旧环境可保留作历史记录，但不参与这些入口。

```bash
cd /data/liuhongda/music-cover-baseline
/opt/conda/bin/python -m pip install --no-deps -r configs/requirements-seed-vc-base.txt
export PATH="/opt/conda/bin:$PATH"
bash scripts/05_voice_conversion/zero_shot/run.sh \
  --run runs/lhd_lv_zhou_shan_yi_chun_modular_test --gpu 0 --preflight
```

此清单用于已配置的 base，只补充 Seed-VC 所需依赖，不是空白环境安装配方。
保留现有 PyTorch、CUDA、NumPy 和 Transformers；不要安装上游整套 requirements。
实际包快照见 `configs/seed-vc-base.installed.txt`，参与换声及音色评估的缓存身份。
运行记录的 `command.json` 保存实际解释器，换声模型和推理参数不因环境迁移而改变。
共享项目所在的另一台服务器也要单独检查其 `/opt/conda`，共享文件不会同步 Python 包。

2026-10-30 在 6969 的 base 验证：Python 3.11.13、PyTorch 2.8.0+cu128、NumPy 2.2.6、
Transformers 5.14.1、huggingface-hub 1.26.0。只添加 16 个缺失包，未替换原有包。
Seed-VC 适配层为旧版 BigVGAN 本地加载器补齐 Hub 1.x 不再传递的两个参数默认值，
不修改第三方代码、模型网络、权重或全局依赖版本。
单依纯参考＋10 秒《绿洲》生成干声的零样本推理成功；另完成单候选微调推理＋ECAPA 评估、
一个训练步的梯度更新。候选测试使用黄绮珊既有权重和缓存中的配置副本，将旧挂载路径改为当前路径；
没有修改原导出音色配置与权重。旧 profile 内嵌的外部绝对路径仍需在迁移时明确处理，
这次测试不代表任意旧导出 profile 均能直接跨机器使用。
验证数据在 `cache/seed_vc_base_20261030/`，用户的手动测试 run 未新增任何阶段结果。
136 项回归测试使用已有测试环境执行；源码、配置和文档链接检查通过。

## 记录环境，保持运行缓存身份

```bash
/opt/conda/bin/python scripts/setup/capture_environment.py \
  --output-dir cache/environment_snapshot --skip-assets
```

输出包含四个仍在使用的项目环境、base、四个上游 commit、GPU 与 FFmpeg；不再调用旧 yue2 解释器。省略 `--skip-assets` 会额外遍历 models 计算哈希；可能耗时较长。省略 `--output-dir` 才会更新 configs 下的运行锁；运行锁参与缓存指纹，更新后相关阶段可能需要重新计算。

`configs/*.installed.txt` 中的旧局部 freeze 记录不等于完整安装配方，其中可能有原服务器 wheel 路径。此轮快照用于核对现有机器，并未声称已在全新机器从零安装通过。迁移需重建环境、恢复锁定权重和素材，再执行预检、短推理与完整任务验收。步骤见 [仓库说明](08-repository.md)。

最初环境设想另存 [历史环境文档](archive/03-environment_before_cleanup_20261029.md)，其中“尚未安装”等描述只反映当时状态。
