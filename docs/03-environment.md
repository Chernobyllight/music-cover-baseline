# 当前运行环境与恢复边界

本服务器使用现有 `/opt/conda` 和项目 `envs/`。当前实际环境快照见 `configs/environments.lock.json`，各模型资源见 [权重清单](12_model_weights.md)。

| 环境 | 主要用途 |
|---|---|
| `envs/control/bin/python` | 主控制器、音频准备和混音 |
| `envs/sheetsage2/bin/python` | 01/08 转谱 |
| `envs/yue2/bin/python` | 03 曲风生成 |
| `envs/separator/bin/python` | 02/04/07 分离、09 MIDI、评审打包与单元测试 |
| `envs/seed-vc/bin/python` | Seed-VC 训练/推理、RMVPE 诊断、ECAPA 候选评估 |
| `/opt/conda/bin/python` | SVC、RVC 训练/推理；不另建环境 |

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

## 记录环境，保持运行缓存身份

```bash
/opt/conda/bin/python scripts/setup/capture_environment.py \
  --output-dir cache/environment_snapshot --skip-assets
```

输出包含五个项目环境、base、四个上游 commit、GPU 与 FFmpeg。省略 `--skip-assets` 会额外遍历 models 计算哈希；可能耗时较长。省略 `--output-dir` 才会更新 configs 下的运行锁；运行锁参与缓存指纹，更新后相关阶段可能需要重新计算。

`configs/*.installed.txt` 中的旧局部 freeze 记录不等于完整安装配方，其中可能有原服务器 wheel 路径。此轮快照用于核对现有机器，并未声称已在全新机器从零安装通过。迁移需重建环境、恢复锁定权重和素材，再执行预检、短推理与完整任务验收。步骤见 [仓库说明](08-repository.md)。

最初环境设想另存 [历史环境文档](archive/03-environment_before_cleanup_20261029.md)，其中“尚未安装”等描述只反映当时状态。
