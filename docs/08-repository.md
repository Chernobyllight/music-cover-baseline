当前版本：`v0.3`，版本标识见根目录 `VERSION`，变更见 [CHANGELOG.md](../CHANGELOG.md)。逐步操作见 [实操手册](11_modular_walkthrough.md)。

模型资产的集中清单、固定版本、下载命令与校验方法见[模型权重指南](12_model_weights.md)。

本仓库用于保存音乐换曲风与参考音色转换 baseline 的代码、文档及版本记录。原始工作目录为 `/data/liuhongda/music-cover-baseline`；现有模型和音频产物继续保留在该目录，不随 Git 提交上传。

v0.3 分支的模块化布局见 [模块化文档](09_modular_pipeline.md)。模型和环境目录继续复用；旧运行只读保留，新的模块化执行创建新运行目录。

| 内容 | Git 管理方式 |
| --- | --- |
| `src/`、`scripts/`、`tests/` | 提交项目源码和测试 |
| `README.md`、`USAGE.md`、`VALIDATION.md`、`docs/` | 提交说明和验证记录 |
| `configs/` | 提交模型 revision、校验值、环境快照和当前配置 |
| `data/demo_sources.json` | 提交内置演示素材的来源和哈希，不包含音频或歌词正文 |
| `third_party/YuE`、`third_party/seed-vc` | 作为 Git 子模块固定上游代码 commit |
| `models/`、`envs/`、`cache/`、`checkpoints/` | 本地资源，不提交 |
| `data/raw/`、`data/prepared/`、`data/voice_datasets/`、`runs/` | 本地输入、中间结果和交付音频，不提交 |

在新目录获取代码时，同时初始化两个上游子模块：

```bash
git clone --branch v0.3 --recurse-submodules https://github.com/Chernobyllight/music-cover-baseline.git
cd music-cover-baseline
```

如果已经普通克隆，在仓库内执行：

```bash
git submodule update --init --recursive
```

子模块固定版本如下；使用普通 `git submodule update` 恢复记录的版本，不使用 `--remote` 更新至上游最新分支。

| 上游仓库 | 已验证的 commit |
| --- | --- |
| [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE) | `1647252d68b70cbead046ffd3f7caf036dde47ee` |
| [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) | `51383efd921027683c89e5348211d93ff12ac2a8` |

克隆代码不会同时获得可直接运行的 GPU 环境。现有 `envs/` 复用了原服务器 `/opt/conda` 的基础包，不能直接复制到另一台机器使用。迁移时需分别重建 `control`、`sheetsage2`、`yue2`、`separator`、`seed-vc` 环境，实际 Python、Torch、CUDA 与关键包版本见 `configs/environments.lock.json`。

`configs/*.installed.txt` 是当时各环境的局部安装记录，部分条目引用原服务器 wheel 路径，而且没有列出从基础环境继承的全部包；它们不是可直接在任意机器执行的完整安装清单。当前验证记录见 [VALIDATION.md](../VALIDATION.md)。

在所需 Python 环境和下载依赖准备好后，可使用现有脚本恢复固定版本的模型：

```bash
envs/control/bin/python scripts/download_models.py
envs/control/bin/python scripts/download_voice_models.py
envs/control/bin/python scripts/download_separator_model.py
```

下载脚本需要相应的 `huggingface_hub`、`PyYAML` 等依赖。它们会创建模型目录并更新本地模型清单；`download_voice_models.py` 还会按当前仓库位置重建 `configs/seed_svc.local.yml` 中的绝对路径。版本清单里的原始绝对路径是原服务器记录，迁移时不能直接照搬。

从 `v0.1` 起默认开启成品分轨和 MIDI 评审，`v0.3` 继续保留。基础 separator 环境配置好后，还需要安装附加依赖和模型：

```bash
envs/separator/bin/python -m pip install --no-deps -r configs/review-requirements.txt
envs/separator/bin/python scripts/setup_review_assets.py
export PATH="/opt/conda/bin:$PATH"
bash scripts/run_baseline.sh --preflight
```

该附加清单复用已有 Torch 2.8 / CUDA、NumPy、Librosa、ONNX Runtime、Julius 和 audio-separator 环境，不是一个独立环境的完整 requirements 文件。验证版本及模型 URL、哈希在 `configs/review_models.lock.json`。Basic Pitch 使用 CPU ONNX，不要求安装 TensorFlow。只恢复旧基础流程可使用 `--no-review`，对应预检为 `bash scripts/run_baseline.sh --preflight --no-review`。

需要内置演示素材时，执行已有的准备脚本：

```bash
envs/control/bin/python scripts/prepare_demo.py
```

自己的歌曲、歌词和参考录音按 [USAGE.md](../USAGE.md) 放入 `data/raw/`。当前仓库不包含《落叶归根》《秋意浓》《喜欢你》以及目标人物参考录音，相关文件仍保存在原服务器。`runs/` 下的试听页、音轨、MIDI 和评分表也不会出现在 GitHub。历史运行目录已按需清理，验证文档中的旧编号只作为历史记录。

代码检查可在现有服务器执行，不会启动整首音乐生成：

```bash
PYTHONPATH=src envs/separator/bin/python -m unittest discover -s tests -v
```

该检查需要初始化 YuE 子模块，因为谱面移调复用了其中的 ABC 解析器。新环境中需要 NumPy、SciPy、支持 MP3 解码的 SoundFile/libsndfile，以及新增测试所需的 Mido / Pretty MIDI；在 PATH 中提供带 `libmp3lame` 编码器的 FFmpeg。音频测试会实际生成和解码 MP3，完整测试集应使用已配置评审依赖的 separator 环境。完整推理仍需完成上述环境、模型与输入准备，再通过 `scripts/run_baseline.sh` 执行。

对已经完成的评审运行执行独立检查，不会重新调用生成模型：

```bash
envs/separator/bin/python scripts/verify_review.py /绝对路径/runs/某次运行 \
  --output /绝对路径/runs/某次运行/delivery_verification.json
```

`configs/project_code.lock.json` 记录本次归档时项目 Python 和 Shell 源码的 SHA256。后续代码变化由 Git commit 追溯；需要新的文件级快照时应同步更新该清单。

## v0.3 的微调资源恢复

仅恢复代码不会获得歌手模型。已导出的音色版本需从自己的训练/备份恢复整个 `models/voices/<voice_id>/seed_vc/<version>/`，包含 `voice.json`、`model.pth`、`config.yml`、`validation.json`；解析器核对文件哈希、44.1 kHz 和 F0 条件。模型配置中的本机路径也需适配新机器。标准训练/验证/导出步骤见 [音色微调手册](13_voice_adaptation.md)。

微调推理（候选数为 1 也一样）需 ECAPA/RMVPE；OpenVoiceV2 和 `se_db.pt` 只在训练时需要。配置好原有 Seed-VC 环境后，按 [权重指南](12_model_weights.md) 安装附加依赖并执行：

```bash
envs/seed-vc/bin/python scripts/setup/setup_voice_training.py --scope evaluation
envs/seed-vc/bin/python scripts/setup/setup_voice_training.py --scope evaluation --check-only
bash scripts/run_baseline.sh --config configs/tasks/bu_bai_de_ying_xiong_celtic_epic_yang_kun.yaml --preflight
```

最后一条要求已经恢复该任务指定的杨坤 profile；其他歌手请换自己的任务配置。旧歌手录音、导出权重及训练状态均不随仓库发布；本版不包含单曲伪标签方案、歌词审核模型或歌词检查接口。
