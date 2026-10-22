本仓库用于保存音乐换曲风与参考音色转换 baseline 的代码、文档及版本记录。原始工作目录为 `/data/liuhongda/music-cover-baseline`；现有模型和音频产物继续保留在该目录，不随 Git 提交上传。

| 内容 | Git 管理方式 |
| --- | --- |
| `src/`、`scripts/`、`tests/` | 提交项目源码和测试 |
| `README.md`、`USAGE.md`、`VALIDATION.md`、`docs/` | 提交说明和验证记录 |
| `configs/` | 提交模型 revision、校验值、环境快照和当前配置 |
| `data/demo_sources.json` | 提交内置演示素材的来源和哈希，不包含音频或歌词正文 |
| `third_party/YuE`、`third_party/seed-vc` | 作为 Git 子模块固定上游代码 commit |
| `models/`、`envs/`、`cache/`、`checkpoints/` | 本地资源，不提交 |
| `data/raw/`、`data/prepared/`、`runs/` | 本地输入、中间结果和交付音频，不提交 |

在新目录获取代码时，同时初始化两个上游子模块：

```bash
git clone --recurse-submodules https://github.com/Chernobyllight/music-cover-baseline.git
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

需要内置演示素材时，执行已有的准备脚本：

```bash
envs/control/bin/python scripts/prepare_demo.py
```

自己的歌曲、歌词和参考录音按 [USAGE.md](../USAGE.md) 放入 `data/raw/`。当前仓库不包含《落叶归根》《秋意浓》以及目标人物参考录音，相关文件仍保存在原服务器。`runs/` 下的试听页和音频也不会出现在 GitHub；报告中的这些链接是本地验收入口。

代码检查可在现有服务器执行，不会启动整首音乐生成：

```bash
PYTHONPATH=src envs/control/bin/python -m unittest discover -s tests -v
```

该检查需要初始化 YuE 子模块，因为谱面移调复用了其中的 ABC 解析器。新环境中至少需要 NumPy、SciPy 和 SoundFile 等基础依赖。完整推理仍需完成上述环境、模型与输入准备，再通过 `scripts/run_baseline.sh` 执行。

`configs/project_code.lock.json` 记录本次归档时项目 Python 和 Shell 源码的 SHA256。后续代码变化由 Git commit 追溯；需要新的文件级快照时应同步更新该清单。
