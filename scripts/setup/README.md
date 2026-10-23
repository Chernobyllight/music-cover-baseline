# 环境与素材准备

本目录集中管理模型下载、演示素材准备和环境记录。原 `scripts/` 下同名入口保持兼容。它不属于每首歌的 00–11 阶段，不读取 `runs/` 中的中间产物；模型与环境准备好后，每首歌从 00 开始。

## 已配置服务器上的使用示例

检查当前任务所需的环境和模型，不生成歌曲：

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"

bash scripts/run_baseline.sh \
  --config configs/tasks/qiu_yi_nong_synthwave_sun_yanzi.yaml \
  --preflight
```

环境检查通过后，开始准备“孙燕姿＋秋意浓＋Synthwave / Dance-pop”的输入：

```bash
bash scripts/00_prepare/run.sh \
  --config configs/tasks/qiu_yi_nong_synthwave_sun_yanzi.yaml \
  --run runs/qiu_yi_nong_modular_test
```

00 读取配置中的 `data/raw/songs/qiu_yi_nong.mp3`、`data/raw/lyrics/qiu_yi_nong.txt` 和 `data/raw/references/sun_yanzi/reference.wav`，输出到 `runs/qiu_yi_nong_modular_test/stages/00_prepare/attempt_001/artifacts/`。具体数据衔接见 [00 的 README](../00_prepare/README.md)。

## 本目录脚本的具体用法

以下命令均在项目根目录执行，仅在需要对应准备操作时使用。下载脚本会写入模型或素材及相应锁记录；环境捕获会更新环境和资产锁文件。环境创建与恢复步骤见 [使用说明](../../USAGE.md) 和 [仓库恢复说明](../../docs/08-repository.md)。

| 操作与具体命令 | 读取内容 | 产生的数据及消费者 |
| --- | --- | --- |
| `envs/control/bin/python scripts/setup/download_models.py` | 脚本中固定 revision 的模型仓库 | `models/SheetSage2`、`models/MERT-v2-FullSong`、`models/YuE2-3B`、`models/YuE2-Vae`，以及 `configs/generation_models.lock.json`；供 01、03、08 使用 |
| `envs/control/bin/python scripts/setup/download_separator_model.py` | 固定的分离权重下载地址 | `models/separation/` 和 `configs/separation_model.lock.json`；供 02、04 使用 |
| `envs/control/bin/python scripts/setup/download_voice_models.py` | 固定 revision 的 Seed-VC、RMVPE 等模型仓库 | `models/` 中换声及音高模型、`configs/voice_models.lock.json`、`configs/seed_svc.local.yml`；供 02、05、10 使用 |
| `envs/separator/bin/python scripts/setup/setup_review_assets.py` | 固定的评审分离权重及已安装的 Basic Pitch 包 | `models/separation/` 中的 karaoke 权重、`models/review/` 和 `configs/review_models.lock.json`；供 07、09 使用 |
| `envs/control/bin/python scripts/setup/prepare_demo.py` | 官方演示歌曲、歌词和 LibriSpeech 参考素材 | `data/raw/` 下《今晚不眠》演示素材、`configs/demo.json`、`data/demo_sources.json`；供使用 `--demo` 的 00 读取。本脚本不下载本例的《秋意浓》和孙燕姿素材 |
| `/opt/conda/bin/python scripts/setup/capture_environment.py` | 已安装的五个环境、模型文件、GPU/FFmpeg 信息及上游 commit | `configs/*.installed.txt`、`configs/environments.lock.json`、`configs/local_assets.lock.json`；用于环境追溯与缓存身份 |
| `/opt/conda/bin/python scripts/setup/fetch_wheels.py` | 脚本中列出的安装包下载地址 | `cache/wheels/`；供后续安装使用，不自动安装，也不生成歌曲产物 |

例如，调整环境并完成安装后重新记录实际环境和模型哈希：

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"

/opt/conda/bin/python scripts/setup/capture_environment.py
```
