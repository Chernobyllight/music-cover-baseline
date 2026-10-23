# 00_prepare · 准备输入音频和歌词

读取《秋意浓》原曲、歌词和孙燕姿参考录音，统一音频格式并裁剪参考片段。

## 使用示例

以下示例统一使用“孙燕姿参考音色＋秋意浓＋Synthwave / Dance-pop”，运行目录为 `runs/qiu_yi_nong_modular_test`。各模块使用同一个 `--run`，串联同一首歌曲的中间产物。

本模块直接读取原始素材，没有上游生成模块。首次创建运行需要通过 --config 提供任务参数。

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"

bash scripts/00_prepare/run.sh \
  --config configs/tasks/qiu_yi_nong_synthwave_sun_yanzi.yaml \
  --run runs/qiu_yi_nong_modular_test
```

本示例任务配置指定孙燕姿参考录音的前 16 秒、auto 移调及 Synthwave / Dance-pop 曲风。00 只准备输入，曲风生成由 03 执行，目标音色转换由 05 执行。

## 读取哪些输入

以下路径相对于项目根目录，由任务配置指定。

| 来源 | 输入文件 | 用途 |
| --- | --- | --- |
| 原始素材 | `data/raw/songs/qiu_yi_nong.mp3` | 原曲音频 |
| 原始素材 | `data/raw/lyrics/qiu_yi_nong.txt` | 原曲歌词 |
| 原始素材 | `data/raw/references/sun_yanzi/reference.wav` | 目标音色参考录音 |

## 本模块输出

首次执行的输出目录：`runs/qiu_yi_nong_modular_test/stages/00_prepare/attempt_001/artifacts/`。

| 文件 / 目录 | 内容及下游用途 |
| --- | --- |
| `source.wav` | 48 kHz 立体声原曲；供 01、02、10、11 使用 |
| `reference.wav` | 44.1 kHz 单声道参考片段；供 02、05、11 使用 |
| `lyrics.txt` | 清理后的歌词；供 03、11 使用 |
| `metadata.json` | 原曲与参考片段的音频信息 |

## 查看记录与重跑

每次尝试在 artifacts 同级保存 `inputs.json`（本次实际读取的完整路径、生产模块与哈希）、`config.json`、`command.json`、`process.log`、`checks.json` 和成功后的 `outputs.json`。有多次尝试时，以运行目录的 `stage_state.json` 及其中记录的 attempt 为准。

重复执行示例命令会复用本模块已有的有效结果；在命令末尾加 `--force` 可只重跑本模块并创建新 attempt。上游缺失或失效时，按报错提示从相应上游模块开始执行。

本模块完成后，如需从这里继续到最终交付，在同一项目根目录执行：

```bash
bash scripts/run_baseline.sh --run runs/qiu_yi_nong_modular_test --from 00_prepare
```

总入口先校验并复用有效的本模块及上游结果，再执行后续阶段。

业务实现：`src/music_cover/modules/prepare.py`。默认模块配置：`configs/modules/00_prepare.yaml`。已有运行保留初始化时的配置；调整已有运行请通过 `--config` 显式传入任务配置覆盖。

完整依赖、缓存及失败恢复说明见 [模块化说明](../../docs/09_modular_pipeline.md)。

另一个完整实例：[孙燕姿＋秋意浓＋Bossa Nova / Samba Jazz，输出 runs/lhd_test_modular](../../docs/11_modular_walkthrough.md)。该手册包含从 00 到 11 的每条命令、上下游路径和评估解读。
