# v0.2 模型权重清单与准备方法

本文集中说明：**每个模块需要什么模型、从哪里获取、放在哪里、怎样检查，以及对应代码在哪里。** 按 2026-10-23 的 `v0.2` 代码、锁文件和现有服务器文件核对。

当前完整流程（`--transpose auto --review`）需要 **13 组模型资产**，包含主模型和附属模型；同一个模型可被多个阶段复用。GitHub 仓库不包含这些大文件。孙燕姿或詹姆斯的音色由参考录音提供，当前 Seed-VC 路线无需单独下载对应人物的专用权重。

本机项目根目录：

```text
/mnt/afs_f/liuhongda/music-cover-baseline
```

下文 `models/...`、`envs/...` 等均相对此根目录。现有 `/data/liuhongda/music-cover-baseline` 指向同一共享项目；迁移到新机器时应使用新机器实际路径。本文负责模型资产准备；Python/CUDA 环境的前提及迁移限制见[仓库恢复说明](08-repository.md)，运行步骤见[00–11 实操手册](11_modular_walkthrough.md)。

## 1. 先按模块看需要哪些模型

| 模块 | 需要的模型 | 用来做什么 |
| --- | --- | --- |
| `00_prepare` | 无新增模型权重 | 整理原曲、歌词、参考录音 |
| `01_source_score` | SheetSage2 + MERT-v2-FullSong | 原曲转谱；MERT 是配套音频编码器 |
| `02_pitch_adaptation` | BS-RoFormer + RMVPE | 分离原曲人声、提取原曲与参考的音高，推荐升调或降调；`--transpose 0` 时跳过这些模型调用 |
| `03_style_generation` | YuE2-3B + YuE2-Vae | 根据乐谱、歌词和风格生成音乐，并解码为音频 |
| `04_vocal_separation` | BS-RoFormer | 新生成音乐拆成人声和伴奏 |
| `05_voice_conversion` | Seed-VC + Whisper-small + CAMPPlus + RMVPE + BigVGAN | 提取内容、参考音色和音高，换声并合成波形 |
| `06_mix` | 无新增模型权重 | 对齐、增益处理和混音 |
| `07_stem_separation` | Mel-Band RoFormer Karaoke + Demucs `htdemucs_6s` | 估计主唱/背景人声，以及鼓、贝斯、吉他、钢琴等分轨 |
| `08_final_score` | 复用 SheetSage2 + MERT-v2-FullSong | 对最终混音重新转谱 |
| `09_stem_midi` | Basic Pitch ONNX | 分轨转 MIDI；和弦 MIDI 复用 08 结果，鼓 MIDI 使用频带起音规则，无额外鼓转谱模型 |
| `10_evaluation` | 条件性复用 RMVPE | `pitch_check=true` 且请求的 `transpose` 不为 `0` 时检查实际移调与换声后的音高保持 |
| `11_delivery` | 无新增模型权重 | 整理交付与评审包 |

`--no-review` 会跳过 07、08、09，因此不需要 Mel-Band RoFormer Karaoke、Demucs 和 Basic Pitch 这三组评审模型。SheetSage2 仍被 01 使用，RMVPE 仍被 05 使用；不能因为跳过某个复用阶段就把这些权重删掉。

“分轨”与“MIDI”均为模型或规则的估计结果；有权重、有输出文件，不代表分离无串音或乐谱完全正确。

## 2. 13 组模型与实际文件位置

大小仅统计表内**主权重文件**，使用十进制 MB/GB，按现有文件四舍五入；不包含配置、缓存、Python 环境及运行结果，也不代表显存需求。

| 模型 | 主权重文件（相对项目根目录） | 大小 | 下载来源或安装来源 |
| --- | --- | --- | --- |
| SheetSage2 | `models/SheetSage2/model.safetensors` | 229 MB | `m-a-p/SheetSage2`，见下方固定 revision |
| MERT-v2-FullSong | `models/MERT-v2-FullSong/model.safetensors` | 2.53 GB | `m-a-p/MERT-v2-FullSong` |
| YuE2-3B | `models/YuE2-3B/model.safetensors` | 7.26 GB | `m-a-p/YuE2-3B` |
| YuE2-Vae | `models/YuE2-Vae/model.safetensors` | 531 MB | `m-a-p/YuE2-Vae` |
| BS-RoFormer | `models/separation/model_bs_roformer_ep_317_sdr_12.9755.ckpt` | 639 MB | [UVR 模型发布资产](https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt) |
| Seed-VC（44.1 kHz、F0 歌唱分支） | `models/Seed-VC/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth` | 821 MB | `Plachta/Seed-VC` |
| Whisper-small | `models/Whisper-small/model.safetensors` | 967 MB | `openai/whisper-small` |
| CAMPPlus | `models/CAMPPlus/campplus_cn_common.bin` | 28 MB | `funasr/campplus` |
| RMVPE | `models/RMVPE/rmvpe.pt` | 181 MB | `lj1995/VoiceConversionWebUI` |
| BigVGAN | `models/BigVGAN/bigvgan_generator.pt` | 489 MB | `nvidia/bigvgan_v2_44khz_128band_512x` |
| Mel-Band RoFormer Karaoke | `models/separation/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` | 913 MB | [UVR Karaoke 发布资产](https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt) |
| Demucs `htdemucs_6s` | `models/review/demucs/5c90dfd2-34c22ccb.th` | 55 MB | [Demucs 权重](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th) |
| Basic Pitch | `models/review/basic_pitch.onnx` | 230 KB | 安装 `basic-pitch==0.4.0` 后，由准备脚本复制包内 ONNX 模型 |

例如，BS-RoFormer 的完整路径为：

```text
/mnt/afs_f/liuhongda/music-cover-baseline/models/separation/model_bs_roformer_ep_317_sdr_12.9755.ckpt
```

不要只复制上表的大文件。Hugging Face 模型需要保留下载脚本准备的整个模型目录，包括配置、分词器和自定义 Python 实现。下列配套文件也要保留：

| 模型 | 配套配置及说明 |
| --- | --- |
| SheetSage2、MERT、YuE2-3B、YuE2-Vae、Whisper-small | 各自模型目录中的配置、处理器/分词器及 Python 文件，按下载脚本完整保留 |
| BS-RoFormer | `models/separation/model_bs_roformer_ep_317_sdr_12.9755.yaml` |
| Seed-VC | `models/Seed-VC/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml`；实际运行使用派生配置 `configs/seed_svc.local.yml` |
| BigVGAN | `models/BigVGAN/config.json` |
| Mel-Band RoFormer Karaoke | 准备脚本登记 `models/separation/config_mel_band_roformer_karaoke.yaml`；现有 `audio-separator` 运行目录还包含 `models/separation/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956_config.yaml`，迁移时一并保留 |
| Demucs | `models/review/demucs/htdemucs_6s.yaml`，其中指定对应的 `.th` 权重 |

`audio-separator` 首次加载还可能获取模型目录信息或配套配置，不能仅凭权重文件齐全就认定已支持完全离线运行。准备离线机器时，应复制已验证的分离模型目录及相关缓存，并完成一次实际加载/短段推理验证。

Seed-VC 主权重名称包含 `whisper_base`，但本项目的实际运行配置使用 **Whisper-small**；应按当前配置与锁文件准备，不要凭主权重文件名另换编码器。

## 3. 固定版本与校验记录在哪里

Hugging Face 下载脚本固定以下 revision，不以最新 `main` 替代。链接打开对应的固定模型版本：

| 模型 | 固定 revision |
| --- | --- |
| [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B/tree/8f24312f187763b9854adeec87e437e2b03bbff1) | `8f24312f187763b9854adeec87e437e2b03bbff1` |
| [YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae/tree/152733a19ad43aa67e367f9b5503ef8075bb5126) | `152733a19ad43aa67e367f9b5503ef8075bb5126` |
| [SheetSage2](https://huggingface.co/m-a-p/SheetSage2/tree/b83c3d79a6d98bcffaf8f9ed112c90aa887887bc) | `b83c3d79a6d98bcffaf8f9ed112c90aa887887bc` |
| [MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong/tree/7001f7657276238629e61c7746ce70735e0604c0) | `7001f7657276238629e61c7746ce70735e0604c0` |
| [Seed-VC](https://huggingface.co/Plachta/Seed-VC/tree/257283f9f41585055e8f858fba4fd044e5caed6e) | `257283f9f41585055e8f858fba4fd044e5caed6e` |
| [BigVGAN](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x/tree/95a9d1dcb12906c03edd938d77b9333d6ded7dfb) | `95a9d1dcb12906c03edd938d77b9333d6ded7dfb` |
| [Whisper-small](https://huggingface.co/openai/whisper-small/tree/973afd24965f72e36ca33b3055d56a652f456b4d) | `973afd24965f72e36ca33b3055d56a652f456b4d` |
| [RMVPE](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/e6d0c1a17da07c33557852f9dfa2bd44cc75737d) | `e6d0c1a17da07c33557852f9dfa2bd44cc75737d` |
| [CAMPPlus](https://huggingface.co/funasr/campplus/tree/e4b6ede7ce16997aff4ae69fbca1f0175e2afede) | `e4b6ede7ce16997aff4ae69fbca1f0175e2afede` |

| 清单 | 记录内容 |
| --- | --- |
| [generation_models.lock.json](../configs/generation_models.lock.json) | 4 组转谱/生成模型的仓库、revision、下载目录 |
| [voice_models.lock.json](../configs/voice_models.lock.json) | 5 组换声/音高模型的仓库、revision、下载目录 |
| [separation_model.lock.json](../configs/separation_model.lock.json) | BS-RoFormer 权重与配置的 URL、大小/哈希 |
| [local_assets.lock.json](../configs/local_assets.lock.json) | 已归档本地模型文件的大小与 SHA256 快照 |
| [review_models.lock.json](../configs/review_models.lock.json) | Karaoke、Demucs、Basic Pitch 及评审配置的 URL、大小、SHA256、包版本 |

`generation_models.lock.json` 和 `voice_models.lock.json` 的 revision 记录不是逐文件 SHA256。下载脚本会更新相应锁文件；首次恢复时先保留 Git 中的已验证记录，再与下载结果比较。分离/评审下载脚本通常记录本次实际哈希，只有 Demucs 额外检查上游文件名中的短哈希；“脚本运行成功并写出新哈希”不等于已与原验证版本比对。

## 4. 在服务器准备模型：按顺序执行

**现有 7830 服务器已经具备这些权重，正常运行无需重新下载；可以直接执行第 5 节检查。** 以下命令适用于恢复模型的情况，前提是项目的 `envs/control`、`envs/separator` 等环境已配置。它们不是从空白机器安装全部 Python/CUDA 依赖的一键脚本。

### 4.1 初始化路径，保留恢复前的版本记录

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate base
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"

# 下载脚本会更新部分锁文件；保留当前分支的原始记录，便于比较。
mkdir -p cache/model-restore-baseline
git archive HEAD configs/generation_models.lock.json configs/voice_models.lock.json \
  configs/separation_model.lock.json configs/local_assets.lock.json \
  configs/review_models.lock.json | tar -x -C cache/model-restore-baseline

# 下载脚本依赖检查；若导入失败，应先补齐 control 环境。
envs/control/bin/python -c "import huggingface_hub, yaml; print('download dependencies OK')"
```

新机器的项目路径不同，应先修改上面的 `cd`。环境构建前提见[仓库恢复说明](08-repository.md)。

### 4.2 下载转谱与生成模型（4 组）

```bash
envs/control/bin/python scripts/download_models.py
```

得到 `models/SheetSage2/`、`models/MERT-v2-FullSong/`、`models/YuE2-3B/` 和 `models/YuE2-Vae/`。对应实现是 [scripts/setup/download_models.py](../scripts/setup/download_models.py)。

### 4.3 下载换声与音高模型（5 组）

```bash
envs/control/bin/python scripts/download_voice_models.py
```

得到 `models/Seed-VC/`、`models/Whisper-small/`、`models/CAMPPlus/`、`models/RMVPE/` 和 `models/BigVGAN/`。脚本同时根据**当前项目根目录**生成 `configs/seed_svc.local.yml`，更新其中的声码器和编码器绝对路径。对应实现是 [scripts/setup/download_voice_models.py](../scripts/setup/download_voice_models.py)。

### 4.4 下载基础人声分离模型（1 组）

```bash
envs/control/bin/python scripts/download_separator_model.py
```

得到 `models/separation/model_bs_roformer_ep_317_sdr_12.9755.ckpt` 和配套 YAML。该模型同时服务 02 的原曲人声分析及 04 的生成歌曲分离。对应实现是 [scripts/setup/download_separator_model.py](../scripts/setup/download_separator_model.py)。

### 4.5 准备分轨与 MIDI 评审模型（3 组）

```bash
# 这里复用已配置的 separator 环境；此清单仅包含附加依赖。
envs/separator/bin/python -m pip install --no-deps -r configs/review-requirements.txt
envs/separator/bin/python scripts/setup_review_assets.py
```

得到 Karaoke 权重/配置、Demucs 权重/配置，以及 `models/review/basic_pitch.onnx`。Basic Pitch 模型从安装包复制，运行使用 CPU ONNX；无需为此安装 TensorFlow。对应实现是 [scripts/setup/setup_review_assets.py](../scripts/setup/setup_review_assets.py)。

如果已经在别处下载了评审资源，可把下列四个同名文件放到 `/mnt/afs_f/liuhongda/music-cover-baseline/cache/review-downloads/`：

```text
mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt
config_mel_band_roformer_karaoke.yaml
5c90dfd2-34c22ccb.th
htdemucs_6s.yaml
```

然后改用：

```bash
envs/separator/bin/python scripts/setup_review_assets.py \
  --local-assets /mnt/afs_f/liuhongda/music-cover-baseline/cache/review-downloads
```

缺少的文件仍会尝试联网获取，已存在的目标文件不会因此被覆盖；Basic Pitch 仍从本机安装包复制。这个参数仅适用于评审准备脚本，其他三个下载脚本没有 `--local-assets` 参数。

## 5. 准备后怎么验证

### 5.1 完整任务预检

在项目根目录执行：

```bash
export PATH="/opt/conda/bin:$PATH"
bash scripts/run_baseline.sh \
  --config configs/tasks/qiu_yi_nong_bossa_nova_sun_yanzi.yaml \
  --preflight
```

该配置对应孙燕姿＋秋意浓＋Bossa Nova / Samba Jazz，启用自动选调和完整评审。预检检查选中后端要求的文件、依赖导入、FFmpeg/MP3 编码支持，并对评审资产进行大小及 SHA256 校验。成功会显示 `Selected module preflight OK`；不会开始生成整首歌曲。

当前预检不是所有模型的逐文件校验，也不加载所有大模型完成推理。完整运行是否可用，还应按[逐模块实操手册](11_modular_walkthrough.md)执行并检查结果。

### 5.2 按已验证快照比对权重哈希

下面只读检查已登记的主权重、评审配置及 BS-RoFormer 配置；不下载、不改模型，也不更新锁文件。若执行过 4.1，优先使用该步骤保存的原始记录，避免拿刚生成的哈希与自身比较。现有服务器可直接执行：

```bash
envs/control/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path.cwd()
saved = root / 'cache/model-restore-baseline/configs'
locks = saved if saved.is_dir() else root / 'configs'
print('Compare against:', locks)
local = json.loads((locks / 'local_assets.lock.json').read_text())
weight_suffixes = {'.safetensors', '.pth', '.pt', '.bin', '.ckpt', '.th', '.onnx'}
expected = {p: item for p, item in local.items()
            if Path(p).suffix in weight_suffixes}
review = json.loads((locks / 'review_models.lock.json').read_text())
expected.update({item['path']: item for item in review['assets']})
sep = json.loads((locks / 'separation_model.lock.json').read_text())
expected['models/separation/model_bs_roformer_ep_317_sdr_12.9755.ckpt'] = {
    'bytes': sep['size'], 'sha256': sep['sha256']}
expected['models/separation/model_bs_roformer_ep_317_sdr_12.9755.yaml'] = {
    'sha256': sep['config_sha256']}
failures = []
for relative, record in sorted(expected.items()):
    path = root / relative
    ok = path.is_file()
    if ok and 'bytes' in record:
        ok = path.stat().st_size == record['bytes']
    if ok:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                digest.update(block)
        ok = digest.hexdigest() == record['sha256']
    print('OK' if ok else 'MISSING/DIFFERENT', relative, flush=True)
    if not ok:
        failures.append(relative)
print(f'{len(expected) - len(failures)}/{len(expected)} files matched')
raise SystemExit(1 if failures else 0)
PY
```

读取大权重计算哈希会产生磁盘 I/O。若出现不匹配，应核对下载来源、固定 revision 和具体文件，不能通过重新生成锁文件把差异掩盖掉。本检查不覆盖完整 Python 环境、全部模型配置或未登记的运行时缓存。

## 6. 权重与模型代码分别在哪里

| 用途 | 项目调用入口 | 模型实现来源 |
| --- | --- | --- |
| 转谱 | [backends/sheetsage.py](../src/music_cover/backends/sheetsage.py) | `models/SheetSage2/` 与 `models/MERT-v2-FullSong/` 内固定版本的自定义实现 |
| 生成 | [backends/yue.py](../src/music_cover/backends/yue.py) | `third_party/YuE/` 子模块及对应模型资产 |
| BS-RoFormer 分离 | [backends/separator.py](../src/music_cover/backends/separator.py) | `audio-separator==0.30.2` 安装包 |
| 换声 | [backends/seed_vc.py](../src/music_cover/backends/seed_vc.py) | `third_party/seed-vc/` 子模块 |
| 音高提取 | [backends/rmvpe.py](../src/music_cover/backends/rmvpe.py) | `third_party/seed-vc/modules/rmvpe.py` |
| 成品分轨 | [backends/review_stems.py](../src/music_cover/backends/review_stems.py) | `audio-separator==0.30.2` 与 `demucs==4.0.1` |
| 分轨 MIDI | [backends/midi.py](../src/music_cover/backends/midi.py) | `basic-pitch==0.4.0` + ONNX Runtime |

现有服务器 BS-RoFormer 网络结构的具体文件：

```text
/mnt/afs_f/liuhongda/music-cover-baseline/envs/separator/lib/python3.11/site-packages/audio_separator/separator/uvr_lib_v5/roformer/bs_roformer.py
```

加载权重与分块推理的包装实现：

```text
/mnt/afs_f/liuhongda/music-cover-baseline/envs/separator/lib/python3.11/site-packages/audio_separator/separator/architectures/mdxc_separator.py
```

这些 `site-packages` 路径取决于 Python 版本和安装环境，不是 GitHub 中本项目的源码目录。修改调用参数或选择权重时查看项目后端；若修改依赖库网络结构，应单独记录修改和版本，避免迁移时丢失。

## 7. 常见问题定位

| 现象 | 先检查什么 |
| --- | --- |
| 克隆仓库后提示缺少模型 | Git 不带 `models/`；按第 4 节恢复，并保留整个模型目录 |
| 02/10 缺 `rmvpe.pt` | 运行换声模型下载脚本；RMVPE 同时服务选调、换声与音高评估 |
| 已有 Seed-VC 主权重仍不能换声 | 检查 Whisper-small、CAMPPlus、RMVPE、BigVGAN 及 `configs/seed_svc.local.yml` |
| 迁移后仍引用旧 `/data/...` 路径 | 重新执行换声模型准备脚本生成本机配置；锁文件中的历史绝对路径也应按迁移后的实际位置更新 |
| 07/09 缺模型，基础歌曲却能生成 | 补齐评审依赖，并执行 `setup_review_assets.py`；只有不需要评审输出时才使用 `--no-review` |
| BS-RoFormer/Karaoke 权重存在但报配置缺失 | 检查对应 YAML、`audio-separator` 版本及首次加载时的网络/缓存 |
| `ffmpeg is not available` | 当前服务器执行 `export PATH="/opt/conda/bin:$PATH"`，再运行预检 |
| 不知道该下载哪个人物模型 | 此路线使用通用 Seed-VC 权重，人物参考音频放在 `data/raw/references/`，通过 `--reference` 指定 |

本清单反映当前实现的实际依赖。后续替换模型时，应同步修改后端配置、下载脚本、锁记录和本文，并重新验证受影响模块。
