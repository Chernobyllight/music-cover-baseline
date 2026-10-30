# v1.0 模型权重清单与准备方法

本文集中说明：**每个模块需要什么模型、从哪里获取、放在哪里、怎样检查，以及对应代码在哪里。** 按 2026-10-30 的 v1.0 代码与服务器资产核对，覆盖 Seed-VC、SVC、RVC 和 00–11 评审流程。基础模型沿用 v0.2，微调/评估依赖见第 9 节。

原有零样本完整流程（`--transpose auto --review`）需要 **13 组模型资产**，包含主模型和附属模型；同一个模型可被多个阶段复用。GitHub 仓库不包含这些大文件。孙燕姿或詹姆斯的音色由参考录音提供，零样本 Seed-VC 路线无需单独下载对应人物的专用权重。新增微调路线可用目标歌手录音训练专用版本，所需附加资产见本文末尾。

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
| 不知道该下载哪个人物模型 | 零样本路线使用通用 Seed-VC；微调路线加载本地训练导出的 voice.json，两者仍通过 --reference 指定参考片段 |

本清单反映当前实现的实际依赖。后续替换模型时，应同步修改后端配置、下载脚本、锁记录和本文，并重新验证受影响模块。


## 9. 可选：目标音色微调和独立评估资产

原有零样本接口保持独立，不要求安装下列训练资产。完整逐步命令见 [音色微调手册](13_voice_adaptation.md)。

| 资产 | 本地文件 | 用途 | 大小（十进制） |
| --- | --- | --- | --- |
| OpenVoiceV2 converter | `models/OpenVoiceV2/converter/checkpoint.pth`、`config.json` | 训练时扰动输入音色，模型自身冻结 | 约 131 MB |
| Seed-VC 说话人向量库 | `models/Seed-VC/se_db.pt` | 提供随机音色扰动目标，固定不训练 | 约 102 MB |
| SpeechBrain ECAPA-TDNN | `models/voice_evaluation/ecapa/embedding_model.ckpt`、`hyperparams.yaml` | 独立音色相似度评估，冻结 | 约 83 MB |
| 自行微调的 Seed-VC | `models/voices/<voice_id>/seed_vc/<version>/model.pth` | 05 finetuned 模式使用；配套 config.yml、voice.json、validation.json | 取决于导出模型 |

前三项在 `configs/voice_training_assets.lock.json` 固定仓库 revision、文件路径、用途、字节数和 SHA256；下载后校验完整哈希。基础 Seed-VC、Whisper-small、CAMPPlus、RMVPE 和 BigVGAN 仍沿用前文资产，不会覆盖通用主权重。

| 来源 | 固定 revision |
| --- | --- |
| [myshell-ai/OpenVoiceV2](https://huggingface.co/myshell-ai/OpenVoiceV2/tree/f36e7edfe1684461a8343844af60babc2efbb727) | `f36e7edfe1684461a8343844af60babc2efbb727` |
| [Plachta/Seed-VC](https://huggingface.co/Plachta/Seed-VC/tree/257283f9f41585055e8f858fba4fd044e5caed6e) | `257283f9f41585055e8f858fba4fd044e5caed6e` |
| [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/tree/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286) | `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` |

```bash
cd /mnt/afs_f/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
envs/seed-vc/bin/python -m pip install --no-deps \
  -r configs/requirements-voice-adaptation.txt
envs/seed-vc/bin/python scripts/setup/setup_voice_training.py --scope all
# 只检查，不重新下载。
envs/seed-vc/bin/python scripts/setup/setup_voice_training.py --scope all --check-only
```

`--scope training` 只准备 OpenVoice 和向量库；`--scope evaluation` 只准备 ECAPA。独立验证和主流程10的 voice_check 都会核验评估资产。ECAPA 使用原始编码器向量的单位归一化余弦，不使用说话人分类头；hyperparams.yaml 保留用于追溯结构。训练 checkpoint 含优化器等状态，体积显著大于只含网络参数的导出模型，不要将两者混用。

## 微调候选评估需要的权重

05 的 Seed-VC / RVC `finetuned` 模式默认产生三个候选并自动评估，SVC 当前使用一个候选；即使只有一个候选，也需要冻结的 ECAPA-TDNN 和 RMVPE。零样本 05 仍不需要 ECAPA；10 的可选 voice_check 启用时才需要。加载已导出的通用音色模型进行推理，不需要 OpenVoiceV2 或训练向量库。

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
envs/seed-vc/bin/python scripts/setup/setup_voice_training.py --scope evaluation
```

按本文件前述环境准备步骤安装评估依赖；该命令准备并校验 `configs/voice_training_assets.lock.json` 中 evaluation 资产。模型位置为 `models/voice_evaluation/ecapa/embedding_model.ckpt` 和 `models/RMVPE/rmvpe.pt`，音色 profile 中列出的 checkpoint/config/validation 三个文件均须保留。

当前杨坤实例使用 `models/voices/yang_kun/seed_vc/catalog_v001_20261027/voice.json`，`selected_step=1000`。此 profile 为用户指定 checkpoint，状态是 `unverified_user_selected`，不等于通过独立测试集模型选优。模型 SHA256 为 `292cae9459eba3fad2905beec480857d5d3f57d818c3ee2a06f21edcc3a7674a`；仓库只提供任务配置与使用说明，不分发歌手模型或录音。

本版主流程不调用歌词 ASR、歌词门禁或静音修复；不需要此前实验使用的 Whisper large-v3、Paraformer 或 Qwen3-ASR 权重。表内 Whisper-small 是 Seed-VC 的内容编码器，仍为换声必需资源。


## 新增 so-vits-svc 4.1 后端

支持独立歌手微调与 05 音色转换，复用已有 base 环境；Seed-VC 零样本与微调入口继续保留。SVC 目前一个候选，不支持零样本。训练数据、权重清单、完整示例及中间产物见 [SVC 接入指南](18_so_vits_svc.md)。


SVC 独立权重（均在工程根目录下）：

| 文件 | 作用 |
|---|---|
| `models/SoVitsSVC/contentvec.pt` | 固定 ContentVec 第 12 层内容编码器 |
| `models/SoVitsSVC/rmvpe.pt` | SVC 兼容格式 RMVPE；与 Seed-VC 同名文件不可直接互换 |
| `models/SoVitsSVC/G_0.pth`、`D_0.pth` | 训练底模；导出后的推理不需要 D |
| `models/voices/<歌手>/so_vits_svc/<版本>/model.pth`、`config.json`、`voice.json` | 训练后歌手模型及不可拆分的配置、身份与校验信息 |

固定来源、大小和 SHA256 见 [so_vits_svc_assets.lock.json](../configs/so_vits_svc_assets.lock.json)。下载/校验使用 `/opt/conda/bin/python scripts/setup_svc.py --download --training`。SVC 当前使用 base，保留现有核心包；不安装 fairseq，不需要额外扩散或增强器权重。

## SVC优化实验可选权重（2026-10-28）

直接使用推荐杨坤模型只需要原SVC推理资产ContentVec/RMVPE及 `models/voices/yang_kun/so_vits_svc/optimized_v002_20261028/` 的profile、model/config/validation；用 `configs/voice_inference/so_vits_svc/yang_kun_optimized.yaml`指定推理参数。

复现全部实验额外需要 `configs/so_vits_svc_experiments_assets.lock.json` 中的9个资产：vec256l9、HuBERT-Soft、Whisper-PPG匹配生成器，HuBERT-Soft编码器、Whisper-medium编码器，full/shallow diffusion底模、NSF增强器压缩包、FCPE权重。它们放在 `models/SoVitsSVC/optimization_20261028/`，具有来源URL、大小和SHA256；由 `scripts/05_voice_conversion/so_vits_svc/experiments/assets.py --download` 下载并验证，不是普通Seed-VC推理的依赖。详见 [实验安装命令](../scripts/05_voice_conversion/so_vits_svc/experiments/README.md)和[优化报告](19_svc_optimization.md)。

## RVC 独立权重（v1.0）

RVC 固定源码版本、下载地址、revision、字节数与 SHA256 在 `configs/rvc_assets.lock.json`；不复用并覆盖 Seed-VC/SVC 的权重文件。

| 路径（工程根目录下） | 用途 |
|---|---|
| `models/RVC/hubert_base/{config.json,preprocessor_config.json,pytorch_model.bin}` | 768 维内容特征 |
| `models/RVC/rmvpe.pt` | F0 与清浊音特征 |
| `models/RVC/pretrained_v2/f0G48k.pth` | 初始化 48 kHz F0 生成器 |
| `models/RVC/pretrained_v2/f0D48k.pth` | 初始化判别器，仅训练需要 |
| `models/voices/huang_qi_shan/rvc/v001_20261029/voice.json` | 最终 4000 步黄绮珊模型档案 |
| 同目录 `model.pth`、`config.json`、`added.index`、`index_manifest.json`、`validation.json` | 推理模型、训练集索引、来源与验收记录，缺一不可 |

在工程根目录执行 `/opt/conda/bin/python scripts/setup_rvc.py --training --download` 获取缺少的公共资产并校验（服务器必须能访问下载地址）；本地已有资产时省略 `--download`。人物专用模型需按 [RVC 指南](20_rvc.md)训练并导出，不能从公共底模推断任意歌手音色。导出是本工程严格加载的模型包，不宣称直接兼容上游 WebUI 的模型导入格式。

## 当前可用人物模型与保留策略（2026-10-29）

以下路径相对工程根目录，已导出人物模型需连同同目录的配置、校验记录和索引整体保留：

| 用途 | profile | 已使用训练步数 |
|---|---|---:|
| 黄绮珊 Seed-VC | `models/voices/huang_qi_shan/seed_vc/v001_20261029/voice.json` | 1000 |
| 黄绮珊 RVC | `models/voices/huang_qi_shan/rvc/v001_20261029/voice.json` | 4000 |
| 杨坤 SVC 推荐 | `models/voices/yang_kun/so_vits_svc/optimized_v002_20261028/voice.json` | 6000（4000＋2000） |
| 杨坤 Seed-VC | `models/voices/yang_kun/seed_vc/catalog_v001_20261027/voice.json` | 1000 |

孙燕姿、詹姆斯的既有 Seed-VC，以及杨坤初版 SVC profile 也保留在 `models/voices/`。名称中的日期与步数是模型版本信息，不代表本次又做了完整训练或选模。新人物的模型使用各自训练入口生成，不能只修改 `voice_id` 复用另一人物权重。

RVC 默认预设 `configs/voice_inference/rvc/huang_qi_shan.yaml` 使用检索比例 0.5；`huang_qi_shan_index075.yaml` 显式使用 0.75，同一份 4000 步权重和索引。两者不是两个不同训练模型。SVC 优化模型使用 ContentVec 768l12 / RMVPE / 标准 NSF-HiFiGAN，推荐推理未开启扩散或增强器；实验资产不是默认推理的必需项。
