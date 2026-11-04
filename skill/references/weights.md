# 模型权重与下载命令

所有文件位置相对项目根目录。以下链接来自本仓库下载器/资产锁，不使用上游最新 main 替换固定版本。Hugging Face 链接指向模型快照目录时，应保留整个下载器选取的目录，包括 JSON、分词器、自定义 Python 和配置；不要只下载主权重。

## 01–06 公共模型

| 节点/组件 | 本地主要位置 | 固定版本下载来源 |
|---|---|---|
| 01 SheetSage2 | `models/SheetSage2/model.safetensors` | [SheetSage2](https://huggingface.co/m-a-p/SheetSage2/tree/b83c3d79a6d98bcffaf8f9ed112c90aa887887bc) |
| 01 MERT 编码器 | `models/MERT-v2-FullSong/model.safetensors` | [MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong/tree/7001f7657276238629e61c7746ce70735e0604c0) |
| 02/04 BS-RoFormer | `models/separation/model_bs_roformer_ep_317_sdr_12.9755.ckpt` | [权重](https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt)、[配套 YAML](https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_317_sdr_12.9755.yaml) |
| 02/05 RMVPE | `models/RMVPE/rmvpe.pt` | [RMVPE](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/e6d0c1a17da07c33557852f9dfa2bd44cc75737d/rmvpe.pt) |
| 03 YuE2 生成模型 | `models/YuE2-3B/model.safetensors` | [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B/tree/8f24312f187763b9854adeec87e437e2b03bbff1) |
| 03 YuE2 音频解码 | `models/YuE2-Vae/model.safetensors` | [YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae/tree/152733a19ad43aa67e367f9b5503ef8075bb5126) |
| 05 Seed-VC 歌唱模型 | `models/Seed-VC/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth` | [Seed-VC](https://huggingface.co/Plachta/Seed-VC/tree/257283f9f41585055e8f858fba4fd044e5caed6e) |
| 05 Seed-VC 结构配置 | `models/Seed-VC/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml` | [同一快照](https://huggingface.co/Plachta/Seed-VC/tree/257283f9f41585055e8f858fba4fd044e5caed6e)；下载器派生 `configs/seed_svc.local.yml` |
| 05 Whisper-small | `models/Whisper-small/model.safetensors` 及配置 | [Whisper-small](https://huggingface.co/openai/whisper-small/tree/973afd24965f72e36ca33b3055d56a652f456b4d) |
| 05 CAMPPlus | `models/CAMPPlus/campplus_cn_common.bin` | [CAMPPlus](https://huggingface.co/funasr/campplus/resolve/e4b6ede7ce16997aff4ae69fbca1f0175e2afede/campplus_cn_common.bin) |
| 05 BigVGAN | `models/BigVGAN/bigvgan_generator.pt`、`config.json` | [BigVGAN](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x/tree/95a9d1dcb12906c03edd938d77b9333d6ded7dfb) |
| 微调候选/训练验证 ECAPA | `models/voice_evaluation/ecapa/embedding_model.ckpt`、`hyperparams.yaml` | [ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/tree/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286) |

00 和 06 无新增模型。`transpose=0` 时 02 可跳过分析，但 04 仍需 BS-RoFormer。Seed-VC 虽然主文件名有 `whisper_base`，本工程实际用 Whisper-small，不要擅自替换。

### 公共主流程下载

首次准备时先备份现有锁记录；下列三个下载器会更新锁，Seed-VC 下载器还会按本机位置重建运行配置。已有权重时先检查，不重复覆盖用户改过的配置。

```bash
mkdir -p cache
DEPLOY_LOCK_BACKUP=$(mktemp -d "$PWD/cache/weights-before-deploy.XXXXXX")
cp configs/generation_models.lock.json configs/voice_models.lock.json configs/separation_model.lock.json configs/seed_svc.local.yml "$DEPLOY_LOCK_BACKUP/"

/opt/conda/bin/python scripts/setup/download_models.py
/opt/conda/bin/python scripts/setup/download_separator_model.py
/opt/conda/bin/python scripts/setup/download_voice_models.py

# 选择任意微调/SVC/RVC 路线时还需要评估资产；零样本 05 本身不要求 ECAPA。
/opt/conda/bin/python scripts/setup/setup_voice_training.py --scope evaluation
/opt/conda/bin/python scripts/setup/setup_voice_training.py --scope evaluation --check-only
```

`download_voice_models.py` 会一次准备五组 Seed-VC/F0 资产；仅用 SVC/RVC 时其中部分不是推理必需，但 02 和质量评估仍需 `models/RMVPE/rmvpe.pt`。按需精简时以 `common/resources.py` 的 requirements 为准，不能删除复用资源。

HF revision 是快照版本，并非每个文件的 SHA256。BS-RoFormer 下载器会写入本次哈希，必须与备份的 `separation_model.lock.json` 比对；不能仅因新锁写出就声称原始哈希校验通过。其 YAML URL 未固定 commit，下载内容若与仓库记录不同应调查，不静默接受漂移。

## SVC 与 RVC 公共资产

| 路线/组件 | 本地路径 | 下载链接 |
|---|---|---|
| SVC ContentVec 768l12 | `models/SoVitsSVC/contentvec.pt` | [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/e6d0c1a17da07c33557852f9dfa2bd44cc75737d/hubert_base.pt) |
| SVC RMVPE | `models/SoVitsSVC/rmvpe.pt` | [rmvpe.zip](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip)，由脚本取出 `model.pt` 并按锁重命名 |
| SVC 预训练生成器，仅训练 | `models/SoVitsSVC/G_0.pth` | [G_0.pth](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/resolve/a42897099a881c0545a466da1161933785e19fa2/vec768l12/G_0.pth) |
| SVC 预训练判别器，仅训练 | `models/SoVitsSVC/D_0.pth` | [D_0.pth](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/resolve/a42897099a881c0545a466da1161933785e19fa2/vec768l12/D_0.pth) |
| RVC HuBERT/ContentVec | `models/RVC/hubert_base/pytorch_model.bin`、`config.json`、`preprocessor_config.json` | [完整子目录](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/e6d0c1a17da07c33557852f9dfa2bd44cc75737d/hubert_base) |
| RVC RMVPE | `models/RVC/rmvpe.pt` | [rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/e6d0c1a17da07c33557852f9dfa2bd44cc75737d/rmvpe.pt) |
| RVC v2/F0/48k 预训练生成器，仅训练 | `models/RVC/pretrained_v2/f0G48k.pth` | [f0G48k.pth](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/e6d0c1a17da07c33557852f9dfa2bd44cc75737d/pretrained_v2/f0G48k.pth) |
| RVC v2/F0/48k 预训练判别器，仅训练 | `models/RVC/pretrained_v2/f0D48k.pth` | [f0D48k.pth](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/e6d0c1a17da07c33557852f9dfa2bd44cc75737d/pretrained_v2/f0D48k.pth) |

SVC 的 RMVPE 与 RVC/Seed-VC 的 RMVPE 文件格式、大小不同，不能因同名就互相覆盖。G/D 是公共训练底模，不是人物音色成品。

```bash
# 只运行选定路线；推理已有人物模型时省略 --training。
/opt/conda/bin/python scripts/setup_svc.py --download --training
/opt/conda/bin/python scripts/setup_rvc.py --download --training
# 不下载，仅检查：
/opt/conda/bin/python scripts/setup_svc.py --training
/opt/conda/bin/python scripts/setup_rvc.py --training
```

这两个脚本根据 `configs/so_vits_svc_assets.lock.json` / `configs/rvc_assets.lock.json` 验证大小和 SHA256，不负责安装 Python 包，也不会训练人物模型。默认 SVC 路线为 ContentVec 768l12 + RMVPE + 标准 NSF-HiFiGAN，未使用浅层扩散或增强器；标准声码器在生成器模型内，不需要另下扩散权重。

## 新训练人物音色时的附加资产

| 用途 | 本地路径 | 固定来源 |
|---|---|---|
| Seed-VC 训练音色扰动 | `models/OpenVoiceV2/converter/checkpoint.pth`、`config.json` | [OpenVoiceV2](https://huggingface.co/myshell-ai/OpenVoiceV2/tree/f36e7edfe1684461a8343844af60babc2efbb727/converter) |
| Seed-VC 训练说话人库 | `models/Seed-VC/se_db.pt` | [se_db.pt](https://huggingface.co/Plachta/Seed-VC/resolve/257283f9f41585055e8f858fba4fd044e5caed6e/se_db.pt) |
| 原始录音两级主唱分离 | `models/separation/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` | [Karaoke 权重](https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt)；配套配置见 `configs/review_models.lock.json` |

```bash
# 只在准备 Seed-VC 训练时需要 training scope。
/opt/conda/bin/python scripts/setup/setup_voice_training.py --scope all
/opt/conda/bin/python scripts/setup/setup_voice_training.py --scope all --check-only
```

录音分离默认 `--mode lead`，需 BS-RoFormer 与 Karaoke 两套权重及 YAML。现有 `setup_review_assets.py` 同时处理 Demucs/Basic Pitch；只为训练分离素材时，不必连带部署 07–09。可用现有 audio-separator 的下载模式准备 Karaoke，再检查仓库锁定哈希：

```bash
/opt/conda/bin/audio-separator \
  --model_filename mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt \
  --model_file_dir models/separation --download_model_only
```

包可能将运行配置命名为 `mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956_config.yaml`。锁中另外登记的 `config_mel_band_roformer_karaoke.yaml` 可按 [配套配置链接](https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/config_mel_band_roformer_karaoke.yaml) 下载；保留已验证的配套配置。当前锁定主权重 SHA256 为 `1de20d459332fe8869aeb01327a31df0032262706e1365114e852dc271779813`，锁中 YAML SHA256 为 `30364842112e18500aca7ff7b1acf235251eb4ee93b377278176b432410eae33`，以随仓库发布的锁为准。之后使用 `scripts/voice_data/separate/run.sh ... --plan` 检查；下载模式本身不等于推理成功。下载失败应保留原因，不用同名未知权重替代。

## 人物模型部署

以下没有公共下载链接，由用户训练或从自己的可信备份恢复：

| 路线 | 整个导出目录 | 主要内容 |
|---|---|---|
| Seed-VC | `models/voices/my_voice/seed_vc/v001/` | `voice.json`、`model.pth`、`config.yml`、`validation.json` |
| SVC | `models/voices/my_voice/so_vits_svc/v001/` | `voice.json` 及其登记的模型、结构配置、验证报告 |
| RVC | `models/voices/my_voice/rvc/v001/` | `voice.json` 及其登记的模型、结构配置、检索索引、索引清单和验证报告 |

以 `voice.json` 记录的实际文件名和哈希为准。迁移时保留完整目录，不手改已签名模型文件；旧档案内嵌路径需要与当前解析器核对。训练和导出命令见 [training.md](training.md)。主流程的 `--voice-profile` 指向 `voice.json`。
