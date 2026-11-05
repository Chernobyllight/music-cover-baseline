# Fish Diffusion 部署、训练与 05 接入

本路线独立位于 `scripts/05_voice_conversion/fish_diffusion/`，后端 `fish_diffusion`、模式 `finetuned`。训练、特征、推理和评估使用 base `/opt/conda/bin/python`；不改变主流程 00–04 或其他模型的解释器。本功能随 v1.2 提供；部署前获取该版本并核对独立资产锁和兼容补丁。

## 选择操作手册

- 亲自从原始录音创建全部配置和中间结果：[13 步手动实操](24_fish_diffusion_manual.md)。每步包含输入、输出与命令，采用新版本 `manual_v001`。
- 新机器、不同歌手：[部署 Skill](../skill/SKILL.md)、[通用训练步骤](../skill/references/training.md)、[权重下载](../skill/references/weights.md)。服务器素材不随仓库分发。
- 了解 5000 步方案的选择依据与限制：[优化实验](23_fish_diffusion_optimization.md)。其中旧 1000 步是历史接入验证，用户反馈音色不佳，不是当前推荐质量方案。

## 模型与训练范围

当前组合是 ContentVec 第 9 层投影到 256 维、Parselmouth F0、DiffSinger/WaveNet 扩散声学模型和 NSF-HiFiGAN。44.1 kHz、hop=512、128 维 log10 Mel、log F0、单歌手 embedding、20 层 WaveNet、1000 个扩散时间点。

公共底模的歌手 embedding 已移除；微调初始化新的单歌手 embedding，并更新整个声学模型（55,061,632 个参数）。ContentVec 与声码器冻结，Parselmouth 是固定算法；没有 LoRA 或只训 embedding。人物音色来自模型，不用主流程参考作为推理条件，不是 Seed-VC 式任意参考零样本转换，也不是 Fish Speech/商业 Voice Changer。

当前示例训练 5000 步，batch=4、crop=384 帧、bf16、seed=42。AdamW 的 betas=(0.9,0.98)、eps=1e-9、weight_decay=0.01；梯度范数裁剪 1。学习率从 1e-5 预热 1000 步到 4e-4，再余弦衰减到第 5000 步的 2e-5。仅训练集做 [-5,+5] 半音增强，probability=1.5 表示额外一次增强及 50% 概率的第二次；与推理移调无关。

5000 是本次经验配置，可显式修改，不能当成官方固定预算。配置生成器会提示训练结束于预热中或早于衰减终点，但不禁止用户进行短流程检查。

## 环境与公共权重部署

先按 [环境说明](03-environment.md)核对 base。必要依赖清单为 `configs/requirements-fish-diffusion.txt`；`configs/fish-diffusion-base.installed.txt` 仅是机器快照。新机器不照装上游整套旧 requirements，已有机器只在确认缺项后安装，并保留原包约束：

```bash
mkdir -p cache/fish_setup
/opt/conda/bin/python -m pip list --format=freeze > cache/fish_setup/base-before.txt
/opt/conda/bin/python -m pip install -c cache/fish_setup/base-before.txt \
  -r configs/requirements-fish-diffusion.txt
bash scripts/05_voice_conversion/fish_diffusion/download_assets/run.sh
/opt/conda/bin/python scripts/setup/setup_voice_training.py --scope evaluation
bash scripts/05_voice_conversion/fish_diffusion/preflight/run.sh \
  --scope training --model-load --gpu 0
```

以上从实际项目根目录执行。下载脚本准备源码和权重，不安装包、不训练人物；预检也不生成歌曲。公共 ECAPA/RMVPE 资产与评估依赖需另行准备。

| 组件 | 项目内位置 | 来源 |
|---|---|---|
| 固定源码 | `third_party/fish-diffusion` | [上游](https://github.com/fishaudio/fish-diffusion)，commit `8e8f8cd849717f2db561e6c194df1a320c910e62` |
| 训练底模 | `models/FishDiffusion/content-vec-pretrained-v1.ckpt` | [v2.0.0 下载](https://github.com/fishaudio/fish-diffusion/releases/download/v2.0.0/content-vec-pretrained-v1.ckpt) |
| ContentVec | `models/FishDiffusion/contentvec/pytorch_model.bin`、`config.json` | [权重](https://huggingface.co/lengyue233/content-vec-best/resolve/main/pytorch_model.bin)、[配置](https://huggingface.co/lengyue233/content-vec-best/resolve/main/config.json) |
| NSF-HiFiGAN | `models/FishDiffusion/nsf_hifigan/` 完整目录 | [下载包](https://github.com/fishaudio/fish-diffusion/releases/download/v2.0.0/nsf_hifigan-stable-v1.zip) |
| Parselmouth | base 包，无神经网络权重 | `praat-parselmouth` |
| ECAPA/RMVPE 评估 | `models/voice_evaluation/ecapa/embedding_model.ckpt`、`models/RMVPE/rmvpe.pt` | [公共资产](../skill/references/weights.md) |

大小、SHA256、来源与下载包成员见 [资产锁](../configs/fish_diffusion_assets.lock.json)。ContentVec URL 的 main 不是固定 revision，实际内容由锁内哈希约束，漂移会报错。保留锁内全部文件，包括声码器 ONNX/LICENSE；当前检查并非只看推理使用的两个文件。

Fish checkout 由下载器独立获取，不是 `.gitmodules` 中的第五个子模块。部署需携带 [兼容补丁](../patches/fish_diffusion/README.md)，补丁前后哈希均受校验。只作用于 Fish 源码的兼容导入、检查点加载和 F0 边界，不覆盖其他后端包。可传 `download_assets/run.sh --cache-dir cache/fish_assets` 复用下载文件；源码也需已准备或可联网获取。

普通 `preflight --scope inference` 只要求推理资产；当前 `--model-load` 始终试载公共底模，仍需该训练权重，且不能代替人物模型的真实推理检查。

## 独立阶段与数据流

```text
目标人物原始录音 → voice_data/separate → recordings/lead.wav + manifest
→ Fish training/create_config → 本次 YAML
→ training/00_prepare → train/validation/test WAV
→ training/01_features → 内容/F0/Mel 特征
→ training/02_train → 检查点
→ training/03_validate → 验证音频与指标
→ training/04_export → voice.json + model.ckpt + config.json + validation.json
→ create_inference_config → 推理 YAML
主流程 04 人声 + 人物模型 → Fish run.sh（05）→ converted.wav
04 伴奏 + converted.wav → 06_mix → final.wav / final.mp3
```

上述训练编号属于 05 内部，不会调用主流程 00 或更改主流程 00–04。`--recordings-dir` 显式使用本次分离录音；`--source-dataset` 是另一种可选输入，必须是哈希匹配且无跨集合录音泄漏的现成 train/validation/test 数据集。要亲自生成全部文件时采用前者。

训练状态在训练 run 的 `state.json`，歌曲状态在歌曲 run 的 `stage_state.json`。根据状态查当前有效 attempt，不能固定取最大编号或假定一直是 attempt_001。重跑训练上游会使下游阶段失效。继续训练用 `02_train --resume`，延长预算可加 `--max-steps N`；更换数据或模型身份使用新 run。

## 验证和交付边界

训练 03 默认验证最终检查点，也支持 `--checkpoint`。当前使用最多两个留出录音重建探针和配置的歌曲转换窗口；不是全部测试切片遍历，也不直接复现优化实验的全部筛选指标。`--evaluation-split test` 产生独立探针报告，不能用于选模/导出；导出前须有匹配 checkpoint 的 validation 报告。

推理默认 PLMS、sampler_interval=10、8 秒分块、0.2 秒重叠、0.5 秒上下文、seed=42，保持输入音高。1000 个扩散时间点、采样间隔和 Seed-VC 的 `--steps` 是不同参数。

质量检查未通过或评估失败仍保留可解码音频，导出/混音标记待复核；损坏模型、错误配置和没有可用输出等执行问题仍报错。ECAPA、F0 和技术检查不能代替自然度、发音及音色试听。

可选 `fish_diffusion/validate/run.sh` 负责 05→06 和跨运行整曲报告；对比要求 04 人声与伴奏哈希一致。它会调用 05/06 并复用有效缓存，不是纯读取报告的工具。首次可用 `--source-run` 创建目标；目标已存在时省略该参数，不需要重算就不加 `--force`。跳过它不影响已有模型、人声和最终 MP3。

手动验证记录见 [VALIDATION](../VALIDATION.md)；具体产物和已完成版本见 [手动实操](24_fish_diffusion_manual.md)。
