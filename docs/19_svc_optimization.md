# 杨坤 SVC 优化实验与推荐配置（2026-10-28）

本轮只训练杨坤 SVC，Seed-VC 使用已有 `catalog_v001_20261027` 的 1000 步权重，未重新训练。所有方法使用完全相同的已生成歌曲人声、伴奏和混音算法。旧模型、旧数据和 Seed-VC 两条接口均保留；优化版需要显式指定配置。

## 结论及适用范围

推荐 **ContentVec vec768l12 + RMVPE + 标准 NSF-HiFiGAN**，使用清理后的 448.18 秒杨坤训练音频，先以初始学习率 1e-4 训练 4000 步，再从该检查点同时加载生成器和判别器，以初始学习率 3e-5 训练 2000 步。推理采用 noise_scale=0.1、8 秒分块、0.2 秒重叠、0.5 秒上下文；保留输入 F0，不额外移调，不启用自动 F0 预测、检索、增强器或扩散。

四个整曲 case 的平均 ECAPA：旧 SVC **0.5349**，优化 SVC **0.5814**，固定 Seed-VC **0.5629**。独立于选型的两个测试 case 平均分别为 **0.5191 / 0.5727 / 0.5533**。

这是本轮客观指标选择的推荐方案。**没有收集人工盲听分数，不能据此断言“电音感消失”或听感全面超过 Seed-VC。** 单一目标歌手、有限曲目、单次推理种子，且两个系统训练数据量和训练步数不同；结果不能证明架构普遍优劣。

## 直接使用：接入现有 05 模块

工程目录 `/data/liuhongda/music-cover-baseline`（服务器实际解析到 `/mnt/afs_f/liuhongda/music-cover-baseline`）。推荐配置和模型：

- `configs/voice_inference/so_vits_svc/yang_kun_optimized.yaml`
- `models/voices/yang_kun/so_vits_svc/optimized_v002_20261028/voice.json`
- 同目录 `model.pth`、`config.json`、`validation.json`，有 SHA256 校验。

使用一个已经成功完成 00–04 的 run，创建独立比较目录，避免覆盖原结果：

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"

# 输出目录必须尚不存在；该命令引用并校验原来的 00–04 中间产物。
/opt/conda/bin/python scripts/fork_voice_run.py \
  --source-run runs/lhd_yang_kun_svc_optimized_validation_20261028 \
  --run runs/my_yang_kun_svc_optimized

bash scripts/05_voice_conversion/so_vits_svc/run.sh \
  --run runs/my_yang_kun_svc_optimized \
  --config configs/voice_inference/so_vits_svc/yang_kun_optimized.yaml \
  --gpu 0

bash scripts/run_baseline.sh \
  --run runs/my_yang_kun_svc_optimized --from 06_mix --gpu 0
```

输入是原 run `04_vocal_separation` 的 vocals；输出 `stages/05_voice_conversion/attempt_*/artifacts/converted.wav` 供 06 混音使用，最终 MP3/WAV 在新 run 的 `deliverables/`。具体中间产物位置以 `stage_state.json` 为准。若不需要后续分轨/MIDI，可只运行 `bash scripts/06_mix/run.sh --run runs/my_yang_kun_svc_optimized --gpu 0`。

**只指定 voice.json 不会自动修改推理参数**，所以推荐同时指定上述 YAML。原版 SVC 默认 noise_scale=0.4、20 秒分块不变，Seed-VC zero_shot/finetuned 默认行为也不变。

## 为什么原来的 SVC 容易出现电音感

1. **原模型只训练 1000 步，约 328 秒训练数据，目标歌手适配不足是有证据支持的原因。** 在相同原数据、noise_scale=0.1、20秒分块下，训练从1000步增加至12000步，初筛ECAPA从0.3921提升到0.4279；再使用8秒分块达到0.4401。增加步数并非单调改善，因此不能盲选最后检查点。
2. **训练目标含人声分离残留，可能把分离伪影学进模型。** 同样的片段、编码器、4000 步和推理设置，始终使用二次分离 lead 的 ECAPA 为 0.4138；在无需去和声时保留第一次分离 vocals 的版本为 0.4573，自重建频谱误差也下降。这支持减少不必要二次分离的策略，但“残留造成听感电音”的因果仍需听辨确认。
3. **杨坤的沙哑、气声和不规则谐波是较难重建的信号。** NSF 声码器依赖 F0 和谐波激励，噪声控制、清浊音边界、数据覆盖都会影响音色细节。这是机理解释，不是已隔离证明的唯一原因。降低 noise_scale 本身没有明显提高初筛 ECAPA，不能把它当成万能去电音开关。
4. **不能通过破坏音高换取更高 ECAPA。** DIO/PM 初筛相似度看似更高，但F0误差明显增大：±1 半音保持率分别约 67.9%/73.2%，因此排除。最终坚持 RMVPE；FCPE、CREPE、Harvest 在本数据上也未形成更好的综合结果。
5. **兼容移植没有发现数值不一致。** ContentVec 原权重方程的独立函数实现与当前 Transformers 实现，在 1 秒噪声和 2 秒真实人声 CPU FP32 测试中，768l12、256l9 最大绝对误差均为 0。它缩小了接口移植错误的可能性，但不代表整个模型没有任何问题。

## 数据和训练

原始素材、歌手歌曲来源及质量记录仍在 `data/raw/references/yang_kun/`；本轮不覆盖它们。

| 数据版本 | 训练片段 | 总时长 | 用途 |
|---|---:|---:|---|
| 原始 dataset/train | 41 | 328 秒 | 旧 SVC、原数据继续训练对照 |
| expanded_lead | 45 | 448.18 秒 | 统一使用二次分离 lead 的消融 |
| expanded_clean | 45 | 448.18 秒 | 推荐训练数据，18 个歌曲组 |
| expanded_full | 53 | 527.50 秒 | 增加完整《空城》合格片段；本轮未胜出 |

推荐训练集完整路径：

```text
/data/liuhongda/music-cover-baseline/data/voice_datasets/yang_kun/so_vits_svc/optimization_20261028/expanded_clean_vec768l12/train/yang_kun/
```

原始整理清单是上一级 `expanded_clean/manifest.json`；编码器缓存目录附带 `source_manifest.json`、train/validation/test 文件列表，`.soft.pt`、`.f0.npy`、`.spec.pt` 等。

处理规则：4–12 秒左右的自然短句，低能量处切分；活动比例≥70%、有声比例≥55%；单歌曲组最多55秒；55Hz二阶高通，RMS目标0.1、峰值上限0.9，增益最多约6dB，约5ms淡入淡出。二次分离移除分量相对lead低于−20dB时使用第一次分离vocals，否则保留lead。这里的−20dB是工程筛选阈值，不是保证无伴奏/无和声的标准。

验证集6段48秒、测试集5段40秒，保留原歌曲分组；训练/验证/测试歌曲组互斥，推理参考提示片段排除在训练集之外。《空城》完整音频仅扩充原训练组，不接触验证/测试组。没有为了扩充而把质量不合格的整首全部加入。新增完整音频来自工程已有 `data/raw/songs/kong_cheng_yang_kun.mp3`；本轮无需重新爬取新歌。

训练全参更新生成器与判别器，ContentVec/RMVPE冻结。batch=6（末批可小于6），FP32、允许TF32，AdamW，betas=(0.8,0.99)，eps=1e-9，随有效epoch衰减；裁剪长度10240采样点约0.232秒，mel损失权重45、KL权重1，还包括对抗、特征匹配与F0预测损失。实际实验随机种子42（实验训练入口设置；上游配置遗留的train.seed字段不负责设定此次运行种子）。第二阶段加载生成器/判别器参数并重建低学习率优化器，选中其2000步检查点，总更新6000步；不是把原1000步模型继续到6000步。

## 对照设计与整曲结果

先用2个转换片段和2个杨坤自重建片段初筛392份输出，再将候选扩大到4个转换片段与6个自重建片段。选型约束：每段F0 ±1半音保持率≥90%，自重建平均log谱L1不超过旧模型的1.05倍，无空白/非有限/采样峰值越界。在满足约束的候选中按平均ECAPA选型，冻结 `selection.json` 后才计算最终测试指标。

| 曲目 / 编曲 | 角色 | 旧 SVC ECAPA | 优化 SVC ECAPA | 固定 Seed-VC ECAPA |
|---|---|---:|---:|---:|
| 不败的英雄 / 凯尔特史诗 | 选型曲目的整曲 | 0.5424 | **0.5999** | 0.5994 |
| 坏女孩 / Funk-Soul | 选型曲目的整曲 | 0.5589 | **0.5806** | 0.5457 |
| 不败的英雄 / Blues-Rock | 留出编曲测试 | 0.5444 | **0.6281** | 0.6043 |
| 秋意浓 / Bossa Nova–Samba Jazz | 留出歌曲测试 | 0.4937 | **0.5173** | 0.5022 |
| 四组等权平均 | 描述统计 | 0.5349 | **0.5814** | 0.5629 |

四组F0 ±1半音保持率平均：98.25% / **99.03%** / 98.83%。凯尔特case优化SVC与Seed的ECAPA仅差0.0005，不应解读为显著优势。留出的Blues-Rock与选型凯尔特版是同一首歌，不等于两首完全陌生歌曲；秋意浓保留了原先偏高的女声生成音区，作为跨音区测试，没有为SVC单独降调。

ECAPA在**混音前转换人声**上计算，使用5段独立测试参考人声取平均，而非推理时同一20秒提示；所以不能直接和旧页面采用同一提示的0.7/0.8分值比较。F0指标是RMVPE估计、时延校正后对齐有效有声帧的误差比例，不是人工唱准评分；原始报告同时保留coverage。秋意浓Seed版本的对齐检查提示起始窗口与后续窗口时延不一致（约110ms与−10ms），需要试听核查；没有为提高指标而强行修正。自重建频谱误差只适用于输入已经是杨坤的片段，不应用来奖励普通转换输出保留原歌手音色。

扩大验证中，旧SVC→推荐SVC：ECAPA **0.4118→0.4809**（固定Seed-VC为0.4676）；自重建多尺度log谱L1 **1.0582→0.9874**，约下降6.7%。普通clean4k的自重建误差更低0.9754，是可保留的备选；推荐模型按事先定义的约束与ECAPA规则选中，并非每一个指标都最优。

所有版本使用相同伴奏和06混音算法；人声RMS匹配增益与最终限幅增益由各版本波形计算，数值不必相同。实验没有重新生成上游曲风。

## 测了哪些方法，哪些没有采用

- 编码器/匹配底模：vec768l12、vec256l9、HuBERT-Soft、Whisper-medium PPG。256l9有效但未胜出；HuBERT/Whisper在本轮综合指标不优于推荐方案。不同编码器也对应不同预训练底模，不能把差别归因于编码层数一个因素。
- F0：RMVPE、DIO、Harvest、Parselmouth、CREPE、FCPE全部实际运行；最终选RMVPE。
- 推理：噪声0/0.1/0.2/0.4/0.7、20/8秒分块、RMS包络混合；检索比例0.25/0.5/0.75、64中心聚类；NSF增强器自适应key0/6/12。
- 训练：原数据至12000步、清理数据至8000步、更多《空城》片段、音量嵌入、低学习率继续优化、加长声码器训练片段至32768采样点。加长片段并未超过当前选中方案；扩充数据量本身也没有保证更好。
- Diffusion：完整扩散6000步；浅扩散至20000步；k_step50/100/200、有/无二次编码。扩大验证最佳浅扩散ECAPA0.4588，低于推荐0.4809；部分浅扩散配置的自重建频谱误差略好，可以作为音质侧备选，未据此替换默认路线。
- Snake：将已有标准NSF预训练参数迁移到Snake激活，缺乏匹配Snake预训练声码器；2000步时生成器损失约190–234、F0检查失败，主动早停。**这是本轮迁移配方失败，不是Snake架构普遍无效。**
- 未实际训练：Chinese HuBERT、DPHuBERT、WavLM、Whisper-large等，以及每一种模型/参数的笛卡尔积。在查阅的预训练目录中没有与此次方案同样现成匹配的全部底模；ONNX主要是执行格式，不作为新的音质模型比较。此次是有证据的候选收敛，不声称穷举上游所有组合。

## 2026-10-29 结果保留范围

已按确认的方案清理 runs：保留 ContentVec 768l12 + RMVPE + 标准 NSF-HiFiGAN、清理后的 45 段 / 448.18 秒训练集、4000 + 2000 步检查点，以及 noise_scale=0.1、8 秒分块、0.2 秒重叠的四组最终结果。原始数据、训练特征、models 中的已导出权重和各模型接口不变。

正式验证 run 的 00–04 已复制为自包含文件，其余三组共同输入保存在实验目录的 cases/<case>/upstream/；不再依赖已删除的旧 run。推荐训练入口的架构模板固定在 configs/voice_training/so_vits_svc/model_template.json，不再读取旧训练 run。

summary.json、selection.json、verification.json 是历史实验记录，其中未选中方案的音频/检查点路径已清理；报告中的数值仍保留供审计。当前文件范围见实验目录 retention.json。若重新开展旧消融，应创建新实验并重新生成所需输出，不可把历史汇总误当成所有音频仍存在。

## 文件和隔离

```text
scripts/05_voice_conversion/so_vits_svc/experiments/  # 独立实验工具
configs/requirements-svc-experiments.txt            # 只供隔离目录安装的可选包
configs/so_vits_svc_experiments_assets.lock.json    # 9个可选资产：来源/大小/SHA256
models/SoVitsSVC/optimization_20261028/              # 实验底模、扩散、增强器
cache/svc_optimization/python_deps/                 # pip --target；不升级base核心依赖
runs/svc_optimization_yang_kun_20261028/             # 训练/筛选/固定输入/最终比较
  training/low_lr_refine/checkpoints/step_00002000.pth
  selection.json                                  # 先冻结再测试
  summary.json                                    # 所有汇总与校验
  delivery/<case>/low_lr_refine_2000/final.mp3     # 保留4份推荐整曲
  delivery/<case>/low_lr_refine_2000/vocal.mp3     # 对应干人声
runs/lhd_yang_kun_svc_optimized_validation_20261028/ # 正式模块衔接验证
```

实际改动只增加SVC独立实验脚本/资产配置/导出profile，并让SVC标准适配器显式支持匹配的vec256l9（原vec768默认不变）。未经验证的架构不会静默降级到默认模型。旧SVC和Seed-VC原权重校验仍匹配；ContentVec两个变体做了独立数值对照。实验依赖通过当前base的 `pip --target ... --no-deps` 放到独立目录，未创建新的环境；指标/分离复用工程既有环境。

复现实验与依赖安装命令见 [实验脚本 README](../scripts/05_voice_conversion/so_vits_svc/experiments/README.md)。模型文件较大，按工程惯例不进入Git；本地/服务器上生成不代表已经上传仓库。

## 上游依据

- [so-vits-svc 4.1 中文说明](https://github.com/svc-develop-team/so-vits-svc/blob/4.1-Stable/README_zh_CN.md)，工程固定提交 `730930d337d171479eadf305f96cbed4bb393e77`。
- [匹配预训练模型目录](https://huggingface.co/Sucial/so-vits-svc4.1-pretrain_model/tree/main)，使用的下载URL及SHA256锁在资产清单。
- [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC)，实际调用当前SVC固定版本中的Unit2Mel/Vocoder实现。

## 最终工程验证

97项测试通过。正式run `runs/lhd_yang_kun_svc_optimized_validation_20261028` 的12个阶段状态成功，其中00–04复用哈希校验过的既有产物，05–11实际执行；交付验证检查182个文件，12个音频分轨、10个叶子分轨，MIDI无空轨。分轨求和最大重建误差约1.30e-7。Seed-VC zero_shot与既有finetuned分别实际生成有效音频，原权重SHA256仍匹配。

导出模型的所有tensor与选中的实验checkpoint完全相等。跨两个默认GPU进程的整曲波形不是逐采样相同（RMS差约0.000121）；在同一进程显式启用确定性CuDNN后，实验与正式适配器最大绝对差为0。不要把普通GPU推理的固定seed误解为跨进程逐位可复现。详见实验目录 `verification.json` 和 `logs/regression_final.log`（临时 parity_probe 文件已清理）。
