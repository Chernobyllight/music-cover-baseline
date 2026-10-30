# 杨坤 SVC 独立实验工具

这些是2026-10-28实验的复现入口，不会被普通baseline自动执行。结论、数据和最终使用命令见 [`docs/19_svc_optimization.md`](../../../../docs/19_svc_optimization.md)。执行根目录始终为 `/data/liuhongda/music-cover-baseline`。

## 环境/权重检查

不新建环境，不升级base的torch/transformers等核心包：

```bash
cd /data/liuhongda/music-cover-baseline
export PATH="/opt/conda/bin:$PATH"
/opt/conda/bin/python -m pip install \
  --target cache/svc_optimization/python_deps --no-deps \
  -r configs/requirements-svc-experiments.txt

PYTHONPATH=src /opt/conda/bin/python \
  scripts/05_voice_conversion/so_vits_svc/experiments/assets.py --download
```

已下载权重只校验：去掉 `--download`。输入是 `configs/so_vits_svc_experiments_assets.lock.json`，输出为 `models/SoVitsSVC/optimization_20261028/verification.json`。原始SVC基础权重先按 `docs/18_so_vits_svc.md` 安装。网络不可达时可以在可连接机器下载锁定URL，核对SHA256后复制到相同资产目录，再执行校验；不能用同名但不同权重替代。

## 数据准备

```bash
PYTHONPATH=src /opt/conda/bin/python \
  scripts/05_voice_conversion/so_vits_svc/experiments/prepare_data.py
```

读取 `data/raw/references/yang_kun/manifest.json`、`sources/*/stems/`、`quality/*/` 和原dataset分组，生成 `data/voice_datasets/yang_kun/so_vits_svc/optimization_20261028/expanded_clean/`、`expanded_lead/` 及manifest。脚本还从四个历史run冻结04人声、两个18秒窗口和cases.json；这些源run必须存在；它们已于2026-10-29清理，因此不要直接重跑这个历史准备脚本。已冻结的cases和训练集仍保留，推荐训练、推理和混音命令可直接使用。这是本轮固定实验，不是适用于任意项目的通用下载器。已有数据可直接使用，正常推理不需要重复准备。

## 推荐训练配方复现

下面使用新名字，避免覆盖本轮检查点。第一段先生成features，再训练生成器/判别器；第二段必须等第一段成功：

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH=src:cache/svc_optimization/python_deps

/opt/conda/bin/python -u scripts/05_voice_conversion/so_vits_svc/experiments/train.py \
  --name demo_clean_vec768 --encoder vec768l12 --data expanded_clean \
  --steps 4000 --learning-rate 0.0001

/opt/conda/bin/python -u scripts/05_voice_conversion/so_vits_svc/experiments/train.py \
  --name demo_low_lr --encoder vec768l12 --data expanded_clean \
  --steps 2000 --learning-rate 0.00003 \
  --warmstart runs/svc_optimization_yang_kun_20261028/training/demo_clean_vec768/checkpoints/step_00004000.pth
```

中间产物：`runs/svc_optimization_yang_kun_20261028/training/<name>/model_config.json`、`loading.json`、`checkpoints/step_*.pth`、`report/losses.jsonl`、`report/training.json`。第二阶段加载第一阶段生成器和判别器，重新创建优化器；`--steps 2000`是第二阶段的更新次数。`--resume`仅允许代码、数据、配置身份匹配的同一训练，不用于改变学习率；旧实验在训练器升级前保存的checkpoint若身份校验失败，不应绕过校验，使用显式warmstart创建新实验。

编码器实验参数：`--encoder vec256l9` / `hubertsoft` / `whisper-ppg`；原始数据保持不动。音量嵌入/Snake/扩散是独立消融入口，不是推荐设置。

## 推理与评估

保存的request包含每个模型的checkpoint/config/variants和固定输入cases。以下请求已经存在，普通SVC sweep会复用已完成conversion.json：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:cache/svc_optimization/python_deps \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=4 \
  /opt/conda/bin/python -u scripts/05_voice_conversion/so_vits_svc/experiments/sweep.py \
  --request runs/svc_optimization_yang_kun_20261028/optimized_full_request.json

# 只在冻结selection.json后评估最终测试集；使用已有Seed环境加载指标。
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src OMP_NUM_THREADS=4 \
  envs/seed-vc/bin/python -u scripts/05_voice_conversion/so_vits_svc/experiments/evaluate.py \
  --outputs runs/svc_optimization_yang_kun_20261028/optimized_full/outputs.json --split test

PATH="/opt/conda/bin:$PATH" PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  envs/control/bin/python scripts/05_voice_conversion/so_vits_svc/experiments/mix_delivery.py \
  --outputs runs/svc_optimization_yang_kun_20261028/optimized_full/outputs.json
```

转换输出：`optimized_full/low_lr_refine_2000/chunk8/<case>/converted.wav`。指标：`optimized_full/evaluation/results.json`。最终混音：`delivery/<case>/low_lr_refine_2000/final.mp3`；干人声：同目录`vocal.mp3`。混音使用 `cases.json` 指向的原03 cover和04 instrumental，不重新生成编曲。

不要修改已运行request来冒充同一次实验：更改模型/输入/参数请使用全新output目录。评估会复用单项缓存，拒绝在同一输出目录混用validation/test参考。汇总前核对request预期输出数量，防止在转换未结束时把部分指标当成全量。

`prepare_final_validation.py`生成扩大验证请求；`select_model.py`按保存的规则冻结选择，已有selection.json时拒绝重选；`export_selected.py`创建新的不可覆盖profile。当前导出脚本固定本轮profile版本，用于审计复现，不能直接用来覆盖线上模型。新的通用歌手训练/导出仍使用上层训练模块。

## 清理后保留的结果结构（2026-10-29）

```text
runs/svc_optimization_yang_kun_20261028/
  training/clean_vec768/checkpoints/step_00004000.pth
  training/low_lr_refine/checkpoints/step_00002000.pth
  cases/ / cases.json                       # 自包含的共同输入
  final_validation_b/low_lr_refine_2000/     # 选定模型的验证片段
  optimized_full/ / optimized_full_request.json
  delivery/<case>/low_lr_refine_2000/        # 四组final.wav/mp3及干声
  selection_protocol.json / selection.json
  summary.json / verification.json / retention.json / logs/
```

非选定方案的检查点、中间音频和整曲已删除；历史汇总数值保留。原 summarize.py、select_model.py 等完整实验汇总入口依赖已删除的对照文件，重新开展消融时须先生成完整数据。推荐 train.py 使用 configs/voice_training/so_vits_svc/model_template.json 架构模板，不依赖旧 voice_training 目录。

其它脚本：`separate_extra.py`、`extend_data.py`为《空城》扩充实验；`diffusion_lab.py train|infer`为完整/浅扩散训练推理；`resume_original.py`复制原checkpoint后继续训练；`seed_control.py`只推理固定Seed模型；`summarize.py`核验392份初筛输出、12份整曲评估与原模型校验。原始下载草稿不作为依赖安装入口，以`assets.py`和锁文件为准。
