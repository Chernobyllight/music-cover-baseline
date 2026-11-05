# 运行环境与迁移

供新机器部署、已有机器缺包、迁移工作目录时阅读。主入口见 [SKILL.md](../SKILL.md)。以下命令只用于实际部署时；阅读 Skill 不代表已经执行。

## 1. 获取与 Skill 匹配的代码

从 [项目仓库](https://github.com/Chernobyllight/music-cover-baseline) 获取包含本 Skill 和当前脚本的发布分支/commit。先确认发布版本，不假定 `main` 或历史 `v1.0` 已包含服务器后来新增的工具。代码包也必须附带四个上游子模块的准确版本。Fish 不是第五个 gitlink 子模块，由独立下载脚本按 `configs/fish_diffusion_assets.lock.json` 获取固定 checkout 并应用校验过的兼容补丁；需要一并交付 `patches/fish_diffusion/`。

```bash
git clone --branch v1.2 https://github.com/Chernobyllight/music-cover-baseline.git
cd music-cover-baseline
# 本 Skill 对应 v1.2；若使用其他 commit，应同步核对接口。
git submodule update --init third_party/YuE third_party/seed-vc third_party/so-vits-svc third_party/rvc
git rev-parse HEAD
git submodule status
```

不要使用 `git submodule update --remote`；不要递归拉取 RVC 的实时 VST 等无关子模块。对已有脏工作树先看 `git status`，不能用强制 checkout/reset 覆盖用户修改。

确认至少有以下文件；缺少时是代码版本不匹配，不是少装了一个 pip 包：

```bash
test -f skill/SKILL.md
test -f scripts/05_voice_conversion/so_vits_svc/create_inference_config/run.sh
test -f scripts/05_voice_conversion/rvc/training/create_config/run.sh
test -f src/music_cover/voice_mixing.py
test -f scripts/05_voice_conversion/fish_diffusion/training/create_config/run.sh
test -f configs/fish_diffusion_assets.lock.json
```

## 2. 当前代码实际需要的运行环境

| 用途 | 当前固定位置 | 说明 |
|---|---|---|
| 00–04 的 Shell 控制入口、总入口 | `envs/control/bin/python` | 入口导入项目并调度子进程，不是所有模型都由它加载 |
| 01 转谱后端 | `envs/sheetsage2/bin/python` | 旧版 Transformers 与谱面模型配套 |
| 02 分离/F0、03 YuE2、04 分离、05 五条路线、06 独立入口 | `/opt/conda/bin/python` | 使用 base；所有人物模型训练入口同样使用 base |

项目根目录可以变化；**base 安装前缀在当前代码中仍是 `/opt/conda`**。仅设置 `PATH` 或 `PYTHON` 不能改变硬编码的 base 路径。若目标机器没有该前缀，应使用满足此布局的 Linux 部署环境；需要支持其它前缀属于另一个代码适配任务，不能假称已经支持。

不要为 02–06 创建 `envs/yue2`、`envs/seed-vc`、`envs/separator`；01/control 是当前源码已有的例外。只跑到 06 时不用安装 07–11 的 separator 评审环境。

已验证硬件为 Linux / NVIDIA H200；当前 YuE2 后端写有 `memory_budget_gib=80`，不是“最低显存为 80 GiB”的实测结论，也不是已经验证消费级 24 GB GPU。新硬件需实际验证显存与 CUDA 支持，不能自动缩短歌曲掩盖 OOM。

```bash
nvidia-smi
source /opt/conda/etc/profile.d/conda.sh
conda activate base
export PATH="/opt/conda/bin:$PATH"
/opt/conda/bin/python -c 'import sys, torch; print(sys.executable); print(sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
ffmpeg -version
ffprobe -version
ffmpeg -hide_banner -encoders | grep libmp3lame
```

要求 FFmpeg/FFprobe 可执行且 FFmpeg 支持 `libmp3lame`。缺失时由部署系统的包管理器安装 FFmpeg、libsndfile；编译 CUDA 扩展时还需要匹配工具链，不能仅凭 `nvidia-smi` 显示的 CUDA 数字认定已安装 nvcc。

## 3. 基础依赖恢复

**不要直接 `pip install -r configs/*.installed.txt`。** 这些是历史快照，含继承包、原服务器 wheel 路径或无关包；`requirements-*-base.txt` 也只是已有 base 的补充清单。仓库没有已经在空白机器验证过的完整环境锁或一键安装器。

当前核对的公共主流程核心版本如下。不同机器历史快照可能不同，已有可用环境应先验证，不要为“统一版本”直接覆盖。

| 组件 | base 已验证版本 |
|---|---|
| Python | 3.11.13 |
| PyTorch / Torchaudio | 2.8.0+cu128 / 2.8.0+cu128 |
| NumPy / SciPy / Librosa / SoundFile | 2.2.6 / 1.15.3 / 0.11.0 / 0.13.1 |
| Transformers / Tokenizers / HF Hub | 4.57.3 / 0.22.2 / 0.36.0 |
| Safetensors / Accelerate / Tiktoken | 0.7.0 / 1.12.0 / 0.12.0 |
| audio-separator / ONNX Runtime | 0.30.2 / 1.23.2 |
| SpeechBrain / FAISS | 1.0.3 / faiss-cpu 1.11.0 |

若需要初装依赖，在**专用于本项目的空白 base** 中可从下列核心组合开始；这是安装起点，不是已经完成空白环境验收的声明。已有工作环境应只补缺项，先查看 pip 解析计划，保留可工作的 Torch/CUDA 组合。

```bash
/opt/conda/bin/python -m pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
/opt/conda/bin/python -m pip install \
  numpy==2.2.6 scipy==1.15.3 librosa==0.11.0 soundfile==0.13.1 \
  transformers==4.57.3 tokenizers==0.22.2 huggingface-hub==0.36.0 \
  safetensors==0.7.0 accelerate==1.12.0 tiktoken==0.12.0 \
  PyYAML==6.0.2 einops==0.8.1 numba==0.61.2 omegaconf==2.3.0 \
  faiss-cpu==1.11.0 mido==1.3.3 pretty-midi==0.2.11
/opt/conda/bin/python -m pip install --no-deps \
  -r configs/requirements-pitch-base.txt \
  -r configs/requirements-seed-vc-base.txt \
  -r configs/requirements-voice-adaptation.txt
/opt/conda/bin/python -m pip install --no-deps --no-build-isolation -e third_party/YuE
```

`--no-deps` 保留底层栈，也意味着仍需处理缺少的传递依赖。先运行选定节点预检，根据实际 ImportError 和包元数据补齐缺项；复查解析计划，避免意外重装 Torch/NumPy/Transformers。不要直接安装上游 WebUI 全部 requirements。SVC/RVC 当前无界面适配使用仓库自己的 ContentVec/HuBERT 加载器，不要求为了上游 WebUI 另建环境。

audio-separator 的 NumPy < 2 声明与本项目已验证 BS-RoFormer 路径不同；YuE 上游元数据固定的 Torch 版本也与这里的兼容路径不同。`pip check` 的这些已知声明冲突应单独记录，不能忽略其它缺依赖，也不能把全局依赖检查称为通过。所选模型的实际加载和音频推理是进一步验证环节。

### 恢复已有的 control 与 01 环境布局

仅在对应环境不存在时创建；不要覆盖已有虚拟环境。它们继承已准备的 base，01 的版本隔离发生在自己的目录中。

```bash
test -x envs/control/bin/python || /opt/conda/bin/python -m venv --system-site-packages envs/control
test -x envs/sheetsage2/bin/python || /opt/conda/bin/python -m venv --system-site-packages envs/sheetsage2
envs/sheetsage2/bin/python -m pip install --no-deps \
  numpy==1.24.3 scipy==1.13.1 transformers==4.45.2 tokenizers==0.20.3 \
  huggingface-hub==0.36.0 safetensors==0.5.3 \
  mido==1.3.3 pretty-midi==0.2.10 mir-eval==0.8.2
```

下载 SheetSage2 后核对其 `requirements.txt` 与实际加载；绘图/浏览器渲染依赖不是 01 旋律提取的默认部署目标。环境目录不要从另一台机器原样复制，虚拟环境可能引用原基础 Python。

### Fish 路线的附加依赖

只在使用 Fish 时检查 [requirements-fish-diffusion.txt](../../configs/requirements-fish-diffusion.txt)。保留当前可工作的 base 包版本，再补缺项，不照装上游整套旧依赖：

```bash
mkdir -p cache/fish_setup
/opt/conda/bin/python -m pip list --format=freeze > cache/fish_setup/base-before.txt
/opt/conda/bin/python -m pip install -c cache/fish_setup/base-before.txt \
  -r configs/requirements-fish-diffusion.txt
```

依赖冲突时核对具体约束，不解除约束去升级其他路线的底层包。`configs/fish-diffusion-base.installed.txt` 是原机器快照，不是空白机器一键安装清单。内容编码器本地加载；预检需要公共 ECAPA/RMVPE 评估依赖。安装包和准备权重是两个独立步骤。

## 4. 模型下载后分层验证

先按 [weights.md](weights.md) 准备模型，再从仓库根目录执行：

```bash
bash scripts/01_source_score/run.sh --preflight
bash scripts/02_pitch_adaptation/run.sh --transpose auto --preflight
bash scripts/03_style_generation/run.sh --preflight
bash scripts/04_vocal_separation/run.sh --preflight
bash scripts/05_voice_conversion/zero_shot/run.sh --preflight
bash scripts/06_mix/run.sh --preflight
```

02 必须带 `--transpose auto` 才能覆盖分离与 F0 依赖；原调 0 的预检可能跳过它们。微调/SVC/RVC/Fish 还需按主文件中带 profile 的实际 05 命令预检。检查成功后再执行有真实输入的 00–06；不要把预检的退出码当成音质验收。

部署成功后记录本机包版本、GPU、代码/submodule commit 和模型校验值。`scripts/setup/capture_environment.py` 会枚举范围外的历史环境；只部署 00–06 的机器不要直接照搬运行它而假定所有环境都存在。保存选定解释器的 `pip freeze` 到新 `cache/` 记录即可，不覆盖运行锁来伪造一致性。

## 故障定位

| 现象 | 检查/处理 |
|---|---|
| `ffmpeg is not available` | PATH 中提供 FFmpeg 和 FFprobe，确认 `libmp3lame` |
| `No module named music_cover` | 用本仓库 `bash scripts/.../run.sh`，它会设置 PYTHONPATH；不要从任意目录直接裸跑 `python -m music_cover.pipeline` |
| `/mnt/afs_f/...` 文件不存在 | 先确认项目根目录解析和文件是否存在；该挂载路径可能只是 `/data/...` 的真实路径。配置缺失时先创建配置，不因看到挂载名就重跑全部上游。迁移时使用用户自己的相对路径；旧 profile 内嵌路径另行核对 |
| Fish 训练 `00_prepare` 报训练 YAML 不存在 | 先执行 `fish_diffusion/training/create_config/run.sh`，看到 `Created ...yaml`；`--plan` 不创建文件。已有分离结果无需重做 |
| Fish 配置或导出目标已存在 | 使用新 version/输出位置；不要覆盖已成功版本。`dataset-dir` 与 `export-dir` 也由后续阶段创建，不提前建空目录 |
| 01 读模型时 Transformers 报错 | 核查 `envs/sheetsage2`，不要把该节点静默改成 base，也不要降级整个 base |
| 03 导入 `yue2` 失败 | 检查固定 YuE 子模块是否以 `--no-deps -e` 安装到 base |
| 03 CUDA/注意力扩展或 OOM | 看当前 `command.json`、GPU、Torch/CUDA 与实际日志；对照固定 YuE 源码，不盲装最新 vLLM/FlashAttention |
| profile 或模型哈希不匹配 | 恢复完整原导出包，或用本次训练重新导出；不要改 SHA256 绕过校验 |
| 上游缺失/失效 | 按顺序补跑或重跑受影响的最早节点，再继续下游；`--force` 不会自动补齐上游 |
| 05 有音频但 `recommended_id` 为 null | 查看质量报告，这是无合格推荐，不等于未生成；06 可混音默认输出并标记待复核 |

新任务配置支持项目相对路径，不代表每一份历史训练状态都可任意搬家。尤其 SVC 训练 filelist/模型配置存在运行时绝对路径，迁移续训需明确重建或验证，不能直接搜索替换已签名产物。使用已导出的模型时优先保留完整导出目录和文件哈希。
