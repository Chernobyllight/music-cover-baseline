# RVC base 环境检查

第 2 步；固定 base `/opt/conda/bin/python`，相对路径按项目根目录解析。配置生成器可加 `--plan` 只检查预览，已有输出不覆盖。

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/rvc/preflight/run.sh \
  --scope all --gpu 0
```

`all` 检查 RVC 训练/推理、ECAPA 评估、两级人声分离、CUDA、FFmpeg/MP3 编码；`training` 检查 RVC 训练/推理及评估；`inference` 不要求预训练 G/D 权重。检查包含版本锁和模型哈希；评估 F0 的存在性由运行依赖检查负责。缺项时退出码非零，不安装包、不创建环境、不运行模型。终端输出 `status: passed`、实际 python/prefix、GPU 和公共资产路径。`--plan` 同样只执行检查。

脚本自身固定使用 base。要让调用它的交互终端也显示 `(base)`，在终端执行 `source /opt/conda/etc/profile.d/conda.sh`、`conda activate base`；子进程不能修改父 shell 的环境。


完整四步见[配置脚本说明](../../../../docs/rvc_config_scripts.md)。
