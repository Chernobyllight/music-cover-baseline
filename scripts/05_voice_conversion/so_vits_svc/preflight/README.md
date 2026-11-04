# SVC base 环境预检

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/so_vits_svc/preflight/run.sh \
  --scope all --gpu 0
```

输入为现有 base、SVC 公共权重、ECAPA/RMVPE 评价组件、人声分离模型和 FFmpeg；无需上一阶段产物。只读检查，不安装、不下载、不创建实验文件。终端预期输出 `status: passed`、`python: /opt/conda/bin/python`。

`training` 范围省略分离组件；`inference` 进一步省略 G/D 底模。所有范围都检查评价依赖和 MP3 编码器。完整实例见 [SVC 手动验证](../../../../docs/svc_config_scripts.md)。
