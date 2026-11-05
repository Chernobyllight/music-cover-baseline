# Fish Diffusion · download_assets

下载固定版本的上游源码与公共权重，应用受哈希校验的 base 兼容补丁。不会安装 Python 包。

全部使用 `/opt/conda/bin/python`。脚本自行定位项目根目录，不依赖当前目录或 VC_* 变量。

输入：configs/fish_diffusion_assets.lock.json。支持 --cache-dir models/FishDiffusion/downloads 离线缓存；相对路径按项目根目录解析。

输出：third_party/fish-diffusion 和 models/FishDiffusion；不创建训练或歌曲 run。

## 具体使用示例

```bash
bash /data/liuhongda/music-cover-baseline/scripts/05_voice_conversion/fish_diffusion/download_assets/run.sh
```

[完整部署、训练和验证说明](../../../../docs/22_fish_diffusion.md)。
