# Fish Diffusion · validate

完整验证 Fish 05→06，并可比较相同源人声和伴奏的其他后端结果。

全部使用 `/opt/conda/bin/python`。脚本自行定位项目根目录，不依赖当前目录或 VC_* 变量。

输入：已有歌曲 --run 与 Fish 推理 YAML。仅首次需要自动 fork 且目标不存在时，额外传已完成 00–04 的 --source-run。--compare-run 可重复，必须已完成 05/06 且 04 两条音轨哈希相同。继续已有 run 时省略 --source-run。

输出：歌曲 run 的 05/06 产物及 fish_validation/comparison.json、index.html。不重跑主流程 00–04，不修改对照 run。

## 具体使用示例

```bash
bash scripts/05_voice_conversion/fish_diffusion/validate/run.sh \
  --run runs/lhd_lv_zhou_shan_yi_chun_fish_manual_v001 \
  --config configs/voice_inference/fish_diffusion/shan_yi_chun_manual_v001.yaml \
  --compare-run runs/lhd_lv_zhou_shan_yi_chun_modular_test --gpu 0
```

[完整部署、训练和验证说明](../../../../docs/22_fish_diffusion.md)。

这是可选对比：05 和 06 完成后已有 final.wav/final.mp3，跳过本入口不影响前面的训练、导出或混音。本入口会调用 05/06 并复用有效缓存，不是纯读取工具；已有目标时省略 --source-run，无需重算不加 --force。

[13 步实操与结果路径](../../../../docs/24_fish_diffusion_manual.md)。
