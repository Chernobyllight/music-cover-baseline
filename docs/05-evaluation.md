# 当前评估口径

技术成功表示产物完整、数值与格式符合约定、可衔接下游；音乐效果仍需人工试听。当前主流程没有自动歌词识别、歌词质量门禁或静音修复。

## 音色转换

微调 05 对每个候选计算时长、静音/削波、对齐和 F0 检查；技术合格者按与参考片段的 ECAPA 余弦相似度排序。Seed-VC 和 RVC 默认三个候选，SVC 一个。细则见 [候选手册](15_voice_candidates.md) 与 `src/music_cover/voice_candidates.py`。

ECAPA 衡量说话人嵌入的接近程度，不是身份概率、自然度或歌曲质量百分比。音区、语言、说话/歌唱域差异、分离残留均会影响分数。候选筛选所用 prompt 的分数与独立 test 参考分数必须分别标注，不得混表或用 test 反选候选。

RVC/SVC 的参考片段不参与其推理条件，但仍需核对是否与训练数据重叠，不能仅凭“推理不用参考”就称独立测试。

## 主流程 10 和交付检查

| 检查 | 范围 |
|---|---|
| `evaluation.json` | 音频、分轨、MIDI 等技术检查；可含候选报告 |
| `pitch_adherence.json` | 启用选调且开启 pitch_check 时，核对生成移调及换声前后 F0 |
| `voice_conversion_evaluation.json` | voice_check=true 时额外计算音色与音高诊断 |
| `scripts/verify_review.py` | 只读核对交付及全部阶段文件、音轨重建、MIDI、压缩包、链接 |

示例使用已保留的完整 RVC 运行：

```bash
cd /data/liuhongda/music-cover-baseline
PYTHONPATH=src envs/separator/bin/python scripts/verify_review.py \
  runs/lhd_chuan_yue_shu_guang_orchestral_soul_huang_qi_shan_rvc_20261029 \
  --output cache/chuan_yue_delivery_check.json
```

人工对比应听原曲、03 换风格结果、04 重混、最终成品以及换声前后干声，覆盖主歌、副歌、高音、气声、齿音、停顿和结尾。分轨小重建误差不代表串音少；MIDI 有文件不代表转谱准确。

历史评估设计另存 [存档](archive/05-evaluation_before_cleanup_20261029.md)；实际已经完成的实验记录见 [VALIDATION](../VALIDATION.md)，日期与数据保留状态分别说明。
