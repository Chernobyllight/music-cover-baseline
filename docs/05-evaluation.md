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

## 音频产出与质量验收分离

三条音色路线使用统一有效人声对齐：筛除低活动窗口，检测足够窗口的 RMS 包络一致性；主检查不可靠时按音节起音复核。默认相关性阈值 0.3、延迟跨度 30 ms，长音频要求至少 3 个且至少 70% 有效窗口被接受。未能确认对齐时不自动平移，保留音高统计但标为 unassessable，不能据此断言跑调。训练验证、05 候选、06 混音和 10 音色评价采用同一方法。

05 的 generation_status 与 assessment 独立：passed、failed、unassessable、error 不改变已成功生成音频的保存。仅合格者进入推荐排名；无合格者仍以有限 ECAPA 最高的已生成候选作默认试听输出，缺指标时按 seed/ID 确定。recommended_id 可为 null，selected_id 是默认输出；06 和 11 不再把两者混用。评价组件失败也记录错误并保留音频。权重/音频哈希不匹配、损坏文件或推理没有可读取结果仍是执行错误。

训练后质量未通过允许导出已验证且哈希一致的模型；profile 的 assessment_status=review_required、validation_status=quality_review_required。该行为不改变训练数据、权重或推理参数，也不重新训练。

06 对时长不一致的转换结果保留较长时间轴，较短音轨尾部补静音，不做时间拉伸；对静音结果仍输出混音并记录警告。输出格式、哈希、有限数值等文件完整性检查继续执行。最终结果质量仍需人工试听，程序完成不等同质量合格。
