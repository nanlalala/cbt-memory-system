# 客观基准 V4 开发修复：实际结果（2026-11-05）

**真实修复批次已完成，但新增收益很小：managed 从 480/800 增至 483/800，raw 净变化为 0。** 当前 managed 对 raw 的 8.0 个百分点差异包含原来已有的 7.625 个百分点，不能全部归因于本轮修复。它也不能证明 CBT 对话收益或临床效果。

本轮在真正 hosted Colab 执行，只对提示发生变化的 **206 条回答**重新调用模型：managed 96 条、raw 110 条。其余 **2,194 条**沿用此前提示未变的冻结回答，其中 short-term 的 800 条全部复用。已形成 800 题 × 3 组的完整 2,400 行结果清单，并在清单中逐行标明新生成与旧回答复用。

## 结果和剩余缺口

| 条件 | 原 v3 官方正确数 | 当前 V4 官方正确数 | v3→V4 变好 / 变差 | 本轮新回答 / 旧回答复用 |
| --- | ---: | ---: | ---: | ---: |
| Short-term | 10/800 | 10/800（1.25%） | 0 / 0 | 0 / 800 |
| Raw-history RAG | 419/800 | 419/800（52.375%） | 2 / 2 | 110 / 690 |
| Managed-memory RAG | 480/800 | 483/800（60.375%） | 3 / 0 | 96 / 704 |

Managed 相对自身原结果增加 3/800，即 **0.375 个百分点**。当前同题 managed 对 raw 为 72 胜、720 平、8 负；原结果为 68 胜、725 平、7 负。**“对自身原结果没有退步”不等于“每题都优于 raw”。** 例如当前单跳 64k 任务 managed 为 96/100，raw 为 97/100。

| 当前任务类别 | Short-term | Raw history | Managed memory |
| --- | ---: | ---: | ---: |
| 单跳 fact_sh | 9/400 | 374/400 | 385/400（96.25%） |
| 多跳 fact_mh | 1/400 | 45/400 | 98/400（24.5%） |

**多跳仍有 302/400 未满足官方标准，是主要未解决缺口。** 不能用单跳的高分概括多跳能力。官方既定评分保持不变；预先保留的严格格式/明确答案辅助检查为 short-term 10、raw 418、managed 481，不替代官方结果，也不构成专业 CBT 评分。

本批次仅涉及 MemoryAgentBench，**没有更新 PersonaMem**；其原结果与原工件保持不变。CBT 过程的 36 个虚构开发案例及其中的写入契约错误另见 [CBT 过程报告](cbt-process-followup-results-20261105.md)。本次客观基准修复的三题增加，不能代替 CBT 契约修复后的效果复测。

## 研究解释边界

这是在已暴露开发材料上的版本修复诊断。2,194 条旧回答与 206 条新回答来自不同执行时点，**不是三组全部同期重跑**。提示未变的回答复用可避免无必要的调用，但不能完全排除时间、服务端或输出波动的影响，也不是独立的确认性试验。

800 题包含同一 fact-consolidation 家族的长度变体，保守视为 **1 个依赖家族**，不能当成 800 名独立用户或多个独立人群样本。人群级置信区间无法据此估计；报告中保留 `not_estimable_one_family`，不提供虚假的独立题目置信区间。

逐题分数与提示/回答哈希能核对版本、完整性和数值，不能证明模型依赖了哪段证据，或错误究竟发生在抽取、更新、检索、裁剪还是最终回答。没有独立来源标注的回答错误阶段继续保持 unresolved。没有真人、独立 CBT 专业评分或临床结局，不能称为治疗效果。

## 本轮成本和身份

本批次 **206 次实际调用全部成功**，记录失败 0、未解决尝试 0、usage 缺失 0。报告使用量为输入 **462,998**、输出 **449**，合计 **463,447 tokens**；模型报告名均为 `deepseek-v4-1-flash-260910`。金额未知，未以推测单价补齐。旧回答的原始费用保留在原运行中，本轮不重复计费，也不把其历史费用抹去。

与新增 CBT 批次合计为 **406 次调用、666,875 tokens**（CBT 200 次 / 203,428 tokens；本批次 206 次 / 463,447 tokens）。这是这两批新增工作的合计，**不是项目全部历史成本**。报告生成和本次元数据审核没有新增模型调用。

- 基础执行提交：`efd75116f985749a96a6070cb80c4b38d5c94dba`
- 修复 helper SHA-256：`5e1f862f615a871a09e498da0199d43b43b09a3b745c827a6c8152aeeae89bdc`
- 新协议身份：`bd14178402c2b713e5a4e9eeac250477c4ee66951ff3a06493295ae62a3552ae`
- 每题元数据、全部来源哈希及独立复算：[V4 结果目录](../evaluation/objective-memory/results/v4-repair-20261105/README.md)

## English boundaries

This completed development repair grid contains 800 cases and 2,400 arm rows: **206 newly generated replies and 2,194 reused frozen replies from an earlier execution**. It is not a contemporaneous full rerun. Managed official correctness rose from 480 to 483, with three gains and no regressions against its old replies; raw remained at 419, with two gains and two regressions. Short-term stayed at 10 and was entirely reused. The current managed–raw gap is eight percentage points, while the incremental managed change is only 0.375 percentage points.

All length variants belong conservatively to one fact-consolidation family. A population confidence interval is not estimable from this single dependent family. Results are development-exposed; reuse and timing prevent a clean separation of repair effects from execution-time or output variability. Multi-hop correctness remains 98/400 (24.5%). PersonaMem was not changed. Metadata reconciliation does not establish answer-error causal stages, independent authenticity, user benefit or CBT/clinical efficacy.

The new batch used 206 successful calls and 463,447 reported tokens; monetary cost is unknown. Question, history, reply and gold text were not read outside Colab or included in these public result exports.
