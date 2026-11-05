# CBT 过程对照：真实模型运行结果（2026-11-05）

**本轮已实际完成 36 个虚构开发案例、108 条三臂回答记录。结果尚不能证明稳定的记忆管理收益或 CBT 实际疗效。** 这是同模型辅助评分的开发诊断，不是独立专业评审，也没有真实参与者。研究材料为已暴露的 3 个参数变体 × 12 类任务，不能当成 36 名独立用户。

执行与完整工件处理位于真正 hosted Colab。冻结执行提交为 `efd75116f985749a96a6070cb80c4b38d5c94dba`，协议 SHA-256 为 `b59ee0082d150faa61536a7a20042a441048b5e6b279c6199f85c12426c24815`。本公开更新仅处理已导出的安全元数据，没有在 Colab 之外读取原始问题、回答、金标准或模型 payload，也没有调用模型。原运行、协议和失败记录保持原样。

## 对照结果与 CBT 含义

下表中“满足”是模型对研究者草案标准的判定，**不能称为经独立建立的准确率、CBT 专业分数或疗效**。未解决项保留在全部 36 题的分母中。

| 记忆条件 | 满足 met | 部分 partial | 违反 violated | 未解决 unresolved | 总计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short-term | 18 | 2 | 5 | 11 | 36 |
| Raw-history RAG | 20 | 7 | 0 | 9 | 36 |
| Managed-memory RAG | 21 | 6 | 0 | 9 | 36 |

Raw 与 managed 的 27 个双边有评分案例中，managed 有 2 个更好、1 个更差、24 个同等级；另有 9 个案例未解决。有 5 个案例在同一案例内使用相同输入，raw 与 managed 因而共享该案例的一条回答，不能当作独立支持管理机制的重复实验。

- **目标跟进没有总体提升：**T01 两臂均为 1/3 满足、2/3 部分满足。B 变体中 managed 从满足降为部分，C 变体从部分升为满足，恰好相抵。
- **唯一的任务级净增在 T07：**比较过去预测与一次实际观察时，raw 为 1/3 满足，managed 为 2/3。Short-term 在该任务反而为 3/3，不能只挑选 managed 对 raw 的一题差异来宣称整体更优。C:T07 的 managed 前缀还带有抽取不可用/无效标志；它的较高草案评分不证明完整语义流水线有效。
- **练习障碍仍未解决：**T11 的 raw 与 managed 都是 3/3 部分满足；short-term 为 3/3 违反。更长记忆提供了一些开发信号，但两种 RAG 都尚未满足完整任务标准。
- **删除与不保存尚无结果结论：**T09、T10、T12 的 27 条逻辑组别记录因隐私参照未提供给模型评分器而未进入模型判定，全部保持 unresolved。不得据此称通过、失败或证明删除隔离；需要受控的来源/生命周期审核与独立评审。

逐题类型计数如下，单元格顺序固定为 **met / partial / violated / unresolved**，每格分母为 3 个虚构变体：

| 任务 / Task | Short-term | Raw history | Managed memory |
| --- | --- | --- | --- |
| T01 目标与初始练习 / Goal | 0/2/0/1 | 1/2/0/0 | 1/2/0/0 |
| T02 部分作业复盘 / Partial homework | 3/0/0/0 | 3/0/0/0 | 3/0/0/0 |
| T03 想法、事实与历史 / Source distinctions | 0/0/2/1 | 3/0/0/0 | 3/0/0/0 |
| T04 未确认提议 / Unanswered proposal | 3/0/0/0 | 3/0/0/0 | 3/0/0/0 |
| T05 目标更正 / Goal correction | 3/0/0/0 | 3/0/0/0 | 3/0/0/0 |
| T06 练习量修订 / Quantity revision | 3/0/0/0 | 3/0/0/0 | 3/0/0/0 |
| T07 预测与观察 / Prediction versus observation | 3/0/0/0 | 1/2/0/0 | 2/1/0/0 |
| T08 歧义澄清 / Ambiguous update | 3/0/0/0 | 3/0/0/0 | 3/0/0/0 |
| T09 删除后查询 / Deleted source | 0/0/0/3 | 0/0/0/3 | 0/0/0/3 |
| T10 不保存后查询 / Non-stored source | 0/0/0/3 | 0/0/0/3 | 0/0/0/3 |
| T11 保留的练习障碍 / Retained barrier | 0/0/3/0 | 0/3/0/0 | 0/3/0/0 |
| T12 新会话不回流 / New-session isolation | 0/0/0/3 | 0/0/0/3 | 0/0/0/3 |

CBT 过程的关键区别是：准确复述目标不等于协作地设定目标；记住作业不等于帮助实践；正确回顾一次观察不等于用户学会了技能。当前未测独立专业适切性、真实用户打扰次数或重复追问负担、练习技能迁移、症状变化、日常功能和持续效果。真人数、独立 CBT 专业评分数和临床结局数均为 **0**。

## 错误归因的可核查边界

200 次 provider 请求全部成功，108 条回答的结构校验通过；这不等于所有阶段成功。公开元数据独立复算发现，managed 的 36 个前缀中有 12 个标记所有抽取可用/有效，24 个包含抽取不可用或无效。前缀共享来源事件，因此 **24 个前缀不等于 24 次独立失败**，其他有效候选仍可能存在。

真正 Colab 中的只读来源阶段诊断现已导出，并完成独立的结构元数据与冻结适配器代码核验（诊断 SHA-256：`ace19582b40c1a8a86b8b64fcb7749df908dcef59eb9b02813951e3f9bfd1f5b`）。39 个去重后的合资格来源抽取事件中，27 个通过结构校验、12 个无效，涉及 A-S06/07/08/10/12、B-S06/08/12、C-S06/08/09/12。

12 个无效事件均在冻结版本 `source_replay.py` 的 `_ingest_semantic` 第 181 行抛出同一种 `ValueError`：`user_correction` 缺少适配器要求的规范目标。对应候选把 `target_source_id` 放在 `structured_payload` 中；当时适配器读取顶层 `candidate.target_source_id`，再映射到 `structured_payload.proposed_update_target_source_id`。这些候选既没有顶层字段，也没有规范字段，因此在派生更正内容前被拒绝。导出的结构标志均显示：嵌套目标指向合资格的更早用户来源，引用只指向当前来源，支持引文与当前原文精确匹配。由此可定位一个**抽取输出与写入适配器的字段位置契约错位**；不能把它误写成网络失败、原文全部丢失或用户意图已获确认。

本次独立审核读取了导出的结构标志，并核对身份一致的公开冻结代码；没有在 Colab 之外重新打开原始文本或模型 payload。27 个结构有效输出不代表语义解释一定正确。三个位于最后前缀的重放快照与原准备工件一致，也不等于所有回答的因果归因已经解决。

来源写入阶段的上述结构错误已经定位，但它不足以断定某一道回答的评级变化究竟由抽取、更新、检索、裁剪还是回答造成。没有来源级受控证据的因果阶段保持 unresolved；答案子串命中、输入包含某个 ID 或同模型给出的理由都不足以证明因果。

## 全部尝试与成本

| 用途 / Purpose | 实际调用 | Prompt tokens | Completion tokens | Total tokens |
| --- | ---: | ---: | ---: | ---: |
| 抽取 / Extraction | 39 | 35,270 | 10,027 | 45,297 |
| 最终回答 / Response | 93 | 61,063 | 14,330 | 75,393 |
| 模型辅助判断 / Judge | 68 | 68,616 | 14,122 | 82,738 |
| 合计 / Total | 200 | 164,949 | 38,479 | 203,428 |

108 条逻辑回答来自 93 次去重后的回答调用。保留 200 次实际尝试，provider 失败 0、未解决尝试 0；抽取输出无效另计，不从成本中删除。200 次均报告模型名 `deepseek-v4-1-flash-260910`。使用量字段完整，但没有核验货币价格或账单，**金额未知**。API 延迟之和约 570.38 秒、中位数约 2.76 秒；它们不是端到端耗时、并发墙钟时间或用户任务时间。

## English evidence statement

A genuine hosted-Colab run completed 36 fully exposed fictional development cases and 108 three-arm answer rows. Same-model-assisted draft criteria were met on 18 short-term, 20 raw-history and 21 managed-memory rows; unresolved counts were 11, 9 and 9 respectively. These are not independently established accuracy, professional CBT ratings or clinical outcomes.

Of 27 raw/managed pairs with resolved draft grades, managed was higher on two, lower on one and unchanged on 24; nine pairs remained unresolved. In each of five cases, raw and managed shared an identical prompt and one canonical answer. The only net task-level increase was prediction-versus-observation task T07, from one to two met grades out of three variants; short-term met all three. Both RAG arms remained partial on every T11 retained-practice-barrier variant. These findings do not establish stable management benefit, adequate CBT process or actual learning.

All 200 provider attempts succeeded, but that does not establish semantic pipeline validity. Exported metadata show extraction issues affecting 24 managed prefixes. The now-exported structural diagnostic and identity-matched frozen code were independently checked: 27 of 39 unique source events passed structural validation; all 12 invalid events failed at the same correction-ingestion target check. Their candidates placed a valid prior-source `target_source_id` inside `structured_payload`, while the frozen adapter expected a top-level field mapped to canonical `proposed_update_target_source_id`. Exported flags showed exact current-source quotes and current-source-only citations. This establishes a field-placement contract mismatch, not semantic correctness, confirmed user intent or the cause of any answer-grade difference. Raw source/model text was not reopened outside Colab; answer-error causal attribution remains unresolved.

Costs include 39 extraction, 93 answer and 68 judge calls: 203,428 reported tokens in total. Currency cost is unknown. Zero real participants, zero independent human CBT assessments and zero clinical outcomes were observed. Independent process review, participant burden/skill measurements and an appropriately approved matched-comparator clinical study remain distinct evidence requirements.

## Public artifacts and audit scope

- [Frozen public aggregate](../evaluation/dialogue-benefit/results/cbt-followup-20261105/public_aggregate.json)
- [108 text-free score metadata rows](../evaluation/dialogue-benefit/results/cbt-followup-20261105/public_score_metadata.jsonl)
- [Execution/code/source identities](../evaluation/dialogue-benefit/results/cbt-followup-20261105/report_summary.json)
- [Independent derived-metadata audit](../evaluation/dialogue-benefit/results/cbt-followup-20261105/derived_metadata_independent_audit.json)
- [Recomputed task-grade counts](../evaluation/dialogue-benefit/results/cbt-followup-20261105/per_task_grade_counts.json)
- [Exported structural diagnostic](../evaluation/dialogue-benefit/results/cbt-followup-20261105/extraction_structural_diagnostic.json)
- [Independent structural-metadata and frozen-code audit](../evaluation/dialogue-benefit/results/cbt-followup-20261105/extraction_structural_independent_audit.json)

The independent audit recomputes exported metadata and numeric reconciliation; it does not reopen original journals outside Colab or verify provider billing. These result exports contain no question, answer, gold/reference text, raw extraction payload or credential. Original hashes remain recorded. This update makes no clinical claim and does not overwrite an earlier experiment. Any later optimization is a separate frozen run; its results are not filled in here before the corresponding safe artifacts are available.
