# P2 任务规范

## 目标

在 P1 `PaperDoc`、Provider、去重和评测协议上实现可执行的结构化查询树、引用感知迭代、证据加载、可解释分量评分与 `StopDecision`。P2 不实现 P3 五 Agent 圆桌，不修改 `repos/` 原始仓库。

## 固定流程

```text
QueryPlanner → SourceRouter → RecallRunner → FusionDeduper
→ ConstraintGate → 初排/种子选择 → CitationExpander/EvidenceLoader
→ Scorer → gap 驱动下一轮 QueryPlan → StopController
```

## 固定边界

- 最大迭代次数 2，最大引用深度 1，每个 gap 最多生成 2 条子查询。
- Provider 调用有界并发；错误进入 `source_errors`，不得转换成 `relevance=0`。
- 引用种子必须通过硬约束、至少具备摘要、相关性达到配置阈值。
- `metadata` 不能冒充正文证据；EvidenceItem 必须带 `evidence_ref` 或明确 unavailable。
- 分量固定为 relevance、constraint、evidence、quality、citation、novelty；默认权重为 0.30/0.25/0.20/0.10/0.10/0.05，仅作为可配置基线。
- 强停止：`BUDGET_EXHAUSTED`、`MAX_ITERATION`、`ALL_PROVIDER_FAILED`、`NO_NEW_PAPER_2_ROUNDS`。`MAX_CITATION_DEPTH` 只限制引用扩展并写入 `triggered_conditions`，不终止整个查询迭代。
- 软停止至少满足三项中的两项：新增高相关论文 < 2、子查询覆盖率 >= 0.8、证据覆盖率 >= 0.7。
- 所有结构化对象必须可校验和落盘回放；长正文只通过 `content_ref`/`evidence_ref` 引用。

## Agent Team 与测试隔离

- 开发 Agent 按互不重叠的模块写代码；主 Agent 负责集成与冲突处理。
- 开发 Agent 可以运行其模块定向测试，但不得出具 P2 最终验收结论。
- 集成完成后必须创建一个此前未参与开发、无开发任务上下文的全新测试 Agent。
- 最终测试 Agent 只读取本规范、代码和 artifact，执行独立测试并提交问题清单；不得修改代码。
- 若问题需要修复，由开发 Agent 或主 Agent 修复；最终复验再次使用新的测试 Agent。

## 验收

- 同一查询可回放 QueryPlan、节点、候选、引用关系、EvidenceVerdict 和 StopDecision。
- 引用扩展在 fixture 中真实调用 `relations()` 并带父论文、关系类型、来源与深度；关闭引用时有消融。
- 每个最终结果有分量分、硬约束状态及 evidence_ref/evidence_status。
- 输出 Recall/Precision/F1@10/@20、MRR、引用覆盖率、证据覆盖率、延迟和 API 调用量。
- AutoScholarQuery 小样本必须作为查询分解回归集，不能继续把完整疑问句逐词 AND。
- P1 全部测试保持通过；P2 mock 在无 LLM Key、无付费 API 时可运行。
- `git diff --check`、secret scan、compileall 和独立测试 Agent 验收通过后，才允许创建 P2 Git 提交。
