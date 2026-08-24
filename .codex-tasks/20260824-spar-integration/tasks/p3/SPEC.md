# P3 任务规范

## 目标

在 P2 的 QueryPlan、PaperDoc、引用、证据和停止协议上实现五角色结构化圆桌。角色只传输短 JSON 消息和 artifact 引用，长正文必须落盘；不修改 `repos/`。

## 五角色与流程

```text
Planner → Retriever → CitationExplorer → EvidenceJudge → Arbiter
```

固定角色：`planner` 生成计划，`retriever` 召回，`citation_explorer` 扩展关系，`evidence_judge` 评分和约束判断，`arbiter` 生成停止和最终选择。实现允许串行 fixture，但消息协议必须保留角色边界。

## 消息协议

- 协议版本：`spar-agent.v1`。
- 消息类型固定为 `QUERY_PLAN`、`RESULT_BATCH`、`RELATION_BATCH`、`EVIDENCE_VERDICT`、`STOP_DECISION`、`FINAL_SELECTION`。
- 每条消息必须包含 `run_id`、`message_id`、`type`、`sender`、`receiver`、`seq`、短 `payload` 和 `payload_ref`；正文只能通过相对 artifact 路径读取。
- 禁止绝对路径、URL、路径穿越、长正文和未定义角色；每种消息的 payload 由协议校验。
- 消息写入 JSONL，artifact 通过 root containment 校验，能够独立回放。

## 运行边界

- P3 fixture 无 Key 可运行，引用支持 enabled/disabled 消融。
- 单次运行最多两轮、引用深度为 1、候选和 API 调用受 QueryPlan budget 限制。
- Provider/DeepSeek 错误写入错误 artifact；不得将错误转换为论文低相关分。
- DeepSeek 为可选理解/判断层；未配置或失败时回退 P2 确定性规则并保留 degraded 状态。

## 验收标准

- 五角色消息均通过 `validate_message`，协议 JSONL 可加载；每个最终 PaperDoc 有分量分、硬约束状态、证据状态和可读取 evidence_ref。
- 以 fixture 分别运行引用开启和关闭；开启必须有关系边和 relations 调用，关闭必须为 `citation_disabled` 且无关系调用。
- P3 artifact 可读取，消息、PaperDoc、关系、证据、停止和最终选择均可追溯到同一 `query_id`。
- 输出至少包含候选数、选择数、错误数、消息字节数/估算 token 数，并能比较结构化消息的开销。
- P2 全量测试保持通过；compileall、git diff --check、secret scan 和全新只读 Agent 验收通过后才创建 P3 Git 提交。
- 真实 Gold 缺失时不得声称 F1 或效果提升；只能报告 fixture/真实 smoke 的可执行性和限制。

## P3 修复补充（2026-08-25）

- 圆桌必须至少运行两轮：首轮 `STOP_DECISION.action=NEXT_QUERY` 时由 `QueryPlanner.next_iteration()` 根据 gap 产生带 `parent_id/iteration` 的下一轮子查询；达到预算后记录 `MAX_ITERATION`。
- `EvidenceJudge` 对未被硬约束排除的词法初排候选做批量判断，单轮最多一次批量调用；规则分与模型分差值大于 0.25 时，Arbiter 触发一次批量复核并在 verdict artifact 记录 `conflict_reviewed`。
- `EVIDENCE_VERDICT` 必须引用包含全部候选 verdict 的 artifact；`STOP_DECISION`、`FINAL_SELECTION` 的接收方固定为 `orchestrator`，不得发给自身。
- 最终选择必须是 `spar.final.v1`，包含标题、年份、venue、分区（`high/partial/reserve`）、六项分数、证据引用和关系图。
- P3 replay 必须读取 `protocol.jsonl`、manifest、final_selection 和 evidence_ref，不能只验证文件存在。
- P3 指标复用 P1 `identity.match_papers`；对照必须支持 P2/P3、citation enabled/disabled、structured/long-text communication，并同时报告 Recall/F1、延迟、Provider/LLM 调用和 Token。
