# Agent 结构化通信协议（方案版）

## 目标

Agent 间默认传递 JSON/JSONL。长查询解释、论文全文、摘要和审稿意见写入 artifact，以 ID 或 `*_ref` 引用；不在消息中复制长自然语言。

## 通用消息封套

```json
{
  "protocol": "spar-agent.v1",
  "run_id": "run-0001",
  "message_id": "msg-0001",
  "type": "RESULT_BATCH",
  "sender": "retriever",
  "receiver": "arbiter",
  "seq": 3,
  "payload_ref": "artifacts/run-0001/result-batch-003.json",
  "payload": {}
}
```

`payload_ref` 存在时，消息只保留摘要统计和 ID；`payload` 不得放论文全文。`diagnostic_code` 使用枚举，例如 `OK`、`DEGRADED`、`RATE_LIMIT`、`SCHEMA_ERROR`、`NO_EVIDENCE`。

## 消息类型

- `QUERY_PLAN`：Planner 输出结构化子查询和硬/软约束。
- `SEARCH_ACTION`：Retriever 执行某个 `subquery_id`、source、page 和预算。
- `RESULT_BATCH`：返回 PaperDoc ID、召回分数、source 和 provenance 引用。
- `RELATION_BATCH`：CitationExplorer 返回父论文、关系类型、子论文 ID 和深度。
- `EVIDENCE_VERDICT`：EvidenceJudge 返回分量分、约束结果、证据引用和置信度。
- `STOP_DECISION`：Arbiter 返回是否停止、原因代码、边际收益和预算统计。
- `FINAL_SELECTION`：输出最终 PaperDoc ID、排序分量和证据引用。

## 五 Agent 的固定职责

| Agent | 读取 | 输出 |
|---|---|---|
| Planner | 用户问题、历史 QueryPlan | `QUERY_PLAN` |
| Retriever | `SEARCH_ACTION`、Provider 配置 | `RESULT_BATCH` |
| CitationExplorer | 高相关 PaperDoc ID | `RELATION_BATCH` |
| EvidenceJudge | PaperDoc 和 `content_ref` | `EVIDENCE_VERDICT` |
| Arbiter | 所有 verdict、关系和预算 | `STOP_DECISION` / `FINAL_SELECTION` |

## 传输限制

1. 消息正文只允许 ID、枚举、数值、布尔值、短标签和路径引用。
2. 长文本统一写 artifact；Agent 之间只传 `content_ref`、`evidence_ref`、`paper_id`。
3. 每个消息必须可重放；不得依赖 Agent 的内存状态。
4. 所有错误显式传递，不得用空数组或低分隐藏错误。
5. P3 初期使用 JSONL 便于调试；测得 token/字节收益后再评估 MessagePack/CBOR，不提前增加二进制依赖。
