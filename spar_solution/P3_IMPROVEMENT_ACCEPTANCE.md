# P3 修复验收记录

## 分工

- 代码审查：`/root/p3_roundtable_review`，只读检查逐条问题和回归风险。
- 修改执行：主 Agent 与已分配实现 Agent，只修改 `spar_solution/` 和 P3 任务文件，不修改 `repos/`。
- 最终验收：`/root/p3_fresh_acceptance_20260825`，本轮未参与代码修改，只读执行全量测试、fixture 回放、协议校验和密钥扫描。

## 当前修复范围

1. 两轮 `NEXT_QUERY`/`MAX_ITERATION` 圆桌调度。
2. 全量 verdict artifact、批量 Judge、分歧复核和失败降级标记。
3. QueryPlan 候选预算、来源级 API 调用/延迟、DeepSeek token/延迟记录。
4. `spar.final.v1` 元数据、相关性分区、六项分数、证据引用和关系图。
5. `replay_p3()` 对消息顺序、artifact 内容、query_id、证据文件归属和最终选择一致性进行校验。
6. P1 identity 匹配、P2/P3 对照、citation 消融和 structured/long-text 通信对照。
7. 共享 P2 PaperDoc 准备/评分路径，fixture 增加零种子和证据篡改回归。

## 已执行验证

- 定向 P3 测试：15 项通过。
- 全量测试：174 项通过。
- `compileall`：通过。
- fixture：引用开关两套、P2/P3 对照和结构化/长文本对照已生成到 `artifacts/p3/`。
- 全新只读 Agent（`/root/p3_fresh_acceptance_20260825`，未参与修改）验收通过：174/174 单测、compileall、两套 fixture replay、协议 seq/receiver/payload 内容一致性、证据篡改拒绝和 final_selection 篡改拒绝均通过。
- communication 对照同时包含 `recall`、`f1`、`latency_ms`、`provider_calls`、`llm_calls`、`total_tokens`；P2/P3 对照也保留同一组成本字段。

## 结论边界

状态：`PASS（fixture/协议验收）`。

已验证：代码、协议、artifact 回放、两轮 NEXT_QUERY/MAX_ITERATION、批量 Judge、分歧复核、证据归属和最终输出结构。

仅 mock 验证：P2/P3、引用开关、structured/long-text 通信对照，以及无 Key fixture 的成本统计。

真实 API 已验证：本次 P3 修复未发起 DeepSeek、Bohrium、OpenAlex 或真实 LocalLibrary 请求，因此不能宣称真实 API 效果已验证。

Gold 边界：当前没有可用于证明效果提升的官方 Gold；任何 provisional Gold 结果只能作为调试信息，不能作为效果提升结论。

当前阻塞：真实 API smoke、真实本地库路径和正式 Gold 评测仍待后续阶段；本批次不进入 P4/P5 范围。
