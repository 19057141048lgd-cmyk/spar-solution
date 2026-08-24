# P3 进度日志

- **状态**：P3 修复开发完成，独立只读验收通过，待最终 Git 提交确认
- **范围**：五角色 JSON/JSONL 协议、artifact store、P3 fixture、引用消融、消息开销统计
- **当前**：已补两轮 NEXT_QUERY、批量 Judge、冲突复核、统一 identity、P3 replay 和竞赛格式最终输出；未修改 `repos/`，未读取或写入密钥
- **已验证**：P3 定向测试 15 项通过；全量测试 174 项通过；compileall 通过
- **已完成**：全新 Agent 只读验收、全量回归、fixture/replay、通信对照、Git 提交前检查
- **限制**：当前 P3 结果是 fixture/协议验证，不等于真实 Gold 效果提升
- **独立验收**：`/root/p3_fresh_acceptance_20260825` 未参与修改；174/174 tests、compileall、两套 fixture replay、证据和 final_selection 篡改拒绝均通过

## 2026-08-25 P3 修复批次

- 修复单遍直线：P3 每轮记录 Planner/Retriever/CitationExplorer/EvidenceJudge/Arbiter，首轮可发 `NEXT_QUERY`，第二轮受 `MAX_ITERATION` 收敛。
- Arbiter 现在接收全量 verdict artifact；规则与 Judge 分歧超过 0.25 时触发一次批量复核。
- DeepSeek Judge 改为每轮一次批量调用，不再逐篇请求；候选上限进入 QueryPlan budget `max_judge_candidates`。
- 最终选择为 `spar.final.v1`，包含论文元数据、相关性分区、六项分数、证据引用和关系图；`replay_p3()` 校验 JSONL、manifest、final_selection 和证据文件。
- P3 指标复用 P1 identity，并提供 P2/P3、citation 和结构化/长文本通信对照入口；没有 Gold 时不作效果提升结论。
- P3 现在实际写入 P2/P3 与通信对照 artifact；成本包含 provider/LLM 调用、token、延迟，回放会读取并核对证据内容。
