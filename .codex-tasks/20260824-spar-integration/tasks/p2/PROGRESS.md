# P2 进度日志

## Context Recovery Block

- **任务**：完成 P2 结构化查询、引用迭代、分量评分、停止与可回放评测
- **形态**：single-full（Epic 子任务）
- **进度**：15/15
- **当前**：P2 已完成优化：宽召回、DeepSeek 理解/判断层和失败回退已接入，等待最终独立验收与 Git 定位
- **文件**：`.codex-tasks/20260824-spar-integration/tasks/p2/TODO.csv`
- **验证**：全量测试、compileall、fixture/replay 已通过；AutoScholar 20 条和 WiFi DeepSeek smoke 已运行；最终独立验收由全新只读 Agent 执行
- **下一步**：P3 五 Agent 结构化协议已实现，进入独立验收和 Git 提交

## 2026-08-24 Start

- P2 按 `PLAN.md` 最新实施设计启动。
- P3 保持 TODO，不在 P2 中实现五 Agent 圆桌。
- 最终验收必须由未参与开发的全新 Agent 执行；开发 Agent 的定向测试不能替代独立验收。
- 当前不要求新增 LLM Key；mock 和确定性 QueryPlanner 必须先可运行。真实 LLM/Provider smoke 使用本地未跟踪配置。

## 2026-08-24 P2 完成与优化

- 固定流程已落地：QueryPlanner → SourceRouter → RecallRunner → 去重 → ConstraintGate → EvidenceLoader → Scorer → CitationExpander → StopController。
- `P2Pipeline` 生成 `query_plan/recall/papers/citation/evidence/verdicts/stop/errors/run_manifest` 九类 JSON；证据摘要会写入 artifact 目录，`replay_p2` 会校验 PaperDoc、manifest query_id 和 evidence_ref 文件存在性。
- arXiv 召回改为短语/OR 宽召回，并将 QueryPlan 时间范围传递给 API 和本地过滤；不再把自然语言问题整体按 AND 查询。
- 新增 DeepSeek 理解/判断层：先生成结构化 QueryPlan，再对候选逐篇判断相关性和硬约束；DeepSeek 失败时保留确定性规则回退，`unknown` 不误删，只有明确 `fail` 才排除。
- WiFi fixture：引用开启为 2 papers、1 条 `references` 边、1 次 relations API；关闭引用为 1 paper、0 边、0 次 relations API。
- 每个最终 PaperDoc 均有六项分数、硬约束状态、证据状态和有效 evidence_ref；AutoScholarQuery 前 20 条查询规划回归通过。
- P2 fixture 和真实 WiFi smoke 均保持 API 错误与论文低相关分离；真实 smoke 使用本地配置和临时环境变量，密钥未写入 artifact。
- P2 的旧基线提交保留；本轮优化单独进入后续 Git 提交，便于回滚和比较。
