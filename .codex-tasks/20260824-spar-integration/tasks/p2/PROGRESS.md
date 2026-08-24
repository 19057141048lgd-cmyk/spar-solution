# P2 进度日志

## Context Recovery Block

- **任务**：完成 P2 结构化查询、引用迭代、分量评分、停止与可回放评测
- **形态**：single-full（Epic 子任务）
- **进度**：20/20
- **当前**：P2 改进批次已完成代码修复、独立只读审查、全量回归、全新 Agent 验收和 Git 定位
- **文件**：`.codex-tasks/20260824-spar-integration/tasks/p2/TODO.csv`
- **验证**：全量测试、compileall、fixture/replay 已通过；AutoScholar 20 条和 WiFi DeepSeek smoke 已运行；最终独立验收由全新只读 Agent 执行
- **下一步**：保留当前提交作为 P2 改进基线；P3 保持既有实现，不在本轮扩展

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

## 2026-08-24 老师改进批次收尾

- A：时间边界、首轮停止、OpenAlex 空结果和 arXiv smoke 已修复；旧 AND/错误 Gold 产物归档到 `spar_solution/artifacts/_invalid/`。
- B：Gold 身份、年份过滤、重复评分、统一指标已修复；P2 指标按关系调用内的实际 HTTP 次数计量。
- C：OpenAlex relations 支持 citations/references/all；DeepSeek 计划与候选判断接入主管线，非法输出触发规则回退；引用子论文再次判断；token、调用预算与 `spar.final.v1` 已落盘。
- 当前代码全量 `unittest`、compileall、diff check 已通过；真实 DeepSeek/Bohrium 是否可用只由本地配置状态决定，不把缺 Key 当作论文不相关。

## 2026-08-24 最终验收

- 全新只读 Agent 独立回放 citation enabled/disabled 两套 fixture：两轮均可回放，最终停止记录为 `MAX_ITERATION`，证据文件存在，`spar.final.v1` 可校验。
- 当前全量测试为 167 项；P2 TODO 20/20 完成。
- 历史 `live-wifi-deepseek`、旧 `self-test-final`、旧 `self-test-optimized` 已移入 `spar_solution/artifacts/_invalid/`，不参与效果声明。
- P2 仍不能声称真实效果提升：WiFi Gold 是 provisional，DeepSeek/Bohrium/LocalLibrary 的生产路径未同时具备可用配置。
