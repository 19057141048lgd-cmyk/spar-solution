# P1 进度日志

## Context Recovery Block

- **任务**：实现 P1 的 PaperDoc/schema/mock 首个切片
- **形态**：single-full
- **进度**：17/17；P1 评测协议已验收
- **当前**：身份、Gold、指标、arXiv/LocalLibrary、A/B/C/D 编排、WiFi artifacts 已完成
- **文件**：`.codex-tasks/20260824-spar-integration/tasks/p1/TODO.csv`
- **验证**：60 tests passed；compileall、artifact 清单、secret scan 通过；arXiv live smoke 已返回结构化结果
- **边界**：本地库无真实路径，状态为 mock；Gold 为 provisional；不进入 P2/P3

## P1 Evaluation Protocol — 2026-08-24

- A：arXiv only；B：LocalLibrary only；C：两路去重合并；D：去重后 P1 确定性排序。
- 身份匹配按 DOI → arXiv → 稳定 ID → 标题/年份/第一作者；ambiguous 不强算 TP。
- 评测固定输出 @10/@20、Macro/Micro-F1、延迟、API 次数、来源数量、去重数、错误数。
- WiFi 四查询 artifact：`spar_solution/artifacts/p1/wifi-heart-rate/`。
- 当前结论只能是 fixture/provisional 验证，不能证明效果提升。

## Session Start — 2026-08-24

- 已确认 P1 只在 `spar_solution/` 新增基线代码，不修改 `repos/`。
- 初始 mock 切片阶段未读取真实 Key、未调用真实 API。
- 后续 Provider contract 完成后，OpenAlex 使用本地 `.env.local` 完成 live-smoke；Bohrium 因 Key 缺失显式失败。

## P1 Slice 1 — PaperDoc/schema/mock

- **Status**：DONE
- **Files**：`spar_solution/src/spar_baseline/paperdoc.py`、`mock_pipeline.py`、`spar_solution/tests/`
- **What was done**：实现 PaperDoc v1 校验、身份键、跨源合并、显式 Provider 错误和双来源 mock 闭环。
- **Validation**：5 个 unittest 通过；compileall 通过；git diff --check 通过。
- **Known issue**：todo-list-csv 脚本不能读取 taskmaster 的子任务 TODO 表头，已改用 apply_patch 按 taskmaster 模板更新；不影响项目代码。

## P1 Full Acceptance — 2026-08-24

- **Status**：DONE
- **Parallel work**：配置/Provider contract、Bohrium、OpenAlex、search service、SPAR compat isolation 并行完成。
- **Live query**：`WiFi heart rate monitoring`。
- **Artifact**：`spar_solution/artifacts/p1/wifi-heart-rate-live-local-env.json`。
- **Result**：OpenAlex 成功返回 5 条结构化 PaperDoc；Bohrium 返回 `config` / `BOHR_ACCESS_KEY is not configured`，未被伪装为空结果。
- **Top result**：`WiFi-Based Real-Time Breathing and Heart Rate Monitoring during Sleep`（2019，DOI 已保留在 artifact）。
- **Validation**：27 unittest passed；compileall、git diff --check、secret scan passed；上游 `repos/SPAR` 未修改。
- **Security**：OpenAlex 测试 Key 仅保存在被忽略的 `.env.local`；Bohrium Key 未在当前本地配置中发现；artifact 未包含 Key 或 Bearer 值。
