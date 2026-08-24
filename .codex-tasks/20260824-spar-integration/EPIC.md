# SPAR 三阶段融合 Epic

## Goal

在不丢失 SPAR 查询树和引用演化思想的前提下，吸收 YFR 的数据治理、smartsearch 的 Sciverse 适配与证据协议、K-Dense 的 API 可靠性，并形成可测试、可回放、Agent 间结构化通信的 P1/P2/P3 交付路线。

## Non-Goals

- 本次不修改 SPAR、YFR、smartsearch 或其他原仓库源代码。
- 本次不调用真实学术 API，不验证真实 Key 的效果。
- 本次不实现五 Agent，不做最终比赛 UI 或完整论文综述。

## Constraints

- 全程使用固定 `PaperDoc` 和 JSON/JSONL artifact 协议。
- 密钥只从本地未跟踪环境文件读取，禁止写入 Git。
- P1 的 mock/contract 验收通过后才能进入 P2；P2 消融通过后才能进入 P3。
- Sci-Hub 禁用；只使用合法 OA 和用户提供的测试 URL。

## Child Deliverables

- P1：可运行基线、PaperDoc、Provider contract、安全配置和 mock 测试。
- P2：结构化查询树、引用感知迭代、分量评分和边际收益停止。
- P3：五 Agent 结构化圆桌、可回放 artifact、评测和竞赛封装。

## Dependency Notes

- P2 depends_on P1；P3 depends_on P2。
- 具体子任务状态以 `SUBTASKS.csv` 为准；根目录临时 TODO 以 `SPAR融合方案 TO DO list.csv` 为准。

## Done-When

- [ ] P1、P2、P3 的子任务全部通过各自验收。
- [ ] PaperDoc 和 Agent 协议保持向后兼容。
- [ ] mock、contract、live-smoke 和消融指标均有 artifact 记录。
