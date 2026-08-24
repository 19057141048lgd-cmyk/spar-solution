# P1：可运行基线、PaperDoc、安全数据层和检索评测协议

## P1 完整目标

完成可运行学术检索基线：PaperDoc、Provider contract、本地配置、Bohrium/OpenAlex 适配、跨源检索/去重/结构化 artifact、mock/contract/live-smoke 测试，以及 SPAR 已知阻塞的兼容修复记录。最终对“WiFi 心率监测”执行真实检索并保存可复验结果。

## 非目标

- 不修改 `repos/` 中任何原始仓库。
- 不实现查询树、引用扩展、LLM 重排或五 Agent。

## 验收

- 缺失字段、非法枚举、列表变字符串、错误证据状态会被拒绝。
- 相同 DOI 的两条来源记录会合并，数组字段不会变成字符串。
- Provider 错误会进入结构化 `source_errors`，不会写成论文低相关分。
- mock 闭环能输出 PaperDoc、合并统计和错误统计。
- 测试密钥只从未跟踪的 `.env.local` 读取，输出和日志不回显密钥。
- Bohrium/OpenAlex 至少一个来源真实返回可解析响应；失败来源必须记录明确错误。
- “WiFi 心率监测”检索结果保存为结构化 JSON，包含 query、来源、PaperDoc、错误和统计。
- `python -m unittest discover -s spar_solution/tests -v`、compileall、secret scan 通过。
- A/B/C/D 四组检索模式和可执行指标协议必须通过 fixture 验收，不能只检查说明文档。

## P1 可执行论文检索评测协议

### 四组对照模式

- `A_arxiv`：只调用 arXiv API。
- `B_local`：只调用 `LocalLibraryProvider`。无路径时状态为 `unavailable`；fixture 时状态为 `mock`，不得伪造 live 结果。
- `C_fusion`：调用 arXiv 与本地库，先按固定身份规则去重合并，再计算指标。
- `D_reranked`：在 C 的去重结果上执行 P1 确定性元数据排序，不冒充 P2 语义重排。

### 身份与 Gold

评测前必须先去重。身份优先级固定为 DOI、arXiv ID、Semantic Scholar/OpenAlex/本地稳定 ID、规范化标题+年份+第一作者；冲突或字段不足返回 `ambiguous`，不得强行计 TP。Gold 使用 `spar.gold.v1`；当前 WiFi 集合为人工 `provisional` 标注，不是官方答案，每条记录保存判断依据和标注时间。

### 指标与安全规则

`TP=预测集合∩Gold集合`，`FP=预测集合-TP`，`FN=Gold集合-TP`。固定输出 Precision/Recall/F1@10/@20、Macro-F1、Micro-F1、平均延迟、API 调用次数、各来源返回数、去重数和 source_errors 数。分母为零时返回 `0.0`；Provider/API/解析错误只进入审计字段，不作为低相关论文。

### 验收门槛

- C 的 Recall@10、Recall@20 不低于 A/B 两者较优者，否则 `fusion_regression=true`。
- D 的 F1@10、F1@20 不低于 C，否则 `rerank_regression=true`。
- 没有官方 Gold 时只能标记 provisional，必须输出“暂不能证明效果提升”。
- 每个查询目录固定写入 `query.json`、`gold.json`、四种 `results_*.json`、`metrics.json`、`errors.json`、`run_manifest.json`；根目录同时写汇总版固定文件名。
