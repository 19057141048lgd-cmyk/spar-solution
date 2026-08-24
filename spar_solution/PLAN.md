# SPAR 科研检索助手三阶段融合方案

## 0. 当前状态与交付边界

本文件是后续开发会话的方案基线。P1 已形成 PaperDoc、Provider、去重、错误协议、arXiv/本地库双路和 A/B/C/D 可执行评测闭环；P2 已形成结构化查询、宽召回、DeepSeek 理解/判断、引用感知迭代、证据评分、停止和 artifact replay；P3 已形成五角色结构化消息、artifact 回放和开销统计。当前 WiFi Gold 仍为 provisional、本地库仍为 mock，因此不能宣称真实效果提升。`P1_ACCEPTANCE.md` 记录已验证项、mock-only 项和阻塞项。

P2/P3 本次补充为实施级设计，但开发仍遵循 `P1 评测闭环 → P2 → P3` 的依赖顺序。本文件只规定方案和验收，不在这里伪造测试集结果。

方案判断来自对以下代码的逐文件分析：

- SPAR：`pipeline_spar.py`、`search_engine.py`、`api_web.py`、`search_node.py`、`rerank.py`。
- YFR：`services/agent/app/daily_review.py`、`paper_search_client.py`、`sciverse_client.py`。
- smartsearch：`src/smart_search/service.py`、`intent_router.py`、`providers/sciverse.py`。
- scientific-agent-skills：`skills/paper-lookup/SKILL.md`、`scripts/paginate.py`、`jats_to_text.py`、`arxiv_atom.py`、`openalex_abstract.py`。
- paper-search-cli / paper-search-mcp / openreview_search：已完成的适配器、混合检索、RRF、全文和引用接口分析。

## 1. 总体判断与保留边界

SPAR 保留为算法主干，因为它唯一已经形成“查询融合 → 搜索树 → 上下文查询演化 → 引用扩展 → 相关性判断”的研究检索闭环雏形。但以下代码当前不能原样使用：引用搜索默认关闭，最终 `rerank` 没有接入，停止条件没有真正使用高相关论文数量，API 字段不统一且存在未定义变量、错误变量、日期失效和依赖缺失。

其他仓库按职责吸收，不整体拼接：

| 组件 | 吸收内容 | 明确不吸收 |
|---|---|---|
| SPAR | `AcademicSearchTree`、`SearchNode`、查询融合、上下文演化、引用树骨架 | 当前默认配置、摘要单分、失效的停止和死代码 rerank |
| YFR | `_normalize_paper`、`_merge_paper`、`_dedupe`、OA 校验、证据/质量/新颖度分层、Daily Delta、draft/resume | 词法质量分作为最终判断、只取 Sciverse 首 3500 字、整套日报流程 |
| smartsearch | Sciverse provider、严格 schema、分页/offset、错误分类、`evidence_items`、`gap_check`、`provider_attempts` | 面向网页的 planner、最多 6 URL 的 research 主循环、网页综合器 |
| scientific-agent-skills | API 选源规则、分页核对、HTTP 200 异常识别、JATS/Atom/倒排摘要解析、provenance | 将 Skill 当成检索算法；它没有排序、查询演化或统一 Paper schema |
| paper-search-cli | 多源 `PaperSource`、缓存/限流/重试、S2 引用、OA PDF 链路 | 跨源首条保留的聚合排序、文本包装输出 |
| paper-search-mcp | Python 适配器、错误隔离、PDF 提取原型 | `to_dict()` 的列表转字符串、默认全源并发、Sci-Hub 回退 |
| openreview_search | 向量+词法+RRF 基线、Chroma 缓存、并发评估框架 | 固定会议库、子串关键词、摘要单维重排 |

## 2. 固定数据协议

所有 Agent、Provider、评估器之间只传输固定的 `PaperDoc`、`QueryPlan`、`EvidenceVerdict`、`StopDecision` 等结构化对象，长正文通过 `content_ref`/`evidence_ref` 指向文件或对象存储。具体字段见 [PAPERDOC_SCHEMA.md](./PAPERDOC_SCHEMA.md) 和 [AGENT_PROTOCOL.md](./AGENT_PROTOCOL.md)。

硬规则：

1. `PaperDoc` 的列表字段永远保持 JSON 数组，不转成分号字符串。
2. DOI、arXiv、S2、OpenAlex、Sciverse 等标识全部保留，不以单一 `paper_id` 覆盖来源身份。
3. 元数据、摘要、局部正文、完整正文必须用 `evidence_status` 区分，不能把 PDF URL 当成全文。
4. 每个分数都必须能回溯到 `query_id`、`iteration`、`source`、`endpoint`、`evidence_ref`。
5. Agent 之间传 ID、枚举、数字、布尔值和短代码；解释性长文本只落盘，不放在消息正文。

## 3. 三阶段实施方案

### P1：可运行基线、统一 PaperDoc 和安全数据层

#### 目标

- 让 SPAR 在没有真实 Key 时可用 mock/fixture 完成端到端运行。
- 清理 SPAR 的导入阻塞和确定性代码错误。
- 固定 `PaperDoc`、Provider、证据和错误协议。
- 接入至少两个真实学术来源和一个正文/引用来源；优先 Sciverse、Semantic Scholar/OpenAlex、Europe PMC/CORE 其中可用者。
- 建立可重复的 baseline：原始 SPAR、P1 修复版、openreview_search RRF 作为对照。

#### 具体结合

1. SPAR 的 `AcademicSearchTree` 继续生成查询节点；不在 P1 重写查询算法。
2. 用 smartsearch 的 `SciverseProvider` 作为一个 `PaperSource`，保留 `/meta-search`、`/agentic-search`、`/content`、`/meta-paper-relations` 的 schema 校验和错误分类。
3. 用 paper-search-cli/mcp 的 API 适配器补充 Semantic Scholar、OpenAlex、arXiv、PubMed/PMC；只保留原始 Paper 对象，不走文本包装输出。
4. 用 YFR 的 normalize/merge/dedupe 逻辑形成唯一数据层，但增加 DOI 规范化、标题 Unicode/标点归一化和版本标记，避免误合并。
5. 用 K-Dense 的分页 reconciliation、JATS/Atom/OpenAlex 解析器补充 `provenance` 和 `full_text_available`。

#### P1 必做任务

- 补齐依赖清单和可复现安装方式；隔离默认 localhost LLM、代理和 Serper 配置。
- 修复 SPAR 已确认错误：Semantic Scholar 未定义 `query`、`"|".json(...)`、引用分支错误变量、`arxivId` 硬取、`end_date` 被清空、`fieldsOfStudy` 类型不统一、可变默认参数。
- 为 Provider 定义 `search/read/relations` 三个最小接口，并统一错误为 `config/auth/rate/timeout/parse/network/empty`。
- 生成每次运行的 `run_id`、`query_id`、`iteration`、`source_errors` 和结构化 artifact。
- 默认关闭 Sci-Hub；只允许合法 OA 站点和用户提供的测试 URL。

#### P1 验收标准

- 在无 Key 的 mock 模式下，固定 fixture 查询可完整生成 `PaperDoc`、候选集、错误记录和最终 JSON；不允许未捕获异常。
- 论文对象通过 schema 校验，DOI 重复、预印本/正式版和跨源字段合并均有单元测试；精确重复去重结果 100% 可复现。
- Provider 失败必须显式进入 `source_errors`，不能静默变成空结果或 `relevance=0`。
- `full_text_available=false` 时，任何评估器不得把该论文标成全文证据充分。
- 真实测试 Key 到位后，选定来源的最小 smoke 用例成功率、响应字段和分页完整性都有日志记录；未配置 Key 时测试必须安全失败。

### P2：查询分解、引用感知迭代和可解释重排序

#### 目标

- 把 SPAR 的平面自然语言扩展改成可执行的结构化查询树。
- 真正启用引用/参考文献扩展和上下文查询演化。
- 将相关性、硬约束、论文质量、证据完整度、引用影响力、新颖度分开判断。
- 用边际收益、覆盖度和预算共同停止，而不是只按 `max_depth` 停止。

#### 具体结合

1. 保留 SPAR `query_fusion`、`query_expand_from_context` 和 `SearchNode` 状态；替换 `expand_query` 的扁平输出为 `QueryPlan`。
2. 用 smartsearch Sciverse `meta-paper-relations`，以及 S2/Europe PMC 的 references/citations 接口填充引用图。
3. 用 YFR 的 relevance/quality/evidence/novelty 字段作为特征，不直接沿用其词法权重。
4. 用 openreview_search 的 RRF 作为召回层消融基线；RRF 后必须保留原始向量、词法、Provider 和引用分数。

#### P2 实施级模块设计

| 模块 | 输入 | 输出 | 主要来源/实现 |
|---|---|---|---|
| `QueryPlanner` | 用户问题、历史 QueryPlan | `QueryPlan`、`QUERY_PLAN` artifact | SPAR `expand_query` + smartsearch intent signals；必须 JSON 校验 |
| `SourceRouter` | 子查询 kind、领域、时间、证据需求 | Provider 列表和预算 | smartsearch capability-first + K-Dense 选源规则 |
| `RecallRunner` | `SEARCH_ACTION` | PaperDoc 批次、分页和错误 | P1 Provider；arXiv/本地库/OpenAlex/Bohrium 并发，限并发数 |
| `FusionDeduper` | 多 Provider PaperDoc | 去重后的 PaperDoc、source agreement | P1 `canonical_paper_key` + YFR merge；保留字段来源 |
| `ConstraintGate` | QueryPlan 硬约束、PaperDoc 元数据/摘要 | pass/fail/unknown 和原因码 | 规则优先；unknown 不得伪装 pass |
| `EvidenceLoader` | PaperDoc、证据需求 | chunk、`EvidenceItem` | smartsearch `/content`、Bohrium content、合法 OA/JATS |
| `CitationExpander` | 通过门控的种子 PaperDoc | relation edges、下一轮候选 | Sciverse relations、S2/Europe PMC references/citations |
| `Scorer` | PaperDoc、evidence、QueryPlan | 分量分和 `EvidenceVerdict` | YFR 特征 + SPAR LLM/embedding judge；分数可回溯 |
| `StopController` | 每轮增益、覆盖、成本、错误 | `StopDecision` | 固定规则，不让 LLM 单独决定是否停止 |

P2 不直接复制 SPAR 的大类 `SearchNode` 字典，而是将每个节点写成可校验 artifact：

```json
{
  "node_id": "n_001",
  "parent_id": null,
  "iteration": 0,
  "subquery_id": "sq_topic_01",
  "kind": "topic",
  "query_text": "WiFi heart rate monitoring",
  "providers": ["arxiv", "local_library", "openalex"],
  "paper_ids": [],
  "new_unique_papers": 0,
  "new_relevant_papers": 0,
  "relation_edges": [],
  "stop_reason": null,
  "provenance_ref": "artifacts/run/n_001.json"
}
```

P2 的最小可执行流程固定为：

```text
QueryPlanner
  → SourceRouter
  → RecallRunner（并发、有界）
  → FusionDeduper
  → ConstraintGate
  → 初排和种子选择
  → CitationExpander / EvidenceLoader
  → Scorer
  → QueryPlanner 只根据 gap 生成下一轮子查询
  → StopController
```

查询迭代不是“每轮重新让 LLM 自由改写”，而是由 gap 驱动：`missing_method`、`missing_dataset`、`missing_time_range`、`missing_application`、`citation_neighbor_gain`。每个 gap 最多生成 2 条子查询，单次运行最多 2 次迭代、引用深度最多 1；这些上限先保证可运行和可验收，后续由消融实验调整。

P2 的默认排序分量暂不宣称学术最优，先作为可校准特征：

```text
relevance       0.30
constraint      0.25
evidence        0.20
quality         0.10
citation        0.10
novelty         0.05
```

硬约束不通过时直接排除；unknown 不排除，但最高只能进入“待核验”区。最终权重必须在开发集标注上校准，并保留无重排、RRF、LLM 重排三个消融结果，不能把初始权重写成事实结论。

P2 强停止条件：`BUDGET_EXHAUSTED`、`MAX_ITERATION`、`MAX_CITATION_DEPTH`、`ALL_PROVIDER_FAILED`、`NO_NEW_PAPER_2_ROUNDS`。软停止条件至少同时满足“新增高相关论文低于 2、子查询覆盖率 ≥ 0.8、证据覆盖率 ≥ 0.7”中的两项。所有阈值写入运行配置和 `StopDecision`，不得隐藏在提示词中。

#### P2 查询与迭代规则

`QueryPlan` 至少包含：主题、方法、数据集/任务、时间范围、硬约束、软约束、候选来源能力、预算和停止策略。每条子查询有 `subquery_id`、`parent_id`、`kind`（topic/method/dataset/constraint/comparison/reference）、`source_capabilities` 和 `iteration`。

每轮流程固定为：

```text
QueryPlan → 并行召回 → PaperDoc 去重合并 → 约束门控
→ 相关性初排 → 选择引用种子 → citation/reference 扩展
→ 从新增证据生成下一轮子查询 → StopDecision
```

引用扩展只允许来自：硬约束通过、证据状态至少 abstract、相关性超过阈值的种子；禁止把全部 `irrelevant_docs` 送入引用图。引用节点必须带父节点、关系类型、来源 API 和深度。

最终分数先保存分量，不直接硬编码一个不可解释的总分：

```text
final = calibrated(relevance, constraint, evidence, quality, citation, novelty)
        - hard_constraint_penalty - duplicate_penalty - warning_penalty
```

初始权重只作为配置和消融变量，待标注集校准；引用数不得直接等同论文质量。

#### P2 停止规则

满足任一强停止条件：预算耗尽、达到最大引用深度、连续两轮无新 PaperDoc、API 全部不可用。满足以下组合条件可提前停止：新增高相关论文数低于阈值、子查询覆盖率达到目标、引用图新增有效节点低于阈值、证据覆盖度达到目标。每次停止必须输出 `StopDecision.reason_code` 和数值依据。

#### P2 验收标准

- 同一查询可回放出完整查询树、每轮候选数、引用关系和停止理由。
- 引用扩展实际发生并能回溯到父论文；关闭引用模块时能做消融对照。
- 每个最终结果均有分量分数、硬约束判断和至少一个 `evidence_ref`；仅摘要结果必须明确标记。
- 在固定评测集上至少比较 Recall@K、Precision@K、F1@K、MRR、引用覆盖率、证据覆盖率、平均延迟和 API 调用量；P2 不得以“看起来更合理”代替指标。

### P3：五 Agent 结构化圆桌、评测闭环和竞赛交付

#### 目标

- 将 P2 的单一判断改为五角色 Agent 的受限协作。
- Agent 之间只传 JSON/JSONL artifact 引用，不传长自然语言。
- 建立可重放、可审计、可做消融的比赛版本。

#### 五 Agent 角色

1. `Planner`：生成/修订 `QueryPlan`，只能输出结构化子查询和约束。
2. `Retriever`：执行 Provider 搜索，返回 `PaperDoc` 引用和召回统计。
3. `CitationExplorer`：扩展 references/citations/related works，返回关系边。
4. `EvidenceJudge`：按硬约束、相关性、证据、质量分量给出 `EvidenceVerdict`。
5. `Arbiter`：比较各 Agent 的分量分、证据覆盖和冲突代码，生成 `FinalSelection` 或 `StopDecision`。

每个 Agent 的输入输出、字段上限和文件位置固定在 [AGENT_PROTOCOL.md](./AGENT_PROTOCOL.md)。长摘要/全文只由 EvidenceJudge 通过 `content_ref` 读取，不在 Agent 消息中复制。

#### P3 实施级调度与仲裁

P3 采用有向的 fan-out/fan-in，而不是五个 Agent 无限自然语言辩论：

```text
Planner
  → Retriever ─────┐
  → CitationExplorer ─┤
  → EvidenceJudge ────┘
          → Arbiter
          → STOP / NEXT_QUERY / FINAL_SELECTION
```

- Planner 每轮最多一次；只有 `StopDecision=NEXT_QUERY` 才能再次调用。
- Retriever 和 CitationExplorer 可以并发，但必须写入不同 artifact 文件。
- EvidenceJudge 只读取 `PaperDoc` 和 `content_ref`，不负责扩大召回。
- Arbiter 先用规则处理硬约束、重复和错误状态；只有冲突时才调用 LLM 仲裁。
- 一次运行最多 2 个圆桌轮次；每个候选最多 1 次 EvidenceJudge；预算耗尽立即输出 degraded 结果。

Arbiter 的固定决策优先级：

```text
Provider/schema error → 排除该证据，不降低论文相关性分
hard_constraint=false → 排除
evidence_status=metadata 且问题要求全文 → 待核验，不进入最终首位
多个 Judge 分歧 > 0.25 → conflict，触发一次复核
否则按 calibrated final_score + evidence confidence 排序
```

P3 的新颖性只在有实验数据时宣称：结构化 PaperDoc/证据图、引用关系和增量收益共同驱动 Agent 调度；不是单纯增加 Agent 数量。必须比较“P2 单 Agent”“P3 五 Agent JSON/Artifact”“P3 五 Agent 长文本通信”三组，若五 Agent 不能提升 F1、证据覆盖或成本效率，就不把它作为最终主路径。

#### P3 验收标准

- 可从 artifact 目录重放一次运行，重放结果包含 query tree、PaperDoc、relations、verdicts、stop decision 和 provenance。
- 五 Agent 的消息均符合协议；自然语言备注仅允许短诊断字段，不能承载论文正文或完整综述。
- 任何 Provider、LLM 或 Judge 失败都产生可区分的 degraded/fail 状态，不能伪装成正常低分。
- 对比 P1/P2/P3 的 Recall@K、F1@K、证据覆盖、引用覆盖、延迟、Token/API 成本；P3 不得降低召回，且必须证明结构化通信相对长文本通信的 token/字节收益。
- 竞赛输出只包含经过证据校验的 PaperDoc 选择、理由代码、证据引用和指标，不输出无法追溯的结论。

## 4. 融合与引用规则

### 4.1 来源路由

- 主题/跨学科：Sciverse semantic + Semantic Scholar/OpenAlex。
- 生物医学全文：Europe PMC/PMC 优先；PubMed 只作摘要元数据。
- 预印本：arXiv；正式出版信息用 Crossref/OpenAlex 补齐。
- 引用图：Semantic Scholar 或 Europe PMC relations；OpenAlex/Crossref 作为补充，不以 citation count 代替关系图。
- OA 正文：Unpaywall → PMC/Europe PMC/CORE/OpenAIRE 等合法来源；Sci-Hub 永久禁用。

### 4.2 去重与合并

优先级：规范化 DOI → arXiv/S2/OpenAlex/Sciverse/PMCID 等稳定 ID → 标题+年份+作者相似度。重复记录必须合并来源、摘要、PDF 候选、引用数和 provenance；冲突字段保留 `field_sources`，不得静默覆盖。

### 4.3 证据规则

- `metadata`：只能用于召回和基础过滤。
- `abstract`：可用于主题相关性，不能声称验证正文方法/实验。
- `partial_text`：只能引用明确 chunk；必须记录 offset/section/page。
- `fulltext`：才允许做方法、实验和结论核验。
- HTTP 200 但响应体是错误、JATS 无 body、分页短缺、解析警告均写入 `warnings` 并降低 evidence confidence。

### 4.4 失败与停止规则

Provider 错误不降为论文相关性 0，而是进入错误集合；同一错误达到重试上限后触发 fallback 或 `degraded`。停止由预算、增益、覆盖和证据共同决定，并输出数值和 `reason_code`。

## 5. 评测标准、官方边界与公开基准

### 5.1 赛题已明确和未明确的内容

根据赛题文档，已明确：

- F1 Score 占竞赛评测 70%；
- 运行效率占 20%，关注 API 调用次数、Token 消耗和端到端延迟；
- 结构化结果占 10%；
- 竞赛得分由公开测试集 30% + 隐藏测试集 30% 自动评分组成，专家评分占 40%；
- 必须接入至少一种学术搜索 API。

赛题文档没有明确提供：

- Gold 论文答案文件；
- DOI/arXiv/标题匹配规则；
- `K` 值（如 F1@10 或 F1@20）；
- Macro-F1/Micro-F1 的汇总方式；
- 官方评测脚本或本地复现命令。

因此，队友的 TP/FP/FN 公式是正确的通用定义，但不是完整的官方验收协议。我们必须把自己的评测分为“官方分数不可复现”和“本地可复现实验”两层，不能把临时 Gold 称为官方答案。

### 5.2 三层评测集

1. **赛事层**：提交官方公开/隐藏测试流程；只记录官方返回分数，不能提前知道隐藏答案。
2. **标准基准层**：优先接入 [SPARBench](https://huggingface.co/datasets/XiaofengAlg/SPARBench)、[AutoScholarQuery/RealScholarQuery](https://github.com/bytedance/pasa)、[LitSearch](https://github.com/princeton-nlp/LitSearch)；这些数据有论文集合或相关性标签/可运行评测代码。
3. **领域回归层**：WiFi 心率监测四查询和自建库查询，用 provisional Gold 做回归，不用于声称官方成绩。

公开基准的定位不同：SPARBench 与本项目算法最贴合，支持专家相关性标签；AutoScholarQuery/RealScholarQuery 适合验证查询迭代和引用扩展；LitSearch 有 597 个 ML/NLP 查询、语料和 BM25/密集检索/重排代码，适合检索层回归，但不覆盖 WiFi 生理监测领域。所有外部数据的许可证、时间截断和可访问性必须记录。

可选扩展是 [AstaBench](https://github.com/allenai/asta-bench) 的 PaperFindingBench。它提供 validation/test split、日期受限语料和更完整的 agent 成本评估，但环境和数据访问要求更高，适合 P3 交付前做外部验证，不作为 P1 阻塞项。

### 5.3 统一本地评测协议

每个 query 先按 DOI → arXiv ID → 稳定来源 ID → 标题+年份+首作者匹配；无法确认的记录为 `ambiguous`，不计 TP。分别计算 `P@10/R@10/F1@10`、`P@20/R@20/F1@20`、Macro-F1、Micro-F1、MRR、延迟、API 调用、Token、来源错误、去重数、证据覆盖率和引用覆盖率。

必须保留以下消融：

```text
A：单 arXiv
B：单自建论文库
C：A+B 直接合并去重
D：A+B 统一排序
E：D + 查询迭代
F：E + 引用扩展
G：F + P3 五 Agent
```

回归门槛采用相对比较而不是拍脑袋绝对值：`C` 的 Recall 不低于 A/B 中较优者；`D` 的 F1 不低于 C；`E/F/G` 每增加机制都必须报告增益、成本和失败率。若没有 Gold，只能报告“可运行”和“候选质量人工抽查”，不能报告效果提升。

## 6. 密钥、URL 和测试管理

### P1 评测协议补充（2026-08-24）

P1 仅新增可执行评测闭环，不改变 P2/P3。`experiment.py` 固定 A/B/C/D 四模式，`identity.py` 固定身份优先级，`gold.py` 校验 `spar.gold.v1`，`metrics.py` 固定 @10/@20、Macro/Micro-F1 和运行统计，`providers/arxiv.py` 与 `providers/local_library.py` 提供 arXiv 及本地库边界。四个 WiFi 查询的 fixture artifact 位于 `spar_solution/artifacts/p1/wifi-heart-rate/`；本地库当前没有真实路径，所有本次 A-D 结果均明确为 mock，Gold 为 provisional，不能据此宣称效果提升。

根目录和每个查询目录都保留固定结果文件，PaperDoc 通过 `paperdoc.v1` 校验，Provider 错误保存在 `errors.json`/`source_errors`，不会转成不相关论文。D 只使用可复现的 P1 元数据排序，P2 才允许引入语义重排。

本地使用 `.env.local` 或 `config/test.env`，两者必须加入根 `.gitignore`；只提交 `.env.example` 的变量名和 URL 占位符，不提交任何 token、邮箱、Cookie 或真实响应。

建议变量名：`SCIVERSE_API_BASE_URL`、`SCIVERSE_API_TOKEN`、`SEMANTIC_SCHOLAR_API_KEY`、`OPENALEX_MAILTO`、`CROSSREF_MAILTO`、`CORE_API_KEY`、`UNPAYWALL_EMAIL`、`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`SERPER_API_KEY`。实际收到的 Key 只写入本地测试文件，由统一 config loader 自动读取；日志对 token、email、URL query key 做脱敏。

测试分三层：

1. `mock`：无网络、无 Key、固定 fixture，验证协议和算法。
2. `contract`：对真实测试 URL 做 HTTP 状态、schema、分页和内容类型验证。
3. `live-smoke`：少量查询、严格预算、密钥不回显，只记录 source、状态、耗时、数量和错误码。

## 7. 后续执行顺序

当前会话只提交本方案和工作区初始化。后续新会话必须按以下顺序推进：

1. P1 待办 11-17 已完成：身份、Gold、指标、arXiv/LocalLibrary、A/B/C/D 对照和 WiFi artifact；当前保持 P1 边界。
2. P1 评测闭环通过后实现 P2 的 `QueryPlan`、SourceRouter、RecallRunner、CitationExplorer、Scorer 和 StopController。
3. P2 先在 SPARBench/AutoScholarQuery/LitSearch 中做外部基准和消融，再决定是否调整默认权重和阈值。
4. P2 的查询树、引用图、评分和停止规则通过后才进入 P3；P3 不重复实现 Provider。
5. P3 只增加受限五 Agent 调度、结构化通信、仲裁和回放，并以 P2 为不可退化基线。

所有阶段以任务追踪文件和 artifact 为准，不以聊天内容为准。
