# 改进方案与执行计划（IMPROVEMENT_PLAN）

日期：2026-08-24
性质：本文件是本轮改进的唯一执行依据。所有问题、设计、改动方式、验收办法、待办事项集中记录在此。
执行原则：**先本方案获批，后动代码**；每项改完必须跑通对应验收，再进入下一项。

---

## 0. 背景摘要（为什么会有这份方案）

独立审查结论：项目方向正确、数据协议层（PaperDoc/identity/错误协议）质量合格，不需要推翻重来；
但存在 4 个 P0 级 bug 和若干 P1 级缺口，导致"迭代检索、引用扩展、LLM 判断"三大核心能力
目前实际是死代码或仅存在于 fixture 中。本方案分三批（A/B/C）修复和补齐，全部完成前不启动 P3。

审查已验证的关键事实（证据详见各项"问题"小节）：

1. 时间范围硬约束会误杀区间内的正常年份论文（已复现）。
2. 停止条件在第 0 轮就终止循环，第 2 轮迭代从未执行过（artifact 证实）。
3. OpenAlex 把合法空结果当错误抛出，可误触 ALL_PROVIDER_FAILED。
4. AutoScholarQuery Recall=0 的两个产物基于旧版 AND 查询代码，修复至今未提交，数字不可信。
5. 没有任何真实 Provider 支持引用关系（relations），引用扩展只能跑 fixture。
6. LLM（DeepSeek）只在评测脚本里用过，主管线的相关性评分是纯词法重叠。
7. Token 计量完全缺失，而赛题效率分（20%）明确计入 Token 消耗。
8. 赛题要求的结构化展示（10%）没有任何实现。

---

## 1. 改动范围总览

| 编号 | 名称 | 级别 | 批次 | 改动类型 | 依赖 |
|---|---|---|---|---|---|
| A-1 | 时间范围约束格式统一 | P0 | A | 改代码+补测试 | 无 |
| A-2 | 停止条件语义修复（迭代可达） | P0 | A | 改代码+补测试 | 无 |
| A-3 | OpenAlex 空结果不再当错误 | P0 | A | 改代码+改测试 | 无 |
| A-4 | 提交 arXiv 修复 + 重建可信 smoke 基线 | P0 | A | git 提交+新增 CLI | A-1 |
| B-1 | Gold/fixture 共享 mock DOI 防御 | P1 | B | 改代码+补测试 | 无 |
| B-2 | OpenAlex 支持年份过滤参数 | P1 | B | 改代码+补测试 | 无 |
| B-3 | 管线跳过已评分论文 | P1 | B | 改代码+补测试 | A-2 |
| B-4 | 统一 F1/身份匹配实现 | P1 | B | 重构+补测试 | 无 |
| B-5 | 文档同步（测试数、MRR 说明） | P1 | B | 改文档 | 无 |
| C-1 | OpenAlex 引用关系（relations）实现 | P1 | C | 新功能+测试 | A-3 |
| C-2 | DeepSeek 接入主管线（规划+相关性） | P1 | C | 新功能+测试 | A-1 |
| C-3 | Token/成本计量入 manifest | P1 | C | 新功能+测试 | C-2 |
| C-4 | FINAL_SELECTION 结构化输出模块 | P1 | C | 新功能+测试 | B-3 |

本轮明确不做（遗留清单，见第 4 节）：五 Agent 圆桌、多模型、BM25/向量检索、
`_norm_text` 正则统一、DOI 冲突判 ambiguous 的校准、O(n²) 去重优化等。

建议执行方式：新建分支 `improve/p0-p1-batch`，每项一个提交，全部验收后合回 main。

---

## 2. 批次 A：P0 修复（不修完不做任何其他事）

### A-1 时间范围约束格式统一

**存在的问题**

- `query_planner.py:171` 生成的硬约束值是冒号格式：`f"{start or ''}:{end or ''}"`，
  即 `"2018:2021"`；只有起始年时是 `"2018:"`。
- `p2_evidence.py:122` 的区间正则只认 `-` / `to` / `~` / `至`：
  `re.search(r"(\d{4})\s*(?:-|to|~|至)\s*(\d{4})", expected)`，匹配不上冒号，
  于是落入"年份 ∈ 精确集合 {2018, 2021}"的分支。
- 后果（已本地复现）：查询 "published between 2018 and 2021"，2019 年论文被判
  `fail ('year_mismatch',)` → `hard_constraints_pass=False` → Scorer 将其
  `excluded=True, final=None`，论文直接从最终结果中消失。
- 现有测试为什么没拦住：`tests/test_p2_evidence.py:11` 手写的是 `"2020-2025"`
  连字符格式，恰好是 gate 支持的格式；planner→gate 的集成测试不存在。

**为什么需要这么做**

硬约束误杀直接损失 Recall，而 F1 占赛题总分 70%。这是目前对检索效果伤害最大的单点 bug。

**怎么改动（设计）**

1. 改 planner 输出为自描述、无歧义的格式（`query_planner.py` `_deterministic` 内）：
   - 双边界：`"2018-2021"`
   - 只有起始年（since/after）：`">=2018"`
   - 只有结束年（before/until）：`"<=2021"`
2. 改 gate 解析（`p2_evidence.py` `_evaluate_one` 的 year 分支），新增两条规则并保留旧格式兼容：
   - `(\d{4})\s*-\s*(\d{4})` → 区间判断（原 `-`/`to`/`~`/`至` 继续支持）
   - `>=\s*(\d{4})` → `year >= n`；`<=\s*(\d{4})` → `year <= n`
   - 解析不出任何年份 → 维持 `unknown/year_unparseable`，不许推断为 pass
3. 新增集成测试 `tests/test_planner_gate_integration.py`：
   从 `QueryPlanner().plan("... between 2018 and 2021 ...")` 取真实 `hard_constraints`
   喂给 `ConstraintGate`，断言 2017 fail、2019 pass、2021 pass、2022 fail；
   再断言 `since 2018` 与 `before 2021` 两种开放区间。

**验收办法**

```bash
cd C:\Users\richd\Desktop\科研助手
python -m unittest discover -s spar_solution/tests -q   # 全部通过，新增 >=4 个用例
```
外加手工复现脚本：2019 年论文对 "between 2018 and 2021" 必须返回 `pass`。

**待办事项**

- [ ] 修改 `query_planner.py` 时间约束值格式（含单边界情形）
- [ ] 修改 `p2_evidence.py` 年份解析（新增 `>=`/`<=`/区间 `-`，保留旧格式）
- [ ] 新增 `tests/test_planner_gate_integration.py`（≥4 用例）
- [ ] 全量测试通过

---

### A-2 停止条件语义修复（让第二轮迭代真正可达）

**存在的问题**

- `p2_stop.py:122-123`：`if citation_depth >= max_citation_depth` 就加入强停止
  `MAX_CITATION_DEPTH`。而管线在 iteration 0 做了一次引用扩展后 `citation_depth=1`、
  阈值也是 1，于是**第 0 轮结束即整体终止**。
- 软停止同样在第 0 轮就计数：首轮"新增高相关 < 2"几乎必然成立（`LOW_RELEVANT_GAIN`），
  子查询首轮全部调用成功又给 `SUBQUERY_COVERAGE_MET`，两个软条件凑齐即软停。
- 证据：`artifacts/p2/self-test/stop.json` = `('strong','MAX_CITATION_DEPTH',0,1)`；
  `artifacts/p2/self-test-no-citation/stop.json` = `('soft',...,0,0)`。
  **两个 fixture 的第 1 轮迭代都从未执行。**
- 概念错误：引用深度预算应该约束"还要不要继续扩引用"，而不是"整个检索循环停不停"。
  深度约束在 `CitationExpander(max_depth=1)` 内部已经生效，StopController 里这条是重复且有害的。

**为什么需要这么做**

赛题 3.1(2) 明确要求"迭代式检索策略：根据已找到的相关论文动态调整检索策略"。
当前迭代是死代码，等于没有这个能力；且这是 SPAR 思路（查询演化）的核心价值点。

**怎么改动（设计）**

1. `p2_stop.py`：
   - 强停止集合收敛为 `BUDGET_EXHAUSTED / ALL_PROVIDER_FAILED / NO_NEW_PAPER_2_ROUNDS / MAX_ITERATION`
     四个；`MAX_CITATION_DEPTH` 从"强停止原因"降级为"触发的条件"（只出现在
     `triggered_conditions` 里供审计，不再令 `should_stop=True`）。`STRONG_REASON_CODES`
     常量保留它以兼容旧 artifact 读取，但 `decide()` 不再因它返回停止。
   - 软停止前置条件：`iteration >= 1` 才允许评估软条件（首轮没有"历史增益"可比）。
2. `p2_pipeline.py`：`citation_depth` 仍上报给 `decide()`（审计用），逻辑不变。
3. 更新测试 `tests/test_p2_stop.py`：
   - iteration=0 + citation_depth=1 → `should_stop=False`，`triggered_conditions` 含
     `MAX_CITATION_DEPTH`；
   - iteration=0 + 软条件全满足 → 仍 `continue`；
   - iteration=1 + 软条件满足（2 项）→ `soft` 停止；
   - iteration 达到 max_iterations → `strong/MAX_ITERATION`。
4. 更新管线级断言：`run_p2_fixture(citation_enabled=True)` 产出的 `stop.json`
   必须有 2 条记录（iter0=continue，iter1=stop），`run_manifest.json` 的
   `iterations == 2`。

**验收办法**

```bash
python -m unittest discover -s spar_solution/tests -q
python -m spar_solution.src.spar_baseline.p2_cli fixture --output spar_solution/artifacts/p2/self-test
# 检查 stop.json：第一条 decision_type=continue，第二条为 stop；run_manifest.json iterations=2
```

**待办事项**

- [ ] `p2_stop.py` 强停止集合调整 + 软停止 iteration>=1 前置
- [ ] `tests/test_p2_stop.py` 用例更新（≥4 用例）
- [ ] `tests/test_p2_pipeline.py` 迭代数断言更新
- [ ] 重生成两个 self-test artifact 并核对 stop.json

---

### A-3 OpenAlex 空结果不再当错误

**存在的问题**

- `openalex_provider.py:288-289`：`if not results: raise ProviderError("openalex", "empty", ...)`。
- 违反项目自己的协议（PLAN 硬规则：API 返回为空 ≠ 检索失败）。
- 连锁后果：`RecallRunner` 把该调用记为 `ok=False`；若所有来源对某子查询都合法地
  返回 0 条，`provider_successes==0` → 误触 `ALL_PROVIDER_FAILED` 强停止，
  把"没搜到"误诊为"供应商全挂"，后续轮次被错误终止。
- 注意：arXiv Provider 没有这个问题，只有 OpenAlex 抛。

**为什么需要这么做**

空结果是正常业务结果（查询过窄、时间窗内无文献）。把它记为故障会污染失败率统计、
误导停止决策，并且和"错误不伪装成空结果/低分"的协议方向正好搞反。

**怎么改动（设计）**

1. `openalex_provider.py` `search()`：删除该 raise；`results` 为空时正常返回
   `ProviderResult(records=[], total=meta.count 或 0, warnings=["no_results"])`。
2. `providers/base.py` 的 `ErrorCode` 中 `empty` 枚举保留（其他来源仍可用），
   但 OpenAlex 不再使用它表达"成功但零结果"。
3. 更新 `tests/test_openalex_provider.py` 中断言抛错的用例：改为断言
   `ok=True`、`records==[]`、`total==0`、warnings 含 `no_results`。
4. 新增 `tests/test_p2_recall.py` 用例：两个 Provider 都合法返回空时，
   StopController 不得给出 `ALL_PROVIDER_FAILED`。

**验收办法**

```bash
python -m unittest discover -s spar_solution/tests -q
```

**待办事项**

- [ ] 删除 `openalex_provider.py` 空 results 的 raise，改为正常返回
- [ ] 更新对应单测；新增"全空不误诊"的 RecallRunner/Stop 联动用例
- [ ] 全量测试通过

---

### A-4 提交 arXiv 修复 + 重建可信 smoke 基线

**存在的问题**

- 工作区有**未提交**的 `providers/arxiv.py` 改动（AND → OR 宽召回，+64 行）和
  `p2_recall.py`、`test_arxiv_provider.py` 改动。HEAD 版本
  `_search_expression` 是 `" AND ".join(...)`——把整个问句逐词 AND，
  `artifacts/p1/auto-scholar-smoke-5/`（5 条全部 0 篇）和
  `artifacts/autoscholar/raw-arxiv-sample20/`（20 条全部 0 篇）都是这个坏版本跑出来的。
- 当时生成 smoke-5 的脚本已被删除，无法复现实验；且其 Gold 构造受 mock DOI
  误合并影响（qid test_2：gold_count=4 但 fn=1，4 篇被并成 1 篇）。
- 因此现有两个"Recall=0"产物**不可信也不能比**，但它们还躺在 artifacts 里容易被当成结论。

**为什么需要这么做**

没有可信基线，后面一切消融（E=迭代、F=引用、G=Agent）都没有对照物。
且修复代码不提交，任何一次误操作都可能把它丢掉。

**怎么改动（设计）**

1. 逐行 review 工作区未提交 diff（arxiv.py / p2_recall.py / test_arxiv_provider.py），
   确认 OR 表达式、引号短语、12 词上限、日期本地复滤逻辑无误后提交，
   commit message：`fix(arxiv): OR-based wide-recall query expression`。
2. 在 `eval_cli.py` 新增正式子命令 `auto-scholar-smoke`（替代被删的一次性脚本）：
   - 输入：`--dataset <jsonl> --offset --limit（默认 5）`
   - Gold：从数据集行 `answer_arxiv_id` 构造，**只用 arxiv_id，doi 置 None**
     （配合 B-1 的安全构造器，杜绝 mock DOI）。
   - 检索：`ArxivProvider` + `QueryPlanner`（规则版，不需要 Key），
     相邻调用间隔 `--sleep`（默认 3.1s，arXiv 礼貌限速）。
   - 输出：`artifacts/p1/auto-scholar-smoke/<n>/` 下 summary.json + 逐查询明细，
     指标复用 `metrics.py`（身份层匹配），manifest 记录 arxiv 版本日期与查询表达式。
3. 旧的两个无效产物移入 `artifacts/_invalid/`（不删除，保留审计痕迹），
   README 性质的 manifest 里注明失效原因。

**验收办法**

```bash
python -m spar_solution.src.spar_baseline.eval_cli auto-scholar-smoke \
  --dataset <AutoScholarQuery_test.jsonl 路径> --limit 5 \
  --output spar_solution/artifacts/p1/auto-scholar-smoke/run-001
# 预期：5 条查询绝大多数 returned_count>0（不再全 0）；summary.json 含查询表达式与延迟；
# 成本上限：5 次 arXiv API 调用 + ~16 秒 sleep，无其他网络访问。
```

**待办事项**

- [ ] review 并提交工作区三个未提交文件
- [ ] `eval_cli.py` 新增 `auto-scholar-smoke` 子命令（含安全 Gold 构造）
- [ ] 新增对应单测（mock transport，不联网）
- [ ] 归档旧无效产物到 `artifacts/_invalid/`
- [ ] 联网跑一次 5 条真实 smoke 并核对结果（需用户允许联网，成本见上）

---

## 3. 批次 B：协议与一致性修复（A 批全部验收后进行）

### B-1 Gold/fixture 共享 mock DOI 防御

**存在的问题**

- `mock_pipeline.py:14-15`：`_paper()` 每次都带同一个 `doi:10.1234/mock.paper`。
- `experiment.py:203-218` `_gold_paper_doc` 只做 `identifiers.update(有值的键)`，
  **不会清掉**模板里的 mock DOI。Gold 条目一旦没有 DOI（例如只有 arXiv ID），
  多篇不同论文共享同一 DOI → `deduplicate_papers` 按 DOI 判 matched → 误合并。
- 已发生过的实锤：smoke-5 中 qid test_2 的 4 篇 Gold 被并为 1 篇（fn=1）。
  WiFi fixture 因 Gold 全有真 DOI 而幸免——是运气不是设计。

**为什么需要这么做**

Gold 被误合并直接使 FN 失真、F1 不可信。这是评测正确性问题，不是风格问题。

**怎么改动（设计）**

1. 新增安全构造器 `experiment.py::gold_paper_doc(item)`：从空 identifiers 模板起步，
   只填 Gold 真实拥有的标识（doi/arxiv_id/openalex_id/...），缺的键一律 None；
   `_gold_paper_doc` 改为调用它（保留旧名做别名，避免破坏导入）。
2. 同样原则适用于 `p2_pipeline.py::run_p2_fixture`（已手工覆盖 DOI，风险低，加注释说明）。
3. 新增测试：构造两条无 DOI、有不同 arxiv_id 的 Gold → `deduplicate_papers`
   必须保留 2 条；两条无任何标识的 Gold → 保留 2 条（ambiguous 不合并）。

**验收办法**

```bash
python -m unittest discover -s spar_solution/tests -q   # 新增 >=2 用例
python -m spar_solution.src.spar_baseline.eval_cli wifi-fixture --output <临时目录>
# fixture 指标与既有基线一致（Gold 全有 DOI，行为不应变化）
```

**待办事项**

- [ ] 实现 `gold_paper_doc` 安全构造器并替换 `_gold_paper_doc` 内部实现
- [ ] 新增误合并回归测试（≥2 用例）
- [ ] 重跑 wifi-fixture 确认指标不变

---

### B-2 OpenAlex 支持年份过滤参数

**存在的问题**

- `p2_recall.py::_call_search` 会把计划里的 `start_year/end_year` 传给所有 Provider；
  arXiv 接收并在服务端+本地双过滤（`arxiv.py:147-209`），而
  `openalex_provider.py:235` 的 `search(..., **_)` 把它们**静默吞掉**。
- 后果：同一份 QueryPlan，arXiv 按年份过滤、OpenAlex 不过滤，跨源结果口径不一致；
  依赖 ConstraintGate 兜底时又会被 A-1 修掉的格式 bug 二次伤害。

**为什么需要这么做**

口径不一致会让消融实验（E=迭代 等）的对照失真；这也是"效率分"的一部分——
服务端过滤比取回再丢省 API 流量。

**怎么改动（设计）**

1. `openalex_provider.py::search` 显式接收 `start_year: int|None, end_year: int|None`，
   映射为 OpenAlex 过滤语法：`filter=publication_year:2018-2021`
   （单边界则 `publication_year:>2018` / `:<2021`；与未来其他 filter 用逗号合并）。
2. 参数校验复用 arXiv 的年份规则（1900–2200，start<=end）。
3. mock transport 单测：断言请求 URL 含正确 filter；无年份时不带 filter。

**验收办法**

```bash
python -m unittest discover -s spar_solution/tests -q   # 新增 >=2 用例
```

**待办事项**

- [ ] `search()` 新增年份参数与 filter 拼装
- [ ] URL 断言用例（有/无年份、单边界）
- [ ] 全量测试通过

---

### B-3 管线跳过已评分论文

**存在的问题**

- `p2_pipeline.py:145-163`：每轮对 `seen` **全量**重新做 gate → evidence → score。
  A-2 修好后迭代真正会跑第 2 轮，于是第 0 轮已评分的论文会被再评一遍：
  verdicts/evidence artifact 重复翻倍；将来接 LLM（C-2）后等于 Token 和费用翻倍。

**为什么需要这么做**

成本（赛题效率分 20%）+ artifact 可审计性（同一论文多条 verdict 会干扰统计口径）。

**怎么改动（设计）**

1. 管线维护 `scored_ids: set[str]`；每轮只对 `seen` 中 `paper_id` 未评分过的论文
   执行 gate/evidence/score 并写 artifact，已评分的跳过。
2. 例外：若未来计划硬约束变化需重评，留一个 `rescore_if_plan_changed=False` 参数占位，
   本轮默认不实现重评逻辑（当前两轮共用同一约束集）。
3. 测试：citation 开启 + 2 轮迭代 fixture 中，`verdicts.json` 每个论文恰好 1 条记录。

**验收办法**

```bash
python -m unittest discover -s spar_solution/tests -q
python -m spar_solution.src.spar_baseline.p2_cli fixture --output <临时目录>
# verdicts.json 无重复 paper_id
```

**待办事项**

- [ ] `P2Pipeline.run` 增加 scored_ids 跳过逻辑
- [ ] 无重复 verdict 测试
- [ ] 全量测试通过

---

### B-4 统一 F1 / 身份匹配实现

**存在的问题**

全仓有三处独立的匹配/指标实现，口径不一：

1. `metrics.py`：走 `identity.match_papers`，DOI/arXiv/稳定 ID/标题+年份+首作者，权威实现。
2. `p2_metrics.py:17-24`：**裸 paper_id 字符串集合匹配**。Provider 的 paper_id 是
   `arxiv:2301.12345` 这种带前缀形式，Gold 若是裸 `2301.12345` 就全部匹配失败，
   P2 的指标会系统性偏低。
3. `autoscholar_baseline.py:44-52`：arXiv ID 精确匹配（对 AutoScholar 数据集本身是合理的，
   保留，但要在 docstring 标注它与 metrics.py 的关系）。

**为什么需要这么做**

三套口径 = 消融实验里 A 模式和 P2 模式的数字不可比。指标不可信会直接污染所有后续决策。

**怎么改动（设计）**

1. `p2_metrics.evaluate_p2_run` 改为复用 `metrics.evaluate_at_k`：
   `gold_ids` 参数兼容三种输入——PaperDoc 记录列表（直接用）、
   裸字符串（自动识别 `^\d{4}\.\d{4,5}$` 为 arxiv_id、`W\d+$` 为 openalex_id、
   `10\..+` 为 doi，包成最小 Gold 记录）、已是映射的保持原样。
2. 返回结构保留原字段名（`by_cutoff` 等），内部 tp/fp/fn 来自统一实现。
3. 测试：预测 `paper_id="arxiv:2301.12345"` vs gold `"2301.12345"` → tp=1
   （当前实现是 0）；gold 带 DOI vs 预测带同 DOI → tp=1。

**验收办法**

```bash
python -m unittest discover -s spar_solution/tests -q   # 新增 >=2 用例
```

**待办事项**

- [ ] `evaluate_p2_run` 切换到 `metrics.evaluate_at_k`，保留旧签名兼容
- [ ] 裸字符串 Gold 的类型识别辅助函数 + 测试
- [ ] 全量测试通过

---

### B-5 文档同步

**存在的问题**

- `P1_ACCEPTANCE.md` 写"60 passed"，实际现已 98 项。
- `PLAN.md` §5.3 承诺本地评测含 MRR 与 Token，但 `metrics.py` 无 MRR
  （`p2_metrics.py` 有）、Token 计量全仓没有（C-3 补）。

**怎么改动（设计）**

1. `P1_ACCEPTANCE.md` 增补一段"2026-08-24 复核"：测试数 98；MRR 现状说明；
   Recall=0 两个产物已归档失效（引用 A-4）。
2. `PLAN.md` §5.3 加一句现状注记：MRR 由 `p2_metrics` 提供、Token 计量在 C-3 落地。

**验收办法**：文档 diff 人工 review。

**待办事项**

- [ ] 两份文档补注记

---

## 4. 批次 C：能力补齐（B 批验收后进行；完成后才具备谈 P3 的资格）

### C-1 OpenAlex 引用关系（relations）实现

**存在的问题**

- 四个 Provider 的 `relations()` 全部 raise unsupported（`arxiv.py:226`、
  `openalex_provider.py:332`、`bohrium.py:270`、`local_library.py:102`）。
- P2 的引用扩展（`p2_citation.py`）设计完好、种子门控正确，但**没有任何真实数据源**，
  只能跑 fixture。P2 验收标准"引用扩展实际发生并能回溯到父论文"无法真实验证。
- 讽刺点：`openalex_provider._to_paper_doc` 已经在解析 `referenced_works` 字段，
  只差查询端点没有实现。

**为什么需要这么做**

引用网络探索是赛题题面明确点名的能力（"引文网络探索"），也是 SPAR 相对普通检索的
核心增益来源。不做这个，P2 的 F 项消融（引用扩展）永远无法测。

**怎么改动（设计）**

实现 `OpenAlexProvider.relations(paper_id, relation, page_size)`，纯 stdlib，transport 可注入：

1. **paper_id 归一**：接受 `W123`、`openalex:W123`；若是 `doi:10.x/...` 形式，先调
   `/works/https://doi.org/10.x/...` 解析成 W-id（多 1 次调用，计入预算）。
2. **citations（被引）**：`GET /works?filter=cites:W123&per_page=page_size`，
   结果走现成 `_to_paper_doc`，`provenance` 加 `relation_source=openalex/citations`。
3. **references（参考文献）**：先 `GET /works/W123?select=id,referenced_works`
   拿 ID 列表（截断前 page_size 个），再用
   `GET /works?filter=openalex_id:W1|W2|...&per_page=page_size` 批量取完整记录
   （OpenAlex filter 的 `|` OR 语法，单次最多 50 个 ID，超出截断并在 warnings 记录）。
4. **relation="all"**：先 citations 后 references，调用数 ≤3 次/种子。
5. 每条记录附 `relation_type`，`CitationExpander` 现有的 `_relation_type/_nested_paper`
   即可消费，无需改 p2_citation。
6. 调用次数与耗时记入 `ProviderResult.provenance`（与效率分挂钩）。
7. 单测全部用 mock transport：构造 W-id 解析、citations 列表、references 两段式
   三组 fixture JSON；错误路径（404/超时/畸形响应 → ProviderError 对应错误码）。

**验收办法**

```bash
python -m unittest discover -s spar_solution/tests -q      # 新增 >=5 用例，全部离线
# 可选真实 smoke（需用户允许联网，成本：1 个种子 × ≤3 次 OpenAlex 调用）：
# 断言 edges>0、子 PaperDoc 通过 validate_paper_doc、provenance 可回溯父论文
```

**待办事项**

- [ ] `relations()` 实现（citations/references/all + id 归一）
- [ ] mock transport 单测 ≥5 用例（含错误路径）
- [ ] P2 fixture 增加"openalex relations"端到端用例（fixture transport）
- [ ] （可选，需允许）1 次真实 OpenAlex relations smoke

---

### C-2 DeepSeek 接入主管线（查询规划 + 相关性判断）

**存在的问题**

- DeepSeek 目前只在 `autoscholar_baseline.py`（评测对照脚本）里被调用过，
  P2 主管线没有任何 LLM：`Scorer._relevance`（`p2_scoring.py:158-170`）是
  词元重叠比率，`QueryPlanner._deterministic` 是英文硬编码词表规则。
- 词法相关性对"target networks for Deep Q-learning"这类语义查询排序能力弱；
  规则 planner 的 gap 子查询是 `"{topic} methods techniques algorithms"` 模板拼接，
  泛化性差（专家评分中"算法泛化性"占 10%）。
- 赛题核心要求就是"基于大模型的自主搜索策略迭代"。

**为什么需要这么做**

没有 LLM 的 P2 只是检索管道；接入 LLM 才是赛题的主体。同时必须保证：
LLM 失败不得伪装成"论文不相关"（协议红线），所以降级路径和计量必须一起做。

**怎么改动（设计）**

新增 `src/spar_baseline/llm_client.py`（stdlib urllib，模式照抄
`autoscholar_baseline._deepseek_query` 的安全实践：超时、脱敏、显式错误）：

1. **DeepSeekClient**
   - `__init__(api_key, base_url, model="deepseek-chat", timeout=45)`；
     `from_env()` 读 `DEEPSEEK_API_KEY`（经 `config.py` loader），无 Key → 实例为
     `configured=False`，所有调用直接抛 `LLMUnavailable`（调用方降级，不算错误）。
   - `chat_json(messages, max_tokens) -> tuple[dict, usage]`：
     解析 JSON（容忍 ```json 围栏），usage 提取 prompt/completion tokens 并累计
     （喂给 C-3）。HTTP 429/5xx → 抛 `LLMError(retryable=True)`，重试 1 次后放弃。
2. **注入点 1：查询规划**（复用现成接口 `QueryPlanner.from_llm_json`）：
   - `make_llm_planner(client) -> Callable`：prompt 给出 query_plan.v1 的字段骨架，
     要求只输出 JSON；返回的 dict 由 `from_llm_json` 强校验，
     校验失败 → 抛异常 → 外层捕获 → 回退 `_deterministic` 并在计划里记
     `planner_source: "llm_fallback_rules"`（新增可选字段，向后兼容）。
   - P2Pipeline 构造时可选传入 `planner=`，默认仍是规则版（无 Key 行为不变）。
3. **注入点 2：相关性判断**（新增 `p2_judge.py::LlmRelevanceJudge`）：
   - `judge(papers: list[PaperDoc], plan) -> dict[paper_id, {"relevance": float, "reason_code": str}]`。
   - **批量**：一次调用判 ≤15 篇，每篇只送 title + 摘要截断 600 字符 +
     编号；输出 `{"items":[{"n":1,"relevance":0.0-1.0,"reason":"..."}]}`。
   - 管线集成（`p2_pipeline.py`）：词法评分后取**未 excluded 的 Top-M（默认 20）**
     送 judge；结果经 `Scorer.score(component_overrides=...)` 注入
     （该参数 `p2_scoring.py:106` 已预留，不用改 Scorer）。
   - **降级红线**：judge 抛错/超时/返回不合法 → 该批论文保留词法分，
     verdict.warnings 加 `llm_judge_unavailable`；**绝不允许**因 LLM 失败把
     relevance 写 0 或把论文标记不相关。
   - 同一 run 内按 `(query_id, paper_id)` 缓存，避免重复判。
4. 成本上限写入 QueryPlan.budget：`max_llm_calls`（默认 10/查询），
   超限即只用词法分并记 warning。
5. 单测全部用 FakeClient（不联网）：合法规划/非法 JSON 回退、批量 relevance
   覆盖词法分、FakeClient 抛错时 warnings 且词法分保留、token 计数累计、
   无 Key 时管线行为与现状完全一致。

**验收办法**

```bash
python -m unittest discover -s spar_solution/tests -q       # 新增 >=8 用例，全部离线
# 可选真实联调（需用户允许并提供 Key，成本上限：1 个查询 ×
#   1 次规划(<=1k token) + 2 批相关性(每批 <=3k token) ≈ <1 万 token，约几分钱）：
#   跑一个真实问题，manifest 中 planner_source=llm，verdicts 含 llm relevance，
#   无任何 relevance 因 LLM 失败被写成 0
```

**待办事项**

- [ ] `llm_client.py`（DeepSeekClient + LLMError/LLMUnavailable + token 累计）
- [ ] `make_llm_planner` + planner_source 字段 + 回退测试
- [ ] `p2_judge.py` 批量判断 + 缓存 + 降级红线测试
- [ ] P2Pipeline 可选注入（无 Key 行为不变）测试
- [ ] budget.max_llm_calls 限流测试
- [ ] （可选，需允许+Key）1 次真实联调

---

### C-3 Token / 成本计量入 manifest

**存在的问题**

- 赛题效率分 20% = API 调用次数 + Token 消耗 + 端到端延时。当前只有 API 次数
  （`stats.api_calls`）和逐调用延迟，**Token 一处都没有记**；LLM 接入（C-2）后
  没有计量就无法证明成本效率，也无法做"结构化通信省 Token"的 P3 对比。

**怎么改动（设计）**

1. `P2Run` 增加 `cost` 字段，`run_manifest.json` 固定输出：
   ```json
   "cost": {
     "provider_calls": {"arxiv": 6, "openalex": 3},
     "llm_calls": 3,
     "prompt_tokens": 5412,
     "completion_tokens": 860,
     "total_tokens": 6272,
     "llm_failures": 0,
     "wall_ms": 21450,
     "per_stage_ms": {"recall": 8100, "citation": 6200, "judge": 5400, "evidence": 1750}
   }
   ```
2. 数据来源：LLM tokens 来自 C-2 client 的累计器；provider_calls/延迟从现有
   recall/citation stats 汇总；wall_ms 管线整体计时。
3. `p2_metrics.evaluate_p2_run` 的 `stats` 增补 `total_tokens/llm_calls`（若有）。
4. 单测：FakeClient 跑 fixture → manifest 数值精确等于注入的假 usage 之和。

**验收办法**

```bash
python -m unittest discover -s spar_solution/tests -q    # 新增 >=2 用例
# fixture 运行产物 run_manifest.json 含完整 cost 块
```

**待办事项**

- [ ] cost 汇总与 manifest 落盘
- [ ] p2_metrics 透传 tokens
- [ ] 数值精确性测试

---

### C-4 FINAL_SELECTION 结构化输出模块

**存在的问题**

- 赛题"回复结果结构化"占 10%（列表、关系图等）。`AGENT_PROTOCOL.md` 定义了
  `FINAL_SELECTION` 消息类型，但**没有任何代码**生成它；当前产物只有内部
  PaperDoc/verdicts，没有面向提交的最终答案格式。

**怎么改动（设计）**

新增 `src/spar_baseline/final_output.py`：

1. `build_final_selection(run, top_k=20) -> dict`，schema `spar.final.v1`：
   - `results[]`：rank、paper_id、title、year、venue、doi/arxiv_id、landing_url、
     `relevance_zone`（`high`: final≥0.6 / `partial`: 0.3–0.6 / `reserve`: 其余；
     对应赛题"区分高度相关和部分相关"）、final_score、component_scores、
     evidence_refs、reason_codes；
   - `relation_graph`：nodes（入选论文）+ edges（来自 run.citations 的
     parent/child/relation_type，child 不在 top_k 也保留为节点但标记
     `outside_topk=true`）；
   - `summary`：各区数量、citation 边数；
   - `degraded`：沿用 manifest 判定；
   - `cost`：引用 C-3。
2. `P2Pipeline.write_artifacts` 自动写 `final_selection.json`；
   `p2_cli` 增加 `finalize --input <artifact_dir>` 可从已有产物离线重生成
   （replay 后调用，便于改格式不重跑）。
3. 单测：fixture run → final_selection 通过校验；zone 边界值正确；
   citation on 时 edges>0 且父论文可回溯；excluded 论文不得出现在 results。

**验收办法**

```bash
python -m unittest discover -s spar_solution/tests -q
python -m spar_solution.src.spar_baseline.p2_cli fixture --output <临时目录>
# 产物含 final_selection.json，结构符合 spar.final.v1
```

**待办事项**

- [ ] `final_output.py` + schema 校验函数
- [ ] write_artifacts 集成 + p2_cli finalize 子命令
- [ ] zone/edges/excluded 测试 ≥4 用例

---

## 5. 执行顺序与依赖

```text
批次 A（P0）：A-1 → A-2 → A-3 →（提交+联网 smoke）A-4
批次 B（协议）：B-1 → B-2 → B-3(依赖A-2) → B-4 → B-5     顺序可并行，B-3 必须在 A-2 后
批次 C（能力）：C-1(依赖A-3) → C-2 → C-3(依赖C-2) → C-4(依赖B-3)
每个批次收尾：全量 unittest + 重生成 fixture artifact + git commit（一项一提交）
```

## 6. 总验收清单（本轮完成的定义）

- [ ] `python -m unittest discover -s spar_solution/tests -q` 全绿，用例数 ≥ 98+30
- [ ] 复现脚本：2019 年论文对 "between 2018 and 2021" 约束为 pass
- [ ] self-test stop.json 出现 continue→stop 两轮记录，iterations=2
- [ ] OpenAlex 空结果为 ok=True 空 ProviderResult
- [ ] arXiv OR 修复已提交；auto-scholar-smoke 重跑不再全 0（联网项）
- [ ] verdicts.json 无重复 paper_id
- [ ] p2_metrics 能匹配 `arxiv:xxx` vs 裸 ID 的 Gold
- [ ] openalex relations 离线测试通过（可选：1 次真实 smoke）
- [ ] 无 DEEPSEEK_API_KEY 时全流程行为与现状一致；有 Key 时 manifest 含 token 数（可选联网项）
- [ ] fixture 产物含 final_selection.json（spar.final.v1）
- [ ] P1_ACCEPTANCE.md / PLAN.md 已同步现状

## 7. 风险与回退

| 风险 | 缓解 |
|---|---|
| A-2 改停止语义后 fixture 指标变化 | 指标变化是预期行为（迭代真的跑了）；重生成 artifact 并在 commit message 说明 |
| A-4/C 真实联网调用 | 全部有硬上限（A-4：5 次 arXiv；C-1：3 次；C-2：<1 万 token），执行前再次向用户确认 |
| LLM 输出不稳定破坏管线 | 双保险：JSON 强校验 + 失败回退规则版；无 Key 时零行为变化 |
| 重构 p2_metrics 影响旧 artifact 回放 | 保留旧字段名；replay 对旧产物做一次兼容性测试 |
| 改动互相踩踏 | 分支 improve/p0-p1-batch + 每项独立提交，可按提交粒度回退 |

## 8. 本轮明确不做（遗留清单，防止 scope 膨胀）

1. P3 五 Agent 圆桌与消息封套（等 C 批验收 + P2 真实基线出来后按 PLAN 的三组消融再议）。
2. 第二个 LLM 模型 / 多模型路由。
3. BM25/向量检索、embedding 重排（词法粗排 + LLM 精判已覆盖当前需求）。
4. `_norm_text` 与 `normalize_title` 正则统一、DOI 冲突判 ambiguous 的校准、
   O(n²) 去重优化、search_service 真并行（均为低风险清理，下轮处理）。
5. 真实本地论文库接入（等用户提供数据源）。

## 9. 2026-08-24 实施状态

本批次已完成 A/B/C 的代码修复与测试补齐：

- 全量 `python -m unittest discover -s spar_solution/tests -q`：166 项通过。
- `compileall`：通过；`git diff --check`：无错误（仅换行符提示）。
- A-4 新 smoke 已产生非零结果；旧全零/错误 Gold 产物已移动到 `artifacts/_invalid/`，禁止用于效果结论。
- OpenAlex `relations` 真实接口、调用预算和年份边界已纳入代码；DeepSeek 非法判断会触发规则回退，引用子论文会再次判断，判断依据写入 verdict/final artifact。
- 新鲜独立验收仍需在本批次最终代码稳定后执行；真实 DeepSeek、Bohrium 和 LocalLibrary live 状态以 `.env.local` 配置为准，缺失配置只记录 `config_missing`。

状态口径：fixture/离线协议已验证；OpenAlex live smoke 已验证；DeepSeek/Bohrium live 本轮不因本地缺失配置而伪造成功；没有 provisional Gold 之外的效果提升结论。
