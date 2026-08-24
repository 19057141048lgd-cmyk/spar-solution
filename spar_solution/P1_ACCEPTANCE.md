# P1 验收报告

日期：2026-08-24

## 结论

P1 的可执行检索评测协议已实现并通过 fixture 验收；没有进入 P2/P3。四组模式和四个 WiFi 查询 artifact 已生成，但当前 LocalLibrary 是 `mock`，Gold 是 `provisional` 人工标注集，因此不能证明真实效果提升，也不能称为官方基准。

## 已验证

- `PaperDoc v1` 校验、数组类型约束、来源合并和错误事件。
- 身份匹配优先级：DOI → arXiv ID → S2/OpenAlex/本地稳定 ID → 标题+年份+第一作者；冲突/字段不足返回 `ambiguous`，不强算 TP。
- Gold `spar.gold.v1`：四个 WiFi 查询、人工依据、标注时间、`annotation_status=provisional`。
- 固定 TP/FP/FN、Precision/Recall/F1@10/@20、Macro-F1、Micro-F1，以及延迟、API 次数、来源返回数、去重数、source_errors。
- Provider/API/解析失败仅进入审计字段，不转换成论文低相关分。
- `A_arxiv`、`B_local`、`C_fusion`、`D_reranked` 四模式先去重后计分；C/D 回归门槛已实现。

## 仅 Mock 验证

命令：

```powershell
python -m spar_solution.src.spar_baseline.eval_cli wifi-fixture --output spar_solution/artifacts/p1/wifi-heart-rate
```

结果目录：`spar_solution/artifacts/p1/wifi-heart-rate/`。根目录和每个 `wifi_hr_q1` 至 `wifi_hr_q4` 子目录均包含 `query.json`、`gold.json`、四种 `results_*.json`、`metrics.json`、`errors.json`、`run_manifest.json`。fixture 结果的 `fusion_regression=false`、`rerank_regression=false`，但这只说明协议运行正常。

汇总指标（fixture/provisional）：

| 模式 | Recall@10 | F1@10 | Recall@20 | F1@20 |
|---|---:|---:|---:|---:|
| A arXiv | 0.667 | 0.800 | 0.667 | 0.800 |
| B LocalLibrary | 0.667 | 0.800 | 0.667 | 0.800 |
| C fusion | 1.000 | 1.000 | 1.000 | 1.000 |
| D reranked | 1.000 | 1.000 | 1.000 | 1.000 |

这些数值来自 mock fixture，不能外推到真实论文库。

## 真实 API 已验证

OpenAlex WiFi live smoke 已成功返回结构化论文（本次复核返回 3 条，API total=10865）；历史结果保存于 `artifacts/p1/wifi-heart-rate-live-local-env.json`。Bohrium 因本地 `BOHR_ACCESS_KEY` 未配置而显式返回 `config`/`config_missing`，未伪装为空结果。此次 A-D 对照实验没有使用真实本地库，也没有把 OpenAlex 结果冒充 arXiv 或 Gold。

## Gold 缺失与效果边界

当前 Gold 不是官方或穷尽集合，状态为 `provisional`。因此报告中的 A/B/C/D 指标不能证明效果提升；只有在真实本地库配置和经过独立复核的 Gold 到位后，才能进行 live 对照结论。

## 验证命令

- `python -m unittest discover -s spar_solution/tests -v`
- `python -m compileall -q spar_solution/src spar_solution/tests`
- 历史快照：60 passed（已过期，仅保留审计记录）。当前本轮全量测试结果以改进验收报告为准。
- arXiv live smoke：四个 WiFi 查询均按真实 API 请求执行；返回数量受 arXiv 语料覆盖影响，未写入 A-D fixture Gold。
- 目标范围 secret scan：不得出现真实 API key、Bearer 值或 `.env.local` 内容。
- 根及查询 artifact 文件清单检查。

## 当前阻塞

1. LocalLibrary 没有真实路径或接口，只能保持 `mock`/`unavailable`。
2. Gold 仍为 provisional，不能宣称 A/B/C/D 的真实优劣。
3. P1 的 D 仅为确定性元数据排序；语义重排、查询迭代、引用扩展属于 P2，未实现。

## 2026-08-24 改进复核

- B 批改动完成时执行全量测试：131 项通过；该数字是本次复核快照，不替代后续最终验收记录。
- P2 的 MRR 已由 `p2_metrics.py` 提供，并与 Precision/Recall/F1 共用 `metrics.py` 的论文身份匹配规则。
- Token/LLM 成本计量已落地到 P2 manifest；P1 fixture 不产生 LLM token，因此 P1 仍不宣称真实效率提升。
- 旧版 arXiv 全词 AND 查询生成的 Recall=0 产物不作为有效基线；以 A-4 重建的可复现 smoke 为准。
