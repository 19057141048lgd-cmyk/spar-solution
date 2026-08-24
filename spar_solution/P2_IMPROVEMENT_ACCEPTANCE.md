# P2 改进验收报告

日期：2026-08-24
分支：`improve/p0-p1-batch`

## 角色隔离

1. **代码审查**：`integration_code_review`，只读检查老师方案、代码、测试与 artifact，未修改文件。
2. **实施**：A/B/C 实施 Agent 分模块修改；主 Agent 只负责集成、冲突处理和补丁。
3. **最终验收**：`fresh_final_acceptance`，未参与本轮开发，仅读取规范并独立运行检查；验收 Agent 不修改代码。

## 已验证

- 最终全量回归：167 项 `unittest` 全部通过，`compileall` 通过，现行 P2 fixture/replay 通过。

- A-1：`YYYY-YYYY`、`>=YYYY`、`<=YYYY` 时间约束端到端一致，区间中间年份不再误杀。
- A-2：引用深度不再终止总查询；两轮流程可达，`MAX_ITERATION` 在最后一轮有明确记录。
- A-3：OpenAlex 合法空结果返回成功空集并带 `no_results`，不计 Provider 故障。
- A-4：arXiv 使用短语/OR 宽召回；旧全零基线已归档到 `artifacts/_invalid/`，新 smoke 不再全零。
- B-1/B-4：Gold 身份不再继承 mock DOI；P2 指标复用统一 DOI/arXiv/稳定 ID匹配。
- B-2/B-3：OpenAlex 年份过滤生效；已评分论文和已扩展引用种子不会重复评分。
- C-1：OpenAlex 支持 `citations`、`references`、`all`，父子边、关系类型和实际 HTTP 调用数均记录。
- C-2/C-3：DeepSeek 可选接入 QueryPlan 与候选判断；无 Key 规则回退；非法输出不覆盖词法分；token/调用/耗时进入 manifest。
- C-4：产物含 `spar.final.v1`、分区结果、关系图、证据引用和成本字段。

## 仅 fixture / 离线验证

- 当前全量单元测试、compileall、P2 fixture/replay、citation enabled/disabled 消融属于离线验证。
- 全新只读验收 Agent 在独立临时目录重跑 citation enabled/disabled：两套均为两轮流程，最终记录 `MAX_ITERATION`；enabled 为 2 篇/1 条引用边/6 次 Provider API，disabled 为 1 篇/0 条引用边/5 次 Provider API。
- DeepSeek 的真实 HTTP 联调未执行，因为本地配置没有可用 `DEEPSEEK_API_KEY`；这不是把论文判为不相关，而是记录 `config_missing`/规则回退。
- LocalLibrary 仍没有真实路径，状态保持 `unavailable`，没有伪造真实本地库结果。

## 真实 API 已验证

- OpenAlex WiFi smoke 已返回结构化论文；OpenAlex relations 的 mock 契约与真实搜索入口均通过。
- AutoScholar arXiv smoke 已完成 5 条真实查询：5/5 请求成功，返回总记录 50，不再是旧版全零结果。该 smoke 只证明可运行和召回非零，不证明官方效果提升。
- Bohrium 本轮因本地 `BOHR_ACCESS_KEY` 未配置未执行 live；没有把配置缺失算作低相关论文。

## 效果边界与当前阻塞

- WiFi Gold 仍为 `provisional`，不能据此声称 P2 相对 P1 或四组模式有真实效果提升。
- 旧版 `live-wifi-deepseek`、`self-test-final`、`self-test-optimized` 已移入 `artifacts/_invalid/`，不应作为当前 schema 的验收结果。
- 没有真实 LocalLibrary、DeepSeek 和 Bohrium 三者同时配置前，无法完成完整生产路径的效果/成本对照。
- P3 既有代码未在本轮修改；五 Agent、多模型和通信消融不属于本轮验收。

## 复现命令

```powershell
python -m unittest discover -s spar_solution/tests -q
python -m compileall -q spar_solution/src spar_solution/tests
python -m spar_solution.src.spar_baseline.p2_cli replay --input spar_solution/artifacts/p2/self-test
python -m spar_solution.src.spar_baseline.p2_cli replay --input spar_solution/artifacts/p2/self-test-no-citation
```
