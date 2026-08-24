# 本地工具与全文转换说明

## PDF → Markdown 转换工具（老师提供，2026-08-25）

位置：`tools/`（不入库，已被 .gitignore 忽略；原始压缩包在微信文件目录）。

- 用途：把下载的论文 PDF 转成 Markdown，供 P2 证据层读取全文（赛题要求
  "基于标题、摘要及可用全文信息"评估）。
- 机制：本地 `pymupdf`/`pdfplumber` 解析 + **PaddleOCR-VL-1.5 云端 API**
  （`https://paddleocr.aistudio-app.com`）做版面/公式识别。
- 使用前必须配置：`tools/api_config.py` 里的 `API_TOKEN`（或运行 GUI 右上角
  ⚙️ 设置），token 需在百度 AI Studio 申请。**不得把 token 写入任何入库文件。**
- 依赖：见 `tools/requirements.txt`（pymupdf、pdfplumber、Pillow 等）。
- 接入点规划：下载 OA PDF（PaperDoc 的 `access.pdf_url`）→ 调用
  `tools/pdf_converter.py` 生成 md → 存为 EvidenceItem（`evidence_status=fulltext`，
  `content_ref` 指向 md 文件）→ EvidenceJudge 按 chunk 引用。此链路在检索
  效果（F1）稳定后再接。

## 外部参考仓库

位置：`repos/`（不入库）。7 个参考项目的吸收对账与可抄参数见
`IMPROVEMENT_PLAN.md` 与会话审查记录；SPAR 搜索树超参（max_depth=2、
每层剪枝 2 条查询、DOCS_TO_EXPAND=40、高相关阈值 0.75、分桶排序公式）
已落实到 `src/spar_baseline/search_tree.py`。
