# 资料索引

## 方法基础(Microsoft LLMLingua-2)

- 论文:[arXiv:2403.12968](https://arxiv.org/abs/2403.12968)(ACL 2024
  Findings)——抽取式 token 分类、GPT-4 数据蒸馏管线、2–5× 压缩接近持平的
  基准数字、作者自述局限
- 基底 checkpoint:[microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank](https://huggingface.co/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank)(Apache-2.0)
- 蒸馏数据集:[MeetingBank-LLMCompressed](https://huggingface.co/datasets/microsoft/MeetingBank-LLMCompressed)(注意:MeetingBank 是**训练集**,评测必须排除,见 data-strategy.md 硬约束)

## 对标(The Token Company / bear)

- 完整调研与口径分析:`../../tokencamp-pro/docs/research/thetokencompany-compression.md`
  ——产品事实、bear 机制(同族抽取式)、双方数字口径差异、路线建议
- 要点:同技术族;对方全部优势数字为厂商自述小样本;其分场景分档指导与
  `<ttc_safe>` 保护标签是产品化参考

## 旧项目(0xnicholas/tokencamp,设计源头)

- [2026-06-09 抽取式压缩设计](https://github.com/0xnicholas/tokencamp/blob/main/docs/superpowers/specs/2026-06-09-extractive-prompt-compression-design.md)——对标记录(bear 系列、LLMLingua-2)
- [2026-07-18 refine 设计](https://github.com/0xnicholas/tokencamp/blob/main/docs/superpowers/specs/2026-07-18-refine-prompt-refinement-design.md)——两阶段管线、guardrail 选型实验(mean-pool 否决的实测数据)
- [压缩 benchmark 报告(2026-07-28)](https://github.com/0xnicholas/tokencamp/blob/main/docs/benchmarks/compression-llmlingua2-tokens.md)——公开 checkpoint 的实测基线,本项目验收的对照组
- [agent-notes/refine](https://github.com/0xnicholas/tokencamp/blob/main/docs/agent-notes/refine.md)
- 评测 harness 起点:`compression/eval/run_eval.py`、`build_corpus.py`、`conversation_gen.py`(同仓)

## 运行时宿主(tokencamp-pro)

- `../../tokencamp-pro/sidecars/refine/compressor.py`——模型加载、token 打分、
  分块、阈值过滤;接入点 = `MODEL_ID`
- `../../tokencamp-pro/sidecars/refine/refiner.py`——rewrite 阶段与 guardrails;
  `_FACT_PATTERN`(数字事实抽取)在两侧复用
- `../../tokencamp-pro/crates/gateway/src/pipeline/transform.rs`——网关管道集成
  (本项目的模型最终在这里影响线上请求)
