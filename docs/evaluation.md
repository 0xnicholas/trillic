# 第一阶段:评测基线

评测基线是 Trillic 的**验收尺**:没有它,"微调后优于公开 checkpoint"无法
证明,后续所有迭代(query-aware、更激进档位)都没有度量基础。它在任何
**验收声称**之前完成;plumbing 性质的 pilot 训练(打通管线、不做质量
声称)不受此门约束(2026-08-30 修订,见 `roadmap.md`)。

## Harness

以旧项目 `compression/eval/run_eval.py` 为起点移植(见 references.md)。
流程(每个 golden prompt):

1. 经 refine sidecar 的 `POST /refine` 压缩/精炼;
2. 用**原文**和**精炼后**的 prompt 各跑一次固定下游任务(经网关,成本进
   内部账本);
3. LLM judge 按 golden 标注的 key_points 给两个答案打分;
4. 输出质量 delta vs 总 token 成本(压缩 + 下游)。

golden set 格式(旧仓 `eval/golden.jsonl` 先例):

```json
{"id": "support-001", "prompt": "...", "key_points": ["必须存活的事实", "..."]}
```

**规模(pilot-ramp,2026-08-30 定案)**:旧仓只有 3 条。先每类 10 条
pilot 验证 harness 端到端;用 pilot 实测 judge 方差,按"能分出效应量"
倒推每类样本数(预期 30–50 区间)后补齐,对齐 data-strategy.md 的三类
负载。

**冻结纪律**:golden set 在基线数字产出时冻结。之后只允许"扩充 +
重跑基线",不允许改题——考卷跟着学生改,验收作废。judge 模型、下游
作答模型、rubric 全部落版本进报告。

## 指标

| 指标 | 口径 | 说明 |
|---|---|---|
| 压缩率 | tiktoken(计费口径)+ WordPiece(模型原生)双计数 | 扫档 0.1–0.5,旧 benchmark 只扫到 0.3 |
| fact recall | 数字型事实逐字存活率 | 现网短板:公开 checkpoint 默认档 0.776 |
| token F0.5 | BERTScore 式逐 token 对齐 | refine 运行时 guardrail 同款实现 |
| 下游任务质量 | LLM judge 按 key_points 打分 | 任务级,本项目的主验收指标 |

口径注记(issue #6 落地):harness 的 fact recall 与运行时 guardrail
共用同一正则(`_FACT_PATTERN` 逐字复用,两侧必须同漂移);token F0.5
与 guardrail 同 F0.5 代数(recall/precision 方向、β²=0.25),但相似度
用 token 恒等对齐替代 embedding 余弦(harness 不带模型运行时)——对
抽取式压缩(严格子序列)恒等对齐是精确的,harness F0.5 是 embedding
口径的下界,方向与运行时阈值一致。压缩延迟 p95 用线性插值
percentile(旧 benchmark 口径)。模型原生计数器参数化:`word` /
`wordpiece` / `sentencepiece` 三 flavor(`[metrics]` 配置),换基底只改
配置不改代码。

## 验收标准

微调模型 vs 公开 checkpoint(`llmlingua-2-bert-base-multilingual-cased-meetingbank`)
同档 aggressiveness 对比;显著性用 paired bootstrap(按 prompt 重采样)
报 95% CI,判据 = 任务级质量 delta 的 CI 下界不为负:

- **go 标准**:下游任务质量不掉(统计显著),且压缩率或 fact recall 有一项
  显著提升;
- 仅在代理指标(fact recall / F0.5)上赢不算赢——TTC 调研的教训:代理指标
  与任务质量会背离。

## 基线范围与延迟记录(2026-08-30 增补)

基线覆盖**两个公开 checkpoint**:现网 mBERT-base
(`llmlingua-2-bert-base-multilingual-cased-meetingbank`)+ 公开
XLM-R-large 变体(零训练成本,只加推理)。同时记录各 checkpoint 的
**压缩延迟**(p95)——这是 roadmap 阶段 1 的产出之一,为"XLM-R-large
微调"后手牌提供激活判据(质量显著更高且延迟可接受才激活)。

## 对照数字(公开 checkpoint,旧仓 2026-07-28 实测)

| 档位 | 压缩率(tiktoken) | fact recall | token F0.5 |
|---|---|---|---|
| 0.1 | 19.8% | — | — |
| 0.2(默认) | 27.6% | 0.776 | 0.821 |
| 0.3 | 34.3% | — | — |

语料:60 条(RAG 20 / system prompt 20 / 多轮对话 20)。完整报告见
references.md 的 benchmark 链接。
