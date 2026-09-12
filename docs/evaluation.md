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

口径注记(issue #7 落地):任务级质量闭环在假桩下端到端可演示——三类负载
(RAG 问答 / system prompt 约束遵守 / 对话记忆)按 load_type 分派各自的下游
任务指令,经网关客户端对原文与压缩后两份 prompt 各作答一次(同一任务框
架,仅载荷不同),LLM judge 按 key_points 对两份答案独立打分(逐点 0/1,
条分 = 覆盖率);质量 delta = 压缩后分 − 原文分,按 prompt 重采样(种子化
10k 次)的 paired percentile bootstrap 给 95% CI,判据字段(CI 下界不为负)
直接落报告。judge rubric 版本化:版本号 + rubric 全文的 sha256 进报告,措辞
任何变动都会变哈希;judge 与下游作答模型钉死在配置(`[quality]` 段)并进
报告。假桩语义:stub 网关对作答请求回显 prompt,对 judge 请求按机检规则
(关键点全部有效词在答案中出现 = 1)打分——因此假桩下压缩只会降分,
管线接线的方向性可验证;真实跑(http 模式)换真模型,零代码改动。
bootstrap 的 RNG 口径:每抽一次下标消耗一次 `random.Random(seed).random()`
(跨版本稳定的唯一生成器),CI 端点用与延迟 p95 同款的线性插值 percentile;
测试用种子化字面量钉死数值,漂移即红。

## 基线数字(权威,2026-09-12 双 checkpoint 全量)

数字权威 = 仓内冻结基线报告(本节为摘要,完整数据见 `runs/` 两份 run 目录与
`docs/baselines/2026-09-12-full149-dual-checkpoint-summary.md`)。旧仓 2026-07-28
的 60 条对照数字自此仅作历史参考。

- **考卷**:149 条(RAG 49 / system prompt 50 / 多轮对话 50),内容寻址
  sha256 `fb728d6a5042…`(逐文件清单与冻结记录见
  `eval/manifests/golden-freeze.json`;冻结钉 = golden 哈希 + 仓库 commit)。
- **钉子**:作答 `deepseek/deepseek-v4-pro`,judge `zhipu/glm-5.1`,rubric v1
  (`f48af57b1a80…`),harness commit 进每份报告。
- **mBERT-base(现网公开 checkpoint)**:

| 档位 | 压缩率(计费) | fact recall | token F0.5 | 质量 delta [95% CI] | 延迟 p95 |
|---|---|---|---|---|---|
| 0.1 | 0.2203 | 0.7699 | 0.6965 | +0.0095 [-0.0140, +0.0345] | 3.2ms |
| 0.2(默认) | 0.2948 | 0.7699 | 0.6626 | +0.0009 [-0.0330, +0.0363] | 6.4ms |
| 0.3 | 0.3590 | 0.7699 | 0.6393 | -0.0151 [-0.0507, +0.0209] | 5.0ms |
| 0.4 | 0.4179 | 0.7699 | 0.6159 | -0.0670 [-0.1075, -0.0279] | 2.9ms |
| 0.5 | 0.4748 | 0.7699 | 0.5883 | -0.0729 [-0.1172, -0.0293] | 3.0ms |

- **XLM-R-large(公开变体,CPU 推理)**:同考卷同扫档,压缩更保守
  (0.1 档保留率 0.790 vs mBERT 0.780)、fact recall 全档 1.0000,但质量
  delta 与 mBERT 无统计差异(仅 0.4 档 CI 下界 > 0),压缩延迟 p95
  17.8–25.8s(≈ mBERT 的 4000×)。
- **后手牌判定**:**不激活**——质量未全面显著更高(1/5 档)、延迟不可
  接受、接口非 drop-in(`.roberta` 无 `.bert`,decisions §6)。v1 继续
  mBERT-base 基底。
- **口径注记**:judge 中途从 kimi/kimi-k2 切到 zhipu/glm-5.1(kimi 上游故障,
  owner 授权,切在冻结前;智谱 key 为 GLM Coding Plan,仅 coding 端点可用);
  两 checkpoint 同 judge 内部一致;glm 对个别答案样本触发内容审查,有界
  重答后通过(计入 content_filter_retries)。延迟为生产形态测量(含 HTTP)。

## 验收标准(不变)

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

## 历史对照(已被基线取代)

旧仓 2026-07-28 的 60 条对照数字(默认档压缩率 27.6%、fact recall 0.776)
已由上节冻结基线取代为数字权威,仅作历史参考;链接见 references.md。
