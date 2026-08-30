# 数据策略

微调的核心瓶颈是数据,不是算力。本文件定训练数据的来源、管线与约束。

## 蒸馏管线(复用 LLMLingua-2 的公开方法)

LLMLingua-2 论文(arXiv:2403.12968)的数据构造管线可直接复用:

1. **teacher 压缩**:GPT-4 级模型对原始语料分块(≤512 token)做压缩;
2. **词级标注**:模糊匹配算法从 teacher 的压缩结果导出每个词的
   保留/删除标签;
3. **质控**:Variation Rate 过滤(丢弃波动最大的 5%)、Alignment Gap
  过滤(丢弃对齐最差的 10%)。

teacher 调用走 tokencamp 自有网关——成本进内部账本,且网关的路由/计费/
fallback 全部复用。参考实现:旧项目 `compression/eval/`(见 references.md)。

## 语料来源

- **公开语料打底**:LongBench 各子集(gov_report、qasper 等,许可证见旧仓
  `eval/corpus/SOURCES.md` 的先例)及其他开放文档集;
- **合成语料补配比**:LLM 生成"典型网关负载"。旧仓
  `eval/conversation_gen.py` 的种子对话生成器是现成起点。

## 配比:对齐网关真实负载

沿用旧 benchmark 的三类负载模型,并**加重后两类**(公开数据集对 RAG 长文
已有覆盖,缺口在另两类):

| 负载类型 | 来源 | 配比倾向 |
|---|---|---|
| RAG 长文(检索拼接文档) | LongBench 等公开集 | 基准量 |
| system prompt(生产风格) | 手写/合成(无公开数据集) | 加重 |
| 多轮对话历史 | 合成(种子场景族 × 变体) | 加重 |

## 硬约束

1. **MeetingBank 排除**:它是基底 checkpoint(LLMLingua-2)的训练集。混入
   训练或评测都会污染对比——评测时模型在背答案。
2. **客户 prompt 不可用**:隐私边界与 ZDR workspace 承诺。训练与评测语料
   只能来自公开数据集与合成数据。
3. query-aware 扩展(第二阶段)的样本需要带任务上下文(question/task),
   语料 schema 从第一天就要预留该字段,避免返工。
4. **train/eval 切分**:训练语料与评测 golden set 必须切分——LongBench
   子集显式分半;合成生成器共享代码,但场景族/种子分流。同种子同分布的
   合成数据既当训练集又当考卷,与 MeetingBank 污染同构的错误。
5. **训练使用许可**:逐子集检查"训练用途"许可。旧仓 `SOURCES.md` 的
   许可先例是**评测**口径,不覆盖训练;research-only 授权的子集只能进
   golden set,不能进训练集。
