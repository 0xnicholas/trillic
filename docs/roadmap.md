# Roadmap

来源:2026-08-30 grilling 会话(范围:全程 roadmap)。上游定案见
`decisions.md`;数据与评测的细化策略见 `data-strategy.md` /
`evaluation.md`(本文件修订了其中两处,见 decisions.md §4)。本文件是
阶段结构与门禁的唯一权威;日历排期刻意不在其内。

## 定位:北极星与验收分离

bear(The Token Company)框定能力面——抽取式本体 + query-aware 扩展
——是方向参照,但**不进任何验收标准**:其数字全部厂商自述、口径不可比
(TTC 调研 §3.2:"双方数字各自成立、互不证伪")。验收尺始终是"自家
任务级评测上优于公开 checkpoint"。bear 的产品化特性(分场景分档、
`<ttc_safe>` 标签)属 tokencamp-pro 侧,不在本仓范围。

## 总览

```
阶段 1 评测基线 ──出口:pilot-ramp golden set 冻结 + 双基线数字(含延迟)
   ∥ (基建并行)
阶段 2 数据管线 ──语料构造(切分+许可检查)→ 小规模 pilot 打通
   → [等阶段 1 基线] → 大规模 teacher 蒸馏
阶段 3 v1 微调(mBERT-base)
   硬门:阶段 1 完成
   止损:最多 3 次验收尝试;后手牌仅当已激活;3 败 → 降级收尾,v2 不启动
阶段 4 v2 query-aware
   进入:v1 验收通过;第一项工作 = 接入面 spike;spike 被拒 → research artifact
阶段 5 交付
   产物包四件套;延迟闸 p95 ≤ 现网 1.2×;发布动作归 tokencamp-pro
```

## 阶段明细

### 阶段 1:评测基线(验收尺)

- **pilot-ramp 定容**:先每类 10 条 pilot(共 30)验证 harness 端到端;
  用 pilot 实测 judge 方差,按"能分出效应量"倒推每类样本数(预期落在
  30–50 区间)后补齐。
- **双公开 checkpoint 基线**:现网 mBERT-base + 公开 XLM-R-large
  (零训练成本,只加推理);同时记录各 checkpoint 的压缩延迟(p95)。
- **后手牌激活判据**:双基线显示 XLM-R-large 质量显著更高 **且** 延迟
  可接受,"XLM-R-large 微调"升级点才激活。
- **冻结纪律**:golden set 在基线数字产出时冻结。之后只允许"扩充 +
  重跑基线",不允许改题——考卷跟着学生改,验收作废。
- **判据**:同一 prompt 的原文/压缩后答案,同 judge 模型同 rubric,
  paired bootstrap(按 prompt 重采样)报 95% CI;验收判据 = 任务级
  质量 delta 的 CI 下界不为负。
- **版本钉死**:judge 模型、下游作答模型、rubric 全部落版本进报告。

### 阶段 2:数据管线(基建与阶段 1 并行)

- **切分纪律**:训练语料与评测 golden set 强制切分——LongBench 子集
  显式分半;合成生成器共享代码,但场景族/种子分流。同种子同分布的
  合成数据既当训练集又当考卷,与 MeetingBank 污染同构。
- **许可检查**:逐子集检查**训练用途**许可;research-only 授权的子集
  只能进 golden set,不能进训练集(旧仓 `SOURCES.md` 先例是评测口径,
  不覆盖训练)。
- **小规模 pilot 先行**:几百条打通端到端(语料 → 蒸馏 → 标注 → 质控
  → 训练 plumbing)。pilot 训练只验证管线,不做任何质量声称。
- **花费门**:大规模 teacher 蒸馏(花钱的部分)等阶段 1 基线数字
  产出后才启动。

### 阶段 3:v1 微调与验收

- **基底**:mBERT-base(与现网同架构)——验收对比归因干净("自有
  蒸馏数据 > MeetingBank 数据",架构变量锁死),CPU 延迟零风险。
- **硬门**:阶段 1 完成(验收尺就位且基线数字已产出)。
- **止损**:最多 3 次验收尝试 = 初版 + 2 轮数据迭代;每轮迭代前必须
  写下"这轮赌什么"(假设驱动,不是无脑加数据)。
- **后手牌**:预算内可换 XLM-R-large 重训一次,**仅当**阶段 1 双基线
  已激活其条件;激活条件从未满足时,不许把它当救生牌——架构升级救
  不了数据问题。
- **降级收尾**(3 次仍败):产出物转为——评测基线 + harness 移交
  tokencamp-pro 侧作长期 benchmark 资产;负结果报告(管线哪里不行、
  什么配比试过、数字如何);refine 继续用公开 checkpoint,现网零影响;
  **v2 不启动**(其前提"自有蒸馏数据有价值"已塌)。

### 阶段 4:v2 query-aware

- **进入条件**:v1 验收通过(不要求 v1 已上线——训练线连续推进,上线
  观察并行)。
- **第一项工作是接入面 spike**,不是数据:task 信号从哪来(refine
  管道有无天然任务上下文)、`compress()` 接口怎么扩、tokencamp-pro 侧
  是否接受配合性改动(跨仓谈判,host 有自己的 ADR 纪律)。
- **设计输入(bear 参照,2026-08-30 增补)**:query-aware 的行业参照形态 =
  focus statement(自然语言任务描述)+ retention budget(保留比例)双参数,
  与 aggressiveness(删除阈值)是两套参数化;接入面 spike 时评估 budget
  作为第二参数语义候选,并受 decisions.md §5 KV cache 硬规则约束
  (已缓存前缀不动)。来源:`docs/research/bear.md` 消费点 B。
- **收缩路径**:spike 被拒 → v2 降级为 research artifact——数据管线、
  checkpoint、评测数字照常产出归档,不上生产。v2 的成功定义从"上线"
  变为"证明 query-aware 在自家数据上有效"。
- **验收尺扩展**:golden set 扩 query-aware 版(每类样本带任务上下文),
  扩集后重跑基线(遵守冻结纪律:扩容可以,改题不行)。

### 阶段 5:交付

- **产物包**(四件缺一不算交付):
  1. HF 格式 checkpoint(与 LLMLingua-2 同接口);
  2. model card 含数据溯源:基底 Apache-2.0、各语料子集许可与用途
     声明、合成数据自产声明、MeetingBank 排除声明;
  3. 对照报告(冻结版 golden set 上的验收数字);
  4. 延迟档案(阶段 1 双基线顺带产出)。
- **延迟闸**:compress p95 ≤ 现网 checkpoint 的 1.2×。mBERT-base 同
  架构基底天然满足;此闸主要防后手牌场景(XLM-R-large 换底后延迟翻车)。
- **发布动作归 tokencamp-pro 侧**(MODEL_ID 切换 + 其自身的 canary/
  观察纪律),本项目不越界、不催促。

## 刻意不在本文件内

- 日历排期(独立排期,带宽由项目所有者定);
- golden set 编写方式、下游任务定义、judge rubric——阶段 1 的 spec
  材料,进入实现阶段再定。
