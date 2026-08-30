# bear 深度研究(北极星参照)

调研日期:2026-08-30。定位:Trillic 的北极星参照文档,按项目的三个消费面
(评测方法学 → 阶段 1;query-aware 能力形态 → v2;产品化特性边界 → 防重议)
+ 版本演进考古组织。本文是 tokencamp-pro 侧
[2026-08-29 基线调研](../../tokencamp-pro/docs/research/thetokencompany-compression.md)
的**深挖层**,不重复其结论;基线已定结论直接引用(机制为抽取式 token 分类、
与 LLMLingua-2 同族;厂商数字未经独立验证;两家数字口径互不证伪)。

**北极星纪律(先行声明)**:bear 框定能力面,但**不进任何验收标准**(见
根目录 CONTEXT.md)。本文所有"准确率/提升"类数字均为厂商自述,引用时保持
该定性,不得写成已验证事实。

标注约定:[官方] = TTC 自有渠道自述(官网/blog/docs/OpenAPI/自家包仓库);
[官方代码] = TTC 公开仓库中的代码事实;[第三方] = 独立来源(含 Wayback、
搜索缓存);[推断] = 本文分析;[论文] = arXiv 文献。抓取日期除注明外均为
2026-08-30。

---

## 0. 新增事实总览(相对 2026-08-29 基线)

1. **TTC 公开了完整的评测 harness 源码**:GitHub org `TheTokenCompany/Benchmarks`
   (创建 2026-01-14,最后 push 2026-07-29),含 CoQA / SQuAD 2.0 /
   FinanceBench / LongBench-v2 四套 QA 评测 + 一份独立 latency 基准报告;
   judge prompt、retry 策略、落盘格式全部可读。[官方代码]
2. **n=150 的抽样真相**:不是随机抽样,是**数据集顺序的前 150 条**
   (head-slice,`items[:limit]`)。SQuAD"全部来自 Normans 单篇文章"正是
   head-slice 的副作用;CoQA"Gutenberg 未覆盖"同理。[官方代码]
3. **Judge 判分机制全文曝光**:gpt-5-mini 作 judge,输出二值
   `CORRECT`/`INCORRECT` + 简短解释;FinanceBench 版 judge prompt 明确
   "数值等价(3.5B==$3,500,000,000)、2% 内舍入可接受"。仓库默认答题模型
   也是 gpt-5-mini——与 CoQA blog 自述"GPT-5.4 作答"不一致(§A.5)。
4. **QA 系 blog 无 judge 一致性/人工校验/置信区间**;但 bear-2-safety 文有
   显著性噪声带分析,bear-2-finance 文有 5 seed × 3 read 方差控制——
   方法学成熟度在其各篇发表物之间不均衡。
5. **bear-2-finance 的能力形态**:query-aware = "focus statement"(用户
   问题/指令/agent 子任务)随主文档传入 + **retention budget 10%–90%**
   (保留比例,与 aggressiveness 0–1 是两套参数化);private preview,无
   自助入口;公开 OpenAPI 契约中**无任何 query/task/focus 字段**——
   query-aware 尚未进入公开 API 面。
6. **OpenAPI 3.1 规格公开**(v2.2.0):唯一端点 `POST /v1/compress`;默认
   aggressiveness 0.2;**fail-open 语义明文写入**("压缩故障永不打断调用方
   流程");错误码全套定义;422 校验错误为 FastAPI 风格 → 后端 FastAPI。
7. **未入 spec 的 `compression_ratio` 模式**:latency blog 实际使用
   "对任意内容删掉固定比例"的静态 ratio 参数,但该参数不在 OpenAPI 中——
   API 存在文档外模式。
8. **版本时间线大幅精确化**(§D):bear-1.1 是 2026-02 案例的生产模型;
   bear-1.2 在 2026-02 已有吞吐实测(fan-out 路径 1.7M tok/s);bear-2 于
   2026-06 登场;bear-2-finance 2026-08;bear-1/1.1 发布公告现已 soft-404
   (内容仅存搜索缓存)——旧模型营销页被系统性回收。
9. **SDK 双语言全量公开**(PyPI/npm,均 2026-06-03 首发):MIT;SDK README
   自述"**默认压缩所有角色,含 assistant/agent**"——与 docs("assistant
   消息永不压缩以保 KV cache")**正面矛盾**(§C.5,矛盾如实记录)。
10. **latency blog(2026-08,未列入其自索引)**:9 模型 × 5 provider,
    200K token,TTFT −29%~−49% @ratio 0.5;披露反例(Gemini 3.5
    Flash-Lite 压缩后 +44% 更慢)与排除项(生成端安全层拒收部分压缩文本)。
11. **他们在自家 harness 里记录了自己的 API bug**:"output_tokens
    sometimes exceeds input; cap to original"。[官方代码]

---

## A. 评测方法学(bear 怎么证明自己)

### A.1 CoQA blog(2026-06,bear-2)[官方 + 官方代码]

- **设置**:压缩 story 文本后交 GPT-5.4 问答;对话历史不压缩(story 级
  压缩缓存,每 story 压一次复用)。
- **抽样**:CoQA validation 展开为逐轮条目后取**前 150 条**(无 seed、
  无分层)。150 条覆盖五个 in-domain 源中的四个——head-slice 的自然结果。
- **判分**:LLM judge;harness 默认 judge=gpt-5-mini,只输出
  `CORRECT`/`INCORRECT`+解释,允许同义/改写/包含式匹配。**单次判分,
  无 judge 一致性、无自洽投票、无人工校验**。
- **数字**:control 93.3% → τ=0.05:94.7%(+1.3pp,仅省 0.1%)→ τ=0.2:
  95.3%(+2.0pp,省 8.2%);分域 MCTest +7.3pp。全部厂商自述。
- **"+2pp"归因**:去噪效应(删限定词/回指/套话,数字与名字全保留)——
  机制层定性论证 + 单点实验,**无消融**(不存在"随机删除等量 token"对照)。
- **自述局限**:仅 150 题、未跑完整集、CoQA 文本偏短。**未提及** CI、
  显著性、judge 噪声。
- **口径缺口**:准确率分母只计 judge 成功判分的条目(judge 失败被剔除)
  ——极端情况下轻微抬高准确率。[官方代码]

### A.2 SQuAD 2.0 blog(2026-03,bear-1.2)[官方 + 官方代码]

- validation **前 150 题**,全部来自同一篇 Wikipedia 文章(blog 自己承认);
  7 档扫描(0.05–0.7)+ control,共 1,200 次评测。
- **唯一公开完整剂量-响应曲线的 QA 基准**:0.05:+4.0pp/−17.3% tokens;
  0.3:−2.0pp;0.5:−4.0pp;0.7:−7.3pp/−45.4%。**重压缩掉到基线以下**——
  与 CoQA"压越多越好"叙事形成内部张力(他们自己的解释:抽取式 span 问答,
  删多了把答案 span 删掉了)。
- 不可答检测 +7.5pp 是最大增益来源;自列五条局限。
- **可复现性声明只兑现一半**:blog 链接 "Code & results on GitHub",
  仓库含 harness,但 QA results/ 目录未提交。[官方代码]

### A.3 其他两篇的方法学层级 [官方]

- **bear-2-finance(2026-08)**:FinanceBench 全 150 题(全集);两个
  reader(Claude Haiku 4.5 / Gemini 3.5 Flash Lite);**5 seed × 每 seed
  3 次读取**,报五 seed 均值——其发表物中方差控制最强的一篇;与
  truncation / BM25 / embedding RAG 做**等 token 预算对比**(33% 预算:
  bear-2-finance 76.9 vs Embedding RAG 74.9 vs BM25 69.8 vs 头部截断
  46.4)。图注泄露内部配方名 "bear-2-fin-1"。
- **bear-2-safety(2026-06,准技术报告)**:6 个安全基准全集(500–1,660
  例);F1 对 gold 标签;**显著性分析**:"二项比例正态近似 95% CI,
  <0.025(或 0.015)视为持平";τ∈[0.05,0.5] 每 0.1 一档完整扫描;
  局限一节披露 FP:FN 偏向。**其全部公开材料中方法学最强的一篇**。
- **latency(2026-08)**:33 篇听证转录(10K–200K token);每配置 10 次
  取中位;披露反例与排除项——反例披露是加分项。

### A.4 与 LLMLingua-2 论文的方法学对照

| 维度 | TTC(bear) | LLMLingua-2 | 谁强 |
|---|---|---|---|
| 判分 | QA 系全靠 LLM judge(单次、无一致性披露) | 任务原生指标(EM 等,零判分噪声) | LLMLingua-2 |
| 样本 | n=150 head-slice / safety 全集 | 各基准全集、多任务 | LLMLingua-2 |
| 方差控制 | 仅 finance 与 safety 有 | 多 seed、同行评审 | 平手 |
| 可复现 | harness 公开(无 QA 结果数据);模型/数据闭源 | 数据+代码+checkpoint 全公开 | LLMLingua-2 |
| 端到端口径 | **强项**:生产 LLM 答题 + token 成本 + TTFT + pipeline 对比 | 压缩器延迟 + 端到端提速 | TTC |
| 负结果披露 | SQuAD 高档掉点、反例、排除项 | 论文自述局限 | 平手 |

**结论[推断]**:TTC"生产口径强、学术口径弱"。若把其 safety 文的噪声带
公式套回 CoQA:150 题二项 95% CI ≈ ±3.5pp,**+2.0pp 在统计上不可与其
自定阈值区分**——用其自家方法学推出的、对其旗舰数字的内部不一致。

### A.5 口径不一致清单 [官方]

1. CoQA blog 称 GPT-5.4 作答;harness 默认 gpt-5-mini——仓库与 blog
   不同步。
2. SQuAD blog 头部只写 bear-1.2;harness config 跑了 bear-1.2 + bear-1.1
   (结果未发布)。
3. FinanceBench 公开 harness 只压缩 oracle evidence 页全文,与 blog 的
   "整篇 419K token filing、超窗回退"是两套评测——**公开代码 ≠ blog
   数字的生成器**(finance 部分)。
4. 延迟口径四处并存:<50ms(首页)/ p95 150ms(FAQ)/ ~85ms per 100K
   (SDK README)/ 0.1–0.6s 中位 + fan-out sub-120ms(latency blog)。

### 消费点 A(喂阶段 1)

1. **评测协议可抄骨架**:三份 judge prompt(会话式/抽取式/金融数值式)
   是现成模板;resume-on-interrupt、增量落盘、按源域/轮次/可答性分解的
   汇总输出值得移植进 harness。
2. **必须补上他们缺的三件事**:分层抽样(拒绝 head-slice)、judge 噪声带
   (复用 ±1.96√(p(1−p)/n) 公式写进报告)、judge 一致性抽检(如 5% 人工
   双判)——补完即在其 QA 系口径之上。
3. **负结果先例存在**(SQuAD 高档掉点):我们的评测应预设"压缩率-质量
   权衡曲线"为标准产出,而非单点数字——与既定的"扫档覆盖 0.3 以上"一致。

---

## B. query-aware 能力形态(bear-2-finance 线索)

### B.1 已知 [官方]

- **定义**:"在主文档之外提供一个 focus statement 即可获得更强压缩;
  focus statement 可以是用户问题、一条指令、或 agent 的子任务"。用户设定
  **retention budget(10%–90%)**,模型返回相关 span 纯文本,<1 秒。
- **与 bear-2 的关系**:基础 bear-2 任务无关;bear-2-finance = query-aware
  化 + 金融调优。**没有任何"bear-2 vs bear-2-finance 同题同档"量化数字**。
- **数字(厂商自述,五 seed 均值)**:两 reader 在 90%→50% 保留档全部与
  全文档基线不可区分;33% 档:76.9 vs Embedding RAG 74.9 vs BM25 69.8 vs
  截断 46.4;成本 −50%,TTFT −25%~−45%(含压缩时间)。
- **准入**:private preview,面向 design partners;无公开价格、无自助
  开通、无 SDK 支持;quickstart 主流程仍 invite-only。
- **生产脚注**:800K token 已生产验证;419K 文档 50% 档自动回退 33%
  (budget 是目标值非硬保证,存在兜底逻辑)。

### B.2 API 层表达:公开材料不足(如实记录)

- **已知**:公开 OpenAPI 的请求只有 model / input / aggressiveness /
  app_id 四字段,**没有 task/query/focus/budget 字段**;公开 harness 的
  payload 同样三字段;`compression_ratio` 只在 latency blog 叙述里。
- **未知**:私有 preview 的实际 schema(focus 是新字段、塞 input、还是
  独立端点)、budget 字段名与类型、响应是否带 span 坐标、计费口径。
  硬约束(不注册不调用)下无法验证。
- **[推断]**:blog 说 "One endpoint, one budget parameter, no index to
  build"——大概率是同端点加 budget(+focus)参数扩展,非新端点。

### B.3 方法论意义 [推断]

其 33% 等预算对比是"学习型压缩 vs 检索"的同成本对比——LongLLMLingua
之外的第二个独立信号,且是唯一带 pipeline 级对照的。**注意反读**:
embedding RAG 与其差距仅 2pp,在 judge 噪声带(±~0.035 @n=150)边缘,
"显著优于 RAG"的表述要打折扣;真正被碾压的只是截断类方案。

### 消费点 B(喂 v2)

1. **接口形态已收敛出行业参照**:focus statement(自然语言任务描述)+
   retention budget(保留率)双参数——与已定案的"`[task; context]` 拼接、
   只对 context token 打标"数据形态同构;budget 参数化值得作为 aggressiveness
   之外的第二档 API 语义记入候选(面向"我有 N 个 token 预算"的调用方)。
2. **窗口期判断成立**:query-aware 尚未公开可用、无 SDK、无公开 schema;
   自训模型若在窗口期内做出同能力,不存在"落后一代"的既成事实。
3. **等预算对比设计应进评测蓝图**:压缩器 vs BM25 vs embedding RAG 的
   同 token 预算横评是最有说服力的评测形态,harness 可预置这三条对照线。

---

## C. 产品化特性机制(细节层)

### C.1 分场景分档指导 [官方]

| 档位 | 范围 | 场景 |
|---|---|---|
| Light | 0.05–0.15 | 金融报告与法律合同;医疗记录 |
| Moderate | 0.15–0.4 | 会议转录与通话录音;网页抓取 |
| Aggressive | 0.4–0.9 | 聊天历史压缩;替代 Claude compact |

OpenAPI 内嵌口径略不同(0.05–0.2 给"模型要直接阅读的文本";0.5–0.8 给
对话历史);bear-2-safety 的 τ 定义域是 [0.0, 0.5]——变体有自己的参数域。

### C.2 按 role 分档的语义 [官方]

- dict 按 role 分档:推荐 system 0.1 / user 0.3–0.5 / tool 0.5–0.7;
  **不在 dict 里的 role 不压缩**。
- **assistant 消息不压缩的官方理由**:"so the LLM cache is fully
  preserved"——保护 provider 侧 KV/prompt cache 命中,不是内容原因。
- 场景配方:RAG(system 0.1 / user 0.3 / 检索文档 0.7)、Agentic(system
  0.1 / tool 0.6)、Chat(system 0.1 / user 0.4)。

### C.3 `<ttc_safe>` 保护标签 [官方]

- 语法:`<ttc_safe>...</ttc_safe>` 包裹;SDK 侧 `protect()` 生成;另有
  `protect_json`(仅 agents.md 一句)。
- 语义:包裹 span 逐字通过。解析规则(嵌套、转义、跨消息、闭合缺失)
  **未公开**;无 "experimental" 字样但无硬保证。
- **[推断]**:保护责任交给调用方手工标注;**他们没有公开任何自动事实
  保护(数字/日期强制保留)**——CoQA 例文数字全保留是模型行为,不是 API
  保证。这仍是可差异化的点。

### C.4 API 工程细节 [官方]

- **Fail-open**:"压缩故障永不打断调用方流程";agents.md 版:返回原文
  且带 `error` 字段。**内部矛盾**:response schema 无 `error` 字段,spec
  又定义 503——两种故障语义并存,何种故障走哪条路未说明。
- 错误码 401/402/413/422/429/500/503 全套;429 无配额数字、无
  Retry-After;413 上限数值未公布;422 为 FastAPI 风格。
- **确定性承诺**:"same input + same aggressiveness → same output",
  官方明示可安全缓存、保 prompt cache 有效。
- `input` 接受 string/object/array(先 JSON 编码);`app_id` 用于用量
  分账;SDK 支持 gzip 请求体。
- **文档外模式**:`compression_ratio`(latency blog 实证使用);API 接受
  的模型集合大于文档声明集合(harness config 仍列 bear-1.1)。

### C.5 SDK 契约 [官方]

- Python(0.5.1):`TheTokenCompany(api_key).compress(text, model=,
  aggressiveness=, app_id=)`;同步+异步;依赖仅 httpx;MIT。响应对象
  加算 `tokens_saved`、`compression_ratio`(SDK 侧便捷字段,不在 wire
  format)。包装器:`thetokencompany.openai.with_compression(...)`、
  `thetokencompany.anthropic` 变体。
- TypeScript(0.5.0):subpath exports `/openai` `/anthropic` `/ai-sdk`;
  AI SDK 提供 `withCompression()` 与 `compressionMiddleware()` 两种集成。
- 发布时间线:双语言均 2026-06-03 首发、2026-08-13 同日 0.5.x;发布人
  rasmus-u(2 人团队之一)。
- **README 与 docs 的 assistant 矛盾(重点)**:README"默认压缩所有角色,
  含 assistant;传省略 assistant 键的 dict 以保 KV cache"vs docs"自动压缩
  所有非 assistant 消息"。至少一处与代码实况不符(未装包验证,硬约束内
  无法仲裁)。[官方,矛盾如实记录]

### 消费点 C(喂边界,防重议)

1. **fail-open 语义与"数字事实强制保留"**:TTC 把"返回原文+error"作为
   产品承诺;`<ttc_safe>` 是手工标注,无自动事实保护——tokencamp-pro 侧
   的 compress 数字事实保护小项可对齐 fail-open 并以自动化差异点设计。
2. **per-role 分档 + 三场景配方是现成产品化范式**:system 0.1 / user
   0.3–0.5 / tool 0.5–0.7 与网关三类负载一一对应,可作 refine 档位文档的
   对照基准;assistant 不压缩的 KV cache 理由同样适用于网关前缀缓存。
   **此类产品化特性归 tokencamp-pro 侧,不在本仓范围**(基线调研已定案,
   此处仅存机制细节供 host 侧取用)。
3. **`compression_ratio` 预算模式是真实的第二接口形态**(8 月 blog 实证
   使用):如要支持"按保留率输出",阈值语义之外还需预算控制——v2 数据
   形态设计时预留。

---

## D. 版本演进考古

### D.1 时间线(证据 + 精度)

| 时间 | 事件 | 证据/精度 |
|---|---|---|
| 2025(年内) | 公司成立,bear-1 面世 | [官方]YC 页;公告无日期、现已 soft-404;精度:季 |
| 2026-01-14 | Benchmarks 仓库创建 | [官方]GitHub API;精度:日 |
| 2026-02 | bear-1.1 在 Pax Historia 生产(193B tokens/月) | [官方]blog 索引;精度:月 |
| 2026-02 | bear-1.2 吞吐实测:fan-out 1.7M tok/s、P50 <120ms | [官方]latency blog 引述;精度:月 |
| 2026-03-16 | Wayback 首录 bear-1/bear-1-1 公告页(存在下界) | [第三方];精度:日(下界) |
| 2026-03 | SQuAD 2.0 blog(bear-1.2,7 档扫描) | [官方]页头;精度:月 |
| 2026-04-22 | Gateway 延迟基准(EC2) | [官方]REPORT.md 自注;精度:日 |
| 2026-06-03 | 双语言 SDK 开源首发 | [官方]包仓库时间戳;精度:秒级 |
| 2026-06 | **bear-2 登场** + bear-2-safety | [官方]blog 索引;精度:月 |
| 2026-07-29 | Benchmarks 仓库最后 push(config 仍列 1.2+1.1) | [官方];精度:日 |
| 2026-08-13 | 双 SDK 同日 0.5.x | [官方];精度:秒级 |
| 2026-08 | bear-2-finance private preview;latency blog | [官方];精度:月 |

YC W26 批次与节奏交叉:仓库创建落在批次开始期,bear-2 与 SDK 开源在
Demo Day 后约一个季度。[推断]

### D.2 版本叙事 [官方 + 推断]

- **bear-1**:"first LLM input compression model… semantic compression"。
  注意**早期用"semantic compression"话术,现官网全面改口"deterministic
  delete-only"**——定位从"智能语义压缩"收敛到"确定性抽取"。[推断]
- **bear-1.1**:精度修复型("more precise about which tokens to remove")。
- **bear-1.2**:无独立公告;docs 定位"Faster compression. Lower
  latency"——**低延迟/高吞吐型迭代**。
- **bear-2**:质量跃升型("Most accurate. Best quality preservation")。
- **bear-2-safety**:安全分类器预处理,非对称保留(危险内容 95–100%
  保留、良性 25–70% 压缩);**家族首个"输出不可复用"变体**(仅供分类,
  不得进入检索/摘要/审计/人读)。
- **考古障碍**:`/blog/bear-1`、`/blog/bear-1-1` 现返回博客索引页
  (soft-404),`/blog/bear-1-2`、`/docs/python-sdk` 硬 404;旧版
  docs/python-sdk 缓存显示模型表曾是"bear-1.2 (recommended), bear-1.1,
  or bear-1"——**bear-2 上线后旧模型营销页被系统性回收**;而 API 实际
  接受集(config 仍调得动 bear-1.1)大于文档声明集。
- 命名体系:bear-{maj}.{min} 主线 + bear-2-{domain} 变体;finance 图注
  泄露内部配方名 "bear-2-fin-1"——变体内部也在版本化。

### D.3 从演进序列推断:他们持续在修什么 [推断]

1. **延迟/吞吐是第一迭代轴**(1 → 1.1 "faster" → 1.2 "lower latency"
   + 1.7M tok/s 实测 + Gateway 基准):压缩自身延迟直接吃掉节省;bear-2
   发布后 1.2 仍以低延迟档并存——**单模型没有同时做到质量最优+延迟最低,
   双 SKU 并存是质量-延迟权衡的产品化**。
2. **上下文超窗是真实痛点**:评测从"问答准确率"(3 月)演进到"pipeline
   级:装得下+更快+更便宜"(8 月)——卖点从质量转向 fit/latency/cost
   三合一。
3. **KV cache 相容性是被踩过的坑**:assistant 不压缩的三处重复强调 +
   "keeps your prompt caches valid"——推断早期有客户因压缩破坏前缀缓存
   而受损。
4. **压缩文本触发生成端安全层是已知风险**(Fable 5 拒收部分压缩文本被
   排除出基准):抽取式删除可能产出"像注入"的碎片文本,他们知道但只在
   脚注披露。[官方披露+推断其普遍性]
5. **家族扩张跟着客户负载走**:主线修通用质量/延迟;safety/finance 变体
   各绑定一类买家;Enterprise 自定义微调始终在列。

### 消费点 D(校准预期)

1. **"质量 vs 延迟双档并存"是可借鉴的发布形态**:若单一 checkpoint 无法
   两全,可用同一底座的两个蒸馏规格复刻;checkpoint 产物接口(替换
   `MODEL_ID`)天然支持。
2. **迭代节奏校准**:主线 4 版/8 个月,变体 2 个/3 个月——小分类器模型
   可以月级迭代;他们每版发布都带一篇基准 blog,评测产出物就是发布物料,
   与本仓"评测先行"的结构同构。
3. **KV cache 相容性必须进设计约束**:compress 阶段若按 role 分档,
   assistant/已缓存前缀不动是硬规则(TTC 用真金白银换来的教训)——
   decisions.md 候选条目。

---

## Gaps 与重抓触发

未能确认:bear-2-finance 私有 API schema;bear-1/1.1 精确发布日期;CoQA
blog 实际 judge 模型;rate limit / 每请求上限数字;`compression_ratio` 与
aggressiveness 的关系;SDK 对 assistant 消息的真实默认行为(可读公开 SDK
源码仲裁,本轮未展开)。

**重抓触发**(满足任一则重查):bear-2-finance 转 GA(重查 openapi.json
diff);新模型版本发布(bear-3 或新变体);SDK 0.6+;Benchmarks 仓库新
push;独立第三方出现可复现实测(触发基线调研的重议条件)。

## 重议触发:第一方实测 bear API

本文档在"不注册、不调用"约束下完成。第一方实测(bear 跑我们的冻结
golden set)列为**条件后续项**,三个入口条件同时满足时再议:

1. golden set 已冻结(阶段 1 产出);
2. harness 就绪(可加 API adapter);
3. TTC 准入可得(preview/GA)。

注意:若实测结果为 bear 显著优于同档 LLMLingua-2,恰好满足基线调研 §5
的重议触发条件之一——那是数据输入决策,不是恐慌理由。

---

## 来源清单

抓取日期除注明外均为 2026-08-30。

### TTC 官网 / docs / blog
- [官方] [blog/coqa](https://thetokencompany.com/blog/coqa)、[blog/squad-v2](https://thetokencompany.com/blog/squad-v2)、[blog/bear-2-safety](https://thetokencompany.com/blog/bear-2-safety)、[blog/compressing-sec-filings](https://thetokencompany.com/blog/compressing-sec-filings)、[blog/latency](https://thetokencompany.com/blog/latency)、[blog 索引](https://thetokencompany.com/blog)
- [官方] [docs/compression](https://thetokencompany.com/docs/compression)、[docs/aggressiveness](https://thetokencompany.com/docs/aggressiveness)、[docs/protect-text](https://thetokencompany.com/docs/protect-text)、[docs/quickstart](https://thetokencompany.com/docs/quickstart)、[docs/openai](https://thetokencompany.com/docs/openai)、[docs 索引](https://thetokencompany.com/docs)
- [官方] [openapi.json](https://thetokencompany.com/openapi.json)(v2.2.0)、[agents.md](https://thetokencompany.com/agents.md)、[llms-full.txt](https://thetokencompany.com/llms-full.txt)
- [官方→404] /blog/bear-1、/blog/bear-1-1(soft-404,内容经搜索缓存);/blog/bear-1-2、/docs/python-sdk(硬 404,旧版经搜索缓存)

### TTC GitHub / 包仓库
- [官方代码] [TheTokenCompany/Benchmarks](https://github.com/TheTokenCompany/Benchmarks)(创建 2026-01-14、push 2026-07-29;config.yaml、compress.py、coqa/、squad_v2/、financebench/、latency/REPORT.md)
- [官方] [GitHub org repos API](https://api.github.com/orgs/TheTokenCompany/repos)(三仓库时间戳)
- [官方] [PyPI the-token-company](https://pypi.org/pypi/the-token-company/json)(版本时间线、README)、[npm the-token-company](https://registry.npmjs.org/the-token-company)(subpath exports、SLSA provenance)

### 缓存与存档
- [第三方缓存] Exa 搜索摘要(bear-1 / bear-1-1 / helonic / docs-python-sdk,查询于 2026-08-30)
- [第三方] [Wayback CDX:首页](http://web.archive.org/cdx/search/cdx?url=thetokencompany.com)、[blog 路径](http://web.archive.org/cdx/search/cdx?url=thetokencompany.com/blog*)(查于 2026-08-30)

### 排除
- 2026 横评/目录页(pointfive 等):基线已覆盖,无新增方法学信息,属 GTM 叙事(任务排除项)
- YC 公司页、LLMLingua-2 论文/模型卡/数据集:基线已覆盖,仅作对照引用
