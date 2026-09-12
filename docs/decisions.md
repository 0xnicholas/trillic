# 决策记录

来源:2026-08-29 tokencamp-pro 仓库内的 grilling 会话(问题:"我们是否应该
开发一个类似 bear 的模型来支持 refine"),同日修订为独立立项。完整背景见
`../../tokencamp-pro/docs/research/thetokencompany-compression.md` §5–§6。

## 定案

### 1. 动机(四项全成立,战略持有级)

- **压缩质量**:现网实测默认档 fact recall 0.776(低于 rewrite 侧 0.9 的
  guardrail 阈值),数字型事实在压缩中丢失;
- **压缩率**:公开 checkpoint 实测 27.6% @ 默认档 0.2,有提升空间;
- **query-aware 能力**:bear-2-finance(The Token Company)与 LongLLMLingua
  (学术)两个独立证据指向按任务定向压缩的价值;
- **自主可控**:核心能力不绑定第三方 2024 年的公开 checkpoint。

### 2. 形态

- **领域微调公开 checkpoint 起步**,不从零训练(从零的收益需微调先证明
  自有数据的价值);
- **query-aware 作为数据形态扩展**,不动模型架构:训练样本为
  `[task; context]` 拼接,只对 context 部分的 token 打保留/删除标签;
- **明确排除生成式压缩模型**:输出不再是输入的子序列,会废掉 compress 环节
  "严格子序列"的安全保证,且与 refine 的 rewrite 阶段功能重叠。

### 3. 立项形态(同日修订,推翻原"三段式 + go/no-go 门槛"版本)

- **独立项目、独立仓库、独立排期**;不占 tokencamp-pro 的 v1 / memory 试点
  带宽,不碰其模块化单体纪律(ADR-0001);
- **任务级评测基线是本项目的第一阶段**:角色是验收尺(证明微调后优于公开
  checkpoint),不是做不做的门槛——立项本身已 go;
- refine 侧的接入面只有一个:`compressor.py` 的 `MODEL_ID`。
  tokencamp-pro 仓不做任何配合性改动。

### 4. Roadmap 与训练开工门(2026-08-30 grilling 修订)

- **两段式能力线**:v1 任务无关微调 → v2 query-aware 数据形态扩展。拒绝一步到位:失败归因要干净、eval 面要最小、接入面承诺分阶段兑现(query-aware checkpoint 无法靠 MODEL_ID 替换上线,`compress()` 只有 text 参数)。
- **"训练零开工"字面门放宽**:原"评测基线在任何训练开工之前完成"修订为"在任何**验收声称**之前完成"——plumbing 性质的 pilot 训练(打通管线、不做任何质量声称)放行;大规模 teacher 花费与正式训练仍等阶段 1 基线数字。
- **止损与降级路径**:v1 最多 3 次验收尝试;3 败则降级收尾(harness 资产化 + 负结果报告,v2 不启动)。

完整阶段结构、门禁与理由见 `roadmap.md`。

### 5. KV cache 相容性硬规则(2026-08-30,bear 研究输入)

- 当压缩面扩展为 role 感知(v2 接入面设计,或 host 侧 per-role 分档)时,
  **assistant 消息与已缓存前缀不可压缩**是硬约束。依据:TTC 将其写进核心
  承诺("keeps your prompt caches valid",docs 三处强调 assistant 永不压缩),
  推断是用生产事故换来的教训;网关的多轮会话同样依赖前缀缓存,压缩破坏
  缓存 = 延迟与成本双重回退,直接吃掉压缩收益。
- 现状注记:现网 compressor 无 role 概念、整段压缩(含 assistant 轮)——
  这是 host 侧(tokencamp-pro)的已知缺口,记录在案防止 v2 设计时重议;
  本仓 v2 接入面 spike 必须把"不破坏已缓存前缀"列为设计前提。
- 来源:`docs/research/bear.md` §C.2、§D.3。

### 6. 接入与供货决策(2026-09-11 grilling:Trillic → refine 的替换)

来源:2026-09-11 grilling 会话(问题:"Trillic 完成后,能替换 tokencamp 的
refine 吗")。事实依据为宿主仓 `sidecars/refine/` 与 `crates/gateway/` 的
侦察结论;可执行细节(drop-in 契约、patch 草案、切换 runbook)见
`docs/delivery/refine-integration.md`。

**替换的实质** — 只替换 compress 阶段的 checkpoint 权重。rewrite 阶段(网关
LLM 回调)、`/refine` 协议、`x-tc-refine*` 头、fail-open 语义(宿主
ADR-0011)、guardrail 一律不变。compress 默认关闭(宿主 `refine_tier` 默认
`off`、`x-tc-refine-compress` 默认 false),影响面 = 显式启用
`rewrite_compress` 档的工作区;收益天花板同此。

**完成锚点与判据** — "完成后" = 阶段 5 产物包五件齐(见 roadmap 阶段 5)。
判据:任务级质量 delta 的 95% CI 下界 ≥ 0 **且** 压缩率或 fact recall 有一项
显著提升;**持平 = 不替换**(负结果归档,refine 继续用公开 checkpoint)。

**drop-in 契约(产物硬性要求)** — `BertForTokenClassification` 兼容
(2 标签 id2label)+ fast tokenizer(force-keep 依赖
`return_offsets_mapping`)+ 暴露 `.bert` 属性(`refiner.py:263` 的相似度
Guardrail 直接调 `self.compressor.model.bert(**inputs)`)。mBERT-base 基底
天然满足;XLM-R-large 不满足(属性为 `.roberta`)——后手牌激活判据据此
修订(见 roadmap 阶段 1)。

**宿主改造集(全部四处)**

1. `sidecars/refine/compressor.py:11` 的 `MODEL_ID` → 本地快照路径;
2. `refine_cache.py:make_cache_key` 加入压缩模型身份——现状只含
   `meta_prompt_version` + `refine_model`,不含压缩机模型,换权重后旧模型的
   压缩结果会**静默继续命中**(看起来切了、实际没切);
3. `crates/gateway/src/refine.rs` 的 `RefineResponse` 补
   `original_tokens`/`refined_tokens`(sidecar 早已返回,网关只反序列化
   `refined_text` + `fallback`);
4. `crates/gateway/src/metrics.rs` + `pipeline/transform.rs` 记录 token 节省
   指标(`model` 标签 = 压缩模型版本,新旧可比;标签来源在接入包里为待定
   实现项)。

**不做**:模型版本走环境变量覆盖(版本应单一来源、可审计;回滚用 git revert
足够)、影子双跑、按工作区分批切换。

**交付与锁定** — 私有 HF 仓作传输通道 → 目标机落成本地快照 + 逐文件 sha256
清单,运行时不再依赖 HF 可达性与凭据;旧快照原地保留作回滚素材。

**评测与回归归属** — harness 与 golden 留在本仓(口径单一来源,复制进宿主
必然漂移);冻结评测包不物理复制进商业仓(同时解决 CC BY-NC 条目的用途解释
与双份漂移),宿主以 pinned ref 调用,项目收尾时整体移交。回归纪律:stub
模式进宿主 CI(零成本);影响压缩面/权重的改动必须跑真链路全量(≈600 次
网关调用),由 Trillic 侧执行、走宿主网关账本,结果摘录入
`docs/baselines/`。

**force-keep 保留** — 宿主 `_FACT_PATTERN` 强制保留不动;对照报告须标注它对
新旧模型同时生效、可能封顶 fact recall 差异(实际可赢的轴 = 压缩率与任务级
质量)。

**上线与回滚** — 直接切换 + applied/fallback 计数 + 节省指标观察;回滚 =
git revert + 重启。宿主无灰度机制,压缩仅 2–3ms,影子双跑与分批切换的成本
吃掉全部收益。

**v2 不纳入"能替换"的定义** — v1 替换成功即答案;v2 query-aware 需扩
`compress()` 与宿主头协议(跨仓,宿主从未承诺配合),独立评估,spike 被拒 →
research artifact。

**供货关系 = 触发式持续供货** — 非一次性交付。再交付触发器:① 真链路回归
跌破验收线;② 负载分布漂移(操作化为 golden 主动扩集 + 线上聚合指标
(节省率/fallback/延迟)漂移告警——均为代理信号,不声称测质量);③ 训练语料
许可变化;④ v2 作为独立能力线立项。无触发则不交付。

**许可** — hotpotqa(CC BY-SA 4.0)可进训练集,model card 如实声明衍生关系;
权重不进公开仓(SaaS-only,不分发)。断供性质:产物自包含(本地快照 + 旧模型
并存),Trillic 停止不影响 refine 运行与回滚。

## 排除项汇总

| 排除 | 理由 |
|---|---|
| 从零训练 token 分类器 | 收益需微调先行证明;成本与维护负担高一个数量级 |
| 生成式压缩 | 破坏严格子序列保证;与 rewrite 功能重叠 |
| 使用客户 prompt 训练 | 隐私边界 / ZDR 承诺,无讨论余地 |
| 接入 TTC API 作后端 | 违背自托管定位;fail-open 语义外包给第三方 |
| 压缩模型走 env 覆盖 | 模型版本应单一来源、可审计;回滚用 git revert 已足够 |
| 影子双跑 / 按工作区分批切换 | 压缩仅 2–3ms,双跑成本吃掉收益;宿主无 per-workspace 模型维度 |
| 评测包复制进商业仓 | CC BY-NC 条目用途需重新解释 + 双份漂移;改为 pinned ref 引用 |
| XLM-R-large 当 drop-in 底座 | 无 `.bert` 属性,直接踩断宿主相似度 guardrail(decisions §6) |
