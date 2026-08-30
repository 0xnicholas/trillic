# Trillic

自训 prompt 压缩模型项目——为 tokencamp-pro 的 refine 子产品训练自有压缩 checkpoint。

## 定位

tokencamp-pro 的 refine 压缩环节当前使用 Microsoft LLMLingua-2 公开 checkpoint
(`bert-base-multilingual-cased-meetingbank`,2024)。Trillic 的目标是在公开
checkpoint 上领域微调出自有模型,在压缩质量、压缩率、query-aware 能力上超过它,
实现核心能力的自主可控。

**产出物**:一个 token 分类 checkpoint(与 LLMLingua-2 同接口:逐 token 保留
概率)。refine sidecar 替换 `compressor.py` 的 `MODEL_ID` 即接入,网关零改动。

**方法族**:抽取式压缩(只删不改,输出是输入的严格子序列)。明确不做生成式
压缩——那会破坏压缩输出的安全保证(见 docs/decisions.md)。

## 文档地图

- `docs/decisions.md` — 立项决策(2026-08-29 grilling 定案,含修订史)
- `docs/roadmap.md` — 阶段结构、门禁与止损(2026-08-30 grilling 定案)
- `docs/data-strategy.md` — 训练数据策略:蒸馏管线、语料来源与配比、硬约束
- `docs/evaluation.md` — 第一阶段:任务级评测基线(项目验收尺)
- `docs/references.md` — 论文、旧项目文档、对标调研的索引

## 上游参考

- 对标调研:`../tokencamp-pro/docs/research/thetokencompany-compression.md`
  (The Token Company / bear 的产品事实、技术对比、口径分析)
- refine 实现:`../tokencamp-pro/sidecars/refine/`(本项目模型的运行时宿主)

## 状态

初始化完成,仅文档。Roadmap 已定案(见 `docs/roadmap.md`);第一阶段
(评测基线)未开工。
