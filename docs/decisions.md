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

## 排除项汇总

| 排除 | 理由 |
|---|---|
| 从零训练 token 分类器 | 收益需微调先行证明;成本与维护负担高一个数量级 |
| 生成式压缩 | 破坏严格子序列保证;与 rewrite 功能重叠 |
| 使用客户 prompt 训练 | 隐私边界 / ZDR 承诺,无讨论余地 |
| 接入 TTC API 作后端 | 违背自托管定位;fail-open 语义外包给第三方 |
