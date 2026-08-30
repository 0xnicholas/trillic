# Trillic

自训 prompt 压缩模型项目:在公开 LLMLingua-2 checkpoint 上微调出自有压缩模型,为 tokencamp-pro 的 refine 供货。本文件是术语表,只收录本项目语境特有的词汇。

## Language

### 定位

**北极星**:
The Token Company bear 系列在本项目中的角色——框定能力面(抽取式本体 + query-aware 扩展)的方向参照。北极星不进任何验收标准:其数字全部为厂商自述、口径不可比。
_Avoid_: 验收对象、benchmark 对手、竞品指标

**验收尺**:
任务级评测基线的角色代称——"微调后优于公开 checkpoint"的唯一度量。没有尺,不许声称赢。
_Avoid_: 门槛、go/no-go gate

### 能力线

**v1**:
任务无关的领域微调 checkpoint(mBERT-base 基底),项目第一条能力线。
_Avoid_: MVP、beta 版

**v2**:
query-aware 数据形态扩展(`[task; context]` 拼接,只对 context 打标)产出的 checkpoint,第二条能力线;进入前提是 v1 验收通过。

**后手牌**:
XLM-R-large 微调升级点。仅当阶段 1 双基线同时证明"公开 XLM-R-large 显著更强"且"延迟可接受"时激活;不作为止损救生牌。
_Avoid_: 备选方案、fallback

**research artifact**:
v2 接入面 spike 被拒后的收缩形态——数据管线、checkpoint、评测数字照常产出归档,不上生产。

### 交付

**产物包**:
v1 的交付单元,四件缺一不算交付:checkpoint + 溯源 model card + 对照报告 + 延迟档案。
