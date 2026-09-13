# 训练语料配比设计(v1)

来源:issue #16(阶段 2 数据管线地基)。本文回答一个问题:**v1 训练集里
三类负载各占多少、为什么**。语料构造管线、许可核查与切分纪律的机制性
部分见 `data-strategy.md` 与 `training/manifests/training-corpus-v1.json`;
本文只谈配比主张及其理由,是后续定容决策(parent #15)的输入之一。

## 输入与对照

三类负载沿用 `data-strategy.md` 的模型:

| 负载类型 | 来源 | 本设计的角色 |
|---|---|---|
| RAG 长文(检索拼接文档) | LongBench 公开子集(train 半侧) | 公开打底,基准量 |
| system prompt(生产风格) | 手写场景族 × 种子合成 | 加重 |
| 多轮对话历史 | 场景族 × 种子合成(旧仓生成器移植 + 新族) | 加重 |

对照物是评测侧:golden 149 条按 rag / system_prompt / dialogue ≈
49 / 50 / 50(≈1:1:1)构造——那是**考卷**口径,三类都要考,均衡合理。
训练口径不同:训练要对着**网关真实负载的缺口**投喂,而不是对着考卷
投喂。

## 配比主张

**rag : system_prompt : dialogue = 30% : 35% : 35%**(定容时容差 ±5pt,
三类合计归一)。

理由逐条:

1. **公开打底不撤、但不加重(30%)**。基座 checkpoint(LLMLingua-2,
   MeetingBank + 维基文风蒸馏)已经在"公开文本文档"分布上训练过;
   LongBench RAG 长文对它不是新分布。保留三成的作用是**锚住通用
   抽取能力、防止微调把模型带偏到合成模板**,而不是补能力缺口。
2. **加重后两类,因为缺口在那里(35% + 35%)**。生产压缩面
   (decisions §6)是显式启用 rewrite_compress 的工作区:被压缩的文本
   是请求里的 system prompt、多轮历史与注入文档。公开数据集对
   前两者**没有覆盖**(生产风格 system prompt 无公开数据集,多轮
   会话历史受隐私边界约束不可采集,只能合成)——微调收益只能落在
   这两类上。
3. **两类均分,避免偏科**。两类合成机制同源(场景族 × 种子变体),
   没有证据支持某一类更缺;均分把"哪类收益更大"留给阶段 3 的
   分负载验收数据回答,而不是让配比预判结论。
4. **合成总量封顶 ≈ 2/3,防模板过拟**。合成层的多样性受场景族数量
   约束(sysprompt / dialogue 各 10 族 × 种子槽位变体),占比过高会
   把模型过拟到模板文风——表现在考卷上就是 golden 合成条目虚高、
   真实负载不涨。2/3 封顶 + 公开 30% 打底是显式的对冲。
5. **配比是定容决策的变量,不是本文件的铁律**。pilot 全链路跑通后,
   定容决策(parent #15)在同一 manifest 版本化框架下可调整比例——
   但每次调整必须当作新数据版本(新 manifest + sha256),不许静默改。

## v1 语料层现状

- 基础层(issue #16)= LongBench train 半侧、训练用途许可核查通过的子集:
  **hotpotqa 150 + gov_report 150 = 300 条**,全部 `load_type=rag`;
- 合成两层(issue #17)已生成:**system_prompt 350 + dialogue 350**,
  落地本文 30/35/35 锚点配比(300 rag + 700 合成 = 1000 条,实际
  达成 30.0/35.0/35.0);语料 = `training/corpus/synthetic-train-v1.jsonl`,
  manifest = `training/manifests/synthetic-training-v1.json`(逐族
  seed 计划 + sha256 + mix 记录);
- qasper(train 半侧 112 条)**许可排除**:逐文档 CC-BY-* 变体
  (含 NC/SA)无法从 LongBench 分发行逐条核验,维持保守排除
  (eval-only)。证据链见 `eval/manifests/subsets.toml` 的
  `train_use_evidence` 与 `training/manifests/training-corpus-v1.json`;
- 合成层的多样性受生成器 slot 池约束(sysprompt 每族 9–486 种组合,
  dialogue 每族 ≥243 种):选取器扫描 train 种子段(901–990,预留
  headroom 而非配额),**内容去重**后按族轮转录取至目标条数,因此
  低多样性族贡献少(hr_policy 5 条)、高多样性族贡献多
  (support_logistics 70 条),dialogue 侧恰好每族 35 条——逐族条数
  以 manifest 为准;后续 pilot(几百条)与全量定容的具体数字由
  定容决策(parent #15)在同一版本化框架下调整。

## 合成层纪律(已落地,issue #17)

- **种子分流**:合成生成器代码与 golden 共享,但场景族种子段互斥
  (eval 101–110 / train 901–990,train 段为 headroom 而非配额);
  训练合成只用 train 段,双方校验器互斥断言——golden 侧生成器对
  train 种子 raise,训练侧对 eval 种子 raise,且选取器/校验器在
  **内容层**再断言一次零重叠(不同种子可渲染出 byte 相同的 prompt,
  slot 池小干这种情况是常态,选取器直接跳过);
- **选取规则**(确定性,同输入同输出):按注册表族序轮转,族内
  从 train 段升序扫描,仅录取渲染内容在语料内唯一且不在 golden
  指纹集中的种子,至目标条数为止;条数配比与逐族 seed 计划全部
  版本化进 manifest(sha256 锚定);
- **v2 query-aware 预留**:schema 第一天含 `question` / `task` 字段
  (v1 恒空);v2 填充时样本形态为 `[task; context]` 拼接、只对
  context 打标(decisions §2),多轮对话条目的 role 行序列化让
  "assistant 轮与已缓存前缀不可压缩"(decisions §5)在数据层可执行;
- **零网关调用纪律**:本层产出(合成、manifest、校验)全部为本地
  确定性模板渲染;teacher 蒸馏的花费全部发生在后续蒸馏 issue,进
  网关账本。重建入口:`scripts/build_synthetic_training.py`(等价于
  `trillic corpus build-synthetic` + 全量校验)。

## 许可与排除速览

| 子集 | 许可 | 训练用途 | 关键依据 |
|---|---|---|---|
| hotpotqa | CC BY-SA 4.0 | **可**(义务:署名 + 同享;model card 声明,权重不分发) | hotpotqa.github.io 许可声明 |
| gov_report | 美国联邦政府作品(公有领域);GovReport 再分发层 CC BY 4.0 | **可** | 17 U.S.C. §105 + 语料内 GAO 声明 |
| qasper | 记录口径 CC BY-NC 4.0(保守) | **不可**(eval-only) | 创建者卡片 CC BY 4.0,但逐文档 CC-BY-* 变体不可核验 → 保守排除 |
| MeetingBank | — | **硬拒**(任何来源) | 基座 checkpoint 训练集(硬约束 1) |
| 客户 prompt | — | **硬拒**(无例外) | 隐私边界 / ZDR(硬约束 2) |

完整证据链接以 `eval/manifests/subsets.toml`(curation)与
`training/manifests/training-corpus-v1.json`(冻结记录)为准。
