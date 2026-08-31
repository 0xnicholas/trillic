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

## 开发

技术栈:Python 3.12+,uv 管理依赖,pytest 测试。依赖刻意保持最小
(httpx、tiktoken、tokenizers、pytest),不含训练栈。

```bash
uv sync                    # 安装依赖
uv run pytest              # 全量测试(零网络:sidecar/网关用确定性假桩)
uv run trillic eval run \
  --config eval/configs/fixture.toml \
  --golden eval/golden/fixture.jsonl \
  --out runs               # 跑 fixture 迷你集(stub 模式,零网络零花费)
```

说明:

- **评测骨架**(`trillic eval run`,issue #2):golden jsonl + TOML 配置进 →
  不可变 run 目录(`metrics.json` + `report.md`)出。
- **指标全集 + 扫档**(issue #6):压缩率双口径(tiktoken 计费 + 模型原生
  token 计数器参数化:`word` / `wordpiece` / `sentencepiece` 三种 flavor,
  换基底只改配置 `[metrics]` 不改代码);fact recall 与 token F0.5 为纯函数,
  口径与运行时 guardrail 同源(`_FACT_PATTERN` 逐字复用;F0.5 同公式、
  恒等对齐替代 embedding,harness 不带模型运行时);`[run] levels` 扫档
  0.1–0.5 每档一节;sidecar 调用逐条计时,p50/p95 按档入报告。假桩下全链
  路可演示。
- **任务级质量 + judge + bootstrap**(issue #7):三类负载按 load_type
  分派下游任务(RAG 问答 / system prompt 约束遵守 / 对话记忆),原文与
  压缩后两份 prompt 经网关各作答一次,LLM judge 按 key_points 逐点 0/1
  打分(条分 = 覆盖率);质量 delta = 压缩后 − 原文,paired bootstrap
  (按 prompt 重采样,种子化 10k 次)95% CI 与“下界不为负”判据进报告;
  judge rubric 版本化(版本号 + 全文 sha256 进报告),judge 与作答模型
  钉死在 `[quality]` 配置段;bootstrap 为纯函数,种子化字面量单测钉死
  数值。stub 网关回显作答 + 机检打分,假桩下压缩只降分(方向性可验
  证);原文作答每 run 只算一次,扫档不加价。
- **golden 工具族**(`trillic golden *`,issue #3):schema v2(含 load_type
  与 source 溯源)校验、LongBench 切分/许可 manifest、种子化 RAG 起草。
  切分纪律:内容寻址半分(sha1 排序,前 ceil(n/2) 为 eval 半),golden
  只从 eval 半取样,同一条目永远不跨 split;MeetingBank 被校验器硬拒。
- **RAG pilot**(`eval/golden/rag_pilot.jsonl`):10 条,源自 LongBench
  qasper / hotpotqa / gov_report,种子 20260831,逐条许可与 split 溯源
  (见 `eval/manifests/longbench.json`);key_points 自数据集标注派生后
  人工修剪。
- **system prompt pilot**(issue #4):生产风格系统提示无公开数据集可抄——
  10 个手写场景族模板(`src/trillic/sysprompt.py`)+ 种子化槽位合成。
  种子分流硬约束(data-strategy 硬约束 4):每族两个不相交整数种子池
  (eval 101–110 / train 901–910),golden 只用 eval 种子,训练侧共用
  生成器代码但永不交叉;分流权威在 `eval/manifests/sysprompt_families.toml`,
  生成记录在 `eval/manifests/sysprompt.json`(含 pilot 冻结校验:逐条
  content_sha1 对重生成比对,漂移即拒)。golden 文件
  `eval/golden/sysprompt_pilot.jsonl`(10 条,key_points 全部为行为约束型:
  格式/拒绝/语气,激活用户消息内嵌于 prompt)。
- **多轮对话 pilot**(issue #5):多轮对话历史无公开数据集可抄——旧仓
  `compression/eval/conversation_gen.py`(同作者)的 5 个场景族移植为
  `src/trillic/dialogue.py` 并新增 5 族,共 10 族种子化槽位合成。结构 =
  多轮 user/assistant 历史(`"role: content"` 行序列化,含代码块)+ 末轮
  用户消息(要求回忆历史事实);key_points 全部为历史事实型(名称/日期/
  先前决定),逐条以原文措辞引用历史(机检 ≥4 连续词重叠),压缩后可判
  存活。种子分流与 sysprompt 同构:分流权威在
  `eval/manifests/dialogue_families.toml`(eval 101–110 / train 901–910
  永不相交),生成记录在 `eval/manifests/dialogue.json`(含逐条
  content_sha1 漂移拒)。golden 文件 `eval/golden/dialogue_pilot.jsonl`
  (10 条,682–1266 tiktoken tokens/条)。
- **中断恢复 + 定容工具**(issue #8 零花费前置):`--golden` 支持多文件
  拼合(三类 pilot 一条命令跑);`--resume-from <run_dir>` 以内容寻址复用
  上一次运行的网关台账(原文答案按 golden sha + 质量钉子复用,压缩侧
  仅当重压缩逐字一致才复用,漂移即诚实重计费),报告与 metrics 记
  fresh/reused 计数与恢复来源;`eval sizing --run <dir>` 从已跑 run 出
  judge 方差、效应量(Cohen's d)与单侧 z 检验反推的每类定容 n。
- **pilot 基线与定容**(issue #8):真链路(sidecar mBERT-base + 真网关)
  30 条 pilot 全量跑通;任务级质量 delta −0.0767,95% CI [−0.1722,
  +0.0044](基线摘录与台账见 `docs/baselines/`);定容决策 **50/类**
  (合并分析为主判据,依据与局限在 sizing-decision);golden 扩产至
  149 条(RAG 49 + sys 50 + dlg 50,合并去重校验通过,生成类 manifest
  冻结含漂移拒);中断恢复在两次真实网关停摆下验证(36 fresh /
  84 reused)。RAG 缺口 1 条:短答案子集(qasper/hotpotqa)机械派生
  要点质量不足、富答案 eval 半已耗尽,补齐需人工修剪(后续票)。
- **假桩注入**:sidecar(POST /refine)与网关客户端均为可注入协议,
  配置 `mode = "stub" | "http"` 切换;全部测试零网络。
- **离线 tiktoken**:测试通过仓内缓存(`eval/assets/tiktoken_cache/`)
  跑通,不依赖外网;真实跑可设 `TIKTOKEN_CACHE_DIR` 指向同一目录。
- **网关密钥**只经 `REFINE_SERVICE_KEY` 环境变量注入,不进配置文件、不落仓。
- **LongBench 数据**不入仓(`_downloads/` 已 gitignore);用
  `curl -L <subsets.toml 里的 source_url> -o _downloads/longbench-data.zip`
  重新下载并解压子集文件后,可重生成 manifest 与起草。

## 状态

阶段 1(评测基线)进行中:#2–#8 已落地(真链路冒烟、pilot 基线、定容
50/类、golden 149 条);剩 #9 双 checkpoint 全量基线 + 冻结收官。
