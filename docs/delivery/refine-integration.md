# refine 接入包(骨架草案)

状态:**草案**(2026-09-11 建立于 grilling 会话);阶段 5 交付前冻结为 v1 定稿。
对象:tokencamp-pro 的 refine sidecar(`sidecars/refine/`)与网关 transform 阶段。
事实基线:2026-09-11 对宿主仓的侦察(下文行号以该次侦察为准)。
决策依据:`docs/decisions.md` §6;阶段结构见 `docs/roadmap.md` 阶段 5。

本包是产物包第五件:让宿主**照做即可接入**的单元,不是说明文档。

## 1. drop-in 契约(产物校验清单)

一次切换是否可行,先看这份清单——任何一条不满足,都要求宿主改 `refiner.py`,
那就不再是"替换权重"而是"改造宿主"。

- [ ] `config.json` 的 `architectures` 含 `BertForTokenClassification`,
      `num_labels == 2`,`id2label` = {0: drop, 1: keep};
- [ ] tokenizer 为 **fast**(`return_offsets_mapping=True` 必须可用——
      force-keep 靠 offsets 定位);
- [ ] 模型类暴露 **`.bert`** 属性(`refiner.py:263` 的相似度 Guardrail 直接调
      `self.compressor.model.bert(**inputs)`);
- [ ] 与现网同架构基底(mBERT-base 系列)→ 延迟档案与内存占用可直接对比;
- [ ] 随附对照报告与延迟档案(阶段 5 产物包其余各件)。

交付前在本仓跑一遍(离线、零网络):

```bash
uv run python - <<'PY'
from transformers import AutoTokenizer, AutoModelForTokenClassification
p = "<checkpoint dir>"
tok = AutoTokenizer.from_pretrained(p)
m = AutoModelForTokenClassification.from_pretrained(p)
assert m.config.num_labels == 2, m.config.num_labels
assert hasattr(m, "bert"), "guardrail needs .bert (XLM-R would fail here)"
assert tok.is_fast and tok("x", return_offsets_mapping=True)["offset_mapping"]
print("drop-in OK:", m.config.architectures)
PY
```

> 已知反例:XLM-R-large 的类属性是 `.roberta`,**不是** drop-in。后手牌激活
> 判据已据此修订(roadmap 阶段 1)。

## 2. 交付物与传输

传输走私有 HF 仓,落盘为**本地快照**——运行时不再依赖 HF 可达性与凭据。

```
sidecars/refine/models/trillic-v1/
  config.json  model.safetensors  vocab.txt  tokenizer_config.json  ...
  SHA256SUMS                       # 逐文件校验清单
sidecars/refine/models/models--microsoft--llmlingua-2-...   # 旧快照,原地保留
```

```bash
huggingface-cli download <org>/trillic-v1 --revision <sha> \
  --local-dir sidecars/refine/models/trillic-v1
(cd sidecars/refine/models/trillic-v1 && shasum -a 256 -c SHA256SUMS)
```

旧快照目录**不删**:它就是回滚素材(见 §4 第 10 步)。

## 3. 宿主改动 patch 草案

全部改动就四处(+1 待定,见 3.4)。其余文件一行不动。

### 3.1 `sidecars/refine/compressor.py`

```diff
-MODEL_ID = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
+# Trillic v1 — local snapshot, sha256-verified (SHA256SUMS)
+MODEL_ID = os.path.join(os.path.dirname(__file__), "models", "trillic-v1")
```

```diff
     def __init__(self, model_id: str = MODEL_ID):
         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
+        # 缓存键用的模型身份标签,与加载路径同源(避免手工漂移)
+        self.model_tag = os.path.basename(os.path.normpath(model_id)) or model_id
         self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
```

注:本地路径下 `cache_dir` 参数被 `transformers` 忽略,无副作用;启动加载
完全离线。失败语义不变——`server.py:22-36` 仍会落到 light mode
(`compressor=None`,继续服务 rewrite)。

### 3.2 `sidecars/refine/refine_cache.py`

现状键 = `text ‖ rewrite ‖ compress ‖ aggressiveness ‖ min/max_tokens ‖
meta_prompt_version ‖ refine_model`,**不含压缩模型** → 换权重后旧模型的
压缩结果会继续命中(静默失效,最危险的一类 bug)。

```diff
     meta_prompt_version: str,
     refine_model: str,
+    compressor_model: str,
 ) -> str:
     parts = [
         ...
         meta_prompt_version,
         refine_model,
+        compressor_model,
     ]
```

调用点 `refiner.py:151-154` 增加实参 `self.compressor.model_tag if
self.compressor else "none"`(`compressor=None` 时 compress 路径本来就不跑)。

### 3.3 `crates/gateway/src/refine.rs`

sidecar 早已返回 token 计数,网关只反序列化了 `refined_text` + `fallback`:

```diff
 pub struct RefineResponse {
     pub refined_text: String,
     #[serde(default)]
     pub fallback: bool,
+    /// Token counts straight from the sidecar; no prompt content.
+    #[serde(default)]
+    pub original_tokens: u64,
+    #[serde(default)]
+    pub refined_tokens: u64,
 }
```

### 3.4 `crates/gateway/src/metrics.rs` + `pipeline/transform.rs`

```rust
pub fn record_refine_tokens(direction: &str, model: &str, count: u64) {
    metrics::counter!(
        REFINE_TOKENS_TOTAL,
        "direction" => direction.to_owned(),   // original | refined
        "model" => model.to_owned(),
    )
    .increment(count);
}
```

调用点:`transform.rs:85-103`(`record_refine` / `record_refine_duration`
附近),按 sidecar 响应各记一次。

**待定实现项(唯一一个)**:`model` 标签从哪来——

- (A) 宿主侧部署常量(如 env `REFINE_COMPRESSOR_MODEL=trillic-v1`):零 sidecar
  改动,但与 `compressor.py` 的 `MODEL_ID` 两个来源,存在漂移风险;
- (B) sidecar `RefineResponse` 加 `compressor_model` 字段(取自
  `Compressor.model_tag`,light mode 为 `"none"`):单一来源,改动 +1 处。

**推荐 (B)**:标签是"新旧可比"的唯一凭据,值得多改一处。

隐私:只记计数,不记 prompt 内容——与宿主既有边界一致。

## 4. 切换 runbook

1. 落盘快照并跑 §2 的 sha256 校验;
2. 打 §3 的 patch(4 处 + 3.4 的 A/B 选择);
3. 冷启动 sidecar(uvicorn);
4. `GET /health` → `{"status":"ok","compressor":true}`(false = light mode,停);
5. stub 冒烟:本仓 `uv run trillic eval run --config eval/configs/fixture.toml`
   (零网络零花费);
6. 真链路回归:本仓 pinned harness 跑冻结评测包全量(147 条 ≈ 600 次网关
   调用),走宿主网关的账本;
7. 对照冻结基线核判据:质量 delta 95% CI 下界 ≥ 0 且压缩率或 fact recall 有
   一项显著提升——**持平即停,不切换**;
8. 切换(改常量 + 重启);
9. 观察:applied/fallback/skipped 计数 + token 节省指标(按 `model` 标签);
10. 异常回滚:`git revert` + 重启;旧快照仍在原地,缓存键已变,老条目自然
    失效——回滚不残留脏缓存。

## 5. 非差异清单(切换后**不变**的东西)

- `/refine` 与 `/compress` 协议、请求/响应字段语义;
- `x-tc-refine*` 头与 `refine_mode` / `refine_tier` / aggressiveness 语义;
- rewrite 阶段(LLM 经网关回调)与 `x-tc-refine: false` 递归防护;
- fail-open 与 light mode(宿主 ADR-0011)及 `x-tc-refine: skipped` 标记;
- 三项 guardrail 与 `_FACT_PATTERN` force-keep;
- 缓存 LRU/TTL 行为、超时(`REFINE_TIMEOUT_MS`)、隐私边界(sidecar 只见
  prompt 文本)。

## 6. 回归与再交付触发器

回归纪律:stub 模式进宿主 CI(零成本);**影响压缩面/权重的改动**必须跑真链路
全量(Trillic 侧执行,结果摘录入 `docs/baselines/`)。

再交付(Trillic → refine)触发器,无触发不交付:

1. 真链路回归跌破验收线;
2. 负载分布漂移——操作化为 golden 主动扩集 + 线上聚合指标(节省率/
   fallback/延迟)漂移告警;**均为代理信号,不声称测质量**(客户 prompt 不可
   用于评测,硬约束 2);
3. 训练语料许可变化;
4. v2 作为独立能力线立项。

## 7. 断供说明

产物自包含:本地快照 + 旧模型并存,`from_pretrained(local_path)` 离线可跑。
**Trillic 项目停止不影响 refine 运行与回滚**——这是"交付物供货"优于"代码
依赖"的地方。

## 8. 待现场确认(本仓侦察无法确定)

- 生产宿主是否有仓外的容器/编排(宿主仓内无 Dockerfile/compose/k8s 证据);
- `sidecars/refine/models/` 在目标环境是否持久(裸机进程重启后仍在);
- 是否存在会带上 sidecar 的自托管客户包(SaaS-only 判断来自宿主 ADR-0017,
  但环上没有直接证据)。
