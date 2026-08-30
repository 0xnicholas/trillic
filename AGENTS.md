# AGENTS.md

Trillic — 自训 prompt 压缩模型项目(为 tokencamp-pro 的 refine 供货)。

## 语言与风格

- 代码、标识符、代码注释、commit message:English。
- 文档:中文。
- Commit:约定式提交(`feat:` / `fix:` / `docs:` / `chore:` …)。

## 必读

动手前读 `docs/decisions.md`(已定决策与排除项)与 `docs/data-strategy.md`
的两条硬约束:

1. MeetingBank 语料禁止进入训练或评测集(基底 checkpoint 的训练集,混入即
   污染);
2. 禁止使用客户 prompt 做训练/评测(隐私边界,无例外)。

## 上游关系

- 模型的运行时宿主是 `../tokencamp-pro/sidecars/refine/`(本仓不存放其代码);
- 产出物 = 与 LLMLingua-2 同接口的 token 分类 checkpoint;验收标准见
  `docs/evaluation.md`。
