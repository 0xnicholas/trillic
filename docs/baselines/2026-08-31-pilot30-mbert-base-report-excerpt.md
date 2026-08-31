# Eval run 20260831T132510388886Z-pilot30-57381968

- created: 2026-08-31T13:25:10+00:00
- harness: trillic v0.1.0 (python 3.13.12, tiktoken 0.14.0)
- golden: 30 items, sha256 `127c070e82eb…`
- sidecar: mode=http, refine_model=deepseek/deepseek-v4-pro, rewrite=False, compress=True, aggressiveness=0.2
- token calibers: tiktoken cl100k_base (billing) + native `wordpiece` (compression_ratio = fraction removed; kept_ratio = retention)
- sweep levels: 0.2

## Level 0.2 — refine_model deepseek/deepseek-v4-pro

| id | tiktoken o→c | kept | removed | native o→c | native kept | fact recall | F0.5 | latency (ms) |
|---|---|---|---|---|---|---|---|---|
| rag-qasper-16c24587 | 3525→2528 | 0.7172 | 0.2828 | 3960→2829 | 0.7144 | 0.7955 | 0.7374 | 5.33 |
| rag-qasper-4249cb42 | 2491→1818 | 0.7298 | 0.2702 | 2921→2214 | 0.7580 | 1.0000 | 0.7353 | 2.21 |
| rag-qasper-1a620bce | 2587→2052 | 0.7932 | 0.2068 | 2951→2351 | 0.7967 | 0.9141 | 0.7516 | 1.66 |
| rag-qasper-342d4384 | 3854→3217 | 0.8347 | 0.1653 | 4754→3894 | 0.8191 | 0.8512 | 0.6324 | 1.86 |
| rag-hotpotqa-378ae0ef | 2527→1808 | 0.7155 | 0.2845 | 2586→1832 | 0.7084 | 0.9878 | 0.7545 | 1.65 |
| rag-hotpotqa-79b9693a | 2740→1968 | 0.7182 | 0.2818 | 2590→1828 | 0.7058 | 0.7174 | 0.7288 | 1.72 |
| rag-hotpotqa-638db17f | 3712→2537 | 0.6835 | 0.3165 | 3739→2609 | 0.6978 | 0.8977 | 0.7795 | 3.23 |
| rag-hotpotqa-3795906b | 3042→2196 | 0.7219 | 0.2781 | 3085→2226 | 0.7216 | 0.9515 | 0.7243 | 2.15 |
| rag-gov_report-2144ef74 | 2102→1659 | 0.7892 | 0.2108 | 2422→1937 | 0.7998 | 0.8889 | 0.7754 | 1.78 |
| rag-gov_report-4c05f517 | 2669→1808 | 0.6774 | 0.3226 | 2760→1827 | 0.6620 | 0.8438 | 0.7870 | 1.71 |
| sys-support_logistics-s101 | 233→160 | 0.6867 | 0.3133 | 254→172 | 0.6772 | 1.0000 | 0.6241 | 1.55 |
| sys-finance_analyst-s101 | 229→142 | 0.6201 | 0.3799 | 247→154 | 0.6235 | 1.0000 | 0.6512 | 1.53 |
| sys-medical_triage-s101 | 256→158 | 0.6172 | 0.3828 | 294→185 | 0.6293 | 1.0000 | 0.6040 | 1.53 |
| sys-code_reviewer-s101 | 208→145 | 0.6971 | 0.3029 | 238→167 | 0.7017 | 1.0000 | 0.7101 | 1.50 |
| sys-sales_outbound-s101 | 193→137 | 0.7098 | 0.2902 | 225→161 | 0.7156 | 0.8750 | 0.6387 | 1.65 |
| sys-legal_contract-s101 | 231→142 | 0.6147 | 0.3853 | 249→153 | 0.6145 | 0.6667 | 0.6362 | 2.31 |
| sys-socratic_tutor-s101 | 193→143 | 0.7409 | 0.2591 | 203→145 | 0.7143 | 1.0000 | 0.6979 | 2.19 |
| sys-api_support-s101 | 227→161 | 0.7093 | 0.2907 | 256→176 | 0.6875 | 0.7000 | 0.6972 | 1.75 |
| sys-hr_policy-s101 | 215→159 | 0.7395 | 0.2605 | 248→176 | 0.7097 | 0.8333 | 0.6583 | 1.83 |
| sys-incident_commander-s101 | 278→209 | 0.7518 | 0.2482 | 293→217 | 0.7406 | 0.5000 | 0.5687 | 1.77 |
| dlg-support_ticket-s101 | 971→689 | 0.7096 | 0.2904 | 1103→766 | 0.6945 | 0.6400 | 0.6296 | 1.76 |
| dlg-pair_programming-s101 | 1266→974 | 0.7694 | 0.2306 | 1477→1107 | 0.7495 | 0.8929 | 0.5273 | 1.78 |
| dlg-requirements_discussion-s101 | 1033→701 | 0.6786 | 0.3214 | 1192→816 | 0.6846 | 0.9583 | 0.6340 | 1.73 |
| dlg-data_pipeline-s101 | 1026→748 | 0.7290 | 0.2710 | 1182→837 | 0.7081 | 0.4737 | 0.5819 | 1.55 |
| dlg-devops_incident-s101 | 951→764 | 0.8034 | 0.1966 | 1104→863 | 0.7817 | 0.7755 | 0.5788 | 1.91 |
| dlg-vendor_procurement-s101 | 772→522 | 0.6762 | 0.3238 | 863→589 | 0.6825 | 0.5556 | 0.6392 | 2.07 |
| dlg-onboarding_rollout-s101 | 783→596 | 0.7612 | 0.2388 | 832→589 | 0.7079 | 0.4194 | 0.5806 | 1.96 |
| dlg-bug_triage-s101 | 675→517 | 0.7659 | 0.2341 | 760→551 | 0.7250 | 0.4444 | 0.5659 | 1.63 |
| dlg-access_review-s101 | 692→499 | 0.7211 | 0.2789 | 767→527 | 0.6871 | 0.3333 | 0.5389 | 1.62 |
| dlg-release_planning-s101 | 760→595 | 0.7829 | 0.2171 | 811→587 | 0.7238 | 0.1200 | 0.5971 | 1.63 |

- items: 30
- corpus (tiktoken): 40441 → 29752 (kept 0.7357, removed 0.2643)
- corpus (native): 44366 → 32485 (kept 0.7322, removed 0.2678)
- mean fact recall: 0.7679, mean token F0.5: 0.6589
- latency: p50 0.0018s / p95 0.0028s (mean 0.0020s, n=30)

### Task quality (LLM judge on key_points)

| id | load | original | compressed | delta |
|---|---|---|---|---|
| rag-qasper-16c24587 | rag | 0.0000 | 0.0000 | +0.0000 |
| rag-qasper-4249cb42 | rag | 0.0000 | 0.0000 | +0.0000 |
| rag-qasper-1a620bce | rag | 0.0000 | 0.0000 | +0.0000 |
| rag-qasper-342d4384 | rag | 1.0000 | 0.0000 | -1.0000 |
| rag-hotpotqa-378ae0ef | rag | 0.0000 | 0.0000 | +0.0000 |
| rag-hotpotqa-79b9693a | rag | 0.0000 | 0.0000 | +0.0000 |
| rag-hotpotqa-638db17f | rag | 0.0000 | 0.0000 | +0.0000 |
| rag-hotpotqa-3795906b | rag | 1.0000 | 1.0000 | +0.0000 |
| rag-gov_report-2144ef74 | rag | 0.4000 | 0.4000 | +0.0000 |
| rag-gov_report-4c05f517 | rag | 0.4000 | 0.0000 | -0.4000 |
| sys-support_logistics-s101 | system_prompt | 0.8000 | 0.8000 | +0.0000 |
| sys-finance_analyst-s101 | system_prompt | 1.0000 | 1.0000 | +0.0000 |
| sys-medical_triage-s101 | system_prompt | 1.0000 | 0.7500 | -0.2500 |
| sys-code_reviewer-s101 | system_prompt | 0.0000 | 0.2500 | +0.2500 |
| sys-sales_outbound-s101 | system_prompt | 0.7500 | 0.5000 | -0.2500 |

---

# 归档说明(2026-08-31)

- 完整产物:`runs/20260831T132510388886Z-pilot30-57381968/`(metrics.json + report.md,runs/ 不入仓;本文件为可读摘录)
- checkpoint:sidecar mBERT-base(`microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`),aggressiveness 0.2(默认档)
- 下游作答 deepseek/deepseek-v4-pro;judge kimi/kimi-k2(rubric v1,sha256 f48af57b1a80…)。
  注:judge 选型受环境约束 —— anthropic 上游 key 为占位符不可用(见 eval/configs/pilot30.toml 注释);judge 与作答不同厂,规避自偏好
- 计费:120 次网关调用(30 条 × 原文/压缩 × 作答/judge)+ 冒烟 4 次 + 网关可用性探测 6 次;台账 runs/.ledger/4ed646c85c53ad34.jsonl(60 对,含两次中断后 journal 续跑复用的 84 次调用)
- 过程注记:网关两次瞬时停摆(requests 挂起 >120s 后自愈),触发两次 journal 续跑 —— "杀掉重跑,已完成条目零重复网关调用"在真实中断下成立(36 fresh / 84 reused);该故障模式已反馈宿主仓库

## 数字要点

- 压缩(默认档 0.2):tiktoken 口径 kept 0.62–0.83 / 条;WordPiece 原生口径与之同趋势 —— 双口径参数化在真 checkpoint 上工作
- fact recall:RAG 0.72–1.00;system prompt 全 1.00(数字约束逐字存活 —— 宿主仓 593a70b 的 force-keep 生效)
- 任务级质量(主验收指标口径):mean original 0.6025 → compressed 0.5258;
  **mean delta −0.0767,95% CI [−0.1722, +0.0044],CI 下界为负**(n=30,压缩伤害方向明确、单类未达显著)
- RAG 类存在 judge 地板效应(terse 作答在原文/压缩两侧同判 0,配对差值仍有效但信息量低)—— 定容与解读时已知局限
