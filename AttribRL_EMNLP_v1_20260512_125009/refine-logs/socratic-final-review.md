# Socratic Final Review — AttribRL

**Model**: gpt-5.5 (xhigh reasoning)
**Thread ID**: 019e17d4-3da8-7633-97d7-624fdcc569cd
**Dialogue turns**: 3 (Turn 0 → Turn 2 → understanding declaration)

## Final Scores

| Dimension | Weight | Score |
|---|---:|---:|
| Problem Fidelity | 15% | 8.5 |
| Method Specificity | 25% | 8.0 |
| Contribution Quality | 25% | 7.4 |
| Frontier Leverage | 15% | 7.0 |
| Feasibility | 10% | 6.4 |
| Validation Focus | 5% | 8.0 |
| Venue Readiness | 5% | 6.8 |

**OVERALL: 7.62 / 10**
**Verdict: REVISE** (not RETHINK; core mechanism clear; EMNLP Findings 强潜力，main 取决于实证干净度)

## Per-dimension comments

- **Problem Fidelity (8.5)**: discovery-use gap 直接 optimize target，不是 outcome / CoT proxy → 切中 tool-agent 训练痛点
- **Method Specificity (8.0)**: 三轮修正后 cf construction / KL token target / normalization / copy defense / D_t / extractor 都可执行
- **Contribution Quality (7.4)**: trajectory-level utilization 可训练 reward + DiscUseBench 绑定，是最有价值的；扣分点 = Pearl/backdoor framing 太强难成立，建议降调为 *diagnostic counterfactual sensitivity* 而非 *causal identification*
- **Frontier Leverage (7.0)**: GRPO/Qwen/tool-bench 是当前有效路线；没有新 RL 算法、没有 frontier-scale、没有 retrieval；胜负靠 reward design 和诊断质量
- **Feasibility (6.4)** ⚠️: 18 GPU-day 对 10-variant ablation + 7B/14B + multi-seed 偏乐观；SWE-Bench + 14B + reward computation 并行有工程摩擦
- **Validation Focus (8.0)**: V7 vs V1/V2/V3/V8 能隔离 dense shaping / lexical-only / cf-only / CST predictability；建议四主指标 = conversion + accuracy + copy audit + cf sensitivity
- **Venue Readiness (6.8)** ⚠️: 方法机制清楚，但论文叙事过载风险：matched distractor / argument-slot KL / delayed KL / EMA / null baseline / D_t / Mask / extractor / 10 ablations 全展开 → 主贡献漂到工程修补

## Critical action items

### Feasibility fix (IMPORTANT)
- 主实验保留 V0/V1/V2/V3/V7/V8 (6 个)
- V4/V5/V6/V9 → 1-seed diagnostic 或 appendix pilot
- 14B 只跑 V0/V7/V2 (3 个)
- 提前 stopping rule: V7 在 BFCL+ToolBench 不超过 V2 则停止扩展到全部 benchmark

### Venue Readiness fix (IMPORTANT)
- 主文只讲一个核心对象: `evidence-specific action sensitivity reward`
- delayed KL / adaptive α / V9 AgenTracer / 部分 extractor details → appendix
- 主实验 2-3 个最强诊断图支撑核心 claim

## Simplification Opportunities

1. **Drop adaptive α**: 当前 `min(normalized cf, lex)` 已提供 conjunctive gate，adaptive α 加复杂收益不清
2. **Delayed KL → diagnostic only**: 主方法先用 argument-slot KL + EMA；delayed KL 作 robustness analysis
3. **V9 AgenTracer post-hoc filter → related baseline (no training variant)**

## Modernization Opportunities

1. D_t judge calibration section: human agreement / judge sensitivity / FP-FN taxonomy
2. Reward hacking audit dashboard: copy ratio / non-evidence copy / evidence pass-through / null-distractor gap / KL floor hit rate
3. CST/AgenTracer 区别 object-of-optimization 表: CoT predictability vs post-hoc attribution vs train-time action sensitivity

## Drift Warning

**轻微 drift**: 原是 "simple trajectory-level utilization reward"，现扩为 multi-gate / multi-baseline / multi-calibration system。主文必须收束，否则 contribution 看起来是 heuristic stack。

## Path to Outstanding

若 V7 干净击败 V1/V2/V3/V8 在 conversion 上 (+30pp)，且 reward hacking audit 显示 copy ratio < 5%，paper 可冲 Outstanding；否则 Findings 落点稳健。
