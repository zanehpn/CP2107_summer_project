# Round 3 Review

| Dim | Wt | R1 | R2 | R3 | Weighted (R3) |
|---|---:|---:|---:|---:|---:|
| Problem Fidelity | 15% | 8.5 | 8.7 | **8.8** | 1.32 |
| Method Specificity | 25% | 6.5 | 6.8 | **7.4** | 1.85 |
| Contribution Quality | 25% | 7.0 | 8.0 | **7.8** | 1.95 |
| Frontier Leverage | 15% | 8.0 | 8.3 | **8.5** | 1.28 |
| Feasibility | 10% | 6.5 | 6.8 | **7.2** | 0.72 |
| Validation Focus | 5% | 6.0 | 7.4 | **8.0** | 0.40 |
| Venue Readiness | 5% | 8.0 | 8.2 | **8.2** | 0.41 |
| **Overall** | | 7.45 | 7.71 | **7.93** | |

**Verdict**: REVISE
**Drift Warning**: NONE
**MAX_ROUNDS reached** — 用 Round 2 refinement (7.93 highest) 作 FINAL 基础；Round 3 的 surgical fixes 在 Phase 5.5 expansion 整合

## Anchor & 状态

- Anchor preserved（仍是 multi-modal offline RL 中 rare value-relevant mode preservation）
- Dominant contribution sharper（"global W₂ fails; mode-conditional W₂ fixes mode recall with bound"）
- Method simpler（unit 统一到 chunk）但仍有理论/统计 overclaim
- Frontier leverage appropriate（FiLM + classifier-free + frozen contrastive 都自然）

## Round 3 — 4 个 Surgical Fixes

### Surgical #1 (CRITICAL): Theorem N_z 应换 N_eff(z)
- 当前: balanced replay 让 N_z = N/K → ε_W(z) bound
- 问题: oversampling 不创造独立数据；statistical sample size 应是有效样本数 `N_eff(z) = (Σ_i w_i(z))² / Σ_i w_i(z)²`
- Fix: 定理 statement 用 N_eff(z)；balanced replay 仍保 gradient budget 平等，但 statistical error 取决于 effective sample size；ρ_min 显式条件保 N_eff(z*) > 0
- Priority: CRITICAL (定理 correctness)

### Surgical #2 (IMPORTANT): chunk 执行歧义
- 当前: "apply a_chunk[0] (or every-H replan)" 二选一未定
- 问题: 如果 every-H execute，Q(s, a_chunk[0]) 不是正确改进目标；如果只 a_chunk[0]，chunk 是 mode-conditioning object 不是 open-loop plan
- Fix: 明确 receding-horizon——always execute a_chunk[0]；chunk 是 mode-conditioning + diffusion 稳定化对象，控制仍单步
- Priority: IMPORTANT

### Surgical #3 (IMPORTANT): invalid (s, z) pair
- 当前: `(1/K) Σ_z E_s W₂` 隐含每 mode 在每 state 都 valid
- 问题: 真实轨迹中许多 (s,z) 无 support
- Fix: 改采样为 `z ~ Uniform(K), (s, a_chunk) ~ D weighted by c_ω(z | s, a_chunk)`——天然只在 valid (s,z) 上做 W₂ + DSM
- Priority: IMPORTANT

### Surgical #4: pseudo-novelty 风险
- Reviewer 可能描述为 "Q-DOT per cluster + FiLM"
- Fix: novelty 定位明确写：(a) global W₂ 的 failure mode 反例；(b) mode-recall 定理；不靠 K ICNN engineering
- Priority: MINOR (positioning, not mechanism)

## Simplification

1. Delete "every-H replan" — receding-horizon a_chunk[0] only
2. Replace N_z = N/K with N_eff(z)；balanced replay = optimization 不 statistics
3. 简化术语：f_ω + GMM posterior 之后统一称 "frozen mode classifier"

## Modernization

1. Reuse diffusion-policy encoder features for f_ω 若 silhouette OK；SimCLR fallback
2. Keep classifier-free dropout 但不作 contribution
3. 不用 LLM/Q-cost/蒸馏/online

## Drift Warning

NONE。Adroit/NeoRL-2 严格 appendix 即可。

## Remaining Action Items (for Phase 5.5)

1. 重写 theorem 用 N_eff(z) 分离 statistical error 与 replay/optimization error
2. 承诺 receding-horizon execution
3. 重定义 balanced sampler over valid weighted chunks
4. 重定位 novelty narrative
