# Round 2 Review

| Dimension | Weight | R1 | R2 | Δ |
|---|---:|---:|---:|---:|
| Problem Fidelity | 15% | 8.5 | **8.7** | +0.2 |
| Method Specificity | 25% | 6.5 | **6.8** | +0.3 |
| Contribution Quality | 25% | 7.0 | **8.0** | +1.0 |
| Frontier Leverage | 15% | 8.0 | **8.3** | +0.3 |
| Feasibility | 10% | 6.5 | **6.8** | +0.3 |
| Validation Focus | 5% | 6.0 | **7.4** | +1.4 |
| Venue Readiness | 5% | 8.0 | **8.2** | +0.2 |
| **Overall** | | 7.45 | **7.71** | +0.26 |

**Verdict**: REVISE
**Drift Warning**: NONE for core proposal

## Anchor & Direction

- **Anchor preserved**: 仍攻 multi-modal offline RL 中 rare value-relevant mode preservation
- **Dominant contribution sharper**: ✓ "global W₂ fails; mode-conditional W₂ fixes mode recall with a bound"——可读、单一焦点
- **Method simpler**: ✓ 但仍轻度 overbuilt（marginal TV 项可换更直接的 conditional consistency）
- **Frontier appropriate**: ✓ contrastive encoder + FiLM 是 conditional generation 标准现代 primitive

## Remaining Critical Issues

### CRITICAL #1: action vs action-chunk unit 不一致（Method Specificity）
- `c_ω(z|s, a_{0:H})` 期望 trajectory/action chunk 但 Diffusion-QL 通常输出 single action
- Fix: 二选一并贯穿到底：
  - (a) **action-chunk policy**: π_θ 输出 H 步 chunk a_{0:H-1}；c_ω, W₂, DSM, theorem 都以 chunk 为 unit
  - (b) single-action policy: c_ω(z|s,a) 仅看 state-action 对；定理改为 action-mode recall

### IMPORTANT #1: 弱训练信号矛盾核心 claim
- W₂ 项用 m_β(z|s) 加权 → rare mode 得 rare gradient → 与 paper 核心 claim "保 rare mode" 直接冲突
- Fix: balanced mode replay (每 z 等量) + soft GMM responsibilities (而非 hard c_ω > 0.5)

### IMPORTANT #2: schedule 乐观
- 96 runs "5 天" 不可信（ICNN tuning + 失败 seed + D4RL variance）
- Fix: core = 4 tasks × 4 baselines × 3 seeds = 48 runs；8 tasks 作 completion target

## Simplification Opportunities

1. **Replace D_TV(m_θ, m_β) with conditional mode-consistency loss**:
   ```
   L_consistency = E_{z~m_β, a~π_θ(.|s,z)} [- log c_ω(z | s, a_chunk)]
   ```
   直接问 "条件 z 下生成的 a 是否仍在 mode z"——比 marginal TV 更直接、更易估计、更易读
2. Soft GMM responsibilities 替代 hard c_ω > 0.5
3. Adroit/NeoRL-2 严格 appendix；不让其重新进入主 claim

## Modernization Opportunities

1. FiLM mode tokens + classifier-free-style **conditioning dropout**：训练时 ~ 5% 概率 z = null，让 denoiser fallback to marginal behavior（mode 信心低时鲁棒）
2. Reuse diffusion-policy state encoder features for f_ω 若已有，否则保独立 SimCLR
3. 不要 LLM / Q-cost / 蒸馏

## Remaining Action Items

1. **Define the generated object**: single action or action chunk. 全套公式对齐
2. **Replace m_θ TV with conditional consistency**
3. **Add balanced mode replay** 让 rare mode 真获训练信号
4. **State novelty around** "global W₂ failure + recall theorem" 不要落在 "per-mode ICNN engineering"
