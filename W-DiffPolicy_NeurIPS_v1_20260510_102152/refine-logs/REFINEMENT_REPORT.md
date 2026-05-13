# Refinement Report — IDEA-01 Mode-Conditional W-DiffPolicy

**Problem**: 离线 RL 中 diffusion policy 的 KL/score-BC 行为正则在 multi-modal 上系统失效（rare value-relevant mode 被吞）
**Initial Approach**: W-DiffPolicy v1 — KL → W₂ replacement + mode-preservation theorem
**Date**: 2026-05-10
**Rounds**: 3 / MAX_ROUNDS=3 (自动收敛触发)
**Final Score**: 7.93 / 10
**Final Verdict**: REVISE
**Codex thread**: `019e0d5d-3c03-79c0-9ec5-dfe562c2f248`

## Problem Anchor [verbatim]

- Bottom-line: 离线 RL 中 behavior policy 多模态，diffusion policy 的 KL/score-BC 行为正则在多模态上系统失效
- Must-solve: rare value-relevant mode preservation
- Non-goals: 不解决 inference 加速；不重写 Q-learning；不做 online；不主张首个 Q-cost OT
- Constraints: 2× 4090; 6-8 weeks NeurIPS 2026; D4RL Kitchen/AntMaze (核心) + Adroit/NeoRL-2 (附录)
- Success: chunk-mode recall × return Pareto dominate global-W₂/KL on 3/4 core tasks; finite-sample theorem 含 N_eff(z) + ε_W + ε_clf + ε_c; toy 反例 figure; inference overhead < 5%

## Skeleton (final)

- State A → State B；Path: Diagnosis (toy 反例) → Formalization (chunk unit + frozen mode classifier + receding horizon) → Theory (N_eff bound) → Empirical (4 core tasks × 4 baselines × 3 seeds) → Differentiation

## Output Files

- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`
- Skeleton: `refine-logs/skeleton.md`
- Round 0 initial: `refine-logs/round-0-initial-proposal.md`
- Round 1 review + refinement: `refine-logs/round-{1-review,1-refinement}.md`
- Round 2 review + refinement: `refine-logs/round-{2-review,2-refinement}.md`
- Round 3 review + expanded: `refine-logs/round-{3-review,3-expanded}.md`
- Score history: `refine-logs/score-history.md`

## Score Evolution

| Round | PF | MS | CQ | FL | Feas | VF | VR | Overall | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8.5 | 6.5 | 7.0 | 8.0 | 6.5 | 6.0 | 8.0 | **7.45** | REVISE |
| 2 | 8.7 | 6.8 | 8.0 | 8.3 | 6.8 | 7.4 | 8.2 | **7.71** | REVISE |
| 3 | 8.8 | 7.4 | 7.8 | 8.5 | 7.2 | 8.0 | 8.2 | **7.93** | REVISE |

## Round-by-Round Review Record

| Round | Main Reviewer Concerns | Top-2 Issues Targeted | What Was Changed | Result |
|---|---|---|---|---|
| 1 | z internal inconsistency (CRITICAL); m_θ(z|s) undefined (CRITICAL); contribution sprawl (IMPORTANT) | (1) z FiLM-cond + frozen mode classifier; (2) contribution focus | frozen contrastive encoder + FiLM；删除 joint EM；ModeBench 降级；scope 缩小 Kitchen+AntMaze | partial — chunk vs single-action 未定 |
| 2 | action vs chunk unit (CRITICAL); rare-mode m_β 加权矛盾 (IMPORTANT); schedule (IMPORTANT) | (1) unit = chunk; (2) balanced replay + L_cons | unit chunk H=4; balanced uniform-z replay; conditional consistency 替 marginal TV; soft GMM; classifier-free dropout | yes — theorem 仍 N_z=N/K 是 statistical overclaim |
| 3 | N_eff theorem (CRITICAL); receding-horizon ambiguity (IMPORTANT); invalid (s,z) (IMPORTANT) | Phase 5.5 expansion 整合所有 surgical fix | N_eff 替 N_z; receding-horizon execution; balanced sampler over valid weighted (s,z); novelty positioning | yes (in expansion) |

## Final Proposal Snapshot (3-5 bullets)

1. **Failure-mode 反例首发**: closed-form + numeric 证明 global W₂ → 0 但 mass 漂移使 mode 消失（toy figure 作 paper Section 1）
2. **方法 = chunk diffusion + frozen mode classifier (Stage 0) + per-mode ICNN W₂ (Stage 2) + conditional consistency + receding-horizon inference**：每个组件都有 formula + pseudocode + 接口 + 超参范围；总新 trainable < 1M
3. **N_eff-aware finite-sample theorem**：明确 statistical error 与 optimization budget 解耦；rare mode 在 ρ_min/n_min 阈值下有 high-prob recall lower bound
4. **Empirical core**: 4 multi-modal D4RL tasks × 4 baselines (Diffusion-QL / global W₂-DiffPolicy / BDPO / ours) × 3 seeds = 48 runs ≈ 7-8 days dual-4090；stretch 4 tasks + appendix Adroit/NeoRL-2
5. **Inference overhead < 5%**：chunk 仅作 mode-conditioning + DSM 稳定化对象，receding-horizon execute first action only

## Method Evolution Highlights

1. **从 "global W₂" 到 "per-mode W₂ + frozen classifier + balanced valid sampler"**——把 mode preservation 从抽象目标变成可优化、可度量、可证明的形式化对象
2. **从 single-action 到 chunk diffusion (H=4)**——chunk 是 mode-meaningful unit；与 RSS 2023 / NeurIPS 2025 chunk diffusion 一致
3. **从 m_β-weighted W₂ 到 balanced replay + N_eff theorem**——让 rare mode 真获训练信号，但理论上诚实区分 optimization 与 statistical sample size
4. **从 D_TV(m_θ, m_β) marginal TV 到 conditional consistency loss**——直接、易估计、易读
5. **从模糊 "apply a_chunk[0] or every-H replan" 到 receding-horizon only**——chunk 不是 open-loop plan

## Pushback / Drift Log

| Round | Reviewer Said | Author Response | Outcome |
|---|---|---|---|
| 1 | 把 πθ(a|s,z) 重写为 batch-stratified marginal constraint | 拒绝——会削弱定理可证明性，retreat 到 batch-level | rejected |
| 1 | 用 LLM classifier 替 GMM | 拒绝——引入语义噪声 + 与 D4RL 几何脱节 + 杀鸡用牛刀 | rejected |
| 1 | 接 Q-cost OT 增 novelty | 拒绝——drift；与 OTPR/P16 重叠 | rejected |
| 2 | single-action policy + 定理改 action-mode recall | 拒绝——chunk 才是 mode-meaningful unit；与 Diffusion Policy by Chi RSS 2023 / Mixed-Density Diffuser P07 一致 | rejected |
| 2 | Adroit/NeoRL-2 重新进主 claim | 严格拒绝——确认严格 appendix only | rejected |
| 3 | 把 N_z = N/K 当 statistical sample size | 接受——改为 N_eff(z)，并诚实在 paper 标注 balanced replay = optim ≠ statistics | accepted |
| 3 | every-H replan 还是 receding-horizon | 接受——commit receding-horizon only | accepted |
| 3 | invalid (s,z) pair | 接受——balanced sampler 限制在 N_eff(z) ≥ n_min 的 valid 模式上 | accepted |

## Remaining Weaknesses

1. **未达 READY (≥9) 阈值**：剩余 ~1.07 分主要在 Method Specificity (7.4) 与 Contribution Quality (7.8)；前者需要在 paper 写作时把 chunk DSM ↔ IQL critic 联训 schedule 细化；后者需要在 narrative 上进一步抑制 "K ICNNs" 视角
2. **N_eff 风险**：若 D4RL rare mode 在某些 task 上 N_eff < n_min，empirical Pareto 会无法构建——需 Stage 0 BIC + oracle 双 protocol 在数据上提前 verify。论文中诚实标注为 "low-multimodality regime" 即可不主张优势
3. **ICNN-OT conjugate stability**：高维 chunk 空间 (H=4 × d_a=18 ≈ 72 dim for Adroit) 上 inner-max 可能不稳；已建议 Sinkhorn warm-up，但若所有 ablation 都需 Sinkhorn 则 ICNN 的 discriminator-free 优势削弱
4. **W₂ 是 chunk-mode recall 而非 trajectory-mode 或 occupancy preservation**：scope 已诚实标注；occupancy-aware 留作 IDEA-03 follow-up

## Raw Reviewer Responses

<details>
<summary>Round 1 Review</summary>

详见 `round-1-review.md` 的结构化分析。Codex thread: `019e0d5d-3c03-79c0-9ec5-dfe562c2f248`。

Key verdict: REVISE @ 7.45/10. 两 CRITICAL: z 不一致 + m_θ undefined.

</details>

<details>
<summary>Round 2 Review</summary>

详见 `round-2-review.md`。

Key verdict: REVISE @ 7.71/10. 一 CRITICAL: action vs chunk unit. 一 IMPORTANT: rare-mode m_β-weighted training signal contradicts core claim.

</details>

<details>
<summary>Round 3 Review</summary>

详见 `round-3-review.md`。

Key verdict: REVISE @ 7.93/10. 4 surgical fixes integrated in Phase 5.5 expansion. MAX_ROUNDS reached → automatic convergence rule applied. 用 Round 2 refinement (7.93) 作 FINAL base + Round 3 surgical fixes via expansion.

</details>

## Next Steps

- **状态**: 提案 REVISE (not READY)；剩余 ~1.07 分需在 paper 写作阶段补强 (Method Specificity 与 Contribution Quality 细节)
- **建议**:
  1. 进入 experiment-plan 阶段：固定 Stage 0 BIC + oracle 双 protocol 验证 D4RL multi-modal task 的 N_eff 充分性；若任 task N_eff < n_min，提前在 paper 中标注
  2. 实施 Stage 0：复现 Diffusion-QL chunk-variant + 训练 frozen f_ω 与 GMM；t-SNE/UMAP 可视化作 paper Section 1 figure
  3. Round 4 review (可选)：在 paper draft 写好后再做一轮 review 看是否能达 READY
  4. 与 IDEA-10 的 sibling theory paper 协调：IDEA-10 的 "two-mode continuous-control MDP" 解析结果可作为本论文 Section 1 的 motivating example，IDEA-10 standalone short paper 可同时投 NeurIPS workshop / theory track
