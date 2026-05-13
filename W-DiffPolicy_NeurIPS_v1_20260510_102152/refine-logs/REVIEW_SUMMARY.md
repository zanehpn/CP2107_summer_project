# Review Summary — Mode-Conditional W-DiffPolicy (IDEA-01)

**Problem**: 离线 RL 中 diffusion policy 的 KL/score-BC 行为正则在 multi-modal 上系统失效（rare value-relevant mode 被吞）
**Initial Approach**: W-DiffPolicy v1 — KL → W₂ replacement + mode-preservation theorem
**Date**: 2026-05-10
**Rounds**: 3 / MAX_ROUNDS=3
**Final Score**: 7.93 / 10
**Final Verdict**: REVISE (MAX_ROUNDS 触达 + 自动收敛规则)

## Problem Anchor (vintage)

- Bottom-line: rare value-relevant mode preservation in multi-modal offline RL
- Must-solve: per-mode mass not averaged or drifted to dominant mode
- Non-goals: 不解决 inference 加速；不重写 Q-learning；不做 online；不主张首个 Q-cost OT
- Constraints: 2× 4090; 6-8 weeks NeurIPS 2026; D4RL Kitchen/AntMaze (核心) + Adroit/NeoRL-2 (附录)

## Skeleton (final)

- State A: reviewer 相信 diffusion 自然保多模态，正则只是 distance 选择问题
- State B: reviewer 接受 global W₂ 也吞稀有 mode + per-mode 形式化对象的必要性
- Path: Diagnosis → Formalization (chunk + frozen classifier + receding horizon) → Theory (N_eff bound) → Empirical (4 core × 4 baselines × 3 seeds) → Differentiation

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | What This Round Simplified / Modernized | Top-2 Issues Targeted | Solved? | Remaining Risk |
|-------|-------------------------|------------------------------------------|-----------------------|---------|----------------|
| 1     | z 内部不一致 (CRITICAL)；m_θ(z|s) 未定义 (CRITICAL)；contribution sprawl (IMPORTANT)；schedule 乐观 | 引入 frozen contrastive trajectory encoder + FiLM；删除 joint EM；ModeBench 降级 protocol；NeoRL-2/Adroit 推附录 | (1) z FiLM + frozen mode classifier，(2) contribution focus 收紧 | partial | unit (action vs chunk) 仍模糊 |
| 2     | action vs chunk unit (CRITICAL)；rare-mode m_β 加权矛盾核心 claim (IMPORTANT)；schedule 乐观 (IMPORTANT) | 统一 unit 到 chunk H=4；balanced mode replay；conditional consistency loss 替 marginal TV；soft GMM；classifier-free dropout | (1) chunk unit 全套对齐，(2) balanced replay + L_cons | yes | theorem 用 N_z=N/K 是 statistical overclaim |
| 3     | N_eff theorem (CRITICAL)；receding-horizon ambiguity (IMPORTANT)；invalid (s,z) sampler (IMPORTANT)；novelty positioning (MINOR) | (Phase 5.5 expansion) 全 [EXPAND] 段补 formula/pseudocode/接口/超参；理论改 N_eff；commit receding-horizon；balanced sampler over valid weighted (s,z) | (1) N_eff 修正；(2) chunk 执行歧义解决 | yes (in expansion) | balanced replay 仍是 optim 而非 statistics——已诚实标注 |

## Score Evolution

| Round | PF | MS | CQ | FL | Feas | VF | VR | Overall | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8.5 | 6.5 | 7.0 | 8.0 | 6.5 | 6.0 | 8.0 | **7.45** | REVISE |
| 2 | 8.7 | 6.8 | 8.0 | 8.3 | 6.8 | 7.4 | 8.2 | **7.71** | REVISE |
| 3 | 8.8 | 7.4 | 7.8 | 8.5 | 7.2 | 8.0 | 8.2 | **7.93** | REVISE |

## Overall Evolution

- **Method 更具体**: 从 "用 latent z 与 per-mode W₂" → 完整 chunk-unit + frozen contrastive encoder + GMM + FiLM + ICNN semi-dual + balanced valid sampler + receding horizon，每步都有 formula + pseudocode + 接口 + 超参范围
- **Dominant contribution 更聚焦**: 从初期 sprawl (4-6 平行 claim) → 单一 "global W₂ failure + mode-conditional W₂ + recall theorem"；ModeBench / NeoRL-2 transfer / P19 推广 / BDPO critique 全部降级到 protocol/appendix/Discussion 单段
- **Unnecessary complexity 删除**: joint EM fine-tune of q_ξ → 全冻结；marginal D_TV → conditional consistency；m_β 加权 W₂ → balanced replay；硬 c_ω > 0.5 → soft posterior
- **Modernity 升级**: SimCLR-style contrastive trajectory encoder + FiLM + classifier-free dropout 都是现代但非 over-foundation；不 import LLM/Q-cost/distillation
- **Drift 避免**: 始终 anchored 在 rare value-relevant mode preservation；非主线 ideas (NeoRL-2 cross-domain, P19 control 推广) 控制在 appendix/Discussion 1 段
- **Skeleton gaps 填**: Step 2 (Formalization) 在 R1/R2 有 superficial 缺陷，R2 重写 unit + frozen classifier 后修复；Step 3 (Theory) 在 R3 review 后通过 N_eff 修正去除 vacuity

## Final Status

- **Anchor status**: ✅ preserved（始终是 multi-modal offline RL rare value-relevant mode preservation）
- **Focus status**: ✅ tight（一个 dominant contribution + 一个内嵌 evaluation protocol + 一个 stretch appendix）
- **Modernity status**: ✅ appropriately frontier-aware（contrastive encoder + FiLM + classifier-free dropout；不 over-engineer）
- **Skeleton completeness**: 全 5 步覆盖；Step 3 (Theory) 在 expansion pass 用 N_eff 修正后干净
- **Strongest parts of final method**:
  1. **Counter-example narrative**：global W₂ → 0 但 mass 漂移使 mode 消失；可用 closed-form 2-mode + numeric 验证
  2. **N_eff-aware theorem**：把 statistical error 与 optimization budget 解耦；rare mode 不被 ε vacuous
  3. **Receding-horizon chunk inference**：chunk 是 mode-conditioning + DSM 稳定化对象，不是 open-loop plan——避免 Q-improvement objective mismatch
  4. **Balanced valid (s,z) sampler**：只在 N_eff(z) ≥ n_min 的有效 mode 上正则
  5. **End-to-end implementation precision**：Stage 0/1/2 全 formula + pseudocode + 接口 + 超参范围
- **Remaining weaknesses**:
  1. 论文最终 verdict 仍 REVISE 而非 READY；READY 需要 overall ≥ 9 — 现 7.93。剩余的 ~ 1.07 分主要在 Method Specificity (7.4 → 9 还需 chunk DSM 与 IQL critic 联训的更精细 schedule) 与 Contribution Quality (7.8 → 9 还需在 paper 写作时进一步抑制 "K 个 ICNN engineering" 视角)
  2. N_eff 理论虽然 honest 标注 rare mode 的 unverified 区域，但若 D4RL 任务 rare mode 真 N_eff 全部 < n_min，empirical Pareto 会做不出来——这是 fundamental risk，需 BIC + oracle 双 protocol 在 Stage 0 提前检查
  3. ICNN-OT 的 conjugate inner-max 在高维 chunk 空间可能不稳——已建议 Sinkhorn warm-up，但若所有 ablation 都需 Sinkhorn 则 ICNN 的 discriminator-free 优势削弱
- **Pushback / drift log**:
  - Round 1: rejected reviewer's "stratified batch constraints" alternative（会破坏定理可证明性）
  - Round 2: rejected "single-action policy + 定理改 action-mode recall"（chunk 才是 mode-meaningful unit）
  - 全程 rejected: LLM mode classifier、Q-cost OT 引入、online RL fine-tuning（避开 OTPR 重叠 + 与 v1 anchor 冲突）

