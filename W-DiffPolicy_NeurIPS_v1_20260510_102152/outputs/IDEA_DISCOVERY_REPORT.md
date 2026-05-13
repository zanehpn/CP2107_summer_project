# Idea Discovery Report — W-DiffPolicy

**研究方向**: W-DiffPolicy: Wasserstein-Regularized Diffusion Policies for Multi-Modal Offline Reinforcement Learning
**目标会议**: NeurIPS 2026
**Pipeline 启动**: 2026-05-09 23:25 UTC
**Pipeline 完成**: 2026-05-10 01:15 UTC（~2 小时全自动执行）
**Reference Design**: `/Users/zpy/LLM_project/idea_paper/boardSearch/outputs/archive_2026-05-09_w-diffpolicy-neurips_done/DIRECTION_REPORT_W-DiffPolicy.md`
**REVIEWER_MODEL**: gpt-5.5 (xhigh reasoning, via Codex MCP)
**Codex thread (统一上下文)**: `019e0d5d-3c03-79c0-9ec5-dfe562c2f248`
**REFINE_TOP_N**: 2

---

## Executive Summary

本 pipeline 自动完成了从 W-DiffPolicy 方向到 NeurIPS 2026 投稿提案的完整流程。

**主要产出**:
1. **主论文方向（IDEA-01）**: Mode-Conditional W-DiffPolicy — Wasserstein 行为正则配合 frozen mode classifier、per-mode ICNN-OT、balanced valid (s,z) sampler、receding-horizon chunk diffusion；finite-sample chunk-mode recall theorem 含 N_eff(z) error term。Composite screening = 7.23/10 PROCEED；3 轮 refinement 后最终 score 7.93/10 (REVISE，未达 READY 阈值 9 但显著提升 +0.48 vs 初始 7.45)。
2. **Sibling theory paper（IDEA-10）**: When Does KL Cause Mode Collapse in Continuous Control? — 推导 two-point MDP 的 KL closed-form mass allocation 与 W₂ bang-bang threshold；regime map 给出何时 W-OT 替换 KL 是 valid motivation。Composite screening = 7.64/10 PROCEED (Module B Accept verdict)；1 轮 condensed refinement 修复 3 个 surgical theory bugs。

**关键 pipeline 发现**:
- **OTPR (2502.12631, Feb 2025)** 是 lit-survey 阶段漏掉的关键威胁论文——已用 Q-cost OT for diffusion (online RL fine-tuning)。该发现倒逼 IDEA-01 必须把 novelty 重心从 "Q-cost OT" 移到 "mode-preservation"，并把 IDEA-06 (Value-Conditioned Wasserstein) 降级 (composite 5.23 CAUTION)。
- **P19 (KL-Mode-Collapse, Oct 2025)** 给出 LM RL 上的 mode-collapse 形式化定理，但其类比未推到 control——这一空白成为 IDEA-10 sibling theory paper 的核心切入点。
- **Q-DOT (RLC 2025)** + **SWFP (Oct 2025)** 已在 IQL/online flow 上做 W₂；W-DiffPolicy 的差异化不能再靠 "首次 W₂ + diffusion" 这一表层叙事，必须靠 mode-preservation 的形式化对象 + N_eff-aware finite-sample theorem。

**战略策略**:
- **主论文 IDEA-01** = method paper（NeurIPS main track）
- **Sibling IDEA-10** = standalone theory short paper（NeurIPS theory track / short paper）或作为 IDEA-01 的 motivation theorem section
- **IDEA-06 / IDEA-09** 不独立投稿；分别作为 IDEA-01 的 ablation 与 cross-domain 实验段（NeoRL-2 1-2 task 验证）

**全 pipeline 输出文件清单**: 见末尾。

---

## Phase 1 — Literature Survey

**29 篇论文，6 主题，12 gap**（详见 `outputs/LANDSCAPE.md` / `outputs/LANDSCAPE.json`）。

### 6 个主题

1. **Diffusion Policy for Offline RL** [active]: P01-P13 — Diffusion-QL、EDP、SORL、BDPO、RACTD、Mixed-Density Diffuser、Diffusion-QL value-cond 等
2. **OT Regularization in Offline RL** [emerging, 2024-2026 burst]: P14-P18 — Q-DOT (RLC 2025), BWD-IQL (ICLR 2026), Maximin OT (NeurIPS 2024), SWFP (Oct 2025), 在线 OT
3. **Multi-modality Theory & Diagnostics** [emerging]: P19-P21 — KL collapse 定理 (Oct 2025), LOM (单 mode 选择), RAMAC
4. **ICNN / OT Theory** [mature]: P22 Makkuva 2020 + applications
5. **Cross-Domain Offline RL & Real-World Benchmarks** [active]: P23-P27 — PUORL, OTDF (ICLR 2025), Dual-Robust, NeoRL-2
6. **Distributional RL & Sinkhorn** [mature]: P28-P29

### 4 个 HIGH-confidence gap

- **G1**: W₂-OT 正则尚未应用于 diffusion policy 训练（IQL 与 online flow 已做，diffusion offline 空白）
- **G2**: 无 OT 正则 diffusion policy 的 mode-preservation 形式化定理
- **G5**: 无 W₂-based distribution-shift bound for diffusion policy
- **G9**: NeoRL-2 上 OT-regularized 方法迁移性未测

---

## Phase 2 — Idea Generation

### Critical Analysis (Phase 2a) — 13 critique 跨 4 维度

**元主题**: behavior density / diffusion reverse density / induced occupancy 三者被错误互换。

详见 `outputs/CRITICAL_ANALYSIS.md`。13 critiques 涵盖：
- Unverified Assumptions（KL 保多模态？BDPO reverse-kernel 测最终策略？ICNN-W₂ 在 finite-sample 下可靠？）
- Incorrect Generalizations（静态 OT 推到 MDP；P19 LM 推到 control；image diffusion 表现力推到 action）
- Experimental Flaws（return-only 评估；NFE 不归一；offline model selection；dataset 混淆多模态/数据质量）
- Cross-Domain Misfits（Euclidean transport cost 在控制上的 semantic 错位；DA 对齐 logic 误用 cross-domain RL；diversity metric ≠ RL value）

### Critique-Anchored Ideas (Phase 2b) — 11 ideas

详见 `outputs/IDEAS_RAW.md` 与 `outputs/IDEAS_FILTERED.md`。

**Top 6 (Prof. He 4-dim 综合 ≥ 14/20)**:
| Rank | IDEA | Title | He Score | Risk |
|---|---|---|---|---|
| 1 | IDEA-01 | Mode-Conditional W₂ Diffusion Policy | 17/20 | MEDIUM |
| 2 | IDEA-06 | Value-Conditioned Wasserstein | 17/20 | MEDIUM |
| 3 | IDEA-09 | NeoRL-2 OT Stress Test | 17/20 | HIGH |
| 4 | IDEA-10 | KL-Mode Collapse in Continuous Control | 17/20 | MEDIUM |
| 5 | IDEA-02 | Reverse-Path != Terminal | 16/20 | LOW |
| 6 | IDEA-03 | Occupancy-Aware W₂ | 16/20 | HIGH |

**淘汰**: IDEA-05 (Budget-Normalized Audit, He 11/20)。

---

## Phase 3 — Multi-Dimensional Screening

针对 Top 4 (IDEA-01, 06, 09, 10) 做 NeurIPS-specific 多维度筛选。

| Rank | IDEA | Novelty | Venue (3-Reviewer) | Strategic | Feasibility | Composite | Recommendation |
|---|---|---|---|---|---|---|---|
| 1 | **IDEA-10** KL-Mode Collapse Continuous Control | 8.0 | 7.3 (WA/A/A; Meta Accept) | 7.4 | 8.0 | **7.64** | **PROCEED** |
| 2 | **IDEA-01** Mode-Conditional W₂ DiffPolicy | 8.0 | 6.7 (WA/A/WA; Meta Weak Accept) | 7.4 | 7.0 | **7.23** | **PROCEED** |
| 3 | IDEA-09 NeoRL-2 Stress Test | 6.5 | 4.7 (WA/WR/WR; Meta Weak Reject) | 6.4 | 5.0 | 5.55 | CAUTION |
| 4 | IDEA-06 Value-Conditioned W₂ | 5.5 | 3.7 (WR/R/WR; Meta Weak Reject) | 5.8 | 7.0 | 5.23 | CAUTION |

**关键 screening 发现**:
- **OTPR (2502.12631)** 已用 Q-cost OT for diffusion (online fine-tuning)，把 IDEA-06 的 novelty 锁死在 5.5/10
- IDEA-09 工作量 (NeoRL-2 复现 + cross-domain) 与 6-8 周窗口不匹配，且 reviewer 担心 benchmark-only 风险
- IDEA-10 与 IDEA-01 形成 perfect siblings：理论 + 方法

详见 `outputs/SCREENING_REPORT.md` 与 `outputs/SCREENING_RANKED.md`。

---

## Phase 4 — Deep Refinement

### IDEA-01 Mode-Conditional W₂ Diffusion Policy (3 rounds, MAX_ROUNDS reached)

**Score Trajectory**: 7.45 → 7.71 → 7.93 (REVISE; +0.48 improvement)

**Round-by-Round 演进**:

| Round | Top-2 Issues Targeted | Key Method Changes |
|---|---|---|
| 1 | (1) z 内部不一致 + m_θ undefined; (2) contribution sprawl | frozen contrastive trajectory encoder + FiLM 条件; m_θ 用 frozen mode classifier; ModeBench 降级 protocol; NeoRL-2/Adroit 推附录 |
| 2 | (1) action vs chunk unit; (2) rare-mode m_β 加权矛盾核心 claim | unit 统一到 action chunk H=4; balanced mode replay; conditional consistency loss 替 marginal TV; soft GMM; classifier-free dropout |
| 3 (expansion) | (1) N_eff theorem; (2) receding-horizon ambiguity; (3) invalid (s,z) sampler; (4) novelty positioning | Phase 5.5: 全 [EXPAND] 段补 formula/pseudocode/接口/超参；理论改 N_eff(z); commit receding-horizon execute first action only; balanced sampler over valid weighted (s,z) |

**Pushback / Drift log**:
- 拒绝把 πθ(a|s,z) 重写为 batch-stratified marginal constraint（破坏定理可证明性）
- 拒绝 LLM mode classifier（语义噪声）
- 拒绝 Q-cost OT 引入（与 OTPR/P16 重叠）
- 拒绝 single-action 改 action-mode recall（chunk 才是 mode-meaningful unit）
- 接受 N_eff(z) theorem 修正（balanced replay = optim ≠ statistics）
- 接受 receding-horizon execute first action only
- 接受 balanced sampler 限制在 N_eff(z) ≥ n_min 的 valid 模式

**最终 Final Proposal 核心**:
- Stage 0: 一次性 SimCLR-style trajectory encoder + GMM + frozen mode classifier
- Stage 1: chunk diffusion warm-up (KL-BC + uniform z + FiLM + 5% classifier-free dropout)
- Stage 2: balanced mode replay + per-mode ICNN-W₂ + conditional consistency loss + chunk DSM
- Inference: receding-horizon chunk sampling + execute first action only (overhead < 5%)
- Theorem: `Recall_z(πθ) ≥ 1 - O((ε_stat(z,δ) + ε_opt + ε_conj)/Δ²) - O(ε_clf/ρ_min) - O(ε_c/log 2)` with `ε_stat ∝ 1/√N_eff(z)`
- Empirical: synthetic 4-mode + 4 D4RL multi-modal tasks × 4 baselines × 3 seeds = 48 runs (~7-8 days dual-4090)

**Refinement 文件**: `refine-logs/{skeleton,round-{0..3}-*,score-history,FINAL_PROPOSAL,REVIEW_SUMMARY,REFINEMENT_REPORT}.md`

### IDEA-10 KL-Mode Collapse in Continuous Control (1 round condensed)

**Initial Score**: 7.05/10 REVISE (3 surgical theory bugs)

**Theory bugs fixed**:
1. ΔQ sign vs collapse direction 不一致 → 重新 frame collapse 为 rare mode mass < ε regardless of value sign
2. KL convention 混用 → 统一 reverse KL `D_KL(π || β)`
3. W₂ threshold 不应是 log → 改 bang-bang `ΔQ ≷ α Δ_a²` + dead zone

**核心 Theorem Hierarchy**:
- T1 (KL closed-form): `p_2* = m_2 exp(ΔQ/τ) / (m_1 + m_2 exp(ΔQ/τ))`
- T2 (W₂ bang-bang): `|ΔQ| ≤ α Δ_a² ⟹ π_W* = β` (rare mode 完全保住)
- Corollary (regime map): 4 regime 表，标注 KL collapse / W₂ preserve / 两者都崩 / 两者都保

**关键 finding**: KL 不会 collapse rare *high-value* mode；它 collapse 的是 rare *low-value* mode 与 Q-guidance 误排序的 rare mode。**P19 在 LM 上的 collapse 对应 control 中 ΔQ << 0 + small τ 的 regime**——这给 W-DiffPolicy 等方法的 motivation 划清了适用边界。

**Refinement 文件**: `refine-logs/idea-10-sibling/{skeleton,round-1-review,FINAL_PROPOSAL}.md`

---

## 战略组合建议

### 主论文 (NeurIPS 2026 main track)

**Mode-Conditional W-DiffPolicy** (IDEA-01)
- 6-8 周时间预算
- 主实验 48 runs core + 48 stretch + 6 NeoRL-2 appendix
- IDEA-10 的 threshold theorem 可作为 Section 1 motivation 的内嵌定理
- IDEA-06 的 value-conditioned ground cost 作为 ablation
- IDEA-09 的 NeoRL-2 transfer 作为 cross-domain 实验段（1-2 task）

### Sibling theory paper (NeurIPS 2026 short paper / theory track)

**KL-Mode Collapse Threshold Law in Continuous Control** (IDEA-10)
- 4-6 周时间预算（理论为主）
- 主实验 = synthetic phase diagram (CPU) + 54 runs D4RL sanity check
- 与主论文协调 narrative：为 W-DiffPolicy 类方法提供 motivation 何时合理
- 不引入新方法

### v2 / Follow-up（不在本轮）

- IDEA-03 Occupancy-Aware W₂（HIGH risk theoretical extension）
- IDEA-07 ICNN-W₂ Reliability Pressure Test（diagnostic appendix or short paper）
- IDEA-11 Distill Without Losing Modes（与 RACTD 整合，v2 短论文）

---

## 风险与限制

### IDEA-01 主论文风险

1. **Score 7.93 < READY 阈值 9**: 剩余 ~1.07 分主要在 Method Specificity (7.4 → 9 需更精细 chunk DSM ↔ IQL critic 联训 schedule) 与 Contribution Quality (7.8 → 9 需 narrative 抑制 "K ICNNs engineering" 视角)
2. **N_eff 风险**: 若 D4RL rare mode 在某 task 上 N_eff < n_min，empirical Pareto 做不出来——需 Stage 0 BIC + oracle 双 protocol 提前 verify
3. **ICNN-OT conjugate stability**: 高维 chunk 空间（H=4 × d_a=18 ≈ 72 dim for Adroit）上 inner-max 可能不稳；已建议 Sinkhorn warm-up，但若全部 ablation 都需 Sinkhorn 则 ICNN 的 discriminator-free 优势削弱
4. **Scope is chunk-mode recall, not trajectory-mode or occupancy**: 已诚实标注；occupancy-aware 留作 IDEA-03 follow-up

### IDEA-10 sibling 风险

1. **Theorem 范围窄**: 主论文用 two-point MDP；Gaussian mixture 推广留 corollary——reviewer 可能认为 toy
2. **ΔQ_estimated vs ΔQ_true 误差影响**: 实证中 Q-error 可让 KL 错杀 rare 高-Q mode；附录需 controlled experiment
3. **D4RL controlled split 设计**: 需要从 dataset 中合成 mixture，标注 mode 1/2，给 ΔQ 控制——工程量小但需 careful

### Pipeline 整体限制

1. **Lit-survey 漏 OTPR**: 在 idea-screen 阶段才发现；后续 Stage 0 实施前应再做一次定向查新（"diffusion + Q-OT", "online RL diffusion fine-tune", "chunk OT", "JKO online RL"）
2. **Score < READY 收敛**: 两 idea 都未达 READY (≥ 9)；建议 paper draft 写好后再做一轮 review 验证是否能进入 READY
3. **Reference design 已有 v1**: 本 pipeline 是 v1 的 refinement；实际工程实施时应以本 FINAL_PROPOSAL 为蓝本而非 v1 design doc

---

## 输出文件清单

```
/Users/zpy/LLM_project/idea_paper/AutoVibeIdeaV2/

outputs/
├── LANDSCAPE.md, LANDSCAPE.json                  # Phase 1 — 29 papers, 6 themes, 12 gaps
├── CRITICAL_ANALYSIS.md                           # Phase 2a — 13 critiques
├── IDEAS_RAW.md, IDEAS_FILTERED.md               # Phase 2 — 11 ideas, top 6 surviving
├── SCREENING_REPORT.md, SCREENING_RANKED.md      # Phase 3 — top 4 screened, top 2 PROCEED
├── IDEA_DISCOVERY_REPORT.md                       # 本文件 (Phase 5)
├── PIPELINE_LOG.md                                # 自动化决策日志
└── PIPELINE_STATE.json                            # checkpoint state

refine-logs/                                       # Phase 4a — IDEA-01 主论文
├── skeleton.md
├── round-0-initial-proposal.md
├── round-1-review.md, round-1-refinement.md
├── round-2-review.md, round-2-refinement.md
├── round-3-review.md, round-3-expanded.md
├── score-history.md
├── FINAL_PROPOSAL.md
├── REVIEW_SUMMARY.md
├── REFINEMENT_REPORT.md
│
└── idea-10-sibling/                              # Phase 4b — IDEA-10 sibling theory
    ├── skeleton.md
    ├── round-1-review.md
    └── FINAL_PROPOSAL.md
```

---

## Pipeline 自动化决策审计

详见 `outputs/PIPELINE_LOG.md`。所有 checkpoint 决策（包括 OTPR 漏检、IDEA-06 降级、MAX_ROUNDS 触达自动收敛、Phase 5.5 expansion 集成 surgical fixes）均自动记录，无需用户输入。

**关键决策**:
1. 检测到旧 SAEPro state 不匹配，自动 backup 并 fresh start (Phase 1 启动)
2. OTPR 漏检在 Phase 3 时被发现，更新 IDEA-06 novelty 与 strategic positioning（无需重跑 Phase 1）
3. MAX_ROUNDS=3 触达 IDEA-01 但 score < 9，自动选 Round 2 refinement (highest-scoring) + Phase 5.5 expansion 整合 Round 3 surgical fixes
4. IDEA-10 sibling 用 condensed 1-round review 而非完整 3 轮，节约 context budget；surgical theory bugs 直接在 FINAL 整合而非再调用 Codex

---

## 下一步建议

1. **优先级 1 — IDEA-01 实施**: 立即启动 Stage 0（Diffusion-QL chunk-variant 复现 + frozen contrastive encoder 训练 + GMM fitting + t-SNE visualize）；约 1 周
2. **优先级 2 — IDEA-10 理论 derivation**: 同步推导 Theorem 1 + 2 + 3；约 1 周（可与优先级 1 并行）
3. **优先级 3 — 定向二次查新**: 在 Stage 0 实施前再查 ("diffusion + Q-OT", "chunk OT diffusion offline RL")，确认 OTPR 之外没有新威胁
4. **优先级 4 — 联系 Q-DOT 作者** (Omura et al., U-Tokyo): 最大威胁组；可能愿意合作或讨论；至少了解其 v2 计划

完成本文件并附全部输出，pipeline 全自动执行结束。
