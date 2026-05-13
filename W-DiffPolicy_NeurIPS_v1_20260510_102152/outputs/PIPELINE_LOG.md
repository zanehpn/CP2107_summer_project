# Pipeline Log — W-DiffPolicy (NeurIPS)

**研究方向**: W-DiffPolicy: Wasserstein-Regularized Diffusion Policies for Multi-Modal Offline Reinforcement Learning
**目标会议**: NeurIPS
**启动时间**: 2026-05-09 23:25 UTC
**Reference Design**: /Users/zpy/LLM_project/idea_paper/boardSearch/outputs/archive_2026-05-09_w-diffpolicy-neurips_done/DIRECTION_REPORT_W-DiffPolicy.md
**REVIEWER_MODEL**: gpt-5.5
**REFINE_TOP_N**: 2

---

## 自动化决策日志

### Checkpoint 1 — Literature Survey (2026-05-09 23:35 UTC)

- **Status**: ✅ COMPLETED
- **Papers analyzed**: 29 (across 6 themes)
- **Sources**: web (arXiv, OpenReview, NeurIPS/ICLR/ICML proceedings) + boardSearch 历史档案
- **关键威胁论文**:
  - **Q-DOT (P14, RLC 2025, 2507.10843)** — W₂+ICNN+IQL，技术基底相同，最大 incumbent
  - **BWD-IQL (P15, ICLR 2026, 2510.12638)** — Bellman-Wasserstein + IQL
  - **SWFP (P17, Oct 2025, 2510.15388)** — W₂+JKO+flow policy (online，非 offline diffusion)
  - **BDPO (P04, ICML 2025)** — KL 严格化 diffusion policy 直接竞品
  - **RACTD (P06, 2506.07822)** — consistency distillation 路线占位（与训练时正则正交）
- **关键理论支撑**: P19 (KL is Designed to Mode Collapse, Oct 2025) 给出 KL 正则 mode-collapse 形式化定理，但仅在 LLM 上验证；W-DiffPolicy 可同时验证 KL 失效 + W 修复
- **核心 GAP**:
  - G1 (HIGH): W₂-OT 正则 × diffusion policy training 完全空白
  - G2 (HIGH): OT 正则的 mode-preservation 形式化定理缺失
  - G5 (HIGH): W₂-based distribution-shift bound for diffusion policy 缺失
  - G9 (HIGH): NeoRL-2 上 OT-regularized 方法迁移性未测
- **机会窗口**: ~6-8 周 (NeurIPS 2026 deadline)；最大威胁是 Q-DOT 作者延伸到 diffusion，需抢先
- **输出文件**: `outputs/LANDSCAPE.md`, `outputs/LANDSCAPE.json`
- **Decision**: 继续 Phase 2 (Idea Generation)

### Checkpoint 2 — Idea Generation (2026-05-09 23:55 UTC)

- **Status**: ✅ COMPLETED
- **Phase 2a Critique**: 13 个 critique 跨 4 维度 (Unverified / Incorrect Generalization / Experimental Flaw / Cross-Domain Misfit)；元主题 = "behavior density / diffusion reverse density / induced occupancy 三者被错误互换"
- **Phase 2b Ideas**: 11 个 idea 全部 anchor 到 critique
- **Codex thread ID**: `019e0d5d-3c03-79c0-9ec5-dfe562c2f248`
- **筛选结果**: 11 → 10 (淘汰 IDEA-05 budget-normalized audit, He 11/20 < threshold 12)
- **Top 6 (recommend for screening)**:
  - **Rank 1**: IDEA-01 Mode-Conditional W₂ DiffPolicy (17/20, MEDIUM, 5-7w) — 直接 sharpen v1 多模态保留
  - **Rank 2**: IDEA-06 Value-Conditioned Wasserstein (17/20, MEDIUM, 5-7w) — Q 嵌入 transport cost
  - **Rank 3**: IDEA-09 NeoRL-2 Transfer Stress Test (17/20, HIGH, 7-8w) — cross-domain real-world
  - **Rank 4**: IDEA-10 KL-Mode Collapse in Continuous Control (17/20, MEDIUM, 4-6w) — P19 推到控制
  - **Rank 5**: IDEA-02 Reverse-Path != Terminal (16/20, LOW, 3-4w) — BDPO critique
  - **Rank 6**: IDEA-03 Occupancy-Aware W₂ (16/20, HIGH, 7-8w) — induced MDP occupancy
- **Sibling alternates (rank 7-10)**: IDEA-07, 11, 04, 08
- **关键发现**: IDEA-01 + IDEA-06 与作者 v1 W-DiffPolicy 设计高度对齐，分别从 "保哪些 mode" 和 "保的方向" 完美互补
- **输出文件**: `outputs/CRITICAL_ANALYSIS.md`, `outputs/IDEAS_RAW.md`, `outputs/IDEAS_FILTERED.md`
- **Decision**: 继续 Phase 3 (Multi-Dimensional Screening) 在 Top 4 (IDEA-01, 06, 09, 10) 上

### Checkpoint 3 — Idea Screening (2026-05-10 00:15 UTC)

- **Status**: ✅ COMPLETED
- **Modules**: A (Novelty + Web Cross-Verify), B (NeurIPS 3-Reviewer + Meta), C (5-Dim Strategic Fit)
- **关键 Module A 发现**: **OTPR (2502.12631, Feb 2025)** 是 lit-survey 阶段漏掉的关键威胁——Q-function as transport cost + diffusion policy 已存在（online RL fine-tuning，非 offline）。**LANDSCAPE.json 应在后续 refine 阶段补加此论文**。
- **Composite 排名**:
  - **Rank 1**: IDEA-10 KL-Mode Collapse in Continuous Control — **7.64/10 PROCEED** (Meta Accept)
  - **Rank 2**: IDEA-01 Mode-Conditional W₂ Diffusion Policy — **7.23/10 PROCEED** (Meta Weak Accept)
  - Rank 3: IDEA-09 NeoRL-2 Stress Test — 5.55 CAUTION (Meta Weak Reject)
  - Rank 4: IDEA-06 Value-Conditioned W₂ — 5.23 CAUTION (Meta Weak Reject)
- **关键策略决策**: 推荐 IDEA-01 + IDEA-10 进入 refinement；IDEA-06/09 整合为 IDEA-01 的 ablation/实验段。即 "主论文 = IDEA-01 (mode-conditional W₂ + 内嵌 IDEA-10 的 threshold theorem) + sibling short paper = IDEA-10 standalone"
- **输出文件**: `outputs/SCREENING_REPORT.md`, `outputs/SCREENING_RANKED.md`
- **Decision**: 继续 Phase 4 (Deep Refinement) on top 2 (IDEA-01, IDEA-10)

### Checkpoint 4a — Refinement of IDEA-01 (Main Paper) (2026-05-10)

- **Status**: ✅ COMPLETED (REVISE verdict; MAX_ROUNDS=3 reached without READY)
- **Score Trajectory**: Round 1 (7.45) → Round 2 (7.71) → Round 3 (7.93) — improving, but did not reach SCORE_THRESHOLD=9
- **Final verdict**: REVISE — 用 Round 2 refinement 作 FINAL_PROPOSAL.md base + Round 3 surgical fixes 在 Phase 5.5 expansion 整合
- **Anchor status**: ✅ preserved
- **Focus status**: ✅ tight (single dominant contribution)
- **Modernity status**: ✅ appropriately frontier-aware
- **Skeleton status**: 全 5 步覆盖
- **Phase 5.5 Expansion**: 8 sections expanded (full formula + pseudocode + interfaces + hyperparam ranges); 4 Round 3 surgical fixes integrated (N_eff theorem, receding-horizon, valid (s,z) sampler, novelty positioning)
- **New Theory-Experiment flags from expansion**: 0 NOT FEASIBLE; 3 new theoretical claims absorbed into existing experimental budget
- **Key method upgrades across rounds**:
  1. z 改为 inference-time FiLM 条件 + frozen mode classifier (R1)
  2. Unit 统一到 action chunk H=4 (R2)
  3. Balanced mode replay + conditional consistency loss + soft GMM (R2)
  4. N_eff-aware theorem + receding-horizon execution + valid (s,z) sampler (R3 expansion)
- **Remaining concerns**:
  1. Score 7.93 < READY 阈值 9 — 剩余 ~1.07 主要 Method Specificity (7.4) 与 Contribution Quality (7.8)
  2. N_eff < n_min 风险需 Stage 0 双 protocol 提前 verify
  3. ICNN-OT conjugate stability on 高维 chunk space
- **Output files**:
  - `refine-logs/skeleton.md`
  - `refine-logs/round-{0..3}-*.md`
  - `refine-logs/score-history.md`
  - `refine-logs/FINAL_PROPOSAL.md`
  - `refine-logs/REVIEW_SUMMARY.md`
  - `refine-logs/REFINEMENT_REPORT.md`
- **Decision**: 继续 Phase 4b (refine IDEA-10 sibling theory paper)

### Checkpoint 4b — Refinement of IDEA-10 (Sibling Theory) (2026-05-10)

- **Status**: ✅ COMPLETED (1 round condensed review + surgical theory fixes integrated)
- **Initial Score**: 7.05/10 (REVISE)
- **Round 1 critique**: 3 surgical theory bugs (ΔQ sign vs collapse direction; KL convention 不一致; W₂ threshold 应是 bang-bang 不是 log)
- **Fixes integrated** in FINAL_PROPOSAL.md:
  1. 重新 frame collapse: rare mode mass < ε regardless of value sign；P19 LM 的 collapse 对应 control 中 ΔQ << 0 + small τ regime
  2. 统一 reverse KL convention `D_KL(π || β)`；π* ∝ β · exp(Q/τ)
  3. W₂ threshold 改为 bang-bang `ΔQ ≷ α Δ_a²` + dead zone `|ΔQ| ≤ α Δ_a²` 内 rare mode 完全保住
  4. Theorem hierarchy: T1 KL closed-form, T2 W₂ bang-bang, T3 Gaussian smoothing corollary
  5. 缩 D4RL scope: 54 runs sanity check + synthetic phase diagram 主 figure
- **关键 theoretical finding**: KL 不会 collapse rare *high-value* mode；它 collapse 的是 rare *low-value* mode 与 Q-guidance 误排序的 rare mode。这与 W-DiffPolicy main paper 的 mode preservation argument 互补
- **Sibling positioning**: 不引入新方法；为 W-DiffPolicy 等 W-OT 类方法提供 motivation 何时合理（Δ_a 大 + |ΔQ| 小 regime）何时不合理（Δ_a 小 或 |ΔQ| 大）
- **Output files**:
  - `refine-logs/idea-10-sibling/skeleton.md`
  - `refine-logs/idea-10-sibling/round-1-review.md`
  - `refine-logs/idea-10-sibling/FINAL_PROPOSAL.md`
- **Decision**: 继续 Phase 5 (Final Aggregated Report)

### Checkpoint 5 — Final Aggregated Report (2026-05-10 01:15 UTC)

- **Status**: ✅ COMPLETED — Pipeline 全自动执行结束
- **总耗时**: ~2 小时（lit-survey 10 分钟 + idea-gen 20 分钟 + idea-screen 20 分钟 + idea-refine 主 40 分钟 + sibling 10 分钟 + final report 10 分钟）
- **总产出**:
  - 1 主论文 FINAL_PROPOSAL（IDEA-01 Mode-Conditional W-DiffPolicy，3 轮 refine 后 7.93/10 REVISE）
  - 1 sibling theory FINAL_PROPOSAL（IDEA-10 KL-Mode Collapse Continuous Control，1 轮 condensed refine + theory bugs fixed）
  - 完整文献 landscape (29 papers, 6 themes, 12 gaps)
  - 完整 critical analysis (13 critiques) + 11 anchored ideas + screening reports
  - 全 pipeline checkpoint 自动化决策日志
- **关键里程碑**: OTPR (2502.12631) 在 idea-screen 阶段被发现，倒逼 IDEA-01 novelty positioning 重心转向 mode-preservation
- **输出文件**: `outputs/IDEA_DISCOVERY_REPORT.md` (final)
- **Decision**: Pipeline 完成；下一步 = IDEA-01 Stage 0 实施 + IDEA-10 理论 derivation 并行启动

