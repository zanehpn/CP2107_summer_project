# Screening Results: Ranked Ideas

**Direction**: W-DiffPolicy: Wasserstein-Regularized Diffusion Policies for Multi-Modal Offline RL
**Venue**: NeurIPS 2026
**Date**: 2026-05-09
**Ideas screened**: 4

## Rankings

| Rank | Idea | Novelty | Venue Score | Strategic | Feasibility | Composite | Recommendation |
|------|------|---------|-------------|-----------|-------------|-----------|----------------|
| 1 | IDEA-10 KL-Mode Collapse in Continuous Control | 8.0 | 7.3 | 7.4 | 8.0 | **7.64** | **PROCEED** |
| 2 | IDEA-01 Mode-Conditional W₂ Diffusion Policy | 8.0 | 6.7 | 7.4 | 7.0 | **7.23** | **PROCEED** |
| 3 | IDEA-09 NeoRL-2 OT Stress Test | 6.5 | 4.7 | 6.4 | 5.0 | **5.55** | CAUTION |
| 4 | IDEA-06 Value-Conditioned Wasserstein | 5.5 | 3.7 | 5.8 | 7.0 | **5.23** | CAUTION |

---

## Rank 1: IDEA-10 KL-Mode Collapse in Continuous Control — **PROCEED**

### Module A: Novelty
- **Score**: 8/10
- **Key differentiator**: 把 P19 (KL-mode-collapse, LM-only) 推到 state-conditioned continuous-control MDP；解析 KL/W₂/score-BC/Q-guidance 在 two-mode 下的 mass allocation 阈值
- **Closest prior work**: P19 KL-Mode-Collapse (LM only, fund/mod overlap)；2602.02250 LQG control (sup overlap)

### Module B: NeurIPS Simulation
- Reviewer 1 (Empiricist): **Weak Accept** — toy + Kitchen/AntMaze mode-controlled splits；担心若只 toy 影响有限
- Reviewer 2 (Innovator): **Accept** — 高概念清晰，纠正错误类比，正负结果均可发表
- Reviewer 3 (Rigorist): **Accept** — threshold theorem 形式好，可证明可解释可实验验证
- Meta-review: **Accept**
- **Top risk**: theorem 范围太窄被判 toy；score-BC vs KL 等价关系在 diffusion 下要严谨

### Module C: Strategic Fit
- Longevity: 9/10 — 理论论文老化慢；KL/W₂ 在 RL 是基础问题
- Roadmap Viability: 8/10 — Paper1 theorem → Paper2 empirical validation → Paper3 其他 policy class
- Application Grounding: 5/10 — 理论性，间接应用
- Execution Uniqueness: 7/10 — 作者 v1 阅读 + P19 深度
- Iteration Readiness: 8/10 — toy MDP 解析快速反馈

---

## Rank 2: IDEA-01 Mode-Conditional W₂ Diffusion Policy — **PROCEED**

### Module A: Novelty
- **Score**: 8/10
- **Key differentiator**: OT 从 global behavior distance 改为 mode-conditional safety constraint——不是 "学一个 latent"，而是证明/测量每个 value-relevant mode 是否被保住
- **Closest prior work**: LOM (单 mode 选择，方向相反，mod)；Q-DOT (IQL global W₂，mod)；Latent Diffusion ORL (无 OT，mod)

### Module B: NeurIPS Simulation
- Reviewer 1 (Empiricist): **Weak Accept** — 补 return-only 评估盲点；担心 latent z 学习不稳
- Reviewer 2 (Innovator): **Accept** — 把 diffusion ORL+OT+多模态诊断三线合并成新问题定义
- Reviewer 3 (Rigorist): **Weak Accept** — Theorem scaffold 有方向；要求 finite-sample bound 含 ICNN error
- Meta-review: **Weak Accept**
- **Top risk**: mode extraction 不稳；theorem 与 ICNN 实现脱节；return 提升不明显

### Module C: Strategic Fit
- Longevity: 8/10 — 多模态保留是 RL/IL 根本问题
- Roadmap Viability: 8/10 — method+theorem → mode-aware distillation → real-robot transfer
- Application Grounding: 7/10 — Kitchen/AntMaze + NeoRL-2 + 机器人 manipulation
- Execution Uniqueness: 7/10 — 作者 v1+ICNN+硬件就绪；U-Tokyo 直接竞争但 framing 更清晰
- Iteration Readiness: 7/10 — Synthetic mixture <1 周；Kitchen 单卡数小时

---

## Rank 3: IDEA-09 NeoRL-2 OT Stress Test — **CAUTION**

### Module A: Novelty: 6.5/10
### Module B: NeurIPS Simulation: Meta **Weak Reject** — Empiricist Weak Accept；Innovator/Rigorist Weak Reject
### Module C: Strategic: 6.4/10
### **Top risk**: 变成 "我们在 NeoRL-2 上跑了很多 baseline" 工程报告
### **Action**: 不作主线；可作 IDEA-01 的 cross-domain 实验段或 v2 follow-up

---

## Rank 4: IDEA-06 Value-Conditioned Wasserstein — **CAUTION**

### Module A: Novelty: 5.5/10
**关键发现**: OTPR (2502.12631, Feb 2025) 已用 Q-cost OT for diffusion + RL fine-tuning；P16 Rethinking OT 已用 Maximin Q-cost。Q-cost OT 不再 novel
### Module B: NeurIPS Simulation: Meta **Weak Reject** — 三 reviewer 一致负面
### Module C: Strategic: 5.8/10
### **Top risk**: 被判定为 OTPR/P16 的 obvious variant
### **Action**: 不独立成文；合并入 IDEA-01 作 ablation 章节 ("Q-aware ground cost vs Euclidean")

---

## Next Steps

### For PROCEED Ideas (REFINE_TOP_N=2):
- **IDEA-10** → 进入 `/idea-refine` 做深度精炼（理论 short paper 路线）
- **IDEA-01** → 进入 `/idea-refine` 做深度精炼（主论文方向，匹配 v1 W-DiffPolicy 设计）

### Strategic Composition Recommendation:
1. **主论文** = IDEA-01 (Mode-Conditional W₂ Diffusion Policy)
   - Theorem section 内嵌 IDEA-10 的 threshold theorem（作为 motivation）
   - Ablation section 包含 IDEA-06 的 value-conditioned cost variant
   - Cross-domain experiment section 选 1-2 个 NeoRL-2 task 验证 (IDEA-09 缩水版)
2. **Sibling short paper** (可选) = IDEA-10 standalone (KL-collapse threshold theory)
   - 若主论文写得快，可在 NeurIPS deadline 前同时投 short paper / workshop
   - 若时间紧，IDEA-10 完全融入 IDEA-01 的 motivation section

### For CAUTION Ideas:
- IDEA-06 / IDEA-09 不独立投稿；价值在于补强 IDEA-01 主论文的 ablation 与 cross-domain 实验

---

## 与 v1 W-DiffPolicy 设计的对比

作者 v1 W-DiffPolicy DIRECTION_REPORT 设计：W₂ + ICNN + diffusion + KL→W₂ + mode-preservation theorem + D4RL Kitchen/AntMaze + NeoRL-2 cross-domain。

经 screening 后建议的精炼版（基于 IDEA-01 + IDEA-10 融合）：
- **保留**: W₂ + ICNN + diffusion 主框架
- **强化**: 把 "KL→W₂ 替换" 升级为 "Mode-Conditional W₂ + mode-mass regularizer"——回应 reviewer 的 "global W₂ 也会平均化 mode" 质疑
- **新增 motivation theorem (来自 IDEA-10)**: KL collapse 在 offline diffusion control 中何时真发生（threshold law），W₂ 在何条件下严格优于 KL
- **保留**: NeoRL-2 cross-domain，但缩水到 1-2 个 task 作 transfer 验证
- **避免**: 不要主打 "Q-cost OT" 角度（OTPR 已占）；改为 mode-preservation 角度
