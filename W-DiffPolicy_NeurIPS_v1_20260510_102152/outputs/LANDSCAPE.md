# Literature Landscape: W-DiffPolicy — Wasserstein-Regularized Diffusion Policies for Multi-Modal Offline RL

**Date**: 2026-05-09
**Papers analyzed**: 29
**Sources**: web (arXiv, OpenReview, NeurIPS/ICLR/ICML proceedings), boardSearch 历史档案
**目标会议**: NeurIPS 2026

---

## Executive Summary

围绕「将 Wasserstein/最优运输 (OT) 正则替换 KL 正则用于 diffusion policy 离线 RL」这一交叉方向，文献分布在三条平行线上：(i) **Diffusion policy for offline RL**（Diffusion-QL → EDP → BDPO → SORL → RACTD），核心问题是 expressive policy class，但仍以 KL 或行为克隆为主要正则器；(ii) **OT regularization for offline RL**（Q-DOT、BWD-IQL、Maximin OT、SWFP 在线 flow），核心证据是 Wasserstein 在 IQL/flow policy 上稳定且优于 KL，但**全部未涉及 diffusion policy training-time 正则**；(iii) **Multi-modality 理论**（KL-Mode-Collapse 定理 2510.20817、LOM、RAMAC）刚刚在 2025-2026 给出 KL 正则导致 mode 塌缩的形式化证明，**但尚无对应的 OT mode-preservation 定理**。

**最关键的交叉空白**: Q-DOT (RLC 2025) 与 BWD-IQL (ICLR 2026) 已分别将 W₂ via ICNN 和 Bellman-Wasserstein 用于 **IQL** (point-estimate policy)；而 BDPO (ICML 2025) 将 KL 严格扩展到 **diffusion policy** 的反向传播过程；却没有任何工作把 **W₂-OT 正则与 diffusion policy 训练**结合。这是本提案的「第一性创新点」。

**机会窗口**: 2510.20817 (KL is Designed to Mode Collapse, Oct 2025) 在 LLM RL 上首次给出 KL 正则的 mode-collapse 形式化定理，但**未推广到 control / offline RL**。Diffusion policy 的多模态优势 (D4RL Kitchen/AntMaze) 与 KL 正则的 mode-collapse 理论之间存在一个明显的「**理论与方法的失配**」——这正是 W-DiffPolicy 想填补的。

**威胁评估**: (a) Q-DOT 作者 (Omura et al., U-Tokyo) 极有可能延伸到 diffusion，但目前公开的 follow-up 集中在 IQL；(b) RACTD 占据 reward-aware consistency 路线，但与训练时正则正交，可叠加；(c) SWFP (2510.15388, Oct 2025) 在 *online* flow policy 上做了 W₂-JKO 正则，但未触及 offline + diffusion + multi-modal 这条线。整体看，**6-8 周窗口仍开放**，关键是抢在 NeurIPS 2026 deadline 之前完成 v1。

---

## Paper Table

| ID | Paper | Authors | Year | Venue | Method | Key Result | Relevance | Source |
|----|-------|---------|------|-------|--------|------------|-----------|--------|
| P01 | Diffusion Policies as an Expressive Policy Class for Offline RL (Diffusion-QL) | Wang, Hunt, Zhou | 2023 | ICLR 2023 | KL-regularized diffusion policy + Q-guided sampling | 首篇将 diffusion 用于 offline RL 的工作；D4RL SOTA | 直接 baseline + 主要竞品 (KL 正则) | web |
| P02 | Efficient Diffusion Policies (EDP) | Kang et al. | 2023 | NeurIPS 2023 | Action approximation + 兼容 TD3/CRR/IQL | 训练 5d→5h，多算法兼容 | baseline; 训练加速可借鉴 | web |
| P03 | SORL: Scaling Offline RL via Shortcut Models | Espinosa-Dice et al. | 2025 | NeurIPS 2025 | Shortcut models + Q-verifier 推理时间扩展 | antmaze-large/antsoccer SOTA | 当前 diffusion-family SOTA baseline | web |
| P04 | BDPO: Behavior-Regularized Diffusion Policy Optimization | Gao, Wu, Cao et al. | 2025 | ICML 2025 | KL 解析地按 reverse-time transition kernel 累加 | D4RL 全套 SOTA on diffusion | 直接竞品 (KL 严格化路线) | web |
| P05 | ReFORM: Reflected Flows On-support Offline RL | Anonymous | 2025 | OpenReview | Reflected flow 保证 on-support | flow-based offline RL | flow 路线对照 | boardSearch |
| P06 | RACTD: Accelerating Diffusion Planners via Reward-Aware Consistency Distillation | Daza et al. (CMU) | 2025 | arXiv 2506.07822 | Reward-aware consistency distillation; 142× speedup | +8.7% over SOTA, 1-NFE | 推理加速威胁；与 W-正则正交 | web |
| P07 | Mixed-Density Diffuser | Anonymous | 2025 | arXiv 2510.23026 | 非均匀时序分辨率 + planning | Maze2D/Kitchen/AntMaze SOTA | planning-based 对照 | web |
| P08 | One-Step Flow Q-Learning | Anonymous | 2025 | arXiv 2508.13904 | 单步 flow 解决 diffusion bottleneck | offline RL 训练加速 | 推理加速对照 | web |
| P09 | DreamFuser: Value-Guided Diffusion Policy | Anonymous | 2025 | OpenReview | Value-guided sampling | diffusion offline RL | 同期路线 | web |
| P10 | Diffusion Policies with Value-Conditional Optimization | Anonymous | 2025 | arXiv 2511.08922 | Value-conditional diffusion | offline RL | 同期路线 | web |
| P11 | Decision Flow Policy Optimization | Anonymous | 2025 | arXiv 2505.20350 | Flow-based decision policy | online + offline | flow 路线对照 | web |
| P12 | Entropy-Regularized Diffusion Policy with Q-Ensembles | Anonymous | 2024 | arXiv 2402.04080 | Entropy reg + Q-ensembles | offline RL | 熵正则 vs Wasserstein 对照 | web |
| P13 | Adaptive Diffusion Policy Optimization | Anonymous | 2025 | arXiv 2505.08376 | 自适应 diffusion 步长 | robotic manipulation | 同期工程优化 | web |
| **P14** | **Q-DOT: Offline RL with Wasserstein Regularization via OT Maps** | Omura et al. (U-Tokyo) | 2025 | RLC 2025 (RLJ) | **ICNN-based discriminator-free OT map; Brenier 定理** | hopper-medium-v2 +10.4, kitchen-partial-v0 +21.5 | **核心威胁：W₂+ICNN+IQL；启发本提案的 ICNN 设计** | web |
| **P15** | **BWD-IQL: Bellman-Wasserstein Distance for IQL** | Anonymous | 2026 | ICLR 2026 | **Value-aware OT score 用作数据集质量诊断 + IQL 训练正则** | BWD 高度相关 oracle score；IQL 提升 | **核心相关：W-OT for IQL；W-DiffPolicy 是 diffusion 版** | web |
| **P16** | **Rethinking Optimal Transport in Offline RL** | Anonymous | 2024 | NeurIPS 2024 (2410.14069) | **Q-function as transport cost; Maximin OT formulation** | D4RL 改进 | 理论基底：OT 视角看 offline RL | web |
| **P17** | **SWFP: Iterative Refinement of Flow Policies in Probability Space** | Sun et al. | 2025 | arXiv 2510.15388 (Oct 2025) | **JKO 离散化 + W₂ Trust Region for flow policy** | online robotic control | **直接关联：W₂ + flow + JKO；与 W-DiffPolicy 在 online 上对应** | web |
| P18 | Online Reward-Weighted Fine-Tuning of Flow Matching with Wasserstein Regularization | Anonymous | 2025 | arXiv 2502.06061 | W₂ 微调 flow matching | online generative | OT + flow 路线参考 | web |
| **P19** | **KL-Regularized RL is Designed to Mode Collapse** | GX-Chen, Prakash et al. | 2025 | arXiv 2510.20817 (Oct 2025) | **形式化定理：KL 正则的 mode-collapse 不可避免** | LLM/化学语言模型实验验证 | **核心理论支撑：KL bad → 需 W₂ alternative** | web |
| **P20** | **LOM: Learning on One Mode** | Wang, Jin, Montana | 2025 | ICLR 2025 (2412.03258) | GMM 识别 mode + 选择 best mode 加权模仿 | D4RL 改进 | **多模态 RL 的非-OT 路线对照** | web |
| P21 | RAMAC: Multimodal Risk-Aware Offline RL | Anonymous | 2025 | arXiv 2510.02695 | Multi-modal 风险感知 + 行为正则 | offline RL 风险 | 多模态 + 正则化对照 | web |
| **P22** | **Optimal Transport Mapping via ICNN** | Makkuva, Taghvaei, Lee | 2020 | arXiv 1908.10962 | **首次用 ICNN 学 Brenier OT map** | 通用 OT mapping | **理论基底：Q-DOT 与 W-DiffPolicy 共用基础** | web |
| P23 | PUORL: Offline RL with Domain-Unlabeled Data | Anonymous | 2025 | RLC 2025 | PU learning + cross-domain | 1-3% domain labels | cross-domain offline RL benchmark | boardSearch |
| P24 | Cross-Domain Offline Policy Adaptation with OT and Dataset Constraint (OTDF) | Anonymous | 2025 | ICLR 2025 (LRrbD8EZJl) | Transition-level OT alignment + selective sharing | 跨 dynamics shift SOTA | cross-domain OT 对照 | web |
| P25 | Dual-Robust Cross-Domain Offline RL Against Dynamics Shifts | Anonymous | 2025 | arXiv 2512.02486 | Performance bound + dual robustness | cross-domain | 同期 cross-domain | web |
| P26 | Beyond OOD State-Actions: Supported Cross-Domain Offline RL | Anonymous | 2025 | AAAI | Support-based cross-domain | OOD analysis | cross-domain 对照 | boardSearch |
| **P27** | **NeoRL-2: Near Real-World Benchmarks** | Liu et al. (Polixir) | 2025 | arXiv 2503.19267 | 7 个真实场景 task；conservative dataset | SOTA 仍超不过 behavior | **核心 benchmark 之一** | web |
| P28 | Distributional RL by Sinkhorn Divergence | Anonymous | 2022/2024 | arXiv 2202.00769 | Sinkhorn divergence for distributional RL | distributional RL | Sinkhorn 路线参考 | web |
| P29 | Distribution Shift / OOD in Offline RL Survey | Anonymous | 2026 | Neural Comp 2026 | OOD 综述 | 综合视角 | 综述 | boardSearch |

---

## Thematic Analysis

### Theme 1: Diffusion / Flow Policy for Offline RL

**Status**: active (2023-2026 持续高产)
**Dominant approach**: Score-based diffusion policy + Q-learning + 行为正则 (KL or BC loss)
**Papers**: P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13

主线脉络：Diffusion-QL (P01, ICLR 2023) 奠定了用 diffusion 做 multi-modal action distribution 的范式，KL 正则用 BC loss 近似。EDP (P02) 解决了训练效率；BDPO (P04, ICML 2025) 将 KL 解析地按 reverse-time transition kernel 累加，是当前 KL-正则化路线的最强版；SORL (P03, NeurIPS 2025) 用 shortcut models 把训练和推理 scale 起来，在 antmaze-large 上取得 SOTA；RACTD (P06) 走另一条路——consistency distillation 实现 single-step 推理 (142× speedup)，与训练时正则化正交。Flow-based 变体 (ReFORM/Decision Flow/One-Step Flow Q-Learning，P05/P11/P08) 是平行替代方案。

**关键债务**: 几乎所有方法都用 KL 或 BC loss 作为行为正则；P12 (Entropy-Regularized + Q-Ensembles) 是少数尝试替代正则的工作，但仍未触及 OT。**目前没有任何 diffusion policy 工作使用 W₂-OT 正则**。

### Theme 2: Optimal Transport Regularization in Offline RL

**Status**: emerging (核心论文 2024-2026 集中爆发)
**Dominant approach**: W₂ via ICNN (Q-DOT, P14) 或 value-aware OT (BWD-IQL, P15) 或 Maximin OT (P16)
**Papers**: P14, P15, P16, P17, P18

Q-DOT (P14, RLC 2025) 是关键节点：首次用 ICNN-based discriminator-free OT map 实现 W₂ 正则，在 hopper-medium-v2 (+10.4) 和 kitchen-partial-v0 (+21.5) 上比 IQL 大幅领先。**关键限制：仅在 IQL（point-estimate policy）上验证，未涉及 diffusion**。BWD-IQL (P15, ICLR 2026) 是稍微独立的 angle——把 Bellman-Wasserstein 当作数据质量诊断，附带验证 IQL+BWD 训练效果。SWFP (P17, Oct 2025) 是最相关的——在 online flow policy 上用 JKO 离散化得到 W₂ trust region，但目标是 *online* fine-tuning，不是 offline + diffusion。

**关键缺失**: OT regularization 在 IQL/online-flow 上已成熟，但 **diffusion policy training-time OT 正则是空白**。这是 W-DiffPolicy 的「卡位点」。

### Theme 3: Multi-modality Theory & Diagnostics

**Status**: emerging (理论 2025 才形式化)
**Dominant approach**: GMM 识别 mode (LOM)；KL collapse 定理 (P19)；mode-aware 风险正则 (RAMAC)
**Papers**: P19, P20, P21

P19 (KL is Designed to Mode Collapse, Oct 2025) 是关键里程碑——在 LLM RL 上形式化证明 KL 正则的 mode-collapse 不可避免，**但只在生成式语言模型上验证，未推到 control/offline RL**。LOM (P20, ICLR 2025) 走另一条：放弃保留所有 mode，转而用 GMM 选最佳 mode 加权模仿——避免了 multi-modality 但放弃了多样性。RAMAC (P21) 在风险约束设定下做 multi-modal regularization。**没有任何工作给出 OT 正则的 mode-preservation 定理**——这是本提案的理论贡献空间。

### Theme 4: ICNN / OT Theory

**Status**: mature (2020 奠基) + active (2024-2025 应用爆发)
**Dominant approach**: ICNN 学 Brenier convex potential 的梯度作为 OT map
**Papers**: P22, P14, P17

P22 (Makkuva et al. 2020) 是基底：用 ICNN 参数化凸势函数，最优性由 Brenier 定理保证。Q-DOT (P14) 把它直接迁到 offline RL；SWFP (P17) 用 JKO 视角接到 flow policy。这条路线技术上成熟，**主要工程问题是 ICNN 的训练速度（比 GAN-style W₂ 慢但更稳定）**。

### Theme 5: Cross-Domain Offline RL & Real-World Benchmarks

**Status**: active (2024-2026)
**Dominant approach**: PU learning (PUORL)、OT alignment (OTDF)、support-based (Beyond OOD)
**Papers**: P23, P24, P25, P26, P27

PUORL (P23) 用 PU learning 做 cross-domain；OTDF (P24, ICLR 2025) 是直接相关——用 transition-level OT 做 cross-domain dataset filter，但不涉及 policy class。NeoRL-2 (P27) 提供 7 个 near-real-world tasks，且报告了**当前 SOTA 仍超不过 behavior policy** 的悲观结果，是检验 offline RL 是否真有效的「试金石」。

**关键空白**: cross-domain OT (P24) 与 multi-modal diffusion policy 没有结合工作；W-DiffPolicy 在 D4RL → NeoRL-2 zero-shot 上的迁移性是天然实验设计。

### Theme 6: Distributional RL & Sinkhorn

**Status**: mature
**Papers**: P28, P29

Sinkhorn divergence 在 distributional RL（值分布）上有探索（P28），但**作为 offline RL 行为正则的 Sinkhorn 是空白**——这是 W-DiffPolicy 的可选 ablation 维度（W₂ vs Sinkhorn）。

---

## Gap Identification Matrix

| Gap ID | Gap Description | Evidence (papers) | Gap Type | Confidence |
|--------|----------------|-------------------|----------|------------|
| **G1** | **Wasserstein OT 正则尚未应用于 diffusion policy 训练**：W₂-via-ICNN 已在 IQL (point-estimate policy, P14) 和 online flow policy (P17) 上验证；KL 正则在 diffusion 上 (P01, P04) 严格化；但 diffusion + W₂-OT 训练时正则的交集是空白。 | P14, P15, P17, P01, P04 | cross-domain transfer | **HIGH** |
| **G2** | **缺乏 OT 正则 diffusion policy 的 mode-preservation 形式化定理**：P19 给出了 KL 正则的 mode-collapse 定理，但仅在 LLM/语言模型 setting；未推广到 RL；OT 是否能形式化保证 mode 保留也无证明。 | P19, P20, P14 | overlooked formulation | **HIGH** |
| **G3** | **D4RL 等 multi-modal benchmark 缺少标准化 mode-coverage 量化指标**：所有 diffusion 方法都口头宣称多模态保留，但没有标准 "mode count preserved / true mode count" 指标；P20 (LOM) 用 GMM 识别 mode 但未作为正式 benchmark metric。 | P01, P02, P04, P20 | missing diagnostic | MEDIUM |
| **G4** | **RACTD-style consistency distillation 与训练时 OT 正则的正交性未验证**：RACTD (P06) 加速了 diffusion 推理但 inherit 了 teacher 的 KL 正则；W-OT 正则的 diffusion teacher 能否被 RACTD 蒸馏成 single-step 是开放问题。 | P06, P14, P01 | resolution opportunity | MEDIUM |
| **G5** | **W₂-based distribution-shift bound for diffusion policy 不存在**：CQL/IQL/Diffusion-QL 的保守性分析建立在 KL 或 TV 上 (P01, P04)；P14/P15 给出 IQL 的 W-bound 启示但不直接适用于 diffusion 的 K 步 reverse process。 | P01, P04, P14, P15, P16 | overlooked formulation | **HIGH** |
| **G6** | **Cross-domain offline RL 的 OT (OTDF, P24) 与 multi-modal diffusion 未结合**：P24 在 transition 层做 OT alignment 但不涉及 multi-modal action structure；W-DiffPolicy 天然适合 cross-domain 因为 W₂ 同时给出 source/target 距离。 | P24, P27, P23, P25 | cross-domain transfer | MEDIUM |
| **G7** | **ICNN-based OT 在高维 / 离散-连续混合 action space 的扩展性未测**：Q-DOT (P14) 在 hopper/walker/kitchen 上验证；Adroit (24-dim 灵巧手) 与 hybrid action 上效果未知。 | P14, P22 | scaling frontier | LOW |
| **G8** | **Sinkhorn (entropic OT) 用作 diffusion policy 行为正则未探索**：P28 用 Sinkhorn 做 distributional RL；但作为 policy 行为正则的对比研究不存在。 | P28, P14 | overlooked formulation | MEDIUM |
| **G9** | **NeoRL-2 上 OT-regularized 方法的 cross-domain 迁移性能未测**：P27 报告 SOTA offline RL 仍超不过 behavior；P14/P15/P24 都未在 NeoRL-2 上验证；W-DiffPolicy zero-shot 转 NeoRL-2 是天然实验。 | P27, P14, P15, P24 | untested assumption | **HIGH** |
| **G10** | **「KL is Designed to Mode Collapse」(P19) 的 RL 控制实验扩展未完成**：P19 在生成式 LM 上证明并实验，但未在 D4RL 控制任务上重做；W-DiffPolicy 可同时验证 KL 失效 + W 修复。 | P19, P01, P04 | resolution opportunity | MEDIUM |
| **G11** | **RACTD (P06) 与 KL 正则共生的方法学风险**：P06 占据 reward-aware consistency 路线但其 teacher 仍受 KL 正则；理论上 mode 在 distill 阶段进一步坍塌，但实验未对比 W-teacher → consistency student。 | P06, P19 | untested assumption | LOW |
| **G12** | **离线训练时 OT 估计不稳定的稳健化未充分研究**：W₂ 在 long-horizon (T>1000) batch 中估计噪声大；P14 解决 IQL 但 diffusion 多步累积的 W 估计稳健化未做。 | P14, P17 | scaling frontier | MEDIUM |

---

## Trajectory Analysis

**Trajectory tracing performed**: limited (基于已知作者机构 + 论文时间序列)

### Top Authors / Groups

| 名字 | 机构 | 近期方向 | 关键论文 |
|---|---|---|---|
| Motoki Omura et al. | U-Tokyo / Harada-Kurose-Mukoyama Lab | OT regularization for offline RL；ICNN-based discriminator-free 方法 | P14 (Q-DOT, RLC 2025) |
| Chen-Xiao Gao, Yang Yu | Nanjing U / 朱占星组 | Behavior regularized diffusion policy；KL 严格化 | P04 (BDPO, ICML 2025) |
| Wang, Hunt, Zhou | Twitter/X (now X.AI?) | Diffusion policy 范式开创者 | P01 (Diffusion-QL, ICLR 2023) |
| Espinosa-Dice et al. | Cornell | Scaling offline RL with shortcut models | P03 (SORL, NeurIPS 2025) |
| Mianchu Wang, Giovanni Montana | Warwick | Multi-modality in offline RL；mode selection | P20 (LOM, ICLR 2025) |
| Polixir 团队 | 南京大学 / Polixir | Real-world offline RL benchmark | P27 (NeoRL-2) |

### 趋势图谱

```
2020 ─ 2022 ─ 2023 ─ 2024 ─ 2025 ──── 2026 ────► 当前 (2026-05)
          P22    P01    P02   P16   P14, P15, P17, P19    [W-DiffPolicy 窗口]
       ICNN-OT  D-QL   EDP   OT-RL  Q-DOT  BWD  SWFP  KL-collapse-thm
                              ┃     IQL+W  data  flow+W  formal
                              ▼     ICNN  diag   online  theory
                            Maximin
                             OT
```

**关键时间窗口**:
- Q-DOT 出炉 (RLC 2025) → BWD-IQL (ICLR 2026) → SWFP (Oct 2025) → KL-collapse 定理 (Oct 2025) → **W-DiffPolicy (NeurIPS 2026 deadline) 窗口约 6-8 周**
- 主要威胁：Q-DOT 作者 (Omura et al.) 极有可能延伸到 diffusion；需抢在他们之前

### Co-author Clusters

| Cluster | 成员 | 焦点 | 与其他 cluster 的关系 |
|---|---|---|---|
| **U-Tokyo OT-RL** | Omura, Harada, Kurose, Mukoyama | OT + ICNN + IQL | 与 W-DiffPolicy 直接竞争 (技术基底相同) |
| **Nanjing-LAMDA Diffusion** | Gao, Wu, Cao, Y. Yu, Z. Zhang | KL-strict diffusion policy | 与 W-DiffPolicy 互补 (KL → W 替换) |
| **CMU Distillation** | RACTD authors | Reward-aware distillation | 正交可叠加 (W-teacher + RACTD-student) |
| **Cornell SORL** | Espinosa-Dice 等 | Shortcut models scaling | 互补 (training-time vs inference-time) |
| **Polixir / NeoRL** | 南京 polixir 团队 | Real-world benchmark | 提供 evaluation testbed |

---

## References (selected key entries)

```bibtex
@inproceedings{wang2023diffusion,
  title={Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning},
  author={Wang, Z. and Hunt, J. J. and Zhou, M.},
  booktitle={ICLR},
  year={2023},
  url={https://arxiv.org/abs/2208.06193}
}

@inproceedings{omura2025qdot,
  title={Offline Reinforcement Learning with Wasserstein Regularization via Optimal Transport Maps},
  author={Omura, Motoki and others},
  booktitle={Reinforcement Learning Conference (RLC)},
  year={2025},
  url={https://arxiv.org/abs/2507.10843}
}

@inproceedings{bwdiql2026,
  title={Expert or not? Assessing Data Quality in Offline RL (Bellman-Wasserstein Distance)},
  booktitle={ICLR},
  year={2026},
  url={https://arxiv.org/abs/2510.12638}
}

@inproceedings{gao2025bdpo,
  title={Behavior-Regularized Diffusion Policy Optimization for Offline RL},
  author={Gao, Chen-Xiao and Wu, Chenyang and Cao, Mingjun and Xiao, Chenjun and Yu, Yang and Zhang, Zongzhang},
  booktitle={ICML},
  year={2025},
  url={https://arxiv.org/abs/2502.04778}
}

@article{chen2025klmodecollapse,
  title={KL-Regularized Reinforcement Learning is Designed to Mode Collapse},
  author={GX-Chen and Prakash and others},
  journal={arXiv preprint arXiv:2510.20817},
  year={2025}
}

@article{sun2025swfp,
  title={Iterative Refinement of Flow Policies in Probability Space for Online RL (SWFP)},
  author={Sun, Mingyang and others},
  journal={arXiv preprint arXiv:2510.15388},
  year={2025}
}

@inproceedings{neorl2,
  title={NeoRL-2: Near Real-World Benchmarks for Offline RL},
  author={Liu et al. (Polixir)},
  journal={arXiv preprint arXiv:2503.19267},
  year={2025}
}

@inproceedings{makkuva2020icnn,
  title={Optimal Transport Mapping via Input Convex Neural Networks},
  author={Makkuva, A. and Taghvaei, A. and Lee, J. and Oh, S.},
  booktitle={ICML},
  year={2020},
  url={https://arxiv.org/abs/1908.10962}
}

@inproceedings{rethinkot2024,
  title={Rethinking Optimal Transport in Offline Reinforcement Learning},
  booktitle={NeurIPS},
  year={2024},
  url={https://arxiv.org/abs/2410.14069}
}
```

---

## Sources

- [Q-DOT (Omura et al., RLC 2025)](https://arxiv.org/abs/2507.10843)
- [BWD-IQL (ICLR 2026)](https://arxiv.org/abs/2510.12638)
- [BDPO (ICML 2025)](https://arxiv.org/abs/2502.04778)
- [Diffusion-QL (ICLR 2023)](https://arxiv.org/abs/2208.06193)
- [RACTD (CMU, 2025)](https://arxiv.org/abs/2506.07822)
- [SORL (NeurIPS 2025)](https://arxiv.org/abs/2505.22866)
- [SWFP (Oct 2025)](https://arxiv.org/abs/2510.15388)
- [KL-Mode-Collapse (Oct 2025)](https://arxiv.org/abs/2510.20817)
- [Rethinking OT in Offline RL (NeurIPS 2024)](https://arxiv.org/abs/2410.14069)
- [LOM (ICLR 2025)](https://arxiv.org/abs/2412.03258)
- [NeoRL-2 (Polixir 2025)](https://arxiv.org/abs/2503.19267)
- [ICNN-OT (Makkuva et al. 2020)](https://arxiv.org/abs/1908.10962)
- [OTDF (ICLR 2025)](https://openreview.net/forum?id=LRrbD8EZJl)
- [Behavior Regularized Flow Latent Policy (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/39916)
