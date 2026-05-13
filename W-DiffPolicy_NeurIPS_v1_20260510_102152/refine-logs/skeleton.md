# Skeleton — IDEA-01 Mode-Conditional W₂ Diffusion Policy

## State A（reviewer 在阅读前的认知）

- 离线 RL 中 diffusion policy（Diffusion-QL/EDP/SORL/BDPO）通过 KL 或 score-matching BC 行为正则约束 OOD action，且认为 diffusion 的 expressive class 自然能保留多模态。
- W₂-OT 正则在 IQL（Q-DOT, RLC 2025）与 online flow（SWFP, Oct 2025）上已证有效；P19 (Oct 2025) 给出 KL 在 LM RL 上 mode-collapse 的形式化定理。
- 多模态保留 = "diffusion 把多个 mode 都建模出来"，且 D4RL Kitchen/AntMaze 上 return 提升 = 多模态保留成功。

## State B（reviewer 读完后必须相信的）

- "Diffusion policy 自动保多模态" 是错误信念：global W₂（更不必说 KL）即便距离很小，仍可通过 mass 平移让稀有 value-relevant mode 消失——KL 的崩坏只是更严重版本。
- **Mode preservation 必须成为显式约束、可量化指标、可证明对象**：以 latent mode variable z 为锚，引入 per-mode W₂ + mode-mass regularizer，构成 Mode-Conditional W₂ 行为正则。
- 该机制有 finite-sample mode-recall 理论保证 `Recall_mode(πθ) ≥ 1 - O((ε_W + ε_ICNN)/Δ²) - O(ε_mode-est)`，且在 D4RL 多模态任务上既保多模 (90%+ rare-mode recall) 又不牺牲 return。
- W-DiffPolicy 与 OTPR (online Q-cost OT) / P16 (Maximin Q-OT) / Q-DOT (IQL global W₂) 形成正交：差异化在 **offline + diffusion + per-mode preservation + 形式化定理**。

## Skeleton Path（5 个不可跳节点）

1. **Step 1 — Diagnosis**: 即使 W₂ 替换 KL，**全局 W₂ 仍可吞 rare value-relevant mode**（toy mixture + Kitchen 上的 controlled counter-example figure）。若跳过此步，reviewer 会问 "为何不直接 v1 的 KL→W₂ replacement"。
2. **Step 2 — Formalization**: 给 mode preservation 一个 latent mode variable + per-mode conditional W₂ + mode mass GMM 估计的形式化（loss 公式 + ICNN 实现）。若跳过此步，reviewer 会问 "mode 是 observable / latent / cluster 哪种"。
3. **Step 3 — Theoretical Guarantee**: 证明 finite-sample mode-recall bound，error 项明确含 ICNN-W₂ estimator error + GMM mode-est error + mode separation Δ。若跳过此步，reviewer 认为是 "Mathiness or principled-free derivation"。
4. **Step 4 — Three-tier Empirical Chain**: 从 (a) synthetic Gaussian-mixture MDP 解析对照，到 (b) D4RL Kitchen/AntMaze 多模态 benchmark，到 (c) NeoRL-2 1-2 个 task 的 cross-domain transfer。每层都做 oracle-mode / no-mode-mass / global-W₂ / KL ablation。若跳过，Empiricist 不接受。
5. **Step 5 — Differentiation**: 显式 vs OTPR / Q-DOT / BDPO / P19 / LOM 的差异表 + 不依赖 Q-cost OT 的 narrative。若跳过，reviewer 认为是 OTPR/Q-DOT 的 obvious variant。

每步的非可跳性：
- 跳 Step 1 → 论文沦为 v1 的 KL→W₂ replacement，与 Q-DOT 撞 framing
- 跳 Step 2 → mode-preservation 仍是口号，缺乏 formal handle
- 跳 Step 3 → Rigorist 认为 mathiness；NeurIPS 理论门槛过不了
- 跳 Step 4 → Empiricist 认为 toy-only；return 提升不足以说服
- 跳 Step 5 → Innovator 认为 novelty ceiling 被 OTPR 锁死
