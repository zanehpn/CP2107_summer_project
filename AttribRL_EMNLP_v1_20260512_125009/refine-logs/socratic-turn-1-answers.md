# Socratic Turn 1 — Answers (Claude)

[See full text in main conversation log; key answers summarized here]

## A1. R_cf-ema 传播规则
- Trajectory-time axis EMA, per-trajectory state, γ=0.7
- Probe step: EMA(t+1) = γ·EMA(t) + (1-γ)·R_cf(t+1)^raw
- Non-probe step: carry forward EMA(t+1) = EMA(t)
- Initial value: batch-level mean of computed cf KL (cold-start 0.1)
- 不跨 trajectory leak；不按 obs type 全局聚合

## A2. 量纲校准 — sigmoid + temperature
$\hat{R}_{cf,t} = \sigma((R_{cf,t} - \mu_{cf}) / T_{cf})$
- $\mu_{cf}$ = batch-level 50-percentile
- $T_{cf}$ = batch-level IQR
- Output range [0,1] 与 rouge-L 同量纲
- 用 sigmoid 不用 min-max 因前者对 outliers 鲁棒

## A3. Lemma 1 精确化 — trivial copy only
Lemma 1.a: 对于 *trivial copy policy* (复制 obs 固定 schema position)，verbatim defense 成立。
- semantic-copy edge case (复制 evidence span + transformation) 是 *legitimate utilization*，不被惩罚
- 新增 hacking detector audit only (附录)

## A4. VerbatimCopyRatio LCS-based 定义
$\text{VerbatimCopyRatio}(a_t) = \frac{|\text{LCS}(a_t, o_t) \cap \text{NonEvidenceTokens}(o_t)|}{|a_t|}$
- 只惩罚 non-evidence 复制 (schema/wrapper/metadata)
- evidence pass-through (BFCL value-into-argument) 不计入
- Threshold 0.7，pilot 校准

## A5. 训练成本预算
- K=5 sparse probe rate: 20% step actual cf computation
- 80% step EMA carry forward
- Batch-level distractor cache: 64 distractors per task type
- 2×4090 epoch budget: Qwen-7B ≈ 12 hr (8 hr base + 4 hr cf overhead);
- Total: 18 GPU-day = 9 calendar day parallel 2 GPU ✓
- Failure: K=10 if overhead > 2×; QLoRA r=32 if memory tight
