# Review Summary — AttribRL

**Problem**: Tool-augmented LLM agents 在 trajectory 中发现 tool output 远超使用 tool output；缺 self-supervised utilization training reward。
**Initial Approach**: Counterfactual + lexical 双源 attribution reward + GRPO (from anchor design doc)。
**Date**: 2026-05-12
**Mode**: socratic-auto (3 dialogue turns + understanding declaration + final scoring)
**Rounds**: 1 (Socratic 3-turn = single round)
**Final Score**: 7.62 / 10 (lift from screening baseline 6.54, target ≥7.5 achieved ✓)
**Final Verdict**: REVISE (Findings/Accept potential; Main 依赖实证干净度)

## Problem Anchor
Bottom-line: gpt-oss-120b 97.54% discovery vs 0.53% use → utilization gap 必须直接 reward
Constraints: 2×4090, $30 API, 4 weeks, EMNLP main/Findings, ≥2 size, ≥3 seeds
Success: Qwen-7B conversion +30pp + verbatim < 5% + 14B 复现 + Plan B safety net

## Skeleton (final, post-Socratic refined)
1. DiscUseBench 14×5 grid → motivation 硬证据
2. Causal D/U/C definition via counterfactual replacement
3. AttribRL gated min mixture (R_cf-ema + R_lex + D_t + Mask + λ schedule)
4. **Diagnostic counterfactual sensitivity** guarantees（**降调** Pearl identifiability，soft causal motivation only）
5. 6 variants × 7B × 3 seeds × 5 bench + 14B cross-size + ablation appendix + hacking audit + Plan B

## Round-by-Round Resolution Log

| Round | Reviewer Concerns | What Round Simplified / Modernized | Top-2 Issues | Solved? | Remaining Risk |
|-------|-------------------|-------------------------------------|--------------|---------|----------------|
| Socratic Turn 0 (Q1-Q5) | distractor 构造 / KL token target / delayed KL / span extractor / verbatim defense formalism | matched distractor 5-bench rules + first-decision-token KL + frozen-action off-policy delayed KL + evidence-role-aware span extractor + Lemma 1.a verbatim defense | distractor pool + KL normalization | ✅ partial (Q5 verbatim defense needed upgrade) | KL unbounded vs rouge-L bounded |
| Socratic Turn 1 (Q1-Q5) | EMA propagation rule / KL量纲校准 / Lemma 1 π_copy 定义 / VerbatimCopyRatio 定义 / 训练 cost | trajectory-time EMA carry-forward / sigmoid normalization + absolute floor + null-distractor calibration / Lemma 1.a refined for trivial-copy / LCS-based ratio excluding evidence tokens / 18 GPU-day budget itemized | KL量纲 + π_copy 严格定义 | ✅ resolved | first-decision-token might miss argument slot |
| Socratic Turn 2 (Q1-Q5) | EMA on non-probe / absolute KL floor + null-distractor / argument-slot KL / D_t evidence-vs-seen / ablation matrix | EMA(t+1)=EMA(t) non-update / null-distractor net cf / two-stage token (fn-tok + arg-tok) / D_t judge prompt v2 "evidence-availability" / 10-variant ablation V0-V9 with CST/AgenTracer baselines | argument-slot KL + judge prompt + ablation isolation | ✅ resolved | scope creep risk → triggered final score |
| Final Scoring | 7 dimensions; OVERALL 7.62; REVISE | applied: drop adaptive α; delayed KL → diagnostic; V9 → related baseline (not training variant); Pearl framing → soft motivation only | feasibility (18 GPU-day 偏紧) + venue readiness (叙事过载) | resolved via simplification | Final pilot validation needed in W1 |

## Overall Evolution

- **Method became more concrete**: matched distractor rules (per-benchmark, ~5000 pool each), first-decision-token + argument-slot KL (two-stage with $w_{\text{arg}}=0.7$), gated min mixture with sigmoid normalization + null-distractor calibration + absolute floor + LCS-based mask, judge prompt v2 with evidence-availability framing, EMA trajectory-time propagation
- **Contribution became more focused**: dropped adaptive α, demoted delayed KL to diagnostic/appendix, demoted V9 AgenTracer to related baseline; main paper now centers on "evidence-specific action sensitivity reward"
- **Theoretical claim refined**: downgraded Pearl backdoor identifiability to *soft motivation* (Appendix B); main claim is *diagnostic counterfactual sensitivity proxy* + empirical correlation with human utilization (≥0.7 Spearman)
- **Verbatim defense formalized**: Lemma 1.a covers trivial-copy / schema-boilerplate-copy; semantic-copy explicitly excluded with caveat (legitimate utilization)
- **Drift avoided**: scope locked to text-only tool agents, GRPO black box reuse, no new RL algorithm, no frontier scale, no retrieval

## Final Status

- **Anchor status**: ✅ preserved (Engländer 2026 anchor fact + Discovery/Use/Conversion 三元 + 2×4090 constraint 全部贯彻)
- **Focus status**: tight (1 dominant: AttribRL reward; 1 supporting: DiscUseBench eval/fallback)
- **Modernity status**: appropriately frontier-aware (GRPO + Qwen + tool-agent benchmark + MCP relevance) without forced trendiness
- **Skeleton completeness**: all 5 steps mapped to paper sections
- **Strongest parts**: matched distractor + null-distractor calibration; first-decision-token + argument-slot two-stage KL; gated min mixture as structural defense; 6-variant ablation isolating CST/AgenTracer/dense-shaping
- **Remaining weaknesses**:
  1. KL → ATE 形式 framing 弱（已 downgrade 至 diagnostic）
  2. 14B 只跑子集 (V0/V2/V7 × 3 seed, BFCL+ToolBench only) — reviewer 仍可批
  3. W1 pilot 必须通过（gpt-oss-120b 复现 + Qwen-7B base conversion < 30%）；不过 → Plan B
