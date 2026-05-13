# Refinement Report — AttribRL

**Problem**: Training tool-augmented LLM agents to causally condition actions on tool observations (close 97.54% discovery vs 0.53% use gap)
**Initial Approach**: Counterfactual + lexical 双源 attribution reward + GRPO（anchor design doc base proposal）
**Date**: 2026-05-12
**Mode**: socratic-auto (3 dialogue turns)
**Rounds**: 1 (Socratic) / MAX_ROUNDS=3
**Final Score**: 7.62 / 10
**Final Verdict**: REVISE (Findings/Accept potential, Main 依赖实证)
**Baseline (from idea-screen)**: composite 6.54 → Target ≥7.5 → Achieved 7.62 (+1.08)

## Problem Anchor (immutable)
[See FINAL_PROPOSAL.md §1]

## Skeleton (final)
5 步：DiscUseBench 规模化 → causal D/U/C → AttribRL gated min mixture → diagnostic sensitivity guarantees → 6-variant × multi-size 实证 + audit + Plan B。

## Output Files
- Skeleton: `refine-logs/skeleton.md`
- Initial proposal: `refine-logs/round-0-initial-proposal.md`
- Socratic turn 0/1/2 questions + answers: `refine-logs/socratic-turn-{0,1,2}-{questions,answers}.md`
- Socratic final review: `refine-logs/socratic-final-review.md`
- **Final proposal**: `refine-logs/FINAL_PROPOSAL.md` (canonical clean version)
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Score evolution: `refine-logs/score-history.md`

## Score Evolution

| Round | PF | MS | CQ | FL | Feas | VF | VR | Overall | Verdict |
|-------|----|----|----|----|------|----|----|---------|---------|
| Initial (idea-screen) | — | — | — | — | — | — | — | 6.54 | PROCEED WITH CAUTION |
| Socratic 3-turn (final) | 8.5 | 8.0 | 7.4 | 7.0 | 6.4 | 8.0 | 6.8 | **7.62** | REVISE |

## Round-by-Round Review Record

| Round | Reviewer Concerns | Top-2 Issues Targeted | What Changed | Result |
|-------|-------------------|------------------------|--------------|--------|
| Socratic Turn 0 | distractor 构造规则 / KL 单位 / delayed KL 机制 / span extractor / verbatim defense | matched distractor pool design + verbatim defense formalism | added per-bench rules; first-decision-token KL; off-policy delayed KL; evidence-role span; Lemma 1 | partial — verbatim defense still wasn't conjunctive |
| Socratic Turn 1 | EMA / KL量纲校准 / Lemma 1 π_copy / VerbatimCopyRatio / 训练 cost | KL量纲 + π_copy 严格定义 | trajectory-time EMA carry-forward + sigmoid normalization + absolute floor κ + null-distractor net cf + Lemma 1.a refinement (trivial-copy only) + LCS-based VerbatimCopyRatio excluding evidence | resolved |
| Socratic Turn 2 | EMA on non-probe / absolute floor / argument-slot KL / D_t evidence-vs-seen / ablation isolation | argument-slot KL + ablation isolation vs CST/AgenTracer/dense-shaping | EMA carry-forward formula; two-stage fn-tok + arg-tok KL with $w_{\text{arg}}=0.7$; D_t judge prompt v2; 10-variant ablation V0-V9 with stopping rule | resolved → understanding declaration |
| Final scoring & post-revisions | feasibility 6.4 / venue readiness 6.8 + drift warning | trim ablation matrix 10→6 + collapse adaptive α + Pearl framing downgrade | dropped adaptive α; delayed KL → diagnostic only; V9 → related baseline; Pearl → soft motivation appendix; main paper centers on "evidence-specific action sensitivity reward" | applied to FINAL_PROPOSAL.md |

## Final Proposal Snapshot

- **Dominant contribution**: AttribRL — gated min mixture $U_t = D_t \cdot \min(\hat{R}_{cf-\text{ema}}^{(t)}, R_{lex,t}) \cdot \text{Mask}(t)$ as first training-time utilization reward
- **Supporting contribution**: DiscUseBench — 14 model × 5 bench D/U/C grid (also Plan B Findings fallback)
- **Theoretical anchor**: Lemma 1.a (verbatim trivial-copy defense); diagnostic counterfactual sensitivity (not causal identification)
- **Experiments**: 6 main variants × Qwen-7B × 3 seeds × 5 benchmark + 3 variants × Qwen-14B × 3 seeds × 2 bench + 3 ablation 1-seed appendix + V9 AgenTracer-style related baseline
- **Budget**: 18 GPU-day on 2×4090, $30 API; 4-week timeline with W1 pilot gate (gpt-oss-120b 97.54% 复现 + Qwen-7B base < 30%)
- **Anti-drift**: paper 主线 collapsed to *one core object* (evidence-specific action sensitivity); appendix 承载 delayed KL / adaptive α / V9 / Pearl motivation

## Method Evolution Highlights

1. **Most important simplification (post-Socratic)**: drop adaptive α + collapse 10-variant matrix → 6 + delayed KL → diagnostic only。理由：reviewer 警告 "contribution 漂到 heuristic stack"。
2. **Most important mechanism upgrade (Turn 0→1)**: plain weighted sum $\alpha R_{cf} + (1-\alpha) R_{lex}$ → **gated min mixture** with EMA + Mask + null-distractor calibration。理由：reviewer 指出 weighted sum 无法严格防 verbatim copy (Lemma 1 fails for semantic copy)。
3. **Most important framing modernization (final)**: Pearl backdoor identifiability claim → diagnostic counterfactual sensitivity claim。理由：reviewer Contribution Quality 7.4 扣分在 "Pearl framing 太强难成立"。

## Pushback / Drift Log

| Round | Reviewer Said | Author Response | Outcome |
|-------|---------------|------------------|---------|
| Turn 0 Q5 | "weighted sum 不严格防 verbatim copy" | accepted; upgraded to gated min mixture + Mask + Lemma 1.a | accepted |
| Turn 1 Q3 | "Lemma 1 中 π_copy 含糊；semantic copy 反例" | accepted; refined to Lemma 1.a (trivial-copy only) + caveat 标注 semantic copy 是 legitimate utilization | accepted |
| Turn 2 Q3 | "first-decision-token 可能命中 tool choice 而非 argument" | accepted; upgraded to two-stage fn-tok + arg-tok with $w_{\text{arg}}=0.7$ | accepted |
| Final | "Feasibility 偏紧 + Venue 叙事过载" | accepted; trimmed ablation 10→6 + Pearl framing → soft motivation only | accepted in FINAL_PROPOSAL.md |
| Final | "Drift: simple utilization reward → multi-gate stack" | accepted; main paper collapsed to "evidence-specific action sensitivity" single object | accepted in §9 Anti-Drift Statement |

## Remaining Weaknesses

1. **KL → ATE framing 弱**：已 downgrade 但仍是 main contribution 的 conceptual weak point；mitigation 通过 empirical Spearman correlation ≥ 0.7 with human-annotated utilization (200 sample)
2. **14B 仅子集**：V0/V2/V7 × 3 seed × BFCL+ToolBench；reviewer R2 可批；mitigation 通过 trend statement + 算力 disclosure
3. **W1 pilot 必过**：gpt-oss-120b 复现 97.54%/0.53% + Qwen-7B base conversion < 30%；W1 不过 → Plan B DiscUseBench Findings paper (composite 6.79, safety net)
4. **Reward hacking edge cases**：semantic copy + transformation 边界依赖 transformed answer recognizer 的 rule quality；pilot 100 task validate ≥ 0.85 coverage 是 gating criterion

## Raw Reviewer Responses

<details>
<summary>Socratic Turn 0 — GPT 5 questions on mechanism specificity</summary>

Q1-Q5 见 `refine-logs/socratic-turn-0-questions.md`
</details>

<details>
<summary>Socratic Turn 1 — GPT 5 questions on formalism precision</summary>

Q1-Q5 见 `refine-logs/socratic-turn-1-questions.md`
</details>

<details>
<summary>Socratic Turn 2 — GPT 5 questions on reward semantics + ablation</summary>

Q1-Q5 见 `refine-logs/socratic-turn-2-questions.md`
</details>

<details>
<summary>Final Score & Verdict (gpt-5.5)</summary>

见 `refine-logs/socratic-final-review.md` 完整 7-dim breakdown + simplification/modernization/drift/verdict
</details>

## Next Steps

- **If READY**: proposal is venue-ready, proceed to experiment planning (not applicable here — verdict REVISE)
- **If REVISE (current)**: FINAL_PROPOSAL.md is the canonical version；剩余 weaknesses (Pearl framing / 14B 子集 / W1 pilot / semantic-copy rule) 在 paper writing 阶段处理；建议 W1 pilot 完成后再启动 `/idea-refine` Round 2 attempt 7.5 → 9.0
- **If RETHINK**: 不适用；core mechanism 清楚，仅需 framing + experimental scope 调整
