# Score Evolution — AttribRL

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|------------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| 1 (Socratic 3-turn) | 8.5 | 8.0 | 7.4 | 7.0 | 6.4 | 8.0 | 6.8 | **7.62** | REVISE |

**Initial baseline (idea-screen Phase 3)**: composite 6.54
**Final after Socratic refinement**: 7.62
**Gain**: +1.08 (target was ≥7.5; achieved ✓)

**Key contributors to gain**:
- Method Specificity: +1.5 (matched distractor 5-bench rules, first-decision-token + argument-slot KL, gated min mixture, two-judge D_t with evidence-availability prompt)
- Contribution Quality: +1.0 (verbatim copy defense lemma 1.a; 10-variant ablation matrix with CST/AgenTracer baselines)
- Validation Focus: +1.2 (paired t-test with Bonferroni; 4 main metrics: conversion/accuracy/copy/cf-sensitivity)

**Outstanding weaknesses**:
- Feasibility (6.4): 18 GPU-day for 10 variants 偏乐观；需 trim 主实验 + appendix
- Venue Readiness (6.8): 方法叙事过载；主文需收束
- Pearl backdoor framing 太强，应降调为 diagnostic counterfactual sensitivity
