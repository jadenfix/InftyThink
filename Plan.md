Here’s a full research plan you can actually execute on **CPU + JAX**, while still keeping it academically credible.

## Project title

**Reproducing and Extending InftyThink on CPU with JAX: Iterative Reasoning via Bounded Segments and Progress Summaries**

## 1. Problem statement

InftyThink proposes replacing one monolithic long reasoning trace with an iterative loop of **short reasoning segments** plus **concise progress summaries**. The paper argues this yields bounded computational cost, a “sawtooth” memory pattern, and better reasoning efficiency without architectural changes. It also reports reconstructing OpenR1-Math into **333K** iterative-format training instances and shows **3–11%** gains for Qwen2.5-Math-7B on **MATH500, AIME24, and GPQA_diamond**. ([arXiv][1])

Your version of the project is not to match their frontier-scale training, but to test whether the **core protocol** still helps at small scale: on CPU, in JAX, with a manageable model and a tractable subset of the data. JAX supports CPU-only development directly, including a simple `pip install -U jax` path and a `JAX_PLATFORMS=cpu` option to force CPU execution. ([JAX Documentation][2])

## 2. Main objective

Reproduce the central InftyThink claim at small scale:

**Iterative reasoning with summaries can outperform or match vanilla long-chain reasoning at equal or lower token/computation budget.** This claim is directly aligned with the paper’s framing and reported results. ([arXiv][1])

## 3. Research questions

### RQ1

Does iterative reasoning with summaries improve final-answer accuracy over vanilla CoT under the same token budget?

### RQ2

How sensitive is performance to:

* segment length
* summary length
* number of iterations
* whether the next step conditions on only the summary vs summary + recent segment tail

### RQ3

What information is lost when summaries compress earlier reasoning, and how much does that loss hurt correctness?

### RQ4

Can a better memory representation than plain-text summaries improve the framework?

That last question is your best academic branching point.

## 4. Core hypotheses

**H1.** On long-ish math problems, segmented reasoning with summaries will beat vanilla one-shot CoT at matched budget because it avoids uncontrolled context growth.

**H2.** There will be a nontrivial sweet spot for segment length. Too short creates overhead; too long recreates the original long-context problem.

**H3.** Plain-text summaries will fail mainly through **constraint loss**, **numeric drift**, and **forgotten subgoals**.

**H4.** Adding a more structured progress state will outperform plain free-form summaries.

The first two are direct extensions of the InftyThink motivation; the latter two are the natural academic next step. ([arXiv][1])

## 5. Scope and success criteria

Keep the first pass intentionally narrow.

You are **not** trying to:

* reproduce the exact full paper results
* train a frontier reasoning model
* run expensive RL or multi-node jobs

You **are** trying to:

* recreate the iterative segment-summary mechanism
* train on a small subset of a relevant reasoning dataset
* show at least one rigorous comparison against vanilla CoT
* produce one defensible extension beyond the base method

A successful first paper-style outcome is:

* a working JAX implementation,
* a clean dataset-conversion pipeline,
* a baseline comparison showing when iterative summaries help or hurt,
* an ablation section,
* one extension with measurable effect.

## 6. Dataset plan

The cleanest starting dataset is **OpenR1-Math-220k**. Hugging Face describes it as **220k math problems** with **2–4 reasoning traces** per problem, and notes that the `default` split contains **94k problems** and achieved the best SFT performance among its splits. ([Hugging Face][3])

If you want the larger raw source, **OpenR1-Math-Raw** contains **516,499 problems** and **1,209,403 solutions**, with traces kept under a **16k-token budget** and formatted with `<think>...</think>`. ([Hugging Face][4])

### My recommendation

Use:

* **OpenR1-Math-220k, `default` split** for the main reproduction
* a **2k–10k example subset** for the first serious run
* OpenR1-Math-Raw only if you later want a larger-scale reconstruction study ([Hugging Face][3])

## 7. Experimental design

### Phase A: inference-only reproduction

Goal: test the mechanism before training.

Build a loop like this:

1. Prompt the model with the question.
2. Generate at most `K` reasoning tokens.
3. Stop and generate a summary of current progress in at most `S` tokens.
4. Build the next prompt from:

   * original question
   * prior summary or summaries
   * optionally the most recent segment tail
5. Repeat for `T` iterations.
6. Ask for final answer.

This phase answers whether the **protocol itself** has value, even before supervised tuning.

### Phase B: supervised reproduction

Goal: train the model in the InftyThink-style format.

Convert each long rationale into:

* `segment_1`
* `summary_1`
* `segment_2`
* `summary_2`
* ...
* `final_answer`

Then train the model on two tasks:

* **segment generation**
* **summary generation**

You can do this as one decoder-only model with control tokens such as:

* `<SEGMENT>`
* `<SUMMARY>`
* `<FINAL>`

### Phase C: extension study

Goal: add one genuinely novel variant.

Best first extension:
**structured progress state instead of plain summary text**

Example structured state:

* solved facts
* unresolved subgoals
* current numeric values
* important constraints
* confidence or uncertainty flag

This is the most obvious and academically strong branch because InftyThink’s summaries are useful but potentially lossy. The paper itself is about bounded reasoning via summarization; your extension asks whether **state structure** is better than unconstrained prose. ([arXiv][1])

## 8. Data construction pipeline

For each example from OpenR1-Math:

### Step 1: parse

Extract:

* problem text
* one valid reasoning trace
* final answer

### Step 2: clean

Strip wrapper tokens if needed, normalize whitespace, and preserve the final answer span.

### Step 3: segment

Split the reasoning trace into chunks by token count, not by character count.

Initial values to test:

* segment length = 64, 128, 256 tokens
* summary length = 16, 32, 64 tokens

### Step 4: create summary targets

Three options, in order of simplicity:

**Option 1: heuristic summaries**
Use rule-based compression from the trace. Lowest quality, cheapest.

**Option 2: teacher-generated summaries**
Use a stronger external model once offline to summarize each segment history.

**Option 3: summary-as-latent-supervision**
No teacher summaries at first; only train segment generation and ask the model to create summaries as auxiliary outputs later.

For a clean first paper, Option 2 is strongest.

### Step 5: create training examples

Format each step as something like:

**Input**

* question
* prior summary or summaries
* task token `<SEGMENT>`

**Target**

* next reasoning segment

And:

**Input**

* question
* prior summary or summaries
* latest segment
* task token `<SUMMARY>`

**Target**

* next summary

## 9. Model plan

Stay small enough for CPU.

Use a compact decoder-only language model implemented in JAX/Flax or a minimal JAX transformer you control. Do not optimize for benchmark size; optimize for:

* clean training loop
* stable tokenizer pipeline
* easy ablations
* reproducibility

A practical target is a **small model where full-batch experiments finish locally**, even if training is slow. The academic value here comes more from the protocol and analysis than from squeezing the last bit of accuracy out of model scale.

## 10. Baselines

You need strong baselines or the project will feel undercontrolled.

### Baseline 1: vanilla CoT

Question → one long reasoning trace → final answer

### Baseline 2: token-capped vanilla CoT

Same as above, but with the same effective token budget as the iterative method

### Baseline 3: segmented reasoning without summaries

This tells you whether the summary itself matters

### Baseline 4: iterative reasoning with naive truncation memory

Pass only the latest segment, no summary

### Main method

InftyThink-style segment + summary loop

### Extension

Structured summary/state version

## 11. Metrics

Primary:

* final-answer accuracy

Secondary:

* total generated tokens
* wall-clock runtime
* average tokens per correct answer
* peak context length
* summary compression ratio

Diagnostic:

* error rate from forgotten constraints
* error rate from arithmetic drift
* error rate from wrong subgoal tracking

Those diagnostics are important because they give you something academically interesting beyond “accuracy went up 1.2 points.”

## 12. Ablation plan

This section is where your project starts to look like a paper.

Run ablations on:

### Segment length

64 / 128 / 256

### Summary length

16 / 32 / 64

### Number of iterations

2 / 4 / 8

### Conditioning strategy

* summary only
* summary + latest segment tail
* all prior summaries concatenated
* compressed rolling state only

### Training objective

* segment-only
* summary-only auxiliary
* joint segment + summary

### Summary style

* free-form plain text
* structured bullet state
* key-value state schema

## 13. Failure analysis plan

For every wrong answer in a sampled eval set, classify the failure:

1. **Constraint loss**
   A condition from the question disappears from the summary.

2. **Intermediate result corruption**
   A numeric value, symbol, or relationship mutates.

3. **Subgoal omission**
   A remaining step is not carried forward.

4. **Overcompression**
   Summary is grammatical but too vague to support correct continuation.

5. **Redundant overhead**
   The iterative loop spends budget summarizing trivial progress.

This kind of analysis is exactly what can turn a decent engineering project into a good research artifact.

## 14. The best academic extension

If you want one branch that feels most publishable, do this:

### Structured progress state + verifier

Replace plain summary text with a schema like:

```text
known_facts:
open_subgoals:
derived_values:
important_constraints:
confidence:
```

Then add a lightweight verifier that checks whether:

* all original constraints still appear when relevant,
* important derived quantities are preserved,
* open subgoals are not silently dropped.

### Why this is a strong branch

InftyThink’s core insight is that bounded reasoning is possible through summarization. Your extension asks whether **the summary format is the bottleneck**. That is a clean, adjacent research question rather than a random add-on. ([arXiv][1])

## 15. Alternative extension branches

If you want options, rank them like this:

### Best

**Structured state memory**

### Second-best

**Adaptive summarize/continue policy**
Instead of fixed segment length, let the system decide when to summarize

### Third-best

**Summary retrieval memory**
Retrieve similar prior summaries or solved patterns

### Fourth-best

**Domain transfer**
Use long document QA or scientific-paper reasoning instead of math

The structured-state branch is still the cleanest and most publishable for a first pass.

## 16. Timeline

### Week 1

Set up JAX CPU environment, tokenizer, base model, dataset loader. JAX supports a straightforward CPU install path, and `JAX_PLATFORMS=cpu` can force CPU backend selection. ([JAX Documentation][2])

### Week 2

Implement dataset parsing and segmentation pipeline on a small OpenR1-Math subset. OpenR1-Math-220k `default` is the right curated starting point. ([Hugging Face][3])

### Week 3

Build inference-only iterative loop and compare against vanilla CoT on a few hundred examples.

### Week 4

Train the first supervised segment+summary model on a 2k–5k subset.

### Week 5

Run core baseline comparison and collect metrics.

### Week 6

Add ablations on segment length, summary length, and conditioning style.

### Week 7

Implement structured-state extension and rerun a smaller comparison.

### Week 8

Failure analysis, plots, writing, and result cleanup.

## 17. Deliverables

By the end, you should have:

* a reproducible JAX repo
* a dataset-conversion script
* an inference-only protocol implementation
* a supervised training pipeline
* baseline and ablation results
* a structured-state extension
* a short paper draft or technical report

## 18. What would count as a good result

Any of these is publishable-ish at small scale:

* iterative summaries beat vanilla CoT at matched budget
* iterative summaries reduce context growth with little or no accuracy loss
* structured summaries outperform free-form summaries
* adaptive summarization beats fixed segmentation
* failure analysis shows where summary compression breaks

You do not need frontier-scale absolute scores. You need a clear causal story.

## 19. Risks and mitigations

### Risk 1

CPU training is too slow

**Mitigation:** keep the first pass tiny, prioritize inference-only experiments first, and use a few thousand examples before scaling.

### Risk 2

Summary targets are poor

**Mitigation:** generate teacher summaries once offline and freeze them for the study.

### Risk 3

No measurable improvement over vanilla CoT

**Mitigation:** lean into failure analysis and the structured-state extension. A negative result can still be valuable if it explains when summarization loses critical information.

### Risk 4

The model is too weak to benefit from the protocol

**Mitigation:** evaluate the protocol first with a stronger teacher in the loop, then separate “protocol quality” from “student capacity.”

## 20. Recommended exact first version

If I were setting this up for you, I’d do this:

* Dataset: **OpenR1-Math-220k `default`** ([Hugging Face][3])
* Training subset: **5,000 examples**
* Eval subset: **500 examples**
* Segment lengths: **128 and 256**
* Summary lengths: **32 and 64**
* Baselines: vanilla CoT, token-capped CoT, segmented-no-summary
* Extension: **structured progress state**
* Main success metric: **accuracy at matched token budget**

That is small enough to execute and large enough to say something real.

## 21. One-sentence paper pitch

**We reproduce InftyThink’s bounded iterative reasoning paradigm at small scale in JAX on CPU, show when segment-summary reasoning helps or fails relative to vanilla CoT, and introduce structured progress states that reduce summary-induced information loss.**

That is a real research story.

I can turn this into a **full experiment spec** next: exact dataset columns, example prompt templates, ablation matrix, and what plots to make.

[1]: https://arxiv.org/abs/2503.06692 "[2503.06692] InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models"
[2]: https://docs.jax.dev/en/latest/installation.html "Installation — JAX  documentation"
[3]: https://huggingface.co/datasets/open-r1/OpenR1-Math-220k "open-r1/OpenR1-Math-220k · Datasets at Hugging Face"
[4]: https://huggingface.co/datasets/open-r1/OpenR1-Math-Raw "open-r1/OpenR1-Math-Raw · Datasets at Hugging Face"
