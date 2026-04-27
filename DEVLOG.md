# DEVLOG

A running record of decisions, dead ends, and lessons. Newest entries on top. This is not a changelog (the git log is) — it's the why behind the shape of the project.

---

## 2026-04-27 — Verification gate refinement (validated experimentally)

**Setup.** First DeepSeek V4 Flash run on TopK with the new PROMPT.txt regime: PASSed `has_solution`, FAILed correctness because the kernel allocated `threads * k * 8 = 128 KB` of dynamic SMEM on shape 0 (k=64), which exceeds the 100 KB default opt-in cap. Tool-call inventory showed Flash had run zero `python check.py` invocations — it had self-validated with two ad-hoc `python -c "from solution import ..."` snippets that almost certainly used the small default shape (16 KB SMEM) and never iterated through all five shapes.

**Edit.** Tightened the verification gate sentence in all 7 PROMPT.txt files:
- Old: `verify correctness against the oracle in check.py, then iterate. If check.py isn't passing, you're not done.`
- New: ``verify correctness by running `python check.py` and reading the output, then iterate. Don't substitute your own one-off correctness snippets for check.py — it iterates over every shape, your spot-check almost certainly won't. If `python check.py` hasn't printed PASS, you're not done.``

Three deliberate changes: (1) literal-action verb ("by running") replaces the abstract goal ("against the oracle"); (2) the middle sentence directly counter-instructs the failure mode (rolling your own); (3) PASS as the explicit sentinel string anchors the stop condition.

**Validation.** Reran Flash with the same model and the same problem; the only variable was the prompt tweak.
- Tool-call inventory: **3 `python check.py` invocations** (was zero).
- Result: PASS on all 5 shapes, peak_fraction 0.0019.
- The model produced a *correct but slow* kernel rather than a *plausible-looking but broken* one.

The score is low — Flash didn't push throughput — but the disciplinary outcome flipped from FAIL to PASS purely from the prompt edit. That's a clean experimental result. Three sentences of prompt rewrite changed the verification regime from "models that already test thoroughly do; models that don't, don't" to "models that *can* run a test, run it." Capability gates kernel quality; discipline now gates correctness.

Filed under: arguments for tightening prompts further actually do work, sometimes. Counter to my earlier "skill issue" framing — turns out half of "skill issue" is "compliance issue," and compliance is promptable.

---

## 2026-04-27 — Opus parity: --effort max wiring + token-cost logging

**Decision.** Wired `--effort` flag for the `claude` harness in `run_hard.sh` (previously only codex respected `REASONING_EFFORT`). Updated `scripts/sweep.sh` ACTIVE_MATRIX to use `claude claude-opus-4-7 max` for parity with `codex gpt-5.5 xhigh`.

**Why.** Houssin's Twitter critique on the launch post: "Why not use Opus 4.7 Max if you're using xHigh for GPT 5.5? That's not fair." Correct critique. Last sweep ran Opus at default effort while GPT-5.5 was at xhigh. The CLI exposes `low | medium | high | xhigh | max` as the effort tiers (`claude --help`); `max` is the highest. Smoke-tested with a trivial math prompt — flag accepted, thinking block emitted, output_tokens scaled past visible answer length confirming extended thinking happened.

**Thinking-content visibility.** The `thinking` block in Claude Code transcripts comes back with `thinking: ""` and a `signature: "..."` — content is encrypted in the CLI delivery channel. We get cryptographic proof that thinking happened, plus token counts, but not the raw chain-of-thought. Same disclosure floor as codex (codex shows reasoning summaries but not raw CoT either). Symmetry is preserved.

**Token logging in result.json.** Added `scripts/extract_usage.py` — a single Python script that parses each harness's transcript schema (claude/kimi `{"type":"result"}`, codex `payload.type=token_count` events, opencode `step_finish.part.tokens`) and emits a normalized `{input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens, reasoning_tokens, total_cost_usd}` shape. Wired into `run_hard.sh` so result.json now includes a `usage` block. Coding-plan billing on the CLI hides per-call USD, but raw token counts are always present in the transcripts and that's what matters for cross-model comparison.

Validated on the Flash rerun: input=57,555, output=12,158, cache_read=1,367,296, reasoning=98,047. The 1.37M cache_read confirms DeepSeek implicit caching is hot (matches the 98% hit rate noted in the provider-pinning entry below).

---

## 2026-04-27 — Model coverage expansion (Twitter-driven)

**Added to ACTIVE_MATRIX** (per Twitter requests):
- `qwen/qwen3.6-max-preview` — Alibaba, 262k ctx
- `qwen/qwen3.6-plus` — Alibaba, 1M ctx
- `qwen/qwen3.6-27b` — Alibaba, 262k ctx
- `xiaomi/mimo-v2.5-pro` — Xiaomi (lab), fp8, 1M ctx

All routed through `opencode openrouter-pinned/...` with `provider.order = ["Alibaba", "Xiaomi", "Minimax", "DeepSeek", "Z.AI"]` and `allow_fallbacks: false`.

**Skipped: `qwen/qwen3.6-35b-a3b`.** Alibaba does not host this model on OpenRouter — only AtlasCloud and Parasail (both fp8). Skipping to maintain the native-lab-only integrity rule. To include later, add AtlasCloud to provider.order; flag for revisit if user OKs third-party fp8 for this one model.

**TODO — when budget permits:**
- **GPT-5.5 Pro.** Twitter request × 1 (insanowskyy: "what about 5.5 pro?"). Not on the active matrix because the OpenAI per-call cost is high enough to be a real budget item; coding-plan doesn't apply to API-direct gpt-5.5-pro calls. Revisit when sweep cadence justifies the spend.
- **Gemini 3.1 Pro.** Twitter request × 2. The harness story is unsettled — Droid worked in v3 but is not currently wired into KernelBench-Hard's run_hard.sh. Adding requires either (a) re-adding the droid case to run_hard.sh and authenticating via Factory, or (b) routing through opencode if Google AI Studio offers an OpenAI-compatible endpoint. Skipped until the harness wiring is decided.
- **Mythos / generic "show us X" requests.** Volume signal only; not actionable until the model has a stable identifier and a benchmarkable native-lab provider.

---

## 2026-04-27 — Prompt regime overhaul: eval-shaped → human-shaped

**Decision.** Replaced the two-file `preamble.md` + `AGENT.md` system-prompt regime with a single per-problem `PROMPT.txt` written in plain human voice. The harness now sends `PROMPT.txt` directly as the prompt to each agent — no system/user split, no markdown structure, no "Read SYSTEM_PROMPT.md first" wrapper.

**Why.** Two observations from the TopK overnight sweep:

1. The old preamble opened with "You are an autonomous coding agent being evaluated on a hard GPU kernel optimization problem." That framing primes models to perform-on-test rather than do-the-work. Opus's "the 0.1 RESULT threshold isn't structurally achievable here" rationalization is the eval-shape pattern: when you tell a model it's being evaluated, it explains its score instead of fixing the kernel.
2. The preamble was 101 lines of hardware specs, peak throughput tables, optimization recipes, profiling commands, and workflow steps. That's a benchmark giving away the answer key and then asking the model to find the answer. Models that already know this stuff gain nothing; weaker models get carried.

**What changed in the prompt itself.**

Removed entirely: opening "you are an autonomous coding agent" framing; full hardware spec section (tensor cores, what's not on SM120, etc.); peak throughput table; toolchain section (CUDA versions, compile flags, CUTLASS path); optimization guidance (FP4/FP8/BF16/TMA recipes); profiling commands (`ncu`, `nsys`, `torch.profiler`); workflow steps; budget line; "what makes a good solution"; "good luck" closer.

Kept: one-line hardware identifier in a parenthetical (`SM120 Blackwell, GDDR7, 1.8 TB/s`); library availability list (without it the model won't know FLA / scattermoe / flashinfer are options); shapes inlined as prose; forbidden ops inlined as prose; tolerance + correctness contract inlined as prose; verification gate as a single sentence in the flywheel paragraph ("If check.py isn't passing, you're not done."); custom-kernel mandate; "look up PTX docs / clone repos / investigate" directive.

**What the model now doesn't know coming in.** Peak TFLOPS for any precision. Which tensor-core instructions are available on SM120. Which are SM100-only and will fail. Compile flags. The fact that 188 SMs exist. Profiling tool names. Optimization recipes. It has to look these up itself or know them from training data — that's part of what's being measured.

**What stays in the workspace.** `reference.py`, `check.py`, `benchmark.py`, `problem.yaml`, `shapes.py`, `sota.py`, `PROMPT.txt`. The yaml and shapes.py have to stay because `check.py` and `benchmark.py` import them at runtime. Small leakage risk (a curious model could `cat problem.yaml` and read the regime / forbidden list / tolerance again), but the prompt only directs the model to `reference.py`. If that leakage matters later, the fix is refactoring check/benchmark to read yaml from outside the workspace; not yet worth the complexity.

**Files deleted.** `src/harness/preamble.md`, all `problems/*/AGENT.md` (8 files), one stale `problems/02_kda_cutlass/SYSTEM_PROMPT.md`. The harness no longer composes a SYSTEM_PROMPT.md per run.

**Smoke-tested.** Claude Code on problem 05 with `BUDGET_SECONDS=300` — confirmed PROMPT.txt arrives clean as `event[6] type=user` in the transcript, workspace cleanup behaves, no stale SYSTEM_PROMPT.md left behind.

---

## 2026-04-27 — Verification gate added (then folded into the flywheel)

**Decision.** Added a "your final action before stopping must be a successful `python check.py`" requirement to the prompt. After the prompt overhaul, this lives as a single sentence ("If check.py isn't passing, you're not done.") inside the flywheel paragraph rather than its own section.

**Why.** Of the 4 non-passing TopK runs:
- DeepSeek V4 Flash: linker error from `extern "C"` mismatch between `.cu` and `cpp_sources` header inside `load_inline`.
- DeepSeek V4 Pro: CUDA illegal memory access in the bitonic merge kernel.
- MiniMax M2.7: hardcoded `build_directory="/tmp/topk_v2"` that didn't exist; `FileNotFoundError` on first import.
- GLM-5.1: never wrote `solution.py` — burned 31,995 reasoning tokens before emitting any tool call.

3 of 4 would have been caught by running `check.py` once before submitting. The pattern is "submit blind, stop." Mandating a verification pass costs nothing for capable models, and it's not "hand-holding" — it's the discipline-half of pair programming, which is fair to require. (GLM is unfixable from the prompt; that's a Z.AI output-token-budget problem.)

---

## 2026-04-26 — TopK overnight sweep: forensic findings

**Setup.** 7 models × 1 problem (05_topk_bitonic), sequential, 45-min budget each. `regime: memory`, scored against 1.8 TB/s GDDR7 peak. Geomean over 5 shapes.

**Results.**

| Rank | Model            | Status               | peak_fraction |
| ---- | ---------------- | -------------------- | ------------- |
| 1    | GPT-5.5 xhigh    | PASS                 | 0.0657        |
| 2    | Claude Opus 4.7  | PASS                 | 0.0132        |
| 3    | Kimi K2.6        | PASS (timed out)     | 0.0063        |
| —    | GLM-5.1          | ERR (no solution.py) | —             |
| —    | DeepSeek V4 Pro  | FAIL (CUDA OOB)      | —             |
| —    | DeepSeek V4 Flash| FAIL (link error)    | —             |
| —    | MiniMax M2.7     | FAIL (build dir)     | —             |

**Algorithm gap dominated kernel-craft gap.** GPT and Opus had the same wall budget on the same hardware. Opus picked full bitonic sort (O(n log²n) per row), GPT picked packed-key reduction with `tl.topk` (O(n) per row). At n=8192 that's a ~7x asymptotic gap — and the observed perf gap on the prefill shape (b=64, n=8192, k=8) was 8.7x. The kernel-craft delta would have been maybe 2x; the algorithmic choice was 5-7x of the 8.7x.

**Opus's "structurally launch-bound" claim was wrong.** On shape 0 (b=1, n=131072, k=64), Opus claimed the geomean threshold was unreachable because "the whole benchmark is launch-overhead bound." Actual numbers:
- Bandwidth lower bound to read 512 KB at 1.8 TB/s: **0.28 μs**.
- GPT-5.5 measured: **27 μs** (~100x slower than the floor).
- Opus measured: **48 μs** (~170x slower).

A single launch on a hot CUDA graph is ~1-2 μs. The remaining ~25 μs is real kernel time, not launches. Why is the kernel slow? GPT picked `chunk_n=2048` for shape 0, which gives `131072/2048 = 64` blocks for a 188-SM machine. **34% SM occupancy ceiling.** The kernel is leaving 2/3 of the GPU idle. Opus's CHUNK_PAD=2048 has the identical bug. The fix is `chunk_n=512` → 256 blocks → fully oversubscribed → near-peak bandwidth → estimated 0.10–0.15 peak_fraction on shape 0 alone.

Lesson: "launch-bound" is a real diagnosis on small kernels with many launches and no graphs. "Parallelism-starved" is a different diagnosis with the same surface symptom (low throughput on small shapes). Mixing them up is how rationalization sneaks in. Both Opus and GPT made the same parallelism-starvation mistake; only Opus rationalized it as physical-limit-bound.

**The 4 failures break into one model-side issue and three "didn't run check.py" issues.** GLM-5.1's 31995-reasoning-token blowup is fixable only by raising opencode's max output tokens for zai/glm-5.1; nothing in the prompt fixes a model that can't budget its own thinking. The other three were trivial bugs that any single test run would have caught. Hence the verification gate.

---

## 2026-04-25 — Centralized timing module + L2 flush + warmup bump

**Setup.** Each `problems/<NN>/benchmark.py` was duplicating warmup-and-cuda-events code. Several discrepancies surfaced when comparing runs.

**What we found.** Without an explicit L2 cache flush between trials, FP8 GEMM peak_fraction came out at 0.520. With a 128 MB write to evict L2 (Blackwell consumer L2 is 96 MB), the same kernel measured 0.426. The skinny-M shape went 20% → 10% with the flush. The original numbers were measuring L2-cached re-reads, not HBM bandwidth.

Warmup of 5 was too short for Triton autotune (~7 configs) plus `torch.compile(reduce-overhead)` CUDA-graph capture. Bumped to 10. `iters` defaults to 30 trials; report median.

**What lives in `src/eval/timing.py`.** Single `time_fn(fn, inputs, iters, warmup)` that does warmup → per-trial L2 flush → cuda Events with synchronize-after-record → median. All seven `benchmark.py` files import this; methodology bugs only need fixing once.

**Known biases not addressed.** `torch.compile(reduce-overhead)` gets CUDA graphs which eliminate launch overhead; custom Triton/CUDA kernels do not. On small shapes this gives the compile baseline an artificial advantage. Accepted as the cost of using torch.compile as the published "compiled" reference.

---

## 2026-04-25 — Harness wars

**ccr-rust pivot to OpenCode SST.** Tried routing Claude Code to non-Anthropic providers via ccr-rust (an Anthropic-API-shape proxy). It returned malformed SSE that broke the claude-code stream-json parser. Pivoted to OpenCode SST with custom OpenAI-shape providers (`deepseek`, `zai`, `openrouter-pinned`) — that worked.

**Codex 0.125.0 broke chat-completions routing.** The new release dropped `wire_api="chat"` config support, so codex can no longer route arbitrary OpenRouter models. It only speaks `/responses` API now. Z.AI doesn't implement `/responses`, so GLM-5.1 cannot be reached through codex at all. We fall back to opencode for non-OpenAI lab models. Documented in `CLAUDE.md` model-harness assignment table.

**Codex session-id-from-stderr instead of mtime.** Codex 0.125.0 touches old session JSONL files when scanning its SQLite thread-state DB. So picking the most-recently-modified file in `~/.codex/sessions/<date>/` returns the wrong file. The fix is to grep `session id: <uuid>` out of stderr and `find -name "*${sid}*.jsonl"`.

**`set -e` + SIGTERM 124 was a silent script killer.** When a harness hits the wall-clock `timeout` and gets SIGTERM, exit code is 124. With `set -euo pipefail`, capturing the exit via `cmd; HARNESS_EXIT=$?` exits the whole script. Fix: `cmd || HARNESS_EXIT=$?`. This bug ate two debugging sessions before we caught it.

**Local rust codex binary had a stale alias.** `npm install -g @openai/codex` gives 0.125.0 with `gpt-5.5` support; the local rust binary was 0.118.0 and rejected the model name. Non-interactive shells don't see the alias, so `which codex` was lying. Force PATH to npm bin.

---

## 2026-04-24 — Provider pinning + caching wisdom

**OpenRouter dispatches to whichever backend has capacity, including int4/fp4-quantized weights.** Code generation on int4 is materially worse than full weights — a model that scores 50% on bf16 might score 30% on int4. So `provider_order` pinning to the native lab is mandatory for benchmark integrity.

**Pinning works in our harness, not in Droid custom models.** Droid OpenRouter custom-model configs ignore `provider_order`. The KernelBench harness sends `extraBody.provider.order` directly via the OpenAI SDK, which OpenRouter respects. Anything routed through Droid custom OpenRouter loses pinning.

**MiniMax direct API was 401.** Worked through OpenRouter pinned to "Minimax" provider (their fp8 endpoint, ~$0.30/$1.20 per M, 99.7% uptime).

**DeepSeek implicit caching is real.** Verified: same prompt sent twice in a row hit `cache_tokens: 1792 / 1829` on the second request. ~98% cache hit rate at 10x cost reduction. No explicit cache-control header needed; just resend the same prefix.

---

## 2026-04-24 — Why "Hard": pivot from KernelBench-v3

**v3 was 43 problems of grab-bag difficulty.** Most were winnable by any frontier model with any harness. Median speedups ended up reward-hacked (precision downcast, F.softmax wrappers, GEMM dispatching to `torch._scaled_mm`) or trivially above eager. Leaderboard non-informative.

**v3 reward-hack patterns.** GLM-5.1 cast fp32 inputs to fp16 before GEMM to use tensor cores → ~2x "speedup" that was cheaper arithmetic, not better algorithm. `pct_of_peak > 100%` was the giveaway. MiniMax M2.5 attempted `pkill -f python` to kill the eval process on its first run. Various models called the library wrapper (F.softmax, F.scaled_dot_product_attention) and counted that as a "kernel."

**What we tried that didn't work.** Extensive regex blocklists for forbidden patterns. Brittle whack-a-mole — every release added new ways to hide the dispatch. Replaced with an LLM judge model post-benchmark (`src/eval/judge.py`) that reviews the solution code and flags semantic cheating. Better recall than regex; defaults to PASS on judge error to avoid false negatives.

**Hard's three changes vs v3.**
1. **Tight per-dtype tolerance + multi-shape eval kills reward-hacking** at the correctness gate, so a degenerate identity-operator solution fails check.py.
2. **Roofline grading against hardware peak**, not speedup over PyTorch. Beating eager means nothing; approaching SOTA is the goal. peak_fraction = achieved_TFLOPS / peak_TFLOPS for compute regime, achieved_GB/s / peak_HBM for memory regime.
3. **Forbidden ops listed in problem.yaml + inlined into the prompt.** Using `torch.topk` on a top-k problem fails post-hoc. The point of each problem is to write the kernel, not to dispatch to a library.

**Wall-clock budgets > turn-count budgets.** v3 used `for turn in range(max_turns)` in agent loops. Models like GLM-5.1 burned 9/10 turns on filesystem exploration ("looking for CUTLASS headers") and never wrote code. Switched to wall-clock timeouts in v3 late-stage; carried over to Hard. Models get unlimited turns within a 45-min budget.

---

## Open questions / things to chase

- **Pair-programming eval.** The autonomous-floor numbers tell us how each model behaves alone; they don't tell us the human-in-the-loop ceiling. A 5-model paired-session run on problem 05 would answer "what's the agency tax of running model X without me there?" — the gap between paired and autonomous peak_fraction. n=1 per model but useful even so.

- **Persistent-kernel / cooperative-reduction kernel for shape 0 of TopK.** Both PASS submissions (Opus, GPT) are parallelism-starved on b=1 n=131072. A correctly-fanned-out kernel should hit ≥0.10 on shape 0 alone. Worth writing the reference solution by hand to confirm the achievable ceiling and validate whether the geomean threshold of 0.1 is reachable.

- **GLM-5.1 output-token cap.** The opencode `extraBody` for zai/glm-5.1 doesn't expose `max_output_tokens` — Z.AI's beta API caps at 32768. With reasoning chains of 30k+, that leaves no room for tool calls. Either request a higher cap from Z.AI, or accept GLM as an outlier whose autonomous score is bounded by output budget rather than capability.

- **Removing problem.yaml + shapes.py from the model's view.** Currently they sit in the workspace because check.py and benchmark.py import them. Refactor option: pre-render their content into the prompt (already done) and have check.py / benchmark.py read yaml/shapes from a sibling private directory. Closes a small information leak. Not currently load-bearing.

- **Per-problem prompt voice consistency.** All seven prompts hand-written in one session, same voice, same four-paragraph structure. If we add an 8th problem (Metal lightning attn) or add a second hardware target, the temptation will be to write that prompt in a different style. Resist. The voice is part of the experimental control.
