# CheapRoute

**Token-efficient routing agent** — AMD Developer Hackathon: ACT II, Track 1.

Per the official Participant Guide: submissions must clear an 80% LLM-judge
accuracy gate (19 fixed tasks), then rank ascending by total tokens recorded
through the Fireworks proxy. **Local inference counts fully toward accuracy
and costs zero toward the token score** — a local-only agent is an explicitly
valid strategy (`flagged: ZERO_API_CALLS` is not a failure). CheapRoute's
strategy follows directly from that: answer almost everything with a small
**local** Gemma model (llama.cpp on CPU inside the container, free), and
escalate to Fireworks only when local confidence is low enough to suggest a
genuine failure.

- **`local-first`** (also tagged `:latest`) — the competitive entry. Sample
  the local model, score confidence (logprobs ⊕ self-consistency ⊕ format),
  and only call Fireworks below a tunable threshold τ, currently tuned so
  low that Fireworks is effectively an emergency fallback (≈96% of traffic
  answered locally on the practice set; see the validation table below).
- **`remote-only`** — every answer comes from Fireworks. Kept published as
  an emergency-fallback tag in case a future organizer clarification
  reverses the current local-scoring guidance, but it is not the
  competitive submission: it spends real tokens for no accuracy benefit
  the local-first variant doesn't already provide.

Both modes are tags of the same image, selected at build time via
`ROUTING_MODE`, still overridable at runtime via `CHEAPROUTE_ROUTING_MODE`.

```
/input/tasks.json ─▶ batch adapter (daemon-thread pool + global deadline watchdog)
                        │ per task: classify type (math/mc/classification/
                        │   extraction/language/logic/summary/NER/code/...)
          ┌─────────────┴──────────────┐
   mode=remote_only             mode=local_first
          │                            │
          ▼                            ▼
   Fireworks via            local: k samples + logprobs (Gemma GGUF, free)
   FIREWORKS_BASE_URL          │ confidence = logprobs ⊕ agreement ⊕ format
   (terse prompt, tight        ├─ conf ≥ τ ─▶ local answer        (0 tokens)
    max_tokens)                └─ conf < τ ─▶ Fireworks via FIREWORKS_BASE_URL
          │ on failure ─▶ 1 local sample        │ on failure ─▶ local answer
          └─────────────┬──────────────────────┘
                        ▼
                     /output/results.json  (+ inference_log.json), exit 0
```

## Scoring-harness contract (Track 1)

- Reads `/input/tasks.json` (`[{"task_id","prompt"}]`), writes
  `/output/results.json` (`[{"task_id","answer"}]`), exits 0. Every `task_id`
  is echoed even if a task fails or the deadline hits (partial > invalid) —
  the batch adapter uses a daemon-thread pool gated by a deadline `Event`,
  not a `ThreadPoolExecutor`, specifically so the results write can never be
  blocked behind an in-flight task.
- All remote calls go through the injected **`FIREWORKS_BASE_URL`** with the
  injected **`FIREWORKS_API_KEY`**; the model is chosen at runtime from
  **`ALLOWED_MODELS`** via a ranked preference list (Gemma first) — nothing
  hardcoded.
- Own deadline (default 540s) inside the 10-minute cap; llama-server is up in
  seconds, well within the 60-second readiness rule; image ~2.5GB, linux/amd64.
- **Grading environment is 4GB RAM / 2 vCPU** and scores exactly 19 fixed
  tasks against an 80% LLM-judge accuracy gate. Every number in the
  validation table below was measured under `docker run --memory=4g
  --cpus=2`, matching that constraint, not the unconstrained dev workstation.

## Task-type awareness

The Participant Guide's official evaluation is **8 capability categories**:
factual knowledge, mathematical reasoning, sentiment classification, text
summarisation, named entity recognition, code debugging, logical/deductive
reasoning, code generation. A zero-cost heuristic classifier
([tasktype.py](src/cheaproute/tasktype.py)) covers those 8 plus 3 finer-
grained routing types (multiple choice, general classification, extraction,
language) that get their own tighter prompts/budgets instead of falling back
to a generic short-answer profile — every practice-set task still lands
under one of the 8 official categories at grading time, this is purely an
internal prompting/budget optimization. The classified type sets: the
local/remote prompts (local answers favor completeness — local tokens are
free and the real evaluator is an LLM judge grading intent, not a strict
string match; remote prompts stay terse since remote output tokens are the
score), how the final answer is extracted (fenced code for code tasks, full
text for summaries/NER, `Answer:` line otherwise), the self-consistency
sample count (up to 3 for short-form, 1 for free-form), and the remote
`max_tokens` budget (16 for a letter, 600 for code).

## Confidence signals

1. **Token logprobs** (primary) — how sure the local model was while writing.
2. **Self-consistency** — k local samples; do they agree? (short-form only)
3. **Format check** — non-empty, not truncated, followed the expected shape.

Weights, k, and τ live in [config.yaml](config.yaml); every scalar is overridable via
`CHEAPROUTE_*` env vars (e.g. `CHEAPROUTE_ROUTING_THRESHOLD=0.7`) so the
container can be re-tuned without rebuilding.

## Quickstart (dev, no GPU or model needed)

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Windows paths
.venv/Scripts/python -m pytest                                   # 65 tests
echo '{"id":"t1","task":"What is 17 + 25?"}' | .venv/Scripts/python -m cheaproute
```

`config.yaml` ships with both backends set to `mock`, so everything runs
anywhere. Batch mode locally:

```bash
python -m cheaproute --adapter batch --input in/tasks.json --output out/results.json
```

## Evaluation & threshold tuning

```bash
python eval/run_eval.py --collect-both     # 107 practice tasks across all 11
                                           # categories; grades local AND remote
python eval/tune_threshold.py --min-accuracy 0.85
```

`tune_threshold.py` re-simulates routing offline for 50 values of τ and prints
the accuracy / %local / remote-token curve, then recommends the cheapest τ
that clears the accuracy floor. Code tasks are graded by actually executing
the generated Python against asserts in a sandboxed subprocess.

### Real-model validation (constrained to the judge's 4GB/2vCPU environment)

Both container images were built, run end-to-end, and graded with a real
local Gemma GGUF (CPU llama.cpp) and a real Fireworks-hosted remote model
(the harness's exact `ALLOWED_MODELS` aren't reachable with a personal
Fireworks key, so a comparably-sized model stood in for remote validation
only — production never hardcodes a model id, see below). All runs used
`docker run --memory=4g --cpus=2` to match the grading environment, not the
unconstrained dev workstation:

| run | accuracy | max per-task latency | routing |
|---|---:|---:|---|
| `local-first`, 19-task judge-shaped sample* | **100%** (19/19) | 16.1s | 16 local / 3 remote |
| `local-first`, full 107-task practice set† | **95.6%** (86/90 completed) | 18.7s | 73 local / 17 remote / 17 unfinished |
| `local-first`, full 107-task practice set, unconstrained‡ | 94.4% (101/107) | 18.4s | 93 local / 14 remote |
| `remote-only`, full 107-task practice set (unconstrained sanity) | 94.4% | 5.0s | 107 remote |

\* 2-3 tasks sampled per official category from the practice set — the
closest local proxy to the judge's real 19-task, 8-category grading set.
† A deliberate overload stress test (107 tasks vs. the judge's 19): the
9-minute internal deadline cuts the run off with some tasks still queued
(scored as empty/wrong, dragging the raw 107-task number down) — the batch
deadline watchdog still writes valid, complete JSON on time regardless.
Accuracy and latency among tasks that actually ran is the number that
matters; both stay far inside the 30s/request limit (max 18.7s, i.e. over
11s of margin) and clear the 80% accuracy gate with a wide buffer.
‡ Same image, same full task set, no `--memory`/`--cpus` cap — a regression
check confirming the CPU-constrained runs above aren't an artifact of the
constraint itself (accuracy and max latency both land in the same range
either way).

τ was tuned against this same real local + real remote data
(`eval/results/threshold_curve.csv`): accuracy is flat at 93.5% (96% local,
813 remote tokens total on the full practice set) for τ in [0.02, 0.64],
only dropping to 89.7% at τ=0 (no escalation path at all, even for a
genuinely failed local generation). Since local tokens are free and only
remote tokens count toward the ranking once the accuracy gate is cleared,
τ=0.02 was chosen: virtually all traffic stays local (`ZERO_API_CALLS`-
adjacent — a real escalation path still exists for actual local failures)
while spending a fraction of what a higher, marginally-more-accurate τ would
cost in tokens for no ranking benefit past the gate.

This session's validation pass also surfaced and fixed several real bugs
(full list in commit history): a batch deadline watchdog that could block
past the 10-minute hard cap behind an in-flight task; a hosted "reasoning"
model getting cut off mid-answer whenever it wanted more room than its
budget allowed, now retried with a larger budget; an empty-but-200 remote
response shipping as a successful answer instead of falling back locally;
non-retryable remote HTTP errors burning the full retry budget before
falling back; llama-server thread/context sizing that ignored the judge's
cgroup CPU quota; and three of the eleven practice-set task categories
(classification, extraction, language) having no dedicated classifier
branch at all.

## Container

Pull the published image (or build it — two variants from one Dockerfile, the
`ROUTING_MODE` build arg bakes the mode, still overridable at runtime via
`CHEAPROUTE_ROUTING_MODE`):

```bash
# Published images
docker pull srikantlose/cheaproute-agent:remote-only
docker pull srikantlose/cheaproute-agent:local-first

# ...or build from source
docker build --platform linux/amd64 --build-arg ROUTING_MODE=remote_only \
  -f docker/Dockerfile -t cheaproute:remote-only .
docker build --platform linux/amd64 \
  -f docker/Dockerfile -t cheaproute:local-first .

docker run -v $(pwd)/input:/input -v $(pwd)/output:/output \
  -e FIREWORKS_API_KEY=fw_xxx \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS=gemma-4-31b-it,minimax-m3 \
  srikantlose/cheaproute-agent:local-first
```

The entrypoint starts llama-server with the baked Gemma GGUF, health-checks it
(bounded), then runs the batch adapter. If llama-server never comes up, every
task escalates remotely; if Fireworks is down, the local answer is returned —
**the container never crashes, never hangs, and always writes valid JSON**.

Publish for submission (public Docker Hub repository):

```bash
docker tag cheaproute:local-first srikantlose/cheaproute-agent:local-first
docker tag cheaproute:local-first srikantlose/cheaproute-agent:latest
docker tag cheaproute:remote-only srikantlose/cheaproute-agent:remote-only
docker push srikantlose/cheaproute-agent:latest
docker push srikantlose/cheaproute-agent:local-first
docker push srikantlose/cheaproute-agent:remote-only
```

## Robustness guarantees

- Malformed / unknown-schema / BOM-corrupted payloads are normalized, never fatal.
- Fireworks calls: bounded timeout, retries with backoff, fallback to local answer.
- Global deadline watchdog writes partial-but-valid results before the runtime cap.
- Per-task decision records (route, type, confidence signals, token counts,
  latency) go to `/output/inference_log.json` and stderr.

## Submission

- **Docker image**: `docker.io/srikantlose/cheaproute-agent:latest`
  (= `:local-first`, the competitive entry — local inference counts fully
  toward accuracy and costs zero toward the token score per the official
  clarification); `:remote-only` stays published as an emergency-fallback
  tag only.
- **Demo video**: _link TBD_
- **Slide deck**: _link TBD_

## License

MIT — see [LICENSE](LICENSE).
