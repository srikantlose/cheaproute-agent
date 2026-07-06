# CheapRoute

**Hybrid token-efficient routing agent** — AMD Developer Hackathon: ACT II, Track 1.

For every task, CheapRoute tries a small **local** Gemma model first (llama.cpp on
CPU inside the container — local tokens are free in scoring). It estimates how
confident that answer is; only when confidence falls below a tunable threshold τ
does it escalate to a stronger **remote** Gemma model on Fireworks AI — the only
tokens that count. Submissions pass an LLM-judge accuracy gate, then rank by
fewest proxy-recorded tokens: the router's job is to stay above the gate while
spending as close to zero as possible.

```
/input/tasks.json ─▶ batch adapter (thread pool + global deadline watchdog)
                        │ per task: classify type (math/code/summary/NER/...)
                        ▼
                     local: k samples + logprobs (llama.cpp, Gemma GGUF, free)
                        │ confidence = logprobs ⊕ agreement ⊕ format
                        ├─ conf ≥ τ ─▶ local answer                (0 tokens)
                        └─ conf < τ ─▶ Fireworks via FIREWORKS_BASE_URL
                                        └─ on failure ─▶ local answer
                        ▼
                     /output/results.json  (+ inference_log.json), exit 0
```

## Scoring-harness contract (Track 1)

- Reads `/input/tasks.json` (`[{"task_id","prompt"}]`), writes
  `/output/results.json` (`[{"task_id","answer"}]`), exits 0. Every `task_id`
  is echoed even if a task fails or the deadline hits (partial > invalid).
- All remote calls go through the injected **`FIREWORKS_BASE_URL`** with the
  injected **`FIREWORKS_API_KEY`**; the model is chosen at runtime from
  **`ALLOWED_MODELS`** via a ranked preference list (Gemma first) — nothing
  hardcoded.
- Own deadline (default 540s) inside the 10-minute cap; llama-server is up in
  seconds, well within the 60-second readiness rule; image ~3GB, linux/amd64.

## Task-type awareness

A zero-cost heuristic classifier ([tasktype.py](src/cheaproute/tasktype.py)) maps each
prompt to one of the 8 evaluation categories and sets: the local/remote
prompts, how the final answer is extracted (fenced code for code tasks, full
text for summaries/NER, `Answer:` line otherwise), the self-consistency sample
count (3 for short-form, 1 for free-form), and the remote `max_tokens` budget
(16 for a letter, 600 for code) — because remote output tokens are the score.

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
.venv/Scripts/python -m pytest                                   # 45 tests
echo '{"id":"t1","task":"What is 17 + 25?"}' | .venv/Scripts/python -m cheaproute
```

`config.yaml` ships with both backends set to `mock`, so everything runs
anywhere. Batch mode locally:

```bash
python -m cheaproute --adapter batch --input in/tasks.json --output out/results.json
```

## Evaluation & threshold tuning

```bash
python eval/run_eval.py --collect-both     # 107 practice tasks across all 8
                                           # categories; grades local AND remote
python eval/tune_threshold.py --min-accuracy 0.85
```

`tune_threshold.py` re-simulates routing offline for 50 values of τ and prints
the accuracy / %local / remote-token curve, then recommends the cheapest τ
that clears the accuracy floor. Code tasks are graded by actually executing
the generated Python against asserts in a sandboxed subprocess.

## Container

```bash
docker build --platform linux/amd64 -f docker/Dockerfile -t cheaproute .
docker run -v $(pwd)/input:/input -v $(pwd)/output:/output \
  -e FIREWORKS_API_KEY=fw_xxx \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS=gemma-4-31b-it,minimax-m3 \
  cheaproute
```

The entrypoint starts llama-server with the baked Gemma GGUF, health-checks it
(bounded), then runs the batch adapter. If llama-server never comes up, every
task escalates remotely; if Fireworks is down, the local answer is returned —
**the container never crashes, never hangs, and always writes valid JSON**.

Publish for submission (public GHCR package):

```bash
docker tag cheaproute ghcr.io/srikantlose/cheaproute-agent:latest
docker push ghcr.io/srikantlose/cheaproute-agent:latest
```

## Robustness guarantees

- Malformed / unknown-schema / BOM-corrupted payloads are normalized, never fatal.
- Fireworks calls: bounded timeout, retries with backoff, fallback to local answer.
- Global deadline watchdog writes partial-but-valid results before the runtime cap.
- Per-task decision records (route, type, confidence signals, token counts,
  latency) go to `/output/inference_log.json` and stderr.

## License

MIT — see [LICENSE](LICENSE).
