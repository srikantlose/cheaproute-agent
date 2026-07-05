# CheapRoute

**Hybrid token-efficient routing agent** — AMD Developer Hackathon: ACT II, Track 1.

For every task, CheapRoute tries a small **local** model first (Gemma 3 4B on an
AMD GPU via vLLM/ROCm — local tokens are free in scoring). It estimates how
confident that answer is; only when confidence falls below a tunable threshold τ
does it escalate to a stronger **remote** model (Gemma 3 27B via the Fireworks AI
API — the only tokens that cost anything). Wrong-but-cheap is prevented by tuning
τ against an accuracy floor; expensive-but-safe is prevented by keeping τ as low
as that floor allows.

```
task ─▶ adapter (stdio|http|cli) ─▶ local: k samples + logprobs
                                        │ confidence = logprobs ⊕ agreement ⊕ format
                                        ├─ conf ≥ τ ─▶ local answer      (0 cost)
                                        └─ conf < τ ─▶ Fireworks remote  (tokens)
                                                        └─ on failure ─▶ local answer
```

## Confidence signals

1. **Token logprobs** (primary) — how sure the local model was while writing.
2. **Self-consistency** — k local samples; do they agree? Local tokens are free,
   so extra samples cost only latency.
3. **Format check** — non-empty, not truncated, followed the answer convention.

Weights, k, and τ live in [config.yaml](config.yaml); every value is overridable via
`CHEAPROUTE_*` env vars (e.g. `CHEAPROUTE_ROUTING_THRESHOLD=0.7`) so the
container can be re-tuned without rebuilding.

## Quickstart (dev, no GPU needed)

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Windows paths
.venv/Scripts/python -m pytest                                   # 22 tests
echo '{"id":"t1","task":"What is 17 + 25?"}' | .venv/Scripts/python -m cheaproute
```

`config.yaml` ships with both backends set to `mock`, so everything runs anywhere.
For real runs set `local.backend: openai` (vLLM endpoint) and
`remote.backend: fireworks` (+ `FIREWORKS_API_KEY` in the environment).

## Evaluation & threshold tuning

```bash
python eval/run_eval.py --collect-both     # runs 80 practice tasks, grades
                                           # local AND remote, caches results
python eval/tune_threshold.py --min-accuracy 0.85
```

`tune_threshold.py` re-simulates routing offline for 50 values of τ and prints
the full accuracy / %local / remote-token curve, then recommends the cheapest τ
that clears the accuracy floor.

## Container

```bash
# HF_TOKEN needs accepted access to google/gemma-3-4b-it (gated repo)
docker build -f docker/Dockerfile --build-arg HF_TOKEN=hf_xxx -t cheaproute .
docker run --device=/dev/kfd --device=/dev/dri \
  -e FIREWORKS_API_KEY=fw_xxx -i cheaproute        # stdio adapter (default)
# or: -e CHEAPROUTE_ADAPTER=http -p 8080:8080      # POST /task, GET /health
```

The entrypoint boots vLLM (ROCm) with the local model, waits for health
(bounded), then starts the adapter. If vLLM never comes up, the router
escalates everything remotely; if Fireworks is down, it returns the local
answer — **it never crashes or hangs on a task**.

## Robustness guarantees

- Malformed / unknown-schema / BOM-corrupted task payloads are normalized, never fatal.
- Fireworks calls: bounded timeout, 2 retries with backoff, fallback to local answer.
- Every task appends a JSONL decision record (route, confidence signals, token
  counts, latency) to `logs/decisions.jsonl`; a one-line summary goes to stderr.

## License

MIT — see [LICENSE](LICENSE).
