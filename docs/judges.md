# Judge drivers and execution

Application setup is in [Choosing the judge](../README.md#choosing-the-judge). This page describes driver contracts and native execution limits.

## Hosted requests and ownership

A hosted driver combines one shared engine with a provider (endpoint, authentication, routing and response envelope), a codec (question encoding and score decoding) and a model profile (question kinds and widths, questions per request, request size, truncation, probability precision and normalization). Built-in profiles map known models to their codec: `google/gemma-4-31b-it` and other listed chat models answer one question per request through chat log probabilities (top 20, `require_parameters`, no fallbacks); `typesafe/jev-1.13` and `openai/gpt-6-luna-decisions` answer every question of an input in one System One request under empirical two-decimal profiles pinned to recorded response fixtures. Any other model needs `codec="chat"` or an explicit `profile`; System One needs a profile declaring its serialization precision, and a returned distribution whose mass differs from one by more than `width × 0.5 × 10^-decimals` fails instead of being renormalized. `semantic_switch.testing.drivers` is the conformance kit for new providers, codecs and profiles.

A provider that rejects the request shape fails with an error naming it; pin another with `provider`. `timeout` is each request unit's end-to-end budget, not a fresh per-attempt timeout: it starts before admission and covers admission and active-slot queueing, request encoding, every transport attempt and response-body read, retry waits (408/429/5xx and connection errors, at most 5 attempts) and decoding. Admission bounds how many units hold work, not how long one may queue; each attempt receives only the remaining budget, and a response completing after the deadline is rejected, not retried.

A unit's request body is encoded once after admission and reused byte-for-byte by every retry, so at most `2 × batch_inputs` bodies exist; at most `batch_inputs` attempts are on the wire, and a unit releases its active slot during backoff and decoding.

A `Retry-After` from the provider holds every unit of that judge until it passes; other retries back off privately. Request receipts split elapsed time into queue, service and retry seconds that sum to the total. An encoding, transport or decoding failure fails only its unit; the other units and their receipts are kept.

Each hosted judge runs one HTTPX `AsyncClient` on its own AnyIO portal; `close()` (or `await judge.aclose()`) cancels queued and active units, then releases an owned transport once, and repeating it is harmless.

A supplied `transport` is an async `send(url, body_bytes, remaining_seconds) -> Reply`, borrowed unless `own_transport=True`; an owned transport's `close` hook, synchronous or async, is optional.

A runtime owns and closes a judge it constructs from a source (`ss.Runtime("openrouter:MODEL")`); a judge you supply is borrowed unless `own_judges=True`. `Runtime(bindings={"judge": ..., "hosted": ...})` binds several model roles; a runtime given only roles other than `judge` binds no default judge. Leaving a `with Runtime(...)` block by an exception cancels its in-flight work before closing; `close()` waits for it. Cancelling `await rt.aask(...)` cancels its hosted requests and starts no further rounds; a native decode finishes its current batch first. Locking a hosted judge requires a pinned provider; the lock records the protocol, the requested provider and the returned model and provider.

The CLI uses the same registry and validator: `ss --judge openrouter:MODEL --provider PROVIDER --setting codec=chat check ...`; `--setting KEY=VALUE` reaches any driver setting, and local flags (`--device`, `--repack/--no-repack`, ...) apply only when given. 

## Local execution

Local scoring uses serial decoding and separate KV streams. A request is bounded by `max_input_tokens` (evidence, default 1024), `max_prefix_tokens` (chat template plus guidance, default 1024) and a fixed 1536-token question suffix, including framing. The default logical window is therefore 3584 tokens per stream. `n_ctx` can reduce that per-stream window; it cannot exceed the sum of the caps. To support longer evidence or guidance, raise the corresponding cap. Overlong native requests fail rather than being silently truncated. Authored field-level `max_chars` truncation is a separate application setting.

Physical allocation rounds each stream to native padding and multiplies by `prefix_slots + 2`, including full SWA storage. Four prefix slots allocate six streams; fewer slots save memory without reducing the logical request window. `Runtime.info()` reports realized settings. Native exports and locks bind the execution identity; changing batching, quantization, native build or cache layout requires a new measured comparison and the appropriate lock regeneration.

Determinism is scoped to the recorded model, build and execution shape. On the measured Gemma4/Q8_0 base, CPU and Vulkan first-touch, warm, prefix-residency and whole-sequence controls match exactly. Runtime-adapter repeat/residency controls pass on that Vulkan build, but whole-sequence readout differs by up to 0.017315 for the measured adapters. Changing decode chunk size also changes scores. These are separate execution paths, not a cross-hardware or cross-shape bit-equality promise. See [recorded measurements](ga-measurements.md#full-printer-specialization) and [native diagnostics](../bench/README.md#execution-shape-benchexecution_shapepy).

Grouped decoding remains an explicit backend experiment. Earlier controls were slower; a later printer diagnostic found a workload-specific gain with changed probabilities. Neither establishes a general production throughput result. The production scorer remains serial; use [benchmark commands and recorded scope](../bench/README.md) when comparing execution modes.
