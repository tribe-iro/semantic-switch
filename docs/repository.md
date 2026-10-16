# Repository layout and preservation

## Owners

| Location | Owns | Git policy |
|---|---|---|
| `semantic_switch/` | Runtime SDK, rules, drivers, labels and artifact contracts | Source |
| `packages/semantic-trainer/` | Fitting, selection and recovery | Source; no runtime dependency |
| `packages/semantic-learn/` | Explicit capture, collection, review and rounds | Source; no background learning |
| `packages/semantic-testkit/` | Shared test fixtures and controlled judges | Development only |
| `examples/` | Runnable public SDK examples and small definitions | Source |
| `specialists/` | Reusable application definitions and authored scenario sources | Source |
| `bench/` | Maintained benchmark runners and source materializers | Source |
| `bench/results/` | Selected published measurements and their evidence | Preserve; see its index |
| `bench/results/payloads/` | Required binary evidence named by the payload index | Preserve |
| `packs/` | Published reusable corpora, annotations, labels, reviews and role assignments | Preserve all domains |
| `experiments/` | Explicit exploratory runners, including routing acquisition | Source; not an SDK path |
| `archives/` | Historical acquisitions and inventories | Preserve; unsupported schemas stay historical |
| `docs/history/` | Superseded specifications | Historical, never current usage guidance |
| `work/` | Active runs, resumable journals, intermediate exports and checkpoints | Ignored; not a backup |
| `.cache/`, `.artifacts/`, `.venv*/` | Downloads, temporary model artifacts and environments | Ignored |

`models.py` owns local model configuration and source resolution; `model_jobs.py` owns opt-in admission; `api.py` owns Runtime and convenience calls. `evidence.py` owns label envelopes, claims and ingestion. Public SDK imports remain under `semantic_switch`; trainer integration remains under `semantic_switch.training`.

## Completed work is not scratch

Generate or materialize data explicitly, publish it under `packs/`, then train against that immutable publication. Never regenerate a corpus as a fitting prerequisite. New corpus domains are not ignored. Include each manifest's content-addressed payloads and review history. Fetched proprietary source documents remain outside Git; source metadata does not grant redistribution rights.

Keep active outputs under `work/`. Before treating a completed run as preserved, publish its report, declarations, input identities and required row evidence under `bench/results/`, and archive its model bundles and recovery state outside the checkout. Do not claim a global stage cache: reuse is scoped to each work root and its recorded identities. Do not remove a live or sole-copy run to save space.

The printer coverage comparison is published under `bench/results/printer-coverage-20261016/`. Its two additional corpora are under `packs/zt411-coverage-{random,targeted}-20261015/`; the original `packs/zt411/` is unchanged. A verified external archive preserves the full experiment, including deployed models. The publication records its archive digest and filename, not a machine-specific restore path. That local archive is protection against checkout cleanup, not off-machine disaster recovery.

Historical reports are not rewritten to make old paths look current. `bench/results/payloads/index.json` maps retained historical NPZ references to relative, digest-checked copies. Generated exports are normally ignored; frozen measurement fixtures under `bench/data/exports/` and the routing acquisition archive are explicit exceptions.

## Development installation

From the repository root, create an environment and install the PyTorch wheel for its device first. For a CPU test environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` installs the core extras, shared testkit, learner and trainer. The root `dev` extra alone only adds pytest; it is not the complete multi-package test environment. Accelerator fitting uses its own environment and matching PyTorch wheel; see the trainer README. GPU tests and benchmarks require a GPU-capable llama.cpp build.

```bash
.venv/bin/python -m pytest -m "not slow"
.venv/bin/python -m pytest -c packages/semantic-trainer/pyproject.toml packages/semantic-trainer/tests -m "not slow"
.venv/bin/python -m pytest -c packages/semantic-learn/pyproject.toml packages/semantic-learn/tests -m "not slow"
```

The first command covers core, repository integration, experiments and specialist tests. For CPU miniature-model fixtures, prefix a test command with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` to avoid thread-pool oversubscription; these are test-process settings, not inference defaults. The second and third run the optional packages separately, each with its own test fixtures. Real-model tests are explicitly selected with `-m slow` in an appropriate environment and must not compete with an active model job. Neither application wheel ships the testkit, benchmarks or corpora.

## What the tests establish

Numerical tests compare probability reduction, losses and gradients with independent reference calculations. Kernel tests specify three-valued logic. Persistence, concurrency and recovery tests exercise real files, threads, processes and miniature CPU training. The miniature training fixture simulates a single CPU core to avoid oversized thread pools; production fitting defaults are unchanged. Scripted scorers make SDK, transport and controller behavior repeatable; they do not establish language understanding. README tests with scripted scores check API usage and dispatch.

Controller integration fixtures replace model conversion and final scoring. Their successful promotion means the controller handled its inputs correctly, not that a trained model improved. The real deployment round trip creates a tiny local Llama and a nonzero LoRA, uses the production converter, merger and Q8_0 quantizer, then compares native probabilities against the PyTorch model. It also checks that a zero adapter reproduces the base and that the nonzero adapter changes the native scores:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -c packages/semantic-trainer/pyproject.toml packages/semantic-trainer/tests/test_native_deployment.py
```

This test needs the pinned conversion toolchain already installed under the trainer asset root and the development dependencies. It skips explicitly when that toolchain is absent; it downloads no weights. It validates weight transport and numerical agreement, not task quality. Native history and request-order tests compare executions within the same configuration; a different execution path is not interchangeable merely because its probabilities are approximately close.

Passing counts are software verification results. Model quality comes from separately reported evaluations with their corpus, split, reference and exposure history. The optional `--baseline PATH` report records selected cases, their `slow` marker and outcomes; that marker describes requirements, not whether a test uses mocks or proves semantic quality.

## Documentation and licensing

README owns supported usage and current result summaries; `docs/invariants.md` owns implementation contracts; `bench/README.md` owns measurement scope; TODO owns unfinished work. Historical documents are labeled and retained under `docs/history/`.

No project license was declared in the inherited tree. A maintainer must choose the license before an open-source distribution; third-party model, source-document and dataset licenses remain separate. Do not infer a license from a historical description of the project as open source.
