# Published corpora

Published corpora are reusable input data, not training scratch. All domains under this directory are visible to Git. Keep each `benchmark.json` with its content-addressed situations, annotations, references, labels, definition and review files. A manifest alone is not a preserved corpus.

- `zt411/`: original printer corpus and role assignment; unchanged by the coverage experiment.
- `zt411-coverage-random-20261015/`: separately published random fit additions.
- `zt411-coverage-targeted-20261015/`: separately published targeted fit additions.
- `support/`: published support corpus, previously present only as ignored local files.

Training consumes publications; it does not regenerate them. Token exports depend on model, tokenizer and task presentation and belong in work/cache storage. New source generation, new labels or a changed assignment produce a separate publication. Review history is evidence and is not disposable merely because labels can be recomputed.

Fetched manuals and other proprietary originals remain outside Git under `bench/data/*/sources/`. Their acquisition and redistribution permissions are separate from the library license. The inclusion of computed labels does not imply that a reference has been independently reviewed.
