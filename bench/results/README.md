# Published benchmark evidence

This directory preserves selected completed reports and the evidence needed to inspect them. Active runs, download caches and regenerable training exports belong under ignored `work/` or `.cache/`. A report is not a production quality guarantee: read its input scope, model/build identity, acceptance policy, denominators and reference status.

- [Printer coverage comparison, 2026-10-16](printer-coverage-20261016/README.md): both deployed arms, fixed check population, declarations, training evaluations and compressed full report/journal. Trained models and recovery state are in the verified external archive named by its publication.
- [Native diagnostic payload index](payloads/index.json): relative copies of required NPZ evidence, with hashes. Existing native-segmentation reports retain their original bytes and historical paths; use this index to locate payloads in another checkout.
- [Benchmark catalog](../README.md): maintained commands, provenance, limitations and superseded results.

[Frozen export publications](../data/exports/README.md) are measurement fixtures kept with their manifests under `bench/data/exports/`, separate from reports. Their index maps old report references to preserved payloads. Ordinary content-addressed exports are ignored. Do not prune a payload still named by a retained publication. Historical paths are provenance; missing heavyweight models require separately archived bundles, not an automatic corpus rebuild.
