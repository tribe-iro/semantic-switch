# Frozen export measurement fixtures

These complete export publications were moved byte-for-byte from `bench/results/` to separate inputs from reports. Their manifests and content-addressed payloads remain together. `index.json` maps historical report references to relative paths and SHA-256. Recorded report paths remain historical; use these fixture locations for current inspection with `semantic_switch.training.read_export`.

These are deliberately retained measurement fixtures. Ordinary training exports belong under ignored `work/` and are not publications merely because a benchmark generated them. Do not rerun generation to reconstruct their history or remove payloads still named by their manifests.
