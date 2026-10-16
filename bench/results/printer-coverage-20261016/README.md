# Printer coverage comparison — 2026-10-16

Start with `summary.json`. Both deployed arms match all 560 complete action references, while known labels on the additional 176 incomplete-reference situations expose 4/8 prohibited-action situations (random/targeted). Precision is 99.5%/99.1%; required-action recall is 92.5%/97.4%. This is synthetic regression evidence, not fresh real-language confirmation. No adoption was performed.

`publication.json` records relative payload names, byte counts and SHA-256. `comparison.json.gz` and `comparison.journal.jsonl.gz` preserve the original report and row journal losslessly; decompress them for the original schemas. Original embedded paths remain provenance, not portable executable paths. The publication's relative inventory locates retained copies. Per-arm deployment metadata preserves the selected model hashes; weights and full recovery state are retained in the external archive named by the publication, not in Git.

The original corpus is `packs/zt411/`; additive corpora are `packs/zt411-coverage-random-20261015/` and `packs/zt411-coverage-targeted-20261015/`. `cohorts.json` retains original run locations and hashes. Use the published pack locations for read-only inspection. Scripts are historical source evidence, not another maintained runner; the matching pre-refactor source is retained in `source-7909dfe.tar.zst`.

The full archive is locally retained outside this checkout. Copy it to your own durable storage for another machine or disaster recovery; a Git clone contains the report and small evidence, not the trained weights.
