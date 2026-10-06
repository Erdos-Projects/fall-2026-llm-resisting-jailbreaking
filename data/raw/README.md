# Raw data

All 10 original JSON files used in this pilot are included under `attack-artifacts/<method>/<attack_type>/<target_model>.json`. They preserve the upstream bytes and every field, including `prompt`, `response`, behavior/goal text, query and token counts, and `jailbroken`. The 1,000 source slots include the 114 empty pairs excluded from modeling.

From the package root, verify the bundled files and rebuild the two metadata CSVs:

```sh
python src/data/acquire.py
# Optional alternate raw-data destination:
python src/data/acquire.py --raw-dir /your/data/directory
```

The default is persistent `data/raw/`. Existing files are hash-checked; missing files are downloaded before parsing. A successful run updates acquisition timestamps and derived-file hashes in `provenance.json`. Unexpected bytes fail the run without changing expected source hashes. The upstream README entry records source documentation; it is not a dataset file or downloaded over this README. The original LICENSE is stored as `UPSTREAM_LICENSE.txt`.

Raw files contain harmful or offensive benchmark text. Treat it as research data, not executable instructions. Preserve the MIT license and [source attribution](../../THIRD_PARTY_NOTICES.md) with redistributed originals or derived data. The baseline continues to use neutral metadata only.
