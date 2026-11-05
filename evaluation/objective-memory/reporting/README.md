# Post-run objective reporting

These are post-run reporting tools, copied after the completed 2026-11-05 run.
They are not part of its frozen generation executable or protocol hashes and
must not be inserted into an old run's executable checkout.

The actual run used executable `1511706b3a4b0ac71416998b5b4d752a13c4e313` and
suite plan `940a2b3d8fa02aa683f7855f94931ff1d9710c7835b598c70fc201582ae8d568`.
[Results](../results/full-20261105/README.md) retain all 1,389 cases / 4,167 arm
rows, including failures in the planned denominator.

## Controlled export inside genuine hosted Colab

`colab_export_derived.py` requires genuine Colab before reading experimental
inputs. With the clean frozen checkout and completed run directory, it validates
existing protocols, full journals and source identities under read-only locks.
It recomputes deterministic per-question scores and actual canonical attempt
costs, then writes a fresh directory. It does not repair the live run, call a
model, read a Secret or overwrite an existing export.

```bash
python /content/colab_export_derived.py \
  --repo /content/cbt-objective-v3.1-authorized-1511706b3a4b \
  --work /content/cbt-v3.1-authorized-runall-20261105 \
  --out /content/cbt-v3.1-authorized-runall-20261105/derived-final-20261105 \
  --final
```

The actual export helper's SHA-256 is
`d30e355265dedd7e1018c08061dd3aaf64f88e52109d4afefbf66af7b1d53632`.
The derived directory contains controlled original model replies and score/cost
metadata, but no benchmark question text, options, source histories or gold.
Do not publish the whole derived directory by default.

## Derived-only report and public projection

After controlled cloud transfer and checksum verification:

```bash
python build_full_objective_report.py \
  --input-dir /absolute/cloud/path/derived-final-20261105 \
  --output-dir /absolute/cloud/path/new-report
python export_public_objective_results.py \
  --input-dir /absolute/cloud/path/derived-final-20261105 \
  --report-summary /absolute/cloud/path/new-report/report_summary.json \
  --output-dir /absolute/cloud/path/new-public-results
```

The report builder requires the complete fixed cohort and all named independent
audit checks. It verifies original reply checksums, scores, paired contrasts,
group boundaries and canonical costs. It produces Chinese HTML, per-question CSV,
an audit summary and checksums without any model calls. It reads derived metadata
only, not full source data or scoring gold. Its actual SHA-256 is
`336fcb315b46ab5bfbcf3f9e389a566987e045942dfca1dbc553aab14108b459`.

The public exporter binds its projection to that report and its exact input
bytes. It recursively whitelists identifiers, scores, statuses and numeric costs;
question text, options, histories, reference answers, original model replies and
private source trace text are excluded. Its actual SHA-256 is
`94fc5a652362f21bc3d861fc05d382a45da7356d251c5b30d84cd20ddb5d224f`.
The public files in this repository are the unchanged output of that exporter.

These checks establish reproducible file identity and aggregation, not independent
CBT professional assessment. Content-error causes remain unresolved without
independent evidence annotations. Zero human CBT ratings, actual four-visit
participants and clinical outcomes remain separate from public-task results.
