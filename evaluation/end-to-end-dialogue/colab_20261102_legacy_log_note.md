# Historical Colab output recovered on 4 November 2026

This is a **saved-notebook log observation**, not an independently recomputed
result. It preserves what the old `ad0f75d47c5da3f38cee280d3c449e6e5704f82e`
run printed. It is not the current source-preserving v2 endpoint.

The saved output reports completion of 1,540 LoCoMo questions, 4,620 responses
and 4,620 adapted reference-judge labels across three conditions.

| Condition | Question-micro accuracy as logged | Equal-group accuracy as logged |
| --- | ---: | ---: |
| No history | 0.5195% | 0.5675% |
| Short-term history | 3.8961% | 4.0296% |
| Short-term plus unconfirmed candidate evidence | 52.9221% | 54.0341% |

The logs declare zero technical failures in each condition. These numbers are
transcribed from console output; the original prompts, responses and judge
artifacts were unavailable for this check. The old runtime directory was absent
on 4 November. Declared archive names and hashes are retained in the adjacent
[observation record](colab_20261102_legacy_log_observation.json), but those ZIP
bytes were not verified. The downloaded notebook identity is recorded there;
the complete notebook and its account metadata are not published.

After LoCoMo, the same cell stopped while preparing LongMemEval-S with
`ValueError: duplicate session identity within a question`. It did not report a
LongMemEval result.

The large logged gap is evidence worth recovering and rerunning, not grounds for
a current effectiveness claim. This old experiment retrieved verbatim,
unconfirmed candidate evidence; it did not validate semantic CBT memory, user
consent or treatment outcomes. LoCoMo has only ten conversation groups and prior
development exposure. The adapted same-family model judge is not a human or
clinical evaluation. Keep these log values separate from new v2 runs and do not
replace raw-artifact verification with console aggregates.
