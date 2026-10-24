# Legal biprediction weight sums

While integrating compiler improvements, we found a missing constraint in the
original enumeration: explicit B prediction requires
`-128 <= w0+w1 <= (127 if denominator == 7 else 128)`, in addition to each weight
fitting a signed byte. The original certificate counted some parameter pairs
outside that legal range.

The corrected complete enumeration contains **402,311 classes**:

| Class | Original count | Correct legal count |
| --- | ---: | ---: |
| P | 4,096 | 4,096 |
| B, distinct sources | 524,288 | 394,105 |
| B, repeated source | 8,176 | 4,110 |

All **16 saved functions, instruction witnesses, scalar minima, and packed
minima are identical** after correction. Every saved atlas witness already
satisfied the weight-sum constraint. The parity lower-bound arguments remain
valid: restricting the original larger grammar cannot invalidate an
impossibility result, and the legal witnesses still attain those bounds.
The decoder streams, captures, and measured execution results remain unchanged.

[atlas.json](assets/atlas.json) now contains the corrected certificate and model.
Its internal certificate digest is
`ba35f524e9048a0c2c70141204ef4181f89b915011a5aa0b9d83612c8c7979a9`.
The original bytes are preserved in [atlas-legacy.json](assets/atlas-legacy.json),
matching the atlas hashes in the historical bundle and decoder reports. That
legacy certificate records the earlier overbroad grammar and intentionally
fails the corrected verifier; it is retained for provenance only. Historical
proof source bytes remain available at the source commit recorded by each bundle.

`atlas.py` and [PROOFS.md](PROOFS.md) now use the legal grammar. The shared writer
rejects invalid B sums, synthesis skips them, and affine fusion preserves legal
stages when combining them would violate this constraint. Generic compiler
NOR/NAND synthesis also had invalid parameter choices, although the saved atlas
demonstrations did not; new regressions check their replacement parameters and
actual decoded truth tables.

The restriction is specified in H.264 weighted sample prediction,
[ITU-T H.264, equation 8-242](https://www.itu.int/rec/dologin_pub.asp?id=T-REC-H.264-200305-S!!PDF-E&lang=e&type=items).

```sh
python3 08-decoder-compute-limits/atlas.py \
  --verify 08-decoder-compute-limits/assets/atlas.json
```
