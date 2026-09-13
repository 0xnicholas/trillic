# Teacher distillation run

## Scale

- corpus entries: 1000 (selected 115)
- chunks compressed: 299
- labeled kept: 255 (dropped 44)

## Quality control

- Variation Rate filter: top 5% dropped (threshold 0.0106)
- Alignment Gap filter: top 10% of survivors dropped (threshold 0.0049)

## Alignment distribution (population -> kept)

| metric | p50 | p90 | p95 | mean |
|---|---|---|---|---|
| matching_rate | 0.7044 -> 0.7046 | 0.91904 -> 0.91678 | 0.9731299999999996 -> 0.9462399999999997 | 0.729 -> 0.7291 |
| alignment_gap | 0.0 -> 0.0 | 0.0049 -> 0.0 | 0.0078 -> 0.0025 | 0.0017 -> 0.0003 |
| variation_rate | 0.0 -> 0.0 | 0.0074 -> 0.0034599999999999995 | 0.010609999999999996 -> 0.007089999999999995 | 0.0017 -> 0.0007 |
| word_compressed_rate | 0.7046 -> 0.7046 | 0.9228 -> 0.91678 | 0.9731299999999996 -> 0.9462399999999997 | 0.7319 -> 0.7299 |

## Teacher

- model pin: `deepseek/deepseek-v4-pro` (served: deepseek/deepseek-v4-pro)
- prompt: llmlingua2-paper-v1

## Gateway billing

- fresh calls: 296, replayed: 3
- crash journal: `/Users/nicholasl/Documents/build-whatever/trillic/runs/.ledger/distill-4fa4734ddd57fcec.jsonl` (ledger `4fa4734ddd57fcec`)
