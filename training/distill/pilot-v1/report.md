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
| matching_rate | 0.7044 -> 0.7046 | 0.919 -> 0.9168 | 0.9731 -> 0.9462 | 0.729 -> 0.7291 |
| alignment_gap | 0.0 -> 0.0 | 0.0049 -> 0.0 | 0.0078 -> 0.0025 | 0.0017 -> 0.0003 |
| variation_rate | 0.0 -> 0.0 | 0.0074 -> 0.0035 | 0.0106 -> 0.0071 | 0.0017 -> 0.0007 |
| word_compressed_rate | 0.7046 -> 0.7046 | 0.9228 -> 0.9168 | 0.9731 -> 0.9462 | 0.7319 -> 0.7299 |

## Teacher

- model pin: `deepseek/deepseek-v4-pro` (served: deepseek/deepseek-v4-pro)
- prompt: llmlingua2-paper-v1

## Gateway billing

- fresh calls: 0, replayed: 299
- crash journal: `runs/.ledger/ (content-addressed, gitignored runtime ledger)` (ledger `4fa4734ddd57fcec`)
