# Experiments

The two validations of the paper, Llama 3.1 405B and Nemotron-4 340B. Their four searches are
stored in `mingpu.duckdb`; the figures go to `plots/`.

Run the commands from the root of the repository, with the virtual environment active:

```powershell
.venv\Scripts\Activate.ps1          # Linux and macOS: source .venv/bin/activate
```

The searches are already run. They evaluate the parallel plans with Calculon and take hours; run
one at a time, since a DuckDB file accepts a single writer. Running one again on this database
creates a new search from the stored pairs. The other commands only read the database.

```powershell
# every search, with its id, objective and pairs
mingpu list-searches --db experiments/mingpu.duckdb
```

`show-answer` gives the best triple of each candidate model and the answer; `show-search`, the
best parallel plan of every pair. Add `--json` to either for JSON. `plot-search` draws MFU and
training time against the GPU count, one curve per candidate model, as a PNG in `plots/`.

## Llama 3.1 405B

Target loss 1.89255, sequence 8,192, batch 2,048, vocabulary 128,256, candidate models
`l = 112` to `176`.

| Objective | Budget | Search id | Pairs | Answer |
|---|---|---|---|---|
| 1, `fewest-gpus` | 78 days, every 512 GPUs from 7,680 to 17,408 | `2d908020-bad0-4273-887f-489cdf60c829` | 180 (171 from the cache) | `l = 120` on 14,336 GPUs, (t, p, d) = (8, 4, 448), v 30, MFU 46.31%, 77.8 days |
| 2, `shortest-training` | every 512 GPUs from 8,192 to 24,576 | `81078e8a-2b39-479f-8972-d36663c664f3` | 297 | `l = 128` on 16,384 GPUs, (t, p, d) = (8, 8, 256), v 16, MFU 46.50%, 62.1 days; U* = 47.92% |

```powershell
# Searches
mingpu fewest-gpus --target-loss 1.89255 --max-days 78 --gpu-step 512 --seq-size 8192 --batch-size 2048 --vocab 128256 --workers 8 --db experiments/mingpu.duckdb
mingpu shortest-training --target-loss 1.89255 --max-gpus 24576 --min-gpus 8192 --gpu-step 512 --seq-size 8192 --batch-size 2048 --vocab 128256 --workers 8 --db experiments/mingpu.duckdb

# Objective 1: fewest GPUs within 78 days
mingpu show-answer 2d908020-bad0-4273-887f-489cdf60c829 --db experiments/mingpu.duckdb
mingpu show-search 2d908020-bad0-4273-887f-489cdf60c829 --db experiments/mingpu.duckdb
mingpu plot-search 2d908020-bad0-4273-887f-489cdf60c829 --db experiments/mingpu.duckdb --out experiments/plots --name llama_fewest_gpus

# Objective 2: shortest training on 8,192 to 24,576 GPUs
mingpu show-answer 81078e8a-2b39-479f-8972-d36663c664f3 --db experiments/mingpu.duckdb
mingpu show-search 81078e8a-2b39-479f-8972-d36663c664f3 --db experiments/mingpu.duckdb
mingpu plot-search 81078e8a-2b39-479f-8972-d36663c664f3 --db experiments/mingpu.duckdb --out experiments/plots --name llama_shortest_training
```

## Nemotron-4 340B

Target loss 1.90225, sequence 4,096, batch 2,304, vocabulary 256,000, no TP overlap, candidate
models `l = 96` to `160`.

| Objective | Budget | Search id | Pairs | Answer |
|---|---|---|---|---|
| 1, `fewest-gpus` | 88 days, every 256 GPUs from 3,584 to 7,936 | `bc0004a2-177c-49d4-b3e8-a8b603ce1af6` | 162 (153 from the cache) | `l = 136` on 5,888 GPUs, (t, p, d) = (8, 4, 184), v 17, MFU 44.40%, 83.6 days |
| 2, `shortest-training` | every 256 GPUs from 2,560 to 7,680 | `38d5c25a-a637-4303-8a72-efdd4741417a` | 189 | `l = 104` on 7,680 GPUs, (t, p, d) = (4, 4, 480), v 26, MFU 42.27%, 79.4 days; U* = 48.37% |

```powershell
# Searches
mingpu fewest-gpus --target-loss 1.90225 --max-days 88 --gpu-step 256 --no-tp-overlap --seq-size 4096 --batch-size 2304 --vocab 256000 --workers 8 --db experiments/mingpu.duckdb
mingpu shortest-training --target-loss 1.90225 --max-gpus 7680 --min-gpus 2560 --gpu-step 256 --no-tp-overlap --seq-size 4096 --batch-size 2304 --vocab 256000 --workers 8 --db experiments/mingpu.duckdb

# Objective 1: fewest GPUs within 88 days
mingpu show-answer bc0004a2-177c-49d4-b3e8-a8b603ce1af6 --db experiments/mingpu.duckdb
mingpu show-search bc0004a2-177c-49d4-b3e8-a8b603ce1af6 --db experiments/mingpu.duckdb
mingpu plot-search bc0004a2-177c-49d4-b3e8-a8b603ce1af6 --db experiments/mingpu.duckdb --out experiments/plots --name nemotron_fewest_gpus

# Objective 2: shortest training on 2,560 to 7,680 GPUs
mingpu show-answer 38d5c25a-a637-4303-8a72-efdd4741417a --db experiments/mingpu.duckdb
mingpu show-search 38d5c25a-a637-4303-8a72-efdd4741417a --db experiments/mingpu.duckdb
mingpu plot-search 38d5c25a-a637-4303-8a72-efdd4741417a --db experiments/mingpu.duckdb --out experiments/plots --name nemotron_shortest_training
```
