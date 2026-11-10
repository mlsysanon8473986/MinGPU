# MinGPU

MinGPU plans the training of a language model from a **target loss**. Given the loss, a GPU type and
one budget, it returns the GPU count, the model size, the tokens, the parallel plan, its MFU and the
training time. It evaluates each plan with [Calculon](https://github.com/calculon-ai/calculon), an
analytical performance model of LLM training, used unmodified, so no GPU is needed. This repository accompanies a paper under anonymous
review.

## Objectives

| Objective | You give | MinGPU minimizes |
|---|---|---|
| 1, `fewest-gpus` | target loss, GPU type, training window `T` | the GPU count, with training time at most `T` |
| 2, `shortest-training` | target loss, GPU type, GPU budget `N_gpu_max` | the training time on at most `N_gpu_max` GPUs, at an MFU within `delta` (15%) of the highest |

- **Objective 1** takes the smallest GPU count on which some candidate model trains within `T`.
  Ties go to the shorter training, then to the smaller model. Only GPU counts on which the
  compute-optimal model would need an MFU of 30% to 70% to finish within `T` are allowed.
- **Objective 2** finds `U*`, the highest MFU on `N_gpu_min` to `N_gpu_max` GPUs. Among the plans
  with an MFU of at least `(1 - delta) U*`, it takes the shortest training. Ties go to the higher
  MFU, then to fewer GPUs, then to the smaller model.

## Quick start

```sh
git clone --recursive <repository> MinGPU && cd MinGPU
uv venv .venv && uv pip install --python .venv/bin/python -e ".[plot]" && . .venv/bin/activate
mingpu preview-search fewest-gpus --target-loss 1.89255 --max-days 78   # what a search would cover
mingpu fewest-gpus --target-loss 1.89255 --max-days 78                  # search and answer
mingpu show-answer <search_id>                                          # read the answer again
```

## Install

Python 3.11 or later. Calculon is a git submodule, pinned to the commit the paper used:

```sh
git clone --recursive <repository> MinGPU
cd MinGPU
uv venv .venv
uv pip install --python .venv/bin/python -e ".[plot]" pytest ruff    # Windows: .venv\Scripts\python.exe
```

`.[plot]` adds matplotlib, needed only for `plot-search`. Without `uv`, install Calculon first
(`pip install -e vendor/calculon`), then MinGPU (`pip install -e ".[plot]"`).

## Usage

**1. Preview.** `preview-search` generates and counts the parallel plans, but evaluates none and
writes nothing; it takes about a minute. It reports the candidate models, the candidate GPU counts
and, per model, the plans before and after the restrictions. The last count is what the search would
evaluate.

```sh
mingpu preview-search fewest-gpus --target-loss 1.89255 --max-days 78
mingpu preview-search shortest-training --target-loss 1.89255 --max-gpus 24576
```

**2. Search.** Each objective is a command. It reports each pair as it finishes and ends with the
best triple of each candidate model and the answer over all of them.

```sh
mingpu fewest-gpus --target-loss 1.89255 --max-days 78 --workers 8 --db mingpu.duckdb
mingpu shortest-training --target-loss 1.89255 --max-gpus 24576 --workers 8 --db mingpu.duckdb
```

Each pair is stored as soon as it is done and never evaluated again in the same database, so an
interrupted search, or one with another budget, reuses it. To recompute, use a new database file.

**3. Read the results.** These commands read the database and evaluate nothing.

```sh
mingpu list-searches --db mingpu.duckdb                  # every search, with its id and objective
mingpu show-answer <search_id> --db mingpu.duckdb        # its answer
mingpu show-search <search_id> --db mingpu.duckdb        # the best parallel plan of each pair
mingpu plot-search <search_id> --db mingpu.duckdb        # MFU and training time, a PNG in plots/
```

## Options

| Option | Meaning | Default |
|---|---|---|
| `--target-loss` | target loss `L*` (required) | |
| `--max-days` | `fewest-gpus`: training window `T`, in days (required) | |
| `--mfu-range` | `fewest-gpus`: MFU range `lo:hi` that bounds the GPU counts | `0.3:0.7` |
| `--max-gpus` | `shortest-training`: GPU budget `N_gpu_max` (required) | |
| `--min-gpus` | `shortest-training`: smallest GPU count | `--max-gpus` / 3 |
| `--mfu-tolerance` | `shortest-training`, searches only: MFU tolerance `delta` | 0.15 |
| `--gpu`, `-g` | `h100`, or the path of a system JSON | `h100` |
| `--gpu-step` | step between candidate GPU counts | 256 |
| `--seq-size`, `--batch-size` | sequence length (tokens), largest global batch (sequences) | 8,192, 2,048 |
| `--vocab`, `--datatype` | vocabulary size; `float8`, `float16` or `bfloat16` | 128,256, `bfloat16` |
| `--law` | loss law, `besiroglu` or `chinchilla` | `besiroglu` |
| `--no-tp-overlap` | exclude the overlap of tensor-parallel communication | allowed |
| `--workers`, `-w` | worker processes, below normal priority | physical cores, up to 8 |
| `--db` | database file (searches only) | `./mingpu.duckdb`, or `$MINGPU_DB` |
| `--top-k`, `--all-plans` | candidate parallel plans stored per pair (searches only) | 10 |
| `--json` | print the answer as JSON (searches only) | |

The candidate models are `l_C ± 32` layers, in steps of 8, around the compute-optimal model `l_C`.
The candidate GPU counts are the multiples of `--gpu-step` within the window's range (Objective 1),
or from `--min-gpus` to `--max-gpus`, plus `--max-gpus` (Objective 2).

## The experiments of the paper

The two validations are stored in `experiments/mingpu.duckdb`. The target loss is the loss the law
assigns to the parameters and tokens of the published model.

| Experiment | Target loss | Candidate models | GPU counts, Objective 1 | GPU counts, Objective 2 |
|---|---|---|---|---|
| Llama 3.1 405B | 1.8926 | `l = 112` to `176` | every 512, 7,680 to 17,408 (180 pairs) | every 512, 8,192 to 24,576 (297 pairs) |
| Nemotron-4 340B (no TP overlap) | 1.9023 | `l = 96` to `160` | every 256, 3,584 to 7,936 (162 pairs) | every 256, 2,560 to 7,680 (189 pairs) |

Run one command at a time, because a DuckDB file accepts a single writer; a full search can take
hours. The two searches of an experiment share most pairs: together they cover 306 pairs for Llama
and 198 for Nemotron. In PowerShell, write each command on one line.

### Llama 3.1 405B

```sh
# Preview
mingpu preview-search fewest-gpus --target-loss 1.89255 --max-days 78 --gpu-step 512 \
  --seq-size 8192 --batch-size 2048 --vocab 128256 --workers 8
mingpu preview-search shortest-training --target-loss 1.89255 --max-gpus 24576 --min-gpus 8192 \
  --gpu-step 512 --seq-size 8192 --batch-size 2048 --vocab 128256 --workers 8

# Objective 1: fewest GPUs within 78 days
mingpu fewest-gpus --target-loss 1.89255 --max-days 78 --gpu-step 512 \
  --seq-size 8192 --batch-size 2048 --vocab 128256 --workers 8 --db experiments/mingpu.duckdb
# Objective 2: shortest training on 8,192 to 24,576 GPUs
mingpu shortest-training --target-loss 1.89255 --max-gpus 24576 --min-gpus 8192 --gpu-step 512 \
  --seq-size 8192 --batch-size 2048 --vocab 128256 --workers 8 --db experiments/mingpu.duckdb
```

### Nemotron-4 340B

```sh
# Preview
mingpu preview-search fewest-gpus --target-loss 1.90225 --max-days 88 --gpu-step 256 \
  --no-tp-overlap --seq-size 4096 --batch-size 2304 --vocab 256000 --workers 8
mingpu preview-search shortest-training --target-loss 1.90225 --max-gpus 7680 --min-gpus 2560 \
  --gpu-step 256 --no-tp-overlap --seq-size 4096 --batch-size 2304 --vocab 256000 --workers 8

# Objective 1: fewest GPUs within 88 days
mingpu fewest-gpus --target-loss 1.90225 --max-days 88 --gpu-step 256 --no-tp-overlap \
  --seq-size 4096 --batch-size 2304 --vocab 256000 --workers 8 --db experiments/mingpu.duckdb
# Objective 2: shortest training on 2,560 to 7,680 GPUs
mingpu shortest-training --target-loss 1.90225 --max-gpus 7680 --min-gpus 2560 --gpu-step 256 \
  --no-tp-overlap --seq-size 4096 --batch-size 2304 --vocab 256000 --workers 8 --db experiments/mingpu.duckdb
```

## Terminology

| Term | Meaning |
|---|---|
| Candidate model | a model of the family, set by its layers `l`, trained on the tokens `D_req` that reach the target loss |
| Candidate GPU count | a GPU count that is searched |
| Candidate parallel plan | of a pair: a parallel plan (an execution of Calculon) that passes MinGPU's restrictions |
| Pair | a candidate model and a candidate GPU count |
| Triple | a pair and one of its candidate parallel plans; the paper's `(N, D, N_gpu, P)`, since the model sets `N` and `D` |
| Best parallel plan | of a pair: the plan that fits in memory with the highest sample rate, hence the shortest training and highest MFU |
| Search | the evaluation of all triples, equivalently of all pairs with their best parallel plan |
| Answer | the triple the objective chooses |

## Python API

```python
from mingpu import BESIROGLU, SearchRequest, run_request

loss = BESIROGLU.loss(405_849_243_648, 15.6e12)        # the loss the law assigns to Llama 3.1 405B
request = SearchRequest(target_loss=loss, objective="fewest_gpus", window_days=78)
result = run_request(request, db_path="mingpu.duckdb", workers=8)
a = result.answer
print(a.layers, a.n_gpu, a.plan, a.mfu, a.t_train_days)
```

`SearchSpace` and `run_outer_search` take explicit models and GPU counts. `count_plans` and `inner_search`
handle a single pair.

## The database

The default database is `./mingpu.duckdb` (or `$MINGPU_DB`, or `--db`). The schema is in `migrations/`.

| Table or view | Content |
|---|---|
| `search` | a search: its request, settings and status |
| `candidate_models` | the candidate models of a search, with `N`, `D_req` and `C_req` |
| `cache_pairs` | the pair cache: the plans left after each restriction step |
| `inner_search_pair` | the top-k candidate parallel plans of each cached pair, with Calculon's statistics |
| `search_pair` | the pairs of each search, and whether they came from the cache |
| `best_per_pair` (view) | the best parallel plan of every pair with its MFU, MFU on the `6N` basis, training time and GPU-hours |
| `search_progress` (view) | the progress of each search |
| `fewest_gpus(id, T)`, `shortest_training(id, N_gpu_min, N_gpu_max, delta)` | the two objectives as table macros, with `_per_model` variants |

```sql
SELECT layers, n_gpu, mfu, t_train_days FROM best_per_pair WHERE search_id = '<id>' ORDER BY layers, n_gpu;
SELECT * FROM shortest_training('<id>', 8192, 24576, delta := 0.15);
```

## Development and license

```sh
.venv/bin/python -m pytest                # Windows: .venv\Scripts\python.exe
.venv/bin/ruff check src tests
.venv/bin/ruff format src tests
```

Apache License 2.0 (`LICENSE`). Calculon is Apache-2.0 too, by its authors; see `NOTICE`.
