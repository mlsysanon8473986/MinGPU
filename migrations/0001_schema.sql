-- mingpu schema: planner searches (paper, Sec. 4).
--
-- A search is a set of candidate models (layer counts) and candidate GPU counts for one
-- target loss, batch, sequence length, system and restriction set. Each pair
-- (model, GPU count) is searched once by mingpu.planner.search.inner_search and kept in a
-- content-addressed cache (cache_pairs), with the executions left after each
-- restriction step and its top-k feasible candidate parallel plans (inner_search_pair). Searches
-- link to the pairs they use (search_pair), so a second search with the same
-- pair reuses it. The objectives are macros: the window, the GPU range and the
-- MFU tolerance are always arguments, never stored columns.

-- One planner search: the request and the fixed settings of all its pairs.
CREATE TABLE search (
  id              UUID PRIMARY KEY,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  status          VARCHAR NOT NULL DEFAULT 'created',  -- created, running, done, error
  error           VARCHAR,
  labels          VARCHAR[] DEFAULT [],
  target_loss     DOUBLE NOT NULL,
  law             JSON NOT NULL,                       -- name and constants of the loss law
  seq_len         INTEGER NOT NULL,                    -- s, tokens per sequence
  b_max           INTEGER NOT NULL,                    -- largest global batch, sequences
  vocab           INTEGER NOT NULL,
  datatype        VARCHAR NOT NULL,
  system          JSON NOT NULL,                       -- Calculon system, the same for every pair
  f_peak          DOUBLE NOT NULL,                     -- peak FLOP/s of one device in datatype
  restrictions    JSON NOT NULL,                       -- enabled steps, t_max, fixed t
  top_k           INTEGER,                             -- executions kept per pair, NULL = all valid
  layers          INTEGER[] NOT NULL,                  -- candidate models
  n_gpus          INTEGER[] NOT NULL,                  -- GPU counts searched
  space           JSON NOT NULL,                       -- everything the search covers
  request         JSON,                                -- what the user asked, for provenance
  mingpu_version  VARCHAR,
  calculon_commit VARCHAR
);

-- The candidate models of a search: N(l), the tokens and compute that reach the target loss.
CREATE TABLE candidate_models (
  search_id      UUID NOT NULL REFERENCES search(id),
  layers         INTEGER NOT NULL,
  application    JSON NOT NULL,                        -- Calculon application of the model
  n_params       BIGINT NOT NULL,                      -- N(l), embeddings included
  d_req          DOUBLE,                               -- NULL when the model cannot reach the loss
  c_req          DOUBLE,                               -- 6 N D_req
  PRIMARY KEY (search_id, layers)
);

CREATE SEQUENCE cache_pairs_id_seq;

-- One restricted Calculon search of one (application, system, GPU count, batch, restriction set).
CREATE TABLE cache_pairs (
  id              BIGINT PRIMARY KEY DEFAULT nextval('cache_pairs_id_seq'),
  sha256          VARCHAR NOT NULL UNIQUE,             -- content key of the inputs
  n_gpu           INTEGER NOT NULL,
  b_max           INTEGER NOT NULL,
  datatype        VARCHAR NOT NULL,
  restrictions    JSON NOT NULL,
  calculon_commit VARCHAR,
  generated       BIGINT NOT NULL,                     -- executions of Calculon's generator
  steps           INTEGER[] NOT NULL,                  -- enabled restriction steps, in order
  left_after      BIGINT[] NOT NULL,                   -- executions left after each step
  kept            BIGINT NOT NULL,
  evaluated       BIGINT NOT NULL,
  valid           BIGINT NOT NULL,                     -- evaluated executions that fit in memory
  top_k           INTEGER,                             -- executions stored, NULL = all valid
  seconds         DOUBLE NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The top-k feasible candidate parallel plans of a pair, rank 1 = the best parallel plan.
CREATE TABLE inner_search_pair (
  pair_id BIGINT NOT NULL REFERENCES cache_pairs(id),
  rank    INTEGER NOT NULL,
  -- Calculon's execution fields (Llm.Execution.fields), with Calculon's names
  num_procs INTEGER, tensor_par INTEGER, pipeline_par INTEGER, data_par INTEGER,
  tensor_par_net INTEGER, pipeline_par_net INTEGER, data_par_net INTEGER,
  batch_size INTEGER, microbatch_size INTEGER, datatype VARCHAR, fused_activation BOOLEAN,
  attention_type VARCHAR, activation_recompute VARCHAR, pipeline_interleaving INTEGER,
  optimizer_sharding BOOLEAN, tensor_par_comm_type VARCHAR, tensor_par_overlap VARCHAR,
  seq_par_ag_redo BOOLEAN, data_par_overlap BOOLEAN, weight_offload BOOLEAN,
  activations_offload BOOLEAN, optimizer_offload BOOLEAN, training BOOLEAN,
  -- curated statistics (mirrors mingpu.storage.pair_cache.EXECUTION_METRICS)
  sample_rate DOUBLE, total_time DOUBLE, total_efficiency DOUBLE, compute_efficiency DOUBLE,
  system_efficiency DOUBLE, bubble_time DOUBLE, fw_time DOUBLE, bw_time DOUBLE,
  optim_step_time DOUBLE, recompute_time DOUBLE, tp_comm_exposed_time DOUBLE,
  pp_comm_exposed_time DOUBLE, dp_comm_exposed_time DOUBLE, weight_space DOUBLE,
  act_space DOUBLE, proc_mem_tier1_cap_req DOUBLE, useful_flops DOUBLE,
  stats JSON NOT NULL,                                 -- all of Llm.get_stats_fields
  PRIMARY KEY (pair_id, rank)
);

-- The pairs of a search; reused is true when the pair came from the cache.
CREATE TABLE search_pair (
  search_id  UUID NOT NULL REFERENCES search(id),
  layers     INTEGER NOT NULL,
  n_gpu      INTEGER NOT NULL,
  pair_id    BIGINT NOT NULL REFERENCES cache_pairs(id),
  reused     BOOLEAN NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (search_id, layers, n_gpu)
);

-- The best parallel plan of every searched pair, with its training time (Eq. time of the paper).
-- mfu is Calculon's total_efficiency (Eq. mfu); mfu_6n counts 6 N FLOPs per token, N with
-- embeddings, the basis of the loss law. A pair without a valid plan has NULL plan columns.
CREATE OR REPLACE VIEW best_per_pair AS
SELECT
  sp.search_id, sp.layers, m.n_params, m.d_req, m.c_req, sp.n_gpu, sp.pair_id, sp.reused,
  p.kept, p.valid,
  e.tensor_par, e.pipeline_par, e.data_par, e.pipeline_interleaving, e.microbatch_size,
  e.activation_recompute, e.optimizer_sharding, e.tensor_par_overlap, e.data_par_overlap,
  e.batch_size, e.total_time, e.sample_rate, e.bubble_time,
  e.bubble_time / e.total_time                                  AS bubble_share,
  e.total_efficiency                                            AS mfu,
  6 * m.n_params * e.sample_rate * s.seq_len / (sp.n_gpu * s.f_peak) AS mfu_6n,
  m.d_req / (e.sample_rate * s.seq_len)                         AS t_train_s,
  t_train_s / 86400                                             AS t_train_days,
  sp.n_gpu * t_train_s / 3600                                   AS gpu_hours
FROM search_pair sp
JOIN search s ON s.id = sp.search_id
JOIN candidate_models m ON m.search_id = sp.search_id AND m.layers = sp.layers
JOIN cache_pairs p ON p.id = sp.pair_id
LEFT JOIN inner_search_pair e ON e.pair_id = sp.pair_id AND e.rank = 1;

-- Objective 1, per model: the fewest GPUs in [n_gpu_lo, n_gpu_hi] on which the model trains within
-- window_days.
CREATE OR REPLACE MACRO fewest_gpus_per_model(sid, window_days, n_gpu_lo := 0, n_gpu_hi := 2147483647) AS TABLE
SELECT * EXCLUDE (pick) FROM (
  SELECT b.*, row_number() OVER (PARTITION BY b.layers ORDER BY b.n_gpu) AS pick
  FROM best_per_pair b
  WHERE b.search_id = CAST(sid AS UUID) AND b.t_train_days <= window_days
    AND b.n_gpu BETWEEN n_gpu_lo AND n_gpu_hi
)
WHERE pick = 1;

-- Objective 1: over all models, the fewest GPUs; ties go to the shorter training.
CREATE OR REPLACE MACRO fewest_gpus(sid, window_days, n_gpu_lo := 0, n_gpu_hi := 2147483647) AS TABLE
SELECT * FROM fewest_gpus_per_model(sid, window_days, n_gpu_lo := n_gpu_lo, n_gpu_hi := n_gpu_hi)
ORDER BY n_gpu, t_train_s, layers
LIMIT 1;

-- Objective 2, per model: U* is the model's highest MFU on n_gpu_min to n_gpu_max GPUs; among its plans
-- with an MFU of at least (1 - delta) U*, the shortest training (ties: higher MFU, fewer GPUs).
CREATE OR REPLACE MACRO shortest_training_per_model(sid, n_gpu_min, n_gpu_max, delta := 0.15) AS TABLE
WITH r AS (
  SELECT * FROM best_per_pair
  WHERE search_id = CAST(sid AS UUID) AND n_gpu BETWEEN n_gpu_min AND n_gpu_max AND mfu IS NOT NULL
), u AS (
  SELECT layers, max(mfu) AS u_star FROM r GROUP BY layers
), band AS (
  SELECT r.*, u.u_star,
         row_number() OVER (PARTITION BY r.layers ORDER BY r.t_train_s, r.mfu DESC, r.n_gpu) AS pick
  FROM r JOIN u ON u.layers = r.layers
  WHERE r.mfu >= (1 - delta) * u.u_star
)
SELECT * EXCLUDE (pick) FROM band WHERE pick = 1;

-- Objective 2: U* is the highest MFU of any model on n_gpu_min to n_gpu_max GPUs; among the plans with
-- an MFU of at least (1 - delta) U*, the shortest training (ties: higher MFU, fewer GPUs).
CREATE OR REPLACE MACRO shortest_training(sid, n_gpu_min, n_gpu_max, delta := 0.15) AS TABLE
WITH r AS (
  SELECT * FROM best_per_pair
  WHERE search_id = CAST(sid AS UUID) AND n_gpu BETWEEN n_gpu_min AND n_gpu_max AND mfu IS NOT NULL
), u AS (
  SELECT max(mfu) AS u_star FROM r
)
SELECT r.*, u.u_star
FROM r, u
WHERE r.mfu >= (1 - delta) * u.u_star
ORDER BY r.t_train_s, r.mfu DESC, r.n_gpu, r.layers
LIMIT 1;

-- One row per search with its progress.
CREATE OR REPLACE VIEW search_progress AS
SELECT s.id, s.created_at, s.updated_at, s.status, s.error, s.labels, s.target_loss,
       s.seq_len, s.b_max, len(s.layers) AS models, len(s.n_gpus) AS n_gpus,
       len(s.layers) * len(s.n_gpus) AS pairs,
       count(sp.pair_id) AS pairs_done,
       count(sp.pair_id) FILTER (WHERE sp.reused) AS pairs_reused
FROM search s
LEFT JOIN search_pair sp ON sp.search_id = s.id
GROUP BY ALL;
