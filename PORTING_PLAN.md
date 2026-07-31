# Porting plan: notebooks/scripts -> src/jenkins_et_al/

**Read this file first in any new session working on this port.** It is the
living summary of what's done, what's left, and what to watch for. Update it
as work progresses — mark items done, add newly-discovered flags, note when
an original notebook changes. Keep entries short; this file is meant to be
read in one pass, not to hold full prose (git commit messages and code
comments hold the detailed provenance for anything already ported).

## Constraints (do not relitigate)

- Baseline: git commit `2d19f3a` is the frozen notebook/script state the
  submitted paper's analysis is built on. The analysis and its results must
  not change, even accidentally. The code's *shape* is a real refactor target
  (split into loading/processing/plotting functions) — the constraint is
  output identity, not literal cell-to-function transcription.
- Original files under `notebooks/` and `scripts/` are read-only to any
  agent working on this port. Never edit them, even for a typo. The user may
  edit them directly themselves (see "In-flight original edits" below) —
  that's their call, not something to revert or question.
- Use the `/golden-refactor` skill for every port. One notebook/script per
  invocation. Do not start porting anything, or move to the next item,
  unless the user has named it.
- Environment: `environment.yml` at repo root pins the exact conda env
  (Python 3.9.19) used to run the originals. Use it for both golden capture
  and running the port.
- Never `git add -A` / `git add .` — stage ported files by explicit path
  only. Never stage anything under `notebooks/`/`scripts/`.

## Status

| # | Notebook / script | Status | Ported to | Notes |
|---|---|---|---|---|
| 1 | `config.py` (shared constants) | **Done** (commit `89c23da`) | `src/jenkins_et_al/config.py` | See "Decisions already made" below |
| 2 | `notebooks/preprocessing/process_TRex_output.ipynb` | **Done** (commit `d167892`) | `src/jenkins_et_al/preprocessing/trex.py` | Golden capture redirected `output_dir` to `golden/` to avoid overwriting real production CSVs on `/Volumes/LaCie` |
| 3 | `notebooks/behaviour_free_swim/free_swim_bhv_metrics.ipynb` | Pending | — | Depends on #2's output tree (`/Volumes/LaCie/free_swimming/7dpf_TRex/<stimulus>/{control,stim}/`) |
| 4 | `notebooks/behaviour_free_swim/prefence_heatmaps.ipynb` | Pending | — | Duplicates #3's movement-metric helpers (dedupe on port); calls an undefined `summarize_metric_df`-type function — check before treating as golden; uses `stim`/`base` subfolders vs #3's `stim`/`control` — verify against real folder names |
| 5 | `notebooks/neural/build_brain_map.ipynb` | Pending | — | No confirmed in-repo consumer, but establishes reference brain image / area mask / `area_shorthand_dict.json` that later neural notebooks also read |
| 6 | `notebooks/neural/lifetime_sparseness.ipynb` | Pending — **being edited by user, see below** | — | Owns canonical 12-color forebrain palette + boxplot-vs-OB pattern; re-read fresh at port time, don't trust old survey description |
| 7 | `notebooks/neural/response_latency.ipynb` | Pending | — | Reuses #6's palette/boxplot almost verbatim; hard-codes area-shorthand dict instead of loading JSON |
| 8 | `notebooks/neural/noise_correlations_area_pair_analysis.ipynb` | Pending | — | Owns one of three separate Fisher+FDR implementations |
| 9 | `notebooks/neural/template_matching_classification.ipynb` | Pending | — | Reimplements Fisher+FDR from #8; defines its boxplot-vs-OB function twice (second silently overrides first); owns the 18-color hindbrain palette |
| 10 | `notebooks/neural/functional_clustering_concentration_response.ipynb` | Pending | — | Has 2 apparent bugs: likely missing comma in a dict literal; k-means produced 5 clusters against a 4-name mapping, raised an error in its last saved run — resolve before treating as golden |
| 11 | `notebooks/neural/functional_clustering_valence_neurons.ipynb` | Pending | — | Richest producer/consumer of valence-clustering CSVs; owns int-keyed `CLUSTER_COLORS` (different meaning than the same-named var in #10 — don't merge) |
| 12 | `notebooks/neural/plot_regression_spatial_maps.ipynb` | Pending | — | Simplest consumer of the same CSVs as #11 |
| 13 | `notebooks/neural/lda_svm_spatial_segregation.ipynb` | Pending | — | Owns the "+1"-corrected permutation-test formula (different from #14/#9's formulas — don't unify) |
| 14 | `notebooks/neural/lda_svm_classification.ipynb` | Pending | — | Independent data source (whole-brain response-vector file); owns plain-ratio permutation formula + a third independent Fisher+FDR impl |
| 15 | `notebooks/neural/wholebrain_population_analysis.ipynb` | Pending | — | Has an internal bug: mixed-case stimulus codes in its first `POSITIVE_STIMULI`/`NEGATIVE_STIMULI` def, silently overridden by a corrected lowercase def later in the same notebook. Most plotting is commented-out example calls — lower priority |
| 16 | `scripts/run_bhv_vigour_regression.py`, `run_stimulus_specific_regression.py`, `run_straight_bout_regression.py`, `run_turning_bout_regression.py` | Pending (port together) | — | Share identical z-score/shuffle/correlation core — extract once |
| 17 | `scripts/run_single_stimulus_regression.py` | Pending | — | Deliberate outlier: no permutation test, hard-coded to "4dpf" — make life-stage a parameter |
| 18 | `notebooks/behaviour_head_fixed/bout_frequency_LMM.ipynb` | Pending | — | Owns best-factored LMM-fitting function + paired dot-plot function |
| 19 | `notebooks/behaviour_head_fixed/bout_type_frequency_LMM.ipynb` | Pending | — | Reimplements LMM logic 3x inline (always includes variance term unconditionally, unlike #18/#21); saved output filenames ("intermediate"/"low" conc) don't match the high-conc set actually active when it last ran — check before trusting as golden |
| 20 | `notebooks/behaviour_head_fixed/bout_frequency_PSTH.ipynb` | Pending | — | Different display order + green/magenta valence pair vs #18/#19's limegreen/deeppink (config.py's canonical valence pair is green/magenta, matching this one) |
| 21 | `notebooks/behaviour_head_fixed/reorientation_posture_LMM.ipynb` | Pending | — | Owns most complete LMM function (CI + error handling) + 4-tier significance-star function |

No file anywhere in the codebase cites an explicit figure/panel number — any
notebook-to-figure mapping is a content-based guess. See full survey (in
chat history as of 2026-07-31) for the detailed guess-per-notebook if needed;
not reproduced here since it's low-confidence and not actionable per-port.

## Decisions already made (don't re-litigate)

- **Stimulus code spelling**: `"fex1"` (head-fixed behaviour notebooks) and
  `"fex_1"` (neural notebooks/regression scripts) are both kept as aliases
  in `config.STIMULUS_LABELS` — no canonical form forced.
- **Valence color**: green (positive) / magenta (negative) is canonical
  (`config.VALENCE_COLORS`), chosen by the user over the two other pairs
  found in the original notebooks (limegreen/deeppink; green/red).
- **Stimulus display order**: not forced to one global order — plotting
  functions should accept an explicit `order` argument; `config.py` holds one
  default (`STIMULUS_ORDER_HIGH_CONC`) for convenience only.

## In-flight original edits (not yet reflected in the status table above)

- `notebooks/neural/lifetime_sparseness.ipynb`: user is changing its per-area
  significance test from Mann-Whitney U to Wilcoxon signed-rank (a real
  analysis change, not refactor-shape). Not yet ported, so nothing committed
  depends on the old version. When porting #6, re-read the notebook fresh and
  check whether #7/#9 (same boxplot-vs-OB pattern) still use Mann-Whitney U —
  if so, that's a new cross-notebook inconsistency to flag, not silently fix.

## Known external dependencies

- Golden captures that read real data require `/Volumes/LaCie` mounted
  (contains `free_swimming/`, `larval_HuC/`, `area_masks/`, etc. — not part
  of this git repo).
- Some neural notebooks read correlation CSVs whose names imply an
  aggregation-and-FDR-correction step (and, for the free-swim/behaviour
  notebooks, a bout-aggregation step) that isn't present anywhere in this
  repo. Their absence means those notebooks' full pipelines can't be
  golden-captured end-to-end from raw data alone — golden capture must start
  from whatever intermediate CSV/H5 file the notebook actually reads.
