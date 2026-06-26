# CHECKPOINT — Bayesian Learning course build

## STATUS: COMPLETE ✅

Last updated: 2026-06-25 (session 1, all phases done)

## What was built
Expert-level Bayesian modeling + workflow course as numbered Markdown chapters in
`bayesian_learning/`. Style modeled on `RECSYS_COURSE.md`. Plan frozen in `_COURSE_BIBLE.md`.

## Final file inventory
```
bayesian_learning/
  README.md                                      ← table of contents + recommended path
  00-overview-and-setup.md                       8,327 words
  01-foundations-of-bayesian-inference.md       10,277 words
  02-priors-1-concepts-and-common-priors.md     10,798 words
  03-priors-2-choosing-building-and-checking.md 11,213 words
  04-inference-1-analytic-grid-and-quadratic-approximation.md  9,547 words
  05-inference-2-mcmc-from-metropolis-to-nuts.md               13,475 words
  06-inference-3-variational-inference.md       10,490 words
  07-mcmc-diagnostics-and-debugging.md          10,300 words
  08-the-bayesian-workflow.md                   12,775 words
  09-glms-and-bambi.md                          14,043 words
  10-hierarchical-and-multilevel-models.md      12,581 words
  11-model-comparison-and-evaluation.md         11,704 words
  12-gaussian-processes.md                      11,762 words
  13-mixture-models-and-latent-variables.md     10,030 words
  14-nonparametric-and-flexible-models.md        9,992 words
  15-time-series-and-state-space-models.md      11,293 words
  16-missing-data-the-bayesian-way.md           12,013 words
  17-scaling-and-advanced-computation.md        11,360 words
  18-capstone-case-studies.md                   13,664 words
  appendix-cheatsheets-and-resources.md          8,331 words
  TOTAL: 223,975 words (20 chapters + appendix + README)

  _research/
    _COURSE_BIBLE.md       ← frozen authoring contract
    R1 through R11 dossiers (11 verified API/resource reference files)
    CHECKPOINT.md          ← this file
```

## Pipeline phases completed
1. [SCAFFOLD ✅] Folder + `_COURSE_BIBLE.md` + checkpoint
2. [RESEARCH ✅] 11 dossiers (~31.5k words), verified PyMC v5/ArviZ/Bambi/Stan APIs
3. [WRITE ✅] 20 chapters, 0 failures, 212k words initial
4. [CRITIQUE+REVISE ✅] All 20 critiqued + revised. 150 fixes total.
   Grades: A(01,03,05,11) A-(00,06,07,08,09,10,13,14,17) B(02,04,12,15,16,18,appendix)
   3 confirmed-fixed critical bugs:
   - Ch02: prior_weight formula (0.67→0.80 correct)
   - Ch15: GRW forecasting (pm.set_data shape mismatch → numpy forward-sim)
   - Ch16: MNAR alignment (y_latent row-order bug → manual obs/mis split)
5. [FINALIZE ✅] README.md written. Final word count: 223,975.

## Key design decisions (for future continuation)
- Library stack: PyMC v5 primary, ArviZ diagnostics, Bambi alongside raw PyMC, Stan via
  CmdStanPy, NumPyro/nutpie/BlackJAX for scaling. NO live env — code is illustrative.
- Notation: sigma (not precision), pm.sample returns InferenceData (not "trace")
- Consistency rules frozen in _COURSE_BIBLE.md §10
- priorsense: R-only, not in Python — treated conceptually
- pymc_extras (fit_laplace/find_MAP/fit_pathfinder): fast-moving, flagged with verify-comments
- log_likelihood: NOT computed by default in PyMC v5; need idata_kwargs={'log_likelihood':True}
  or pm.compute_log_likelihood() before az.loo/az.compare

## If continuing / extending
- Add Jupyter notebooks paired to each chapter (like RECSYS_COURSE paired notebooks)
- Add a Chapter 19 on Bayesian causal inference (DAGs, do-calculus, front-door/back-door)
- Add a Chapter 20 on decision theory / utilities
- Consider deeper SBC tooling (sbi library, pymc-bart SBC)
- All 11 dossiers in _research/ remain valid as API references
