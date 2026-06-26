# The Recommender Systems Course
### From first principles to the state of the art — a hands-on, paper-backed curriculum

> *Written the way I'd train a data scientist I'm personally responsible for: to take
> you from "I can call `.fit()`" to "I can design, ship, debug, and defend a
> recommender at Microsoft or Google scale." Every chapter is paired with a tested,
> runnable notebook in this repo so you learn the theory and the engineering in the
> same breath.*

---

## 📚 Table of Contents

- [Part 0 — How to use this course](#part-0--how-to-use-this-course)
- [Part 1 — The Recommendation Problem](#part-1--the-recommendation-problem)
- [Part 2 — Evaluation & Metrics: The Primer](#part-2--evaluation--metrics-the-primer)
- [Part 3 — Neighborhood & Linear Workhorses (start simple — they're hard to beat)](#part-3--neighborhood--linear-workhorses-start-simple--theyre-hard-to-beat)
- [Part 4 — The Matrix Factorization Family (the workhorse decade)](#part-4--the-matrix-factorization-family-the-workhorse-decade)
- [Part 5 — Factorization Machines & Deep CTR/Ranking (how ads & feeds are ranked)](#part-5--factorization-machines--deep-ctrranking-how-ads--feeds-are-ranked)
- [Part 6 — Deep Collaborative Filtering, Autoencoders & Sequential Models](#part-6--deep-collaborative-filtering-autoencoders--sequential-models)
- [Part 7 — Graph Models & the Production Retrieval→Ranking Stack](#part-7--graph-models--the-production-retrievalranking-stack)
- [Part 8 — Online Learning, Cold-Start, Bias & Fairness (the realities of production)](#part-8--online-learning-cold-start-bias--fairness-the-realities-of-production)
- [Part 9 — The Frontier: Generative & LLM-based Recommendation (2024-2026)](#part-9--the-frontier-generative--llm-based-recommendation-2024-2026)
- [Part 10 — The Hyperparameter Tuning Playbook](#part-10--the-hyperparameter-tuning-playbook)
- [Part 11 — Edge Cases & War Stories (what breaks when the numbers look fine)](#part-11--edge-cases--war-stories-what-breaks-when-the-numbers-look-fine)
- [Part 12 — Your Career Roadmap (from here to lead data scientist)](#part-12--your-career-roadmap-from-here-to-lead-data-scientist)
- [Appendix — Cheat-Sheets & Quick Reference](#appendix--cheat-sheets--quick-reference)


## Part 0 — How to use this course

This document is the **theory and judgment** layer. The `projects/` notebooks are the
**hands** — tested `.py` engines plus narrated notebooks where every number is real
(executed headless, 0 errors). Read a Part here, then open its paired notebook and run
it. Theory tells you *why*; the notebook shows you *that it's true*.

### The philosophy I'm teaching you
1. **Always build the dumb baseline first.** Popularity and global-mean are not jokes —
   they are the honest yardstick. A 2019 paper (Dacrema et al.) showed that many
   published "deep" recommenders were beaten by a well-tuned KNN or EASE. If you can't
   beat popularity, you have nothing. (Parts 2–3.)
2. **The metric is the model.** You will optimise *exactly* what you measure, so choose
   the metric before the model. Most real failures are measurement failures. (Part 2.)
3. **Simple, tunable workhorses beat fragile SOTA.** This whole course is biased toward
   methods that *work*, *tune well*, and *don't surprise you in production*: MF, iALS,
   EASE, FM/DeepFM, SASRec, LightGCN, two-tower. We cover the frontier (Part 9) so you
   know where it's going — but you ship workhorses.
4. **Offline win ≠ online win.** The job isn't a better NDCG; it's a better product
   metric in an A/B test, behind a guardrail. (Parts 2, 8, and every notebook's
   Production Lens.)

### The map — Parts ↔ notebooks ↔ what's genuinely new in each
| Part | Topic | Paired notebook(s) | What it teaches that others don't |
|---|---|---|---|
| 1 | The recommendation problem | all | framing, feedback types, the lifecycle |
| 2 | **Evaluation & metrics primer** | `gnnlab/metrics.py` + all | the spine — what to measure and the pitfalls |
| 3 | Neighborhood & linear workhorses | `03`, `06`, **`08` (EASE)** | why simple item-item models are so hard to beat |
| 4 | Matrix factorization family | `03`, `04`, **`08` (iALS)** | explicit *and* implicit MF; the workhorse decade |
| 5 | Factorization Machines & deep CTR | **`05`** | feature interactions, the ranking stage, calibration |
| 6 | Deep CF, autoencoders & sequential | **`06`** | NCF critique, VAE-CF, GRU4Rec→SASRec |
| 7 | Graph models & retrieval→ranking | `04`, **`09` (two-tower)** | LightGCN, the two-stage production stack, ANN |
| 8 | Online learning, cold-start, bias | **`07`** | streaming regimes, FTRL, debiasing, fairness |
| 9 | The frontier: generative & LLM rec | `06` (bridge) | TIGER/semantic IDs, LLM recommenders |
| 10 | **Hyperparameter tuning playbook** | every `tune.py` | how to actually tune each family |
| 11 | Edge cases & war stories | all | what breaks in production and why |
| 12 | Your career roadmap | — | how to become a lead DS |
| Appendix | Cheat-sheets | — | metrics, model-selection tree, dataset guide |

### The datasets, and why each was chosen (each adds information the others lack)
| Dataset | Notebook | What's *new* about it |
|---|---|---|
| MovieLens-100K / 1M | `03`, `04`, `06`, `09` | explicit ratings + (1M) rich demographics & timestamps |
| Frappe | `05` | dense **categorical CTR** features across 10 fields — where FM shines |
| MovieLens-1M (replayed) | `07` | a **time-ordered stream** — for online/streaming learning |
| Last.fm play-counts | `08` | genuine **implicit feedback** (counts = confidence, no ratings) |

### The recommended path
If you're starting cold: **Part 1 → Part 2 (twice — it's that important) → Part 3 →
Part 4**, running notebooks `03` then `08` as you go. Then **Part 5 (`05`) → Part 6
(`06`) → Part 7 (`09`) → Part 8 (`07`)**. Read **Part 9** for the horizon, **Part 10**
whenever you tune, and **Parts 11–12** when you start operating, not just modelling.

> A note on how to *study* this: don't read passively. For each method, force yourself
> to answer the seven questions I make every section answer — intuition, the math, the
> key paper, the hyperparameters, when it breaks, the use case, and which notebook
> proves it. If you can answer those seven for MF, FM, SASRec, LightGCN, two-tower,
> iALS and EASE, you are already ahead of most people who call themselves "RecSys
> engineers."

## Part 1 — The Recommendation Problem

> *A note from me to you:* before we touch a single matrix factorization or graph layer, we need to agree on **what problem we are actually solving**. Almost every painful mistake I have watched a junior team make — optimizing the wrong metric, training on leaked data, shipping a model that "wins offline" and tanks online — traces back to a fuzzy answer to *this* first question. So we slow down here. Get the framing right and the rest of the course is engineering. Get it wrong and the rest is wasted compute.

---

### 1.1 Why recommendation matters

Recommendation is the quiet engine under most of the modern internet. When the catalog is bigger than any human can browse — every song ever recorded, every product in a warehouse, every video uploaded in the last minute — the *interface* to that catalog stops being search and becomes **recommendation**. The system's job is to take a near-infinite supply and surface the handful of items *this* user will care about *right now*.

The business case is not subtle. Gomez-Uribe & Hunt's *The Netflix Recommender System* (ACM TMIS, 2015) put the combined value of personalization and recommendation at **over $1 billion per year** for Netflix alone, mostly through reduced churn. The same story repeats at YouTube, Amazon, Spotify, and every large marketplace: recommendation is not a feature bolted onto the product — for these companies it *is* the product.

The deepest economic argument is **the long tail** (Chris Anderson, *The Long Tail*, 2006). A physical store stocks the few thousand items that sell; a digital catalog stocks everything. Most of the catalog is individually unpopular but collectively enormous. A good recommender is the only mechanism that can profitably connect a niche item to the small set of people who would love it — demand that a "bestsellers" shelf can never reach.

> 💡 **The framing that matters:** recommendation converts an *attention-scarcity* problem into a *matching* problem. The catalog is huge, user attention is tiny (you get ~10 slots on a screen), so the entire game is **ranking under a hard top-K budget**, at the scale of millions of items and billions of interactions.

---

### 1.2 The data: the user–item interaction matrix

Almost everything starts from one object: the **interaction matrix** $R \in \mathbb{R}^{|U| \times |I|}$, with users on the rows, items on the columns, and an entry $R_{ui}$ recording how user $u$ engaged with item $i$ (a rating, a click, a play count, a purchase). Every notebook in this repo — `03_movielens_100k_recsys` through `09_two_tower_retrieval` — is, at heart, a different way of completing or ranking this matrix.

Two structural facts about $R$ shape *everything* downstream:

**It is brutally sparse.** MovieLens-100K (project `03`) holds ~100,000 ratings across 943 users and 1,682 items: a density of about **6.3%**. MovieLens-1M (project `04`) is sparser still; industrial matrices are routinely **>99.9% empty**. So the central modeling challenge is *generalizing from a tiny observed slice to the vast unobserved remainder* — and a missing entry is not a zero, it is an unknown. Sit with that; we return to it in §1.6.

**The cold-start problem.** A brand-new user has an empty row; a brand-new item has an empty column. Pure collaborative methods — which learn *only* from co-occurrence — have literally nothing to go on. This is why project `04` carves out an explicit **cold-start** evaluation, and the repo's headline finding is honest about it: the graph model (LightGCN) wins on warm/hot users but **loses to BPR on cold-start**. Cold-start is the permanent tax on collaborative filtering, and the standard escape hatch is *side information* (content features), which we discuss in §1.7. The 2025 literature confirms this remains unsolved at the frontier — e.g. *Cold-Start Recommendation with Knowledge-Guided Retrieval-Augmented Generation* (2025) attacks it with LLM-retrieved context precisely because collaborative signal is absent.

---

### 1.3 Feedback types: explicit vs implicit

This is the distinction I most want to lodge in your head, because it silently dictates your loss function, your negatives, your metrics, and your evaluation protocol.

#### Explicit feedback — the user *tells* you their preference

A 1–5 star rating is a direct, signed statement: 5 means love, 1 means hate. MovieLens ratings (projects `03`, `04`) are the canonical example. The modeling target is the **rating value**, the task is regression, and you measure error with **RMSE / MAE** (`gnnlab/metrics.py`).

#### Implicit feedback — you *infer* preference from behavior

A click, a play, a purchase, a dwell. The user never declared anything; you are reading intent off behavior. Last.fm listening counts (project `08`) are the archetype. This looks easier — there is *so much more* of it — but it is treacherous, and the trap has a precise shape.

#### The crucial differences

| | **Explicit** (ratings) | **Implicit** (clicks / plays / buys) |
|---|---|---|
| Signal | Signed preference (love ↔ hate) | Unsigned *evidence* of interest |
| Negatives | A 1-star *is* a real negative | **No real negatives exist** |
| Missing entry | Genuinely unknown rating | Could be dislike **or** never-seen |
| Volume | Scarce (rating is effortful) | Abundant (logged passively) |
| Noise | Low per-signal | High (misclicks, autoplay, gifts) |
| Natural loss | Regression (RMSE/MAE) | Ranking / weighted classification |
| Repo home | `03`, `04` | `08`; also `05` (clicks), `06`, `07` |

> 💡 **Why implicit feedback has no real negatives.** If I never played a song, did I dislike it — or never encounter it? You cannot tell. The absence of an interaction is a *confounded mixture* of disinterest and non-exposure. Treating all un-played items as negatives is wrong (most are simply unseen); treating none as negatives leaves you with only positives and nothing to contrast against. This single asymmetry is why implicit-feedback methods look so different from rating predictors.

#### Confidence vs preference — the Hu–Koren–Volinsky reframing

Hu, Koren & Volinsky, *Collaborative Filtering for Implicit Feedback Datasets* (ICDM 2008), give the most elegant resolution: split each observation into **two** quantities:

- **Preference** $p_{ui} \in \{0, 1\}$ — *did any interaction happen at all?* ($p_{ui}=1$ if user $u$ touched item $i$, else $0$).
- **Confidence** $c_{ui} = 1 + \alpha r_{ui}$ — *how strongly do we believe that preference?* More plays ⇒ more confidence.

You then fit *every* user–item cell (including the zeros) but **weight the squared error by confidence**: a zero you are unsure about costs almost nothing; a heavily-played positive costs a lot. That is exactly the **weighted ALS / iALS** objective in project `08`. The other dominant family flips the framing to *ranking*: Rendle et al., *BPR: Bayesian Personalized Ranking from Implicit Feedback* (UAI 2009), optimizes **pairwise** order — for each user, an observed item should score above a sampled unobserved one — which sidesteps the "what is a negative?" question by only ever comparing a positive to a *sampled* candidate. Project `04` (BPR-MF) and project `08` (BPR alongside iALS and EASE) both teach this. Contrast that with projects `03`/`04`'s explicit MF, which minimizes RMSE on *observed ratings only*. Same matrix factorization skeleton; completely different loss because the feedback type changed.

---

### 1.4 The tasks — what are we actually predicting?

"Recommendation" is an umbrella over several genuinely different machine-learning problems. Confusing them is the #1 source of a wrong metric. Here is the clean taxonomy, mapped to this repo.

| Task | Question it answers | Output | Metric family | Repo notebook(s) |
|---|---|---|---|---|
| **Rating prediction** | How many stars would $u$ give $i$? | A scalar per cell | **RMSE, MAE** | `03`, `04` (explicit MF/NMF) |
| **Top-K ranking** | What are the best $K$ items for $u$? | An ordered list | **NDCG@K, Recall@K, Precision@K, MAP@K, MRR, Hit@K** | `03`, `04`, `08` |
| **Candidate retrieval** | Which ~hundreds (of millions) are even worth scoring? | A coarse shortlist | **Recall@K**, latency | `09` (two-tower + ANN) |
| **Sequential / next-item** | Given the *order* of past actions, what's next? | Next-item ranking | **NDCG@K, Hit@K** (leave-one-out) | `06` (GRU4Rec, SASRec) |
| **CTR prediction** | Will $u$ click *this* item in *this* context? | A calibrated probability | **AUC, LogLoss** | `05` (FM family) |

A few teaching points hide in that table:

- **Rating prediction is not what you deploy.** The Netflix Prize trained a generation of us to chase RMSE, but no user ever sees a predicted rating — they see a *list*. Project `03` makes this lesson concrete: matrix factorization wins on RMSE yet its top-K list can still rank *below* a dumb popularity baseline. **Good regression ≠ good ranking.** Part 2 will dwell on why.
- **Sequential recommendation breaks the i.i.d. assumption.** Once *order* carries signal ("you watched episodes 1–3, so episode 4 next"), a model that ignores sequence is leaving information on the table. Project `06` climbs from a Markov chain to **GRU4Rec** (Hidasi et al., 2016) and the self-attentive **SASRec** (Kang & McAuley, 2018) — and the headline result is striking: SASRec wins NDCG@10 (**0.585**) *and* catalog coverage (**0.88**) over a popularity baseline (0.238 / 0.26). Accuracy and diversity together.
- **CTR is a calibration problem, not just a ranking one.** When the output feeds an ad auction or a "% likely to enjoy" badge, the *probability itself* must be trustworthy, which is why project `05` traces **LogLoss** and an **expected-calibration-error (ECE) / reliability curve** alongside **AUC** across LogisticRegression → FM → FFM → DeepFM → DCN-v2. The repo's honest note — plain FM is *better calibrated* than the deeper nets even when they win AUC — is exactly the kind of nuance a lead DS must hold.
- **Retrieval is a recall game; ranking is a precision game.** More on this in §1.8.

---

### 1.5 Beyond accuracy

Even a perfectly accurate ranker can be a bad product. A model that maximizes NDCG by showing everyone the same blockbusters is *accurate* and a *filter bubble*. So `gnnlab/metrics.py` ships a **beyond-accuracy** panel — **catalog coverage, Gini, novelty, ARP, intra-list diversity** — first-class in project `06`. Hold this thought: in §1.8 we will see that *re-ranking* is where these business objectives get enforced, and Part 2 gives them rigorous definitions.

---

### 1.6 Missing-Not-At-Random (MNAR): the bias that haunts everything

Here is the uncomfortable truth that separates a practitioner from a researcher who has actually shipped: **your data is a biased sample of reality, and the bias is baked into the act of observation itself.**

Users do not rate a *random* sample of items. They rate what they chose to watch, which is what was recommended to them, which is what the *previous* model surfaced. So the observed entries of $R$ are **Missing Not At Random** — the probability that $R_{ui}$ is observed depends on the very value $R_{ui}$ you are trying to predict (people rate things they like; the system showed them things it predicted they'd like). Marlin & Zemel, *Collaborative Filtering and the Missing-at-Random Assumption* (UAI 2009), demonstrated empirically that the standard "missing-at-random" assumption underlying RMSE-style training is simply **false** for rating data.

Two engines drive MNAR (per Chen et al., *Bias and Debias in Recommender System: A Survey*, ACM TOIS 2023):

1. **Self-selection bias** — users rate items they already have an opinion about (usually positive); negatives are silently missing.
2. **Previous-system / exposure bias** — you only get feedback on items the prior policy chose to show. Popular items get shown more, accrue more data, look better, get shown even more. A **feedback loop** that ossifies popularity.

Why it haunts *everything*: train naïvely and you learn the *prior recommender's* habits, not true preference. Evaluate naïvely and your offline metric rewards exactly that bias. Schnabel et al., *Recommendations as Treatments: Debiasing Learning and Evaluation* (ICML 2016), reframed recommendation as **causal inference**: an exposed item is a "treatment," and you correct the sample with **inverse-propensity weighting** — down-weighting the over-observed populars. You do not need to implement IPW today; you need the *instinct*: **whenever an offline number looks too good, suspect that selection bias, not skill, produced it.** This is the hidden reason the streaming/prequential evaluation in project `07` matters — evaluating *forward in time on the live stream* is far more honest than shuffling a static log.

---

### 1.7 The classic taxonomy of methods

Three axes you should be able to place any system on instantly.

#### Content-based vs collaborative vs hybrid

| Approach | Learns from | Strength | Weakness |
|---|---|---|---|
| **Content-based** | Item/user *features* (genre, text, demographics) | Handles cold-start; explainable | Trapped in a "more-of-the-same" filter bubble; needs good metadata |
| **Collaborative filtering (CF)** | The interaction matrix *only* ("users like you also liked…") | Discovers surprising cross-genre taste; needs no features | Cold-start helpless; popularity-biased |
| **Hybrid** | Both | Best of both; mitigates cold-start | More moving parts |

Most of this course is CF (it is where the deep ideas live), but the **Factorization Machine** family in project `05` is exactly the principled hybrid: FMs ingest user ID, item ID, *and* arbitrary context features in one model, learning pairwise interactions among all of them — collaborative and content-based unified under one equation. That is why FMs became the workhorse of industrial CTR.

#### Memory-based vs model-based CF

- **Memory-based** — no training; compute similarities at query time and average over neighbors. **KNN-CF** in project `03` is the textbook case (user-user / item-item). Transparent, trivially incremental, but slow at scale and brittle on sparse data.
- **Model-based** — *learn* compact latent factors that *compress* the matrix and *generalize* to unseen cells. **MF / NMF** (`03`, `04`), **BPR-MF** (`04`, `08`), **EASE** (`08`), and the graph models **LightGCN / NGCF** (`03`, `04`) are all model-based. The repo's arc *is* the journey from memory-based KNN to model-based latent factors to graph-based message passing — each step a better way to share statistical strength across a sparse matrix.

---

### 1.8 The industrial recommender lifecycle

A real system is a *pipeline*, not a model. Internalize this loop; the back half of the course builds it piece by piece.

```
        ┌──────────── the feedback loop (this is where MNAR is born) ────────────┐
        ▼                                                                          │
   [ Data / logs ] ─▶ [ Retrieval ] ─▶ [ Ranking ] ─▶ [ Re-ranking ] ─▶ [ Serve ] ─▶ [ Log ]
     interactions      ~M → ~100s        score the       diversity,        top-K to     impressions
     + features        cheap, recall-    shortlist        business rules    the user     + clicks
                       oriented          precisely        de-dup, freshness
```

1. **Retrieval (candidate generation).** From millions of items, fetch a few hundred *cheaply*. The dominant pattern is the **two-tower** model (project `09`): a user tower and an item tower map into a shared embedding space, and an **approximate-nearest-neighbor (ANN)** index turns "score everything" into a millisecond lookup. The metric here is **Recall@K** — you must not *drop* a good item, because nothing downstream can recover it.
2. **Ranking.** Now spend real compute scoring those few hundred with a heavy, feature-rich model (the FM / deep-cross family of project `05`). Here the metric is **precision-flavored** — NDCG, AUC, LogLoss — because you are ordering the final few.
3. **Re-ranking.** Apply diversity, freshness, de-duplication, and business rules (no two items from one seller in a row; honor sponsored slots). This is where the **beyond-accuracy** metrics of §1.5 are enforced.
4. **Serving & logging.** Show the top-K; log impressions *and* outcomes — including **what was shown but not clicked**, the negatives you'll desperately need.
5. **The feedback loop.** Those logs become tomorrow's training data — closing the loop, and (per §1.6) *manufacturing the very MNAR bias* you must defend against. The loop is both the system's power and its pathology.

> 💡 **Why two stages?** This is *the* architectural idea of modern industrial RecSys, from Covington, Adams & Sargin, *Deep Neural Networks for YouTube Recommendations* (RecSys 2016). You cannot run a heavy ranker over millions of items per request — the latency is impossible. So you **split recall from precision**: a cheap retriever guarantees the good items are *in the room* (high recall), and an expensive ranker decides the *order* once the room is small (high precision). Project `09` is the retrieval half; project `05` is the ranking half. Hold this two-stage picture — Parts 7+ develop it in full.

The 2025–26 frontier is rewriting *retrieval* itself. **Generative retrieval** — Rajput et al.'s **TIGER** (*Recommender Systems with Generative Retrieval*, NeurIPS 2023) and its RQ-VAE "semantic IDs" — replaces the embed-and-ANN lookup with a sequence model that *generates the item ID token-by-token*, and LLM-augmented variants are extending it through 2026. The two-stage *frame* survives; what fills each stage keeps evolving.

---

### 1.9 How to read this course

| Part / Notebook | Task | What you'll master |
|---|---|---|
| `03` ML-100K | Rating + top-K | Bias baselines, KNN-CF, MF/NMF, LightGCN; *RMSE ≠ ranking* |
| `04` ML-1M | Rating + ranking + cold-start | BPR, FM, NGCF; when graphs win and when they lose |
| `05` Frappe | CTR | LR → FM → FFM → DeepFM → DCN-v2; AUC vs LogLoss, calibration |
| `06` ML-1M (seq) | Next-item | Markov → GRU4Rec → SASRec; accuracy **+** beyond-accuracy |
| `07` streaming | Online | Decayed-pop, incremental MF, FTRL; **prequential** evaluation, drift |
| `08` Last.fm | Implicit | iALS, EASE, BPR; confidence-vs-preference in practice *(data layer built; models in progress)* |
| `09` two-tower | Retrieval | Two-tower + ANN; the recall half of the two-stage stack *(two-tower model built; eval in progress)* |

Read them roughly in order: each assumes the priors of the last. Where a notebook trains "the classical workhorse first" (it always does), *resist* skipping to the deep model — the baseline is the yardstick that tells you whether the fancy thing earned its complexity. That habit, more than any architecture, is what makes a lead DS.

---

### Where this fits / what to read next

You now have the map: the matrix, its sparsity and cold-start, the explicit/implicit divide and its confidence-vs-preference resolution, the five tasks, the MNAR bias that quietly poisons naïve evaluation, the content/collaborative/hybrid and memory/model axes, and the two-stage retrieval→ranking lifecycle that organizes the whole back half of the course.

Every claim above rested on a *metric* — RMSE, NDCG@K, Recall@K, AUC, LogLoss, coverage. We invoked them loosely. **Part 2 — Evaluation & Metrics** makes them rigorous: how to build an honest train/validation/test split for ranked data (leave-one-out, temporal, and why a random split *leaks*), the exact definitions behind every function in `gnnlab/metrics.py`, the offline-vs-online gap, and why the single most expensive mistake in recommendation is **trusting an offline number that selection bias inflated.** Read it before you train anything.

---

## Part 2 — Evaluation & Metrics: The Primer

> *"You can only improve what you can measure — and you will only chase what you report."* This is the most important part of the course. A recommender is a decision system, and a metric is the loss function of your **judgement**. Pick the wrong metric and you will spend six months shipping a model that wins offline and quietly loses money online. We are going to be slow and rigorous here so that everything after it (Parts 3–9) has a spine to hang on.

The mental model to carry through the whole section: a recommender answers one of four *task shapes*, and each shape has its own native metric.

1. **Rating prediction** — "what score will this user give this item?" (explicit feedback, the 1–5 stars). → Projects [03](../projects/03_movielens_100k_recsys), [04](../projects/04_movielens_1m_recsys).
2. **Top-K ranking** — "given a user, produce an ordered list." This is what you actually deploy. → [`gnnlab/metrics.py`](../gnnlab/metrics.py), Projects [06](../projects/06_sequential_recommendation), 08, 09, Tutorial Lesson 9.
3. **CTR / pointwise classification** — "what is the probability of a click on this (user, item, context)?" → Project [05](../projects/05_factorization_machines).
4. **Beyond-accuracy** — "is the list healthy?" coverage, novelty, diversity, fairness. → [`gnnlab/metrics.py`](../gnnlab/metrics.py), Projects [06](../projects/06_sequential_recommendation), 08.

We will treat each in turn, then spend the back half of the section on the part nobody teaches well: **evaluation protocols, their pitfalls, and the offline→online gap.** That is where careers are made or quietly stalled.

---

### Master metrics table

| Metric | Formula sketch | What it rewards | When to use | Repo ref |
|---|---|---|---|---|
| **RMSE** | $\sqrt{\frac1N\sum (\hat r - r)^2}$ | small errors, *punishes large misses* | explicit-rating regression | `metrics.rmse` (03/04) |
| **MAE** | $\frac1N\sum \lvert\hat r - r\rvert$ | typical error, robust to outliers | interpretable rating error | `metrics.mae` (03/04) |
| **Precision@K** | hits in top-K / K | dense relevance in a short list | fixed-slot UIs | `precision_at_k` |
| **Recall@K** | hits in top-K / #relevant | finding all the good items | retrieval / candidate gen | `recall_at_k` |
| **HitRate@K** | 1 if any hit in top-K | "did we surface *one* good item" | leave-one-out setups | `hit_rate_at_k` |
| **MAP@K** | mean of avg-precision over users | precision *across* the ranking | multi-positive ranking | `average_precision_at_k` |
| **MRR** | $\frac1{\text{rank of 1st hit}}$ | getting *one* answer to the top | single-answer tasks (QA, next-item) | `reciprocal_rank` |
| **NDCG@K** | $\mathrm{DCG@K}/\mathrm{IDCG@K}$ | graded relevance + position | **the headline ranking metric** | `ndcg_at_k` (06/08/09) |
| **AUC** | $P(\text{score}_+>\text{score}_-)$ | correct *ordering* of pos vs neg | CTR ranking quality | `metrics.auc` (05) |
| **LogLoss** | $-\frac1N\sum [y\log p+(1{-}y)\log(1{-}p)]$ | *calibrated* confident probabilities | CTR training & eval | `metrics.log_loss` (05) |
| **ECE** | $\sum_b \frac{\lvert B_b\rvert}{N}\lvert\text{acc}_b-\text{conf}_b\rvert$ | probabilities that *mean* what they say | bidding, thresholding | (see §CTR) |
| **Coverage** | items ever recommended / catalogue | catalogue exploration | filter-bubble check | `catalog_coverage` (06) |
| **Novelty** | mean $-\log_2 p(i)$ | surfacing the long tail | discovery products | `novelty` (06) |
| **ILD** | mean $1-\cos(i,j)$ in list | within-list variety | "more of the same" check | `intra_list_diversity` (06) |
| **ARP / Gini** | mean train-popularity / inequality | (low) popularity bias | fairness of exposure | `average_recommendation_popularity`, `gini_index` (06) |

---

### Rating prediction — RMSE and MAE

When feedback is **explicit** (a user types a 1–5 star rating), the system is doing regression. Two metrics dominate. Let $\hat r_{ui}$ be the predicted rating and $r_{ui}$ the held-out true rating over a test set of $N$ pairs.

$$\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{(u,i)} (\hat r_{ui} - r_{ui})^2}, \qquad \mathrm{MAE} = \frac{1}{N}\sum_{(u,i)} \lvert \hat r_{ui} - r_{ui}\rvert.$$

The single most important difference is the **squaring**. RMSE is a quadratic penalty, so a single 3-star miss hurts it 9× more than a 1-star miss; it is dominated by your worst predictions and is differentiable everywhere, which is why MF and most regressors optimise (a regularised) squared error directly. MAE is a linear penalty — robust to outliers and beautifully interpretable: "on average we are off by 0.74 stars." Because RMSE upweights the tail, **RMSE ≥ MAE always**, and the gap between them is a free diagnostic: a large gap means your error distribution is heavy-tailed (a few users or items you predict terribly), which is usually more actionable than the headline number itself.

Two cautions you must internalise. First, the **scale trap**: both are in the units of the rating, so they are not comparable across datasets with different rating scales — never compare a MovieLens RMSE to an Amazon RMSE. Always anchor to a baseline: a *GlobalMean* predictor on MovieLens-100K scores RMSE ≈ 1.04–1.12 (see Tutorial Lesson 9 and Project [03](../projects/03_movielens_100k_recsys)). If your "fancy" model is not *clearly* below that, it has learned essentially nothing. Second, and more philosophically — **RMSE is almost never the thing you care about.** Users never see a predicted star value; they see an *ordered list*. A model can have excellent RMSE and rank terribly, because squared error spends its capacity getting the *middle* of the rating distribution right (where most of the mass is) while the ranking only cares about the *top*. This is the historical lesson of the Netflix Prize: the field optimised RMSE for years before realising the deployed quantity is a ranking. We carry that lesson into the next subsection. *(Cross-ref: Projects [03](../projects/03_movielens_100k_recsys)/[04](../projects/04_movielens_1m_recsys) implement both via `metrics.rmse` / `metrics.mae`.)*

---

### Top-K ranking — Precision, Recall, MAP, MRR, HitRate, and NDCG

In production you take a user, score the catalogue, sort, and show the top K. The held-out **relevant** set $\mathrm{Rel}_u$ (the items the user actually interacted with, removed from training) is the ground truth. The golden rule, enforced in [`gnnlab/metrics.py`](../gnnlab/metrics.py): **never score an item the user already saw in training.** You rank held-out positives against negatives the user never touched.

**Precision@K and Recall@K** are the set-overlap pair. With $\mathrm{hits} = |\,\text{top-}K \cap \mathrm{Rel}_u\,|$:

$$\mathrm{P@K} = \frac{\mathrm{hits}}{K}, \qquad \mathrm{R@K} = \frac{\mathrm{hits}}{|\mathrm{Rel}_u|}.$$

Precision asks "of what I showed, how much was good?" — the right metric for a fixed-real-estate UI (a 5-slot carousel). Recall asks "of all the good stuff, how much did I surface?" — the right metric for **candidate generation / retrieval**, where a downstream ranker will refine a few hundred items, so you care about catching the positives at all (Project 09, two-tower retrieval). Neither is position-aware: swapping rank 1 and rank K leaves both unchanged.

**MRR** (Mean Reciprocal Rank) fixes that for the single-answer case. If $\mathrm{rank}_u$ is the position of the *first* relevant item:

$$\mathrm{MRR} = \frac{1}{|U|}\sum_u \frac{1}{\mathrm{rank}_u}.$$

It is the natural metric for next-item prediction and QA — there is one right answer and you want it at the top (`reciprocal_rank`). **MAP@K** (Mean Average Precision) generalises to multiple positives by averaging precision *each time you hit a relevant item* as you walk down the list, then averaging over users (`average_precision_at_k`). It rewards packing relevant items early and is the classic IR ranking summary. **HitRate@K** is the binary degenerate case — 1 if *any* relevant item is in the top-K — and is ubiquitous in **leave-one-out** evaluation, where each user contributes exactly one held-out positive (`hit_rate_at_k`).

#### NDCG@K — the headline metric, derived

NDCG (Normalized Discounted Cumulative Gain) is the metric you put on the slide, and you should understand *why*. It starts from **Discounted Cumulative Gain** (Järvelin & Kekäläinen, 2002, *"Cumulated gain-based evaluation of IR techniques"*, ACM TOIS). The idea is two-fold: (a) relevance can be **graded** ($\mathrm{rel}_i \in \{0,1,2,\dots\}$, not just binary), and (b) a hit deep in the list is worth less than a hit at the top — users scan from the top and attention decays. The logarithmic discount encodes that decay:

$$\mathrm{DCG@K} = \sum_{i=1}^{K} \frac{2^{\mathrm{rel}_i} - 1}{\log_2(i+1)}.$$

(In the common binary case $2^{\mathrm{rel}_i}-1$ collapses to $\mathbf{1}[i \in \mathrm{Rel}_u]$, which is exactly what `dcg_at_k` computes.) The problem: DCG is unbounded and incomparable across users with different numbers of relevant items. So we **normalise** by the best achievable DCG — the **Ideal DCG**, obtained by sorting all relevant items to the front:

$$\mathrm{IDCG@K} = \sum_{i=1}^{\min(|\mathrm{Rel}_u|,\,K)} \frac{1}{\log_2(i+1)}, \qquad \boxed{\;\mathrm{NDCG@K} = \frac{\mathrm{DCG@K}}{\mathrm{IDCG@K}} \in [0,1].\;}$$

This matches `ndcg_at_k` exactly. NDCG@K = 1 means a perfect ranking; 0 means no relevant items surfaced.

> 💡 **Why NDCG is the headline.** It is the *only* common metric that is simultaneously (1) position-aware (rank-1 beats rank-10), (2) graded (a 5-star hit can beat a 3-star hit), and (3) normalised to $[0,1]$ so it averages cleanly across heterogeneous users. There is also a *theoretical* reason it works. Wang et al. (2013, *"A Theoretical Analysis of NDCG Type Ranking Measures"*, COLT) proved the surprising fact that NDCG with the log discount converges to 1 as the catalogue grows — which naively suggests it cannot tell good rankers from bad. They resolved the paradox with **"consistent distinguishability"**: for any two *substantially different* ranking functions, NDCG consistently decides which is better. That is the rigorous justification for trusting it. Report **NDCG@10 as your primary number, with Recall@K as the retrieval companion.**

*(Cross-ref: `evaluate_ranking` in [`gnnlab/metrics.py`](../gnnlab/metrics.py) returns precision/recall/ndcg/hit/map per K plus MRR in one call; Projects [06](../projects/06_sequential_recommendation)/08/09 and Tutorial Lesson 9 use it.)*

---

### CTR / classification — AUC, LogLoss, and (crucially) calibration

Click-through-rate models output a *probability*, not a list, so they are judged differently (Project [05](../projects/05_factorization_machines): LR → FM → FFM → DeepFM → DCN-v2 on Frappe). Two metrics, plus a third that the textbooks skip and the industry lives by.

**AUC** (Area Under the ROC Curve) has a clean ranking interpretation: it is exactly the probability that a randomly chosen positive scores above a randomly chosen negative,

$$\mathrm{AUC} = P\big(\hat p_+ > \hat p_-\big) = \frac{1}{n_+ n_-}\sum_{x_+}\sum_{x_-} \mathbf{1}[\hat p_{x_+} > \hat p_{x_-}].$$

This equals the Mann–Whitney U statistic, which is why `metrics.auc` computes it from ranks (the $O(n\log n)$ identity) rather than integrating a curve. AUC is **threshold-free** and **scale-invariant** — it tells you whether the model *orders* clicks correctly. **LogLoss** (binary cross-entropy) is what CTR models actually train on:

$$\mathrm{LogLoss} = -\frac{1}{N}\sum_{i}\big[y_i\log \hat p_i + (1-y_i)\log(1-\hat p_i)\big].$$

It punishes *confident and wrong* predictions brutally (the $\log$ diverges as $\hat p \to$ the wrong extreme), so it cares about the **calibration** of the probability, not just its rank. AUC and LogLoss disagree often, and the disagreement is informative: you can shift every prediction by a monotone transform and leave AUC untouched while wrecking LogLoss.

That brings us to the part that matters most in advertising.

#### Calibration — why it can beat AUC for bidding

A model is **calibrated** if, among all impressions it scores at $\hat p = 0.2$, exactly 20% actually click. AUC does not care about this at all — only the *ordering*. But in a real-time bidding auction, your bid is literally `pCTR × value`, so a 2× miscalibration on pCTR is a 2× error in your bid and a direct hit to ROI. Here calibration matters *more than AUC*. We quantify it with the **Expected Calibration Error**: bin predictions into $M$ bins $B_b$, and measure the gap between average confidence and observed accuracy per bin:

$$\mathrm{ECE} = \sum_{b=1}^{M} \frac{|B_b|}{N}\,\big|\,\mathrm{acc}(B_b) - \mathrm{conf}(B_b)\,\big|.$$

The visual companion is the **reliability diagram**: predicted probability on the x-axis, observed frequency on the y-axis; the diagonal is perfect calibration, and area off the diagonal is miscalibration. The standard fixes are **post-hoc**: Platt scaling (fit a logistic on the logits) or **isotonic regression** (a monotone step function). Recent industrial work pushes this further with field-aware, multi-dimensional calibration — Deep Ensemble Shape Calibration (Yan et al., 2024) and the Auction-Information-Enhanced framework (AIE, RecSys 2024) calibrate pCTR jointly with auction signals; a 2024 survey, *Confidence Calibration for Recommender Systems and Its Applications*, catalogues the landscape.

> 💡 **Pitfall — chasing AUC, shipping a money-loser.** A model can have higher AUC and *worse* calibration than its predecessor. If your downstream system multiplies pCTR by anything (a bid, an expected revenue, a blended score with pCVR), report **LogLoss and ECE alongside AUC** and gate launches on calibration. AUC ranks; calibration *prices*.

---

### Beyond-accuracy — the health of the list

A model that maximises NDCG by recommending the same five blockbusters to everyone is accurate *and* a filter bubble. Accuracy metrics are silent about this, so production teams trace a complementary panel ([`gnnlab/metrics.py`](../gnnlab/metrics.py) §5; Project [06](../projects/06_sequential_recommendation) reports it explicitly). Let $p(i)$ be item $i$'s popularity (interaction share in *train*), and $L_u$ a recommendation list.

- **Catalogue coverage** — fraction of the catalogue ever recommended across all users: $\mathrm{Cov} = |\bigcup_u L_u| / |\mathcal{I}|$. Low coverage = the model only knows how to push a few items (`catalog_coverage`).
- **Novelty** — mean self-information of recommended items, $\mathrm{Nov} = \frac{1}{|L|}\sum_{i \in L} -\log_2 p(i)$. Rare items carry more "surprise," so higher = more long-tail discovery (`novelty`).
- **Average Recommendation Popularity (ARP)** — mean train-popularity of recommended items; *lower* = less popularity-biased (`average_recommendation_popularity`). The blunt, honest popularity-bias number.
- **Gini index** — inequality of how often each item gets recommended across users; 0 = everything recommended equally, 1 = one item hogs every list (`gini_index`). A high Gini is the quantitative signature of a filter bubble.
- **Intra-List Diversity (ILD)** — mean pairwise dissimilarity *within* a list, $\mathrm{ILD}(L_u) = \frac{2}{|L_u|(|L_u|-1)}\sum_{i<j}\big(1 - \cos(\mathbf{v}_i, \mathbf{v}_j)\big)$, using item feature vectors (e.g. genre multi-hot). Low ILD = "ten flavours of the same thing" (`intra_list_diversity`).
- **Serendipity** — the hardest and most valuable: relevant **and** unexpected. The standard form is $\mathrm{Ser}(i) = \mathrm{Unexpectedness}(i)\times \mathrm{Relevance}(i)$, where unexpectedness is the distance from what a *primitive* model (e.g. most-popular) would have shown (Kotkov et al., 2016, *"A Survey of Serendipity in Recommender Systems"*). Note: not yet in `gnnlab`, but it is the north star of "discovery" surfaces; 2025 work even uses LLMs as serendipity judges (Fu et al., 2025).

> 💡 **Pitfall — a high-NDCG model can still be a filter bubble.** NDCG only sees the held-out positives, and your *logged* positives are themselves popularity-biased (popular items get more exposure → more clicks → look more "relevant"). A model that overfits popularity will score high NDCG offline *and* low coverage, high ARP, high Gini. This is why Steck (2018, *"Calibrated Recommendations"*, RecSys) argued for **calibration of the recommendation list** — if a user watched 70% romance / 30% action, the list should too, rather than letting the dominant taste crowd out the rest. Always read NDCG **next to** coverage/Gini/ARP, never alone.

---

### Offline evaluation protocols & PITFALLS

This is the section that separates a senior DS from a junior one. The metric formula is the easy part; the *protocol* that produces the numbers is where almost all published and internal results go wrong.

**Splitting: random vs temporal.** A **random** train/test split (hold out random interactions) leaks the future into the past — you may train on a user's later behaviour and test on their earlier behaviour, which is impossible at serving time. The honest protocol is **temporal**: a global time cutoff, or per-user **leave-last-out** (train on all but the chronologically last interaction, test on it). Sequential models (Project [06](../projects/06_sequential_recommendation), SASRec/GRU4Rec) *require* leave-last-out, and even static models should prefer temporal splits because they expose the **distribution shift** you will actually face. Leakage is insidious: pre-computing normalisation statistics, negative-sampling pools, or popularity features over the *full* dataset all leak test information backwards.

**The sampled-metrics critique — the one every practitioner gets wrong.** To make evaluation cheap, a near-universal shortcut ranks each held-out positive against a small random sample of negatives (the classic "1 positive + 99 negatives" you see in Tutorial Lesson 9 and most NCF-era papers) instead of the **full catalogue**. Krichene & Rendle (2020, *"On Sampled Metrics for Item Recommendation"*, KDD) proved this is **not** just a noisy approximation — it is *inconsistent*: sampled metrics do not preserve the relative ordering of models, **not even in expectation.** Model A can beat B on sampled NDCG and lose on full NDCG. Worse, as the sample shrinks, *all* top-K metrics collapse toward AUC, so you stop measuring the top of the list at all. Their guidance, which you should adopt: **prefer full-catalogue evaluation**; if you must sample for scale, apply their bias/MSE corrections and never compare models evaluated under different sampling regimes.

**Comparing to weak baselines.** Dacrema, Cremonesi & Jannach (2019, *"Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches"*, RecSys) reproduced a wave of "SOTA" neural recommenders and found that most were beaten by **properly tuned simple baselines** — ItemKNN, a well-regularised matrix factorisation, even popularity. The lesson is methodological and brutal: an improvement over a *crippled* baseline is not an improvement. This is exactly why every RecSys project in this repo trains the **classical workhorses first** (bias baselines, KNN, MF, FM) and tunes them as hard as the GNN before any comparison.

> 💡 **Pitfall — popularity bias *in the evaluation itself*.** Offline test sets are built from logged interactions, which are exposure-biased toward popular items, so "accuracy" partly rewards re-recommending what was already popular. Cañamares & Castells (2018) showed this can flip the *measured* winner relative to an unbiased ground truth. Mitigations: propensity-weighted (debiased) metrics, stratifying metrics by item-popularity tiers (head/torso/tail), or collecting a small **uniformly-exposed** test slice. At minimum, report a popularity-only baseline so you can see how much of your "skill" is just popularity.

> 💡 **Pitfall — the seed lottery.** On small data a single run swings by several points. **Average over seeds and report mean ± std** (Tutorial Lesson 9, Project 02 uses 10-fold CV). And the cardinal rule: **select on validation, report on test, never let a test number influence a modelling decision.**

---

### Online metrics & the offline→online gap

Every offline number is a *proxy*. The business is judged on **online** metrics measured on live traffic: short-term engagement (**CTR**, **dwell time**, **completion/watch-time**), conversion (**CVR**, **revenue / GMV**), and the one that actually pays the bills — **long-term retention** (DAU/WAU, session frequency, churn). The offline→online gap is real and large: a model can win NDCG offline and *lose* retention online, because offline data was collected under the *old* policy (feedback-loop bias), because users react to *novelty* not just relevance, and because offline metrics cannot see *delight* or *fatigue*.

**Guardrail metrics.** When you optimise a *goal* metric (say CTR), you protect a set of **guardrails** you refuse to harm — latency, revenue, diversity, complaint rate, unsubscribes. Sequential testing on guardrails lets you abort a harmful experiment early (Letham et al. / industry practice; see "Navigating the Evaluation Funnel," 2024).

**A/B testing vs interleaving.** The gold standard is a randomised **A/B test**: split users, ship A to one arm and B to the other, measure the metric difference with a proper significance test, and *run it long enough* to outlast the **novelty effect** — the transient bump from "this is new" that regresses to the mean. A/B tests are slow and need large samples. **Interleaving** (Joachims 2002; Chapelle et al. 2012, *"Large-scale validation and analysis of interleaved search evaluation"*) is dramatically more sensitive: it blends *both* models' results into one list for the *same* user and attributes clicks back, controlling for between-user variance. Netflix reported interleaving identifies the better algorithm with **orders of magnitude fewer samples** than A/B — making it the standard cheap online pre-filter before a confirmatory A/B.

**Counterfactual / off-policy estimation.** Before you ever go online, you can *estimate* a new policy's online value from *logged* data — if you logged the **propensities** (the probability the old policy showed each item). The **Inverse Propensity Scoring (IPS)** estimator reweights logged rewards to undo the old policy's bias:

$$\hat V_{\mathrm{IPS}}(\pi) = \frac{1}{N}\sum_{i=1}^{N} \frac{\pi(a_i\mid x_i)}{\pi_0(a_i\mid x_i)}\, r_i.$$

IPS is unbiased but **high-variance** when propensities are small (huge weights from rare logged actions). The fix is the **Self-Normalized IPS (SNIPS)** estimator — divide by the sum of weights instead of $N$ — which trades a little bias for a large variance reduction and is the practical default (Swaminathan & Joachims; recent work: *"Counterfactual Risk Minimization with IPS-Weighted BPR and Self-Normalized Evaluation,"* 2025; Δ-OPE, 2024). This closes the loop: offline ranking metrics → counterfactual value estimate → interleaving → confirmatory A/B → retention.

---

### A decision guide — what to report, trace, and optimise

| Task | **Optimise** (loss) | **Report** (headline) | **Also trace** (guardrails) |
|---|---|---|---|
| **Rating prediction** (03/04) | regularised squared error | RMSE | MAE, and the RMSE–MAE gap |
| **Top-K ranking** (06) | BPR / softmax / sampled ranking loss | **NDCG@10** | Recall@K, HitRate@K (LOO), coverage, Gini, ARP |
| **Retrieval / candidate gen** (09) | in-batch softmax / contrastive | **Recall@100/200** | coverage, latency, NDCG of downstream ranker |
| **CTR** (05) | LogLoss (BCE) | **AUC + LogLoss** | **ECE / reliability diagram**, calibration by segment |
| **Discovery surfaces** | ranking loss + diversity re-rank | NDCG@10 | novelty, serendipity, ILD, coverage |

Three rules to carry forward. **(1) Optimise the loss, report the deployed metric, trace the guardrails** — they are usually three different things, and conflating them is the most common mistake. **(2) Never report a single number** — report NDCG *and* a beyond-accuracy panel, AUC *and* calibration, mean *and* std over seeds. **(3) Distrust offline wins until the protocol is bulletproof** — full-catalogue (or corrected) evaluation, temporal splits, strong tuned baselines — then earn the right to test online.

---

### Where this fits / what to read next

This part is the measuring instrument; everything else is the thing being measured. With these metrics and protocols in hand, **Part 3 — Rating Prediction & Matrix Factorization** builds the first real models (bias baselines → KNN-CF → SVD/MF) on MovieLens (Projects [03](../projects/03_movielens_100k_recsys)/[04](../projects/04_movielens_1m_recsys)), and you will immediately see why RMSE is a poor proxy for the ranking you ship — motivating the BPR/implicit-feedback losses of the parts that follow. Keep [`gnnlab/metrics.py`](../gnnlab/metrics.py) open as you go; every model from here on is scored through it.

#### References

- Järvelin, K. & Kekäläinen, J. (2002). *Cumulated gain-based evaluation of IR techniques.* ACM TOIS 20(4). — origin of DCG/NDCG.
- Wang, Y. et al. (2013). *A Theoretical Analysis of NDCG Type Ranking Measures.* COLT. — why the log discount works ("consistent distinguishability").
- Steck, H. (2018). *Calibrated Recommendations.* RecSys. — list-level calibration vs filter bubbles.
- Kotkov, D. et al. (2016). *A Survey of Serendipity in Recommender Systems.* Knowledge-Based Systems.
- Dacrema, M. F., Cremonesi, P. & Jannach, D. (2019). *Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches.* RecSys.
- Krichene, W. & Rendle, S. (2020). *On Sampled Metrics for Item Recommendation.* KDD.
- Cañamares, R. & Castells, P. (2018). *Should I Follow the Crowd? A Probabilistic Analysis of the Effectiveness of Popularity in Recommender Systems.* SIGIR.
- Chapelle, O. et al. (2012). *Large-scale validation and analysis of interleaved search evaluation.* ACM TOIS. (Interleaving; orig. Joachims 2002.)
- Swaminathan, A. & Joachims, T. (2015). *The Self-Normalized Estimator for Counterfactual Learning.* NeurIPS. (SNIPS.)
- Confidence Calibration for Recommender Systems (2024); AIE / Deep Ensemble Shape Calibration (RecSys 2024). — modern CTR calibration.

---

## Part 3 — Neighborhood & Linear Workhorses (start simple — they're hard to beat)

> *"The fastest way to lose credibility as a lead DS is to ship a deep model that a one-line baseline beats on the holdout. The fastest way to build it is to be the person who always builds that baseline first."*

Before we touch a single latent factor or a single graph convolution, we are going to spend a part on the models that the rest of the field keeps quietly rediscovering. These are not warm-up exercises. On a startling number of public benchmarks, a well-tuned item-item neighborhood model or a closed-form linear autoencoder **beats published deep recommenders** — and does it in seconds, with no GPU, and with a model you can read off a piece of paper. If you internalize one thing from this section, let it be this: *simple, honest, well-tuned baselines are not the floor you have to clear — they are frequently the ceiling.*

We will build the ladder in order — non-personalized → neighborhood → learned-linear (SLIM) → closed-form linear (EASE) → embedding-from-co-occurrence (item2vec) — and end with the methodological discipline (Dacrema et al. 2019) that ties it together. Cross-reference the working code in `projects/03_movielens_100k_recsys` (mean baselines + KNN-CF), `projects/06_sequential_recommendation` (ItemKNN), and `projects/08_implicit_feedback_recsys` (EASE) as you read.

---

### 3.1 Non-personalized baselines — the honest yardstick

The very first model you fit should be one with **zero personalization**. Why? Because it tells you how much signal is *in the data at all*, before any cleverness. There is a strict hierarchy of "dumb" predictors, each fixing a weakness of the last (see `03/src/models.py`, the `GlobalMean → UserMean → ItemMean → BiasBaseline` ladder):

- **Global mean** $\hat r_{ui} = \mu$. The absolute floor. Its RMSE is essentially the standard deviation of the ratings (~1.12 on MovieLens-100K). Anything worse than this is *negative information*.
- **User mean** $\hat r_{ui} = \mu_u$. Captures harsh-vs-generous raters.
- **Item mean** $\hat r_{ui} = \mu_i$. Captures good-vs-bad movies — usually the strongest *single* effect (~0.98–1.02 RMSE).
- **Bias baseline** $\hat r_{ui} = \mu + b_u + b_i$ (Koren 2009). Both effects *jointly*, fit by regularized alternating least squares so they don't double-count:

$$ b_u = \frac{\sum_{i\in I_u}(r_{ui}-\mu-b_i)}{\lambda + |I_u|}, \qquad b_i = \frac{\sum_{u\in U_i}(r_{ui}-\mu-b_u)}{\lambda + |U_i|}. $$

For **implicit / top-N** ranking, the non-personalized baseline is **popularity** (`TopPop`): recommend the globally most-interacted items to everyone. This is a brutally strong baseline on skewed catalogs and is *the* one neural papers most often fail to beat. Always report it.

> 💡 **Edge-case callout — popularity is a confound, not just a baseline.** Because clicks beget clicks, popularity correlates with almost everything. A "good" Recall@20 that barely edges out TopPop is mostly measuring the catalog's Gini skew, not your model. Pair accuracy with `gnnlab.metrics.catalog_coverage`, `gini_index`, and `novelty` to see whether you've learned *taste* or just *re-served the head*.

> 🔧 **Tuning.** The only knob is the bias regularizer $\lambda$ (the `reg` shrinkage in `03`). Tune it on a validation split; under sparsity, larger $\lambda$ pulls rarely-seen users/items toward the global mean, which is exactly right. Report **all** baselines in your results table — they cost nothing and they are the bar.

---

### 3.2 Neighborhood collaborative filtering (KNN-CF)

The oldest personalized idea in the field (Resnick et al. 1994, *GroupLens*; Sarwar et al. 2001, *item-based CF*) and still a top-tier baseline. Two flavors:

- **User-based** ("people like you also liked…"): to predict $r_{ui}$, find users similar to $u$ who rated $i$, take their similarity-weighted mean.
- **Item-based** ("movies like this one…"): find items similar to $i$ that $u$ has rated, take $u$'s weighted mean over them.

The item-based prediction (this is exactly what `03/src/models.py::KNNCF` computes for `kind="item"`):

$$ \hat r_{ui} = \mu_i + \frac{\sum_{j \in N_k(i;u)} s_{ij}\,(r_{uj}-\mu_j)}{\sum_{j \in N_k(i;u)} |s_{ij}| + h}. $$

Three design choices live in that formula, and they matter more than the choice of model:

**Similarity function $s_{ij}$.**
- **Cosine** — angle between rating vectors; ignores magnitude.
- **Pearson correlation** — cosine on *mean-centered* vectors. Subtracting each row's mean before computing cosine makes a harsh and a generous rater who agree on *shape* look similar. (In `03`, `_cosine_centered` does precisely this; "cosine on row-centered vectors == Pearson.")
- **Adjusted cosine** — for *item*-item, center by the **user** mean (subtract $\mu_u$ from every entry), removing per-user rating offsets before measuring item agreement (Sarwar et al. 2001). This is usually the right centering for item-item on explicit ratings.
- **Jaccard / cosine on binary vectors** — for **implicit** feedback, where you only have 0/1. Jaccard $= |U_i \cap U_j| / |U_i \cup U_j|$ measures co-occurrence overlap and is the natural choice when "rating" is meaningless.

**Shrinkage $h$.** A similarity estimated from two co-ratings is noise; one from two hundred is signal. The shrinkage term penalizes low-support pairs: $s_{ij} \leftarrow \frac{n_{ij}}{n_{ij}+h}\, s_{ij}$, where $n_{ij}$ is the co-support. In the prediction denominator (the `+ shrink` in `03`) it also damps predictions made from few, weak neighbors toward the item mean. This single regularizer often moves the metric more than $k$.

**Neighborhood size $k$.** Too small → high variance, jumpy predictions; too large → you average in dissimilar items and regress toward the mean. It is a classic bias–variance dial with an interior optimum (typically $k \in [20, 100]$ on MovieLens-scale data).

> 💡 **Why item-item usually wins.** Each **item-item** similarity is estimated from *all users who rated both items* — often hundreds — so it is statistically stable and **changes slowly** as new ratings arrive (you can precompute and cache it). A **user-user** similarity is estimated from the items two users *both* rated — often a handful — so it is noisier, and it goes stale the moment a user rates one more thing. Item-item is also more **scalable**: $|I| \ll |U|$ in most catalogs, so the $I\times I$ similarity matrix is smaller and re-used across all users. This is why `06` ships an **ItemKNN**, and why item-item is the production default.

> 🔧 **Tuning checklist (KNN-CF).** Grid over `kind ∈ {user,item}`, `k ∈ {10,20,40,80,160}`, `shrink ∈ {0,1,10,50,100}`, and similarity ∈ {cosine, Pearson, adjusted-cosine; Jaccard if implicit}. Optimize the *ranking* metric you actually care about (`ndcg_at_k`, `recall_at_k` via `gnnlab.metrics.evaluate_ranking`), not RMSE — a model can have great RMSE and rank terribly.

---

### 3.3 SLIM — the bridge from neighborhoods to *learned* linear models (Ning & Karypis 2011)

The conceptual leap of **SLIM** (*Sparse LInear Methods for Top-N Recommendation*, ICDM 2011) is this: a neighborhood model is just a *heuristically chosen* item-item weight matrix. Why hand-pick similarities when you can **learn** them to directly optimize reconstruction? SLIM models the score as a linear item-item map $\hat X = X B$ and learns the aggregation matrix $B$ by regression:

$$ \min_{B \ge 0,\; \mathrm{diag}(B)=0}\; \tfrac{1}{2}\lVert X - X B\rVert_F^2 + \tfrac{\beta}{2}\lVert B\rVert_F^2 + \lambda \lVert B\rVert_1. $$

The pieces *all* do work:
- **$\mathrm{diag}(B)=0$** — without it, the trivial solution $B=I$ copies $X$ onto itself (perfect reconstruction, zero generalization). This zero-diagonal constraint is the single most important idea in linear item-item models; remember it for EASE.
- **$\ell_1$ (lasso)** — drives most weights to exactly zero, giving a *sparse* $B$. Sparsity is not just regularization; it makes scoring blazingly fast (multiply by a sparse matrix) and the model interpretable ("these 8 items explain this one").
- **$\ell_2$ (ridge)** — the elastic-net partner that stabilizes correlated items.

SLIM is one of the strongest classical methods for **one-class / implicit** feedback. Its costs: training solves an elastic-net regression *per item* (columns of $B$ are independent — embarrassingly parallel), and the $\ell_1$ path is slow on large catalogs. Steck's **ADMM SLIM** (2020) and **Greedy SLIM** (2024) are scalable re-derivations; **ImplicitSLIM** (Shenbin & Nikolenko, ICLR 2024) shows SLIM-style item structure can even improve embedding models. SLIM is the conceptual hinge between the heuristic KNN of §3.2 and the closed-form linear model we turn to now.

---

### 3.4 EASE — the closed-form workhorse (Steck 2019)

**EASE** (*Embarrassingly Shallow Autoencoders for Sparse Data*, Steck, WWW 2019) asks: what happens if we take SLIM's formulation, **drop the $\ell_1$ and the non-negativity**, and keep only the ridge penalty and the zero-diagonal constraint? The objective becomes a convex quadratic with one linear constraint:

$$ \min_{B}\; \lVert X - X B\rVert_F^2 + \lambda \lVert B\rVert_F^2 \quad \text{s.t.}\quad \mathrm{diag}(B)=0. $$

Because it is convex quadratic, it has a **closed-form solution** — no SGD, no epochs, no learning rate. Form the regularized Gram matrix and invert it once:

$$ G = X^\top X + \lambda I, \qquad P = G^{-1}, \qquad B_{ij} = \begin{cases} 0 & i = j \\[4pt] -\dfrac{P_{ij}}{P_{jj}} & i \ne j. \end{cases} $$

That is the *entire model*. Derivation sketch: introduce a Lagrange multiplier vector for the $\mathrm{diag}(B)=0$ constraint, set the gradient to zero (giving $B = I - P\,\mathrm{diagMat}(\gamma)$), then solve the KKT condition $\mathrm{diag}(B)=0$ for the multipliers — which yields exactly $\gamma_j = (1 - 0)/P_{jj}$ and the $-P_{ij}/P_{jj}$ formula above. (See `projects/08_implicit_feedback_recsys`.)

**Why is it so strong?**
1. **It is a *principled* neighborhood model.** Steck's key insight: KNN uses *raw* similarities (entries of $X^\top X$), but EASE uses the entries of the **inverse** Gram matrix. The inverse performs an *implicit partial-correlation / de-confounding* step — it discounts the similarity between items $i$ and $j$ that is merely explained by a third popular item $k$. KNN's similarity matrix is, in Steck's words, "conceptually incorrect"; EASE's is the conditionally-correct one. This is why a closed-form linear model routinely beats hand-tuned KNN *and* deep autoencoders.
2. **One hyperparameter.** Just $\lambda$. The full grid search is a 1-D line search.
3. **No training loop, no non-convexity, perfectly reproducible.** Two practitioners with the same data get the *same* model.

> 🔧 **Tuning $\lambda$ — the U-curve.** $\lambda$ is the *only* knob, and the validation metric (NDCG/Recall) traces a clean **U-shape** against $\log\lambda$. Too small → $G$ is near-singular, $B$ overfits co-occurrence noise. Too large → $B \to 0$, the model collapses toward popularity. The optimum is wide and well-behaved; sweep $\lambda \in \{10^0, 10^1, \dots, 10^4\}$ (good values on dense-ish data are often in the **hundreds to low-thousands**). Beware: EASE is *sensitive* — a bad $\lambda$ can cost 50–80% of the metric, so do not skip the sweep.

> 💡 **When EASE wins vs. fails.**
> - **Wins:** dense-ish implicit feedback, moderate catalogs (up to ~tens of thousands of items), top-N ranking. On MovieLens-20M / Netflix / MSD it matched or beat Mult-VAE and other deep models in Steck's experiments — with a closed form.
> - **Fails:** **pure cold-start.** EASE is *entirely* an item-item co-occurrence model; an item nobody has interacted with has an all-zero column in $X$, so it can never be recommended. It also ignores side features and sequence/time entirely.
> - **The hard limit — $O(I^2)$ memory, $O(I^3)$ compute.** $B$ is a dense $I\times I$ matrix and you must invert $G$. At 50K items, $B$ is ~10 GB in float32; at millions of items it is infeasible. This is the wall that pushes you toward sparse SLIM, blocked/approximate variants (Steck's *Scalable EASE*, RecSys 2022), or — when $|I|$ truly explodes — embeddings and the matrix-factorization / two-tower models of later parts. *Note the duality:* EASE is cheap when items are few and users many; user-user KNN has the opposite profile.

EASE is the model I reach for first on any new implicit-feedback problem of modest catalog size. It sets a bar that is *embarrassing* to miss.

---

### 3.5 item2vec / prod2vec — embeddings from co-occurrence

A different lens on the same co-occurrence signal: borrow **word2vec** (Mikolov et al. 2013) and treat *items as words, baskets/sessions as sentences*.

- **prod2vec** (Grbovic et al., KDD 2015, Yahoo Mail) — trains skip-gram with negative sampling (SGNS) over *purchase sequences*; nearby products in a sequence get nearby vectors.
- **item2vec** (Barkan & Koenigstein, MLSP 2016) — drops the sequence order, treats each user's history as an **unordered bag of items**, and trains SGNS over co-occurring pairs. The result is a dense item-embedding space where cosine similarity captures "bought/watched together" semantics, *even when no user identity is available*.

The mental model: SGNS is implicitly factorizing a shifted PMI co-occurrence matrix (Levy & Goldberg 2014) — so item2vec is doing something *structurally cousin* to a factorization of $X^\top X$, the same Gram matrix EASE inverts. Its practical edge is producing reusable **dense embeddings** for ANN retrieval, clustering, and as features downstream. Its catch is that it is *hyperparameter-sensitive* — window, negative-sampling distribution, subsampling, and epochs matter a lot (Caselles-Dupré et al., RecSys 2018, "Word2vec applied to Recommendation: Hyperparameters Matter"). This is the conceptual ancestor of the embedding-based models in Part 4 and the two-tower retrieval in `09`.

---

### 3.6 The discipline: "Are we really making progress?" (Dacrema et al. 2019)

This is the part to tattoo on your forearm as a future lead. Ferrari Dacrema, Cremonesi & Jannach (RecSys 2019; extended TOIS 2021) tried to **reproduce** the top neural recommenders from KDD/SIGIR/WWW/RecSys. Of 18 papers, only **7 were reproducible with reasonable effort** — and **6 of those 7 were beaten** by properly tuned *simple* baselines: KNN, SLIM, and graph-based heuristics. The recurring failure mode was **weak baselines**: deep models were compared against *untuned* KNN, so they looked good by comparison while losing to a *tuned* one.

Take three operational lessons into every modeling project you lead:

1. **Tune the baselines as hard as you tune the hero model.** An untuned KNN is a strawman; a tuned KNN/EASE/SLIM is a genuine competitor. Same compute budget for both.
2. **Reproducibility is a feature.** Fixed seeds, fixed splits, published configs (this repo's `configs/` + `gnnlab.seed` convention exists for exactly this).
3. **Report the same metric, the same split, the same negative-sampling protocol** for every model. Most "SOTA gains" evaporate under a unified evaluation harness — see Rendle et al. (2020, "Neural Collaborative Filtering vs. Matrix Factorization Revisited"), which showed a well-tuned dot-product MF beats NeuMF.

The 2025-26 landscape only reinforces this: linear and graph-signal-processing methods (e.g., **BSPM**, Choi et al., SIGIR 2023; normalization-aware linear recommenders, 2025) remain at or near the top of public CF leaderboards, often above heavier neural and graph models. The lesson is not "deep learning doesn't work" — it's that **the bar is high, and the bar is cheap to build.**

---

### Cheat-sheet table

| Model | Core idea | Key hyperparameters | When it shines |
|---|---|---|---|
| **GlobalMean / Bias** | $\mu + b_u + b_i$, no interactions | bias reg $\lambda$ | The honest floor; always report |
| **Popularity (TopPop)** | recommend the head to everyone | (none) | Non-personalized top-N floor; hard to beat on skewed catalogs |
| **User-KNN** | "people like you" | $k$, shrink, similarity | Few items, many co-ratings; explainable |
| **Item-KNN** | "items like this" | $k$, shrink, sim (adj-cosine/Jaccard) | Default neighborhood model: stable, cacheable, scalable |
| **SLIM** | learn sparse item-item $B$, elastic-net | $\lambda$ ($\ell_1$), $\beta$ ($\ell_2$) | Implicit feedback; want sparse, fast, interpretable $B$ |
| **EASE** | closed-form $B=-P/\mathrm{diag}(P)$, $P=(X^\top X+\lambda I)^{-1}$ | $\lambda$ only | Dense-ish implicit, ≤ tens of K items; strongest cheap baseline |
| **item2vec** | SGNS on item co-occurrence | dim, window, negatives, epochs | Dense reusable embeddings for ANN/retrieval/features |

---

### 🧭 What to read next → Part 4: Matrix Factorization

Every model here learns or hand-picks an **item-item** structure of size $O(I^2)$. The moment your catalog has millions of items, that wall is fatal — and the inverse-Gram intuition behind EASE quietly *begs* for a low-rank approximation. Part 4 makes that leap: replace the dense $B$ with a **low-rank factorization** $R \approx U V^\top$, giving every user and item a short latent vector whose dot product is the score. We will derive **funkSVD / regularized MF** (Koren et al. 2009), **ALS** and **BPR** for implicit feedback (Rendle et al. 2009), and — crucially for this course — show that **MF is literally a 0-layer LightGCN**, making the graph models of later parts feel inevitable rather than magical. The workhorses of Part 3 are the baseline MF must beat; Part 4 is where personalization finally goes *latent*.

---

#### References

- Resnick, Iacovou, Suchak, Bergstrom & Riedl (1994). *GroupLens: An Open Architecture for Collaborative Filtering of Netnews.* CSCW.
- Sarwar, Karypis, Konstan & Riedl (2001). *Item-Based Collaborative Filtering Recommendation Algorithms.* WWW.
- Koren, Bell & Volinsky (2009). *Matrix Factorization Techniques for Recommender Systems.* IEEE Computer.
- Ning & Karypis (2011). *SLIM: Sparse Linear Methods for Top-N Recommender Systems.* ICDM.
- Mikolov et al. (2013); Levy & Goldberg (2014). *word2vec* / *Neural Word Embedding as Implicit Matrix Factorization.* NeurIPS.
- Grbovic et al. (2015). *E-commerce in Your Inbox: Product Recommendations at Scale (prod2vec).* KDD.
- Barkan & Koenigstein (2016). *Item2Vec: Neural Item Embedding for Collaborative Filtering.* MLSP.
- Caselles-Dupré, Lesaint & Royo-Letelier (2018). *Word2vec applied to Recommendation: Hyperparameters Matter.* RecSys.
- Steck (2019). *Embarrassingly Shallow Autoencoders for Sparse Data (EASE).* WWW. arXiv:1905.03375.
- Ferrari Dacrema, Cremonesi & Jannach (2019). *Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches.* RecSys. arXiv:1907.06902.
- Steck (2020). *ADMM SLIM: Sparse Recommendations for Many Users.* WSDM.
- Rendle, Krichene, Zhang & Anderson (2020). *Neural Collaborative Filtering vs. Matrix Factorization Revisited.* RecSys.
- Steck & Liang (2022). *Scalable Linear Shallow Autoencoder for Collaborative Filtering.* RecSys.
- Choi et al. (2023). *Blurring-Sharpening Process Models for Collaborative Filtering (BSPM).* SIGIR.
- Shenbin & Nikolenko (2024). *ImplicitSLIM and How it Improves Embedding-Based Collaborative Filtering.* ICLR.

---

## Part 4 — The Matrix Factorization Family (the workhorse decade)

> *"For two decades, if you had to ship one recommender on Monday morning and be right, you reached for matrix factorization. It is not a relic. As late as 2022, a properly tuned iALS still beat the deep-learning state of the art on half the standard benchmarks. Before you fall in love with graphs and transformers, you must own this family — its math, its failure modes, and the exact knobs that move the needle."*

Everything in this part operates on one object: the **user–item interaction matrix** `R ∈ ℝ^{m×n}`, with `m` users (rows) and `n` items (columns). The entry `r_ui` is what user `u` did with item `i` — a 1–5 star rating, a play count, a click, or nothing at all. The matrix is **brutally sparse**: in MovieLens-100K (cross-ref `projects/03_movielens_100k_recsys`) only ~6.3% of cells are filled; in industrial settings it is 0.01% or less. The entire game is to predict the missing entries.

---

### 4.1 The core idea: low-rank structure and latent factors

The bet matrix factorization makes is that `R` is **approximately low-rank**. Tastes are not arbitrary — they are governed by a small number of hidden themes (action vs. romance, mainstream vs. arthouse, "has Tom Hanks"). If `k` such themes explain most variance, then a user is summarized by a length-`k` vector `p_u ∈ ℝ^k` (how much they care about each theme) and an item by `q_i ∈ ℝ^k` (how much it expresses each theme). The predicted affinity is their dot product:

```
r̂_ui = p_u · q_i = Σ_f  p_uf · q_if
```

Stack the user vectors into `P ∈ ℝ^{m×k}` and item vectors into `Q ∈ ℝ^{n×k}` and the model is the rank-`k` approximation

```
R  ≈  P Qᵀ        with   k ≪ min(m, n)
```

This is why it is called *factorization*: we factor one giant, mostly-empty matrix into two skinny, dense ones. The compression is enormous (`mk + nk` parameters instead of `mn` cells) and that compression **is** the generalization — forcing all of `u`'s behavior through a single `k`-vector means the model cannot memorize noise; it must find shared structure that transfers to unseen `(u, i)` pairs.

**Why not just use the SVD?** Classical truncated SVD gives the optimal rank-`k` reconstruction of a *fully observed* matrix. But `R` is mostly missing, and SVD treats missing as zero, which is a lie (a movie you never saw is not a movie you rated 0). The two fixes — *only sum the loss over observed cells*, and *regularize* — turn plain SVD into the learned models below.

#### 💡 The Netflix Prize story (Koren, Bell & Volinsky 2009)

In 2006 Netflix offered \$1M to anyone who could beat its Cinematch RMSE by 10% on 100M ratings. The methods that won were not exotic — they were matrix factorization, relentlessly engineered. The winning team **BellKor's Pragmatic Chaos** crossed the line in September 2009. The lasting artifact is the survey *"Matrix Factorization Techniques for Recommender Systems"* (Koren, Bell & Volinsky, **IEEE Computer 42(8):30–37, 2009**), the single most-cited paper in the field. Its thesis: latent-factor models beat neighborhood (KNN) methods *and* gracefully absorb biases, implicit signals, temporal drift, and confidence weights. That paper is the spine of this entire part.

---

### 4.2 Biased Matrix Factorization (FunkSVD): the real workhorse

Simon Funk's blog-famous "FunkSVD" (2006) and its biased extension are what people usually mean by "MF." The key realization: **most of the signal in explicit ratings is not interaction at all — it is bias.** Some users rate everything high; some movies are universally loved. Make the model spend its precious latent dimensions on *taste*, not on these cheap additive effects, by pulling biases out explicitly:

```
r̂_ui = μ + b_u + b_i + p_u · q_i
```

- `μ` — global mean rating (e.g. 3.5 on MovieLens).
- `b_u` — user bias ("this user rates 0.4 stars above average").
- `b_i` — item bias ("this movie is loved, +0.6").
- `p_u · q_i` — the *interaction* term, what's left after bias is removed.

**The objective.** We minimize squared error over the **observed** set `K = {(u,i) : r_ui exists}`, with L2 regularization so the model shrinks toward simplicity under sparsity:

```
min_{P,Q,b}  Σ_{(u,i)∈K} ( r_ui − μ − b_u − b_i − p_u·q_i )²
                          + λ ( ‖p_u‖² + ‖q_i‖² + b_u² + b_i² )
```

**SGD updates.** Let `e_ui = r_ui − r̂_ui` be the prediction error on one observed cell. Take the gradient of that cell's loss and step downhill (learning rate `η`):

```
b_u ← b_u + η ( e_ui − λ·b_u )
b_i ← b_i + η ( e_ui − λ·b_i )
p_u ← p_u + η ( e_ui · q_i − λ·p_u )
q_i ← q_i + η ( e_ui · p_u − λ·q_i )
```

Note the symmetry in the last two lines: each latent vector is nudged in the direction of the *other* one, scaled by the error `e_ui`. One epoch = one sweep over all observed ratings in random order.

#### 🔧 Why biases matter (the part juniors skip)

Drop the biases and your latent factors are forced to *encode* "this user is generous" and "this movie is popular" inside `p_u, q_i`. Those directions then dominate the dot product, drown out genuine taste, and — worse — they don't generalize: popularity is better modeled as a scalar than as a `k`-dimensional direction. In `projects/03_movielens_100k_recsys` and `projects/04_movielens_1m_recsys` the `BiasBaseline` (`μ + b_u + b_i`, *no* latent factors) is implemented first precisely because it is **shockingly strong** — it is the honest bar a latent-factor model must clear, and a freshly-built MF that doesn't beat it has a bug. Note both repos fit biases by **regularized alternating closed-form updates**, e.g. `b_i = Σ_u(r_ui − μ − b_u) / (λ + |U_i|)` — the `λ` in the denominator is exactly the shrinkage that pulls rarely-seen items toward 0. That is the cleanest illustration of regularization-as-shrinkage in the whole course.

---

### 4.3 ALS vs. SGD: two ways to fit the same model

The objective in §4.2 is **biconvex**: fix `Q`, and solving for `P` is a convex ridge regression; fix `P`, and solving for `Q` is too. This gives two optimizers.

**SGD** walks downhill one observation at a time. It is memory-light, trivially handles bias terms and exotic losses (logistic, BPR), and is the default for explicit feedback. Its cost is sensitivity to learning-rate schedules and inherently sequential sweeps.

**ALS (Alternating Least Squares)** exploits the biconvexity: hold `Q` fixed and solve every user's `p_u` in closed form by ridge regression over *that user's observed items*, then swap and solve every `q_i`. Each half-step is exact.

```
p_u = ( Q_uᵀ Q_u + λI )⁻¹ Q_uᵀ r_u      # Q_u = item factors for u's rated items
```

**When to choose which:**

| Situation | Prefer |
|---|---|
| Explicit ratings, fancy/asymmetric loss, biases, online/streaming updates | **SGD** |
| Implicit feedback (every cell is "observed" with a weight) | **ALS** — the magic in §4.4 makes it tractable |
| Massive sparse data, want embarrassing parallelism | **ALS** — each `p_u` solve is independent → trivially shard across users, then across items |
| Limited RAM, need a quick single-machine fit | SGD (or coordinate descent) |

The parallelism point is the practical one: ALS solves are independent across users within a sweep, so Spark/`implicit`-style libraries scale ALS to billions of interactions on a cluster. SGD's sequential dependence makes it fiddlier to distribute (Hogwild!-style lock-free updates exist but need care).

---

### 4.4 iALS / WMF for implicit feedback (Hu, Koren & Volinsky 2008) — THE implicit workhorse

This is the most important model in the part. **Implicit feedback** (clicks, plays, purchases — cross-ref `projects/08_implicit_feedback_recsys`, which uses Last.fm artist play counts ranging 1 to 352,698) breaks the explicit-MF assumptions in three ways:

1. **No negatives.** You see what users *did*, never what they *rejected*. A blank cell is *maybe-dislike-or-maybe-never-seen*, not a 0.
2. **The signal is confidence, not preference.** Playing a song 5,000 times vs. 1 time both mean "likes it" — but with wildly different *certainty*. The magnitude is a confidence level, not a rating.
3. **You must score all cells**, so you cannot just sum over observed entries.

Hu, Koren & Volinsky (*"Collaborative Filtering for Implicit Feedback Datasets,"* **ICDM 2008**; winner of the ICDM 10-year highest-impact award) solve this elegantly. Split each raw count `r_ui` into a **binary preference** and a **confidence**:

```
p_ui = 1 if r_ui > 0 else 0            # did they interact at all?
c_ui = 1 + α · r_ui                    # how sure are we?  (often α·log(1 + r_ui/ε))
```

Every cell now has a target (`p_ui ∈ {0,1}`) and a weight (`c_ui`). Observed cells get high confidence; blanks keep confidence 1 (we *weakly* believe they're negatives). The objective sums over **all m·n cells**, weighted:

```
min_{P,Q}  Σ_{u,i}  c_ui ( p_ui − p_u·q_i )²  +  λ ( Σ_u‖p_u‖² + Σ_i‖q_i‖² )
```

#### 🔧 The efficient solve — why this is genuinely clever

A naive ALS step would sum over all `n` items for every user — `O(m·n·k²)`, hopeless. The trick: write user `u`'s update with the diagonal confidence matrix `Cᵘ = diag(c_u1,…,c_un)`:

```
p_u = ( Qᵀ Cᵘ Q + λI )⁻¹ Qᵀ Cᵘ p(u)
```

Now decompose `Cᵘ = I + (Cᵘ − I)`. The term `QᵀQ` is **shared across all users** — precompute it once per sweep in `O(n·k²)`. What remains, `Qᵀ(Cᵘ−I)Q`, only touches items where `c_ui ≠ 1`, i.e. the handful the user actually interacted with:

```
p_u = ( QᵀQ + Qᵀ(Cᵘ−I)Q + λI )⁻¹ Qᵀ Cᵘ p(u)
```

So per user you do work proportional to *their interaction count*, not the catalog size. Total per sweep: `O((m+n)·k² + nnz·k)` where `nnz` is the number of observed interactions. That `QᵀQ`-sharing is the entire reason iALS scales — internalize it.

#### 🔧 Tuning iALS (`projects/08` builds exactly this)

- **`α` (confidence scaling)** — *the* implicit-specific knob. Controls how much a high count outweighs a blank. Typical `α ∈ [1, 40]` on counts; for heavy-tailed play counts apply `log1p` *first* (project 08 stores raw counts and lets the model choose `1 + α·log1p(count)`) or you let one super-user's 350K plays dominate the loss.
- **`k` (factors)** — explicit MF lives at 20–200; iALS rewards going **much larger** (see below).
- **`λ` (regularization)** — scales with `α` and `k`; tune jointly, not in isolation. Best practice is **frequency-scaled** regularization — penalize each user/item proportional to how many interactions it has — which prevents popular items from being under-regularized.

#### 💡 iALS is still SOTA-competitive (Rendle et al. 2021/2022)

For years iALS was dismissed as a weak baseline that deep models trounce. **Rendle, Krichene, Zhang & Koren, *"Revisiting the Performance of iALS on Item Recommendation Benchmarks"* (RecSys 2022; arXiv 2110.14037)** demolished that narrative. With a "bag of tricks" — chiefly **large embedding dimension (in the hundreds-to-thousands, not the customary 64/128)** and **frequency-scaled regularization** — iALS *outperformed the state of the art on at least half of the comparisons* across four well-studied benchmarks (e.g. MovieLens-20M, Million Song Dataset, on Recall@20/Recall@50/nDCG). The lesson for a lead DS: **most "deep beats classic" claims are tuning artifacts.** Always tune your baseline as hard as your hero model — a theme echoed by Dacrema et al.'s reproducibility work (*"A Troubling Analysis of Reproducibility and Progress in Recommender Systems Research,"* TOIS 2021) and the BPR replicability study (Milogradskii et al., 2024).

---

### 4.5 BPR: Bayesian Personalized Ranking (Rendle et al. 2009)

iALS is *pointwise* — it fits each cell's score. But for top-K recommendation you don't care about absolute scores; you care about **order**. **Rendle, Freudenthaler, Gantner & Schmidt-Thieme, *"BPR: Bayesian Personalized Ranking from Implicit Feedback"* (UAI 2009)** optimizes order directly with a *pairwise* objective.

The assumption: for user `u`, an item `i` they interacted with should rank above an item `j` they didn't. Define `x̂_uij = x̂_ui − x̂_uj` (score gap; the per-MF scores are the usual `p_u·q_i`). Maximize the regularized log-likelihood that all such pairs are correctly ordered:

```
max  Σ_{(u,i,j)∈D_S}  ln σ(x̂_uij)  −  λ‖Θ‖²        σ(x) = 1/(1+e^−x)
```

`D_S` is the set of triples `(u, i, j)` with `i` positive and `j` an unobserved item for `u`. **Sampling** is essential: `D_S` is astronomically large, so LearnBPR draws triples by **bootstrap sampling** — pick a random observed `(u,i)`, then a random `j ∉` u's items — and takes an SGD step on each. The gradient is `(1 − σ(x̂_uij))` times the usual factor derivatives.

#### 🔧 When to prefer BPR over pointwise

- **Use BPR (pairwise / ranking loss)** when the deliverable is a *ranked list* and you're judged by Recall@K / nDCG / MAP. It directly optimizes the AUC-like ordering. Cross-ref the `BPRMatrixFactorization` in `projects/04_movielens_1m_recsys` and the BPR runs in `projects/08`.
- **Use iALS/WMF (pointwise)** when you want calibrated scores, the most scalable ALS solve, or simply a rock-solid baseline. iALS is often *faster to a good answer*; BPR can edge it out on ranking metrics but is sampler- and learning-rate-sensitive.
- **Negative sampling quality matters more than the loss.** Uniform negatives are weak; popularity-corrected or *hard* negatives (high-scoring non-interactions) sharpen BPR substantially — a bridge to the contrastive / two-tower training you'll meet in Part 7.

---

### 4.6 Three essential variants: SVD++, NMF, timeSVD++

**SVD++ (Koren 2008, *"Factorization Meets the Neighborhood,"* KDD 2008)** fuses explicit and implicit signals. Even on a ratings dataset, *which* items a user chose to rate is itself implicit feedback. SVD++ augments the user vector with the items they touched:

```
r̂_ui = μ + b_u + b_i + qᵢᵀ ( p_u + |N(u)|^(−½) Σ_{j∈N(u)} y_j )
```

`N(u)` is the set of items `u` interacted with and `y_j` are secondary item factors. That extra term consistently lowered RMSE on Netflix and is a strong explicit-feedback model — at the cost of a heavier per-user update.

**NMF (Non-negative Matrix Factorization)** constrains `P, Q ≥ 0`. Without sign cancellation, factors become **additive and parts-based** — a latent dimension reads like "amount of comedy," and a prediction is a sum of non-negative contributions. You trade a little accuracy for **interpretability**, which matters when stakeholders demand "why was this recommended?" Cross-ref the `NMFModel` in `projects/03` and `projects/04` (built on scikit-learn's NMF over an imputed rating matrix).

**timeSVD++ (Koren 2009, *"Collaborative Filtering with Temporal Dynamics,"* KDD 2009)** lets parameters drift over time: `b_i(t)`, `b_u(t)`, even `p_u(t)`. Tastes and item perception change (a movie's average rating creeps up after it wins an Oscar; a user goes through a horror phase). Making biases and factors time-dependent gave Netflix a further RMSE drop and is the conceptual ancestor of the **sequential / session models** in Part 6 and the **incremental MF** in `projects/07_streaming_recsys`.

---

### 4.7 Explicit vs. implicit MF — the decision table

| Dimension | Explicit MF (FunkSVD/SVD++) | Implicit MF (iALS/WMF, BPR) |
|---|---|---|
| Signal | Star ratings, scores | Clicks, plays, purchases, dwell |
| Missing cell means | Truly unknown — *skip it* | Weak negative — *include, low confidence* |
| Target | The rating value `r_ui` | Preference `p_ui ∈ {0,1}` + confidence `c_ui` |
| Loss summed over | Observed cells only | **All** cells (weighted) |
| Bias terms | Central (`μ, b_u, b_i`) | Usually dropped; popularity handled by sampling/confidence |
| Typical optimizer | SGD | ALS (iALS) or sampled SGD (BPR) |
| Eval metric | RMSE / MAE | Recall@K, nDCG@K, MAP, HitRate |
| Headline papers | Koren–Bell–Volinsky 2009; Koren 2008 | Hu–Koren–Volinsky 2008; Rendle 2009 |
| Repo | `projects/03`, `projects/04` | `projects/08` |

Note the evaluation switch: explicit MF is scored with `rmse`/`mae`, implicit with the ranking metrics (`recall_at_k`, `ndcg_at_k`, `hit_rate_at_k`, `average_precision_at_k`) in `gnnlab.metrics` (`evaluate_ranking`). **Optimizing RMSE does not optimize ranking** — a model can have great RMSE and recommend boring junk, which is why the implicit world abandoned RMSE.

---

### 4.8 The hyperparameter playbook

| Hyperparameter | Typical range | What it controls | Symptoms / signal |
|---|---|---|---|
| `k` (latent factors) | 16–200 (explicit); **256–2000+** (well-tuned iALS) | Model capacity | Too low → underfit, flat metrics; too high → overfit *unless* `λ` rises with it |
| `η` (learning rate, SGD/BPR) | 1e-3 – 1e-1 | Step size | Too high → loss diverges/NaN; too low → painfully slow, stuck high |
| `λ` (L2 regularization) | 1e-4 – 1e-1 (scale with `k`,`α`) | Shrinkage / overfit control | Train≫val gap → raise `λ`; both bad → lower `λ` or raise `k` |
| `α` (confidence, iALS) | 1 – 40 (on `log1p` counts) | Observed-vs-blank weighting | Too high → over-trusts heavy users, popularity bias; too low → ignores signal |
| epochs / sweeps | 10–50 (SGD); 10–20 (ALS) | Fitting time | Watch val metric; stop when it plateaus/turns (early stopping) |
| #negatives per positive (BPR) | 1–10 (more with hard negatives) | Ranking signal density | Too few → weak gradient; hard negatives help most |

**What matters most, in order:** (1) `k` and `λ` *jointly* — they trade off and must be tuned together; (2) `α` for any implicit model; (3) negative-sampling strategy for BPR; (4) learning rate for SGD/BPR. Use `projects/03`'s `tune.py` pattern: grid/random search on a held-out set, plot train-vs-val to read **underfitting** (both metrics poor, small gap) vs. **overfitting** (train great, val poor, large gap). The single most common rookie error is tuning the hero model hard and the baseline lazily — Rendle 2022 is the cautionary tale.

---

### 4.9 Edge cases and the limits of the family

- **Cold-start — the fatal flaw.** MF learns an embedding *per id seen in training*. A brand-new user or item has **no row in `P` / no column in `Q`**, so the model literally cannot score it — the dot product is undefined. Pure MF cannot embed unseen ids. The fix is to make embeddings a *function of features* (genre, text, demographics) rather than a lookup table — i.e. **content/feature models and two-tower architectures**, which we build in **Part 7**. Forward-reference: this single limitation is *why the field moved past MF*.
- **Fold-in for new users.** A *warm-ish* new user with a few interactions doesn't need full retraining: hold `Q` fixed and solve one ridge regression for their `p_u` (§4.3). Cheap, instant, and exactly what `projects/07_streaming_recsys` does incrementally.
- **Popularity bias.** Implicit MF and uniform-negative BPR over-recommend blockbusters (they dominate the interaction mass). Counter with popularity-corrected sampling, confidence transforms (`log1p`), or post-hoc re-ranking; measure it with `gnnlab.metrics` `average_recommendation_popularity`, `catalog_coverage`, and `gini_index`.
- **Scalability.** ALS parallelizes across users/items (§4.3) to billions of interactions; the binding constraint is usually the `O(nnz·k)` sufficient-statistics pass and the per-entity `k×k` solve — which is exactly why huge-`k` iALS, while accurate, gets expensive.

---

### 💡 What to read next → Part 5 (Factorization Machines & deep CTR)

MF answers *"given a user-id and an item-id, what's the score?"* But real systems drown in **side features** — age, device, time of day, query text, page context — and pure MF can't use them (and can't cold-start, §4.9). **Part 5** generalizes the MF dot-product to *arbitrary feature interactions* via **Factorization Machines (Rendle 2010)** — note `projects/04` already sneaks a `FactorizationMachine` in next to MF — and then climbs the **deep CTR** ladder (Wide & Deep, DeepFM, DCN, the two-tower retrieval of `projects/09`). The thread to carry forward: *latent factors as shared, low-rank parameters* is the idea that survives all the way into modern deep recommenders. Master it here and the rest is engineering.

---

## Part 5 — Factorization Machines & Deep CTR/Ranking (how ads & feeds are ranked)

> *Companion notebook: `projects/05_factorization_machines/` (LogisticRegression → FM → FFM → DeepFM → DCN-v2 on the Frappe CTR dataset). The `gnnlab.metrics` module supplies `auc` and `log_loss`; Tutorial.md Lesson 13 is the narrative twin of this part.*

In Part 4 you built retrieval — generating a few hundred plausible candidates out of millions. This part is about the **stage that decides money**: ranking. Once a feed or an ad auction has a candidate set, a model must assign each candidate a calibrated probability — *will this user click / install / convert?* — and the items are then sorted (and, in ads, multiplied by a bid). This is **click-through-rate (CTR) prediction**, and it is arguably the most economically important supervised-learning problem on earth: a 0.5% relative AUC gain at a large platform is tens of millions of dollars. We will climb a ladder of models — linear → FM → FFM → DeepFM → DCN-v2 → the modern zoo — and at every rung ask the professor's question: *what does this model add over the previous one, and when is the extra capacity actually worth it?*

### 5.1 The CTR framing: sparse categoricals, embeddings, and why pairwise interactions matter

A CTR training row is a pile of **categorical features**: `user_id`, `app_id`, `country`, `hour_of_day`, `device`, `weekday`, `is_weekend`, … Each field is one-hot encoded, so a single example is a vector $x \in \{0,1\}^n$ with $n$ in the **millions** (one slot per possible feature value) but only a handful of non-zeros — one per field. This extreme sparsity is the defining constraint of the domain, and it kills naive interaction models.

Why do we care about **interactions** at all? Because the signal lives in *conjunctions*, not marginals. "User likes pizza" and "it is dinner-time" are each weakly predictive; "user likes pizza **AND** it is dinner-time" is strongly predictive. A logistic regression $\hat y = w_0 + \sum_i w_i x_i$ can never express this — it is additive in features by construction. The classical fix is **Poly2** (a weight $w_{ij}$ per feature pair), but with $n$ in the millions that is $\binom{n}{2}$ parameters, and — fatally — a pair $(i,j)$ that *never co-occurs in training* gets no gradient and therefore no learned weight. On sparse data, Poly2 cannot generalize to unseen pairs.

The whole field is the story of **how to model pairwise (and higher) interactions cheaply and with generalization** on sparse data. The first great idea is to stop storing a number per pair and instead store a **vector per feature** — an *embedding* — and score a pair by the similarity of its two embeddings. That is the Factorization Machine.

🔧 **Practitioner's frame.** Think of each field as having an embedding table; a forward pass is `gather` the active rows, then a cheap interaction op, then a sigmoid. Everything in this part — FM, DeepFM, DLRM, DCN — is a different *interaction op* sitting on top of the same embedding lookup.

### 5.2 Factorization Machines (Rendle, 2010)

A second-order **Factorization Machine** (Rendle, 2010, *"Factorization Machines"*, ICDM) predicts:

$$\hat y(x) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle v_i, v_j\rangle\, x_i x_j$$

The first two terms *are* logistic regression. The third is the magic: every feature $i$ gets a $k$-dimensional latent vector $v_i \in \mathbb{R}^k$, and the interaction strength of features $i$ and $j$ is their **dot product** $\langle v_i, v_j\rangle$. Instead of a free parameter per pair (Poly2), the pair-weight is *factored* as $\langle v_i, v_j\rangle$. The payoff is generalization: even if $(i,j)$ never co-occurred, $v_i$ is learned from every *other* pair $i$ appears in, and $v_j$ from every pair $j$ appears in, so the model can still score their interaction. This factor-sharing is exactly why FMs dominate on sparse, high-cardinality data — and it is why the notebook finds FM crushing the linear baseline on Frappe.

**FM generalizes matrix factorization.** With only two fields, `user` and `item`, the only cross the FM can form is $\langle v_u, v_i\rangle$ — *exactly* the rating model of matrix factorization from Part 4. MF is the two-field special case of an FM; the FM is the generalization that lets you toss in `time`, `device`, `genre`, any side feature, and factorize every field against every other for free. (SVD++ and PITF are likewise FM special cases — see Rendle's paper.)

#### The O(k·n) reformulation — derive it

Computed naively, the double sum is $O(k\,n^2)$ — impossible at web scale. The key identity is the "square-of-sum minus sum-of-squares" trick. For any reals $a_1,\dots,a_n$:

$$\sum_{i<j} a_i a_j = \tfrac12\Big[\big(\textstyle\sum_i a_i\big)^2 - \sum_i a_i^2\Big]$$

because $(\sum_i a_i)^2 = \sum_i a_i^2 + 2\sum_{i<j} a_i a_j$. Now apply this **per latent dimension** $f$, with $a_i = v_{i,f}\,x_i$:

$$\sum_{i<j}\langle v_i, v_j\rangle x_i x_j = \sum_{f=1}^{k}\sum_{i<j} v_{i,f} v_{j,f}\, x_i x_j = \tfrac12 \sum_{f=1}^{k}\Big[\big(\textstyle\sum_i v_{i,f} x_i\big)^2 - \sum_i v_{i,f}^2 x_i^2\Big]$$

The inner sums $\sum_i v_{i,f} x_i$ and $\sum_i v_{i,f}^2 x_i^2$ each cost $O(n)$ (and on sparse data, $O(\#\text{non-zeros})$), and we repeat for $k$ dimensions, so the whole interaction term is **$O(k\,n)$ — linear**. This is the single most important algebraic identity in the part.

💡 **The notebook unit-tests this identity.** `FactorizationMachine.forward` computes the $O(kn)$ form, and `pairwise_naive` computes the $O(n^2)$ double loop; a test asserts they agree to floating-point tolerance. *Always* verify a clever reformulation against the dumb one — the O(kn) trick is exact, not an approximation, and a test makes that auditable. The code is the three lines `sum_then_sq = vx.sum(1).pow(2)`, `sq_then_sum = vx.pow(2).sum(1)`, `inter = 0.5*(sum_then_sq - sq_then_sum).sum(1)`.

**Hyperparameters that matter for FM:** the latent dimension $k$ (capacity of the interaction — the notebook sweeps this and shows AUC rising steeply then plateauing; too small under-fits the field structure, too large over-fits and wastes compute), **L2 regularization** on $w$ and $v$ (FMs over-fit sparse tails without it), and the **learning rate**. With one-hot fields where $x_i = 1$ on the active entries, the embedding lookup *is* the gather, so $vx$ is just the stacked per-field embeddings — which is why FM and a two-tower model share so much plumbing.

### 5.3 Field-aware Factorization Machines — FFM (Juan et al., 2016)

Plain FM gives feature $i$ **one** vector $v_i$ used in *every* interaction. **FFM** (Juan, Zhuang, Chin & Lin, 2016, *"Field-aware Factorization Machines for CTR Prediction"*, RecSys) observes that how `country` interacts with `app` may need a *different* latent direction than how the same `country` interacts with `weather`. So each feature gets a *separate* vector per **field of the partner**: $v_{i, f_j}$ is "feature $i$'s vector as seen by field $f_j$." The interaction becomes:

$$\sum_{i<j} \langle v_{i, f_j},\; v_{j, f_i}\rangle\, x_i x_j$$

where $f_j$ is the field that feature $j$ belongs to. This extra **field-awareness** is what won FFM the **Criteo** and **Avazu** Kaggle CTR competitions (and top-3 in several others). The cost is capacity: FFM has roughly `num_fields ×` the parameters of an FM, the $O(kn)$ collapse **no longer applies** (the vectors differ per pair, so it is an honest $O(\text{num\_fields}^2 \cdot k)$ double loop — fine when fields are few), and it **over-fits faster**. The practical recipe from the LIBFFM authors: a *smaller* $k$, strong early stopping, and careful L2. In the notebook FFM is implemented as one embedding table per field and the double loop over the ~10 Frappe fields.

💡 **The capacity-vs-overfitting lesson in miniature.** FFM is more expressive than FM *on paper*, yet on a clean, randomly-split dataset like Frappe it often does **not** beat FM on held-out AUC — the extra parameters buy variance you cannot afford. FFM's wins came on competition data with billions of rows where the capacity paid for itself. Capacity is only useful when you have the data to fill it.

### 5.4 Wide & Deep (Cheng et al., 2016) — memorization meets generalization

Google's **Wide & Deep** (Cheng et al., 2016, *"Wide & Deep Learning for Recommender Systems"*, DLRS) framed the central tension explicitly. The **wide** part is a linear model over raw and *cross-product* features — it **memorizes** specific, frequent conjunctions ("installed_app=Netflix AND impression_app=Pandora"). The **deep** part is an MLP over embeddings — it **generalizes** to unseen combinations via dense representations. Jointly training both gives you sharp memorization *and* smooth generalization. The catch — and the motivation for everything after — is that the wide part needs **hand-engineered cross features**. The next models automate exactly that.

### 5.5 DeepFM (Guo et al., 2017) — FM and a DNN, sharing embeddings

**DeepFM** (Guo, Tang, Ye, Li & He, 2017, *"DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"*, IJCAI) replaces Wide & Deep's hand-crafted wide part with an **FM**, and — crucially — the FM component and the deep MLP **share the same embedding table**:

$$\hat y = \sigma\big(\,y_{\text{FM}} + y_{\text{DNN}}\,\big)$$

The FM branch models **explicit second-order** interactions via the $O(kn)$ trick; the DNN branch consumes the *concatenated* embeddings and learns **implicit high-order** interactions. Sharing embeddings means low- and high-order signals train jointly with no feature engineering and no separate input pipeline — a genuinely elegant design. In the notebook, DeepFM reuses the exact FM interaction code for its FM head and an `MLP` block for the deep head.

### 5.6 DCN and DCN-v2 (Wang et al., 2017 / 2021) — an explicit cross network

The MLP in DeepFM learns interactions *implicitly* and inefficiently — an MLP is a poor approximator of multiplicative crosses. **Deep & Cross Network** (Wang et al., 2017, *"Deep & Cross Network for Ad Click Predictions"*, ADKDD) introduced a **cross network** that builds **explicit, bounded-degree** feature crosses by construction. **DCN-v2** (Wang et al., 2021, *"DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems"*, WWW) made the cross layer far more expressive. Its cross layer is:

$$x_{l+1} = x_0 \odot (W_l\, x_l + b_l) + x_l$$

where $\odot$ is element-wise product and — the key v1→v2 upgrade — $W_l$ is a full **matrix** (v1 used only a vector/rank-1 weight). Each layer multiplies the running representation back against the **original input $x_0$**, so after $l$ layers you have explicit polynomial crosses of bounded degree $l+1$ — interpretable, finite-degree interactions, in contrast to the MLP's open-ended ones. The residual $+\,x_l$ keeps gradients healthy.

Two engineering ideas make DCN-v2 **web-scale**. First, **low-rank** the cross matrix, $W_l \approx U_l V_l^\top$ with rank $r \ll d$, cutting parameters and FLOPs with little accuracy loss. Second, a **mixture of low-rank experts** with a softmax gate, $W = \sum_i g_i\, U_i V_i^\top$, letting different experts capture interactions in different subspaces — the same MoE idea you see everywhere in 2024-26 scaling work. The cross network can be **stacked** with the deep network (cross → deep) or run in **parallel** (cross ∥ deep, concatenate). DCN-v2 reported significant offline and online gains across many web-scale ranking systems at Google. The notebook's `DCNv2` runs the parallel form: a stack of cross layers alongside an `MLP`, concatenated into a logit.

| Model | What it adds over FM | When to use |
|---|---|---|
| Logistic Regression | (baseline — no interactions) | Sanity baseline; ultra-low-latency / interpretability-mandated |
| **FM** (Rendle '10) | factored pairwise interactions, $O(kn)$, generalizes to unseen pairs | Sparse high-cardinality data; strong, cheap, well-calibrated default |
| **FFM** (Juan '16) | per-field latent vectors (field-awareness) | Competition / very large data where capacity pays off; watch overfit |
| **Wide & Deep** ('16) | memorization (wide) + generalization (deep) | When you have good hand-crafted crosses + want DNN generalization |
| **DeepFM** (Guo '17) | FM **+** DNN sharing embeddings, no manual crosses | Strong general-purpose deep baseline; best AUC/effort ratio |
| **DCN-v2** (Wang '21) | explicit bounded-degree crosses + low-rank/MoE | Web-scale ranking; want explicit crosses *and* deep, efficiently |
| **DLRM** (Naumov '19) | dot-product interaction + huge sharded embeddings | Industrial scale with massive embedding tables (Meta-style infra) |
| **AutoInt** ('19) | self-attention over feature embeddings | When interaction importance is data-dependent / interpretable attention |
| **xDeepFM** ('18) | CIN: explicit *vector-wise* high-order crosses | When you want explicit high-order (beyond degree 2) crosses |

### 5.7 The modern ranking zoo — DLRM, AutoInt, xDeepFM, and what is actually used

**DLRM** (Naumov et al., 2019, *"Deep Learning Recommendation Model for Personalization and Recommendation Systems"*, Meta) is less a clever interaction trick and more a **systems** statement. Dense features go through a bottom MLP; each categorical goes through a (potentially terabyte-scale) **embedding table**; all embeddings (and the processed dense vector) are crossed by **explicit pairwise dot products**; the result feeds a top MLP. Its thesis: *interactions are paramount and a shallow dot-product captures them — you do not need a deep, exotic cross*. Its real contribution is **hybrid parallelism**: model-parallel the giant embedding tables (they do not fit on one device), data-parallel the MLPs. DLRM became the industry's open benchmark for systems/algorithm co-design.

**AutoInt** (Song et al., 2019, *"AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks"*, CIKM) applies **multi-head self-attention** over the feature embeddings, so the *importance* of each interaction is learned and input-dependent (and inspectable via attention weights). **xDeepFM** (Lian et al., 2018, *"xDeepFM"*, KDD) adds a **Compressed Interaction Network (CIN)** that builds explicit high-order crosses at the **vector-wise** level (FM-like), complementing the bit-wise MLP.

**Which is actually used (2026)?** In production, **DCN-v2 and DLRM-style architectures are the workhorses** — they are the strongest accuracy-per-FLOP and serve at latency. The frontier has moved to **scaling**: Meta's **Wukong** (Zhang et al., 2024, *"Wukong: Towards a Scaling Law for Large-Scale Recommendation"*) stacks FM blocks and demonstrated the **first clean scaling law** for recommendation — quality keeps improving across two orders of magnitude of compute, mirroring LLM scaling. ByteDance's **RankMixer** (2025) replaces quadratic self-attention with hardware-friendly **token-mixing + per-token FFNs**, lifting model-FLOPs-utilization from ~4.5% to ~45% and enabling 100× parameter scaling at fixed latency. The lesson: the FM/cross *interaction primitive* survived; the research energy is now in *scaling it efficiently on accelerators*.

### 5.8 Calibration — why LogLoss/ECE can matter more than AUC

Here is the subtlety that separates a senior practitioner from a Kaggler. **AUC measures ranking** — $P(\text{random positive scores above random negative})$ — and is invariant to any monotone rescaling of your scores. **Calibration** asks something AUC cannot see: when the model says **0.03**, do **3%** of those impressions actually click? In a **cost-per-click auction**, effective revenue is $\text{eCPM} = 1000 \cdot \text{pCTR} \cdot \text{bid}$ — you multiply the *raw probability* by money. A model that ranks perfectly but is systematically 2× overconfident will mis-price every auction, over-bid, and burn budget. So for bidding/charging you need **well-calibrated probabilities**, which AUC will happily certify as excellent even when they are badly miscalibrated.

This is measured by **LogLoss** (the cross-entropy CTR models actually optimize) and by **Expected Calibration Error (ECE)** — bin predictions, and within each bin compare *mean predicted probability* to *observed click rate*; ECE is the (weighted) average gap. ECE measures *only* calibration, AUC measures *only* discrimination — they are orthogonal, and a model can win one while losing the other.

💡 **The notebook's headline finding.** On Frappe the ladder is `LR ≪ FFM < FM < DeepFM ≈ DCN-v2` on **AUC** — the deep models *rank* best (highest AUC). But **LogLoss disagrees at the top**: the deep models' LogLoss (~0.196) is slightly **worse** than the plain **FM**'s (~0.174). The MLP branch sharpens the ordering but makes the logits a touch over-confident, which *hurts* calibration. The LR baseline is the cautionary tale: a respectable ranker (~0.93 AUC) but a poor calibrator (LogLoss ~0.37). The operational moral: **if your downstream system bids on the probability, the better-calibrated FM may beat the better-ranking DeepFM** — a decision you can only make by looking at LogLoss/ECE, *never AUC alone*. This is why we trace both. When deep models win AUC but lose calibration, the production fix is usually a cheap **post-hoc recalibration** layer (Platt scaling, isotonic regression, or field-aware calibration — Pan et al., 2020, *"Field-aware Calibration"*, WWW) bolted onto the deep model's logits, getting you the ranking *and* the calibration.

### 5.9 A hyperparameter playbook — what actually moves the needle

| Knob | Typical range | What it controls / when it matters |
|---|---|---|
| Embedding dim $k$ | 8–64 (FM small ~4–16) | Interaction capacity. Steep gains then plateau; bigger $k$ overfits sparse tails. FFM wants *smaller* $k$. |
| DNN hidden sizes | $[400,400,400]$ → $[1024,512,256]$ | Implicit high-order capacity. Width > depth past ~3 layers. |
| DNN depth | 2–4 layers | Beyond 3–4 layers, diminishing returns; harder to optimize. |
| Dropout | 0.0–0.5 | Regularizes the deep branch; the lever against MLP overconfidence. |
| L2 / weight decay | $10^{-6}$–$10^{-4}$ | Essential on sparse embeddings; the FM/FFM overfit lever. |
| # cross layers (DCN) | 2–4 | Max interaction degree = layers+1. More ≠ better; 2–3 usually peaks. |
| Cross rank $r$ (DCN-v2) | 32–512 | Low-rank cost/quality dial; smaller = cheaper, mild accuracy cost. |
| Learning rate (+ warmup) | $10^{-3}$ Adam, linear warmup | LR is the #1 lever; warmup stabilizes huge sparse-embedding training. |
| Batch size | 1k–16k+ | Larger batches → better embedding-table throughput; rescale LR. |

🔧 **Where the needle actually moves.** In rough priority order: (1) **features** — adding a genuinely new signal beats any architecture change; (2) **embedding dimension** and **regularization**, jointly; (3) **learning-rate schedule + warmup**; (4) **model family** (linear → FM is enormous; FM → DeepFM is real but smaller; DeepFM → DCN-v2 is often single-digit relative AUC). Architecture is the *last*, smallest lever — which is exactly why the notebook's ladder shows the biggest jump at LR→FM, not at DeepFM→DCN-v2.

### 5.10 Edge cases: high-cardinality, hashing, cold features, and train/serve skew

**High-cardinality categoricals & the hashing trick.** `user_id` can have hundreds of millions of values; a full embedding row each is infeasible (and useless for the long tail). The **hashing trick** (Weinberger et al., 2009) maps each raw value through a hash into a fixed bucket range $B$, so you carry a fixed-size table regardless of cardinality. The cost is **collisions** — distinct values sharing a row, which injects noise. Mitigations: make $B$ large enough that collisions are rare on the *head*, use **multiple hash functions** / Bloom-filter-style codes (Serrà & Karatzoglou, 2017), or move to learned **semantic IDs** that group similar items deliberately rather than randomly (Singh et al., 2023). The tail collides either way — and that is acceptable, because tail values have little signal.

**Cold features & cold start.** A brand-new `item_id` has an untrained (or hashed-collision) embedding; its prediction leans on whatever side/context features the model has. This is precisely where FM's **factor-sharing** earns its keep — a never-before-seen *pair* can still be scored from each feature's other interactions — and where rich context fields matter most. (The notebook makes exactly this point: adding context fields barely helps a warm random split, but you should expect them to matter when IDs fail.)

**Embedding collisions vs. capacity.** Smaller tables (more collisions) regularize and save memory but blur the head; larger tables sharpen the head but cost RAM and overfit the tail. This bucket-size/collision tradeoff is one of the highest-leverage and most underrated knobs in a production CTR stack.

**Train/serve skew.** The classic production killer: a feature computed one way offline (full-day aggregate) and another way online (partial-day, different default for missing), so the model sees a different distribution at serving than at training. AUC offline looks fine; live metrics tank. Defenses: a **shared feature-transformation library** used by both training and serving, **logging the exact features at serving time** to train on, and monitoring the **train-vs-serve feature distributions** continuously.

### What to read next → Part 6 (Deep CF & sequential recommendation)

We have ranked items as **bags of static features**. But user behavior is a **sequence** — the order and recency of what you watched last carries enormous signal that an FM's set-of-features view throws away. **Part 6** moves to **deep collaborative filtering and sequential recommendation**: from NeuMF/neural CF, through GRU4Rec and the self-attentive **SASRec**/**BERT4Rec**, to transformer-based behavior-sequence models (DIN/DIEN, BST) — how to model *what the user did, in order*, and feed it back into exactly the ranking stage you just built.

---

#### References

- Rendle, S. (2010). *Factorization Machines*. ICDM 2010.
- Juan, Y., Zhuang, Y., Chin, W.-S., & Lin, C.-J. (2016). *Field-aware Factorization Machines for CTR Prediction*. RecSys 2016. (LIBFFM.)
- Cheng, H.-T. et al. (2016). *Wide & Deep Learning for Recommender Systems*. DLRS 2016 (Google).
- Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017). *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*. IJCAI 2017.
- Wang, R., Fu, B., Fu, G., & Wang, M. (2017). *Deep & Cross Network for Ad Click Predictions*. ADKDD 2017 (Google).
- Wang, R. et al. (2021). *DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems*. WWW 2021 (Google).
- Naumov, M. et al. (2019). *Deep Learning Recommendation Model for Personalization and Recommendation Systems* (DLRM). Meta.
- Song, W. et al. (2019). *AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks*. CIKM 2019.
- Lian, J. et al. (2018). *xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems*. KDD 2018.
- Zhang, B. et al. (2024). *Wukong: Towards a Scaling Law for Large-Scale Recommendation*. (Meta.)
- *RankMixer: Scaling Up Ranking Models in Industrial Recommenders* (2025, ByteDance). arXiv:2507.15551.
- Pan, F. et al. (2020). *Field-aware Calibration: A Simple and Empirically Strong Method for Reliable Probabilistic Predictions*. WWW 2020.
- Weinberger, K. et al. (2009). *Feature Hashing for Large Scale Multitask Learning*. ICML 2009.
- Singh, A. et al. (2023). *Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations*. (Semantic IDs.)

---

## Part 6 — Deep Collaborative Filtering, Autoencoders & Sequential Models

> *Where the last part left a user as a static row in a matrix, this part puts the clock back into the model. We will earn the right to use deep nets — and learn, repeatedly, that "deep" is a tool, not a trophy.*

Let me set the frame before any equations. There are two questions a recommender can answer, and confusing them is the most common rookie mistake I see.

1. **"Does user *U* like item *I*?"** — a *static* taste question. Matrix factorization, BPR, FM, LightGCN all answer this. The user is a row; order is discarded.
2. **"Given everything *U* just did, what do they want *next*?"** — a *sequential* question. This is what YouTube, TikTok, Spotify and Amazon actually serve, because *order carries enormous signal that the user–item matrix throws away*: three episodes watched → episode four is obvious; a factorization model has no notion of "next."

This part walks the bridge from (1) to (2). We start with deep *static* CF (Neural CF, autoencoders) — partly because it's foundational, and partly because it delivers the single most important sobering lesson in modern RecSys: **a well-tuned simple model is a brutal baseline**. Then we cross into sequential models — the heart of the part, and the heart of `projects/06_sequential_recommendation`.

---

### 6.1 Neural CF — and the critique that should change how you think

**The idea (He et al. 2017, *Neural Collaborative Filtering*, WWW).** Classical matrix factorization scores a user–item pair with a **dot product** of their embeddings: $\hat{y}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i$. He et al. argued the dot product is an unnecessarily rigid way to combine two embeddings, and proposed *learning* the interaction function with a neural net. **NeuMF** (their flagship) fuses two towers:

- **GMF** (generalized MF): an element-wise product $\mathbf{p}_u \odot \mathbf{q}_i$ followed by a learned linear layer — MF as a special case.
- **MLP**: concatenate $[\mathbf{p}_u; \mathbf{q}_i]$ and push through a multi-layer perceptron, letting the network learn a "similarity" from scratch.

Concatenate the two tower outputs, feed a final sigmoid, train with binary cross-entropy on implicit feedback (observed = 1, sampled negatives = 0). The paper reported NeuMF beating MF on MovieLens and Pinterest, and the field — primed by the deep-learning wave — took "learned similarity > dot product" as gospel.

🔧 **The critique you must internalize — Rendle et al. 2020, *Neural Collaborative Filtering vs. Matrix Factorization Revisited* (RecSys).** Rendle, Krichene, Zhang and Anderson re-ran the comparison with *properly tuned* baselines and found the headline result reverses: **a well-tuned dot product substantially outperforms the learned MLP similarity** on the very same benchmarks. Two reasons, both worth understanding:

- **Learning a dot product with an MLP is hard.** An MLP is a universal approximator *in the limit*, but approximating the simple multiplicative structure $\sum_k p_{uk} q_{ik}$ from finite data is non-trivial — you spend capacity and samples relearning an inductive bias you could have just *built in*. The dot product is not a weakness; it's a strong, correct prior.
- **The original baselines were under-tuned.** Much of NeuMF's reported lift was a tuning artifact, not an architectural one.

💡 **The lesson — deep ≠ automatically better.** This is the thesis of an entire genre of reproducibility papers: Dacrema et al. 2019 (*Are We Really Making Much Progress?*, RecSys) showed that for **11 of 18** recent neural recommenders, simple, well-tuned baselines (ItemKNN, a tuned linear model) matched or beat them. Before you believe any new model, *tune the baseline first*. This is exactly why our notebook climbs a ladder — POP → ItemKNN → Markov → GRU4Rec → SASRec — instead of jumping straight to the Transformer: each rung must *earn* its complexity against the one below.

There's also a **production** corollary Rendle hammers: a dot product enables **maximum-inner-product search** via ANN (FAISS/ScaNN) over millions of items in milliseconds; an MLP scorer forces you to forward-pass every candidate, which is intractable at retrieval scale. The dot product isn't just as-good — it's *deployable*. (We pick this thread back up in Part 7.)

---

### 6.2 Autoencoder CF — reconstructing the user's whole interaction vector

A different deep idea: don't score one pair at a time — feed the model a user's **entire** interaction vector $\mathbf{x}_u \in \{0,1\}^{|I|}$ and ask it to **reconstruct** it. Items the model "fills in" that the user hadn't yet consumed *are* the recommendations.

**AutoRec (Sedhain et al. 2015, WWW).** The first autoencoder CF: a single hidden layer reconstructing the rating vector, MSE loss. Simple, but it set the template.

**Mult-VAE / VAE-CF (Liang et al. 2018, *Variational Autoencoders for Collaborative Filtering*, WWW).** This is the one to study. It upgrades the autoencoder on two axes that *match the structure of the problem*:

- **Multinomial likelihood.** A user's clicks are not independent Gaussian quantities — they're a **budget** of attention spread across items, i.e. draws from a categorical distribution. Mult-VAE models $\mathbf{x}_u \sim \text{Multinomial}(N_u,\, \boldsymbol{\pi}(\mathbf{z}_u))$, where $\boldsymbol{\pi}$ is a softmax over all items produced by the decoder from latent $\mathbf{z}_u$. The log-likelihood is $\sum_i x_{ui} \log \pi_i(\mathbf{z}_u)$ — a ranking-friendly objective that, crucially, **couples items through the softmax**: pushing probability onto one item pulls it off others, which is precisely the competition you want in a top-N list. Liang et al. show multinomial beats the Gaussian/logistic likelihoods that AutoRec-style models used.

- **Partial regularization (β-VAE).** The VAE objective is the ELBO:
  $$\mathcal{L}(\mathbf{x}_u) = \mathbb{E}_{q(\mathbf{z}_u|\mathbf{x}_u)}\big[\log p(\mathbf{x}_u|\mathbf{z}_u)\big] \;-\; \beta \cdot \mathrm{KL}\!\big(q(\mathbf{z}_u|\mathbf{x}_u)\,\|\,p(\mathbf{z}_u)\big).$$
  Standard VAEs set $\beta = 1$. Liang et al.'s key empirical finding: for CF, $\beta = 1$ **over-regularizes** — the KL term forces the latent code toward the prior and the model under-fits the user. They **anneal $\beta$** from 0 up toward a *tuned* value typically $< 1$ (often ≈0.2), trading a little of the Bayesian generative purity for materially better ranking. The intuition: in recommendation we care about reconstruction (ranking) far more than a clean generative prior, so we *deliberately weaken* the regularizer.

🔧 **When Mult-VAE is strong.** It shines on **dense-ish implicit feedback** (MovieLens-20M, Netflix) and gives strong, **diverse** top-N lists because the multinomial softmax spreads mass. It's a *user-encoder* — at inference you pass the user's history vector through the encoder, so it **adapts to new interactions without retraining** (great for warmish users), but a user with an *empty* vector still gets nothing personalized (cold-start, §6.7).

💡 **Kinship to EASE (cross-ref Part 8 / `08` autoencoder section).** Here's the punchline that ties the autoencoder family together. **EASE** (Steck 2019, *Embarrassingly Shallow Autoencoders for Sparse Data*, WWW) is a *linear* autoencoder $\hat{\mathbf{x}}_u = \mathbf{x}_u \mathbf{B}$ with the diagonal of $\mathbf{B}$ constrained to zero (so an item can't reconstruct itself). Its training objective has a **closed-form solution**:
$$\mathbf{B} = \mathbf{I} - \mathbf{P}\,\mathrm{diag}\!\Big(\tfrac{1}{\mathrm{diag}(\mathbf{P})}\Big), \qquad \mathbf{P} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}.$$
No SGD, no epochs — one matrix inverse. And Steck showed this shallow linear model **matches or beats** Mult-VAE on most public datasets. The lesson rhymes with Rendle: the *deep* VAE's advantage over a well-built *shallow* item-item model is smaller than you'd guess. Reach for Mult-VAE when non-linearity and multi-modal user tastes genuinely help (large, dense catalogs); reach for EASE when you want a strong, instant, interpretable item-item baseline — which is most of the time. They are two points on one continuum: *learn an item-item interaction matrix that reconstructs the user vector.*

---

### 6.3 Sequential & session-based recommendation — putting the clock back in

Now the heart of the part, and of `projects/06_sequential_recommendation`. We move from $\mathbf{x}_u$ (an unordered set) to $\mathbf{s}_u = (i_1, i_2, \dots, i_t)$ (an *ordered* sequence) and predict $i_{t+1}$. The evaluation protocol changes with it: **leave-last-out by time** — hold out each user's genuinely *last* interaction as the test target, the second-to-last as validation, train on the rest (§6.7). This is leakage-free because you never let the future predict the past, and it mirrors what production actually does.

#### Markov chains and FPMC — the floor

The simplest sequential model: a **first-order Markov chain** predicts $i_{t+1}$ from $i_t$ alone via item-to-item transition counts $P(i_{t+1}\mid i_t)$. It's our notebook's `Markov` rung and a shockingly strong baseline for repeat/co-occurrence-heavy catalogs.

**FPMC (Rendle et al. 2010, *Factorizing Personalized Markov Chains for Next-Basket Recommendation*, WWW)** fixes its two weaknesses. Raw transition counts are (a) **sparse** (most item-pairs never co-occur) and (b) **impersonal** (same transition for everyone). FPMC **factorizes** a personalized user × item × item transition *tensor*, combining MF (long-term user taste) with a Markov chain (short-term "what you just did"), trained with **BPR**-style pairwise ranking. It is, in spirit, the first model to say *recommendation = long-term preference + short-term sequence*, a decomposition every later model inherits.

#### GRU4Rec — the RNN entry point

**GRU4Rec (Hidasi et al. 2016, *Session-Based Recommendations with Recurrent Neural Networks*, ICLR)** was the first deep session model. A GRU consumes the session item-by-item; its hidden state is a learned summary of "where this session is going," and the output layer scores the next item. Three engineering ideas made it work and are worth knowing:

- **Session-parallel mini-batches.** Sessions have wildly different lengths, so you can't pad them into a tidy matrix cheaply. Instead, stack the *active step* of many sessions side-by-side; when a short session ends, slot a new one into its row. This keeps the GPU busy and decorrelates the batch.
- **Sampling the output.** Softmax over a huge item vocabulary is expensive, so GRU4Rec scores the positive plus **negatives sampled from other sessions in the mini-batch** (popularity-proportional, "for free").
- **Ranking losses: TOP1 and BPR-max.** Plain BPR's gradient *vanishes* once the positive comfortably beats a negative — bad when most negatives are easy. GRU4Rec-v2 (Hidasi & Karatzoglou 2018) introduced **BPR-max** and **TOP1-max**, which weight the loss toward the *highest-scoring negatives* (the ones actually threatening the rank), restoring useful gradient and yielding large lifts. 💡 Lesson: in ranking, *which* negatives you compare against matters as much as the architecture.

#### SASRec — the workhorse

**SASRec (Kang & McAuley 2018, *Self-Attentive Sequential Recommendation*, ICDM)** is the model to know cold. It replaces the RNN with a **causal self-attention Transformer**: at each position it attends only to *earlier* items (a lower-triangular mask), so the representation at step $t$ is an adaptively-weighted blend of the relevant past — long-range dependencies an RNN's single hidden state struggles to keep.

- **Architecture.** Item embedding + learned positional embedding → `num_blocks` Transformer blocks (multi-head self-attention + point-wise feed-forward, residual + LayerNorm) → score next item by **dot product** of the final position's representation with item embeddings (note the dot product again — §6.1's lesson recurs).
- **Objective & training.** It's trained as **next-item prediction at every position simultaneously** (shift the sequence by one; the causal mask makes all positions valid targets at once — very sample-efficient). The original loss is **binary cross-entropy with one sampled negative per position**.
- **`max_len`.** Self-attention is $O(L^2)$ in sequence length, so you keep only the most recent `max_len` items (50–200 typical; our notebook uses 50). Recency dominates, so this rarely hurts.

SASRec is fast, strong, and — this is the headline — **still SOTA-competitive in 2025-26**. Most "new SOTA" papers are measured against it, and many don't survive contact with a *well-tuned* SASRec.

#### BERT4Rec — bidirectional masking, and a replicability nuance

**BERT4Rec (Sun et al. 2019, *BERT4Rec*, CIKM)** borrows the masked-language-model trick: instead of causal left-to-right, it uses **bidirectional** self-attention and trains by **masking random items in the sequence and predicting them** from both sides. The pitch: a user's taste at step $t$ is informed by what came *after*, too (during training), so the encoder learns richer item representations. It reported beating SASRec.

🔧 **The replicability nuance — Petrov & Macdonald 2022 (*A Systematic Review and Replicability Study of BERT4Rec*, RecSys) and Klenitskiy & Vasilev 2023 (*Turning Dross Into Gold Loss: Is BERT4Rec Really Better Than SASRec?*, RecSys).** Two findings reframe the comparison:

- BERT4Rec is **hard to reproduce** — its reported gains often required far more training epochs than papers using it as a baseline gave it.
- The real source of BERT4Rec's edge is **not the bidirectional architecture — it's the loss**. BERT4Rec used full **softmax cross-entropy over all items**; SASRec used **one sampled negative**. When you train SASRec with the *same* full-softmax cross-entropy, **SASRec matches or beats BERT4Rec — and trains faster.** (gSASRec, Petrov & Macdonald 2023, formalizes this: more negatives + a calibrated gBCE loss close the gap and reduce overconfidence.)

💡 **The meta-lesson, a third time:** the gain attributed to a flashy *architecture* was really a *training-objective* difference. Always ablate the loss before crediting the model.

#### Newer directions (2020–2026) — know they exist

- **S³-Rec (Zhou et al. 2020):** self-supervised **pre-training** with mutual-information objectives over items/attributes/segments, then fine-tune. Helps under sparsity; gains over a tuned SASRec are modest.
- **CL4SRec (Xie et al. 2022):** **contrastive learning** — augment a sequence (crop / mask / reorder), pull the two views together in embedding space. A regularizer that helps most when data is sparse.
- **Frequency-enhanced (BSARec, Shin et al. AAAI 2024; FEARec 2023):** self-attention is a **low-pass filter** — it smooths and loses high-frequency (sharp, recent-shift) signal. These add a Fourier/frequency branch to re-inject it. A genuinely *new* inductive bias rather than a loss trick — promising.
- **Generative / Semantic-ID (TIGER, Rajput et al. 2023, NeurIPS):** quantize each item into a short token sequence (RQ-VAE) and **autoregressively generate the next item's ID** like language. The "ChatGPT moment of RecSys" — frontier, covered as the horizon in Part 7.

---

### 6.4 The sequential model → idea → when-to-use map

| Model | Core idea | Reach for it when |
|---|---|---|
| **Markov / item-item** | $P(i_{t+1}\mid i_t)$ from transition counts | strong/cheap baseline; repeat & co-occurrence catalogs; always run it |
| **FPMC** (Rendle 2010) | personalized factorized Markov tensor = MF + MC | small data; you want long-term taste *and* last-action signal, no deep stack |
| **GRU4Rec** (Hidasi 2016) | GRU over the session; BPR-max/TOP1 losses | pure-session (anonymous) data; gentle deep entry point |
| **SASRec** (Kang & McAuley 2018) | causal self-attention Transformer | **default workhorse** — fast, strong, deployable; start here |
| **BERT4Rec** (Sun 2019) | bidirectional masked-item Transformer | only after SASRec+softmax is beaten; budget for long training |
| **BSARec / FEARec** (2023–24) | + frequency branch to recover high-freq signal | rapid taste shifts; SASRec feels "too smooth" |
| **TIGER / generative** (2023→) | Semantic IDs, autoregressive item generation | content-rich catalogs, cold-start via content; frontier (Part 7) |

---

### 6.5 Hyperparameter playbook for SASRec / GRU4Rec

Select on **validation NDCG@10**; read the test number *only* for the winning config. (Same discipline as everywhere in this repo: tune on validation, pick the elbow, never let a test number choose your config.)

| Knob | Typical range | What it does / what matters | Senior-DS guidance |
|---|---|---|---|
| **embedding dim** | 32 – 256 (64 common) | model capacity | **helps then saturates**; on a dense set like ML-1M, 32–64 is the sweet spot — bigger overfits & inflates params |
| **max sequence len** | 50 – 200 | how much history is visible; $O(L^2)$ cost | **helps then saturates** — recency dominates, so pick the *elbow*; longer = slower for little gain |
| **# blocks (depth)** | 1 – 3 (**2** classic) | depth of the Transformer | **diminishing, sometimes negative** returns; **2 blocks** is the canonical SASRec setting — rarely pays to stack more on modest data |
| **# heads** | 1 – 4 (2 common) | attention sub-spaces | low sensitivity; 1–2 is fine |
| **dropout** | 0.1 – 0.5 | regularization on sparse seqs | the **most underrated knob** — sparse data needs *more* dropout (0.2–0.5); too little overfits fast |
| **learning rate** | 1e-3 – 1e-4 (Adam) | optimization | dominant non-architectural knob; warmup helps on long runs |
| **# negatives / loss** | 1 → many; BCE → **softmax** | the §6.3 lesson | **the single biggest lever** — more negatives or full-softmax CE often beats any architecture change |

💡 **Cross-ref the notebook's sensitivity study (`projects/06_sequential_recommendation`, §8).** The one-knob sweeps confirm the textbook picture exactly: **`max_len` and `embed_dim` help then plateau** (you pick the elbow, not the max), and **`num_blocks = 2` is the practical default** — deeper rarely pays on a dataset this size. The meta-lesson the notebook ends on is the one to carry: *most of your gains come from getting the loss and a handful of knobs right, not from a fancier block.*

---

### 6.6 Formula sketches to keep in your head

**SASRec self-attention (causal).** With $\mathbf{Q},\mathbf{K},\mathbf{V}$ projections of the position embeddings $\mathbf{E}\in\mathbb{R}^{L\times d}$:
$$\mathrm{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \mathrm{softmax}\!\Big(\tfrac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}} + \mathbf{M}\Big)\mathbf{V}, \qquad M_{jk} = \begin{cases} 0 & k \le j \\ -\infty & k > j \end{cases}$$
The mask $\mathbf{M}$ enforces causality: position $j$ cannot peek at the future.

**Mult-VAE objective** (multinomial likelihood + annealed-$\beta$ KL): see §6.2. **EASE** closed form: see §6.2. The recurring structural fact across all of §6.1–6.2: **the final score is a dot product** (user-rep · item-emb), which is why these models drop cleanly into retrieval (Part 7).

---

### 6.7 Edge cases a senior DS must handle

- **Short sequences / new users (cold-start).** Sequential models need history; a 1-click user gets near-popularity output. Mitigations: fall back to POP/content features for very short sequences; blend a content/Semantic-ID signal (the generative-model angle); never report metrics only on long-history users — *segment by sequence length* and look at the short-tail.
- **Repeat consumption.** Music/groceries/news are heavily repeat-driven; "next item" is often a *re-listen*. Markov and item-item models exploit this naturally; pure novelty-seeking models *under*-perform here. Decide explicitly whether re-recommending a consumed item is a win (music) or a bug (a movie you finished).
- **Exposure bias in logged sequences.** Your training sequences are what the *old recommender showed*, not free user choice — the data is a feedback loop. A model trained naively learns to imitate the previous system. This is why production teams add **off-policy / IPS-style correction** and log un-personalized exploration traffic (covered with counterfactual evaluation later).
- **Popularity bias & filter bubbles.** A model that maximizes NDCG often does so by **recommending blockbusters to everyone** — accurate *and* a filter bubble. This is why the notebook judges every model on **accuracy AND beyond-accuracy** (coverage, novelty, Gini, ARP — all in `gnnlab.metrics`): SASRec should lift NDCG over POP *while also* raising coverage/novelty and lowering average recommendation popularity. If your "better" model is just more popularity-biased, it isn't better.
- **Evaluation = leave-last-out by time.** Predict the genuinely *next* item; rank the held-out positive against negatives the user never saw (a sampled pool *or*, better, the full catalog — sampled-metric artifacts are a known trap, Krichene & Rendle 2020). Report **NDCG@10 / HR@10** as headline, then the beyond-accuracy panel.

---

### 🎓 What to read next → Part 7 (Graph CF & Production Retrieval)

You now own two model families — deep *static* CF (NeuMF, Mult-VAE, EASE) and *sequential* models (FPMC → GRU4Rec → SASRec → BERT4Rec) — and, more importantly, the **skeptic's discipline** that runs through all of them: *tune the baseline, ablate the loss, judge beyond accuracy.* **Part 7** turns the user–item matrix into a **graph** (LightGCN/NGCF — message passing as collaborative filtering) and then assembles the **production retrieval→ranking stack**: two-tower models, ANN search (FAISS/ScaNN), and where these sequential and graph embeddings actually plug in to serve millions of items per request in milliseconds. The dot product that kept reappearing in this part is the hinge on which that entire architecture turns — and the frontier (generative / Semantic-ID recommendation, TIGER) is where it's all heading.

---

#### References cited

- He, Liao, Zhang, Nie, Hu & Chua (2017). *Neural Collaborative Filtering.* WWW.
- Rendle, Krichene, Zhang & Anderson (2020). *Neural Collaborative Filtering vs. Matrix Factorization Revisited.* RecSys.
- Dacrema, Cremonesi & Jannach (2019). *Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches.* RecSys.
- Sedhain, Menon, Sanner & Xie (2015). *AutoRec: Autoencoders Meet Collaborative Filtering.* WWW.
- Liang, Krishnan, Hoffman & Jebara (2018). *Variational Autoencoders for Collaborative Filtering.* WWW.
- Steck (2019). *Embarrassingly Shallow Autoencoders for Sparse Data (EASE).* WWW.
- Rendle, Freudenthaler & Schmidt-Thieme (2010). *Factorizing Personalized Markov Chains for Next-Basket Recommendation (FPMC).* WWW.
- Hidasi, Karatzoglou, Baltrunas & Tikk (2016). *Session-Based Recommendations with Recurrent Neural Networks (GRU4Rec).* ICLR. — Hidasi & Karatzoglou (2018), *Recurrent Neural Networks with Top-k Gains (BPR-max/TOP1-max).* CIKM.
- Kang & McAuley (2018). *Self-Attentive Sequential Recommendation (SASRec).* ICDM.
- Sun, Liu, Wu, Pei, Lin, Ou & Jiang (2019). *BERT4Rec.* CIKM.
- Petrov & Macdonald (2022). *A Systematic Review and Replicability Study of BERT4Rec.* RecSys. — and Petrov & Macdonald (2023), *gSASRec.*
- Klenitskiy & Vasilev (2023). *Turning Dross Into Gold Loss: Is BERT4Rec Really Better Than SASRec?* RecSys.
- Zhou et al. (2020) *S³-Rec*; Xie et al. (2022) *CL4SRec*; Shin et al. (2024) *BSARec* (AAAI); Rajput et al. (2023) *TIGER* (NeurIPS).
- Krichene & Rendle (2020). *On Sampled Metrics for Item Recommendation.* KDD.

---

## Part 7 — Graph Models & the Production Retrieval→Ranking Stack

Up to now you have learned to *score a (user, item) pair*. That is a beautiful, self-contained problem — and it is also a lie about how recommenders actually run. A real system at Netflix, Pinterest, or YouTube cannot afford to score a pair. It has tens of millions of items and a sub-100ms budget, and it must *narrow the universe down* before it can afford to be careful. This part teaches the two ideas that make that possible: **graph neural networks**, which turn the sparse user–item matrix into a rich propagation structure and give you better embeddings; and the **retrieval→ranking** funnel, the architecture that lets you serve those embeddings to millions of users in milliseconds. They are deeply connected — the graph model *produces* the embeddings the retrieval stage *serves* — which is exactly why they belong in one part.

---

### 7.1 The user–item bipartite graph: why graphs help collaborative filtering

Matrix factorization (Part 3/04) learns a vector `e_u` per user and `e_i` per item so that `e_u · e_i` predicts preference. The training signal is *only* the directly observed interactions — one hop. But collaborative filtering's whole premise is transitive: "users similar to you liked items similar to this." That similarity lives in **paths**, not edges.

Model interactions as a **bipartite graph**: users on one side, items on the other, an edge for every observed interaction. Now "high-order connectivity" becomes literal. A path `u → i → u' → i'` of length 3 says *another user `u'` who shares item `i` with you also liked `i'`* — a second-order collaborative signal. NGCF (Wang et al. 2019, *Neural Graph Collaborative Filtering*) made the key observation: plain MF never sees these paths; it must *infer* them indirectly through the dot product. A GNN, by contrast, **explicitly propagates** embeddings along edges, so a `k`-layer model bakes `k`-hop collaborative signal directly into each node's representation. That is the entire intuition for why GNNs help CF — they make the latent graph structure a *first-class part of the forward pass* instead of something the loss has to reconstruct.

💡 **The mental model:** MF is a GNN with zero propagation layers. Every graph recommender in this part is "MF + structured smoothing of embeddings over the interaction graph." Once you see it that way, the design space is just *how* you smooth.

---

### 7.2 NGCF → LightGCN: the great simplification (cross-ref `04_movielens_1m_recsys`)

**NGCF (Wang et al. 2019).** NGCF imported the standard GCN recipe — at each layer, aggregate neighbours, apply a learned **feature-transformation matrix**, add a nonlinearity. Its propagation also added an element-wise *affinity* term `e_i ⊙ e_u` to model interaction between a node and its neighbour:

```
m_{u←i} = (1/√(|N_u||N_i|)) · ( W1 e_i  +  W2 (e_i ⊙ e_u) )
e_u^(l+1) = σ( W1 e_u + Σ_{i∈N_u} m_{u←i} )
```

It worked, and it beat MF — but it was heavy, slow, and finicky to train. (In `04/src/gnn.py` NGCF is implemented exactly this way, with `W1`/`W2` per layer and per-layer concatenation, precisely so you can *see* the cost.)

**LightGCN (He et al. 2020, *Simplifying and Powering Graph Convolution Network for Recommendation*, SIGIR).** He et al. ran the ablation everyone should run: they removed the feature-transform matrices, then the nonlinearity, then the self-connection — and accuracy went **up** every time. Their argument: in CF the node "features" are just free ID embeddings with no semantic content, so a per-layer linear transform and a ReLU add nothing but optimization difficulty and over-fitting. What remains is the essential operation — **symmetric-normalized neighbour aggregation**:

```
e_u^(k+1) = Σ_{i∈N_u}  1/√(|N_u| |N_i|)  e_i        (and symmetrically for items)
```

with **no weights and no nonlinearity in the propagation at all**. The only learned parameters are the layer-0 embeddings `E^(0)`. Then comes the second clever piece — **layer combination**. Different depths capture different signal (layer 0 = identity, deeper = broader smoothing but more over-smoothing), so LightGCN reads out a *weighted sum across layers*:

```
E = Σ_{k=0}^{K} α_k E^(k)        (typically α_k = 1/(K+1), a simple mean)
```

This self-regularizes against over-smoothing: by always keeping the layer-0 term in the readout, deep propagation can't wash every node toward the same vector. Train it with **BPR loss** (Bayesian Personalized Ranking — pull observed items above sampled negatives) and L2-regularize *only the layer-0 embeddings*. In `04`, LightGCN is fewer lines than NGCF and typically matches or beats it — the canonical "the simpler model won" lesson.

🔧 **Practitioner note:** LightGCN's normalization `1/√(|N_u||N_i|)` is not cosmetic — it down-weights high-degree (popular) neighbours so a blockbuster movie doesn't dominate every user's representation. Drop it and popularity bias explodes.

---

### 7.3 PinSAGE: GraphSAGE at web scale (Ying et al. 2018, Pinterest)

LightGCN's full-graph propagation needs the *entire* normalized adjacency in memory — fine for MovieLens, impossible for Pinterest's **3 billion nodes and 18 billion edges**. PinSAGE (Ying et al. 2018, *Graph Convolutional Neural Networks for Web-Scale Recommender Systems*, KDD) was the first billion-scale industrial GNN recommender, and its innovations are all about *scale*, building on **GraphSAGE** (Hamilton et al. 2017 — cross-ref `01/02`) which you already know: sample a fixed-size neighbourhood, aggregate, repeat.

Three ideas make it work:
- **Random-walk neighbourhoods.** Instead of "all neighbours" (unbounded) or "uniform random neighbours" (noisy), PinSAGE runs short random walks from each node and defines its neighbourhood as the **top visited nodes by visit count**. This both bounds compute *and* gives an **importance weight** per neighbour — importance-pooling aggregation weights neighbours by how often the walk visited them. This is a smarter inductive bias than uniform sampling.
- **Producer–consumer minibatch + MapReduce inference.** GPU does the model math; CPUs stream and build the neighbourhoods; final embeddings for all 3B nodes are produced with a MapReduce pipeline that avoids recomputing shared neighbours.
- **Curriculum hard negatives.** Beyond random negatives, PinSAGE adds *hard* negatives (items somewhat related to the query but not engaged) and ramps their difficulty over epochs.

The payoff was real: A/B tests showed up to a **25% lift** in engagement. PinSAGE is *inductive* — it embeds new pins from features, unlike transductive LightGCN — which is also why it naturally handles cold-start (we return to this in 7.9).

---

### 7.4 Self-supervised & contrastive graph recommenders (the 2021–2023 wave)

LightGCN has two structural weaknesses: it relies entirely on sparse supervised signal (most users have few interactions), and deep propagation causes **over-smoothing** — node embeddings collapse toward each other as `k` grows, hurting the **long tail** (low-degree items get drowned out by their high-degree neighbours). The contrastive-learning wave attacks both.

**SGL (Wu et al. 2021, *Self-supervised Graph Learning for Recommendation*, SIGIR).** SGL adds a **self-supervised auxiliary task** on top of LightGCN: generate two augmented "views" of the graph via **node dropout, edge dropout, or random walk**, propagate each, and add an **InfoNCE contrastive loss** that pulls the two views of the *same* node together and pushes different nodes apart. Why it helps: the contrastive term enforces representation **uniformity**, it acts as automatic **hard-negative mining**, and — crucially — it disproportionately improves **long-tail items** and adds robustness to interaction noise. SGL-ED (edge dropout) was the best variant.

**SimGCL (Yu et al. 2022, *Are Graph Augmentations Necessary?*, SIGIR).** Yu et al. asked the LightGCN question of SGL: *is the expensive graph augmentation even necessary?* Their finding — **no.** What actually drives the gain is the contrastive loss shaping a more **uniform** embedding distribution, not the structural perturbation. So SimGCL drops graph augmentation entirely and instead adds **small random uniform noise directly to the embeddings** to create the two views:

```
e' = e + Δ,   Δ = sign(e) ⊙ (uniform noise),  ‖Δ‖ small
```

It is simpler, trains far faster (no rebuilding augmented adjacency matrices), gives smooth control over the uniformity knob, and beats SGL. XSimGCL (Yu et al. 2023) simplified further by reusing intermediate layers as the views.

**UltraGCN (Mao et al. 2021, CIKM) — the simplification taken to its limit.** Message passing is also the *slow* part. UltraGCN skips explicit propagation entirely: it derives the closed-form limit of *infinite-layer* LightGCN and approximates it with a **constraint loss** that directly pushes embeddings toward where infinite propagation would land, plus an explicit **item–item** term. Result: **>10× faster than LightGCN** with competitive or better accuracy. The throughline of 7.2–7.4 is one of the field's best lessons: **on pure CF, ruthless simplification keeps winning** — strip the transform (LightGCN), strip the augmentation (SimGCL), strip the propagation (UltraGCN).

#### Graph model → idea → when to reach for it

| Model | Core idea | Reach for it when |
|---|---|---|
| **NGCF** (2019) | GCN propagation *with* transforms + nonlinearity | Almost never now — keep as the baseline you beat |
| **LightGCN** (2020) | Strip transform/nonlinearity; layer-averaged readout | Default CF GNN; strong, simple, the one to start with |
| **PinSAGE** (2018) | GraphSAGE + random-walk neighbourhoods, sampled minibatches | Web-scale graphs (>100M nodes) and/or you need inductive/cold-start |
| **SGL** (2021) | Contrastive auxiliary loss via graph dropout views | Sparse data, long-tail matters, noisy interactions |
| **SimGCL** (2022) | Contrastive via embedding noise (no graph aug) | Same as SGL but you want speed + simplicity (preferred over SGL) |
| **UltraGCN** (2021) | Closed-form infinite-layer limit, no message passing | Huge graphs where LightGCN training is too slow |

---

### 7.5 Why two stages: retrieval vs ranking

Here is the constraint that shapes every large recommender. A request arrives; you have ~80ms to fill a slot from a catalogue of, say, 30M items. A good ranking model uses hundreds of features (recency, context, cross-features, sequence) and a deep network — call it 1ms per item if you are lucky. 30M items × 1ms = **8 hours**. Obviously impossible. So the system splits in two:

- **Retrieval (candidate generation):** a *cheap* model reduces 30M → ~hundreds in milliseconds. It is optimized for **recall** — never lose the good items — and for speed. It must use a scoring function so simple it can be precomputed and indexed: a **dot product**.
- **Ranking:** a *heavy*, feature-rich model re-scores those few hundred candidates with every signal available. It is optimized for **precision** at the top — get the *order* exactly right.

This is the single most useful mental model in applied recsys: **"retrieval is recall + speed; ranking is precision + features."** Modern stacks even insert a lightweight **pre-ranking** stage between them. (Tutorial Lesson 12 frames this same funnel; this section is the deep version.)

| | **Retrieval** | **Ranking** |
|---|---|---|
| Job | 30M → ~hundreds | hundreds → ordered top-k |
| Optimize for | **Recall**, latency | **Precision@k**, calibration |
| Scoring | dot product (precomputable) | deep net, full feature set |
| Model | two-tower / LightGCN + ANN | DeepFM / DCN-v2 / DLRM (Part 8 / FM family) |
| Features | mostly IDs + a few side features | hundreds: context, cross, sequence |
| Latency budget | single-digit ms (ANN lookup) | tens of ms over a few hundred items |
| Negatives | in-batch / sampled (millions implied) | logged impressions (true hard negatives) |
| Failure if absent | can't even *find* good items | shows the wrong order of good items |

---

### 7.6 Two-tower / DSSM retrieval (cross-ref `09_two_tower_retrieval`)

The retrieval workhorse is the **two-tower** model, descended from **DSSM** (Huang et al. 2013, *Learning Deep Structured Semantic Models*, CIKM — originally web search) and brought to recommendation at scale by the **YouTube two-tower** paper (Yi et al. 2019, *Sampled Softmax with Random Negatives... for Retrieval*, RecSys).

The architecture: a **user tower** maps the user (ID + features + recent history) to a vector `u`; an **item tower** maps an item (ID + features) to a vector `v`. Score is the dot product, on **L2-normalized** embeddings and divided by a **temperature** `τ`:

```
s(u, i) = (u · v) / τ      with ‖u‖ = ‖v‖ = 1   (a scaled cosine)
```

The towers never interact until that final dot product — and *that is the whole point*. Because the item tower depends only on the item, **every item embedding can be precomputed offline and indexed**; at serving time you run the user tower once and do a nearest-neighbour lookup. (Contrast with a "single-tower" cross-attention model that mixes user and item early: more accurate, but you'd have to recompute for every item — fine for ranking, fatal for retrieval.)

**In-batch negatives + sampled softmax — the trick that makes it scale.** You can't enumerate 30M negatives per step. So take a batch of `B` (user, positive-item) pairs and compute the full `B×B` score matrix. For row `i`, the *diagonal* entry is the true positive; the **other `B−1` items in the batch are negatives for free** — and because batches are drawn from logged data, those negatives are sampled in proportion to **item popularity**, which is realistic. The loss is a per-row cross-entropy (a sampled softmax):

```
L = −(1/B) Σ_i  log  softmax_i( s(u_i, ·) )[i]
```

(`09/src/train.py` implements exactly this `in_batch_softmax_loss` over the `B×B` matrix.)

💡 **logQ correction — don't skip it.** In-batch sampling is *biased*: popular items appear as negatives far more often than chance, so the model over-penalizes them and your retrieval skews unpopular. The fix (Yi et al. 2019, importing Bengio & Senécal's sampled-softmax correction) is to subtract each item's **log sampling probability** from its logit before the softmax:

```
s'(u, i) = s(u, i) − log Q(i)      where Q(i) ≈ item i's popularity share
```

This de-biases the gradient so frequency no longer determines who gets punished. It is *standard* at Google, Pinterest, ByteDance, and Kuaishou. (`09` exposes this as `logq_correction`; it logs `log(pop / pop.sum())`.) A 2025 paper — Hai et al., *Correcting the LogQ Correction* (RecSys 2025) — shows the textbook formula is subtly wrong when the same item is both positive and in-batch negative, and refines it; worth reading once you've internalized the basic version.

🔧 **Temperature `τ`** sharpens or softens the softmax. Too high → fuzzy, under-confident retrieval; too low → the loss fixates on the hardest negative and training gets unstable. Treat it as a *first-class* hyperparameter (typical range 0.05–0.2), not an afterthought.

LightGCN and two-tower are not rivals — they're **complementary**. You can use LightGCN (or PinSAGE) embeddings *inside* the towers, then serve them with ANN. The graph model decides *what a good embedding is*; two-tower + ANN decides *how to serve it in 5ms*.

---

### 7.7 ANN at serving: FAISS, ScaNN, HNSW

Once item embeddings are precomputed and the user tower yields `u`, retrieval becomes **maximum inner-product search (MIPS)**: find the top-k items by `u · v`. Exact search over 30M vectors is too slow, so you use **approximate nearest neighbour (ANN)**, trading a little **recall** for a lot of **speed and memory**. Three families dominate:

- **FAISS** (Johnson et al., Facebook) — the standard toolkit. Its **IVF-PQ** combines an *inverted file* (cluster the space, search only nearby clusters) with **product quantization** (compress each vector into a few bytes). Huge memory savings, GPU support — the default when the index must fit in RAM and budget is tight.
- **HNSW** (Malkov & Yashunin 2018) — *Hierarchical Navigable Small World* graphs. Builds a multi-layer proximity graph and greedily walks toward the query. Outstanding recall–latency in the **high-recall regime**, but memory-hungry (stores the graph) and slower to build/update. The go-to when recall matters more than RAM.
- **ScaNN** (Guo et al. 2020, *Accelerating Large-Scale Inference with Anisotropic Vector Quantization*, ICML) — Google's library. Its insight: standard quantization minimizes reconstruction error *uniformly*, but for MIPS you should preserve the component **parallel** to high-scoring directions. **Anisotropic** quantization weights that direction more, giving state-of-the-art accuracy in the high-recall regime — frequently the fastest at a given recall.

🔧 **The tradeoff you actually tune:** every ANN index exposes a knob (FAISS `nprobe`, HNSW `efSearch`) that buys **recall** with **latency**. Plot recall@k vs p99 latency and pick the point that meets your SLO. The honest failure mode is *silent*: ANN recall loss means good candidates never reach ranking, and you'll only see it as a quiet metric sag, never an error. **Always measure ANN recall against exact search offline.**

Operationally: item embeddings are recomputed in a **batch job** (hourly/daily), the index is rebuilt and atomically swapped behind a serving service; the user tower runs **online** per request.

---

### 7.8 The rest of the production stack

Retrieval and ranking are the load-bearing walls; a real system needs more plumbing:

- **Feature store** — the single source of truth for features, serving **online** (low-latency, per-request) and **offline** (for training), with the *same* transforms both ways. The cardinal sin is **train/serve skew**: a feature computed one way in training and another at serving silently destroys quality.
- **Embedding serving & staleness.** Item embeddings are precomputed; between rebuilds they go **stale**, and brand-new items have *no* embedding until the next batch. The freshness/cost tradeoff (rebuild cadence) is a real lever, and a streaming/nearline path for fresh items is the proper fix (Part 8).
- **Candidate mixing.** Production retrieval is rarely one source — you blend two-tower, a graph model, trending/popular, recently-viewed, and rule-based pools, then dedupe.
- **Re-ranking** for **diversity** and **business rules** runs *after* ranking: MMR-style diversification, per-creator/category caps, freshness boosts, ads/sponsored blending, hard policy filters. (This is where the beyond-accuracy metrics — coverage, novelty, ARP from Tutorial Lesson 12 — get enforced.)
- **Latency SLOs & cost.** End-to-end is usually a **sub-100ms** budget split across the funnel. Retrieval gets single-digit ms (mostly ANN); ranking gets the rest. Cost scales with QPS × model size × candidate count — which is *the* economic reason the funnel exists.

---

### 7.9 Hyperparameters that matter, and edge cases

**LightGCN — what to tune (and what not to):**
- **# layers `K`** (the biggest lever): 2–4. Beyond ~4, **over-smoothing** degrades quality; `K`=3 is a strong default. Layer-combination readout buys you some robustness here.
- **embedding dim**: 64 is standard; 128 if the catalogue is large. Diminishing returns and over-fitting risk past that.
- **L2 reg `λ`**: ~1e-4 on the *layer-0* embeddings; the most important regularizer since there are almost no other parameters.
- **What barely matters:** there is *nothing else* — no learned transforms, no per-layer dropout in propagation. That bareness is the feature.

**Two-tower — what to tune:**
- **temperature `τ`**: highest-leverage and most underrated (≈0.05–0.2; see 7.6).
- **embedding/output dim**: 64–256. Higher dim → better recall but larger ANN index and more memory.
- **batch size = #negatives**: with in-batch negatives, **a bigger batch *is* more negatives**, which directly improves retrieval — the strongest reason to push batch size as far as hardware allows (cross-batch negative caching extends this further).
- **tower depth**: 2–3 layers usually suffices; depth helps far less than getting `τ`, negatives, and features right.

**Edge cases — where systems quietly break:**
- **Cold-start.** Pure-ID LightGCN and ID-only two-tower **cannot embed a new user/item** — no interactions, no embedding. The fix is **feature towers**: condition the tower on side features (genre, text, image), so a brand-new item gets a reasonable embedding from content alone. `09` includes exactly this `use_features` ablation to demonstrate the lift; PinSAGE's inductive feature aggregation is the web-scale version.
- **Popularity bias in in-batch negatives.** As in 7.6 — fix with **logQ correction**; without it, retrieval skews toward unpopular items and recall suffers.
- **Embedding staleness.** Precomputed item vectors drift from the live model; rebuild cadence is a freshness/cost knob, and new items are invisible until the next batch (Part 8's streaming/nearline path).
- **Over-smoothing in deep GNNs.** Stacking layers collapses embeddings toward a common vector and crushes the long tail. Mitigations, in order: keep `K` small, use **layer-combination** readout, add **contrastive regularization** (SGL/SimGCL), or skip propagation entirely (UltraGCN).

---

### 7.10 What to read next → Part 8

You now have both halves: **graph models** that produce excellent CF embeddings, and the **retrieval→ranking funnel** that serves them under a latency budget. But we left three honest gaps wide open — the system is **stale** (batch-rebuilt embeddings, no fresh items), **cold** (new users/items barely handled), and **biased** (popularity and exposure bias baked into the logs we train on). Those are not footnotes; they are where production systems actually live and die. **Part 8 — Streaming, Cold-Start & Bias** takes each one seriously: nearline/streaming embedding updates and incremental graph learning; principled cold-start beyond feature towers (meta-learning, content/LLM priors, semantic IDs); and the debiasing toolkit — exposure/position/popularity bias, inverse-propensity weighting, and off-policy evaluation so you can trust an offline number before you spend an A/B test on it.

---

## Part 8 — Online Learning, Cold-Start, Bias & Fairness (the realities of production)

Everything in Parts 1–7 quietly assumed a fiction: a *frozen world* — train on a snapshot, evaluate on a held-out slice of it, report a number. Production recommenders never get that luxury. The data never stops, the catalogue churns hourly, tastes drift, brand-new users and items arrive with no history, and — the subtle one — **the model's own recommendations decide what data it trains on tomorrow**. This part covers the three forces that separate a model that "works on my machine" from one that ships, keeps working, makes money, and harms no one: *online learning* (keeping fresh on a moving stream), *cold-start* (serving entities you have never seen), and *bias / fairness* (the distortions baked into observational feedback and the loops that amplify them). These are three faces of one fact — **recommenders operate on data they themselves helped generate, in a world that refuses to hold still.**

We lean on what you built. Project 07 (`07_streaming_recsys`) is the live lab for the online half — the three regimes, FTRL, prequential eval, staleness, drift-triggered retraining. Project 04 (`04_movielens_1m_recsys`) gives a *measured* cold-start gap (`per_segment.csv`). Project 09 (`09_two_tower_retrieval`) gives the content-tower fix. I cross-reference all three so the theory has a body.

---

### 8.1 — The stream changes everything: three update regimes

A live recommender must answer one operational question before any modeling question: **how often, and how, do we fold new data into the model?** Project 07's `src/online.py` exists to compare the three canonical answers *fairly* — same model, same stream, same evaluator, only the update policy changes.

1. **Scheduled batch retrain** (`run_offline_retrain`). Every *N* windows, throw the model away and rebuild from scratch on a buffer of recent history. Between retrains the model is **frozen** — which is exactly why we can *measure staleness*: how much accuracy bleeds away as the world drifts past a static model. The "retrain every night" pattern.
2. **Warm-start incremental** (`run_warm_start`). Each window, *continue* from the previous weights with a few fine-tuning steps. Cheaper and fresher than a rebuild; the risk is **catastrophic forgetting** — fine-tune too hard and you forget long-standing taste, too softly and you never adapt. The middle ground most teams ship.
3. **Full online / per-event** (`run_full_online`). Update immediately on every micro-batch, the instant it is scored. Lowest staleness, freshest, noisiest — a single window can yank the weights around.

The decision is a three-way tension between **freshness, cost, and stability**. No regime dominates; you pick the point on the frontier your latency SLOs, compute budget, and drift rate dictate.

| Update regime | What it does | Freshness | Cost / step | Primary risk |
|---|---|---|---|---|
| **Scheduled retrain** | rebuild from scratch every *N* windows | stale between retrains (a **sawtooth**) | highest (full rebuild) | predictable but always lagging the world |
| **Warm-start incremental** | continue from previous weights, few fine-tune steps | fresher | medium | **catastrophic forgetting** / stability–plasticity |
| **Full online** | update per event / micro-batch | always fresh | cheapest per step | noisiest, hardest to debug, drift-sensitive |

> 💡 **Project 07's measured result, in one line.** Going scheduled → warm-start → full-online, **NDCG@10 rose 0.22 → 0.27 → 0.31 while staleness fell 0.078 → 0.037 → 0.000**, and the scheduled full-rebuild had the highest per-update cost. Freshness and accuracy moved *together*; both traded against rebuild cost. The scheduled regime shows a visible **staleness sawtooth** — a bump right after each retrain, then decay as the frozen model drifts. `staleness_from_rolling` in `evaluate.py` measures exactly that decay: metric just after a retrain minus metric just before the next one.

---

### 8.2 — Prequential evaluation: why static splits are a lie on a stream

On a stream there is no "test set," because the future has not happened yet. Worse, any static random split *leaks the future into the past*: a random train/test partition will happily let the model train on a user's later clicks to predict their earlier ones — an offline metric that is wildly optimistic and collapses in production. The honest protocol is **prequential evaluation (test-then-train)**, rooted in Dawid's *prequential principle* (Dawid 1984, "Statistical Theory: The Prequential Approach," *JRSS-A*) and formalized for streams by Gama, Sebastião & Rodrigues (2013, "On evaluating stream learning algorithms," *Machine Learning* 90(3)):

```
for each incoming window W in time order:
    1. PREDICT every event in W with the CURRENT model   ← the test (never-seen data)
    2. SCORE and accumulate the loss for W
    3. UPDATE the model on W                              ← then train
```

This is precisely `PrequentialEvaluator.evaluate_window` in Project 07: it scores the window *before* `model.partial_fit(window)` is ever called. Because every event is evaluated before it is learned from, the metric is leakage-free *by construction* and, crucially, **measures the deployed quantity** — how well the model predicts the very next event. Blum, Kalai & Langford (1999, "Beating the Hold-out," *COLT*) prove progressive validation's generalization bound is essentially as tight as a single hold-out, while using 100% of the stream for both testing and learning — no data permanently sacrificed.

You read the **curve**, not a single number. A model that is great on average but collapses during a trend shift is a bad production model. To stop the lifetime average from drowning out current performance, Gama recommends a **fading factor** instead of a plain cumulative mean:

$$P_e^{\alpha}(t) = \frac{\sum_{s=1}^{t}\alpha^{\,t-s}\,L(y_s,\hat y_s)}{\sum_{s=1}^{t}\alpha^{\,t-s}}, \qquad 0<\alpha\le 1$$

computed recursively in O(1) memory ($S_t = L_t + \alpha S_{t-1}$, $B_t = 1 + \alpha B_{t-1}$). With $\alpha\to 1$ you recover the cumulative average; with $\alpha=0.995$ you get a smoothed *recent* estimate that tracks drift.

---

### 8.3 — FTRL-Proximal: the production online-learning workhorse

The canonical online CTR learner is **FTRL-Proximal** (Follow-The-Regularized-Leader), from McMahan, Holt, Sculley et al. (2013, "Ad Click Prediction: a View from the Trenches," *KDD*) — one of the most-cited applied-ML papers ever, and the model behind Google's display-ads CTR system. Its theoretical legitimacy comes from McMahan (2011, "Follow-the-Regularized-Leader and Mirror Descent: Equivalence Theorems and L1 Regularization," *AISTATS*), which proves that keeping the cumulative L1 penalty in **closed form** (as FTRL does) yields strictly more sparsity than subgradient methods like FOBOS that only approximate the past penalty.

FTRL stores just **two accumulators per coordinate** — $z_i$ (gradient info with a proximal correction) and $n_i$ (sum of squared gradients) — and never stores the weights. Weights are reconstructed *lazily*, only for the features present in the current event:

$$
w_i =
\begin{cases}
0 & \text{if } |z_i| \le \lambda_1 \quad\text{(exact sparsity)}\\[2mm]
-\dfrac{z_i - \operatorname{sgn}(z_i)\,\lambda_1}{\big(\beta + \sqrt{n_i}\big)/\alpha + \lambda_2} & \text{otherwise}
\end{cases}
$$

After predicting $p = \sigma(\sum_i w_i x_i)$ and seeing label $y$, the per-coordinate update is:

$$g_i = (p - y)\,x_i,\qquad \sigma_i = \frac{\sqrt{n_i + g_i^2} - \sqrt{n_i}}{\alpha},\qquad z_i \mathrel{+}= g_i - \sigma_i w_i,\qquad n_i \mathrel{+}= g_i^2$$

This is *exactly* the math in Project 07's `FTRLClassifier` (`models.py`): the soft-threshold `if |z_i| <= L1: w_i = 0`, the denominator `(beta + sqrt(n_i))/alpha + L2`, and the `z += g - sigma*w; n += g*g` update. Three properties make it the workhorse:

- **L1 gives *true zeros*.** Web-scale CTR has billions of sparse one-hot features (every user×item×context cross). The $|z_i|\le\lambda_1$ threshold drives most weights to *exactly* zero, so they need not be stored at all — the model that fits in RAM wins. Plain online SGD + L1 cannot do this; its noisy subgradient steps rarely land on zero. McMahan et al. report large memory savings at equal accuracy.
- **Per-coordinate adaptive learning rates.** The effective rate is $\eta_{t,i} = \alpha/(\beta + \sqrt{n_i})$ — each feature gets its *own* rate (this is AdaGrad's idea, for free). Frequent features get a small, calm step (they have been seen enough); rare features stay plastic and learn fast on the few examples they appear in. Essential when feature frequencies span orders of magnitude. ($\beta = 1$ is "usually fine"; tune $\alpha$.)
- **Lazy, hashed, single-pass.** O(#active-features) per event, no table to resize when a new id appears (the hashing trick absorbs it). This is the *full-online* regime done right.

> 🔧 **Why FTRL and not a deep net for the streaming layer?** Convex, single-pass, sparse, and debuggable — exactly what you want when the model updates per-event and a bad weight can hit live traffic in seconds. Deep retrieval/ranking models live in the *warm-start* or *scheduled* regimes; FTRL-style linear/FM models dominate the genuinely online, second-wise CTR layer.

---

### 8.4 — Drift, drift detection, and drift-triggered retraining

**Concept drift** is the formal reason a frozen model decays. Following Gama et al. (2014, "A survey on concept drift adaptation," *ACM Computing Surveys* 46(4)), drift between times $t_0$ and $t_1$ is any change in the joint distribution, $p_{t_0}(X,y) \ne p_{t_1}(X,y)$, decomposed via $p(X,y)=p(X)\,p(y\mid X)$ into:

- **Real drift:** $p(y\mid X)$ changes — the decision boundary itself moves. *Always* requires adaptation.
- **Virtual / covariate drift:** $p(X)$ changes but $p(y\mid X)$ does not — the input distribution shifts without moving the boundary.

By *dynamics*, drift is **sudden** (instant switch — a policy change), **incremental** (slow monotone morph), **gradual** (old and new concepts alternate during a transition), or **recurring** (seasonality — weekday/weekend, holiday). Recurring drift is why old models can become valuable again.

How do you *detect* it? Two families, for two operational niches:

- **Label-free distribution monitors — PSI (Population Stability Index).** Bin a feature or the model's score into $B$ buckets and compare a baseline ("expected") to current ("actual"):
$$\text{PSI} = \sum_{j=1}^{B}\big(A_j - E_j\big)\,\ln\frac{A_j}{E_j}$$
Industry thresholds: **PSI < 0.1** no meaningful shift; **0.1 ≤ PSI < 0.25** moderate (investigate); **PSI ≥ 0.25** significant (retrain/recalibrate). PSI's superpower is needing *no labels* — you compute it on live traffic *immediately*, long before delayed conversion labels arrive, catching covariate drift early.
- **Label-based error detectors — DDM and ADWIN.** DDM (Gama et al. 2004) tracks the online error rate $p_i$ with std $s_i=\sqrt{p_i(1-p_i)/i}$, stores the running minimum $p_{\min}+s_{\min}$, and fires a **warning** at $p_i+s_i \ge p_{\min}+2s_{\min}$ and **drift** at $p_i+s_i \ge p_{\min}+3s_{\min}$ — a control chart on the premise that for a stable concept, error should not significantly *rise*. ADWIN (Bifet & Gavaldà 2007) keeps a variable-length window and drops the old half whenever two sub-windows' means differ by more than a Hoeffding bound, with *provable* false-positive guarantees and an auto-adapting window length.

> 🔧 **Drift-triggered retraining.** Instead of retraining on a fixed clock, retrain *when statistics say the concept moved* — spend compute only when the model is actually decaying. Project 07's `run_offline_retrain(drift_metric=..., drift_drop=...)` does exactly this: it retrains when the rolling metric drops by more than a threshold below its recent best, with a minimum 2-window gap to prevent thrashing. A fixed schedule is simple and predictable; drift-triggered is efficient but needs a reliable detector and a thrash guard. Most mature teams run *both* — a slow scheduled floor plus a fast drift trigger.

---

### 8.5 — Catastrophic forgetting: the stability–plasticity dilemma

Warm-start and full-online regimes have a failure mode batch training never sees: **catastrophic forgetting** (McCloskey & Cohen 1989, "Catastrophic Interference in Connectionist Networks"; French 1999, "Catastrophic forgetting in connectionist networks," *TiCS*). Sequential gradient updates that fit the new distribution **overwrite** the weights encoding old knowledge. French diagnoses the root cause as *overlapping distributed representations* — the weight-sharing that grants generalization also causes interference — and names the tension the **stability–plasticity dilemma**: plastic enough to learn the new, stable enough to keep the old.

In a recommender this is concrete and quiet. Fine-tune too aggressively on today's hot items and you forget long-tail items, dormant users, and rare patterns absent from the recent window — headline metrics on hot traffic look fine while cold/long-tail quality silently erodes. Because embeddings are densely shared, updating one cohort perturbs unrelated cohorts.

The canonical regularization fix is **Elastic Weight Consolidation** (Kirkpatrick et al. 2017, "Overcoming catastrophic forgetting in neural networks," *PNAS*), which anchors parameters important to old tasks with a quadratic penalty:

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{new}}(\theta) + \sum_i \frac{\lambda}{2}\,F_i\,\big(\theta_i - \theta^*_{\text{old},i}\big)^2$$

where $F_i$ is the diagonal of the Fisher information (how much weight $i$ mattered to the old task) — a spring of stiffness $F_i$ pulling each weight back toward its old value. Important weights stay put; unimportant ones stay free. In practice, the dominant industrial defense is simpler: **replay / rehearsal** — periodic retraining over a *long history buffer* (exactly the `buffer_windows` in Project 07's scheduled regime) rather than pure incremental SGD on the live stream. This is why the three-regime comparison is fundamentally a **stability (scheduled retrain) vs plasticity (full online)** axis.

---

### 8.6 — Cold-start: serving entities you have never seen

The second reality. **Cold-start** is inference for entities with little or no interaction history, in three flavors (Schein, Popescul, Ungar & Pennock 2002, "Methods and Metrics for Cold-Start Recommendations," *SIGIR*):

- **New-user:** a user with no history; nothing to personalize on.
- **New-item:** an item nobody has interacted with; it cannot be retrieved or scored from collaborative signal.
- **New-system (systemic):** a freshly launched platform with a near-empty interaction matrix — the bootstrapping problem for the whole product.

**Why plain matrix factorization fails on cold ids.** MF learns a latent vector *per id* by fitting observed interactions. An id's embedding is meaningful *only* because gradient descent moved it using that id's interaction rows. For an unseen id there are *no rows*, so the embedding sits at its random initialization (or is undefined) and the dot product $p_u \cdot q_i$ is noise. MF is **transductive over ids** — it cannot generalize beyond the ids in its training graph. This is not a tuning problem; it is structural.

> 💡 **The cold-start gap is measurable in your own repo.** Project 04's `per_segment.csv` slices NDCG@10 by user activity. **BPR** scores 0.228 on *cold* users vs 0.308 on *warm* — a ~26% relative drop. **LightGCN** is worse: **0.151 cold vs 0.291 warm**, nearly halving — the richer the collaborative model, the harder it falls when there is no graph neighbourhood to propagate through. The fix in every case below is the same: either *substitute content that exists for cold entities*, or *explore to gather the missing signal*.

**Content / feature-based bootstrapping & hybrids.** Gantner et al. (2010, "Learning Attribute-to-Feature Mappings for Cold-Start Recommendations," *ICDM*) train BPR-MF on warm entities, then learn an explicit map from observable **attributes** (genre, age, occupation) into the latent space; a cold entity's attributes are pushed through the map to *synthesize* a latent vector. **DropoutNet** (Volkovs, Yu & Poutanen 2017, *NeurIPS*) takes a model with both a content input and a collaborative-preference input, and applies **input dropout to the preference vector** during training — randomly zeroing it to *simulate the cold-start condition*, forcing the network to reconstruct a usable embedding from content alone. The elegance: one model, robust to missing inputs, serves warm, user-cold, and item-cold gracefully.

**Two-tower content towers — the cleanest item cold-start fix (cross-ref Part 9 / Project 09).** A **user tower** and an **item tower** each map to a shared embedding space; relevance is the dot product of L2-normalised tower outputs (Yi et al. 2019, "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations," *RecSys*; built on Covington et al. 2016, "Deep Neural Networks for YouTube Recommendations"). Because the **item tower maps content features → embedding**, a brand-new item gets a *usable embedding on day one* from a single forward pass — no history needed — and lands in the same ANN index as everything else, immediately retrievable. Project 09's `TwoTowerModel` makes this the explicit ablation: with `use_features=False` it degenerates to pure id embeddings (i.e., MF, which *cannot* embed a cold item); with features on, the content tower carries cold items. Yi et al.'s contribution — a streaming **sampling-bias correction** ($\text{logit} - \log p_{\text{item}}$) — also matters here, since in-batch negatives are biased toward popular items.

**Meta-learning — MeLU.** Lee et al. (2019, "MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation," *KDD*) treat **each user as a task** and apply MAML (Finn, Abbeel & Levine 2017): learn a global *initialization* such that a *few gradient steps* on a new user's handful of interactions yield a strong personalized model. Cold-start is literally few-shot learning; MAML primes the model to adapt fast rather than needing thousands of examples from scratch.

> 🔧 **2024–2026 SOTA — LLMs and semantic ids.** The frontier attacks cold-start via the one thing every new item/user *does* have: text. **TIGER** (Rajput et al. 2023, "Recommender Systems with Generative Retrieval," *NeurIPS*) quantizes content embeddings into **Semantic IDs** (via RQ-VAE) so semantically similar items *share id prefixes* — a brand-new item inherits generalization from similar warm items, unlike an isolated random hash id; the paper reports improved cold-start generalization. Recent surveys (e.g., arXiv:2412.13432, 2024) frame LLMs synthesizing cold user/item **profiles or embeddings** from metadata, or even **simulating synthetic interactions** to warm up a CF model — all attacking the same root cause Schein named in 2002.

---

### 8.7 — Exploration: ε-greedy, Thompson sampling, contextual bandits

Pure exploitation — always serving the current best-estimate item — never collects data on under-served new items, so their estimates stay bad forever: a self-reinforcing trap that *is* cold-start. **Exploration** deliberately tries uncertain items to gather the missing signal, and bandits formalize the explore/exploit trade-off so its cost is bounded (low regret).

- **ε-greedy.** With probability ε pick a random item, else the greedy best. Simple but undirected — it wastes exploration on clearly-bad arms.
- **Thompson sampling.** Maintain a Bayesian posterior over each arm's reward; each round, *sample* parameters from the posterior and act greedily on the sample. High-uncertainty arms get optimistically sampled often enough to be tried; as data accrues the posterior sharpens and exploration *auto-decays*. Empirically competitive with or better than UCB (Chapelle & Li 2011, *NeurIPS*).
- **LinUCB (contextual).** Li, Chu, Langford & Schapire (2010, "A Contextual-Bandit Approach to Personalized News Article Recommendation," *WWW*) assume reward is linear in context features, $\mathbb{E}[r_{t,a}\mid x_{t,a}] = x_{t,a}^\top\theta_a$, keep per-arm online ridge regression ($A_a = D_a^\top D_a + I$, $b_a = D_a^\top r_a$, $\hat\theta_a = A_a^{-1}b_a$), and pick the arm by **optimism under uncertainty**:
$$a_t = \arg\max_a\Big[\underbrace{x_{t,a}^\top\hat\theta_a}_{\text{estimated reward}} + \alpha\underbrace{\sqrt{x_{t,a}^\top A_a^{-1} x_{t,a}}}_{\text{confidence width}}\Big]$$
A brand-new item starts with $A_a = I$ → a *wide* confidence bound → it is *optimistically shown* (explored); as impressions accumulate the bound shrinks and the system exploits it only if it truly performs. Context features let signal transfer across items so a cold arm is not learned fully from scratch. This is cold-start and exploration as *one* mechanism — and it is the natural production answer to "how do we surface new content without tanking near-term metrics."

---

### 8.8 — Bias and debiasing: the data is not a random sample

The third reality, and the deepest. Observational feedback is **not** a random sample of preferences. Following the authoritative survey (Chen, Dong, Wang, Feng, Wang & He 2023, "Bias and Debias in Recommender System: A Survey and Future Directions," *ACM TOIS*), the load-bearing biases are:

- **Selection bias / MNAR (Missing Not At Random):** users rate the items they *choose* to rate, skewing toward strong opinions; observed ratings are not representative.
- **Exposure bias:** in implicit feedback, a non-interaction conflates "not interested" with "never shown."
- **Position bias:** users click higher-ranked items regardless of relevance.
- **Popularity bias:** popular items are recommended *even more* than their popularity warrants — the long-tail "rich get richer" (Matthew) effect.

| Bias type | Cause | Fix |
|---|---|---|
| **Selection / MNAR** | users choose what to rate; missingness depends on preference | IPS-weighted learning; joint MNAR models |
| **Exposure** | non-interaction ≠ negative; user never saw the item | exposure models; principled negative sampling |
| **Position** | higher rank → more clicks irrespective of relevance | propensity-weighted LTR (IPS-ULTR); click models |
| **Popularity** | long-tail + feedback loop amplify the head | popularity regularization; re-ranking; IPS |
| **Unfairness** | disparate exposure/quality across groups | exposure-fairness constraints; calibration |

**Inverse Propensity Scoring (IPS).** The central debiasing tool, from Schnabel, Swaminathan, Singh, Chandak & Joachims (2016, "Recommendations as Treatments: Debiasing Learning and Evaluation," *ICML*), treats a recommendation as a *treatment* and the rating as the *outcome* (causal inference from observational data). Weight each observed entry by the inverse of its **propensity** $P_{u,i} = P(O_{u,i}=1)$:

$$\hat R_{\text{IPS}} = \frac{1}{|U||I|}\sum_{(u,i):\,O_{u,i}=1}\frac{\delta_{u,i}(Y,\hat Y)}{P_{u,i}}$$

(note the normalizer is the *full* $|U||I|$, not the observed count.) This is **unbiased** for the full-information risk: each term appears with probability $P_{u,i}$ but is weighted by $1/P_{u,i}$, so the two cancel and $\mathbb{E}[\hat R_{\text{IPS}}] = R$ (given every $P_{u,i}>0$). The catch is **variance** — small propensities create enormous weights. Fixes: **clipping** ($\max(P_{u,i}, \tau)$); **self-normalized IPS / SNIPS** (Swaminathan & Joachims 2015, *NeurIPS*), which divides by $\sum 1/P_{u,i}$ instead of $N$; and **doubly robust** estimators (Wang, Zhang, Sun & Qi 2019, *ICML*) that add an imputation model and are unbiased if *either* the propensities *or* the imputed errors are correct — "two chances to be right." For ranking, Joachims, Swaminathan & Schnabel (2017, "Unbiased Learning-to-Rank with Biased Feedback," *WSDM*, best paper) weight each *clicked* result by inverse examination propensity $1/q_k$ (estimated via swap experiments), making click-trained LTR unbiased for relevance.

> 🔧 **This is why offline ≠ online.** A naive offline metric computed only on observed clicks is *biased by exactly the policy that generated the logs*. IPS/SNIPS is how you do honest **off-policy estimation** — predicting a new policy's performance from old logs — before risking an A/B test. It is the rigorous version of the "offline ≠ online" warning in the closing lens of every Project notebook.

---

### 8.9 — Feedback loops, echo chambers, and fairness

**The model biases its own future training data.** This is the most insidious effect: the recommender's outputs determine what users see, which determines what they click, which becomes tomorrow's training data — a closed loop that amplifies whatever bias it started with. Chaney, Stewart & Engelhardt (2018, "How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility," *RecSys*) show this *algorithmic confounding* homogenizes user behavior and *decreases* utility. Mansoury et al. (2020, "Feedback Loop and Bias Amplification in Recommender Systems," *CIKM*) demonstrate empirically that the loop amplifies popularity bias and shrinks coverage over time. Jiang et al. (2019, "Degenerate Feedback Loops in Recommender Systems," *AIES*) analyze echo chambers as *degenerate feedback loops* and show that injecting randomization (exploration — §8.7) slows the degeneration. The practical lesson: **exploration is not only a cold-start tool; it is the antidote to your own feedback loop.**

**Fairness is multi-stakeholder** (Burke 2017, "Multisided Fairness for Recommendation"). Distinguish **consumer-side (C-fairness)** — equitable recommendation quality across user groups — from **provider-side (P-fairness)** — equitable *exposure* across item producers. The headline result is **fairness of exposure** (Singh & Joachims 2018, "Fairness of Exposure in Rankings," *KDD*): because attention decays steeply with rank but relevance often does not, a *tiny* relevance gap produces a *winner-take-all* exposure gap. They enforce exposure *proportional to relevance* via probabilistic rankings solved by a linear program. **Calibration fairness** (Steck 2018, "Calibrated Recommendations," *RecSys*) is the intra-user version: a user who watches 70% action / 30% romance should see roughly that mix, not 100% action because action is the dominant signal; the miscalibration is measured by KL divergence between history and recommendation genre distributions and fixed by a re-ranking step. Standard accuracy-maximizing recommenders *amplify the dominant interest* and crowd out minority tastes — a quiet unfairness.

All of this sits inside the **Responsible AI** frame (Microsoft RAI Standard v2, 2022): six principles — **fairness, reliability & safety, privacy & security, inclusiveness, transparency, accountability** — where transparency and accountability are *foundational* (they enable the rest). For a recommender this maps to fairness across segments, drift monitoring (reliability), privacy of interaction data, inclusive design, explainable recommendations, and clear ownership of harms (filter bubbles, addiction). The 2024–2026 frontier extends this to LLM recommenders, which inherit *new* popularity bias from web-scale pretraining (Zhang et al. 2023, "Is ChatGPT Fair for Recommendation?", FaiRLLM benchmark, *RecSys*) — over-recommending whatever was frequently mentioned online, independent of user data.

---

### 8.10 — Edge cases & monitoring the senior DS owns

The realities that break models *silently*, and what to watch:

- **Training–serving skew.** Features computed differently offline (training) vs online (serving) — a different default, a stale join, a units mismatch — silently destroys a good model. Log serving-time features and assert they match the training pipeline; this is the single most common production-recsys bug.
- **Delayed feedback.** Conversions arrive *late* — a click now, a purchase tomorrow. Train too early and you mislabel slow-converters as negatives. Delayed-feedback models (and a labeling delay window) handle this; never treat "no conversion yet" as "no conversion."
- **Seasonality & recurring drift.** Weekday/weekend, holidays, paydays. A drift detector must not panic over *expected* seasonality (it is recurring drift, not a regression) — calibrate baselines per-period.
- **Data outages.** An upstream feature pipeline dies and ships zeros/nulls; the model serves garbage while every accuracy metric (which needs labels) still looks fine for hours. Label-free monitors — **PSI on input features and on the score distribution** (§8.4) — are your early-warning system precisely because they fire *before* delayed labels can confirm the damage.

> 💡 **The closing discipline.** Building the model is the *start*. Shipping means: connecting model metric → product metric → business metric with a guardrail (latency, revenue, *and* fairness); doing off-policy estimation (§8.8) before the A/B test; watching for drift, skew, and feedback loops; and slicing every metric by segment so a cold-start or fairness regression cannot hide inside a healthy average. That lens — surfaced in every Project notebook's production section and on the `gnnlab` MLOps dashboard — is the difference between "works on my machine" and "ships, makes money, and doesn't harm anyone."

---

### What to read next → Part 9 (the frontier)

We have now closed the loop on *production realities*. **Part 9 takes us to the frontier**: where the field is heading in 2025–26 — generative retrieval and **semantic IDs** (TIGER and successors), **LLM-as-recommender** and conversational recommendation, foundation models for recommendation, graph-based and multimodal retrieval, and the unification of retrieval + ranking + generation into single sequence models. Everything in Part 8 — online learning, cold-start, IPS-debiasing, exposure fairness — does not disappear at the frontier; it becomes *harder* and *more important*, because the larger and more autonomous the model, the more its biases compound and the more its feedback loops bite. Bring this part with you.

---

## Part 9 — The Frontier: Generative & LLM-based Recommendation (2024-2026)

In Part 6 you trained SASRec to attend over a user's history and score the next item against a fixed embedding table. Hold that picture, because the frontier is one conceptual step away from it. SASRec *retrieves* — it produces a query vector and looks items up. The 2024-2026 wave reframes the same next-item objective as **generation**: instead of scoring a table, the model *generates the identifier of the next item, token by token*, exactly the way a language model generates the next word. That single reframing is what people are (loudly) calling RecSys's "ChatGPT moment." This part is your honest map of that territory — what is genuinely new, what is deployed at scale, and where a well-tuned SASRec or two-tower still quietly wins. Keep the Part 6 lesson in your pocket the whole way through: accuracy is necessary but not sufficient, and an offline number is a hypothesis, not a result.

### 9.1 Generative retrieval with Semantic IDs

The atomic item ID is the original sin of classical recommenders. Every item gets a random integer, that integer indexes a row in an embedding table, and the table grows linearly with the catalogue. A brand-new item has a randomly-initialised row that means nothing until it accrues interactions — that is the cold-start problem, baked into the data structure itself. Two items that are near-identical (two episodes of the same show) get unrelated rows. Generative retrieval attacks this at the representation layer.

The recipe, from **TIGER** (Rajput et al., 2023, *Recommender Systems with Generative Retrieval*, NeurIPS), has two stages:

1. **Tokenize items into Semantic IDs.** Take a content embedding of each item (text/image, e.g. from a pretrained encoder) and pass it through an **RQ-VAE** — a residual-quantized variational autoencoder. RQ-VAE quantizes the embedding into a short tuple of discrete codes drawn from a small learned codebook: a coarse code, then a code for the residual, then a code for the residual-of-the-residual, hierarchically. An item becomes something like `(12, 7, 39)` — a *Semantic ID*. Similar items share prefixes; the codebook is tiny and reused across the whole catalogue, so the parameter count no longer scales with the number of items.
2. **Generate the next item autoregressively.** Train a sequence-to-sequence Transformer to take the Semantic IDs of a user's history and *generate the Semantic ID of the next item*, one code at a time, with beam search at inference to produce a ranked list of candidate IDs.

Why this is a genuinely different idea, not a re-skin of SASRec: retrieval becomes **generation over a shared vocabulary**, so the model can in principle produce an ID for an item it has barely seen, because that item's codes are determined by its *content*, not its interaction count. Cold-start items inherit meaningful representations for free; the catalogue can grow without growing the model; and the model exhibits LLM-like behaviour where quality improves with scale.

The successors all attack TIGER's central weakness — that Semantic IDs built purely from content ignore **collaborative signal** (what users actually co-consume):

- **LETTER** (Wang et al., 2024, *Learnable Item Tokenization for Generative Recommendation*, CIKM) aligns the RQ-VAE's quantized codes with collaborative embeddings, so the codebook captures both semantics *and* co-occurrence, and adds diversity regularization to fight code collapse.
- **EAGER** (Wang et al., 2024, *EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration*, KDD) runs two streams — one behavioural, one semantic — and fuses them, rather than forcing one tokenizer to carry both.
- **STAR / LC-Rec and the broader family** push toward end-to-end learnable tokenization, longer IDs generated in parallel, and tokenizers built for recommendation rather than borrowed from LLMs. The 2025 *Practitioner's Handbook* (arXiv 2507.22224) and several survey/reproducibility papers are now consolidating the design space.

> 💡 **Hype vs proven.** *Proven:* on standard academic benchmarks (Amazon review categories, ML), TIGER-style models post strong NDCG/Recall and real cold-start gains, and the approach unifies the retrieval and ranking stages conceptually. *Hype to resist:* (1) gains are reported on small public datasets with **leave-last-out + sampled negatives** — the exact protocol Part 6 warned can flatter a model; (2) a 2026 reproducibility study (*Cold-Starts in Generative Recommendation*, arXiv 2603.29845) finds cold-start is rarely isolated as a primary setting and that papers change tokenizer, scale, and training recipe *together*, so factor-wise attribution is murky. **Beam search over a discrete code tree is also not free** — it is not obviously cheaper than ANN over a two-tower index at serving time. Treat the cold-start and scaling claims as promising and directionally real, not settled.

### 9.2 LLMs as recommenders

A parallel line asks: can a large language model just *be* the recommender? The foundational paper is **P5** (Geng et al., 2022, *Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm*, RecSys). P5's bet is that every recommendation task — rating prediction, sequential next-item, explanation, review summarization — can be cast as **text-to-text**. User histories, item metadata, and reviews all become natural-language prompts, and a single T5-style model is trained to emit the answer as text. One model, many tasks, no task-specific heads.

Since then the space has split along a clean axis — **prompting vs fine-tuning**:

- **Zero-/few-shot prompting.** Hand a frozen LLM a user's history as text and ask it to rank candidates or pick the next item. Hou et al. (2024, *Large Language Models are Zero-Shot Rankers for Recommender Systems*, ECIR) is the reference result, and it is appropriately humbling: LLMs have real zero-shot ranking ability *but* struggle to perceive the **order** of a history, are biased by item **popularity** and by **position in the prompt**, and cannot feasibly score thousands of candidates (the prompt does not fit, and generation is slow). The honest read: useful as a *re-ranker over a short candidate list a classical retriever already produced*, not as a retriever over millions.
- **Fine-tuning / domain adaptation.** Fine-tune the LLM on interaction data (sometimes teaching it an "item-ID dialect," sometimes via RL — Rec-R1, R1-Ranker, and 2025-26 reasoning-enhanced variants). This recovers the collaborative signal that frozen LLMs lack, at real training cost.

Where LLMs earn their keep *today*, concretely:

- **Cold-start.** A model that has read the open web knows that a new sci-fi thriller resembles other sci-fi thrillers before a single user touches it. The 2025 survey *Cold-Start Recommendation towards the Era of LLMs* (arXiv 2501.01945) maps this well — it is arguably the single most defensible LLM-rec use case.
- **Explanations.** "Recommended because you watched X and liked the slow-burn pacing" is natural-language generation, which is exactly what LLMs are good at. Low-risk, high-perceived-value.
- **Conversational / interactive recommendation.** Multi-turn "find me something like X but lighter" is a real product surface LLMs unlock that classical systems cannot touch.

> 💡 **The latency/cost reality.** A two-tower retrieval scores millions of items in single-digit milliseconds with one matrix multiply and an ANN lookup. An LLM forward pass is *orders of magnitude* slower and pricier per request, and you cannot fit a million candidates in a context window. This is not a tuning detail — it is the reason no large platform serves user-facing feed ranking from an LLM in the hot path. LLMs in production rec live **offline or near-line** (precomputing features, embeddings, explanations) or as a **final re-ranker over tens of candidates**, almost never as the primary retriever.

### 9.3 LLM-augmented pipelines (the pragmatic middle)

The least glamorous and most *deployed* use of LLMs in rec is not to replace your model but to **feed it better features**. You keep your battle-tested two-tower retrieval and gradient-boosted/DLRM ranker, and you inject LLM-derived signal into them:

- **Semantic embeddings as features.** Encode item text/metadata and user profiles with an LLM (or an LLM-finetuned embedding model like **LLM2Rec**, Zhang et al., 2025) and concatenate those vectors into the item tower or ranking features. The classical model still does the fast scoring; the LLM contributes open-world semantics, computed offline.
- **Knowledge/reasoning augmentation.** **KAR** (Xi et al., 2024, *Towards Open-World Recommendation with Knowledge Augmentation from LLMs*) prompts an LLM to produce reasoning about user preferences and factual knowledge about items, then distills that into augmentation vectors for any backbone. **RLMRec** (Ren et al., 2024, *Representation Learning with LLMs for Recommendation*) aligns ID-based embeddings with LLM-generated profile text via contrastive learning — model-agnostic, preserving the efficiency of the existing recommender.
- **Data augmentation.** Use LLMs to synthesize side information, denoise interactions, or generate descriptions for sparse items.

This is where most teams should actually start. It is low-risk, it keeps serving cheap (the LLM runs in batch, not in the request path), and it is *additive* to the system you already trust.

### 9.4 Industrial signals — what big labs deploy

The most important industrial result is **HSTU** (Zhai et al., 2024, *Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations*, ICML), Meta's generative recommender. It is worth internalising precisely because it is *not* an LLM-as-recommender story — it is the **sequential-Transformer-at-scale** story, the direct industrial descendant of SASRec:

- It reformulates ranking *and* retrieval as **sequential transduction** over a user's action stream — "actions, not words." The unit of modelling is behaviour, not text.
- The **HSTU** architecture (Hierarchical Sequential Transduction Units) is engineered for high-cardinality, non-stationary, streaming data, and is reported **5.3x-15.2x faster** than FlashAttention2 Transformers on long (8192-token) sequences.
- The headline that made the field sit up: **model quality scales as a power law of training compute across three orders of magnitude, up to GPT-3/LLaMA-2 scale** — the first convincing demonstration that recommenders can ride a scaling law the way LLMs do. Deployed at **1.5T parameters** across surfaces serving billions of users, with **+12.4%** in online A/B metrics.

**Netflix** (2025 tech blog, *Towards Generalizable and Efficient Large-Scale Generative Recommenders*) tells a complementary, more sober story: a foundation model over user-behaviour sequences scaled 50M→1B parameters, with multimodal "semantic item towers" for cold-start and sampled-softmax decoding for efficiency. Crucially, they are candid about the costs — 80 A100s for 240 hours per training cycle, frequent retraining (unlike a train-once LLM), and a *latency misalignment* where cached recommendations drift from the live model.

> 💡 **The signal under the noise.** What big labs are actually deploying is **generative *sequential* recommenders that scale** (HSTU, Netflix's foundation model), not LLMs prompting their way through your feed. The proven industrial idea is "sequence modelling + scaling laws on behavioural data." Semantic IDs are entering production ranking features (Google/YouTube have published on Semantic IDs for *generalization in ranking*, not just retrieval). LLM-in-the-loop is real but lives at the edges — features, cold-start, explanations, conversational surfaces.

### 9.5 A grounded verdict

| Approach | Core idea | Maturity |
|---|---|---|
| Two-tower retrieval + GBDT/DLRM ranker | Cheap dot-product recall, feature-rich rerank | **Deployed** (the default everywhere) |
| SASRec / sequential Transformers | Causal self-attention over history | **Deployed** (strong, cheap baseline) |
| HSTU / generative sequential at scale | Behaviour-as-sequence + scaling laws | **Deployed** (Meta, billions of users) |
| Semantic IDs as ranking features | RQ-VAE codes for generalization/cold-start | **Emerging→deployed** (Google/YouTube) |
| LLM-augmented features (KAR, RLMRec, LLM2Rec) | LLM semantics into classical models | **Emerging** (near-line, additive) |
| TIGER / LETTER / EAGER generative retrieval | Autoregressively *generate* item IDs | **Research→emerging** (benchmark-strong) |
| LLM-as-reranker (zero/few-shot) | Prompt LLM over short candidate lists | **Emerging** (cold-start, conversational) |
| LLM-as-retriever (frozen, over full catalogue) | Generate items from a prompt directly | **Research** (latency/scale-bound) |

Now tie it back to the lesson that anchors this whole course — **Dacrema et al. (2019), *Are We Really Making Much Progress?*** (RecSys). They showed that most "neural" wins of that era evaporated against *properly tuned* nearest-neighbour and linear baselines: only one of seven reproducible methods beat the simple baselines convincingly. That paper is not a historical curiosity — it is the lens you must hold up to every generative/LLM-rec claim. The 2026 generative-rec reproducibility study echoes it almost verbatim: gains are entangled across design choices, evaluated on sampled-negative protocols, and rarely compared against a *well-tuned* SASRec. So the verdict:

- **Where generative/LLM rec genuinely helps today:** cold-start and new-content understanding; explanations and conversational surfaces; *scaling-law* gains on behavioural sequences at industrial compute (HSTU); LLM-derived features feeding classical models. These are real and, in places, deployed.
- **Where well-tuned classical/sequential/two-tower still wins:** warm-start, high-QPS, latency-bound retrieval and ranking on a stable catalogue. A carefully tuned SASRec + two-tower will match or beat a generative model on most public benchmarks *and* serve 100x cheaper. If you cannot beat that, you do not have a result — you have a press release.

### 9.6 What to learn next / how the field is moving

For the lead-DS path, three durable bets. **First, sequence modelling is the substrate** — everything converges on "model the user's action stream with a Transformer." Master SASRec end-to-end (you have) and you can read HSTU and TIGER. **Second, learn the tokenization/Semantic-ID stack** (RQ-VAE, codebooks, collaborative alignment) — it is showing up in *ranking* features even where full generative retrieval has not landed, so the ROI is real now. **Third, internalise scaling laws for recommenders** — the field's center of gravity in 2025-26 is "do recommenders scale like LLMs, and at what cost?" That is an *infrastructure and economics* question as much as a modelling one, which is exactly the altitude a lead operates at.

The throughline from Part 6 to here is unbroken: order is signal, sequences are the bridge, generation is the next reframing — and a strong baseline, honestly evaluated, is the judge of all of it. With that judgment in hand, the final part turns from *what to build* to *who you become* — the career roadmap from senior to lead DS, where the skill that compounds is not knowing the newest model, but knowing which frontier claim to bet the roadmap on and which to let mature in someone else's A/B test first.

---

## Part 10 — The Hyperparameter Tuning Playbook

Most of the gap between a junior and a senior result on the *same model* is tuning
discipline, not cleverness. This Part is the consolidated playbook — the protocol, the
search strategy, and a per-family table of *which knobs actually move the needle*.

### The protocol (non-negotiable)
Every `tune.py` in this repo follows the same loop, and so should you:
1. **Split honestly first.** Train / validation / test, with the *test set sealed*. For
   sequential and streaming data the split must be **temporal** (leave-last-out / by
   time), never random — a random split leaks the future. (Parts 2, 6, 8.)
2. **Select on validation, report on test, once.** The moment a test number influences
   a choice, your reported result becomes fiction.
3. **Average over seeds.** On small data a single run swings several points from the RNG
   alone. Every tuning result in this repo averages over a few seeds and reports the
   std. (See `gnnlab.seed.set_seed` and any project's `tune.py`.)
4. **Log every trial.** `gnnlab.mlops.ExperimentTracker` writes each run's params +
   metrics; `python -m gnnlab.dashboard --open` shows the sweep. You cannot tune what
   you don't record.

### Search strategy — grid vs random vs Bayesian
| Strategy | Use when | Why |
|---|---|---|
| **Grid search** | ≤ ~30 combinations, few knobs | exhaustive, reproducible (`grid_search` in `tune.py`) |
| **Random search** | larger spaces | Bergstra & Bengio (2012): per-unit-compute it beats grid, because few knobs matter and random spends budget exploring them (`random_search` in `tune.py`) |
| **Bayesian (Optuna/Ax)** | each run is expensive | models the response surface; the next step up when a single train takes minutes-to-hours |
| **ASHA / Hyperband** | many configs, early signal | kills bad trials early; great for deep models |

Order of attack: get a **coarse** random sweep to find the right *region*, then a
**fine** grid around the winner. Don't fine-tune in a region you haven't located.

### What actually matters, per model family
Tune the **bold** ones first; the rest are second-order.

| Family | Knobs that matter most | Typical ranges | Failure signal |
|---|---|---|---|
| **MF / BPR** (`03`,`04`) | **#factors**, **L2 reg**, lr, #epochs/negs | factors 16–256; reg 1e-3–1e-1; lr 1e-3–1e-2 | val RMSE/NDCG U-shape; reg too low → overfit, too high → underfit |
| **iALS** (`08`) | **#factors**, **α (confidence)**, **λ** | factors 32–256; α 1–100; λ 1e-2–1e2 | α too high overweights heavy users; λ controls the U-curve |
| **EASE** (`08`) | **λ only** | 1e1–1e4 (log scale) | one clean U-curve — the easiest model in the course to tune |
| **FM / FFM** (`05`) | **embedding k**, **L2 reg**, lr | k 4–32; reg 1e-6–1e-3; lr 1e-3–5e-3 | k plateaus; FFM overfits at high reg/large k |
| **DeepFM / DCN** (`05`) | **DNN width/depth**, **dropout**, #cross layers, lr | hidden [64–512]×[2–4]; dropout 0–0.5; cross 1–4 | deep part overfits without dropout/early stop |
| **SASRec / GRU4Rec** (`06`) | **max_len**, **embedding dim**, dropout, #blocks/heads, lr | max_len 50–200; dim 32–128; blocks 2; heads 1–4 | dim/len help then **saturate**; >2–3 blocks rarely pays |
| **LightGCN / NGCF** (`04`) | **#layers**, **dim**, **λ** | layers 2–4; dim 64; λ 1e-4–1e-3 | too many layers → **over-smoothing** (accuracy drops) |
| **Two-tower** (`09`) | **dim**, **temperature**, tower depth, #in-batch negs | dim 32–128; temp 0.05–0.2 | temperature mis-set collapses or flattens similarities |
| **FTRL / online** (`07`) | **α, β (learning-rate)**, **L1, L2**, window/decay | α 0.005–0.1; L1 for sparsity | drift if window too long; noise if too short |

### Three rules of thumb you'll reuse everywhere
- 🔧 **Embedding dim**: bigger helps then plateaus while params (and serving cost) keep
  growing — find the knee, don't chase the last 0.2%.
- 🔧 **Regularization is the most important deep-RecSys knob with sparse data** (few
  labels per user). Dropout + L2 + early stopping on a validation metric.
- 🔧 **Depth is not free**: in GNNs more layers cause over-smoothing; in sequence models
  returns vanish past 2–3 blocks; in MLPs they invite overfitting. Add depth last.

> **Where this fits / what to read next:** with a model chosen, tuned, and measured,
> the remaining gap to *senior* is operational — Part 11 is the catalogue of things that
> break in production even when your offline numbers are perfect.

---

## Part 11 — Edge Cases & War Stories (what breaks when the numbers look fine)

Offline metrics can be flawless and the system can still fail. This Part is the
catalogue of failure modes a senior DS carries in their head — the things that don't
show up in `model_results.csv` but show up in the incident channel. I've grouped them
by where they bite.

### Data & leakage
- **Temporal leakage.** A random train/test split lets the model "see the future."
  *Symptom:* offline metrics look amazing, online flops. *Fix:* split by time
  (leave-last-out), as `06`/`07` do. The single most common way RecSys results are
  accidentally faked.
- **Sampled-metric distortion.** Evaluating against 100 sampled negatives instead of the
  full catalogue can **reorder which model looks best** (Krichene & Rendle, 2020).
  *Fix:* full-catalogue eval when you can; be explicit about the protocol (`08`/`09`
  evaluate over the catalogue; `06` documents its 1-pos+100-neg pool).
- **Duplicate / bot / scraper rows** inflate popularity and poison co-occurrence. Clip
  per-user counts; cap confidence (relevant to iALS in `08`).
- **Missing-Not-At-Random.** Users rate what they chose to consume; "missing" ≠
  "disliked." Every offline number is computed on a biased sample. (Part 8 / IPS.)

### Cold-start (the perennial)
- **New item / new user / new system.** Pure ID-based MF/LightGCN **cannot embed an id
  it never trained on** — it will silently return garbage or the popular fallback.
  *Fix:* content/feature towers (`09`), hybrid models, exploration. Always have a
  **graceful popularity fallback** wired in.
- 💡 *War story:* a model that "works" in offline eval but the eval set has *no* truly
  cold items — so cold-start is broken in prod and invisible offline. Always slice
  metrics by item age / interaction count (`04` does this).

### Popularity & feedback loops
- **Popularity bias** compounds: the model recommends popular items → they get more
  interactions → they look even more popular next training. The catalogue collapses to a
  bubble. *Detect:* coverage, ARP, Gini (`gnnlab.metrics`, used in `06`). *Fix:*
  re-ranking for diversity, exploration, debiasing.
- **In-batch negatives are popularity-biased**: popular items appear as negatives more
  often, suppressing them — sometimes good, sometimes not. The **logQ correction**
  (Part 7) addresses it.

### Training ↔ serving
- **Training-serving skew** — the #1 silent production bug. A feature is computed one way
  in the training pipeline and another way at serving (different default, timezone,
  normalization). *Fix:* a shared feature definition / feature store; log serving
  features and re-score offline to compare.
- **Embedding staleness.** Two-tower item embeddings are precomputed and indexed in ANN;
  if the index isn't refreshed, new/changed items are served stale vectors.
- **Delayed feedback.** Conversions arrive hours later; a model trained on "no
  conversion yet" learns the wrong label. Use delayed-feedback modeling or appropriate
  attribution windows.

### Evaluation traps
- **Comparing to a weak baseline.** The Dacrema (2019) lesson: tune your baselines as
  hard as your hero model, or your "win" is an artifact. (Part 3.)
- **Average hides the tail.** A great mean metric can hide that the model fails for a
  whole segment (cold users, a demographic). Slice everything (`04`).
- **Novelty effect** in A/B tests: anything new gets a temporary click bump; wait it out
  before declaring victory. (Part 2.)

### Numerical / engineering
- **Tie-breaking in ranking eval** can hand a free rank-1 to the held-out item when all
  scores are equal (cold start, untrained model) — randomize tie-breaks. (A real bug
  caught and fixed in `07`'s evaluator.)
- **Over-smoothing** in deep GNNs: stacking layers makes every node identical and
  accuracy *drops* (`01`'s ablation shows the curve; same caution for LightGCN in `04`).
- **Non-writable array / dtype surprises**, sparse-matrix invariants, silent NaNs in
  LogLoss — clip probabilities (`gnnlab.metrics.log_loss` does), assert no NaN/inf.

### The habit that prevents most of these
Slice, monitor, and alert. Almost every item above is invisible in a single aggregate
number and obvious the moment you (a) slice by segment/recency/popularity and (b)
monitor the live distribution against training. That instinct — *"what's the average
hiding?"* — is the most valuable thing in this Part.

> **Where this fits / what to read next:** you now have the models, the metrics, the
> tuning, and the failure modes. Part 12 is about *you* — how to compound this into a
> career that ends with you running the team.

---

## Part 12 — Your Career Roadmap (from here to lead data scientist)

I'm writing this Part as the person responsible for your growth. Models are learnable in
months; the judgment that makes a *lead* takes deliberate practice. Here's the path I'd
hold you to.

### The three ladders you climb at once
1. **Technical depth** — you can derive, implement, tune, and debug the workhorses
   (this course). Necessary, never sufficient.
2. **Systems & production** — you can design the retrieval→ranking system, reason about
   latency/cost/freshness, and own a model in production (Parts 7–8, every Production
   Lens).
3. **Business & influence** — you translate a vague stakeholder complaint into a metric
   and an experiment, and you can convince people. **This is the ladder most strong ICs
   neglect, and it's the one that makes you a lead.**

### A concrete progression
| Stage | What you can do | The gap to close next |
|---|---|---|
| **Junior** | call `.fit()`, read a metric | build the baseline first; learn evaluation deeply (Part 2) |
| **Mid** | implement & tune the workhorses; honest offline eval | think in systems: retrieval→ranking, serving, cost |
| **Senior** | own a model end-to-end; design A/B tests; debug prod | drive *ambiguous* problems; mentor; pick what *not* to do |
| **Staff / Lead** | set technical direction; trade off across teams; turn business goals into roadmaps | scale yourself through others; be right about the *important* things |

### What to actually practise (in priority order)
1. **Evaluation, until it's reflex.** The single highest-leverage skill. If you can
   design a leakage-free, well-powered offline+online evaluation for any rec problem,
   you are senior. Re-read Part 2.
2. **Strong baselines, always.** Make "did we beat a tuned popularity / EASE / MF?" a
   reflex (Part 3). It will save you from shipping noise and from believing papers.
3. **One system design, deeply.** Be able to whiteboard YouTube/Spotify-style
   retrieval→ranking with feature stores, ANN, candidate mixing, re-ranking, and the
   latency budget (Part 7). This is the most common senior-interview and design-review
   topic.
4. **Production ownership.** Take one model from notebook to A/B test to ramp to
   monitoring. The Production Lens at the end of every notebook is the checklist.
5. **Write.** A crisp one-page design doc that frames a problem, proposes an approach,
   and states the metric & risks is worth more than a clever model nobody understands.
   Lead DSs are made in docs and reviews, not just notebooks.
6. **Responsible AI.** Fairness, privacy, transparency, and feedback-loop harms are
   launch gates at Microsoft/Google. Knowing them deeply differentiates you.

### Reading & staying current (without drowning)
- **Foundational papers** (read them once, properly): Koren MF (2009), Hu/Koren/Volinsky
  iALS (2008), Rendle BPR (2010) & FM (2010), He LightGCN (2020), Kang SASRec (2018),
  Steck EASE (2019), Krichene & Rendle sampled-metrics (2020), Dacrema "are we making
  progress?" (2019). These are cited throughout this course for a reason.
- **Frontier** (skim quarterly): generative retrieval (TIGER), LLM rec, industrial
  sequential transformers (Part 9). Know the direction; don't chase every paper.
- **Venues**: RecSys, KDD, SIGIR, WWW, WSDM, NeurIPS. Engineering blogs from Netflix,
  Spotify, Pinterest, Meta, Google, Microsoft are gold for the *systems* ladder.

### The mindset I want you to internalise
- **Be the person who runs the simple baseline** — it earns trust and catches nonsense.
- **Optimise the business metric, not the loss.** Tie every model metric to a product
  metric to money, and never forget the guardrail.
- **Be honest about uncertainty** (the FM and ENZYMES notebooks in this repo deliberately
  report results that *contradicted* the expected story — that integrity is the job).
- **Scale yourself through others** as you climb: the lead isn't the best modeller in the
  room; they're the reason ten good modellers ship the right thing.

> If you genuinely master the seven questions for every workhorse in this course, can
> design a leakage-free evaluation in your sleep, can whiteboard the production stack,
> and can write the one-pager that gets it funded — you are the person I was asked to
> make you. The rest is reps.

---

## Appendix — Cheat-Sheets & Quick Reference

Keep this open while you work. It's the compression of the whole course into the things
you'll reach for weekly.

### A. Model-selection decision tree
```
What do you have / want?
├─ Explicit ratings, predict the rating ........... Biased MF / SVD        (Part 4 · nb 03,04)
├─ Implicit feedback (clicks/plays), top-K ......... EASE or iALS first     (Part 3/4 · nb 08)
│     └─ need pairwise ranking / huge sparse ....... BPR-MF                 (Part 4 · nb 04,08)
├─ Rich categorical features, CTR/ranking .......... FM → DeepFM / DCN-v2   (Part 5 · nb 05)
├─ Order matters (next-item, sessions) ............. SASRec (GRU4Rec first) (Part 6 · nb 06)
├─ Millions of items, need fast candidates ......... Two-tower + ANN        (Part 7 · nb 09)
├─ Dense interaction graph, collaborative signal ... LightGCN               (Part 7 · nb 04)
├─ Data arrives as a stream / drifts ............... FTRL / incremental MF  (Part 8 · nb 07)
└─ Always, as the yardstick ........................ Popularity baseline    (Part 2/3 · all)
```
Production reality: it's almost never one model. It's **two-tower retrieval → DeepFM/DCN
ranking → diversity re-ranking**, retrained on a schedule with online updates between.

### B. Metrics cheat-sheet
| Task | Optimise | Also report (guardrails) | Repo |
|---|---|---|---|
| Rating prediction | RMSE | MAE | `03`,`04` |
| Top-K ranking | NDCG@K | Recall@K, HitRate@K, MRR | `06`,`08` |
| Retrieval | Recall@K (full catalogue) | coverage, latency | `09` |
| CTR / ranking | LogLoss (calibration) | AUC, ECE | `05` |
| Any deployed recommender | the product metric (A/B) | coverage, novelty, ARP/Gini (popularity bias), fairness, latency, cost | all |
*Golden rule:* select on validation, report on test once, average over seeds, and never
let beyond-accuracy hit zero in pursuit of accuracy.

### C. Dataset guide (and what each teaches)
| Dataset | Signal | Use it to learn | Notebook |
|---|---|---|---|
| MovieLens-100K | explicit, small | the full classical→GNN pipeline | `03` |
| MovieLens-1M | explicit + demographics + time | scale, FM/BPR/NGCF, cold-start slices | `04` |
| Frappe | dense categorical CTR | FM family, feature interactions, calibration | `05` |
| MovieLens-1M (sequences) | temporal order | GRU4Rec/SASRec, leave-last-out | `06` |
| MovieLens-1M (replayed stream) | time-ordered events | online learning, drift, prequential eval | `07` |
| Last.fm | implicit play-counts | iALS confidence, EASE, implicit ranking | `08` |

### D. The seven questions (ask these of every method)
1. **Intuition** — what's the one-sentence idea?
2. **Math** — the objective / update / closed form.
3. **Paper** — who, when, what claim?
4. **Hyperparameters** — which knobs, what ranges, what matters most? (Part 10)
5. **When it breaks** — the edge cases. (Part 11)
6. **Use case** — when would you actually pick it?
7. **Proof** — which notebook implements & measures it?

### E. Command reference (this repo)
```bash
# environment
source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# run all tests (per-project, isolated processes)
./run_tests.sh

# reproduce a project's real numbers, then (re)build + execute its notebook
cd projects/05_factorization_machines
python run_experiments.py
python build_notebook.py
python -m nbconvert --to notebook --execute --inplace factorization_machines.ipynb

# open the MLOps dashboard (aggregates every project's runs)
python -m gnnlab.dashboard --open
```

### F. One-line glossary
- **CF** collaborative filtering · **MF** matrix factorization · **iALS/WMF** implicit
  ALS / weighted MF · **EASE** embarrassingly shallow autoencoder · **BPR** Bayesian
  personalized ranking · **FM/FFM** (field-aware) factorization machines · **DCN** deep
  & cross network · **SASRec** self-attentive sequential rec · **LightGCN** light graph
  conv network · **two-tower** dual-encoder retrieval · **ANN** approximate nearest
  neighbour · **FTRL** follow-the-regularized-leader (online) · **NDCG** normalized
  discounted cumulative gain · **ARP** average recommendation popularity · **IPS**
  inverse propensity scoring · **MNAR** missing-not-at-random · **CTR** click-through
  rate · **ECE** expected calibration error.

---

*This course is a living document. Each Part is paired with a tested notebook so you can
always move between the idea and the running code. Go build.*

