# Research Dossier: Extra Trees, Bagging & Ensemble Methods
**Target file:** `extra-trees.md`
**Prepared for:** Supervised Learning Masterclass — Chapter Author
**Research date:** June 2026

---

## Scope

This dossier feeds the chapter file `extra-trees.md`, which covers:
- Bagging (Breiman 1996) — bootstrap aggregating, variance reduction
- Extremely Randomized Trees / Extra Trees (Geurts, Ernst & Wehenkel 2006) — the central algorithm
- `sklearn.ensemble.ExtraTreesClassifier` and `ExtraTreesRegressor` — all parameters, defaults, differences from `RandomForest`
- `sklearn.ensemble.BaggingClassifier` / `BaggingRegressor` — generic bagging meta-estimator
- Voting ensembles: `VotingClassifier` / `VotingRegressor`
- Stacking: `StackingClassifier` / `StackingRegressor`
- Rotation Forest (Rodriguez 2006), Mondrian Forests (Lakshminarayanan 2014), Deep Forests / gcForest (Zhou & Feng 2017)
- When Extra Trees beats Random Forest and vice versa
- Feature importance: MDI pitfalls, permutation importance, SHAP TreeExplainer

---

## Verified Packages

### Core sklearn (1.5.x)

```python
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    BaggingClassifier,
    BaggingRegressor,
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
    RandomForestClassifier,    # for comparison
)
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor  # single-tree variants
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap  # 0.46.x — TreeExplainer supports ExtraTrees
import optuna  # 3.x — TPE-based hyperparameter optimization
```

**Package inventory:**

| Package | Version (mid-2026) | Role | License |
|---|---|---|---|
| scikit-learn | ~1.5.x | ExtraTrees, Bagging, Voting, Stacking | BSD-3 |
| shap | ~0.46.x | TreeExplainer for ExtraTrees SHAP values | MIT |
| optuna | ~3.x | TPE-based hyperparameter search | MIT |
| sktime | stable | RotationForest classifier (sklearn-compatible) | BSD-3 |
| rotation-forest | 1.0 (PyPI) | Standalone Rotation Forest sklearn wrapper | — |
| IncrementalTrees | (GitHub) | partial_fit for sklearn forest ensembles | — |

**Verified import paths (sklearn 1.5.2):**

```python
# Confirmed working imports
from sklearn.ensemble import ExtraTreesClassifier   # ensemble of Extra Trees
from sklearn.tree import ExtraTreeClassifier         # single Extra Tree (base learner)

# SHAP for ExtraTrees — TreeExplainer works with all sklearn tree ensembles
import shap
explainer = shap.TreeExplainer(model)  # model can be ExtraTreesClassifier/Regressor
shap_values = explainer.shap_values(X_test)
```

**Key API notes verified in docs:**
- `sklearn.ensemble.ExtraTreesClassifier` and `ExtraTreesRegressor` share 19 parameters with nearly identical signatures (differing only in `criterion` defaults and `class_weight`)
- `BaggingClassifier` parameter `base_estimator` was renamed to `estimator` in sklearn 1.2; `base_estimator` is deprecated
- `StackingClassifier`'s `final_estimator` defaults to `LogisticRegression()` if `None`
- `shap.TreeExplainer` supports "most tree-based scikit-learn models" per official docs — explicitly covers ExtraTrees in the SHAP GitHub and tutorials

---

## The Original Algorithm — Breiman Bagging (1996)

**Full citation:**
> Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123–140.
> DOI: https://doi.org/10.1007/BF00058655
> URL: https://link.springer.com/article/10.1007/BF00058655

**The problem solved:** Single unstable learners (deep decision trees, neural networks of the era) had high variance — small perturbations in training data led to wildly different models. Breiman wanted a mechanically simple way to kill that variance without changing the base learner at all.

**The core insight:** If a predictor is unstable (different data → different model), average many versions trained on slightly different datasets. The averaging cancels out uncorrelated errors. Key condition: **the perturbation must change the model substantially** — if the learner is stable, bagging does almost nothing.

**The bagging procedure (pseudocode):**
```
Input: Training set D = {(x_1, y_1), ..., (x_n, y_n)}, base learner L, B bags
For b = 1 to B:
    D_b = bootstrap_sample(D)  # draw n samples WITH replacement from D
    h_b = L(D_b)               # fit base learner on D_b
Output:
    Classification: f(x) = majority_vote({h_b(x) : b = 1..B})
    Regression:     f(x) = (1/B) * sum_b h_b(x)
```

**The bias-variance-bagging relationship:**
For regression, the expected MSE decomposes as:

$$\text{MSE}(f) = \text{Bias}^2(f) + \text{Var}(f) + \sigma^2_\epsilon$$

Bagging approximates the true ensemble predictor $f_A(\mathbf{x}) = \mathbb{E}_{D}[h(\mathbf{x}, D)]$ (averaging over all possible training sets). By Jensen's inequality, for convex loss:

$$L(y, f_A(\mathbf{x})) \le \mathbb{E}_D[L(y, h(\mathbf{x}, D))]$$

So the aggregated predictor has loss ≤ expected loss of a single predictor. **Bagging reduces variance while leaving bias essentially unchanged.** Breiman's empirical finding: 20–40% error reduction on regression trees vs. single tree.

**What was uncertain at publication:** Whether bagging would work on classification (it does — voting also reduces variance). Whether it could hurt stable learners (it can, slightly, by introducing bootstrap noise with no compensating variance reduction).

---

## The Original Algorithm — Extra Trees (Geurts, Ernst & Wehenkel 2006)

**Full citation:**
> Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning*, 63(1), 3–42.
> DOI: https://doi.org/10.1007/s10994-006-6226-1
> URL: https://link.springer.com/article/10.1007/s10994-006-6226-1
> PDF: https://jasonphang.com/files/extratrees.pdf

**The problem solved:** Random Forests (Breiman 2001) reduced variance by (a) bootstrapping and (b) restricting split search to a random subset of $k$ features per node. But **split threshold selection was still greedy** — the best threshold among the $k$ candidate features was found by exhaustive search over all training values. Geurts et al. asked: can we push randomization further and get more variance reduction, faster training, and sometimes better generalization?

**The core innovations — two sources of randomness:**
1. **Random thresholds:** For each candidate feature, draw a random cut-point uniformly from the feature's range in the current node's sample, rather than searching for the optimal threshold.
2. **No bootstrapping (by default):** Use the full training set for every tree — the randomness in split selection alone is the source of tree diversity.

**The ExtraTree algorithm (single tree, pseudocode from the paper):**
```
ExtraTree(S, F):
  Input: Training sample S = {(x_i, y_i)}, feature set F
  If stopping_criterion(S):
      Return Leaf(predict(S))
  
  # Draw K candidate features
  F* = random_subset(F, K)
  
  # For each candidate, draw ONE random split
  splits = []
  For each a in F*:
      x_min, x_max = min(S[:,a]), max(S[:,a])
      t_a ~ Uniform(x_min, x_max)   # random threshold
      score_a = Score(a, t_a, S)    # e.g., variance reduction or Gini
      splits.append((a, t_a, score_a))
  
  # Select best split among K random splits
  a*, t* = argmax_{(a,t) in splits} score(a, t)
  
  # Partition and recurse
  S_left  = {(x,y) in S : x[a*] <= t*}
  S_right = {(x,y) in S : x[a*] >  t*}
  Return Node(a*, t*, ExtraTree(S_left, F), ExtraTree(S_right, F))
```

**The scoring function:** Uses the same impurity measures as standard trees — Gini impurity for classification, variance reduction for regression:

$$\text{Score}(a, t, S) = \frac{n_S}{n} \cdot \left( i(S) - \frac{n_{S_l}}{n_S} i(S_l) - \frac{n_{S_r}}{n_S} i(S_r) \right)$$

where $i(S)$ is the impurity of node $S$, and $n_S$, $n_{S_l}$, $n_{S_r}$ are sample counts.

**The ensemble (Extra Trees forest):**
```
ExtraTreesForest(S, F, T, K):
  For t = 1 to T:
      h_t = ExtraTree(S, F)   # full data, no bootstrap
  
  Regression:     f(x) = (1/T) * sum_t h_t(x)
  Classification: f(x) = majority_vote or probability_average
```

**Key hyperparameter K:** Number of candidate features per split. Special cases:
- $K=1$: Totally randomized trees — splits are independent of the output
- $K=p$ (all features): Extra Trees uses all features but still random thresholds — more randomness than RF with $K = \sqrt{p}$
- $K=\sqrt{p}$: Default for classification (matches RF default)
- $K=p$ (or 1.0): Default for regression in sklearn

**Bias-variance analysis from the paper:** The authors prove that with enough trees, Extra Trees' bias can be *lower* than RF's for certain problem structures (particularly when the optimal split threshold is not at a training point — common in practice). The variance is definitively lower due to the full decorrelation of trees. The total expected error can be better even though individual trees have higher variance from random thresholds.

**What was experimental at publication:** Whether removing bootstrapping (a deliberate design choice in the original paper) would hurt OOB error estimation capability. In sklearn, `bootstrap=False` by default — which means `oob_score=True` will raise a `ValueError` unless you also set `bootstrap=True`.

---

## Major Variants & Evolution

### 1. Random Forests (Breiman 2001) — the immediate predecessor
**Paper:** Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32. DOI: https://doi.org/10.1023/A:1010933404324
**What it contributed:** Combining bagging with feature subspace sampling ($k = \sqrt{p}$ features per split). Threshold selection is still greedy (optimal threshold among the $k$ candidate features is found by exhaustive search). The canonical baseline against which Extra Trees is compared.
**Python:** `sklearn.ensemble.RandomForestClassifier/Regressor`

### 2. Extremely Randomized Trees / Extra Trees (Geurts 2006) — the chapter's central algorithm
Covered above. Key differences from RF:
- Random thresholds (not optimal) → more randomness → lower variance
- Full training set per tree by default (no bootstrap) → slightly higher bias but faster
- Often faster to train: split evaluation is O(K) not O(K · n log n) — no sorting required
- Sometimes better regularization on noisy high-dimensional data
**Python:** `sklearn.ensemble.ExtraTreesClassifier/Regressor`

### 3. Rotation Forest (Rodriguez, Kuncheva & Alonso 2006)
**Paper:** Rodríguez, J. J., Kuncheva, L. I., & Alonso, C. J. (2006). Rotation Forest: A new classifier ensemble method. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 28(10), 1619–1630.
**What changed:** Rather than random feature subsets, Rotation Forest applies PCA to random subsets of features and uses the PC projections as new features. This creates diverse, rotated feature spaces for each tree — oblique splits rather than axis-aligned.
**Algorithm sketch:**
```
For each tree t:
    Randomly partition F into K equal subsets {F_1, ..., F_K}
    For each subset F_j:
        Draw a random sample of 75% of training instances
        Apply PCA to subset F_j on this sample → get rotation matrix R_j
    Combine all R_j into block-diagonal rotation matrix R_t
    Train decision tree on X * R_t
```
**When to prefer:** When decision boundaries are oblique (diagonal) in the original feature space; when the data is dense with correlated continuous features. Competitive with RF and Extra Trees on classification benchmarks.
**Python:** `sktime.classification.sklearn.RotationForest` (sklearn-compatible) or `pip install rotation-forest`

### 4. Mondrian Forests (Lakshminarayanan, Roy & Teh 2014)
**Paper:** Lakshminarayanan, B., Roy, D. M., & Teh, Y. W. (2014). Mondrian Forests: Efficient Online Random Forests. In *Advances in Neural Information Processing Systems* (NeurIPS 2014).
ArXiv: https://arxiv.org/abs/1406.2673
PDF: https://www.gatsby.ucl.ac.uk/~balaji/mondrian_forests_nips14.pdf
**What changed:** Uses Mondrian processes (Roy & Teh 2009) to construct tree structure. The key innovation: **online equivalence** — a Mondrian forest can be updated incrementally as new data arrives, and the distribution over trees remains the same as a batch-trained forest. This is unlike standard Random Forests which must be fully retrained.
**Prediction intervals:** The hierarchical structure enables calibrated uncertainty estimates — trees partition space in a way that naturally provides confidence intervals.
**When to prefer:** Streaming/online settings, when you need prediction intervals, when the data distribution can shift.
**Python:** No major maintained sklearn package as of 2026; the `mondrian-forest` package on GitHub is research-grade. The sktime ecosystem has some variants.

### 5. Deep Forests / gcForest (Zhou & Feng 2017)
**Paper:** Zhou, Z.-H., & Feng, J. (2017). Deep Forest: Towards An Alternative to Deep Neural Networks. arXiv:1702.08835.
URL: https://arxiv.org/abs/1702.08835
**What changed:** Stacks multiple random forest / Extra Trees layers, where each layer's predicted probability vectors are concatenated with the original features and fed to the next layer — mimicking the layer-by-layer representation learning of deep neural networks. Crucially, the number of layers is determined automatically by validation performance (cascade structure).
**Algorithm sketch:**
```
Layer 1: [RandomForest, ExtraTrees] -> predicted probabilities P1
Layer 2: [RandomForest, ExtraTrees] on [X || P1] -> P2
...
Continue until validation accuracy stops improving
Final prediction: average of last layer's outputs
```
**When to prefer:** When you want deep learning's hierarchical feature learning without GPU requirements; on datasets where standard RF/ET plateaus but DNN is overkill. In practice, often similar to a well-tuned XGBoost.
**Python:** `pip install deep-forest` (DeepForest package by the original authors); also `deepforest` on PyPI.

### 6. Aggregated Mondrian Forests (Mourtada, Gaïffas & Scornet 2019)
**Paper:** Mourtada, J., Gaïffas, S., & Scornet, E. (2019). AMF: Aggregated Mondrian Forests for Online Learning. In *NeurIPS 2019*.
URL: https://arxiv.org/abs/1906.10080
**What changed:** Theoretical guarantees on convergence for online learning; the aggregation uses exponential weighting (Bayesian model averaging) rather than uniform averaging, giving better uncertainty quantification.

---

## Hyperparameter Reference

### ExtraTreesClassifier / ExtraTreesRegressor (sklearn 1.5.x)

**Verified from sklearn 1.5.2 documentation:**

| Parameter | Classifier Default | Regressor Default | What it Controls |
|---|---|---|---|
| `n_estimators` | 100 | 100 | Number of trees in the forest |
| `criterion` | `'gini'` | `'squared_error'` | Split quality metric |
| `max_depth` | `None` | `None` | Maximum tree depth (None = no limit) |
| `min_samples_split` | `2` | `2` | Min samples to split a node |
| `min_samples_leaf` | `1` | `1` | Min samples required at a leaf node |
| `min_weight_fraction_leaf` | `0.0` | `0.0` | Min weighted fraction at a leaf |
| `max_features` | `'sqrt'` | `1.0` | **Key: features considered per split** |
| `max_leaf_nodes` | `None` | `None` | Grow best-first with max leaf count |
| `min_impurity_decrease` | `0.0` | `0.0` | Min impurity decrease to split |
| `bootstrap` | `False` | `False` | **Key: False by default (differs from RF)** |
| `oob_score` | `False` | `False` | OOB generalization score (requires bootstrap=True) |
| `n_jobs` | `None` | `None` | Parallelism (-1 = all cores) |
| `random_state` | `None` | `None` | Random seed |
| `verbose` | `0` | `0` | Verbosity |
| `warm_start` | `False` | `False` | Reuse/extend previous fit |
| `class_weight` | `None` | — | Class weighting (classifier only) |
| `ccp_alpha` | `0.0` | `0.0` | Cost-complexity pruning parameter |
| `max_samples` | `None` | `None` | Sample size per tree (requires bootstrap=True) |
| `monotonic_cst` | `None` | `None` | Monotonicity constraints array |

**Critical defaults that differ from RandomForest:**

| Parameter | ExtraTrees default | RandomForest default | Consequence |
|---|---|---|---|
| `bootstrap` | `False` | `True` | ET uses full data per tree; RF uses ~63.2% |
| `max_features` (regressor) | `1.0` (all features) | `1.0` (since sklearn 1.1) | Same now, but ET also randomizes thresholds |
| Split threshold | **Random** (from range) | **Optimal** (best found) | ET's core difference — cannot be changed |

**Criterion options:**
- Classifier: `'gini'` (default), `'entropy'`, `'log_loss'`
- Regressor: `'squared_error'` (default), `'absolute_error'`, `'friedman_mse'`, `'poisson'`

**`max_features` parameter in detail:**
```python
# These are all equivalent ways to specify max_features:
max_features = 'sqrt'     # sqrt(n_features) — default for classifier
max_features = 'log2'     # log2(n_features)
max_features = 1.0        # all features (default for regressor)
max_features = 0.3        # 30% of features
max_features = 10         # exactly 10 features (int)
max_features = None       # same as 1.0 (all features)
```

**Tuning `max_features` for Extra Trees:**
- Lower values → more randomization → lower variance, higher bias
- The original paper finds $K = \sqrt{p}$ often optimal for classification, $K = p$ for regression (random thresholds already provide enough variance without restricting feature space)
- Try `[0.3, 0.5, 0.7, 'sqrt', 'log2', 1.0]` in a search

**`n_estimators` guidance:**
- 100 is often enough; Extra Trees converge faster than RF because trees are less correlated
- Use learning curves (OOB error vs n_estimators, or validation score) to find plateau
- With `warm_start=True`, you can grow the forest incrementally:
  ```python
  et = ExtraTreesClassifier(n_estimators=100, warm_start=True)
  et.fit(X_train, y_train)
  et.n_estimators = 200   # TOTAL desired — not 100 more
  et.fit(X_train, y_train)  # adds 100 more trees
  ```

**`bootstrap` and `oob_score` pitfall:**
```python
# THIS RAISES ValueError at fit time:
et = ExtraTreesClassifier(bootstrap=False, oob_score=True)  # WRONG

# Correct: enable bootstrap to use OOB
et = ExtraTreesClassifier(bootstrap=True, oob_score=True)   # OK
print(et.oob_score_)  # available after fit
```

**`ccp_alpha` (Minimal Cost-Complexity Pruning):**
- Default 0.0 = no pruning; increasing `ccp_alpha` prunes more aggressively
- Find good `ccp_alpha` using `cost_complexity_pruning_path` on a single decision tree, then apply to Extra Trees
- Useful for combating overfitting when trees are allowed to grow very deep

**`max_samples` (added in sklearn 0.22):**
- Only active when `bootstrap=True`
- Controls how many samples each tree sees; smaller values → more diversity → lower correlation
- Combine with `bootstrap=True` if you want both bootstrap and `oob_score`

**`monotonic_cst`:**
- Array of shape `(n_features,)` with values in {-1, 0, 1}
- Not supported for multiclass or multioutput — will raise an error
- Useful for business constraints (e.g., "price must monotonically increase feature importance")

### Optuna search for ExtraTreesClassifier

```python
import optuna
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]
        ),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }
    model = ExtraTreesClassifier(**params, n_jobs=-1, random_state=42)
    return cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
best_params = study.best_params
```

**Tuning playbook:**

| Hyperparameter | Typical Range | Effect (high → low) | Tuning strategy |
|---|---|---|---|
| `n_estimators` | 100–1000 | More trees → lower variance, diminishing returns | Use OOB curve; 200–500 usually sufficient |
| `max_depth` | None, 5–50 | Deeper → more variance; shallower → more bias | Start None; add constraint if overfitting |
| `min_samples_split` | 2–20 | Higher → smaller trees, more regularization | Grid [2,5,10,20] |
| `min_samples_leaf` | 1–10 | Higher → smoother predictions | Grid [1,2,5,10] |
| `max_features` | 0.3–1.0, 'sqrt', 'log2' | Fewer → more diversity, higher bias | Try all: ['sqrt','log2',0.5,1.0] |
| `bootstrap` | True/False | True → adds data diversity (bootstrap noise) | Try both; False is ET default |
| `ccp_alpha` | 0.0–0.05 | Higher → more pruning | Use pruning path; typically 0–0.01 |
| `max_samples` | 0.5–1.0 (with bootstrap=True) | Lower → more diverse trees | Try [0.5, 0.7, 0.9, None] |

### BaggingClassifier / BaggingRegressor (sklearn 1.5.x)

**Verified from sklearn 1.5.2 documentation:**

```python
BaggingClassifier(
    estimator=None,          # Default: DecisionTreeClassifier()
    n_estimators=10,         # number of base estimators
    max_samples=1.0,         # samples per estimator (float = fraction)
    max_features=1.0,        # features per estimator (float = fraction)
    bootstrap=True,          # sample WITH replacement
    bootstrap_features=False,# feature sampling without replacement by default
    oob_score=False,         # OOB estimate (only when bootstrap=True)
    warm_start=False,
    n_jobs=None,
    random_state=None,
    verbose=0
)
```

**Important API note:** `base_estimator` was renamed `estimator` in sklearn 1.2. Using `base_estimator` in sklearn 1.5+ will raise a deprecation warning.

**Bagging variants controlled by parameters:**

| Variant name | `bootstrap` | `bootstrap_features` | Description |
|---|---|---|---|
| Bagging | True | False | Samples with replacement (standard bagging) |
| Pasting | False | False | Samples without replacement |
| Random Subspaces | False | True | Feature subsets without replacement |
| Random Patches | True | True | Both sample and feature subsets with replacement |

### VotingClassifier (sklearn 1.5.x)

```python
VotingClassifier(
    estimators,              # list of (name, estimator) tuples — REQUIRED
    voting='hard',           # 'hard' (majority) or 'soft' (probability avg)
    weights=None,            # array of weights per estimator; None = uniform
    n_jobs=None,
    flatten_transform=True,  # shape of transform() output for soft voting
    verbose=False
)
```

**Hard vs soft voting:**
- `voting='hard'`: Predict class with most votes — works even without predict_proba
- `voting='soft'`: Average predicted probabilities, pick argmax — better when classifiers are well-calibrated; requires all estimators to support `predict_proba`

**VotingRegressor** has no `voting` parameter — always averages predictions:
```python
VotingRegressor(estimators, weights=None, n_jobs=None, verbose=False)
```

### StackingClassifier / StackingRegressor (sklearn 1.5.x)

```python
StackingClassifier(
    estimators,              # list of (name, estimator) tuples — REQUIRED
    final_estimator=None,    # Default: LogisticRegression(); meta-learner
    cv=None,                 # Default: 5-fold CV; or "prefit" if already fitted
    stack_method='auto',     # 'auto', 'predict_proba', 'decision_function', 'predict'
    n_jobs=None,
    passthrough=False,       # If True, include original X in meta-learner features
    verbose=0
)
```

**How stacking works (the key detail):**
1. Base estimators are fitted on the full training set
2. Out-of-fold predictions are generated via `cross_val_predict` using `cv` folds
3. The meta-learner (final_estimator) is trained on these out-of-fold predictions
4. At inference, base estimators predict on new data, meta-learner combines

```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True, random_state=42)),
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(C=0.1),
    cv=5,
    stack_method='predict_proba',
    passthrough=True,  # pass original features to meta-learner too
    n_jobs=-1,
)
stack.fit(X_train, y_train)
```

---

## Common Errors & Pitfalls

### 1. OOB score with bootstrap=False (most common mistake)
```python
# WRONG — raises ValueError at fit time
ExtraTreesClassifier(bootstrap=False, oob_score=True).fit(X, y)
# ValueError: Out of bag estimation only available if bootstrap=True

# RIGHT — must enable bootstrap
ExtraTreesClassifier(bootstrap=True, oob_score=True).fit(X, y)
```
This is confirmed in GitHub issue #19431 where the documentation was flagged for not making this constraint explicit enough.

### 2. warm_start n_estimators semantics
```python
et = ExtraTreesClassifier(n_estimators=100, warm_start=True)
et.fit(X_train, y_train)  # trains 100 trees

# WRONG mental model — this does NOT add 100 trees to make 200:
et.n_estimators = 100
et.fit(X_train, y_train)  # tries to re-use same 100, raises ValueError if n_estimators decreased

# CORRECT — n_estimators is the TOTAL desired count
et.n_estimators = 200  # will add 100 more trees to the existing 100
et.fit(X_train, y_train)  # now has 200 trees total
```

### 3. max_features default changed between sklearn versions
- sklearn < 1.1: `ExtraTreesRegressor` default was `max_features='auto'` (meaning all features)
- sklearn >= 1.1: Default changed to `max_features=1.0` (same as all features, but explicit)
- Classifier default was and remains `'sqrt'`
- **Pitfall:** Code written for sklearn 0.x may have different behavior silently if you assumed the default

### 4. MDI feature importance bias
Extra Trees' `feature_importances_` attribute uses Mean Decrease in Impurity (MDI), which:
- Favors high-cardinality continuous features over low-cardinality/binary features
- Is computed on training data — does not reflect test-set generalizability
- Can rank random noise features highly if the model overfits
```python
# Prefer permutation importance on a held-out set:
from sklearn.inspection import permutation_importance
result = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)
# result.importances_mean, result.importances_std
```

### 5. ExtraTrees on very small datasets
With `bootstrap=False` (default), all trees see all training data. With few samples, the trees will be identical except for random threshold choices → poor diversity → suboptimal ensemble. Use `bootstrap=True` and `max_samples < 1.0` when $n < 500$.

### 6. Random threshold selection producing poor splits
In very high-noise settings with extreme class imbalance, random thresholds may consistently fail to find an informative split in the first few levels of a tree. Check with `max_depth=None` and then try `max_depth=5` — if performance jumps, trees are growing on noise.

### 7. BaggingClassifier predict_proba aggregation
BaggingClassifier's `predict_proba` averages probabilities across base estimators. If the base estimator doesn't support `predict_proba` (e.g., plain `SVC` without `probability=True`), it will raise an `AttributeError`. Always use `SVC(probability=True)` or check `hasattr(estimator, 'predict_proba')`.

### 8. StackingClassifier data leakage with prefit estimators
If using `cv='prefit'`, base estimators must be pre-fitted on a *separate* holdout set, not on `X_train`. Using `cv='prefit'` on estimators fitted on `X_train` causes data leakage — the meta-learner sees training predictions from base estimators that were trained on the same data, inflating out-of-fold performance.

### 9. SHAP TreeExplainer with ExtraTrees — feature_perturbation choice
```python
# Default (no background data) — uses tree_path_dependent
explainer = shap.TreeExplainer(model)

# Better for correlated features — requires background sample
background = shap.sample(X_train, 100)
explainer = shap.TreeExplainer(model, background, feature_perturbation='interventional')
```
With `tree_path_dependent`, SHAP values for correlated features can be misleading because the method conditions on tree path rather than the actual feature distribution.

### 10. Voting with class_weight inconsistency
If using `VotingClassifier` with `voting='hard'`, all sub-classifiers should have consistent class_weight settings. A classifier that upweights class 1 and another that doesn't will produce incomparable vote magnitudes, making the ensemble unpredictable on imbalanced data.

---

## Key Datasets for Examples

### 1. Breast Cancer Wisconsin — classification
```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target
# Shape: (569, 30), binary classification (malignant=0, benign=1)
# Good for: demonstrating ExtraTreesClassifier, feature importance, SHAP
```

### 2. make_classification — synthetic high-dimensional
```python
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=5000,
    n_features=100,     # high-dimensional
    n_informative=10,   # only 10 truly useful
    n_redundant=20,     # 20 correlated to informative
    n_repeated=0,
    n_classes=2,
    random_state=42
)
# Good for: showing Extra Trees vs RF on high-dim noisy data
# Good for: demonstrating MDI bias (noise features get inflated importance)
```

### 3. California Housing — regression
```python
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X, y = data.data, data.target
# Shape: (20640, 8), regression (median house value)
# Good for: ExtraTreesRegressor, PDP plots, SHAP dependence plots
```

### 4. Iris — multi-class (simple demos)
```python
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
# Shape: (150, 4), 3 classes
# Good for: VotingClassifier demos, StackingClassifier basics
```

---

## Curated Resources

### Primary Papers

1. **Geurts et al. 2006 — the Extra Trees paper**
   URL: https://link.springer.com/article/10.1007/s10994-006-6226-1
   DOI: 10.1007/s10994-006-6226-1
   *The definitive source. Contains full pseudocode, bias-variance analysis, score function, and benchmarks vs RF, decision trees, and ridge regression.*

2. **Breiman 1996 — Bagging Predictors**
   URL: https://link.springer.com/article/10.1007/BF00058655
   DOI: 10.1007/BF00058655
   *The original bagging paper. Crucial for understanding why variance reduction works and what "instability" means for an estimator.*

3. **Breiman 2001 — Random Forests**
   URL: https://link.springer.com/article/10.1023/A:1010933404324
   *Context for the Extra Trees paper; shows what Geurts was improving upon.*

4. **Rodríguez et al. 2006 — Rotation Forest**
   URL: https://ieeexplore.ieee.org/document/1688827
   *IEEE TPAMI paper. Key for oblique ensemble methods.*

5. **Lakshminarayanan et al. 2014 — Mondrian Forests**
   URL: https://arxiv.org/abs/1406.2673
   PDF: https://www.gatsby.ucl.ac.uk/~balaji/mondrian_forests_nips14.pdf
   *Online Random Forests with theoretical guarantees and prediction intervals.*

6. **Zhou & Feng 2017 — Deep Forest (gcForest)**
   URL: https://arxiv.org/abs/1702.08835
   *Alternative to deep neural networks using cascaded forest layers.*

### Official Documentation

7. **sklearn ExtraTreesClassifier (1.5.2)**
   URL: https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
   *Canonical API reference — all parameters verified against this page.*

8. **sklearn Ensemble Methods Guide (1.5.2)**
   URL: https://scikit-learn.org/1.5/modules/ensemble.html
   *Comprehensive narrative covering Bagging, Extra Trees, Voting, Stacking with code examples.*

9. **sklearn Permutation Importance vs MDI (1.5.2)**
   URL: https://scikit-learn.org/1.5/auto_examples/inspection/plot_permutation_importance.html
   *Authoritative sklearn example showing MDI bias and how permutation importance corrects it.*

### SHAP

10. **SHAP TreeExplainer documentation**
    URL: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
    *Official API; covers `feature_perturbation`, `model_output`, and supported sklearn models.*

11. **Understanding Tree SHAP for Simple Models (SHAP docs)**
    URL: https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Understanding%20Tree%20SHAP%20for%20Simple%20Models.html
    *Essential worked notebook for understanding TreeExplainer output.*

### Tutorials / Blog Posts

12. **Extra Trees Explained — Visual Guide (Medium)**
    URL: https://medium.com/@samybaladram/extra-trees-explained-a-visual-guide-with-code-examples-4c2967cedc75
    *Visual walkthrough of the random threshold mechanism; good for intuition building.*

13. **Random Forest vs Extra Trees (Baeldung)**
    URL: https://www.baeldung.com/cs/random-forest-vs-extremely-randomized-trees
    *Clear comparison table of algorithmic differences; bias-variance tradeoff discussion.*

14. **Daily Dose of DS — RF vs Extra Trees**
    URL: https://blog.dailydoseofds.com/p/random-forest-vs-extra-trees
    *Practical comparison with benchmarks; shows when each wins.*

15. **Feature importances with a forest of trees (sklearn examples)**
    URL: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    *Official sklearn MDI vs permutation importance comparison example.*

---

## Algorithmic Details the Writer Must Nail

### The key equations

**Bias-variance decomposition for regression (required):**
$$\mathbb{E}[(f(\mathbf{x}) - y)^2] = \underbrace{(\mathbb{E}[f(\mathbf{x})] - y)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(f(\mathbf{x}) - \mathbb{E}[f(\mathbf{x})])^2]}_{\text{Variance}} + \sigma^2_\epsilon$$

**Why averaging reduces variance (the key mathematical insight):**
For $B$ independent identically distributed estimators each with variance $\sigma^2$:
$$\text{Var}\left(\frac{1}{B}\sum_{b=1}^B h_b(\mathbf{x})\right) = \frac{\sigma^2}{B} \to 0 \text{ as } B \to \infty$$

But trees are NOT independent — they're positively correlated (correlation $\rho$):
$$\text{Var}\left(\frac{1}{B}\sum_{b=1}^B h_b(\mathbf{x})\right) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$$

This is the fundamental equation from Breiman (2001). As $B \to \infty$, variance converges to $\rho \sigma^2$. To reduce variance: (a) reduce $\rho$ (decorrelate trees — this is what random feature selection and random thresholds do), (b) reduce $\sigma^2$ (reduce individual tree variance — not the ET/RF strategy).

Extra Trees reduces $\rho$ MORE than RF by adding random threshold selection on top of random feature subsets.

**ExtraTrees score function:**
$$\text{Score}(a, t, S) = \frac{2 \cdot I(a, t, S)}{H(S^{(a)}) + H(S^{(c)})}$$

where $H(S^{(a)})$ is the entropy of the indicator $\mathbf{1}[x_a \le t]$ across the sample, and $H(S^{(c)})$ is the conditional entropy of the output. This is the normalized mutual information criterion from the paper.

For regression, the standard score is variance reduction:
$$\text{Score}(a, t, S) = \text{Var}(S) - \frac{n_{S_l}}{n_S}\text{Var}(S_l) - \frac{n_{S_r}}{n_S}\text{Var}(S_r)$$

### Computational complexity

| Operation | Extra Trees | Random Forest |
|---|---|---|
| Training (per node) | $O(K \cdot n)$ — draw K thresholds, evaluate each in O(n) | $O(K \cdot n \log n)$ — must sort each of K features |
| Total training | $O(T \cdot K \cdot n \cdot d)$ where $d$ = depth | $O(T \cdot K \cdot n \log n \cdot d)$ |
| Inference | $O(T \cdot \log n)$ | $O(T \cdot \log n)$ |

Extra Trees is genuinely faster to train on large $n$ because of the elimination of the sort step.

### When Extra Trees > Random Forest

1. **High-dimensional data with many noisy features:** Random threshold selection acts as implicit regularization — the random split is less likely to overfit to noise than an optimal split.
2. **Fast training is a constraint:** O(Kn) vs O(Kn log n) per split; significant on wide tabular datasets.
3. **Already have enough data:** With large $n$, optimal vs random thresholds make little difference on average (law of large numbers).
4. **Label noise:** Random thresholds are less sensitive to individual mislabeled points than optimal thresholds.

### When Random Forest > Extra Trees

1. **Small, clean datasets:** Optimal split finding squeezes out more signal; Extra Trees' random thresholds waste potential information.
2. **Low-dimensional data:** With $p$ small, the random threshold adds noise without meaningful benefit.
3. **When you want OOB error out of the box:** RF's default `bootstrap=True` gives OOB for free; ET's default is `bootstrap=False`.

---

## Uncertainties

1. **SHAP TreeExplainer explicit ExtraTrees support:** The SHAP documentation says "most tree-based scikit-learn models" but does not explicitly list ExtraTrees in the primary docs page fetched. GitHub issues and tutorials confirm it works. The writer should **verify with a quick test** using `shap.TreeExplainer(ExtraTreesClassifier().fit(X,y))` in the current environment and confirm no errors.

2. **`monotonic_cst` parameter in sklearn 1.5.x:** This parameter was added in sklearn 1.2 for `ExtraTreesClassifier`/`Regressor`. The writer should verify the exact version when it became available and whether there are limitations beyond "not for multiclass/multioutput" (e.g., does it work with `bootstrap=True`?).

3. **`max_features` default for ExtraTreesRegressor:** Search results confirm 1.0 is the default in 1.5.x. But the history is: before 1.1 it was `'auto'` which also meant all features. The writer should test this is indeed 1.0 in the target sklearn version and not changed in 1.5.

4. **Deep Forest (gcForest) Python package stability:** The `deep-forest` PyPI package by Zhou's team was active as of 2022–2023. Its maintenance status and compatibility with Python 3.11+/sklearn 1.5 should be verified before citing it as "installable with pip install deep-forest".

5. **Rotation Forest in sktime API:** The `sktime.classification.sklearn.RotationForest` API may have changed — sktime has had significant API refactoring. Verify the import path in the current sktime stable.

6. **Mondrian Forest Python package:** No actively maintained, sklearn-compatible Python package for Mondrian Forests was found. The writer should either point readers to the `skgarden` library (now unmaintained) or note this as a gap and point to the research implementations only.

7. **Score function in the Geurts 2006 paper:** The exact form of the score function (whether it's normalized mutual information or raw variance reduction) may differ between the classification and regression cases as stated in the paper. The writer should read Section 2 of the paper PDF directly for the precise formula.

8. **`criterion='log_loss'` vs `criterion='entropy'`:** Both are listed in sklearn docs as options for ExtraTreesClassifier; they are mathematically equivalent (log_loss = entropy for classification). But verify whether sklearn 1.5.2 accepts both, or if one is deprecated.

---

## Three-Sentence Summary

Extra Trees (Geurts, Ernst & Wehenkel 2006; DOI 10.1007/s10994-006-6226-1) extend Breiman's Random Forests by randomizing split thresholds uniformly from each feature's range — rather than searching for the optimal cut-point — and by default using the full training set (no bootstrap), which together sharply reduce tree correlation and training time while introducing a modest bias increase that is often more than compensated by variance reduction, especially on noisy high-dimensional problems. The sklearn implementation (`ExtraTreesClassifier`/`ExtraTreesRegressor` in sklearn 1.5.x) is nearly parameter-for-parameter identical to `RandomForestClassifier/Regressor` except for `bootstrap=False` default and the implicit random-threshold mechanism, and the broader sklearn ensemble ecosystem covers generic bagging (`BaggingClassifier`/`BaggingRegressor`), heterogeneous ensembling (`VotingClassifier`/`VotingRegressor`), and stacked generalization (`StackingClassifier`/`StackingRegressor`), all with verified sklearn 1.5.2 APIs. SHAP's `TreeExplainer` supports ExtraTrees for fast exact Shapley values, the MDI feature importance suffers the same high-cardinality bias as Random Forests (permutation importance or SHAP is the recommended remedy), and the chapter must cover the OOB/bootstrap pitfall (`oob_score=True` requires `bootstrap=True`), `warm_start` semantics, and major variants including Rotation Forest (PCA-based oblique splits, sktime), Mondrian Forests (online learning with prediction intervals), and Deep Forests (hierarchical stacked forests as a DNN alternative).
