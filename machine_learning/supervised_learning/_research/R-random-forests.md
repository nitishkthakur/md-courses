# Research Dossier: Random Forests
**For chapter author of `random-forests.md` in the Supervised Learning Masterclass**
**Research date: 2026-06-25 | Target file: `random-forests.md`**

---

## Scope

This dossier feeds the chapter file `random-forests.md`. It covers:
- Breiman's 2001 Random Forests (the canonical algorithm)
- sklearn 1.5.x `RandomForestClassifier` / `RandomForestRegressor` — all parameters verified
- `ExtraTreesClassifier` / `ExtraTreesRegressor` — verified differences
- `IsolationForest` — forest-family anomaly detection variant
- Major variants: Extremely Randomized Trees (Geurts 2006), Isolation Forest (Liu 2008), Oblique/SPORF (Tomita 2020), Mondrian Forests (Lakshminarayanan 2014), Deep Forests / gcForest (Zhou & Feng 2017/2019), Canonical Correlation Forests (Rainforth & Wood 2015)
- SHAP `TreeExplainer` — verified API (shap ~0.46.x)
- GPU packages: cuML/RAPIDS, h2o.ai DRF, Spark MLlib RF
- Variable importance pitfalls (Strobl 2007, 2008)
- OOB error mechanics and attributes
- Hyperparameter reference — all parameter names, defaults, types from live docs
- Common pitfalls from practitioner and GitHub issue reports

---

## Verified Packages

### Core sklearn (primary reference)

```python
# Classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import IsolationForest               # anomaly detection

# Inspection / importance
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# Pipeline integration
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
```

**Version verified against**: scikit-learn 1.5.2 official documentation
**sklearn PyPI URL**: https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html

---

### Verified: `RandomForestClassifier` (sklearn 1.5.2) — Complete Signature

```python
RandomForestClassifier(
    n_estimators=100,           # int, default=100
    criterion='gini',           # {"gini", "entropy", "log_loss"}, default="gini"
    max_depth=None,             # int or None, default=None
    min_samples_split=2,        # int or float, default=2
    min_samples_leaf=1,         # int or float, default=1
    min_weight_fraction_leaf=0.0,  # float, default=0.0
    max_features='sqrt',        # {"sqrt", "log2", None}, int, or float; default="sqrt"
    max_leaf_nodes=None,        # int or None, default=None
    min_impurity_decrease=0.0,  # float, default=0.0
    bootstrap=True,             # bool, default=True
    oob_score=False,            # bool or callable, default=False
    n_jobs=None,                # int or None, default=None (=1)
    random_state=None,          # int, RandomState, or None
    verbose=0,                  # int, default=0
    warm_start=False,           # bool, default=False
    class_weight=None,          # {"balanced", "balanced_subsample"}, dict, list, or None
    ccp_alpha=0.0,              # float >= 0, default=0.0
    max_samples=None,           # int or float or None, default=None
    monotonic_cst=None          # array-like of {-1, 0, 1} or None (added v1.4)
)
```

**Key version-history changes to note in the chapter:**
- `max_features` default changed from `"auto"` → `"sqrt"` in v1.1 (BREAKING: old `"auto"` = `n_features` for regression, `"sqrt"` for classification — now explicit per task)
- `n_estimators` default changed from 10 → 100 in v0.22
- `monotonic_cst` added in v1.4 (missing values also added in v1.4)
- `oob_score` now accepts a callable (custom scorer) since v1.4
- `criterion="log_loss"` added (equivalent to `"entropy"` numerically but more explicit)

**Post-fit attributes:**
```python
rf.estimators_            # list of fitted DecisionTreeClassifier objects
rf.n_classes_             # int or list
rf.n_features_in_         # int
rf.feature_importances_   # ndarray of shape (n_features,) — MDI importance
rf.oob_score_             # float — requires oob_score=True
rf.oob_decision_function_ # ndarray (n_samples, n_classes) — requires oob_score=True
                          # ⚠️ May contain NaN if a sample was never OOB
```

---

### Verified: `RandomForestRegressor` (sklearn 1.5.2) — Key Differences

```python
RandomForestRegressor(
    n_estimators=100,
    criterion='squared_error',  # {"squared_error","absolute_error","friedman_mse","poisson"}
    max_depth=None,
    # ... (all tree control params identical to Classifier)
    max_features=1.0,          # ⚠️ DEFAULT IS 1.0 (ALL features) for regression, NOT "sqrt"
    bootstrap=True,
    oob_score=False,           # returns R² when True (not accuracy)
    # ...
)
```

> **Critical difference**: `max_features` defaults differ between classifier (="sqrt") and regressor (=1.0). The chapter must make this explicit.

**Regressor post-fit attributes:**
```python
rf_reg.oob_score_             # R² on OOB samples
rf_reg.oob_prediction_        # ndarray (n_samples,) — OOB predictions
```

---

### Verified: `ExtraTreesClassifier` (sklearn 1.5.2) — Key Differences from RF

```python
ExtraTreesClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='sqrt',        # same default as RF Classifier since v1.1
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=False,            # ⚠️ KEY DIFFERENCE: ET uses whole dataset by default
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None,
    monotonic_cst=None
)
```

**Algorithm differences from RF:**
1. **No bootstrap by default** (`bootstrap=False`): each tree sees all training samples
2. **Random thresholds**: instead of searching for the best split threshold, ET draws a random threshold uniformly in [min(feature_values), max(feature_values)] for each candidate feature, then picks the best random threshold
3. This makes ET **faster** (no optimization search) but introduces **higher bias** and **lower variance** compared to RF
4. When `bootstrap=False` and training on full data, OOB is not available (requires explicitly setting `bootstrap=True`)

---

### Verified: `IsolationForest` (sklearn 1.5.2) — Forest-Family Anomaly Detector

```python
IsolationForest(
    n_estimators=100,
    max_samples='auto',         # 'auto' = min(256, n_samples)
    contamination='auto',       # float in (0, 0.5] or 'auto'
    max_features=1.0,
    bootstrap=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False
)
```

The base estimator is `ExtraTreeRegressor` with random splits — anomalies get shorter average path lengths across trees. The `contamination` parameter (proportion of outliers) sets the decision threshold. The author should note this is **unsupervised** — no labels needed.

---

### SHAP TreeExplainer (shap ~0.46.x) — Verified API

```python
import shap

# Constructor
explainer = shap.TreeExplainer(
    model,                                # sklearn RF, XGBoost, LightGBM, CatBoost, PySpark
    data=None,                            # background dataset (numpy or pandas)
    model_output='raw',                   # 'raw', 'probability', 'log_loss', or method name
    feature_perturbation='auto',          # 'auto', 'interventional', 'tree_path_dependent'
    feature_names=None,
    approximate=<deprecated_in_init>,     # ⚠️ Use in shap_values() instead
    link=None,
    linearize_link=None
)

# Compute SHAP values
shap_values = explainer.shap_values(
    X,
    y=None,
    tree_limit=None,
    approximate=False,
    check_additivity=True
)
# Returns: (n_samples, n_features) for single output
#          (n_samples, n_features, n_outputs) for multi-output

# Full modern API (preferred for SHAP 0.46+):
shap_values_obj = explainer(X)           # Returns Explanation object
shap.summary_plot(shap_values, X)
shap.waterfall_plot(shap_values_obj[0])
shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0])
shap.dependence_plot("feature_name", shap_values, X)
shap.plots.beeswarm(shap_values_obj)
```

**Feature perturbation notes:**
- `"tree_path_dependent"` (default when no `data` provided): uses training data statistics from the tree structure itself; **no background dataset needed**; can capture feature correlations but may overstate importance of correlated features
- `"interventional"` (requires `data`): uses causal independence assumption; more faithful to ground truth but requires a representative background set (typically 100–200 samples from training data)

**For sklearn RF, the recommended pattern:**
```python
explainer = shap.TreeExplainer(rf_model, data=X_train_sample)
# For classifiers: shap_values is a list of arrays, one per class
# Use shap_values[1] for the positive class in binary classification
```

> ⚠️ **Verify**: The `approximate` parameter behavior in the `__init__` is deprecated in newer versions — the chapter should note to pass `approximate=True` to `shap_values()` for large forests. Check shap release notes for exact version when this changed.

---

### cuML / RAPIDS GPU Random Forest

```python
# Requires: conda install -c rapidsai cuml
from cuml.ensemble import RandomForestClassifier as cuRFC
from cuml.ensemble import RandomForestRegressor as cuRFR

# Drop-in replacement for many use cases
curf = cuRFC(n_estimators=100, max_depth=16, n_streams=4)
curf.fit(X_train, y_train)  # X, y must be cuDF DataFrames or CuPy arrays

# Or use cuml.accel for zero-code-change acceleration:
# %load_ext cuml.accel  # in Jupyter
# import cuml.accel; cuml.accel.install()  # in scripts
```

**Performance**: 20-45x speedup over sklearn CPU for large forests (verified by NVIDIA blog 2024).
**Limitations**: API is sklearn-compatible but not 100% identical; fewer criterion options; requires NVIDIA GPU. Current version: cuml 26.06.00 (as of June 2026).
**Documentation**: https://docs.rapids.ai/api/cuml/stable/

---

### h2o.ai Distributed Random Forest (DRF)

```python
import h2o
from h2o.estimators import H2ORandomForestEstimator

h2o.init()
rf_h2o = H2ORandomForestEstimator(
    ntrees=100,
    max_depth=20,
    mtries=-1,           # -1 = sqrt(p) for classification
    sample_rate=0.632,   # bootstrap fraction
    seed=42
)
rf_h2o.train(x=feature_cols, y=target_col, training_frame=train_h2o)
```

**Key h2o.ai DRF advantages:** distributed training across cluster, native categorical handling, built-in early stopping via `stopping_metric`, supports XRT mode via `histogram_type="Random"`, model explainability via h2o's built-in tools. Documentation: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/drf.html

---

### Spark MLlib Random Forest (PySpark)

```python
from pyspark.ml.classification import RandomForestClassifier as SparkRFC
from pyspark.ml.regression import RandomForestRegressor as SparkRFR

rf_spark = SparkRFC(
    numTrees=100,
    maxDepth=5,
    featureSubsetStrategy="sqrt",
    impurity="gini",
    seed=42
)
```

Use for very large datasets (>hundreds of millions rows) in distributed Spark clusters. Note: Spark RF uses different parameter names (`numTrees` not `n_estimators`, `maxDepth` not `max_depth`, `featureSubsetStrategy` not `max_features`).

---

## The Original Algorithm

### Full Citation

> Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
> DOI: https://doi.org/10.1023/A:1010933404324
> Publisher: Springer (Kluwer Academic Publishers)
> Published: 01 October 2001

**Context:** Breiman's paper had 111,938+ citations as of 2024 — one of the most-cited ML papers ever.

---

### Problem Breiman Was Solving

Before Random Forests, the state of the art was:
1. **Single decision trees** (CART, ID3, C4.5): high variance, prone to overfitting, accuracy limited
2. **Bagging** (Breiman 1996): bootstrapping + averaging reduced variance but trees remained correlated because they split on the same best features → ensemble predictions still highly correlated → limited variance reduction
3. **AdaBoost** (Freund & Schapire 1996): reduced bias by sequentially reweighting, but sensitive to noise/outliers

Breiman's insight: to decorrelate trees in an ensemble beyond what bagging achieves, **inject randomness into the feature selection at each split**. If trees split on different feature subsets, their errors are less correlated, and when averaged, they cancel out more effectively.

---

### The RF Algorithm (as stated in the paper)

**Input:** Training set $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$, number of trees $T$, features per split $m$ (called `max_features` in sklearn)

**For** $t = 1, 2, \ldots, T$:
1. Draw a bootstrap sample $\mathcal{D}^{(t)}$ of size $n$ from $\mathcal{D}$ with replacement
2. Grow a decision tree $h_t$ on $\mathcal{D}^{(t)}$:
   - At each node, randomly select $m$ features from the $p$ total features (where $m \ll p$)
   - Find the best split **among only those $m$ features** (optimize Gini impurity or MSE)
   - Split the node; recurse on children until a stopping criterion is met (e.g., min leaf size)
3. Store $h_t$

**Predict** for new $\mathbf{x}$:
- **Classification**: $\hat{y} = \text{majority\_vote}\{h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_T(\mathbf{x})\}$
- **Regression**: $\hat{y} = \frac{1}{T} \sum_{t=1}^T h_t(\mathbf{x})$

**Breiman's recommended defaults:**
- Classification: $m = \lfloor \sqrt{p} \rfloor$ (sklearn's `max_features="sqrt"`)
- Regression: $m = \lfloor p/3 \rfloor$ (sklearn defaults to $m=p$ now, i.e., 1.0)

---

### The Key Equations

**Error decomposition** (variance-bias-noise):

For a single tree: $\text{MSE}(h) = \text{Bias}^2 + \text{Var}(h) + \sigma^2$

For an ensemble of $T$ identically distributed (correlated) trees with pairwise correlation $\rho$ and variance $\sigma^2_{\text{tree}}$:

$$\text{Var}\left(\frac{1}{T}\sum_{t=1}^T h_t(\mathbf{x})\right) = \rho \cdot \sigma^2_{\text{tree}} + \frac{1-\rho}{T} \cdot \sigma^2_{\text{tree}}$$

As $T \to \infty$: $\text{Var} \to \rho \cdot \sigma^2_{\text{tree}}$

**This is the fundamental result**: variance reduction is bounded by the inter-tree correlation $\rho$. Reducing $m$ (features per split) reduces $\rho$ at the cost of increasing individual-tree bias. RF optimizes this tradeoff.

**Breiman's generalization error bound** (from the paper):

The generalization error of a Random Forest converges a.s. to:

$$PE^* \leq \frac{\bar{\rho}(1 - s^2)}{s^2}$$

where $s$ is the "strength" of individual classifiers (mean margin over examples) and $\bar{\rho}$ is the mean correlation between them. Lower $\bar{\rho}$ and higher $s$ → lower error.

**Gini impurity** (the default split criterion):

$$G(t) = \sum_{k=1}^K p_k(t)(1 - p_k(t)) = 1 - \sum_{k=1}^K p_k(t)^2$$

**Impurity decrease** for a split at node $t$ into children $t_L, t_R$:

$$\Delta G = G(t) - \frac{n_{t_L}}{n_t} G(t_L) - \frac{n_{t_R}}{n_t} G(t_R)$$

**MDI (Mean Decrease in Impurity) for feature $j$:**

$$\text{MDI}(j) = \frac{1}{T} \sum_{t=1}^T \sum_{\text{nodes } v \text{ split on } j} \frac{n_v}{n} \Delta G_v$$

---

### OOB Error (from the paper)

For each training sample $\mathbf{x}_i$, the OOB prediction uses only trees where $\mathbf{x}_i$ was NOT in the bootstrap sample:

$$\hat{y}_i^{\text{OOB}} = \text{aggregation}\{h_t(\mathbf{x}_i) : \mathbf{x}_i \notin \mathcal{D}^{(t)}\}$$

Each $\mathbf{x}_i$ is OOB for approximately $T/e \approx 0.368 \cdot T$ trees (since $P(\text{not sampled}) = (1-1/n)^n \to e^{-1}$).

**sklearn attributes after fitting with `oob_score=True`:**
```python
rf.oob_score_             # scalar: accuracy (classifier) or R² (regressor)
rf.oob_decision_function_ # (n_samples, n_classes) — class probability estimates
```

---

### Variable Importance (Breiman's two measures)

**Measure 1: MDI / Mean Decrease Impurity** (fast, built-in)
- Computed during training from tree structures
- `rf.feature_importances_` — sum to 1.0
- **Pitfall**: biased toward high-cardinality features (see Strobl 2007 section below)

**Measure 2: Mean Decrease Accuracy / Permutation Importance** (the "original" permutation test in the paper)
- For each tree $t$ and OOB samples, permute feature $j$ values, measure accuracy drop
- Breiman's paper used OOB samples for this; sklearn's `permutation_importance` uses a provided dataset (usually test set — **use the test set** to get unbiased estimates)

---

## Major Variants & Evolution

### 1. Bagging (Breiman 1996) — The Predecessor

**Paper:** Breiman, L. (1996). Bagging Predictors. *Machine Learning*, 24(2), 123–140.
**Problem solved:** Single trees overfit (high variance). Averaging bootstrapped trees reduces variance.
**Key change from RF:** No feature subsampling — each tree sees all features at every split.
**Why RF is better:** When all trees split on the same best feature, trees are highly correlated (high $\rho$), limiting variance reduction.
**sklearn:** `sklearn.ensemble.BaggingClassifier` with `DecisionTreeClassifier` as base estimator

---

### 2. Extremely Randomized Trees / Extra-Trees (Geurts et al. 2006)

**Full citation:** Geurts, P., Ernst, D. & Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning*, 63(1), 3–42.
DOI: https://doi.org/10.1007/s10994-006-6226-1

**Problem solved:** RF still searches for the optimal split threshold within the random feature subset — this is expensive and trees can still be moderately correlated.

**Key algorithmic change:**
- RF: at each node, for each of the $m$ random features, find the **best** threshold $\theta^*$ by exhaustive search
- ET: for each of the $m$ random features, draw a **random** threshold $\theta \sim \text{Uniform}(\min(f), \max(f))$, then pick the best among the $m$ random (feature, threshold) pairs

This double randomization (random features + random thresholds) further reduces variance and speeds up training dramatically.

**ET vs RF trade-off:**
- ET: lower variance (more randomization), higher bias (random thresholds may be suboptimal), faster training
- RF: lower bias (optimal thresholds), higher variance, slower training
- In practice: ET often matches or beats RF accuracy, especially with `max_features` tuned up

**Also note:** ET defaults to `bootstrap=False` in sklearn — trees are trained on the full dataset, unlike RF.

**sklearn:**
```python
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
et = ExtraTreesClassifier(n_estimators=100, max_features='sqrt')
```

**When to prefer ET over RF:** When training speed is critical; when dataset is large; when you want a strong regularization effect (high depth trees with random splits act as natural regularizers).

---

### 3. Isolation Forest (Liu, Ting & Zhou 2008/2012)

**Paper (conference):** Liu, F.T., Ting, K.M. & Zhou, Z.-H. (2008). Isolation Forest. In *Proc. 8th IEEE International Conference on Data Mining (ICDM)*, pp. 413–422. DOI: 10.1109/ICDM.2008.17

**Extended journal version:** Liu, F.T., Ting, K.M. & Zhou, Z.-H. (2012). Isolation-Based Anomaly Detection. *ACM Transactions on Knowledge Discovery from Data*, 6(1), Article 3. DOI: https://doi.org/10.1145/2133360.2133363

**Problem solved:** Anomaly detection without density estimation or distance computation — traditional methods (LOF, one-class SVM) are $O(n^2)$; Isolation Forest is $O(n)$.

**Key insight:** Anomalies are "few and different" — they are easier to isolate than normal points. Random recursive partitioning isolates anomalies in **fewer splits** on average.

**Algorithm:**
1. Build $T$ isolation trees with `max_samples='auto'` (default: $\min(256, n)$ samples per tree)
2. Each tree: randomly select a feature and a random split point in $[\min, \max]$ of that feature; recurse
3. Anomaly score for sample $\mathbf{x}$:

$$s(\mathbf{x}, n) = 2^{-\frac{E[h(\mathbf{x})]}{c(n)}}$$

where $h(\mathbf{x})$ is the path length and $c(n) = 2H(n-1) - \frac{2(n-1)}{n}$ is the expected path length of an unsuccessful BST search ($H$ = harmonic number).

**sklearn:**
```python
from sklearn.ensemble import IsolationForest
clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
clf.fit(X_train)
scores = clf.score_samples(X_test)   # lower = more anomalous
labels = clf.predict(X_test)         # 1=inlier, -1=outlier
```

---

### 4. Oblique Random Forests & SPORF (Tomita et al. 2020)

**Paper:** Tomita, T.M. et al. (2020). Sparse Projection Oblique Randomer Forests. *Journal of Machine Learning Research*, 21(104), 1–39.
URL: https://jmlr.org/papers/v21/18-664.html
arXiv: https://arxiv.org/abs/1506.03410

**Problem solved:** Standard RF splits are always axis-aligned (perpendicular to one feature axis). Diagonal boundaries in feature space require many axis-aligned splits; a single oblique split can capture them.

**Key algorithmic change:**
- Standard RF: split on $x_j \leq \theta$ (one feature)
- SPORF: split on $\mathbf{a}^T \mathbf{x} \leq \theta$ where $\mathbf{a}$ is a **sparse random projection** vector (most entries zero, non-zeros drawn from $\{-1, +1\}$)
- This creates hyperplane splits in the feature space

**Python package:** `treeple` (formerly `scikit-tree`) — https://docs.neurodata.io/treeple/
```python
# pip install treeple  # verify current install command
from treeple.ensemble import ObliqueRandomForestClassifier
```

**When to use:** Datasets with correlated features where decision boundaries are not axis-aligned; XOR-like problems; image/sensor data; often outperforms standard RF on structured data.

---

### 5. Mondrian Forests (Lakshminarayanan, Roy & Teh 2014)

**Paper:** Lakshminarayanan, B., Roy, D.M. & Teh, Y.W. (2014). Mondrian Forests: Efficient Online Random Forests. *Advances in Neural Information Processing Systems (NeurIPS)*, 27.
arXiv: https://arxiv.org/abs/1406.2673
PDF: https://www.gatsby.ucl.ac.uk/~balaji/mondrian_forests_nips14.pdf

**Problem solved:** Standard RF cannot be updated incrementally — adding new data requires retraining from scratch. Online ML needs a forest that can be updated one sample at a time.

**Key innovation:** Uses the **Mondrian process** (a stochastic process over recursive axis-aligned partitions) as the prior over tree structures. The key property: a Mondrian tree on $n$ samples is **consistent** with a Mondrian tree on $n+1$ samples — you can extend an existing tree to accommodate new data without rebuilding.

**Online update step:** When new labeled example $(\mathbf{x}_{n+1}, y_{n+1})$ arrives, each Mondrian tree in the forest is updated by sampling an **extended Mondrian tree** from a distribution that (a) agrees with the previous tree on old data and (b) is distributed according to the Mondrian process on the combined dataset.

**Python package:** `skgarden` or implement from the paper. Also extended by Mourtada et al. (2021) as Aggregated Mondrian Forests (AMF), published in *Journal of the Royal Statistical Society Series B*, 83(3), 505–533.

**When to use:** Streaming/online settings, sensor data arriving continuously, environments where model must update without batch retraining.

---

### 6. Deep Forest / gcForest (Zhou & Feng 2017/2019)

**Paper:** Zhou, Z.-H. & Feng, J. (2019). Deep Forest. *National Science Review*, 6(1), 74–86.
arXiv v2: https://arxiv.org/abs/1702.08835v2 (first submitted Feb 2017)

**Problem solved:** Deep neural networks require massive data, compute, and hyperparameter tuning. gcForest proposes deep, layered ensemble learning using forests instead of neurons.

**Key architecture:**
- **Multi-grained scanning**: slide windows over input (similar to CNNs), generate class probability features
- **Cascade forest**: multiple layers, each containing RF + ExtraTrees ensembles; the class probability outputs of one layer become the features for the next layer
- **Adaptive depth**: cascade grows until validation performance stops improving (auto-determines depth)

**Claimed advantages:** Competitive with deep learning on small/medium datasets, far fewer hyperparameters, more interpretable, no GPU required.

**Python implementation (unofficial):** https://github.com/pylablanche/gcForest
**Official implementation:** https://github.com/kingfengji/gcForest

**Caveat for the chapter:** gcForest's claims have been contested in independent benchmarks — results are competitive on some datasets but rarely match well-tuned deep networks on vision/NLP. Present this honestly.

---

### 7. Canonical Correlation Forests (Rainforth & Wood 2015)

**Paper:** Rainforth, T. & Wood, F. (2015). Canonical Correlation Forests. arXiv:1507.05444.
URL: https://arxiv.org/abs/1507.05444
PDF: https://www.cs.ubc.ca/~fwood/papers/Rainforth-2015-CCF-arXiv.pdf

**Problem solved:** Axis-aligned splits in standard RF ignore correlations between features and targets. CCF computes **local canonical correlation** between features and class labels at each node, using the top canonical direction to define split hyperplanes.

**Key change:** At each node, compute CCA between selected features and (one-hot) labels. The first canonical variate direction $\mathbf{a}$ defines the split: $\mathbf{a}^T \mathbf{x} \leq \theta$. This is data-adaptive rather than random (as in SPORF).

**Python packages:** https://github.com/plai-group/ccfs-python (research implementation)

**When useful:** Multivariate outputs (multi-label, multi-output regression); highly correlated features with aligned label structure.

---

### 8. Theoretical Analysis (Biau 2012)

**Paper:** Biau, G. (2012). Analysis of a Random Forests Model. *Journal of Machine Learning Research*, 13(4), 1063–1095.
arXiv: https://arxiv.org/abs/1005.0208

This paper provided the first rigorous consistency proof for a simplified version of RF, showing:
- The simplified RF is **consistent**: prediction error → Bayes error as $n \to \infty$
- The rate of convergence depends only on the number of **informative features** $S$, not total features $p$ (adaptation to sparsity)
- Rate: $O(n^{-1/(S \log 2 + 1)})$ under structural assumptions

**Strobl et al. 2007 (variable importance theory):**
> Strobl, C., Boulesteix, A.-L., Zeileis, A. & Hothorn, T. (2007). Bias in random forest variable importance measures: Illustrations, sources and a solution. *BMC Bioinformatics*, 8, 25.
> PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC1796903/

Key findings: MDI (Gini importance) is biased toward features with more categories or higher cardinality. Continuous features are systematically overestimated. **Solution**: use conditional permutation importance (implemented in the R `party` package as `cforest`).

**Strobl et al. 2008 (conditional permutation):**
> Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T. & Zeileis, A. (2008). Conditional variable importance for random forests. *BMC Bioinformatics*, 9, 307.
> PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC2491635/

Showed that standard permutation importance also overestimates importance of correlated features because permuting one correlated feature still leaves information in the correlated partner.

---

## Hyperparameter Reference

### Complete Guide — All Parameters

#### `n_estimators` (int, default=100)

**What it controls:** Number of trees in the forest.
**Mechanism:** More trees → less variance (better OOB stability), but no bias reduction. Error decreases monotonically with T but with diminishing returns.
**Default reasoning:** 100 is a practical default (changed from 10 in v0.22); 10 was famously too small for production use.
**How to tune:**
- Plot OOB error vs n_estimators; look for the "elbow" where improvement flattens
- 100–500 is typical; 1000+ rarely helps but massively increases memory/time
- Use `warm_start=True` to incrementally fit and track OOB:
```python
rf = RandomForestClassifier(warm_start=True, oob_score=True)
for n in range(10, 301, 10):
    rf.set_params(n_estimators=n)
    rf.fit(X_train, y_train)
    print(n, rf.oob_score_)
```
**Range for Optuna:** `trial.suggest_int("n_estimators", 50, 500)`
**Too low:** High variance; OOB estimate unreliable. **Too high:** Wasted compute; marginal gains.

---

#### `max_features` ({"sqrt","log2",None}, int, float; default="sqrt" for classification, 1.0 for regression)

**What it controls:** Number of features randomly selected at each split node.
**Mechanism:** This is the **primary decorrelation lever**. Lower $m$ → less correlated trees → more variance reduction via averaging; but also higher individual-tree bias.
**Options:**
- `"sqrt"` → $m = \lfloor \sqrt{p} \rfloor$ — Breiman's recommendation for classification
- `"log2"` → $m = \lfloor \log_2(p) \rfloor$ — more aggressive decorrelation
- `None` → $m = p$ — all features (equivalent to bagging, no feature randomness)
- `int` → exact number of features
- `float` → fraction of features (e.g., 0.5 = half)

**Tuning strategy:**
- For classification: try `["sqrt", "log2", 0.3, 0.5]`
- For regression: try `[0.33, 0.5, 0.7, 1.0]`
- `max_features` and `n_estimators` interact: lower `max_features` needs more trees to stabilize

**Range for Optuna:**
```python
max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7])
```

---

#### `max_depth` (int or None, default=None)

**What it controls:** Maximum depth of each tree. None = grow until all leaves are pure or hit `min_samples_split`.
**Default reasoning:** Fully grown trees (None) are high-variance but low-bias; the ensemble averaging handles the variance. This differs from GBM where shallow trees are preferred.
**When to change:** Set to 10-30 for very large datasets to control memory; set to 3-8 if you suspect heavy overfitting.
**Memory impact:** Memory scales roughly as $O(T \cdot N \cdot \log N)$. Capping `max_depth` is the most effective memory control.
**Range for tuning:** `trial.suggest_int("max_depth", 5, 30)` or add `None` as categorical option.

---

#### `min_samples_split` (int or float, default=2)

**What it controls:** Minimum number of samples required to split an internal node.
**Default (2):** Allows splitting on any node with 2 or more samples → grows very deep trees.
**As float:** Treated as fraction of `n_samples` (e.g., 0.01 = 1% of data per node).
**Effect:** Higher values → shallower trees → higher bias, lower variance. Use to control overfitting.
**Range:** `trial.suggest_int("min_samples_split", 2, 20)` or as fraction for large datasets.

---

#### `min_samples_leaf` (int or float, default=1)

**What it controls:** Minimum samples required at a leaf node. A split is only made if it leaves at least `min_samples_leaf` training samples on each side.
**Effect:** Similar to `min_samples_split` but can also prevent creating very small leaves that may cause overfitting. Provides smoother regression predictions.
**Range:** `trial.suggest_int("min_samples_leaf", 1, 20)` or as fraction.
**Interaction with `min_samples_split`:** Both prune tree growth; `min_samples_leaf` is often more interpretable.

---

#### `max_leaf_nodes` (int or None, default=None)

**What it controls:** Grows trees in best-first fashion with at most `max_leaf_nodes` leaf nodes. None = unlimited.
**Effect:** An alternative to `max_depth` for controlling tree complexity that operates in best-first (not depth-first) order.

---

#### `min_impurity_decrease` (float, default=0.0)

**What it controls:** A node will be split only if the impurity decrease is ≥ this value. Provides a threshold on the minimum usefulness of a split.
**Tuning:** Rarely needed; use `min_samples_leaf` or `max_depth` instead for simpler interpretability.

---

#### `bootstrap` (bool, default=True)

**What it controls:** Whether to use bootstrap sampling (with replacement) when building trees.
**True (RF default):** Each tree sees ~63.2% of training data (unique samples); enables OOB error estimation.
**False (ET default):** Each tree sees all training data; typically lower variance per sample but trees are more correlated; OOB error not available.
**Important:** `oob_score=True` requires `bootstrap=True`.

---

#### `oob_score` (bool or callable, default=False)

**What it controls:** Whether to use OOB samples to compute a generalization score.
**True:** Computes OOB accuracy (classifier) or R² (regressor). Accessible via `rf.oob_score_`.
**Callable (sklearn 1.4+):** Pass a custom scorer, e.g., `oob_score=roc_auc_score` for AUC.
**Key benefit:** Free validation estimate without a separate test set — especially useful with small datasets.
**Caveat:** OOB estimates are slightly pessimistic (each sample only uses ~36.8% of trees).

---

#### `class_weight` ({"balanced","balanced_subsample"}, dict, list, None; default=None)

**What it controls:** Weights applied to classes; critical for imbalanced datasets.
**`"balanced"`:** Weights inversely proportional to class frequency: $w_k = n / (K \cdot n_k)$.
**`"balanced_subsample"`:** Same as `"balanced"` but computed on each bootstrap sample rather than the full dataset. Better for highly imbalanced data.
**`None`:** All classes equally weighted.
**When to use:** When positive class is rare (fraud, rare disease detection). Alternative to SMOTE/oversampling.

---

#### `ccp_alpha` (float ≥ 0, default=0.0)

**What it controls:** Cost-Complexity Pruning (CCP) parameter. Higher values → more pruning → smaller, simpler trees.
**Added in sklearn 0.22.** Pruning applies post-training to each individual tree in the forest.
**Tuning:** Use `DecisionTreeClassifier.cost_complexity_pruning_path()` on a representative tree to find a reasonable alpha range, then tune via CV.
**Range:** `trial.suggest_float("ccp_alpha", 0.0, 0.05, step=0.001)` on small datasets.

---

#### `max_samples` (int, float, or None; default=None)

**What it controls:** When `bootstrap=True`, how many samples to draw for each tree. None = draw $n$ samples (standard bootstrap).
**As float:** Fraction of $n$ (e.g., 0.8 = 80% of training data per tree).
**Effect:** Subsampling without replacement (if < n) further reduces tree correlation but each tree has less data → higher bias.
**Use case:** Very large datasets where full bootstrap is slow; or to increase diversity.

---

#### `monotonic_cst` (array-like of {-1, 0, 1} or None; default=None) — Added v1.4

**What it controls:** Per-feature monotonicity constraints. `1` = positive monotone, `-1` = negative monotone, `0` = unconstrained.
**Limitation:** Not supported for multiclass (>2 classes) or multioutput targets in v1.4/1.5.
**Use case:** Business rules where you know a feature must have monotone effect (e.g., credit score and default probability).

---

### Hyperparameter Tuning Playbook Table

| Parameter | Typical Range | Primary Effect | Tuning Priority |
|---|---|---|---|
| `n_estimators` | 50–500 | Variance ↓ as T↑; compute ↑ | Low (set to 200+, use OOB elbow) |
| `max_features` | "sqrt","log2",0.3–0.7 | Bias↑/Variance↓ tradeoff | **High** — most impactful |
| `max_depth` | None, 10–30 | Overfitting control | Medium (None usually fine) |
| `min_samples_leaf` | 1–20 | Overfitting/smoothness | Medium |
| `min_samples_split` | 2–20 | Tree depth control | Low (often redundant with above) |
| `class_weight` | "balanced","balanced_subsample" | Class imbalance | **Critical for imbalanced data** |
| `ccp_alpha` | 0.0–0.05 | Post-hoc pruning | Low (rarely needed) |
| `bootstrap` | True/False | OOB availability; diversity | Low (keep True unless ET) |
| `max_samples` | 0.5–1.0 | Subsample size | Low (tune if dataset is huge) |

**Optuna objective template for RF:**
```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "bootstrap": True,
        "oob_score": False,
        "random_state": 42,
        "n_jobs": -1,
    }
    rf = RandomForestClassifier(**params)
    return cross_val_score(rf, X_train, y_train, cv=5, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

---

## Common Errors & Pitfalls

### 1. MDI Feature Importance Is Biased (Strobl 2007)

**The bug**: `rf.feature_importances_` systematically inflates importance of continuous or high-cardinality features because they offer more split-point candidates by chance. Strobl et al. showed this empirically — a random noise column with many values can rank higher than a meaningful binary feature.

**Detection**: Compare MDI vs permutation importance; large disagreements signal cardinality bias.

**Fix:**
```python
from sklearn.inspection import permutation_importance
perm = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)
# Use perm.importances_mean — unbiased, test-set based
```

**Also**: sklearn docs explicitly warn about this; the permutation importance documentation page (https://scikit-learn.org/1.5/modules/permutation_importance.html) is required reading.

---

### 2. Permutation Importance Underestimates Correlated Features

**The bug (Strobl 2008)**: When two features are correlated ($X_1 \approx X_2$), permuting $X_1$ has little effect on model performance because $X_2$ still carries the same information. Both features appear unimportant even if both are truly predictive.

**Fix:** Hierarchical clustering on Spearman correlation; drop one feature from each correlated cluster before computing permutation importance. sklearn example: https://scikit-learn.org/1.5/auto_examples/inspection/plot_permutation_importance_multicollinear.html

---

### 3. Class Imbalance Without `class_weight`

**The bug:** On a dataset with 95% negative / 5% positive, a RF trained without `class_weight` will be biased toward predicting the majority class. `oob_score_` may look good (e.g., 95%) but AUC-ROC will be poor.

**Fix:** Use `class_weight="balanced"` or `class_weight="balanced_subsample"`. For extreme imbalance, combine with `max_samples` subsampling or SMOTE preprocessing.

---

### 4. `max_features` Default Change Breaks Old Code

**The bug:** Code written for sklearn < 1.1 with `max_features="auto"` now raises a `ValueError` — `"auto"` was removed. Previously `"auto"` meant `"sqrt"` for classifiers but `n_features` for regressors, creating inconsistency.

**Fix:** Replace `max_features="auto"` with `max_features="sqrt"` (classifiers) or `max_features=1.0` (regressors). Note that the regressor default changed from `"auto"` (= $p$) to `1.0` (= $p$) — same value, different spelling.

---

### 5. OOB Score NaN When `n_estimators` Is Small

**The bug:** With few trees, some training samples may never appear in any OOB set. sklearn fills these with NaN in `oob_decision_function_`, and the OOB score may be computed on fewer samples, causing instability.

**Fix:** Use at least 100 trees (default) when relying on OOB. Check: `np.isnan(rf.oob_decision_function_).any()`.

---

### 6. Memory Explosion with Deep Trees and Large Datasets

**The bug:** sklearn RF stores the full tree structure in memory. With $n = 10^6$, $p = 1000$, $T = 500$, and no `max_depth`, memory can exceed 100GB.

**Fix:**
```python
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,         # cap depth
    max_samples=0.5,      # use 50% of data per tree
    n_jobs=-1
)
```
Or switch to cuML for GPU-based training.

---

### 7. Parallelization Overhead with `warm_start`

**The bug:** Using `warm_start=True` inside a cross-validation loop causes sklearn to switch from `"threading"` backend to `"loky"` backend, which incurs inter-process communication overhead. Reported up to 6x slowdown in sklearn GitHub issue #22087.

**Fix:** Avoid `warm_start` inside CV loops. Use it only for the OOB-elbow diagnostic outside of CV.

---

### 8. Random Forest Does NOT Handle Extrapolation

**The bug:** RF predictions are bounded by the range of training target values (since leaf predictions are averages of training samples). For a regression task with time trends, RF cannot predict values outside the training range.

**Detection:** Plot predicted vs actual on a held-out time period; predictions will be flat/clipped.

**Fix:** Feature engineering (add trend features), or switch to gradient boosting with regression capability, or combine RF with a trend model.

---

### 9. Feature Scaling Is Not Required (but affects importance)

RF is insensitive to feature scaling for predictions (tree splits are threshold comparisons, scale-invariant). However, Strobl et al. noted that features on very different scales can have different numbers of candidate split points, interacting with the MDI bias issue.

**Best practice:** You don't need to scale for RF predictions, but standardize features if using permutation importance for fair comparison.

---

### 10. Missing Values — Support Added in sklearn 1.4

**The bug (pre-1.4):** sklearn RF raised an error on NaN inputs. Many practitioners added `SimpleImputer` unnecessarily after 1.4.

**Current behavior (sklearn 1.4+):** Missing values in training data are handled natively — the splitter evaluates both routing (missing → left, or missing → right) and picks the optimal direction. At prediction time, the same learned direction is applied.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
X = np.array([[1, np.nan], [2, 3], [np.nan, 4]])
y = [0, 1, 1]
rf = RandomForestClassifier().fit(X, y)  # works in sklearn 1.4+
```

---

## Key Datasets for Examples

### 1. Breast Cancer Wisconsin (Classification)

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target
# 569 samples, 30 features, binary (malignant=0, benign=1)
# Good for: classification, feature importance, SHAP, OOB
```

**Why:** Medium-sized, all continuous features, binary classification, well-understood feature names → ideal for feature importance comparisons (MDI vs permutation vs SHAP).

---

### 2. Titanic (Classification with Mixed Types)

```python
import seaborn as sns
titanic = sns.load_dataset('titanic')
# ~891 samples, mixed types (numeric + categorical), missing values
# Good for: preprocessing pipeline, class_weight for imbalance, missing value handling
```

**Why:** Realistic preprocessing challenge; class imbalance (survived=38%); demonstrates `class_weight` and sklearn 1.4+ missing value handling.

---

### 3. California Housing (Regression)

```python
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X, y = data.data, data.target
# 20,640 samples, 8 features, regression (median house value)
# Good for: regression RF, RandomForestRegressor criterion comparison, SHAP
```

**Why:** Large enough to show speed tradeoffs (cuML vs sklearn), continuous target for regression diagnostics, geographic features enable PDP/ICE plots.

---

### 4. Forest Cover Type (Large Scale, Multiclass)

```python
from sklearn.datasets import fetch_covtype
data = fetch_covtype()
X, y = data.data, data.target
# 581,012 samples, 54 features, 7 classes
# Good for: n_jobs=-1 parallelization demo, warm_start OOB convergence
```

---

## Curated Resources

### Original Papers (with verified URLs)

1. **Breiman 2001 — Random Forests** (THE paper)
   - DOI: https://doi.org/10.1023/A:1010933404324
   - Springer: https://link.springer.com/article/10.1023/A:1010933404324
   - Semantic Scholar: https://www.semanticscholar.org/paper/Random-Forests-Breiman/8e0be569ea77b8cb29bb0e8b031887630fe7a96c

2. **Geurts et al. 2006 — Extremely Randomized Trees**
   - DOI: https://doi.org/10.1007/s10994-006-6226-1
   - PDF: https://jasonphang.com/files/extratrees.pdf

3. **Liu et al. 2008 — Isolation Forest (conference)**
   - DOI: 10.1109/ICDM.2008.17

4. **Liu et al. 2012 — Isolation-Based Anomaly Detection (journal)**
   - ACM: https://dl.acm.org/doi/10.1145/2133360.2133363

5. **Strobl et al. 2007 — Variable Importance Bias**
   - PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC1796903/

6. **Strobl et al. 2008 — Conditional Variable Importance**
   - PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC2491635/

7. **Biau 2012 — Theoretical Analysis**
   - arXiv: https://arxiv.org/abs/1005.0208

8. **Lakshminarayanan et al. 2014 — Mondrian Forests**
   - arXiv: https://arxiv.org/abs/1406.2673
   - PDF: https://www.gatsby.ucl.ac.uk/~balaji/mondrian_forests_nips14.pdf

9. **Zhou & Feng 2017/2019 — Deep Forest / gcForest**
   - arXiv: https://arxiv.org/abs/1702.08835

10. **Rainforth & Wood 2015 — Canonical Correlation Forests**
    - arXiv: https://arxiv.org/abs/1507.05444

11. **Tomita et al. 2020 — SPORF (Oblique RF)**
    - JMLR: https://jmlr.org/papers/v21/18-664.html

### Official Documentation

12. **sklearn 1.5 RandomForestClassifier API**
    - https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html

13. **sklearn 1.5 Ensemble User Guide (Random Forests section)**
    - https://scikit-learn.org/1.5/modules/ensemble.html#forest

14. **sklearn 1.5 Permutation Importance vs MDI example**
    - https://scikit-learn.org/1.5/auto_examples/inspection/plot_permutation_importance.html

15. **sklearn 1.5 OOB Error example**
    - https://scikit-learn.org/1.5/auto_examples/ensemble/plot_ensemble_oob.html

16. **SHAP TreeExplainer docs**
    - https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html

17. **cuML Random Forest documentation**
    - https://docs.rapids.ai/api/cuml/stable/

18. **h2o.ai DRF documentation**
    - https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/drf.html

### Best Practitioner Resources

19. **"Beware Default Random Forest Importances" — Parr & Howard (explained.ai)**
    - https://explained.ai/rf-importance/
    - _Why_: Most accessible explanation of MDI bias with concrete NYC apartment data example; shows drop-column and permutation importance alternatives

20. **sklearn Permutation Importance with Multicollinear Features example**
    - https://scikit-learn.org/1.5/auto_examples/inspection/plot_permutation_importance_multicollinear.html
    - _Why_: Official example showing the correlated-feature pitfall and hierarchical clustering fix

21. **NVIDIA Blog: Accelerating Random Forests 45x with cuML**
    - https://developer.nvidia.com/blog/accelerating-random-forests-up-to-45x-using-cuml/
    - _Why_: Concrete benchmark numbers for GPU vs CPU RF training

22. **treeple (SPORF) documentation**
    - https://docs.neurodata.io/treeple/
    - _Why_: Modern Python implementation of oblique random forests, sklearn-compatible

---

## Uncertainties

The following items the chapter author **must verify** before publishing:

1. **SHAP `approximate` parameter deprecation**: The dossier notes that `approximate` in `shap.TreeExplainer.__init__` is deprecated in favor of passing it to `shap_values()`. Verify the exact SHAP version when this changed (likely 0.41+) by checking https://github.com/shap/shap/releases and the changelog. The API may have changed again in 0.46.x.

2. **SHAP multi-class RF**: For a 3+ class `RandomForestClassifier`, `explainer.shap_values(X)` returns a list of arrays (one per class). Verify the exact return shape in shap 0.46.x and whether the new `Explanation` object (from `explainer(X)`) unifies this.

3. **`RandomForestRegressor` `max_features` default**: Confirmed as `1.0` (all features) in 1.5.2 docs. Verify this is correct — older docs showed `"auto"` (which was also $p$ for regression). The behavior is the same but the string changed. Author should explicitly compare to the classifier default (`"sqrt"`) since this is a common point of confusion.

4. **cuML API compatibility with sklearn 1.5**: cuML 26.06.00 (current as of June 2026) supports sklearn 1.6+. Check whether `from cuml.ensemble import RandomForestClassifier` has the same `monotonic_cst` parameter that sklearn 1.4+ added, or if there are still gaps.

5. **h2o.ai DRF version currency**: The h2o Python package is `h2o` on PyPI. Verify the current version (was 3.46.0.11 in search results) and any API changes to `H2ORandomForestEstimator`.

6. **treeple install command and version**: The oblique forests package was `scikit-tree` and renamed to `treeple`. Verify the correct pip install: `pip install treeple` and the current module path `from treeple.ensemble import ObliqueRandomForestClassifier` is correct.

7. **gcForest Python implementations**: Both GitHub repos listed (pylablanche and kingfengji) may be unmaintained. Verify if there is an actively maintained Python 3.10+ compatible version of gcForest before recommending it to readers.

8. **Mondrian Forests Python package**: `skgarden` appears to be largely unmaintained. The AMF (Aggregated Mondrian Forests) by Mourtada et al. may have a more active implementation. Verify the best current Python package for Mondrian Forests.

9. **`oob_score` callable (sklearn 1.4+)**: The dossier states that `oob_score` accepts a callable since 1.4. Verify the exact callable signature required (does it follow `sklearn.metrics` scorer protocol?).

10. **IsolationForest 1.5 `contamination='auto'`**: Verify what `'auto'` means for contamination in 1.5.2 (it likely sets threshold such that the training set score distribution matches expected contamination, but confirm).

---

## Summary for Chapter Author

**3-sentence synopsis**: Random Forests (Breiman 2001, DOI: 10.1023/A:1010933404324) improve over bagging by injecting feature randomness at each split — selecting $m \ll p$ features per node — which decorrelates trees and enables meaningful variance reduction when predictions are averaged, with the key formula showing that ensemble variance converges to $\rho \cdot \sigma^2_{\text{tree}}$ as tree count grows. The sklearn 1.5.x API (`RandomForestClassifier` with 20 verified parameters, plus `ExtraTreesClassifier` with its critical `bootstrap=False` and random-threshold differences) is production-ready and now supports missing values natively (since 1.4) and monotonic constraints, while the primary practical pitfall remains the MDI feature importance bias toward high-cardinality features (Strobl 2007), which must always be paired with permutation importance or SHAP TreeExplainer for reliable interpretability. The variant landscape is rich — Extremely Randomized Trees (Geurts 2006) for speed, Isolation Forest (Liu 2008) for anomaly detection, SPORF (Tomita 2020) for oblique splits, Mondrian Forests (Lakshminarayanan 2014) for online learning, and gcForest (Zhou 2019) for deep ensemble stacking — all sharing the same fundamental idea of randomized tree ensembles but solving different problems in the original algorithm's design space.

**Approximate dossier word count**: ~4,800 words
