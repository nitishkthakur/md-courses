# Random Forests: The Complete Masterclass

> **Why this algorithm matters:** Random Forests are the closest thing machine learning has to
> a universal first tool. They win Kaggle competitions, power credit-risk models at JPMorgan,
> diagnose disease at the NIH, and detect fraud at PayPal — all without requiring feature scaling,
> all while handling mixed feature types, missing values, and nonlinear interactions out of the box.
> Breiman's 2001 paper has been cited over 111,000 times, making it one of the most influential
> machine learning publications in history. Yet despite their ubiquity, most practitioners use
> Random Forests as a black box, missing the deep mathematical machinery that makes them work —
> and the precise conditions under which they fail. This chapter changes that.

---

## §0 — Python Ecosystem & Package Guide

Random Forests are implemented in virtually every serious ML library. The choice of package is
not cosmetic: it determines training speed, memory usage, distributed scale, GPU acceleration,
and which hyperparameters you can control. Here is the complete map.

### Complete Package Table

| Package | Import | Version (mid-2026) | Algorithm variant | License |
|---|---|---|---|---|
| `scikit-learn` | `from sklearn.ensemble import RandomForestClassifier` | ~1.5.x | Breiman RF + ExtraTrees + IsolationForest | BSD-3 |
| `cuML / RAPIDS` | `from cuml.ensemble import RandomForestClassifier` | ~26.06.x | GPU RF (CUDA), sklearn-compatible | Apache-2 |
| `h2o` | `from h2o.estimators import H2ORandomForestEstimator` | ~3.46.x | Distributed RF (JVM), native categoricals | Apache-2 |
| `PySpark MLlib` | `from pyspark.ml.classification import RandomForestClassifier` | ~3.5.x | Distributed RF on Spark clusters | Apache-2 |
| `treeple` | `from treeple.ensemble import ObliqueRandomForestClassifier` | ~0.9.x | Oblique/SPORF — hyperplane splits | MIT |
| `shap` | `import shap` (TreeExplainer) | ~0.46.x | Fast exact SHAP for tree ensembles | MIT |
| `optuna` | `import optuna` | ~3.x | Hyperparameter optimization | MIT |
| `river` | `from river.ensemble import AdaptiveRandomForestClassifier` | ~0.21.x | Online/streaming RF (Adaptive RF) | BSD-3 |
| `sklearn (IsolationForest)` | `from sklearn.ensemble import IsolationForest` | ~1.5.x | Anomaly detection via path length | BSD-3 |
| `sklearn (ExtraTrees)` | `from sklearn.ensemble import ExtraTreesClassifier` | ~1.5.x | Extremely Randomized Trees | BSD-3 |

### Top Picks — Recommendation Table

| Package | Best for | Modern/Legacy | Notes |
|---|---|---|---|
| **`sklearn.RandomForestClassifier/Regressor`** | General use, prototyping, production pipelines | Modern | **Start here for 95% of tasks.** Pipeline-compatible, excellent docs, native missing values (v1.4+) |
| **`cuML RandomForestClassifier`** | Large datasets (>100k samples) with NVIDIA GPU | Modern | 20–45x speedup over CPU sklearn; use `cuml.accel` for zero-code-change acceleration |
| **`h2o H2ORandomForestEstimator`** | Distributed training, Spark/Hadoop clusters, AutoML | Modern | Built-in early stopping, native categorical, distributed across cluster |
| **`treeple ObliqueRandomForestClassifier`** | Correlated features, diagonal decision boundaries | Modern | SPORF algorithm; often beats standard RF on structured data |
| **`river AdaptiveRandomForestClassifier`** | Streaming/online learning, concept drift | Modern | Online RF that adapts to distribution shifts; true one-pass learning |

### The Same Fit Across Top Packages

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Option 1: scikit-learn (the workhorse) ────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier

rf_sklearn = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",   # Breiman's recommendation for classification
    random_state=42,
    n_jobs=-1,
)
rf_sklearn.fit(X_train, y_train)
print(f"sklearn RF accuracy: {rf_sklearn.score(X_test, y_test):.4f}")

# ── Option 2: cuML / RAPIDS (GPU) ────────────────────────────────────────────
# from cuml.ensemble import RandomForestClassifier as cuRFC
# import cudf
# X_train_gpu = cudf.DataFrame(X_train)
# y_train_gpu = cudf.Series(y_train)
# rf_gpu = cuRFC(n_estimators=200, max_features="sqrt", random_state=42)
# rf_gpu.fit(X_train_gpu, y_train_gpu)
# (Commented: requires NVIDIA GPU + RAPIDS install)

# ── Option 3: h2o (distributed) ──────────────────────────────────────────────
# import h2o
# from h2o.estimators import H2ORandomForestEstimator
# h2o.init()
# train_h2o = h2o.H2OFrame(np.column_stack([X_train, y_train]))
# rf_h2o = H2ORandomForestEstimator(ntrees=200, mtries=-1, seed=42)
# rf_h2o.train(x=list(range(X_train.shape[1])), y=X_train.shape[1], training_frame=train_h2o)

# ── Option 4: ExtraTrees (ET variant, often faster) ───────────────────────────
from sklearn.ensemble import ExtraTreesClassifier

et_sklearn = ExtraTreesClassifier(
    n_estimators=200,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
et_sklearn.fit(X_train, y_train)
print(f"ExtraTrees accuracy: {et_sklearn.score(X_test, y_test):.4f}")
```

> 🔧 **In practice:** For any new project on a CPU with up to ~500k rows, start with sklearn's
> `RandomForestClassifier`. If training takes more than a few minutes and you have an NVIDIA GPU,
> switch to cuML — the API is nearly identical. For data that lives in a Spark cluster, use h2o or
> Spark MLlib directly rather than pulling data to a single node.

> ⚠️ **Pitfall:** The `max_features` default changed in sklearn 1.1. Code using
> `max_features="auto"` will raise a `ValueError` in 1.1+. Replace with `"sqrt"` for
> classifiers and `1.0` for regressors. The regressor default is also `1.0` (all features),
> not `"sqrt"` — a critical difference that often surprises people switching between task types.

---

## §1 — The Origin: Breiman (2001) and the Paper That Started It All

### Full Citation

> Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
> DOI: https://doi.org/10.1023/A:1010933404324
> Published: October 2001, Springer/Kluwer Academic Publishers.
> Citations as of 2024: 111,938+ (one of the most-cited ML papers in history)

### The State of the Art in 2001: What Was Broken

Imagine the year is 2000. You are a statistician at Berkeley. You have a dataset, a CART decision
tree, and the gnawing knowledge that decision trees are fundamentally unstable: change a handful
of training samples, and the entire tree topology can change. The high variance of individual
trees makes them unreliable. The community had two known partial solutions, both with serious
limitations.

**Solution 1: Bagging** (Breiman himself, 1996). The idea: train $T$ trees on bootstrap samples
of the training data, then average their predictions. Because each tree sees a slightly different
data perturbation, their errors are somewhat decorrelated, and averaging reduces variance.
Mathematically clean. But there was a fundamental ceiling: when you train $T$ trees all using the
same feature set, the best split at the root of every tree tends to be the same powerful feature.
All trees split on the same dominant features → their predictions are highly correlated →
the variance reduction from averaging is severely limited. You have many trees that essentially
agree, not a diverse committee.

**Solution 2: AdaBoost** (Freund & Schapire, 1996). Sequential reweighting of training samples
to focus attention on hard examples. Reduces bias impressively. But AdaBoost is fragile in the
presence of noisy labels — extreme sample weights on noisy examples cause catastrophic overfitting.

Neither solution had cracked the fundamental problem: **how do you build a committee of trees
where members are genuinely different from one another, not just trained on slightly different
subsets of the same data?**

### Breiman's Core Insight: Inject Randomness Into Feature Selection

Breiman's answer was elegant and algorithmic, not statistical: at every node of every tree, before
finding the best split, **randomly restrict the features available to split on**. Instead of
searching over all $p$ features, draw a random subset of $m \ll p$ features, and only consider
splits on those $m$ features.

This one change, combined with bootstrap sampling, has a profound effect: trees that might all
have split on feature $X_3$ at the root (because it's the most globally predictive feature) now
often cannot — feature $X_3$ may not be in the random subset drawn at that node. Different trees
develop different structures from the very first split. Their prediction errors are decorrelated
not just by data perturbation (bagging) but by **structural diversity**.

The mathematics of why this works is captured in what I'll call the **fundamental RF variance
equation**, which Breiman derived in the paper and which we will dissect in §2.

### The Original Algorithm, Formally Stated

> 📜 **Origin/Citation:** The following pseudocode is faithful to Breiman (2001), Section 2, with
> sklearn parameter names added for practical reference.

**Input:**
- Training set $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$, where $\mathbf{x}_i \in \mathbb{R}^p$
- Number of trees: $T$ (sklearn: `n_estimators`)
- Features per split: $m$ (sklearn: `max_features`), with $m \ll p$

**Training (for each tree $t = 1, \ldots, T$):**

1. Draw a bootstrap sample $\mathcal{D}^{(t)}$ of size $n$ **with replacement** from $\mathcal{D}$.
   Approximately 63.2% of original samples appear at least once; the rest are "out-of-bag" (OOB).

2. Grow a decision tree $h_t$ on $\mathcal{D}^{(t)}$ using the following modified split rule:
   - At each internal node $v$ with sample set $S_v$:
     - **Randomly sample** $m$ features (without replacement) from $\{1, \ldots, p\}$
     - **Search exhaustively** for the best split $(j^*, \theta^*)$ among those $m$ features:
       $$j^*, \theta^* = \arg\min_{j \in \text{random subset}, \theta} \left[ \frac{|S_L|}{|S_v|} \text{Impurity}(S_L) + \frac{|S_R|}{|S_v|} \text{Impurity}(S_R) \right]$$
     - Split node $v$ into left child $S_L = \{i \in S_v : x_{ij^*} \leq \theta^*\}$ and right child $S_R$
   - Recurse until leaves contain fewer than `min_samples_split` samples or other stopping criterion

3. Store $h_t$ (the fitted tree)

**Prediction for new $\mathbf{x}$:**

- **Classification (majority vote):**
  $$\hat{y} = \arg\max_k \sum_{t=1}^T \mathbf{1}[h_t(\mathbf{x}) = k]$$

- **Soft probability (class fraction):**
  $$\hat{P}(Y=k \mid \mathbf{x}) = \frac{1}{T} \sum_{t=1}^T \hat{p}_t(k \mid \mathbf{x})$$
  where $\hat{p}_t(k \mid \mathbf{x})$ is the fraction of class $k$ in the leaf that $\mathbf{x}$ lands in for tree $t$.

- **Regression (average):**
  $$\hat{y} = \frac{1}{T} \sum_{t=1}^T h_t(\mathbf{x})$$

**Breiman's recommended defaults:**
- Classification: $m = \lfloor \sqrt{p} \rfloor$ (sklearn's `max_features="sqrt"`)
- Regression: $m = \lfloor p/3 \rfloor$ (sklearn's modern default is `1.0` = all features — a more conservative choice that we will critique in §4)

### Out-of-Bag Error: A Free Cross-Validation

One of the most practically valuable by-products of bootstrap sampling is the OOB error estimate.
Since each tree $h_t$ is trained on only ~63.2% of the data, the remaining ~36.8% are genuinely
unseen samples for that tree. Breiman noticed that you can use these leftover samples as a free
validation set.

For each training sample $\mathbf{x}_i$, define its **OOB prediction** as the aggregated
prediction from only those trees that did NOT use $\mathbf{x}_i$ in their bootstrap sample:

$$\hat{y}_i^{\text{OOB}} = \text{aggregate}\{h_t(\mathbf{x}_i) : i \notin \mathcal{D}^{(t)}\}$$

The fraction of training samples that are OOB for each tree is approximately $e^{-1} \approx 0.368$,
since $P(\text{sample } i \text{ not selected in a bootstrap of size } n) = \left(1 - \frac{1}{n}\right)^n \to e^{-1}$.

Breiman showed empirically that the OOB error estimate is essentially unbiased as an estimate of
the true generalization error, and often converges to the same value as a proper test set. This
matters enormously for small datasets where you cannot afford to hold out a separate validation set.

### The Generalization Error Bound

Breiman derived a theoretical upper bound on the generalization error (for classification) based
on two quantities:

$$PE^* \leq \frac{\bar{\rho}(1 - s^2)}{s^2}$$

where:
- $s$ = the **strength** of the ensemble: the mean over training examples of the margin $\mathbb{E}_\mathbf{x}[P(h(\mathbf{x}) = Y) - \max_{j \neq Y} P(h(\mathbf{x}) = j)]$. Think of this as the average confidence gap of correct predictions over wrong ones.
- $\bar{\rho}$ = the **mean pairwise correlation** between individual trees (computed via their raw margin functions).

This bound reveals the two levers of a Random Forest:
- **Increase $s$**: improve individual tree accuracy (lower `max_features` is NOT the way; deeper trees or better features help)
- **Decrease $\bar{\rho}$**: increase tree diversity (lower `max_features` does help here)

The art of tuning a Random Forest is navigating this tension. We quantify it precisely in §2.

### What Was Uncertain When Published

Breiman was working empirically as much as theoretically. The 2001 paper:
- Lacked a rigorous consistency proof (that came with Biau 2012)
- Used Gini impurity without a theoretical justification for why it outperforms entropy (it often does not)
- Did not analyze the feature importance bias problem that Strobl et al. discovered in 2007
- Had informal arguments about OOB error convergence rather than formal proofs

The algorithm worked spectacularly well in practice, and Breiman knew it. The theoretical
justification caught up over the following decade.

---

## §2 — The Algorithm, Deeply Explained

### The Bias-Variance Decomposition of an Ensemble

To understand why Random Forests work, we need to be precise about what averaging trees actually
accomplishes. Start with the mean squared error of a single regression tree $h(\mathbf{x})$:

$$\text{MSE}[h(\mathbf{x})] = \underbrace{(\mathbb{E}[h(\mathbf{x})] - f(\mathbf{x}))^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(h(\mathbf{x}) - \mathbb{E}[h(\mathbf{x})])^2]}_{\text{Variance}} + \sigma^2_\epsilon$$

where $f(\mathbf{x})$ is the true function and $\sigma^2_\epsilon$ is irreducible noise.

Now consider an ensemble of $T$ trees, each with the same marginal distribution (identically
distributed, but not independent — they share training data). Let $\rho$ be the pairwise
Pearson correlation between any two trees' predictions on a fixed $\mathbf{x}$, and let
$\sigma^2$ be the variance of a single tree's prediction. The variance of the ensemble average is:

$$\text{Var}\left(\frac{1}{T}\sum_{t=1}^T h_t(\mathbf{x})\right) = \rho\sigma^2 + \frac{1-\rho}{T}\sigma^2$$

This is the **fundamental equation of ensemble learning**. Let's unpack it carefully:

- The term $\frac{1-\rho}{T}\sigma^2$ **shrinks to zero** as $T \to \infty$ — averaging uncorrelated
  noise cancels out.
- The term $\rho\sigma^2$ is the **irreducible floor** — no matter how many trees you add, if
  trees are perfectly correlated ($\rho = 1$, as in a single CART tree), averaging buys you nothing.
- The key insight: **$\rho$ is the binding constraint**. Reducing $m$ (features per split) reduces
  $\rho$ but increases individual-tree variance $\sigma^2$ (each tree is less accurate). The RF
  optimizes this tradeoff.

> 💡 **Intuition:** Imagine you are measuring the temperature of a room. If you ask 100 friends
> to read the same thermometer, averaging doesn't help — all measurements are perfectly correlated
> ($\rho = 1$). If instead you give each friend a different thermometer with independent random
> errors, averaging reduces your estimation error by $\sqrt{T}$. Random Forests achieve something
> between these extremes: trees share training data (pushes $\rho$ up) but see different random
> feature subsets at each node (pushes $\rho$ down). The skill is in calibrating how much
> feature randomization to inject.

### The Role of Bootstrap Sampling

Bootstrap sampling (sampling $n$ examples with replacement) provides a first layer of
decorrelation. Each tree is trained on a different bootstrap sample, so their errors are
not identical. But without feature randomization, trees trained on similar bootstrap samples
tend to split on the same highly informative features — particularly at the root, where the
most globally discriminative split is made.

The famous result: when $m = p$ (all features, pure bagging), the inter-tree correlation
stays high because all trees discover the same strong predictors. Reducing $m$ forces trees
to sometimes split on weaker features, creating structural diversity that substantially
lowers $\rho$.

> 🔬 **Deep dive:** The correlation $\rho$ between trees has been studied theoretically
> by Biau (2012) for a simplified RF model. Biau showed that for a forest of Breiman-type
> trees (using random feature subsampling), the generalization error converges to the Bayes
> error at rate $O(n^{-1/(S \log 2 + 1)})$ where $S$ is the number of informative features —
> crucially, the rate depends only on $S$ (the effective dimension), not on the total number
> of features $p$. This is the formal statement of why Random Forests adapt to sparsity:
> irrelevant features, once they stop being selected at nodes, eventually have zero effect.

### Split Criterion: Gini vs Entropy vs MSE

**For classification**, the default is **Gini impurity**:

$$G(t) = 1 - \sum_{k=1}^K p_k(t)^2 = \sum_{k=1}^K p_k(t)(1 - p_k(t))$$

where $p_k(t)$ is the fraction of class $k$ samples at node $t$. At a pure node (one class),
$G = 0$. The maximum impurity is $1 - 1/K$ (uniform distribution over $K$ classes).

The **information gain** (entropy) criterion uses:

$$H(t) = -\sum_{k=1}^K p_k(t) \log p_k(t)$$

Both criteria measure the same thing — class mixing — and produce nearly identical trees in
practice. Gini is slightly faster to compute (no logarithm). For `criterion="log_loss"` (added
in sklearn 1.1), the split minimizes the log-loss, which is numerically equivalent to entropy.

The **impurity decrease** that drives the split decision at node $t$:

$$\Delta I = I(t) - \frac{n_{t_L}}{n_t} I(t_L) - \frac{n_{t_R}}{n_t} I(t_R)$$

A split is chosen only if $\Delta I > 0$ (and $\geq$ `min_impurity_decrease` if set).

**For regression**, the default is **mean squared error** (MSE), equivalently called `"squared_error"`:

$$\text{MSE}(t) = \frac{1}{|S_t|}\sum_{i \in S_t}(y_i - \bar{y}_t)^2$$

where $\bar{y}_t$ is the mean label in node $t$. Leaf predictions are the mean of all labels
in the leaf. Other available criteria: `"absolute_error"` (MAE, more robust to outliers but
slower), `"friedman_mse"` (weighted MSE with Friedman's improvement score — often better for
splitting), `"poisson"` (for count targets with Poisson deviance).

### Mean Decrease Impurity (MDI) Feature Importance

After training, sklearn computes MDI importance (stored in `rf.feature_importances_`) as the
weighted total impurity decrease accumulated at all nodes where feature $j$ was used:

$$\text{MDI}(j) = \frac{1}{T} \sum_{t=1}^T \sum_{\substack{v \in \text{nodes of } h_t \\ \text{split on feature } j}} \frac{n_v}{n} \Delta I_v$$

This is computed during training at zero extra cost, normalized to sum to 1. It is fast,
always positive, and readily interpretable. However, it is **biased** — we address this
critically in §6 and §8.

### Prediction: What Happens at Inference Time

For a new sample $\mathbf{x}$:

1. Drop $\mathbf{x}$ down each of the $T$ trees, following the learned decision rules at each
   internal node until reaching a leaf.
2. Each leaf stores either a class distribution (classifier) or a mean value (regressor).
3. **Classification**: aggregate the $T$ leaf predictions by averaging probabilities across
   trees (soft voting), then take argmax.
4. **Regression**: average the $T$ leaf predictions.

Notice: the prediction is an **average of piecewise-constant functions**. Each tree partitions
$\mathbb{R}^p$ into rectangular (axis-aligned) boxes; the RF prediction is the average over $T$
such partitions. This creates a smooth (in the averaging sense) approximation to any function,
but **it cannot extrapolate** beyond the range of training values — a critical limitation
discussed in §6.

### Computational Complexity

**Training:**
- Building one tree: $O(n p \log n)$ in the general case (sorting $n$ samples on $p$ features at each level, $\log n$ levels)
- With feature subsampling: $O(n m \log n)$ per tree, where $m \ll p$
- Full forest: $O(T \cdot n \cdot m \cdot \log n)$

**Inference:**
- One tree: $O(\log n)$ — follow a path of depth $\log n$ from root to leaf
- Full forest: $O(T \cdot \log n)$ — trivially parallelizable across trees

**Memory:**
- Each fully-grown tree (no `max_depth`) stores $O(n)$ nodes in the worst case
- Forest: $O(T \cdot n)$ nodes total
- For $n = 10^6$, $T = 500$: potentially several gigabytes — cap `max_depth` to control this

**Parallelization:** Both training and prediction are embarrassingly parallel across trees
(`n_jobs=-1` uses all CPUs). This is one of RF's great practical advantages over gradient
boosting, which is sequential by design.

### What the Algorithm Assumes (Implicit Assumptions)

Random Forests are often called "assumption-free" — this is an oversimplification. They assume:

1. **Training and test data are i.i.d.**: drawn from the same distribution. If the test distribution
   shifts (covariate shift), RF performance degrades like all supervised methods.

2. **The signal is recoverable by axis-aligned splits**: RF uses recursive binary splits along
   individual feature axes. Functions with strong diagonal or nonlinear interaction structure
   require many splits to approximate — oblique RF variants (§3.4) address this.

3. **Finite label range (regression)**: Leaf predictions are averages of training labels, so
   the RF cannot predict values outside the range $[\min(y), \max(y)]$. For extrapolation tasks
   (time series forecasting into the future), this is a hard failure mode.

4. **Sufficiently many samples per leaf**: With very small $n$ or very deep trees, leaves may
   contain a single sample. Predictions from such leaves are unregularized and noisy.

5. **Stationarity**: The training data represents the deployment environment. Concept drift
   over time is not handled by standard RF.

---

## §3 — The Full Evolution: From Bagging to Deep Forests

### §3.1 — Bagging: The Direct Ancestor (Breiman 1996)

> 📜 **Citation:** Breiman, L. (1996). Bagging Predictors. *Machine Learning*, 24(2), 123–140.

**The problem Bagging solved:** Individual CART trees have catastrophically high variance. Small
changes in training data produce completely different trees. Breiman's insight: average over
bootstrap replicates to smooth out the variance.

**Algorithm:** Train $T$ trees on bootstrap samples; average predictions. No feature
randomization.

**Why RF improved on it:** All trees in a bagged ensemble split on the same best features,
creating high inter-tree correlation $\rho \approx 0.9$ (empirically). From the fundamental
equation, the variance floor is $\rho\sigma^2 \approx 0.9\sigma^2$ — almost no benefit from averaging.

**sklearn:**
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging = RF with max_features=None (no feature randomization)
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    bootstrap=True,
    random_state=42
)
```

**When to prefer Bagging over RF:** Almost never for decision trees — RF is strictly better.
Bagging shines when your base estimator is not a decision tree (e.g., neural networks, k-NN).

---

### §3.2 — Extremely Randomized Trees / Extra-Trees (Geurts et al. 2006)

> 📜 **Citation:** Geurts, P., Ernst, D. & Wehenkel, L. (2006). Extremely randomized trees.
> *Machine Learning*, 63(1), 3–42. DOI: https://doi.org/10.1007/s10994-006-6226-1

**The problem ExtraTrees solved:** Random Forest still searches exhaustively for the optimal
split threshold within the random feature subset — this is expensive and the trees, while
decorrelated, still share some structural similarities when the same strong feature always wins
the threshold search.

**The key algorithmic change:**

In RF, at node $v$ with randomly selected feature subset $F_v$ of size $m$:
$$\theta^*_j = \arg\min_\theta \Delta I(j, \theta)  \quad \forall j \in F_v$$

ExtraTrees replaces the threshold search with a random draw:
$$\theta_j \sim \text{Uniform}(\min_{i \in S_v} x_{ij},\ \max_{i \in S_v} x_{ij}) \quad \forall j \in F_v$$

Then selects the best (feature, random-threshold) pair. This "double randomization" — random
features AND random thresholds — further decorrelates trees and eliminates the expensive
threshold search entirely.

**Key behavioral differences from RF:**

| Property | Random Forest | ExtraTrees |
|---|---|---|
| Bootstrap | True (default) | **False** (default — sees full dataset) |
| Split threshold | Optimal search | Random uniform draw |
| Speed | Baseline | ~2-3x faster training |
| Variance | Lower (bootstrap + optimal splits) | Even lower (double randomization) |
| Bias | Lower | **Higher** (random thresholds are suboptimal) |
| OOB error | Available | Not available by default (set `bootstrap=True`) |

> 💡 **Intuition:** Think of RF as hiring 100 skilled analysts, each looking at a random
> subset of evidence and finding the best argument. ExtraTrees hires 100 analysts who each
> look at random evidence and make a random argument — then vote. The individual arguments are
> weaker, but the diversity of the committee is higher, and the cancellation of errors in voting
> often compensates.

**When to prefer ExtraTrees over RF:**
- When training speed is critical (large datasets, many features)
- When you want maximum regularization (random thresholds penalize overfitting more)
- When the dataset is large enough that individual-tree bias matters less than ensemble variance
- Empirically: ET and RF perform similarly, with ET slightly worse on average but occasionally
  better when data has many irrelevant features

**sklearn:**
```python
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

et = ExtraTreesClassifier(
    n_estimators=200,
    max_features="sqrt",   # same default as RF for classifier
    bootstrap=False,       # default; set True to enable OOB
    random_state=42,
    n_jobs=-1
)
```

> ⚠️ **Pitfall:** `ExtraTreesClassifier(bootstrap=False)` cannot produce `oob_score_`. If you
> need OOB estimates for ET, explicitly set `bootstrap=True`, understanding that you lose the
> "full dataset per tree" property.

---

### §3.3 — Isolation Forest (Liu, Ting & Zhou 2008/2012)

> 📜 **Citations:**
> Liu, F.T., Ting, K.M. & Zhou, Z.-H. (2008). Isolation Forest. *Proc. 8th IEEE ICDM*, pp. 413–422.
> Liu, F.T., Ting, K.M. & Zhou, Z.-H. (2012). Isolation-Based Anomaly Detection.
> *ACM TKDD*, 6(1), Article 3. DOI: https://doi.org/10.1145/2133360.2133363

**The problem it solves:** Classical anomaly detection methods (LOF, one-class SVM) are $O(n^2)$
in computation and $O(n)$ in memory — they explicitly model the "normal" region and flag
deviations. For large datasets this is prohibitive.

**The key insight:** Anomalies are "few and different" — they have unusual feature values that
are easy to isolate by random splits. Normal points require many splits to isolate (they blend
in with many similar points); anomalies require very few.

**The algorithm:**
1. Build $T$ isolation trees using only `max_samples` observations per tree (default: min(256, n) — deliberately small to avoid trees being too similar)
2. Each tree: randomly select a feature, randomly select a split point in $[\min, \max]$ of that feature, recurse. No class labels needed — this is **unsupervised**.
3. The path length $h(\mathbf{x})$ = number of edges to isolate $\mathbf{x}$

**Anomaly score:**

$$s(\mathbf{x}, n) = 2^{-\frac{E[h(\mathbf{x})]}{c(n)}}$$

where $c(n) = 2H(n-1) - \frac{2(n-1)}{n}$ is the expected path length of an unsuccessful BST
search ($H$ = harmonic number). Scores near 1 indicate anomalies; near 0.5 indicate normal points.

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    n_estimators=100,
    max_samples="auto",       # min(256, n_samples) — deliberately small
    contamination=0.05,       # expected fraction of anomalies
    random_state=42
)
iso_forest.fit(X_train)
scores = iso_forest.score_samples(X_test)   # lower = more anomalous
labels = iso_forest.predict(X_test)         # 1 = inlier, -1 = outlier
```

> ⚠️ **Verify:** `contamination='auto'` sets the threshold such that the training score
> distribution's quantile matches the contamination level. In sklearn 1.5.2, `'auto'`
> sets the threshold at the score corresponding to $1 - 0.1 = 0.9$ quantile of the training
> scores (i.e., 10% contamination assumed). When you know your contamination rate, always
> specify it explicitly.

---

### §3.4 — Oblique Random Forests & SPORF (Tomita et al. 2020)

> 📜 **Citation:** Tomita, T.M. et al. (2020). Sparse Projection Oblique Randomer Forests.
> *Journal of Machine Learning Research*, 21(104), 1–39.
> JMLR: https://jmlr.org/papers/v21/18-664.html

**The problem it solves:** Standard RF splits are always **axis-aligned** — they split on a
single feature $x_j \leq \theta$. Any decision boundary that is diagonal in feature space requires
many axis-aligned splits to approximate, increasing tree complexity unnecessarily.

Classic example: XOR problem. Points at $(0,0), (1,1)$ are class 0; $(0,1), (1,0)$ are class 1.
A single oblique split $x_1 - x_2 \leq 0$ separates them perfectly; axis-aligned splits need
at least two levels.

**The key algorithmic change:** SPORF replaces the axis-aligned split $x_j \leq \theta$ with
a sparse linear combination (oblique split):

$$\mathbf{a}^T \mathbf{x} \leq \theta$$

where $\mathbf{a} \in \mathbb{R}^p$ is a **sparse random projection vector** with entries drawn
from $\{-1, 0, +1\}$ and sparsity parameter $\lambda$ controlling how many features are combined.
At each node, multiple $(\mathbf{a}, \theta)$ candidates are evaluated and the best chosen.

**When to prefer SPORF over RF:**
- Correlated features with diagonal decision boundaries
- XOR/checkerboard pattern data
- Image patches, EEG/sensor data where neighboring features are correlated
- Often matches or exceeds deep networks on tabular biological data

```python
# pip install treeple  # verify install command in current docs
from treeple.ensemble import ObliqueRandomForestClassifier

sporf = ObliqueRandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
sporf.fit(X_train, y_train)
```

> ⚠️ **Verify:** The `treeple` package (formerly `scikit-tree`) is the primary sklearn-compatible
> implementation of SPORF. Check https://docs.neurodata.io/treeple/ for the current install command
> and API. Some parameter names may differ from sklearn.

---

### §3.5 — Mondrian Forests (Lakshminarayanan et al. 2014)

> 📜 **Citation:** Lakshminarayanan, B., Roy, D.M. & Teh, Y.W. (2014). Mondrian Forests:
> Efficient Online Random Forests. *NeurIPS*, 27.
> arXiv: https://arxiv.org/abs/1406.2673

**The problem it solves:** Standard RF requires retraining from scratch when new data arrives.
In streaming environments (financial ticks, sensor readings, recommendation logs), you cannot
wait for a batch retrain.

**The key innovation:** Mondrian Forests use the **Mondrian process** — a stochastic process
over recursive axis-aligned partitions of feature space — as the tree prior. The defining
property: a Mondrian partition of $n$ points is **consistent** with (can be extended to cover)
a Mondrian partition of $n+1$ points. This enables exact online updates.

When new labeled example $(\mathbf{x}_{n+1}, y_{n+1})$ arrives, each Mondrian tree is updated by
sampling an **extended Mondrian tree** from the posterior that (a) preserves the existing
partitions for old data and (b) incorporates the new point. No full retraining required.

**Python implementations:** The `skgarden` library includes Mondrian Forests, though it may
be unmaintained. The AMF (Aggregated Mondrian Forests) variant by Mourtada et al. (2021,
*JRSS-B*) provides theoretical improvements; check for active Python packages before use.

**When to use:** Streaming sensor data, financial data arriving in real time, systems that
must update without batch retraining cycles.

---

### §3.6 — Deep Forest / gcForest (Zhou & Feng 2017/2019)

> 📜 **Citation:** Zhou, Z.-H. & Feng, J. (2019). Deep Forest. *National Science Review*, 6(1), 74–86.
> arXiv: https://arxiv.org/abs/1702.08835

**The problem it purports to solve:** Deep neural networks dominate on large datasets but
are data-hungry, computationally expensive, and sensitive to hyperparameters. gcForest
proposes deep, layered ensemble learning using forests instead of neurons.

**The architecture:**
1. **Multi-grained scanning:** Slide windows of multiple sizes over the input (like CNN
   convolutions but using forests) to generate local feature vectors
2. **Cascade forest:** Stack multiple layers of RF + ExtraTrees ensembles. The class probability
   outputs of layer $\ell$ are concatenated to the original features to form the input of layer $\ell+1$
3. **Adaptive depth:** Add layers until validation performance stops improving

**Honest assessment:** gcForest generated significant interest, but independent benchmarks
showed mixed results. On small/medium structured datasets it can match well-tuned sklearn
ensembles, but it rarely matches modern GBM (XGBoost/LightGBM) on tabular data or deep networks
on vision/NLP. The Python implementations may not be actively maintained — verify before use.

> ⚠️ **Verify:** Both known Python implementations (pylablanche/gcForest and kingfengji/gcForest
> on GitHub) may lack Python 3.10+ compatibility. Check for active forks before using in production.

---

### §3.7 — Canonical Correlation Forests (Rainforth & Wood 2015)

> 📜 **Citation:** Rainforth, T. & Wood, F. (2015). Canonical Correlation Forests.
> arXiv:1507.05444. URL: https://arxiv.org/abs/1507.05444

**The problem it solves:** RF splits are data-agnostic at each node — they look at features
but not at the relationship between features and labels when choosing split directions. CCF
computes the first **canonical correlation** between the randomly selected features and the
one-hot label matrix at each node, using the top canonical variate as the split direction.

**Key change:** At each node, run CCA between $X_{F_v}$ (selected features) and $\mathbf{Y}$
(one-hot labels). The first canonical direction $\mathbf{a}$ defines a data-adaptive oblique split.
This is supervised feature extraction at each node — more powerful than SPORF's random projections,
at the cost of higher per-node computation.

**When useful:** Multi-label and multi-output problems where canonical correlation between
features and labels is meaningful.

---

## §4 — Hyperparameters: The Complete Guide

Random Forest has 20 parameters in sklearn 1.5. Most practitioners tune only 2-3 and leave
the rest at defaults. This section explains every single one — what it does, why it matters,
and how to tune it.

### §4.1 — `n_estimators` (int, default=100)

**What it controls:** The number of trees $T$ in the ensemble.

**Mechanism:** Adding trees reduces ensemble variance monotonically — each additional tree
provides an independent (or partially independent) prediction whose errors partially cancel
with other trees. However, the marginal gain follows the law of diminishing returns:

$$\text{Var}(\text{ensemble}) = \rho\sigma^2 + \frac{1-\rho}{T}\sigma^2$$

The first term $\rho\sigma^2$ is irreducible regardless of $T$. The second term $\frac{1-\rho}{T}\sigma^2$
shrinks quickly for small $T$ (from 10 to 100 is a big reduction) but slowly for large $T$
(from 500 to 1000 gains almost nothing).

**Default reasoning:** The default was 10 in sklearn < 0.22 — famously too small for production
use. It was changed to 100 in v0.22. For most tabular datasets, 100 is adequate; for high-stakes
or very noisy problems, 200–500 is better.

**Diagnostic for setting n_estimators:** Use `warm_start=True` to fit trees incrementally and
track OOB error convergence:

```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

oob_errors = []
n_range = range(10, 301, 10)

rf = RandomForestClassifier(warm_start=True, oob_score=True, random_state=42)
for n in n_range:
    rf.set_params(n_estimators=n)
    rf.fit(X_train, y_train)
    oob_errors.append(1 - rf.oob_score_)

plt.plot(list(n_range), oob_errors)
plt.xlabel("Number of trees"); plt.ylabel("OOB error")
plt.title("OOB Error Convergence"); plt.tight_layout(); plt.show()
# Look for the elbow — set n_estimators just past where the curve flattens
```

> ⚠️ **Pitfall:** Using `warm_start=True` inside a `cross_val_score()` loop causes backend
> switching overhead (sklearn GitHub issue #22087). Use warm_start only for the OOB-elbow
> diagnostic outside of CV loops.

**Too low:** High variance in OOB estimates; ensemble predictions unstable; risk of NaN in
`oob_decision_function_` (some samples never OOB).

**Too high:** Wasted compute and memory; no accuracy gain beyond the elbow.

**Optuna range:** `trial.suggest_int("n_estimators", 50, 500)`

---

### §4.2 — `max_features` (str/int/float, default="sqrt" for clf, 1.0 for reg)

**What it controls:** The number of features $m$ randomly selected at each split node.

**This is the most important hyperparameter for a Random Forest.** It is the primary lever
controlling the bias-variance tradeoff at the tree level and, more importantly, the
inter-tree correlation $\rho$.

| Setting | Value of $m$ | Effect |
|---|---|---|
| `"sqrt"` | $\lfloor\sqrt{p}\rfloor$ | Breiman's default for classification; good bias-variance tradeoff |
| `"log2"` | $\lfloor\log_2 p\rfloor$ | Stronger decorrelation; higher bias; best for high-dimensional data |
| `None` | $p$ (all features) | Pure bagging — no feature randomization; maximum bias risk |
| `0.5` (float) | $\lfloor 0.5p\rfloor$ | Between "sqrt" and all; often optimal for regression |
| `int k` | $k$ exactly | Direct control |

> ⚠️ **Critical**: `max_features` defaults differ between classifier and regressor:
> - `RandomForestClassifier`: `"sqrt"` → $m \approx \sqrt{p}$
> - `RandomForestRegressor`: `1.0` → $m = p$ (ALL features — essentially pure bagging)
>
> Breiman's original recommendation for regression was $m = p/3$, not all features. The sklearn
> regressor default of `1.0` is conservative and often suboptimal for regression tasks.
> Always explicitly tune `max_features` for regression problems.

**Tuning strategy:**
- Classification: try `["sqrt", "log2", 0.3, 0.5]`
- Regression: try `[0.33, 0.5, 0.7, 1.0]`
- With many irrelevant features: lower `max_features` to reduce the chance of splitting on noise

**Interaction with `n_estimators`:** Lower `max_features` increases individual-tree noise →
you need more trees to stabilize the ensemble. When you drop `max_features` below "sqrt",
compensate by increasing `n_estimators`.

**Optuna:** `trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7])`

---

### §4.3 — `max_depth` (int or None, default=None)

**What it controls:** Maximum depth of individual trees. `None` = grow until all leaves are pure or
can't be split (by `min_samples_split`, `min_samples_leaf`, etc.).

**Mechanism:** Deeper trees = lower bias per tree (can capture complex interactions) but higher
variance. The ensemble averaging handles variance, so RF benefits from **fully grown trees** (None)
more than GBMs do (which use shallow trees deliberately). However, with very large $n$, fully
grown trees consume massive memory.

**Default reasoning:** `None` (unlimited depth) works well because ensemble averaging controls
variance. This is opposite to GBMs, which prefer shallow trees (depth 3-8).

**When to change:**
- Memory constraints: cap at 15-30 to prevent RAM exhaustion on large datasets
- Extreme overfitting on very noisy data: cap at 10-15
- Need for faster inference: shallow trees → faster `predict()`

**Range for tuning:** `trial.suggest_int("max_depth", 5, 30)` or add `None` as a categorical option.

---

### §4.4 — `min_samples_split` (int or float, default=2)

**What it controls:** The minimum number of samples required at a node to attempt splitting it.
A node with fewer samples becomes a leaf without further splitting.

**Default (2):** A node with 2 or more samples will be split — allowing the tree to grow until
leaves are singletons. This is deliberate: deep trees + averaging handles the variance.

**As float (e.g., 0.01):** Interpreted as a fraction of `n_samples` — useful for large datasets
where you want to prevent very small splits as a fraction of the data.

**Effect of increasing:** Shallower trees → higher bias, lower variance per tree. Use when you
see clear overfitting.

**Range:** `trial.suggest_int("min_samples_split", 2, 20)`

---

### §4.5 — `min_samples_leaf` (int or float, default=1)

**What it controls:** Minimum number of samples required in any leaf node. A split is rejected
if it would produce a child with fewer than `min_samples_leaf` samples.

**Distinction from `min_samples_split`:** `min_samples_split` checks the parent node size;
`min_samples_leaf` enforces a floor on child node sizes. Both prune tree growth; `min_samples_leaf`
is generally more interpretable and provides smoother regression predictions.

**For regression:** Increasing `min_samples_leaf` to 2-10 significantly smooths predictions —
leaves with more samples average more values, reducing leaf-level noise.

**Range:** `trial.suggest_int("min_samples_leaf", 1, 20)`

---

### §4.6 — `max_leaf_nodes` (int or None, default=None)

**What it controls:** Grows trees in **best-first** order (most impurity-reducing split first)
with at most `max_leaf_nodes` leaves. `None` = no limit.

**Alternative to `max_depth`:** `max_depth` limits depth level-by-level; `max_leaf_nodes`
limits the total size of the tree but allows asymmetric growth — one deep branch where the
data requires it, shallow elsewhere. Often produces better-calibrated trees than `max_depth`.

---

### §4.7 — `min_impurity_decrease` (float, default=0.0)

**What it controls:** A node is split only if the impurity decrease is at least this threshold.
Provides a minimum "usefulness" gate for any split.

**Rarely needed:** Most practitioners prefer `min_samples_leaf` or `max_depth` for cleaner
interpretability. Use `min_impurity_decrease` when you want to explicitly ignore low-signal splits.

---

### §4.8 — `bootstrap` (bool, default=True)

**What it controls:** Whether to bootstrap-sample training data for each tree.

- `True` (RF default): each tree trains on a bootstrap sample (≈63.2% unique samples), enabling
  OOB error estimation
- `False` (ExtraTrees default): each tree trains on the full dataset; trees are more correlated
  (higher $\rho$) but have lower variance per sample

**Important constraint:** `oob_score=True` requires `bootstrap=True`. Setting `bootstrap=False`
disables OOB entirely.

---

### §4.9 — `oob_score` (bool or callable, default=False)

**What it controls:** Whether to compute a generalization estimate using OOB samples.

- `True`: computes accuracy (classifier) or R² (regressor) using OOB predictions
- **Callable (sklearn 1.4+):** pass a custom scorer, e.g., `oob_score=roc_auc_score` for AUC

```python
from sklearn.metrics import roc_auc_score
rf = RandomForestClassifier(
    n_estimators=200,
    oob_score=roc_auc_score,  # callable scorer — sklearn 1.4+
    random_state=42
)
rf.fit(X_train, y_train)
print(f"OOB AUC: {rf.oob_score_:.4f}")
```

> ⚠️ **Verify:** The callable scorer protocol for `oob_score` in sklearn 1.4+ requires a
> function with signature `scorer(y_true, y_score)`. For multi-class, this may differ —
> check current docs at https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html

**OOB estimate is slightly pessimistic:** Each sample uses only ~36.8% of trees for its OOB
prediction vs. the full ensemble at test time. The OOB estimate tends to underestimate true
generalization accuracy by ~0.5-2% — a known and acceptable bias.

---

### §4.10 — `class_weight` (str/dict/list/None, default=None)

**What it controls:** Sample weights applied per class during tree construction.

- `None`: equal weights for all classes
- `"balanced"`: weights proportional to inverse class frequency — $w_k = \frac{n}{K \cdot n_k}$
- `"balanced_subsample"`: like `"balanced"` but recomputed on each bootstrap sample. Better
  for highly imbalanced data because each tree's weight computation reflects its sampled class distribution.
- `dict` or `list of dicts`: manual per-class weights

**Critical for imbalanced data:** A 95%/5% class split will produce high accuracy but terrible
recall for the minority class without `class_weight`. Always use `"balanced"` or
`"balanced_subsample"` for imbalanced problems.

---

### §4.11 — `ccp_alpha` (float ≥ 0, default=0.0)

**What it controls:** Post-hoc **Cost-Complexity Pruning** (Breiman et al. 1984). Higher values
prune each tree more aggressively after training, removing subtrees that don't reduce impurity
sufficiently per added complexity.

**When to use:** When individual trees are extremely deep and overfitting is severe. Use
`DecisionTreeClassifier.cost_complexity_pruning_path()` on a single representative tree to find
a reasonable alpha range, then tune via CV.

**Rarely needed for RF:** Ensemble averaging already controls variance; CCP adds overhead.

---

### §4.12 — `max_samples` (int, float, or None; default=None)

**What it controls:** When `bootstrap=True`, how many samples to draw for each tree.
- `None`: draw $n$ samples (standard bootstrap, ~63.2% unique)
- `float` (e.g., 0.5): draw $\lfloor 0.5 \cdot n \rfloor$ samples per tree
- `int` $k$: draw exactly $k$ samples per tree

**Use case:** Very large datasets where full bootstrap is slow. Setting `max_samples=0.5`
halves training time with modest accuracy cost.

> ⚠️ **Interaction:** When `max_samples < n`, the effective OOB fraction changes from ~36.8% —
> more samples will be OOB, giving a more conservative (pessimistic) OOB estimate.

---

### §4.13 — `monotonic_cst` (array-like of {-1, 0, 1} or None; default=None) — Added v1.4

**What it controls:** Per-feature monotonicity constraints.
- `1`: prediction must be non-decreasing in this feature
- `-1`: prediction must be non-increasing in this feature
- `0`: unconstrained

**Use case:** Business rules. If you know "credit score increases → default probability must
decrease", enforce it: `monotonic_cst = [-1, 0, 0, ...]` for credit score in position 0.

**Limitation (sklearn 1.4/1.5):** Not supported for multi-class (>2 classes) or multi-output
targets. Binary classification and regression only.

---

### §4.14 — The Tuning Playbook

| Hyperparameter | Typical Range | Primary Effect | Tuning Priority |
|---|---|---|---|
| `n_estimators` | 100–500 | Variance ↓ as $T$ ↑; compute ↑ | Low (set 200+, use OOB elbow) |
| `max_features` | `"sqrt"`, `"log2"`, 0.3–0.7 | Bias↑/Variance↓ tradeoff | **High** — most impactful |
| `max_depth` | `None`, 10–30 | Overfitting control, memory | Medium (None usually fine) |
| `min_samples_leaf` | 1–20 | Overfitting/smoothness | Medium |
| `min_samples_split` | 2–20 | Tree depth control | Low (often redundant with above) |
| `class_weight` | `"balanced"`, `"balanced_subsample"` | Class imbalance | **Critical for imbalanced data** |
| `ccp_alpha` | 0.0–0.05 | Post-hoc pruning | Low (rarely needed) |
| `bootstrap` | True/False | OOB availability | Low (keep True) |
| `max_samples` | 0.5–1.0 | Subsample size per tree | Low (tune for huge datasets) |

**Complete Optuna objective:**
```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
        ),
        "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1,
    }
    rf = RandomForestClassifier(**params)
    return cross_val_score(rf, X_train, y_train, cv=5, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
best_params = study.best_params
print(f"Best params: {best_params}")
print(f"Best OOB AUC: {study.best_value:.4f}")
```

> 🏆 **Best practice:** In the Optuna objective, set `n_jobs=1` inside the model and call
> `study.optimize(objective, n_trials=100, n_jobs=4)` at the Optuna level to parallelize
> trials rather than trees — this avoids nested parallelism and is more efficient.

---

## §5 — Strengths

### §5.1 — Robust to Overfitting (With Mechanism)

Adding more trees to a Random Forest cannot increase overfitting — this is a proven theoretical
property. The generalization error converges almost surely as $T \to \infty$ (Breiman 2001,
Theorem 1.2). The variance term $\frac{1-\rho}{T}\sigma^2$ in the ensemble variance formula
only decreases as $T$ grows, while the bias is unchanged.

Contrast with gradient boosting: more GBM rounds eventually overfit (the model is fitting
residual noise). Random Forest has no such sequential residual structure — each tree is
independent. This makes RF much more forgiving to `n_estimators` tuning: you can set it high
(500+) without worrying about overfitting, only about compute time.

> 📊 **Benchmark:** Fernández-Delgado et al. (2014, "Do we Need Hundreds of Classifiers to
> Solve Real World Classification Problems?", *JMLR* 15) evaluated 179 classifiers on 121 UCI
> datasets. Random Forests won or tied the top position on 84.3% of datasets. This established
> RF as the de facto baseline for tabular classification.

### §5.2 — No Feature Scaling Required

Decision trees make split decisions based on thresholds: $x_j \leq \theta$. This threshold
comparison is completely invariant to the scale of $x_j$ — multiplying all values of feature $j$
by 1000 shifts $\theta$ by 1000 but doesn't change which samples go left vs. right. The tree
structures, impurity values, and final predictions are identical regardless of feature scaling.

**Practical consequence:** You can feed raw features directly into a Random Forest without
`StandardScaler`, `MinMaxScaler`, or any normalization. This eliminates a source of data leakage
(fitting the scaler on the full dataset) and simplifies pipelines substantially.

> ⚠️ **Caveat:** Although predictions are scale-invariant, MDI feature importances are
> **not** perfectly scale-invariant. High-cardinality continuous features on large scales may
> have more candidate split points evaluated, potentially inflating their MDI score slightly.
> This interacts with the Strobl 2007 bias issue discussed in §6.

### §5.3 — Handles Mixed Feature Types Natively

RF splits on threshold comparisons that work identically for continuous, ordinal, and
(when numerically encoded) categorical features. Unlike SVMs (which require one-hot encoding
and scaling) or linear models (which assume linearity), RF captures nonlinear relationships
with categorical variables directly.

In sklearn 1.4+, RF also handles missing values natively: during training, the tree learns
which direction (left or right) to route missing-value samples at each split; at prediction
time, the same learned direction is applied. No imputer needed.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# sklearn 1.4+: NaN handled natively
X_with_missing = np.array([[1, np.nan], [2, 3], [np.nan, 4], [3, 2]])
y = [0, 1, 1, 0]
rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_with_missing, y)
```

### §5.4 — Built-in Uncertainty Estimation (OOB + Probability Calibration)

The OOB error gives a nearly free estimate of generalization performance — every training sample
has been the OOB sample for roughly 36.8% of trees, providing a cross-validation-like estimate
without a held-out set. This is particularly valuable for small datasets.

The `predict_proba()` method returns class probability estimates by averaging across trees.
While these probabilities are not perfectly calibrated (they tend toward 0.5 due to the
averaging of tree leaf proportions), they provide useful uncertainty information and can be
post-processed with Platt scaling or isotonic regression.

### §5.5 — Parallelism: Near-Linear Scaling with CPU Cores

The training of $T$ trees is **embarrassingly parallel** — each tree is independent of the
others. With `n_jobs=-1`, sklearn distributes tree training across all available CPU cores.
Training time scales approximately as $O(T / \text{n\_cores})$.

Inference is similarly parallelizable (each tree independently predicts, then results are
aggregated). On a 32-core machine, a 200-tree forest trains approximately 20-25x faster than
on a single core.

For even larger speedups, cuML provides GPU-accelerated RF with 20-45x speedup over CPU sklearn
on datasets with >100k samples (NVIDIA benchmark, 2024).

### §5.6 — Interpretable Feature Importance (With Caveats)

RF provides `feature_importances_` (MDI) at zero extra cost after training. Despite its biases
(discussed in §6), it provides a useful first-pass signal about which features matter most.
For a model with 200 features, MDI will typically identify the 5-10 truly important ones clearly.

Combined with SHAP TreeExplainer (§8), RF is one of the most interpretable non-linear models
available — you get both global importance (SHAP summary plot) and local explanations
(SHAP force plots for individual predictions) with computational efficiency that linear
models cannot match.

### §5.7 — Works Well on Small to Medium Datasets

Neural networks typically need thousands to millions of samples; RF achieves competitive
performance with a few hundred samples. The ensemble of fully-grown trees provides enough
capacity to fit complex patterns from limited data, while the bagging + feature randomization
provides enough regularization to avoid catastrophic overfitting.

> 📊 **Benchmark:** Shwartz-Ziv & Armon (2022, "Tabular Data: Deep Learning is Not All You
> Need") showed that XGBoost and RF consistently outperform deep learning on standard tabular
> ML benchmarks, with RF requiring far less hyperparameter tuning than XGBoost in many scenarios.

---

## §6 — Weaknesses & Failure Modes

### §6.1 — MDI Feature Importance Is Biased

**Mechanism:** Gini impurity and information gain are continuous quantities computed over all
possible split thresholds. A continuous feature with many unique values has many more candidate
split points than a binary feature. Even a random noise column, if it has high cardinality,
will find some split threshold that slightly reduces impurity by chance — and this "lucky split"
inflates its MDI score.

Strobl et al. (2007) demonstrated this with a simulation study: a random noise variable with
many categories consistently ranked higher on MDI than a truly predictive binary variable.

**Detection:**
```python
from sklearn.inspection import permutation_importance

# Compare MDI vs permutation importance
mdi = rf.feature_importances_
perm = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)

# Large disagreements indicate cardinality bias
for i, name in enumerate(feature_names):
    if abs(mdi[i] - perm.importances_mean[i]) > 0.05:
        print(f"Discrepancy for {name}: MDI={mdi[i]:.3f}, Perm={perm.importances_mean[i]:.3f}")
```

**Mitigation:**
1. Always compare MDI with permutation importance — large disagreements indicate bias
2. Use SHAP TreeExplainer for the most reliable importance estimates
3. If you need MDI specifically, normalize feature importances per cardinality group

> ⚠️ **Pitfall:** Many practitioners use `rf.feature_importances_` as the definitive feature
> ranking and make feature selection decisions on it. This can cause real harm: genuinely important
> low-cardinality features (e.g., a binary treatment indicator) may be dropped while noise
> columns with many values are retained.

---

### §6.2 — Permutation Importance Underestimates Correlated Features

**Mechanism (Strobl et al. 2008):** When two features $X_1 \approx X_2$ are highly correlated,
permuting $X_1$ has little effect on model performance because $X_2$ carries the same information.
The permutation-based importance of $X_1$ appears near zero — even if $X_1$ is a genuine causal
driver.

**Detection:** Compute the Spearman rank correlation matrix; features with $|r_s| > 0.7$ are
candidates for this problem.

**Mitigation:**
```python
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import numpy as np

# Compute Spearman correlations
corr, _ = spearmanr(X_train)
corr_linkage = hierarchy.ward(1 - np.abs(corr))
dendro = hierarchy.dendrogram(corr_linkage, no_plot=True)
cluster_ids = hierarchy.fcluster(corr_linkage, t=0.7, criterion="distance")

# Keep one feature per cluster
selected = []
for cluster in np.unique(cluster_ids):
    cluster_features = np.where(cluster_ids == cluster)[0]
    # Keep the feature with highest MDI importance in each cluster
    best = cluster_features[np.argmax(rf.feature_importances_[cluster_features])]
    selected.append(best)

# Recompute permutation importance on selected features only
```

---

### §6.3 — Cannot Extrapolate Beyond Training Range

**Mechanism:** RF predictions are averages of leaf node values. Each leaf value is the
mean of training labels in that leaf. Therefore, the forest prediction is always bounded:

$$\min(y_{\text{train}}) \leq \hat{y} \leq \max(y_{\text{train}})$$

For a regression problem where the test set contains targets outside the training range —
or for time-series forecasting where future values exceed historical values — RF predictions
are clipped at the extremes.

**Detection:** Plot `y_pred` vs `y_true` on a time-ordered holdout. Predictions will appear
flat/capped at historical min/max while true values continue trending.

**Mitigation:**
- Feature engineering: add trend features (e.g., time index, moving average deviation)
- Detrend the target before fitting RF; add trend back to predictions
- Switch to a model that can extrapolate (linear models, GBMs with tree leaves adjusted, neural networks)

> ⚠️ **Pitfall:** This is the most dangerous failure mode for time-series or economic forecasting
> with RF. A practitioner who sees high $R^2$ on training data may not notice the clipping
> until predictions are wrong on a bull/bear market regime the model has never seen.

---

### §6.4 — Memory Footprint for Large Datasets

**Mechanism:** sklearn stores the full tree structure (decision nodes + leaf values) for all $T$
trees. A fully-grown tree can have $O(n)$ nodes (one per training sample, in the worst case).
For $T = 500$ trees on $n = 10^6$ samples, this can require 50-100GB of RAM.

**Detection:** Monitor RAM usage during `rf.fit()`. Use Python's `tracemalloc` or system monitors.

**Mitigation:**
```python
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,      # cap tree depth — most effective memory control
    max_samples=0.5,   # use only 50% of data per tree
    n_jobs=-1
)
# Memory reduction: max_depth=15 reduces node count from O(n) to O(2^15)=32k per tree
```

Or switch to cuML's GPU-accelerated RF which stores trees in GPU VRAM with more efficient
data structures.

---

### §6.5 — Slow Inference for Real-Time Applications

**Mechanism:** Prediction requires dropping each query through $T$ trees of depth $O(\log n)$
each. For $T = 500$ and $n = 10^6$ training samples (depth ~20), each prediction traverses
~10,000 decision nodes.

**When this matters:** Sub-millisecond latency requirements (ad auctions, HFT, real-time
fraud detection). For batch prediction this is rarely a problem.

**Mitigation options:**
- Reduce `n_estimators` at inference time: fit 500 trees, but use `rf.estimators_[:100]`
  for faster inference (slight accuracy reduction)
- Post-training pruning via `ccp_alpha`
- Convert RF to a decision table lookup (hummingbird library converts sklearn models to PyTorch)
- Use a distilled model (train a smaller model to mimic RF predictions)

---

### §6.6 — Class Imbalance Without Correction

**Mechanism:** On a 95%/5% imbalanced dataset, an RF trained without `class_weight` achieves
~95% accuracy by always predicting the majority class. The Gini criterion at each split is
dominated by the majority class, and minority-class samples have insufficient representation
in most bootstrap samples.

**Detection:** High accuracy but low recall for minority class; AUC-ROC << 0.9 despite high accuracy.

**Mitigation:**
```python
rf_balanced = RandomForestClassifier(
    class_weight="balanced_subsample",  # recompute on each bootstrap sample
    n_estimators=200,
    random_state=42
)
# Or combine with SMOTE for extreme imbalance:
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
])
```

---

### §6.7 — High Dimensionality with Sparse Signals

**Mechanism:** When $p \gg n$ (more features than samples), especially when most features are
uninformative (sparse signal), RF struggles. With `max_features="sqrt"`, each split evaluates
$\sqrt{p}$ features randomly. If only $S \ll p$ features are truly informative, the probability
of selecting at least one informative feature at any given node is:

$$P(\text{at least one informative}) = 1 - \binom{p-S}{\sqrt{p}}/\binom{p}{\sqrt{p}} \approx 1 - \left(1 - \frac{S}{p}\right)^{\sqrt{p}}$$

For $S = 5$, $p = 10000$: this is approximately $1 - (0.9995)^{100} \approx 0.049$ — only 5%
of splits involve a truly informative feature. Trees become effectively random.

**Mitigation:**
- Pre-filtering: use marginal feature selection (univariate F-test, mutual information) to
  reduce to top 100-500 features before RF
- Decrease `max_features` (e.g., to `"log2"`) to increase decorrelation with sparse signal
- Consider `Lasso` or `ElasticNet` for truly high-dimensional sparse problems

---

### §6.8 — `max_features="auto"` Deprecated — Legacy Code Breaks

In sklearn ≥ 1.1, `max_features="auto"` raises a `ValueError`. Before 1.1, `"auto"` meant
`"sqrt"` for classifiers and `"auto"` = $p$ for regressors — silently different behaviors.

**Fix:** Replace `"auto"` with `"sqrt"` (classifiers) or `1.0` (regressors).

---

## §7 — Diagnostic Plots & Evaluation

### §7.1 — OOB Error Convergence Plot

**What it shows:** How the OOB error changes as trees are added. Good: smooth convergence to
a flat plateau. Bad: still dropping at your `n_estimators` limit (add more trees) or oscillating
widely (too few trees).

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

oob_errors = []
n_range = range(10, 301, 10)

rf = RandomForestClassifier(warm_start=True, oob_score=True, random_state=42, n_jobs=-1)
for n in n_range:
    rf.set_params(n_estimators=n)
    rf.fit(X_train, y_train)
    oob_errors.append(1 - rf.oob_score_)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(n_range), oob_errors, lw=2, color="steelblue")
ax.axvline(x=100, color="red", linestyle="--", alpha=0.5, label="Default n_estimators=100")
ax.set_xlabel("Number of Trees", fontsize=12)
ax.set_ylabel("OOB Error Rate", fontsize=12)
ax.set_title("OOB Error Convergence — breast_cancer", fontsize=13)
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
# Good: curve flattens well before n_estimators limit
# Bad: still declining at the right edge → add more trees
```

---

### §7.2 — Feature Importance Comparison (MDI vs Permutation vs SHAP)

**What it shows:** Which features drive model predictions, and whether the different importance
measures agree. Disagreements indicate bias (§6.1, §6.2).

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Fit final RF with oob_score for feature importance
rf_final = RandomForestClassifier(
    n_estimators=200, max_features="sqrt", oob_score=True, random_state=42, n_jobs=-1
)
rf_final.fit(X_train, y_train)

feature_names = data.feature_names

# MDI importances
mdi_imp = pd.Series(rf_final.feature_importances_, index=feature_names).sort_values(ascending=False)

# Permutation importances (on test set — unbiased)
perm_result = permutation_importance(
    rf_final, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1
)
perm_imp = pd.Series(perm_result.importances_mean, index=feature_names).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MDI plot
top_mdi = mdi_imp.head(15)
axes[0].barh(range(len(top_mdi)), top_mdi.values, color="steelblue")
axes[0].set_yticks(range(len(top_mdi)))
axes[0].set_yticklabels(top_mdi.index)
axes[0].set_title("MDI Importance (built-in)", fontsize=12)
axes[0].set_xlabel("Mean Decrease Impurity")

# Permutation importance with error bars
top_perm = perm_imp.head(15)
top_perm_std = pd.Series(perm_result.importances_std, index=feature_names)[top_perm.index]
axes[1].barh(range(len(top_perm)), top_perm.values, xerr=top_perm_std,
             color="coral", capsize=3)
axes[1].set_yticks(range(len(top_perm)))
axes[1].set_yticklabels(top_perm.index)
axes[1].set_title("Permutation Importance (test set)", fontsize=12)
axes[1].set_xlabel("Mean Accuracy Decrease")

plt.suptitle("Feature Importance: MDI vs Permutation", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

# Compare top features: large disagreements indicate cardinality bias
print("MDI top 5:", list(mdi_imp.head(5).index))
print("Perm top 5:", list(perm_imp.head(5).index))
```

**Good:** Both methods agree on the top features. The error bars on permutation importance are small.
**Bad:** MDI ranks a feature highly that permutation importance rates near zero → likely cardinality bias.

---

### §7.3 — Validation Curve (max_features sensitivity)

**What it shows:** How train and OOB (or CV) accuracy change as the most important hyperparameter
(`max_features`) varies.

```python
from sklearn.model_selection import validation_curve
import numpy as np
import matplotlib.pyplot as plt

param_range = [0.1, 0.2, 0.3, "sqrt", 0.5, 0.7, 1.0]
# Convert to numeric for plotting
param_range_numeric = [0.1, 0.2, 0.3, np.sqrt(X_train.shape[1])/X_train.shape[1], 0.5, 0.7, 1.0]

train_scores, val_scores = validation_curve(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    X_train, y_train,
    param_name="max_features",
    param_range=param_range,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.fill_between(range(len(param_range)), val_mean - val_std, val_mean + val_std, alpha=0.15, color="coral")
ax.plot(range(len(param_range)), train_mean, "o-", label="Train AUC", color="steelblue")
ax.plot(range(len(param_range)), val_mean, "o-", label="CV AUC", color="coral")
ax.set_xticks(range(len(param_range)))
ax.set_xticklabels([str(p) for p in param_range])
ax.set_xlabel("max_features"); ax.set_ylabel("AUC-ROC")
ax.set_title("Validation Curve — max_features")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
```

**Good:** CV AUC peaks at one of the middle values ("sqrt" or 0.3-0.5) and train AUC is not
much higher — moderate overfitting.
**Bad:** Train AUC = 1.0 everywhere, CV AUC << 1.0 → use lower `max_features` or `max_depth`.

---

### §7.4 — Confusion Matrix and Classification Report

```python
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                              RocCurveDisplay, roc_auc_score)
import matplotlib.pyplot as plt

y_pred = rf_final.predict(X_test)
y_proba = rf_final.predict_proba(X_test)[:, 1]  # probability of positive class

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"OOB Score: {rf_final.oob_score_:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=data.target_names, ax=axes[0], cmap="Blues"
)
axes[0].set_title("Confusion Matrix")

RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
axes[1].set_title(f"ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.4f})")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)

plt.tight_layout(); plt.show()
```

---

### §7.5 — Regression Diagnostics (Residuals + Prediction vs Actual)

```python
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

housing = fetch_california_housing()
X_h, y_h = housing.data, housing.target
X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(
    X_h, y_h, test_size=0.2, random_state=42
)

rf_reg = RandomForestRegressor(
    n_estimators=200, max_features=0.5, oob_score=True, random_state=42, n_jobs=-1
)
rf_reg.fit(X_h_train, y_h_train)
y_h_pred = rf_reg.predict(X_h_test)

rmse = np.sqrt(mean_squared_error(y_h_test, y_h_pred))
r2 = r2_score(y_h_test, y_h_pred)
print(f"RMSE: {rmse:.4f} | R²: {r2:.4f} | OOB R²: {rf_reg.oob_score_:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Predicted vs Actual
axes[0].scatter(y_h_test, y_h_pred, alpha=0.3, s=10, color="steelblue")
lims = [min(y_h_test.min(), y_h_pred.min()), max(y_h_test.max(), y_h_pred.max())]
axes[0].plot(lims, lims, "r--", lw=2)
axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
axes[0].set_title(f"Predicted vs Actual (R²={r2:.3f})")

# Residuals vs Predicted
residuals = y_h_test - y_h_pred
axes[1].scatter(y_h_pred, residuals, alpha=0.3, s=10, color="coral")
axes[1].axhline(0, color="black", lw=2)
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
axes[1].set_title("Residuals vs Predicted")
# Good: random scatter around 0. Bad: funnel shape (heteroscedasticity)
# Note RF-specific: clipping at max training value visible as cluster of large residuals

# Residual distribution
axes[2].hist(residuals, bins=50, color="steelblue", edgecolor="white")
axes[2].set_xlabel("Residual"); axes[2].set_ylabel("Count")
axes[2].set_title("Residual Distribution")

plt.tight_layout(); plt.show()
```

**Good:** Predicted vs Actual cloud lies tightly around the 45° diagonal. Residuals are
randomly scattered around zero with no pattern.
**Bad:** Residuals fan outward with predicted value (heteroscedasticity); large systematic
underprediction at high values (RF extrapolation failure — see §6.3).

---

### §7.6 — Partial Dependence Plots (PDP) and ICE Plots

**What it shows:** The marginal effect of one (or two) features on the prediction, averaging
over all other features. ICE plots show per-sample curves, revealing interaction effects.

```python
from sklearn.inspection import PartialDependenceDisplay

# Plot PDP + ICE for the two most important features
feature_names_list = list(housing.feature_names)
top2_features = [feature_names_list.index("MedInc"), feature_names_list.index("AveOccup")]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

PartialDependenceDisplay.from_estimator(
    rf_reg, X_h_train, features=top2_features,
    kind="both",        # "average" = PDP only; "individual" = ICE only; "both" = PDP + ICE
    subsample=500,      # random subset for ICE lines (performance)
    n_jobs=-1, grid_resolution=50,
    ax=axes,
    ice_lines_kw={"alpha": 0.05, "color": "steelblue"},
    pd_line_kw={"lw": 3, "color": "red"}
)
axes[0].set_title(f"PDP + ICE: {feature_names_list[top2_features[0]]}")
axes[1].set_title(f"PDP + ICE: {feature_names_list[top2_features[1]]}")
plt.suptitle("Partial Dependence + ICE Plots", fontsize=14)
plt.tight_layout(); plt.show()

# Good: ICE lines are parallel → no interaction with other features
# Bad: ICE lines cross → interaction effects present; PDP is misleading; use 2D PDP
```

---

### §7.7 — Calibration Curve

RF probability estimates from `predict_proba()` are not well-calibrated by default — they
tend to be pushed away from 0 and 1 (toward 0.5) because of the averaging of tree leaf
proportions. Always check calibration before using RF probabilities for decision-making.

```python
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 6))

# Uncalibrated RF
CalibrationDisplay.from_estimator(
    rf_final, X_test, y_test, n_bins=10, ax=ax, name="RF (uncalibrated)"
)

# Calibrated RF (isotonic regression)
rf_calibrated = CalibratedClassifierCV(rf_final, method="isotonic", cv="prefit")
rf_calibrated.fit(X_test, y_test)  # note: fitting calibration on test set is illustrative
# In practice: use a separate calibration set from the training split
CalibrationDisplay.from_estimator(
    rf_calibrated, X_test, y_test, n_bins=10, ax=ax, name="RF (isotonic)"
)

ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.set_title("Calibration Curve — breast_cancer")
ax.legend(); plt.tight_layout(); plt.show()
# Good: curve close to diagonal. Bad: S-curve below diagonal (overconfident)
```

---

## §8 — Explainability & Interpretable AI

### §8.1 — Built-in Interpretability: MDI and Decision Paths

Random Forests provide two native interpretability tools:

**1. MDI Feature Importances** (`rf.feature_importances_`): The average impurity reduction
contributed by each feature across all trees and all splits on that feature. Computed during
training at no extra cost. Already covered extensively in §6.1 — always pair with permutation
importance or SHAP to check for cardinality bias.

**2. Decision Path** (per-sample): For any sample $\mathbf{x}$, you can inspect which nodes
it passes through in each tree, and what decision rules were applied:

```python
# Decision path for a specific test sample (tree-level introspection)
sample_idx = 0
tree_idx = 0

# Path through one tree
node_indicator = rf_final.estimators_[tree_idx].decision_path(
    X_test[sample_idx:sample_idx+1]
)
print(f"Nodes visited in tree {tree_idx}: {node_indicator.indices}")

# Feature used at each node, threshold, and direction
tree = rf_final.estimators_[tree_idx]
feature_used = tree.tree_.feature
threshold = tree.tree_.threshold
for node_id in node_indicator.indices:
    if feature_used[node_id] >= 0:  # -2 = leaf
        feat_name = data.feature_names[feature_used[node_id]]
        val = X_test[sample_idx, feature_used[node_id]]
        direction = "<=" if val <= threshold[node_id] else ">"
        print(f"  Node {node_id}: {feat_name} = {val:.3f} {direction} {threshold[node_id]:.3f}")
```

---

### §8.2 — SHAP TreeExplainer: The Gold Standard

SHAP (SHapley Additive exPlanations) decomposes each prediction into the contribution of each
feature, guaranteed to satisfy three fairness properties: local accuracy (contributions sum to
prediction), missingness (absent features have zero contribution), and consistency (increasing
a feature's importance never decreases its SHAP value).

For tree-based models, SHAP provides `TreeExplainer` — a polynomial-time exact algorithm
($O(T \cdot L \cdot D)$ where $L$ = max leaves and $D$ = max depth) versus the exponential
cost of exact Shapley values for arbitrary models.

> 📜 **Citation:** Lundberg, S.M. et al. (2020). From local explanations to global understanding
> with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67.
> DOI: https://doi.org/10.1038/s42256-019-0138-9

```python
import shap
import matplotlib.pyplot as plt
import numpy as np

# ── Setup ──────────────────────────────────────────────────────────────────────
# Use a background sample for interventional SHAP (more theoretically principled)
background = X_train[:100]  # 100-200 samples is sufficient

explainer = shap.TreeExplainer(
    rf_final,
    data=background,
    model_output="probability",          # return SHAP values in probability space
    feature_perturbation="interventional" # causal independence assumption
)

# ── Compute SHAP values ────────────────────────────────────────────────────────
# For binary RandomForestClassifier:
# shap_values returns a list [shap_class0, shap_class1] each of shape (n, p)
# Use the new Explanation object API (SHAP 0.46+):
shap_values_obj = explainer(X_test)
# shap_values_obj has shape (n_test, n_features, n_classes) for multi-class
# For binary: shap_values_obj[:, :, 1] is the positive class

# Or use legacy API:
shap_values = explainer.shap_values(X_test, approximate=False, check_additivity=True)
# shap_values is list of 2 arrays (one per class) for binary classifier
# shap_values[1] = SHAP values for positive class, shape (n_test, n_features)

print(f"SHAP values shape (positive class): {shap_values[1].shape}")
print(f"Expected value (positive class): {explainer.expected_value[1]:.4f}")

# Additivity check: for any sample i:
# sum(shap_values[1][i]) + expected_value[1] ≈ predict_proba(X_test[i])[1]
i = 0
pred_check = np.sum(shap_values[1][i]) + explainer.expected_value[1]
actual_pred = rf_final.predict_proba(X_test[i:i+1])[0, 1]
print(f"SHAP sum + E[f(x)] = {pred_check:.4f} | predict_proba = {actual_pred:.4f}")
```

> ⚠️ **Verify:** The exact shape of `shap_values` for `RandomForestClassifier` has changed
> between SHAP versions. In SHAP ~0.46.x with the `Explanation` object API, binary classifier
> outputs may be consolidated differently than the list-of-arrays format from the legacy
> `shap_values()` call. Test both and check shapes before building plots.

---

#### §8.2.1 — SHAP Summary / Beeswarm Plot (Global Importance)

```python
# Beeswarm plot: global feature importance + direction + density
# Use SHAP values for positive class (index 1)
shap.summary_plot(
    shap_values[1],
    X_test,
    feature_names=list(data.feature_names),
    plot_type="dot",     # "dot" = beeswarm; "bar" = mean |SHAP|; "violin" = violin
    max_display=15,
    show=True
)
# Reading it:
# X axis: SHAP value → positive = increases P(positive class)
# Color: feature value (red=high, blue=low)
# Width: density of samples at that SHAP value
# Good: clear separation between high/low feature values
# Bad: wide spread with no color pattern → feature adds noise
```

---

#### §8.2.2 — SHAP Waterfall Plot (Local Explanation for One Prediction)

```python
# Waterfall plot: decompose one prediction into feature contributions
# Shows the "story" of how we got from the base rate to this individual prediction
sample_idx = 5  # choose an interesting sample

if hasattr(shap_values_obj, '__getitem__'):
    # Modern Explanation object
    shap.plots.waterfall(shap_values_obj[sample_idx, :, 1])
else:
    # Legacy API
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[1][sample_idx],
            base_values=explainer.expected_value[1],
            data=X_test[sample_idx],
            feature_names=list(data.feature_names)
        )
    )
# Reading it:
# Gray bar at bottom = E[f(x)] = base rate across background data
# Colored bars = each feature's contribution (red pushes prediction up, blue pushes down)
# Final bar = f(x) = actual prediction for this sample
```

---

#### §8.2.3 — SHAP Force Plot (HTML visualization)

```python
# Force plot: compact visualization of a single prediction
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][0],
    X_test[0],
    feature_names=list(data.feature_names)
    # Returns an HTML object; in Jupyter use shap.initjs() first
)
# Reading it: features in red push prediction higher; blue push it lower
# Width proportional to magnitude of contribution
```

---

#### §8.2.4 — SHAP Dependence Plot (Feature Interaction)

```python
# Dependence plot: SHAP value vs feature value, colored by interaction feature
top_feature = "worst radius"  # typically the most important feature in breast_cancer
shap.dependence_plot(
    top_feature,
    shap_values[1],
    X_test,
    feature_names=list(data.feature_names),
    interaction_index="auto"  # auto-detect strongest interacting feature
)
# Reading it:
# X axis: feature value
# Y axis: SHAP value (its contribution to prediction)
# Color: automatically selected interacting feature
# Nonlinear curves reveal complex effects; color gradients reveal interaction effects
```

---

### §8.3 — Permutation Importance (When to Prefer Over MDI)

Permutation importance (also called Mean Decrease Accuracy, MDA) measures how much the model's
performance degrades when a feature's values are randomly shuffled:

$$\text{PI}(j) = \frac{1}{K} \sum_{k=1}^K \left[ \text{score}(f, X, y) - \text{score}(f, X^{(\pi_j)}, y) \right]$$

where $X^{(\pi_j)}$ has column $j$ permuted (breaking the association between feature $j$ and $y$).

**When to prefer permutation over MDI:**
- When features have different cardinalities (avoids Strobl 2007 bias)
- When you need a model-agnostic importance metric (same API for any model)
- When you want importance measured in terms of a specific metric (AUC, F1, etc.)
- When features are correlated (though note the Strobl 2008 caveat — both methods struggle)

**When MDI is acceptable:**
- Quick first-pass importance during exploratory analysis
- When all features are continuous and similar in cardinality
- When compute is limited (MDI is free; permutation importance requires multiple forward passes)

```python
from sklearn.inspection import permutation_importance

perm = permutation_importance(
    rf_final, X_test, y_test,
    n_repeats=30,       # repeat permutation 30 times for stable estimates
    random_state=42,
    n_jobs=-1,
    scoring="roc_auc"   # use AUC, not accuracy
)

# Sort by mean importance
sorted_idx = perm.importances_mean.argsort()[::-1]

fig, ax = plt.subplots(figsize=(8, 8))
ax.boxplot(
    perm.importances[sorted_idx[:15]].T,
    vert=False,
    labels=np.array(data.feature_names)[sorted_idx[:15]]
)
ax.set_title("Permutation Importance (test set AUC, 30 repeats)")
ax.set_xlabel("Decrease in AUC")
ax.axvline(0, color="k", lw=1, linestyle="--")
plt.tight_layout(); plt.show()
# Features with error bars crossing zero: not reliably important
# Features with large, non-zero importance: genuine predictive drivers
```

---

### §8.4 — LIME (Local Interpretable Model-agnostic Explanations)

LIME approximates the RF locally around a specific prediction by training a simple linear
model on perturbed samples near $\mathbf{x}$. It works with any black-box model, making it
more general than SHAP TreeExplainer (which is RF-specific).

**When to use LIME over SHAP:**
- When you need explanations that non-technical stakeholders can verify (linear + feature names)
- When the SHAP TreeExplainer API is unavailable for your specific model variant
- When you want to expose the explanation to the end user in a simple tabular format

**When SHAP is better:**
- Theoretical guarantees (Shapley axioms); LIME has no uniqueness guarantee
- Faster for large forests (TreeExplainer is $O(T \cdot L)$; LIME requires many forward passes)
- More faithful globally (LIME is local; SHAP summarizes both)

```python
# pip install lime
from lime.lime_tabular import LimeTabularExplainer

lime_explainer = LimeTabularExplainer(
    X_train,
    feature_names=list(data.feature_names),
    class_names=list(data.target_names),
    mode="classification"
)

# Explain one prediction
i = 5  # sample index
lime_exp = lime_explainer.explain_instance(
    X_test[i],
    rf_final.predict_proba,
    num_features=10,     # show top 10 contributing features
    top_labels=1
)
lime_exp.show_in_notebook()   # in Jupyter
# Or: lime_exp.as_list() for text summary
```

> ⚠️ **Pitfall:** LIME samples in the feature space using a Gaussian perturbation model,
> which may not match the true data distribution. For highly non-Gaussian or correlated features,
> LIME explanations can be unreliable. Prefer SHAP for RF.

---

### §8.5 — Caveats: When Explanations Can Mislead

**1. SHAP with correlated features:** When $X_1 \approx X_2$ are correlated, SHAP distributes
credit between them based on the Shapley value formula, but the allocation is sensitive to
the specific tree structure. The sum is correct but the per-feature attribution can be arbitrary
between correlated features. Use `feature_perturbation="tree_path_dependent"` for consistency.

**2. MDI in presence of correlated features:** MDI double-counts importance when features are
correlated (both may appear in trees and accumulate impurity reduction jointly).

**3. Global SHAP vs local consistency:** The SHAP beeswarm plot aggregates local explanations.
If the model is highly nonlinear, the global summary may obscure important local behaviors
in specific subpopulations.

**4. Explanation stability:** Individual SHAP waterfall plots can vary significantly between
similar samples due to the tree ensemble averaging. Always interpret local SHAP plots in the
context of the broader distribution.

> 🏆 **Best practice:** Use a three-layer interpretability stack:
> 1. **MDI** for quick first-pass signal (cheap, but biased)
> 2. **Permutation importance** for unbiased global ranking (test-set based)
> 3. **SHAP TreeExplainer** for both global (beeswarm) and local (waterfall/force) explanations
>
> Only trust conclusions that all three methods agree on.

---

## §9 — Complete Python Worked Example

This section walks through a full end-to-end Random Forest workflow on two named datasets:
`breast_cancer` (classification) and `california_housing` (regression). Every step is shown,
from raw data to final interpretation.

### §9.1 — Classification on breast_cancer

```python
# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM FOREST COMPLETE WORKED EXAMPLE — Classification (breast_cancer)
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                              ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Step 1: Load Data + EDA ────────────────────────────────────────────────────
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 0=malignant, 1=benign

print("=" * 60)
print("DATASET SUMMARY")
print(f"Shape: {X.shape}")
print(f"Classes: {dict(zip(data.target_names, np.bincount(y)))}")
print(f"Missing values: {X.isna().sum().sum()}")
print(f"\nFeature stats (first 5 features):")
print(X.iloc[:, :5].describe().round(3))

# Class balance check
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
y.value_counts().plot(kind="bar", ax=axes[0], color=["salmon", "steelblue"])
axes[0].set_xticklabels(data.target_names, rotation=0)
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Count")

# Correlation heatmap of top features
sns.heatmap(
    X.iloc[:, :10].corr(),
    ax=axes[1], cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    annot=False, square=True
)
axes[1].set_title("Feature Correlations (top 10)")
plt.tight_layout(); plt.show()

# ── Step 2: Train/Val/Test Split (stratified) ──────────────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(
    X.values, y.values, test_size=0.15, random_state=42, stratify=y.values
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
)

print(f"\nSplit sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Train class balance: {dict(zip(data.target_names, np.bincount(y_train)))}")

# ── Step 3: Baseline RF (no tuning) ───────────────────────────────────────────
# Note: RF does not require feature scaling
rf_base = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
rf_base.fit(X_train, y_train)

print("\nBASELINE RF (no tuning):")
print(f"  OOB Accuracy: {rf_base.oob_score_:.4f}")
print(f"  Val AUC:      {roc_auc_score(y_val, rf_base.predict_proba(X_val)[:,1]):.4f}")
print(f"  Val Accuracy: {rf_base.score(X_val, y_val):.4f}")

# ── Step 4: OOB Convergence Diagnostic ────────────────────────────────────────
oob_errors, oob_train_sizes = [], []
rf_warm = RandomForestClassifier(
    warm_start=True, oob_score=True, random_state=42, n_jobs=-1
)
for n in range(10, 251, 10):
    rf_warm.set_params(n_estimators=n)
    rf_warm.fit(X_train, y_train)
    oob_errors.append(1 - rf_warm.oob_score_)
    oob_train_sizes.append(n)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(oob_train_sizes, oob_errors, color="steelblue", lw=2)
ax.set_xlabel("n_estimators"); ax.set_ylabel("OOB Error")
ax.set_title("OOB Error Convergence"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
# Set n_estimators just past the elbow

# ── Step 5: Hyperparameter Tuning with Optuna ──────────────────────────────────
# Using train+val combined for CV (we have a separate test set for final eval)
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

def rf_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5]
        ),
        "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": 1,  # parallelize at Optuna level, not tree level
    }
    rf = RandomForestClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_trainval, y_trainval, cv=cv, scoring="roc_auc")
    return scores.mean()

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(rf_objective, n_trials=80, n_jobs=4, show_progress_bar=False)

print(f"\nOPTUNA TUNING RESULTS:")
print(f"  Best CV AUC: {study.best_value:.4f}")
print(f"  Best params: {study.best_params}")

# ── Step 6: Final Model with Best Params ──────────────────────────────────────
best_params = study.best_params.copy()
best_params.update({"oob_score": True, "random_state": 42, "n_jobs": -1})
rf_tuned = RandomForestClassifier(**best_params)
rf_tuned.fit(X_trainval, y_trainval)

y_test_pred = rf_tuned.predict(X_test)
y_test_proba = rf_tuned.predict_proba(X_test)[:, 1]

print(f"\nFINAL TUNED RF ON TEST SET:")
print(f"  Test AUC:      {roc_auc_score(y_test, y_test_proba):.4f}")
print(f"  Test Accuracy: {rf_tuned.score(X_test, y_test):.4f}")
print(f"  OOB Accuracy:  {rf_tuned.oob_score_:.4f}")
print(f"\n{classification_report(y_test, y_test_pred, target_names=data.target_names)}")

# ── Step 7: Diagnostic Plots ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_test_pred,
    display_labels=data.target_names,
    ax=axes[0, 0], cmap="Blues"
)
axes[0, 0].set_title("Confusion Matrix (test set)")

# ROC Curve
RocCurveDisplay.from_predictions(y_test, y_test_proba, ax=axes[0, 1])
axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0, 1].set_title(f"ROC (AUC={roc_auc_score(y_test, y_test_proba):.4f})")

# MDI Feature Importances (top 15)
feature_imp = pd.Series(rf_tuned.feature_importances_, index=data.feature_names)
feature_imp.nlargest(15).sort_values().plot(kind="barh", ax=axes[1, 0], color="steelblue")
axes[1, 0].set_title("MDI Feature Importances (top 15)")
axes[1, 0].set_xlabel("Mean Decrease Impurity")

# Calibration curve
CalibrationDisplay.from_estimator(
    rf_tuned, X_test, y_test, n_bins=10, ax=axes[1, 1], name="RF (tuned)"
)
axes[1, 1].plot([0, 1], [0, 1], "k--", label="Perfect")
axes[1, 1].set_title("Calibration Curve")
axes[1, 1].legend()

plt.suptitle("Random Forest — Diagnostic Plots (breast_cancer)", fontsize=14)
plt.tight_layout(); plt.show()

# ── Step 8: Feature Importance — MDI vs Permutation ───────────────────────────
perm = permutation_importance(
    rf_tuned, X_test, y_test, n_repeats=30, random_state=42,
    n_jobs=-1, scoring="roc_auc"
)
perm_series = pd.Series(perm.importances_mean, index=data.feature_names)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
feature_imp.nlargest(15).sort_values().plot(kind="barh", ax=axes[0], color="steelblue")
axes[0].set_title("MDI Importance (built-in)")
perm_series.nlargest(15).sort_values().plot(kind="barh", ax=axes[1], color="coral")
axes[1].set_title("Permutation Importance (test AUC)")
plt.suptitle("MDI vs Permutation Importance Comparison", fontsize=14)
plt.tight_layout(); plt.show()

print("\nTop 5 by MDI:", list(feature_imp.nlargest(5).index))
print("Top 5 by Permutation:", list(perm_series.nlargest(5).index))

# ── Step 9: SHAP Analysis ──────────────────────────────────────────────────────
# Background dataset for interventional SHAP
background = shap.sample(X_trainval, 100, random_state=42)

explainer = shap.TreeExplainer(
    rf_tuned,
    data=background,
    model_output="probability",
    feature_perturbation="interventional"
)

# Compute SHAP values (legacy API — returns list of arrays for binary classifier)
shap_vals = explainer.shap_values(X_test, approximate=False, check_additivity=True)
# shap_vals[1] = SHAP values for positive class (benign), shape (n_test, n_features)

print(f"\nSHAP VALUES:")
print(f"  Shape (positive class): {np.array(shap_vals[1]).shape}")
print(f"  Expected value (positive class): {explainer.expected_value[1]:.4f}")

# SHAP Summary (beeswarm)
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_vals[1],
    X_test,
    feature_names=list(data.feature_names),
    max_display=15,
    show=False
)
plt.title("SHAP Beeswarm — breast_cancer (benign class)")
plt.tight_layout(); plt.show()

# SHAP Waterfall for a malignant prediction
# Find a confidently-predicted malignant sample
malignant_idx = np.where((y_test == 0) & (y_test_proba < 0.2))[0]
if len(malignant_idx) > 0:
    idx = malignant_idx[0]
    print(f"\nWaterfall for sample {idx}: true={data.target_names[y_test[idx]]}, "
          f"P(benign)={y_test_proba[idx]:.3f}")
    shap.waterfall_plot(
        shap.Explanation(
            values=np.array(shap_vals[1][idx]),
            base_values=explainer.expected_value[1],
            data=X_test[idx],
            feature_names=list(data.feature_names)
        )
    )

# SHAP Dependence: top feature
top_feat = feature_imp.idxmax()
top_feat_idx = list(data.feature_names).index(top_feat)
shap.dependence_plot(
    top_feat, shap_vals[1], X_test,
    feature_names=list(data.feature_names),
    interaction_index="auto"
)
plt.title(f"SHAP Dependence: {top_feat}")
plt.show()

# ── Step 10: PDP / ICE ─────────────────────────────────────────────────────────
top2_idx = feature_imp.nlargest(2).index.tolist()
top2_pos = [list(data.feature_names).index(f) for f in top2_idx]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
PartialDependenceDisplay.from_estimator(
    rf_tuned, X_trainval, features=top2_pos,
    kind="both", subsample=300, n_jobs=-1,
    grid_resolution=50, ax=axes,
    ice_lines_kw={"alpha": 0.05, "color": "steelblue"},
    pd_line_kw={"lw": 3, "color": "red"}
)
for i, ax in enumerate(axes):
    ax.set_title(f"PDP + ICE: {top2_idx[i]}")
plt.suptitle("Partial Dependence + ICE Plots (breast_cancer)", fontsize=14)
plt.tight_layout(); plt.show()

# ── Step 11: Final Interpretation ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL MODEL INTERPRETATION")
print("=" * 60)
print(f"Model: RandomForestClassifier with {best_params['n_estimators']} trees")
print(f"Best config: max_features={best_params['max_features']}, "
      f"max_depth={best_params['max_depth']}")
print(f"\nPerformance:")
print(f"  Test AUC-ROC: {roc_auc_score(y_test, y_test_proba):.4f}")
print(f"  Test accuracy: {rf_tuned.score(X_test, y_test):.4f}")
print(f"\nTop predictive features (SHAP + Permutation agreement):")
top3 = perm_series.nlargest(3)
for feat, imp in top3.items():
    print(f"  {feat}: permutation importance = {imp:.4f}")
print(f"\nConclusion: The RF correctly identifies {'%.1f' % (rf_tuned.score(X_test, y_test)*100)}%"
      f" of breast cancer cases. The most important predictors are"
      f" {', '.join(perm_series.nlargest(3).index)}, consistent with domain knowledge"
      f" that tumor size/shape metrics are the strongest discriminators.")
```

---

### §9.2 — Regression on california_housing

```python
# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM FOREST REGRESSION — california_housing
# ═══════════════════════════════════════════════════════════════════════════════

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

housing = fetch_california_housing()
X_h = pd.DataFrame(housing.data, columns=housing.feature_names)
y_h = pd.Series(housing.target, name="MedHouseVal")

print("CALIFORNIA HOUSING DATASET")
print(f"Shape: {X_h.shape}")
print(f"Target range: [{y_h.min():.2f}, {y_h.max():.2f}]")
print(X_h.describe().round(2))

X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(
    X_h.values, y_h.values, test_size=0.2, random_state=42
)

# ── Baseline Regression RF ─────────────────────────────────────────────────────
# Note: max_features defaults to 1.0 (all features) for regressor
# Explicitly set to 0.5 per Breiman's recommendation
rf_reg = RandomForestRegressor(
    n_estimators=200,
    max_features=0.5,      # explicitly set; don't rely on default 1.0
    max_depth=None,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(X_h_train, y_h_train)
y_h_pred = rf_reg.predict(X_h_test)

rmse = np.sqrt(mean_squared_error(y_h_test, y_h_pred))
mae = mean_absolute_error(y_h_test, y_h_pred)
r2 = r2_score(y_h_test, y_h_pred)

print(f"\nRF Regression Performance (california_housing):")
print(f"  RMSE: {rmse:.4f} (units: $100k)")
print(f"  MAE:  {mae:.4f}")
print(f"  R²:   {r2:.4f}")
print(f"  OOB R²: {rf_reg.oob_score_:.4f}")

# ── Regression Diagnostics ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Predicted vs Actual
axes[0].scatter(y_h_test, y_h_pred, alpha=0.2, s=5, color="steelblue")
lim = [0, 5.5]
axes[0].plot(lim, lim, "r--", lw=2)
axes[0].set_xlim(lim); axes[0].set_ylim(lim)
axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
axes[0].set_title(f"Predicted vs Actual\n(R²={r2:.3f})")
# Note: RF clips predictions at training max (~5.0); true values extend to 5.0+

# Residuals
residuals = y_h_test - y_h_pred
axes[1].scatter(y_h_pred, residuals, alpha=0.2, s=5, color="coral")
axes[1].axhline(0, color="black", lw=2)
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
axes[1].set_title("Residuals vs Predicted")
# RF-specific: note underestimation (positive residuals) at high predicted values

# Residual histogram
axes[2].hist(residuals, bins=60, color="steelblue", edgecolor="none")
axes[2].set_xlabel("Residual"); axes[2].set_ylabel("Count")
axes[2].set_title("Residual Distribution")

plt.suptitle("RF Regression Diagnostics — california_housing", fontsize=14)
plt.tight_layout(); plt.show()

# ── SHAP for Regression ────────────────────────────────────────────────────────
background_h = shap.sample(X_h_train, 100, random_state=42)
explainer_h = shap.TreeExplainer(
    rf_reg, data=background_h,
    model_output="raw",  # for regression: raw = predicted value
    feature_perturbation="interventional"
)

# For regression: shap_values is a single (n, p) array, not a list
shap_h = explainer_h.shap_values(X_h_test[:500], approximate=False)
print(f"\nRegression SHAP values shape: {np.array(shap_h).shape}")

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_h, X_h_test[:500],
    feature_names=housing.feature_names,
    max_display=8, show=False
)
plt.title("SHAP Beeswarm — california_housing (RegRF)")
plt.tight_layout(); plt.show()

# 2D PDP: MedInc x Latitude (geographic effect)
fig, ax = plt.subplots(figsize=(9, 7))
PartialDependenceDisplay.from_estimator(
    rf_reg, X_h_train,
    features=[(housing.feature_names.tolist().index("MedInc"),
               housing.feature_names.tolist().index("Latitude"))],
    kind="average", grid_resolution=30, ax=ax
)
ax.set_title("2D PDP: Median Income × Latitude")
plt.tight_layout(); plt.show()
```


---

## §10 — When to Use Random Forests (Decision Guide)

### §10.1 — Flowchart

```
START: You have a supervised learning problem
│
├─ Is your data tabular (rows = samples, cols = features)?
│   ├─ NO → Consider CNN/RNN/Transformer for images/text/sequences
│   └─ YES → Continue
│
├─ How many samples do you have?
│   ├─ < 100 samples → Consider logistic regression, SVM (better with small data)
│   ├─ 100 – 1,000,000 samples → RF is an excellent choice ✓
│   └─ > 1M samples → RF works; consider LightGBM (faster) or cuML RF (GPU)
│
├─ Do you need to extrapolate beyond training target range?
│   ├─ YES (time-series forecasting: future > historical max) → RF will fail
│   │   → Use GBM with trend features, or a linear model for the trend component
│   └─ NO → Continue
│
├─ Do you need sub-millisecond inference latency?
│   ├─ YES → Consider model distillation, XGBoost with few trees, or decision tables
│   └─ NO → Continue
│
├─ Are features predominantly sparse / high-dimensional (NLP, genomics, p >> n)?
│   ├─ YES → Consider L1 logistic regression, linear SVM, or LightGBM with sparse support
│   └─ NO → Continue
│
├─ Is the dataset severely imbalanced (< 5% minority class)?
│   ├─ YES → Use RF with class_weight="balanced_subsample"; optionally add SMOTE
│   └─ Continue
│
├─ Do you need online / streaming updates?
│   ├─ YES → Consider Mondrian Forests or Adaptive RF (river library)
│   └─ Continue
│
└─ RF is likely a strong choice. Start with defaults, then compare with LightGBM.
```

### §10.2 — Decision Table

| Problem Characteristic | RF | GBM (XGBoost/LGBM) | Logistic Regression | Neural Network |
|---|---|---|---|---|
| Tabular, medium size (1k–500k) | **Excellent** | Excellent | Good | Moderate |
| Tabular, large (>500k rows) | Good (cuML for GPU) | **Excellent** | Excellent | Good |
| High-dimensional sparse (p >> n) | Poor | Moderate | **Excellent** (L1) | Good |
| Missing values (no imputation) | **Excellent** (v1.4+) | Excellent | Poor | Poor |
| Categorical features (native) | Good (ordinal encode) | **Excellent** (CatBoost) | Moderate | Moderate |
| Nonlinear interactions | **Excellent** | Excellent | Poor | Excellent |
| Extrapolation beyond range | **Fails** | **Fails** | Good | Good |
| Calibrated probabilities needed | Moderate (post-cal.) | Moderate | **Excellent** | Good |
| No feature scaling needed | **Yes** | **Yes** | No | No |
| Interpretability (global) | Good | Good | **Excellent** | Poor |
| Interpretability (local, SHAP) | **Excellent** | Excellent | Good | Poor |
| Robustness to hyperparameters | **Excellent** | Moderate | Good | Poor |
| Streaming / online learning | Poor (standard) | Poor | Good (SGD) | Good (SGD) |
| Imbalanced classes | Good (`class_weight`) | Good | Good | Moderate |
| Time-series with trends | **Poor** | Poor | **Excellent** | Good |
| Training speed (CPU) | Good (parallel) | Moderate | **Excellent** | Moderate |
| Training speed (GPU) | Excellent (cuML) | **Excellent** | Good | Excellent |

### §10.3 — RF vs GBM: The Practical Comparison

Random Forest and Gradient Boosting Machines (XGBoost, LightGBM, CatBoost) dominate tabular ML.
Here is the decision logic practitioners use:

**Choose RF when:**
- You want a reliable strong baseline quickly — default RF with `n_estimators=200` rarely
  embarrasses itself
- Compute for hyperparameter tuning is limited (RF is far more robust to defaults than GBMs)
- Strict parallelism is required (RF trees are independent → near-linear CPU scaling)
- You need OOB error as a free validation estimate
- Your data has many missing values (sklearn 1.4+ handles them natively, zero imputation needed)
- Monotonicity constraints are required (sklearn 1.4+ `monotonic_cst`)
- You need a quick prototype before committing to full GBM tuning

**Choose GBM (XGBoost/LightGBM) when:**
- Maximizing predictive accuracy justifies extended hyperparameter tuning
- Dataset is very large and histogram-based binning is needed for efficiency
- Native categorical handling is required without preprocessing (CatBoost)
- Early stopping on a validation set is needed to automatically control overfitting
- Memory is very tight (LightGBM's leaf-wise growth uses less memory for equivalent depth)

> 📊 **Benchmark:** Grinsztajn et al. (2022, "Why tree-based models still outperform deep
> learning on tabular data", *NeurIPS*) showed RF and GBMs consistently outperform deep learning
> on standard tabular benchmarks, with RF offering the better hyperparameter robustness profile.
> LightGBM wins on pure accuracy by a small margin when heavily tuned.

### §10.4 — When NOT to Use RF

| Scenario | Why RF Fails | Better Alternative |
|---|---|---|
| Time-series with trend (forecasting) | Prediction bounded by training range | GBM + trend features, ARIMA, Prophet |
| Text classification (bag-of-words) | Sparse features → random subsets meaningless | TF-IDF + LogReg, fine-tuned BERT |
| Image classification | Pixel features lack spatial inductive bias | CNNs |
| Online learning with concept drift | Cannot update without full retrain | Mondrian Forests, Adaptive RF (river) |
| Very small datasets (< 50 samples) | Bootstrap samples too similar | SVM, logistic regression |
| Genomics / ultra-high-dimensional (p > 50k, n < 1k) | Signal detection at √p features fails | Sparse linear models, L1 regression |
| Sub-millisecond single-sample inference | 500 trees × 20 nodes each too slow | Distillation, XGBoost with few trees |

---

## §11 — Resources & Further Reading

### §11.1 — The Foundational Papers

**[1] Breiman, L. (2001). Random Forests.**
*Machine Learning*, 45(1), 5–32.
DOI: https://doi.org/10.1023/A:1010933404324
Semantic Scholar: https://www.semanticscholar.org/paper/Random-Forests-Breiman/8e0be569ea77b8cb29bb0e8b031887630fe7a96c
*The paper that started it all. Unusually clear prose for a theory paper — read Sections 2 (algorithm), 3 (OOB error), and 5 (variable importance) in full.*

**[2] Breiman, L. (1996). Bagging Predictors.**
*Machine Learning*, 24(2), 123–140.
*The predecessor that motivated RF. Understanding where bagging fell short clarifies RF's core contribution.*

**[3] Geurts, P., Ernst, D. & Wehenkel, L. (2006). Extremely randomized trees.**
*Machine Learning*, 63(1), 3–42.
DOI: https://doi.org/10.1007/s10994-006-6226-1
PDF: https://jasonphang.com/files/extratrees.pdf
*Essential companion to the RF paper. The bias-variance analysis in Section 4 is the clearest treatment in the literature.*

**[4] Strobl, C., Boulesteix, A.-L., Zeileis, A. & Hothorn, T. (2007). Bias in random forest variable importance measures: Illustrations, sources and a solution.**
*BMC Bioinformatics*, 8, 25.
PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC1796903/
*Required reading before using `rf.feature_importances_` for any consequential decision. The simulation study is a masterclass in demonstrating algorithmic bias.*

**[5] Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T. & Zeileis, A. (2008). Conditional variable importance for random forests.**
*BMC Bioinformatics*, 9, 307.
PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC2491635/
*Follow-up showing permutation importance also fails for correlated features. The hierarchical clustering solution is described here and implemented in the sklearn multicollinear example.*

**[6] Liu, F.T., Ting, K.M. & Zhou, Z.-H. (2012). Isolation-Based Anomaly Detection.**
*ACM TKDD*, 6(1), Article 3.
DOI: https://doi.org/10.1145/2133360.2133363
*The journal version of Isolation Forest with a cleaner derivation of the path-length anomaly score normalization.*

**[7] Biau, G. (2012). Analysis of a Random Forests Model.**
*JMLR*, 13(4), 1063–1095.
arXiv: https://arxiv.org/abs/1005.0208
*The first rigorous consistency proof for RF. Chapters 2–3 establish the adaptation-to-sparsity convergence rate that explains why RF handles many irrelevant features gracefully.*

**[8] Lakshminarayanan, B., Roy, D.M. & Teh, Y.W. (2014). Mondrian Forests: Efficient Online Random Forests.**
*NeurIPS*, 27.
arXiv: https://arxiv.org/abs/1406.2673
PDF: https://www.gatsby.ucl.ac.uk/~balaji/mondrian_forests_nips14.pdf

**[9] Tomita, T.M. et al. (2020). Sparse Projection Oblique Randomer Forests.**
*JMLR*, 21(104), 1–39.
JMLR: https://jmlr.org/papers/v21/18-664.html

**[10] Zhou, Z.-H. & Feng, J. (2019). Deep Forest.**
*National Science Review*, 6(1), 74–86.
arXiv: https://arxiv.org/abs/1702.08835

**[11] Lundberg, S.M., Erion, G., Chen, H. et al. (2020). From local explanations to global understanding with explainable AI for trees.**
*Nature Machine Intelligence*, 2, 56–67.
DOI: https://doi.org/10.1038/s42256-019-0138-9
*The TreeSHAP paper — the theoretical foundation for all SHAP-based RF interpretability.*

**[12] Rainforth, T. & Wood, F. (2015). Canonical Correlation Forests.**
arXiv:1507.05444.
URL: https://arxiv.org/abs/1507.05444

---

### §11.2 — Books

**Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*. 2nd ed.**
Springer. Free PDF: https://hastie.su.domains/ElemStatLearn/
**Chapter 15** (Random Forests): covers the variance-bias derivation, correlation between trees, and the relationship to bagging. The clearest book-length treatment.

**Breiman, L., Friedman, J., Olshen, R. & Stone, C. (1984). *Classification and Regression Trees*.**
Chapman & Hall. ("CART book")
The foundation for impurity criteria, tree growing, and the pruning techniques that RF builds on. Chapters 2–5 are essential context.

**Murphy, K.P. (2022). *Probabilistic Machine Learning: An Introduction*.**
MIT Press. https://probml.github.io/pml-book/
**Chapter 18** (Trees, Forests, Boosting): modern probabilistic treatment integrating uncertainty quantification perspectives.

**Molnar, C. (2022). *Interpretable Machine Learning*. 2nd ed.**
https://christophm.github.io/interpretable-ml-book/ (free)
**Chapters 8.5 and 9**: the most readable practitioner treatment of permutation importance and PDP for tree models.

---

### §11.3 — Official Documentation

- **sklearn 1.5 RandomForestClassifier API:** https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- **sklearn 1.5 Ensemble User Guide — RF section:** https://scikit-learn.org/1.5/modules/ensemble.html#forest
- **sklearn Permutation Importance vs MDI example:** https://scikit-learn.org/1.5/auto_examples/inspection/plot_permutation_importance.html
- **sklearn OOB Error convergence example:** https://scikit-learn.org/1.5/auto_examples/ensemble/plot_ensemble_oob.html
- **sklearn Multicollinear features + permutation importance:** https://scikit-learn.org/1.5/auto_examples/inspection/plot_permutation_importance_multicollinear.html
- **SHAP TreeExplainer docs:** https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
- **cuML RF (RAPIDS) docs:** https://docs.rapids.ai/api/cuml/stable/
- **h2o.ai DRF docs:** https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/drf.html
- **treeple (SPORF) docs:** https://docs.neurodata.io/treeple/

---

### §11.4 — Practitioner Resources

**"Beware Default Random Forest Importances" — Terence Parr & Jeremy Howard (explained.ai):**
https://explained.ai/rf-importance/
*The single most accessible treatment of MDI bias with a concrete NYC apartment data walkthrough. Every RF practitioner should read this.*

**NVIDIA Blog: Accelerating Random Forests up to 45x with cuML:**
https://developer.nvidia.com/blog/accelerating-random-forests-up-to-45x-using-cuml/
*Concrete GPU vs CPU benchmark numbers across dataset sizes. Use to decide when cuML is worth the setup cost.*

**SHAP Tree SHAP Tutorial notebooks:**
https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/
*Step-by-step notebooks covering RF + SHAP for binary classification, multi-class, and regression.*

**Optuna documentation:**
https://optuna.readthedocs.io/en/stable/
*Reference for the TPE-based hyperparameter search used throughout this chapter.*

---

### §11.5 — Uncertainties Flagged for Verification

Before production use or publication, verify these items against current documentation:

1. **SHAP `approximate` parameter location:** In SHAP ≥ 0.41+, `approximate` may only be accepted in `shap_values()`, not `TreeExplainer.__init__`. Check https://github.com/shap/shap/releases for your version.

2. **SHAP multi-class return shape (SHAP 0.46.x):** For 3+ class `RandomForestClassifier`, verify whether `explainer.shap_values(X)` returns a list of arrays (one per class) or a unified 3D array.

3. **cuML compatibility with sklearn 1.5 parameters:** cuML 26.06.00 may not support `monotonic_cst` or other sklearn 1.4/1.5 additions. Check https://docs.rapids.ai/api/cuml/stable/ for parity status.

4. **h2o.ai DRF current version:** Verify current h2o package version and any API changes to `H2ORandomForestEstimator` at https://docs.h2o.ai/h2o/latest-stable/

5. **treeple install command and version:** Verify `pip install treeple` and `from treeple.ensemble import ObliqueRandomForestClassifier` at https://docs.neurodata.io/treeple/

6. **gcForest Python 3.10+ compatibility:** Both known implementations (pylablanche/gcForest, kingfengji/gcForest) may be unmaintained. Verify before recommending.

7. **Mondrian Forests active Python package:** `skgarden` appears largely unmaintained. Check for current AMF (Aggregated Mondrian Forests) implementation.

8. **`oob_score` callable scorer signature (sklearn 1.4+):** Verify the exact callable signature — does it follow the standard `scorer(y_true, y_score)` protocol or the sklearn scorer API with `(estimator, X, y)`?

---

*Chapter complete. For questions, corrections, or extensions, see the course repository issues.*
