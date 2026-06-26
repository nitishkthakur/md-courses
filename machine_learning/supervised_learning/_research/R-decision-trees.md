# Research Dossier: Decision Trees
**Chapter file:** `decision-trees.md`
**Prepared for:** Supervised Learning Masterclass (internal authoring guide)
**Date:** 2026-06-25
**Researcher:** Claude Code (Sonnet 4.6)

---

## Scope

This dossier feeds the chapter file `decision-trees.md`. It covers the full lineage from ID3 (1986) through C4.5 (1993) through CART (1984), with complete sklearn 1.5.x+ API details, all hyperparameters, CCP pruning workflow, missing value support, monotonic constraints, oblique and optimal tree variants, visualization options (sklearn, dtreeviz), SHAP TreeExplainer, and practitioner pitfalls.

---

## Verified Packages

### Core

| Package | Import | What it implements | Version (mid-2026) | License |
|---|---|---|---|---|
| scikit-learn | `from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor` | CART (Gini/entropy/MSE, binary splits, CCP pruning) | ~1.9.x stable (course bible says 1.5.x target — use 1.5+ APIs) | BSD-3 |
| dtreeviz | `import dtreeviz` | Beautiful tree visualizations (adaptor pattern, supports sklearn/XGBoost/LightGBM) | 2.x | MIT |
| shap | `import shap; explainer = shap.TreeExplainer(model)` | Tree SHAP (exact, polynomial-time) | ~0.46.x | MIT |

### Key Import Paths (verified)

```python
# Core sklearn tree API
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    export_graphviz,      # DOT-format export (requires graphviz)
    plot_tree,            # inline matplotlib (added v0.21)
    export_text,          # human-readable text rules (added v0.21.2)
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap
import dtreeviz           # pip install dtreeviz  (requires graphviz system binary)
```

### DecisionTreeClassifier — Complete Verified Signature (sklearn 1.9.x current stable)

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(
    criterion='gini',              # 'gini' | 'entropy' | 'log_loss'
    splitter='best',               # 'best' | 'random'
    max_depth=None,                # int or None
    min_samples_split=2,           # int or float (fraction of n_samples)
    min_samples_leaf=1,            # int or float
    min_weight_fraction_leaf=0.0,  # float
    max_features=None,             # int | float | 'sqrt' | 'log2' | None
    random_state=None,             # int | RandomState | None
    max_leaf_nodes=None,           # int or None
    min_impurity_decrease=0.0,     # float
    class_weight=None,             # dict | list of dict | 'balanced' | None
    ccp_alpha=0.0,                 # non-negative float (added v0.22)
    monotonic_cst=None,            # array-like of {-1, 0, 1} (added v1.4)
)
```

### DecisionTreeRegressor — Complete Verified Signature

```python
from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(
    criterion='squared_error',     # 'squared_error' | 'absolute_error' | 'poisson'
                                   # NOTE: 'friedman_mse' deprecated in v1.9 (same math as squared_error)
    splitter='best',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    ccp_alpha=0.0,
    monotonic_cst=None,            # added v1.4; not supported for multioutput
)
```

### export_graphviz — Verified Signature

```python
sklearn.tree.export_graphviz(
    decision_tree,
    out_file=None,            # None → returns DOT string
    *,
    max_depth=None,
    feature_names=None,
    class_names=None,
    label='all',              # 'all' | 'root' | 'none'
    filled=False,
    leaves_parallel=False,
    impurity=True,
    node_ids=False,
    proportion=False,
    rotate=False,
    rounded=False,
    special_characters=False,
    precision=3,
    fontname='helvetica'
)
```

### plot_tree — Verified Signature

```python
sklearn.tree.plot_tree(
    decision_tree,
    *,
    max_depth=None,
    feature_names=None,
    class_names=None,          # array-like of str or bool
    label='all',
    filled=False,
    impurity=True,
    node_ids=False,
    proportion=False,
    rounded=False,
    precision=3,
    ax=None,                   # matplotlib Axes; uses current axis if None
    fontsize=None              # auto-determined if None
)
# Returns: list of artists
```

### export_text — Verified Signature

```python
from sklearn.tree import export_text
report = export_text(
    decision_tree,
    feature_names=None,   # list of str or None
    class_names=None,     # list of str or None
    max_depth=10,
    spacing=3,
    decimals=2,
    show_weights=False
)
```

### SHAP TreeExplainer — Verified Signature

```python
shap.TreeExplainer(
    model,
    data=None,                       # background dataset (optional with tree_path_dependent)
    model_output='raw',              # 'raw' | 'probability' | 'log_loss' | method name
    feature_perturbation='auto',     # 'auto' | 'interventional' | 'tree_path_dependent'
    feature_names=None,
    approximate=<object>,            # deprecated
    link=None,
    linearize_link=None
)
```

Minimal working example:

```python
import shap
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
clf = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X, y)

explainer = shap.TreeExplainer(clf, data=X)
shap_values = explainer(X[:100])          # returns Explanation object

shap.plots.waterfall(shap_values[0])       # local: single prediction
shap.plots.beeswarm(shap_values)           # global: feature importance
```

**Note:** For multiclass (`n_classes > 2`), `explainer.shap_values(X)` returns a list of arrays of shape `[n_samples, n_features]`, one per class.

### dtreeviz — Current API (adaptor pattern)

```python
import dtreeviz
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
clf = DecisionTreeClassifier(max_depth=3).fit(iris.data, iris.target)

viz_model = dtreeviz.model(
    clf,
    X_train=iris.data,
    y_train=iris.target,
    feature_names=iris.feature_names,
    target_name='species',
    class_names=list(iris.target_names)
)
viz_model.view()                           # show the tree
viz_model.explain_prediction_path(iris.data[10])  # highlight decision path
```

**Note:** dtreeviz requires graphviz system binary (`brew install graphviz` on macOS).

---

## The Original Algorithm

### ID3 — Quinlan 1986

**Full citation:**
> Quinlan, J.R. (1986). Induction of Decision Trees. *Machine Learning*, 1(1), 81–106.
> DOI: 10.1007/BF00116251
> URL: https://link.springer.com/article/10.1007/BF00116251

**What problem it solved:** Before ID3, decision tree construction was ad hoc and manual. Concept learning had been framed as a search through a hypothesis space, but no algorithm systematically built trees from data. Quinlan's Iterative Dichotomiser 3 was the first principled, recursive, data-driven algorithm.

**Key innovation:** Use Shannon entropy and information gain to choose which attribute to split on at each node. Greedy, top-down, no backtracking.

**Information gain:**
$$IG(S, A) = H(S) - \sum_{v \in \text{values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

where entropy is:
$$H(S) = -\sum_{k} p_k \log_2 p_k$$

**ID3 Pseudocode:**
```
ID3(examples S, target_attribute, attributes):
  if all examples in S have same class:
      return leaf node with that class
  if attributes is empty:
      return leaf node with most common class in S
  A = attribute in attributes that maximizes IG(S, A)
  create decision node branching on A
  for each value v of A:
      S_v = subset of S where A = v
      if S_v is empty:
          add leaf node with most common class in S
      else:
          add subtree = ID3(S_v, target_attribute, attributes \ {A})
  return node
```

**Key limitations of ID3:**
- Categorical attributes only (cannot handle continuous features)
- No pruning (grows full tree → overfits)
- Biased toward attributes with many values (higher information gain by construction, even for irrelevant attributes)
- Cannot handle missing values
- No regression support

---

### CART — Breiman, Friedman, Olshen, Stone 1984

**Full citation:**
> Breiman, L., Friedman, J.H., Olshen, R.A., & Stone, C.J. (1984). *Classification and Regression Trees*. Wadsworth & Brooks/Cole Advanced Books & Software, Pacific Grove, CA. ISBN: 978-0-412-04841-8
> Semantic Scholar: https://www.semanticscholar.org/paper/Classification-and-Regression-Trees-(Wadsworth-Breiman-Friedman/2203c20aaefc87c72e494b45dc77ed10f3013cb5

**What problem it solved:** Existing statistical methods (logistic regression, discriminant analysis) were parametric and assumption-heavy. CART provided a fully nonparametric, interpretable method for both classification and regression. It also provided rigorous cross-validation-based pruning — the first principled solution to the overfitting problem in trees.

**Key innovations:**
1. **Binary splits only** — every internal node splits into exactly two children
2. **Gini impurity** — a new splitting criterion alternative to entropy
3. **Handles continuous and categorical features** — for continuous $x_j$, the split is $x_j \leq \theta$ for some threshold $\theta$
4. **Handles regression** — predict the mean $\bar{y}$ of samples in each leaf; split on variance reduction (MSE)
5. **Cost-complexity pruning** — a mathematically principled pruning procedure via the regularized objective $R_\alpha(T) = R(T) + \alpha |\tilde{T}|$

**Gini impurity at node $m$:**
$$H(Q_m) = \sum_{k} p_{mk}(1 - p_{mk}) = 1 - \sum_k p_{mk}^2$$

where $p_{mk}$ = fraction of class $k$ samples at node $m$.

**CART split criterion:** For a proposed split $(\theta, j)$ (feature $j$, threshold $\theta$) that partitions node $m$ into left set $Q_m^{\text{left}}(\theta)$ and right set $Q_m^{\text{right}}(\theta)$:

$$G(Q_m, \theta) = \frac{n_m^{\text{left}}}{n_m} H(Q_m^{\text{left}}(\theta)) + \frac{n_m^{\text{right}}}{n_m} H(Q_m^{\text{right}}(\theta))$$

Choose $(j^*, \theta^*)$ that minimizes $G(Q_m, \theta)$.

**Cost-complexity pruning:** Define the cost-complexity measure:
$$R_\alpha(T) = R(T) + \alpha |\tilde{T}|$$

where $R(T)$ = total weighted impurity of all terminal nodes, $|\tilde{T}|$ = number of terminal nodes, $\alpha \geq 0$ = complexity parameter. For any $\alpha$, there is a unique smallest minimizing subtree $T(\alpha)$. As $\alpha$ increases from 0 to $\infty$, we get a nested sequence of subtrees from the full tree to the root-only tree. The algorithm computes the effective alpha for pruning each internal node $t$:
$$\alpha_{\text{eff}}(t) = \frac{R(t) - R(T_t)}{|\tilde{T}_t| - 1}$$

where $R(t)$ = impurity of node $t$ treated as a leaf, and $R(T_t)$ = impurity of the subtree rooted at $t$.

**sklearn CCP workflow:**
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Step 1: Get the full pruning path
clf = DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]   # exclude root-only tree

# Step 2: Train one tree per alpha
clfs = [
    DecisionTreeClassifier(random_state=0, ccp_alpha=alpha).fit(X_train, y_train)
    for alpha in ccp_alphas
]

# Step 3: Select alpha by CV score (or test score)
test_scores = [clf.score(X_test, y_test) for clf in clfs]
best_alpha = ccp_alphas[np.argmax(test_scores)]

# Step 4: Retrain with chosen alpha (ideally use CV here, not test set)
final_clf = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=0)
final_clf.fit(X_train, y_train)
```

**Regression criterion (MSE):**
$$\bar{y}_m = \frac{1}{n_m}\sum_{y \in Q_m} y, \quad H(Q_m) = \frac{1}{n_m}\sum_{y \in Q_m}(y - \bar{y}_m)^2$$

**Friedman MSE (deprecated in sklearn 1.9):** An improvement score variant:
$$\text{FriedmanMSE} = \frac{n_l \cdot n_r \cdot (\bar{y}_l - \bar{y}_r)^2}{n_l + n_r}$$

This is mathematically equivalent to `squared_error` and was deprecated in sklearn 1.9 because it produces identical results.

---

### C4.5 — Quinlan 1993

**Full citation:**
> Quinlan, J.R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann Publishers, San Mateo, CA. ISBN: 978-1-558-60238-0

**What it solved over ID3:**
1. **Continuous attributes** — sort values, try each midpoint as a threshold
2. **Missing values** — distribute samples with missing values proportionally across branches
3. **Gain ratio** to fix information gain's bias toward many-valued attributes
4. **Pruning** — pessimistic error pruning (post-hoc subtree replacement)
5. **Multi-valued categorical splits** — native support

**Gain ratio:**
$$\text{GainRatio}(S, A) = \frac{IG(S, A)}{H_A(S)}$$

where $H_A(S) = -\sum_{v} \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}$ is the split information (entropy of the attribute values themselves). This penalizes attributes with many values.

**Pessimistic pruning:** Replace a subtree with a leaf if the estimated generalization error (using a continuity correction on the training error) does not increase. No separate validation set needed, but somewhat ad hoc.

---

### C5.0 / See5 — Quinlan 1997

**Commercial successor to C4.5.** Not open source. Available via R's `C50` package.

Key additions:
- Boosting integration (similar to AdaBoost)
- Substantially faster training (orders of magnitude)
- Case weighting and misclassification costs
- Global cost-complexity pruning (improvement over C4.5's pessimistic pruning)

**Reference:** Quinlan, J.R. (1997). *Data Mining Tools See5 and C5.0.* RuleQuest Research. https://www.rulequest.com/see5-info.html

---

## Major Variants & Evolution

### Conditional Inference Trees (ctree) — Hothorn, Hornik, Zeileis 2006

**Paper:** Hothorn T., Hornik K., Zeileis A. (2006). Unbiased Recursive Partitioning: A Conditional Inference Framework. *Journal of Computational and Graphical Statistics*, 15(3), 651–674.

**Problem solved:** Information gain and Gini impurity are biased toward features with many possible split points (high-cardinality continuous features or features with many categories). CART and C4.5 will preferentially split on such features even when a lower-cardinality feature is equally or more informative.

**Key change:** Separate variable selection from split threshold search using a permutation test. For each feature, test $H_0$: independence between feature and response (using a linear test statistic). Select the feature with the smallest p-value (after Bonferroni correction). Then find the best threshold only for that selected feature.

**Result:** Unbiased feature selection and valid statistical stopping rule (no pruning needed if test is not significant).

**Python:** Not in sklearn. Available in R's `partykit` package. Python wrappers exist but are not first-class.

---

### Extremely Randomized Trees (Extra-Trees) — Geurts, Ernst, Wehenkel 2006

**Paper:** Geurts P., Ernst D., Wehenkel L. (2006). Extremely Randomized Trees. *Machine Learning*, 63(1), 3–42. DOI: 10.1007/s10994-006-6226-1

**Change over CART:** For each split, instead of searching all possible thresholds on the selected feature, draw a random threshold. Both the feature and the threshold are randomized.

**Python:**
```python
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor
# As ensemble:
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
```

---

### Oblique Decision Trees — Non-Axis-Aligned Splits

Standard CART splits are axis-aligned: $x_j \leq \theta$. Oblique trees use linear combinations:
$$w_1 x_1 + w_2 x_2 + \cdots + w_p x_p \leq \theta$$

This can create diagonal decision boundaries that are impossible with axis-aligned splits (e.g., the XOR problem needs only one oblique split but exponentially many axis-aligned splits).

**Key algorithms:**
- **OC1** (Murthy, Kasif, Salzberg 1994) — local search over split coefficients
- **HHCART** (Wickramarachchi et al. 2015, arXiv:1504.03415) — Householder reflection to find oblique splits via eigendecomposition of class covariance matrices. Fits axis-aligned splits in the reflected space.
- **CART-ELC** (2025, arXiv:2505.05402) — exhaustive oblique split search

**Python:** `scikit_obliquetree` package; not in sklearn core.

**When to use:** When decision boundary is oblique/diagonal; especially helpful with correlated features.

---

### Optimal Decision Trees — OSDT, GOSDT

**Problem:** Greedy (top-down) CART finds locally optimal splits at each node but the resulting tree is globally suboptimal. Optimal methods search the entire space of decision trees.

**OSDT Paper:** Hu X., Rudin C., Seltzer M. (2019). Optimal Sparse Decision Trees. *NeurIPS 2019*.
URL: https://arxiv.org/abs/1904.12847

**GOSDT Paper:** Lin J., Zhong C., Hu D., Rudin C., Seltzer M. (2020). Generalized and Scalable Optimal Sparse Decision Trees. *ICML 2020*.
URL: https://arxiv.org/pdf/2006.08690

**Key idea:** Use branch-and-bound search with tight lower bounds on the training objective + a sparsity penalty, combined with clever data structures (bit arrays for fast subset operations).

**Python:** `gosdt-guesses` package (pip install gosdt-guesses). For small datasets and interpretability requirements. Impractical for $n > 10^5$.

**When to use:** When you need provably optimal (not just heuristically good) trees, especially for high-stakes decisions where maximum interpretability and fairness guarantees matter.

---

## Hyperparameter Reference

### DecisionTreeClassifier Hyperparameters

| Hyperparameter | Default | Type | What it controls | Tuning range |
|---|---|---|---|---|
| `criterion` | `'gini'` | str | Split quality metric | `'gini'`, `'entropy'`, `'log_loss'` |
| `splitter` | `'best'` | str | Greedy vs. random split search | `'best'`, `'random'` |
| `max_depth` | `None` | int/None | Maximum tree depth | 1–30; try 3–10 first |
| `min_samples_split` | `2` | int/float | Min samples to consider splitting | 2–20 (int) or 0.01–0.1 (float) |
| `min_samples_leaf` | `1` | int/float | Min samples at each leaf | 1–20 (int) or 0.01–0.05 (float) |
| `min_weight_fraction_leaf` | `0.0` | float | Like min_samples_leaf but weighted | 0.0–0.5 |
| `max_features` | `None` | int/float/str/None | Features considered per split | `None`, `'sqrt'`, `'log2'`, 0.3–0.8 |
| `random_state` | `None` | int/None | Reproducibility seed | any int |
| `max_leaf_nodes` | `None` | int/None | Max number of leaves (best-first) | 10–1000 |
| `min_impurity_decrease` | `0.0` | float | Min impurity gain to split | 0.0–0.05 |
| `class_weight` | `None` | dict/'balanced'/None | Sample weighting by class | `None`, `'balanced'`, `{0: 1, 1: 5}` |
| `ccp_alpha` | `0.0` | float | Cost-complexity pruning strength | 0.0–0.05 (use CCP path to find) |
| `monotonic_cst` | `None` | array-like/None | Monotonicity constraints per feature (added v1.4) | array of -1, 0, 1 |

### DecisionTreeRegressor Hyperparameters

Same as classifier except:
- `criterion`: `'squared_error'` (default), `'absolute_error'`, `'poisson'`
- No `class_weight`
- `monotonic_cst`: supported (not for multioutput)
- **Note:** `'friedman_mse'` accepted but deprecated in v1.9; use `'squared_error'` instead

### Detailed Hyperparameter Notes

#### `criterion`
**Classifier:**
- `'gini'` (default): $H(Q_m) = \sum_k p_{mk}(1-p_{mk})$. Faster to compute (no log). Tends to isolate the most frequent class. Nearly identical results to entropy in practice (accuracy differences < 1–2%).
- `'entropy'` / `'log_loss'`: $H(Q_m) = -\sum_k p_{mk} \log p_{mk}$. Slightly more sensitive to class distribution. Can produce marginally deeper trees. `'entropy'` and `'log_loss'` are aliases (identical).
- **Are gini and entropy equivalent for binary classification?** Mathematically *not* identical — they produce slightly different split evaluations — but empirically the resulting trees are almost always similar. The functional forms are monotonically related near $p=0.5$, which is why they agree most of the time.

**Regressor:**
- `'squared_error'` (default): minimizes mean squared error in leaves. Prediction = mean($y$).
- `'absolute_error'`: minimizes mean absolute error. Prediction = median($y$). Robust to outliers but 3–6× slower to fit.
- `'poisson'`: mean Poisson deviance. Use when $y \geq 0$ and represents counts/rates. Prediction = mean($y$) but split evaluation uses Poisson deviance. Useful for insurance claims, event counts.

#### `max_depth`
**Mechanism:** Hard cap on tree depth. Depth 0 = root only. Default `None` = grow until all leaves are pure or contain < `min_samples_split` samples.

**Too high:** Overfitting (memorizes training data). A tree with depth = log₂(n_samples) can potentially give each sample its own leaf.
**Too low:** Underfitting; simple decision boundaries.

**Tuning:** Start with 3–5 for interpretability. For performance, try 5–20 via CV. Often the primary hyperparameter to tune.

#### `min_samples_split`
**Mechanism:** An internal node is only considered for splitting if it has at least this many samples. Prevents splits on very small groups.
**Default:** `2` (almost any node can split). Equivalent to no constraint.
**As float:** Interpreted as `ceil(min_samples_split × n_samples)`.
**Tuning range:** 2–50, or 0.01–0.1 as fraction.

#### `min_samples_leaf`
**Mechanism:** A split is only made if both resulting children have at least this many samples. More directly controls leaf size than `min_samples_split`.
**Effect:** Acts as pre-pruning. Also smooths predictions in regression.
**Tuning:** 1–50; for regression, higher values reduce variance. Often prefer this over `min_samples_split`.

#### `min_weight_fraction_leaf`
**Mechanism:** Like `min_samples_leaf` but as a fraction of the sum of sample weights. Only relevant when `sample_weight` is used in `fit()`. Default `0.0` applies no constraint.

#### `max_features`
**Mechanism:** Number of features randomly sampled before each split decision. Random Forest uses `'sqrt'` by default here, while a single CART tree defaults to `None` (all features).
- `None` (default for single tree): consider all features at each split
- `'sqrt'`: $\lfloor\sqrt{p}\rfloor$ features — standard for RF classifiers
- `'log2'`: $\lfloor\log_2 p\rfloor$ — standard for RF regressors
- float (e.g., `0.5`): fraction of features
**Single tree effect:** Setting `max_features < p` introduces randomness; makes the tree faster and less prone to overfitting a dominant feature, but generally worsens single-tree performance unless used in an ensemble.

#### `max_leaf_nodes`
**Mechanism:** Grows the tree in best-first order (always splitting the leaf with highest impurity reduction), stopping when the tree has `max_leaf_nodes` leaves. Alternative to `max_depth`.
**Advantage over max_depth:** Best-first growth is more efficient — it spends splits where they matter most. Often produces better trees at the same node count.

#### `min_impurity_decrease`
**Mechanism:** A split is only made if the weighted impurity decrease is at least this value:
$$\frac{n_m}{n} \cdot [H(Q_m) - \frac{n_l}{n_m} H(Q_l) - \frac{n_r}{n_m} H(Q_r)] \geq \text{min\_impurity\_decrease}$$
**Default:** `0.0` (no minimum). Useful for pre-pruning when you want to avoid tiny gains.

#### `class_weight`
**Mechanism:** Scales the contribution of each sample by its class weight. `'balanced'` sets $w_k = n / (k \cdot n_k)$ where $n$ = total samples, $k$ = number of classes, $n_k$ = samples in class $k$.
**Use when:** Class imbalance. `'balanced'` is the go-to default for imbalanced binary classification.
**Note:** Affects the impurity criterion calculation, not just sample counts.

#### `ccp_alpha` (Cost-Complexity Pruning)
**Mechanism:** After growing the full tree, prunes back the subtree(s) with weakest links (lowest $\alpha_{\text{eff}}$) until all remaining subtrees have $\alpha_{\text{eff}} \geq$ `ccp_alpha`.
**Default:** `0.0` (no pruning).
**Workflow:** Use `cost_complexity_pruning_path()` to get all candidate alphas, then cross-validate. See CCP workflow in The Original Algorithm section above.
**Key property:** This is post-pruning (not pre-pruning like `max_depth`). It is the principled, statistically-motivated pruning method from the CART book.

#### `monotonic_cst` (Added v1.4)
**Mechanism:** Constrains the prediction function to be monotonically increasing (+1) or decreasing (-1) with respect to specified features.
**Format:** Array of shape `(n_features,)` with values in $\{-1, 0, 1\}$.
**Limitations:**
- Not supported for multiclass classification (`n_classes > 2`)
- Not supported for multioutput regression
- For classifiers, the constraint applies to the probability of the positive class
**Example:**
```python
# Feature 0: monotone increasing, feature 1: no constraint, feature 2: monotone decreasing
clf = DecisionTreeClassifier(monotonic_cst=[1, 0, -1])
```
**When to use:** When domain knowledge dictates direction (e.g., "higher credit score → higher probability of repayment").

#### `splitter='random'`
**Mechanism:** Instead of exhaustively searching all features and thresholds, randomly selects a threshold for each randomly selected feature. Much faster. Equivalent to `ExtraTreeClassifier` behavior.
**Use case:** Large datasets where training speed matters more than per-tree quality. The randomness is compensated when used in an ensemble.

---

## Missing Value Support (sklearn)

**Version introduced:** sklearn **1.3** (not 1.5 as stated in the topic brief — verify in release notes).

**Supported configurations:**
- `DecisionTreeClassifier` / `DecisionTreeRegressor` with `splitter='best'` and criterion in `{gini, entropy, log_loss}` for classification OR `{squared_error, friedman_mse, poisson}` for regression
- `ExtraTreeClassifier` / `ExtraTreeRegressor` with `splitter='random'`

**How splits handle missing values:** For each threshold candidate on non-missing data, evaluate the split twice: once routing all NaN samples left, once routing all NaN samples right. Take the better option.

**Prediction with missing values:** If no NaN was seen during training for a feature, missing values are routed to the child with the most samples. If the evaluation is tied, missing values go right.

**Example:**
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

X = np.array([0, 1, 6, np.nan]).reshape(-1, 1)
y = [0, 0, 1, 1]
tree = DecisionTreeClassifier(random_state=0).fit(X, y)
print(tree.predict([[np.nan]]))  # [1] — routed correctly
```

**Constraint:** Missing value support requires `splitter='best'`. When `max_features` is set (triggering random feature subsampling), missing values may not be supported in all criterion combinations — **verify in current docs**.

---

## Common Errors & Pitfalls

### 1. Overfitting with Default Parameters
**Symptom:** 100% training accuracy, poor test accuracy. Training a `DecisionTreeClassifier()` with no constraints on any real dataset will almost certainly overfit.
**Mechanism:** The default `max_depth=None`, `min_samples_leaf=1`, `min_samples_split=2` combination allows the tree to grow until every leaf has a single sample (pure leaves).
**Fix:** Set `max_depth`, `min_samples_leaf`, or `ccp_alpha`. Always evaluate on a held-out set.

### 2. Instability (High Variance)
**Mechanism:** Decision trees are highly sensitive to the training data. A small change in the training set (removing even 1% of samples) can result in a completely different tree structure due to the greedy splitting procedure — different top-level splits cascade into completely different subtrees.
**Detection:** Retrain on bootstrap samples and compare tree structures — they'll look drastically different.
**Fix:** Use Random Forests or gradient boosting (ensembles average out the instability). For a single tree, use `random_state` to get reproducibility, but the instability is inherent.

### 3. Impurity Bias Toward High-Cardinality Features (MDI)
**Mechanism:** Features with many possible split points (continuous features, or categoricals with many values) inherently offer more candidate splits and thus get higher MDI (feature importance from `feature_importances_`).
**Example:** A random numeric column often has higher MDI than a genuinely predictive binary column.
**Fix:** Use permutation importance (`from sklearn.inspection import permutation_importance`) computed on a held-out test set. This is unbiased.

### 4. Forgetting that sklearn requires numeric input
**Mechanism:** sklearn's `DecisionTreeClassifier` does not natively support categorical features. If you pass string categories, you'll get a `ValueError` (or silently encode them as numbers if you use `.cat.codes`).
**Fix:** One-hot encode categoricals via `OneHotEncoder` in a pipeline. Note: one-hot encoding with a tree requires more splits to recover the same logical grouping a native categorical split would achieve.

### 5. Using Training Set for ccp_alpha Selection
**Pitfall:** Running `cost_complexity_pruning_path` and selecting the alpha that maximizes training score (always selects `alpha=0`, the full tree).
**Fix:** Cross-validate alpha selection using `cross_val_score` for each alpha value, or use a dedicated validation split (not the final test set).

### 6. Ignoring class_weight with Imbalanced Data
**Mechanism:** With 95% class 0 and 5% class 1, the tree will very quickly predict all class 0 (low impurity). The minority class is systematically ignored.
**Fix:** Set `class_weight='balanced'` or manually provide weights. Also consider `SMOTE`/oversampling if the imbalance is extreme.

### 7. Misinterpreting feature_importances_ from a Single Tree
**Mechanism:** A single tree's `feature_importances_` reflects only the splits in that particular tree. Due to tree instability, a different random seed or slightly different data will give completely different importances.
**Fix:** Use `RandomForestClassifier` for more stable feature importances. Or use permutation importance.

### 8. Missing value support constraints
**Pitfall:** Setting `splitter='random'` with `DecisionTreeClassifier` and expecting missing value support — this combination is not supported (missing values require `splitter='best'` for the standard classifiers).

### 9. Forgetting that `friedman_mse` is deprecated in sklearn 1.9+
If you are supporting code written before sklearn 1.9, note that `criterion='friedman_mse'` will trigger a `FutureWarning` in 1.9+ and may be removed in 1.11. Replace with `criterion='squared_error'` — they are mathematically identical.

### 10. Decision Tree ≠ Decision Rules (directly)
**Misconception:** Practitioners sometimes assume any tree path is a usable rule. A tree with depth 15 has up to $2^{15} = 32768$ leaves, each with a unique rule. These are not interpretable. Only shallow trees (depth ≤ 4) produce human-interpretable rules.
**Fix:** Use `export_text()` or `export_graphviz()` to inspect rules. Prune aggressively if interpretability is needed.

---

## Impurity Measures: Mathematical Equivalence Analysis

For binary classification with $p$ = probability of positive class:

| Criterion | Formula | Maximum (most impure) | Minimum (pure) |
|---|---|---|---|
| Gini | $2p(1-p)$ | 0.5 at $p=0.5$ | 0 at $p=0$ or $p=1$ |
| Entropy | $-p\log_2 p - (1-p)\log_2(1-p)$ | 1.0 at $p=0.5$ | 0 at $p=0$ or $p=1$ |
| Log-loss | Same as entropy (alias) | 1.0 at $p=0.5$ | 0 |

**Are Gini and entropy equivalent for binary classification?** They are NOT mathematically identical. Gini is a second-order Taylor approximation of entropy (specifically, using $-\ln p \approx 1-p$, so $-p \ln p \approx p(1-p)$). They are monotonically related near $p=0.5$ but diverge near extremes. In practice, the resulting trees are almost always the same — empirical accuracy differences are typically < 1–2%.

**When they might differ:** With highly skewed class distributions (e.g., 1% vs 99%), Gini is more aggressive at isolating the majority class (lower impurity achievable faster), while entropy maintains higher sensitivity to the minority class.

**Computational:** Gini has no logarithm — slightly faster. On modern hardware, this difference is negligible for typical datasets.

---

## Key Datasets for Examples

### Iris (Classification — 3 class, 4 features)
```python
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names
target_names = load_iris().target_names
# n=150, p=4, k=3, fully balanced
```
**Best for:** Visualization (depth-2 tree achieves ~97% accuracy), showing plot_tree.

### Breast Cancer Wisconsin (Binary Classification)
```python
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
# n=569, p=30, binary (malignant/benign)
```
**Best for:** CCP pruning demo (sklearn's official example uses this), SHAP TreeExplainer.

### California Housing (Regression)
```python
from sklearn.datasets import fetch_california_housing
X, y = fetch_california_housing(return_X_y=True)
# n=20640, p=8 (longitude, latitude, etc.), target=median house value
```
**Best for:** DecisionTreeRegressor, showing regression impurity (squared_error, absolute_error, poisson), feature importances.

### Titanic (Imbalanced Classification)
```python
import seaborn as sns
df = sns.load_dataset('titanic')
# Good for: class_weight='balanced', missing value handling
```

---

## Curated Resources

### Original Papers
1. **Quinlan 1986 (ID3):** https://link.springer.com/article/10.1007/BF00116251
   *The paper that started it all. Full description of ID3 and information gain.*

2. **Quinlan 1993 (C4.5 book):** Available on Amazon/Semantic Scholar. Key extension adding continuous features, pruning, missing values.
   Semantic Scholar: https://www.semanticscholar.org/paper/C4.5%3A-Programs-for-Machine-Learning-Quinlan/807c1f19047f96083e13614f7ce20f2ac98c239e

3. **CART 1984 book:** https://www.amazon.com/Classification-Regression-Wadsworth-Statistics-Probability/dp/0412048418
   *The foundational text. Still the definitive reference for binary splits, Gini, and CCP.*

4. **Hothorn, Hornik, Zeileis 2006 (ctree):** https://www.tandfonline.com/doi/abs/10.1198/106186006X133933
   *Unbiased recursive partitioning via conditional inference.*

5. **GOSDT 2020:** https://arxiv.org/pdf/2006.08690
   *Globally optimal sparse decision trees — the interpretable ML frontier.*

6. **HHCART oblique trees 2015:** https://arxiv.org/abs/1504.03415
   *Householder-based oblique splits.*

### Official sklearn Documentation
7. **sklearn Decision Trees User Guide:** https://scikit-learn.org/stable/modules/tree.html
   *Complete reference for all sklearn tree parameters, algorithms, and examples.*

8. **DecisionTreeClassifier API:** https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

9. **CCP Pruning Example:** https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
   *Full working code for the CCP alpha selection workflow.*

10. **Permutation Importance vs MDI:** https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
    *Shows why MDI is biased and when to use permutation importance instead.*

### SHAP
11. **SHAP TreeExplainer docs:** https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
    *Constructor signature, supported models, and output format.*

12. **Tree SHAP for simple models notebook:** https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Understanding%20Tree%20SHAP%20for%20Simple%20Models.html
    *Working code: sklearn DecisionTreeClassifier + SHAP waterfall/beeswarm plots.*

### Visualization
13. **dtreeviz GitHub:** https://github.com/parrt/dtreeviz
    *The beautiful tree visualization library. Install: `pip install dtreeviz`.*

14. **mljar viz comparison (5 ways to visualize):** https://mljar.com/blog/visualize-decision-tree/
    *Good tutorial covering plot_tree, graphviz, dtreeviz side by side.*

### Tutorials / Deep Dives
15. **sklearn monotonic constraints demo:** https://scikit-learn.org/stable/auto_examples/ensemble/plot_monotonic_constraints.html
    *Shows monotonic_cst parameter for both trees and ensembles.*

16. **CART-ELC oblique trees 2025 (arXiv):** https://arxiv.org/html/2505.05402v1
    *Exhaustive oblique split search — current research frontier.*

---

## Uncertainties

1. **`friedman_mse` removal timeline:** Deprecated in sklearn 1.9 with warning that it "will be removed in 1.11." The course bible targets 1.5.x, but given the current stable is 1.9.x, the chapter should note both: (a) `'friedman_mse'` exists in sklearn 1.5.x but is deprecated in 1.9.x; (b) the chapter author should decide which version to target. **The chapter should use `'squared_error'` as the canonical name.**

2. **Missing value support version:** Research confirms it was **sklearn 1.3** (not 1.5 as the topic brief suggested). The author should verify in the 1.3.0 release notes (https://scikit-learn.org/stable/whats_new/v1.3.html) and double-check which criterion/splitter combinations are supported in each version.

3. **`monotonic_cst` for multiclass classifiers:** Documented as "not supported for multiclass or multioutput." The author should verify whether this restriction was relaxed in versions > 1.4. Check the changelog for 1.5–1.9.

4. **dtreeviz version compatibility:** dtreeviz 2.x uses an adaptor pattern that requires `dtreeviz.model()` (not the old `dtreeviz()` function from 1.x). The author should verify the current import and usage against the actual installed version and update any examples accordingly.

5. **SHAP 0.46.x compatibility:** The `explainer(X)` call syntax (returning an `Explanation` object for waterfall/beeswarm) requires shap >= 0.40.x. The older `explainer.shap_values(X)` returns raw arrays. Both work but have slightly different plotting APIs. The author should confirm which shap version the course standardizes on and use that API consistently.

6. **`splitter='random'` + missing values:** The documentation states missing value support exists for `splitter='best'`. Whether `ExtraTreeClassifier` (which uses `splitter='random'`) truly supports missing values without restriction needs verification in the actual 1.5+ docs — some combination constraints may apply.

7. **Optimal decision trees scalability:** GOSDT is practical only for small $n$ (< 100k) and sparse binary feature matrices. For real use with larger datasets, the author should note OSDT's limitations clearly.

8. **C4.5 DOI:** The book itself does not have a DOI. The author should use the Morgan Kaufmann ISBN (978-1-558-60238-0) and note it is a book, not a journal article.

---

## Key Algorithmic Details the Writer Must Nail

### The CART Training Loop (must be stated correctly)

```
BuildTree(node, S):
  if stopping_criteria_met(S):
      node.is_leaf = True
      node.value = majority_class(S)  # or mean(y) for regression
      return
  (j*, θ*) = argmin_{j, θ} G(S, j, θ)   # exhaustive search over all (feature, threshold) pairs
  node.feature = j*
  node.threshold = θ*
  S_left  = {(x, y) ∈ S : x_{j*} ≤ θ*}
  S_right = {(x, y) ∈ S : x_{j*} > θ*}
  BuildTree(node.left,  S_left)
  BuildTree(node.right, S_right)
```

Stopping criteria: depth limit hit, min_samples_split, min_samples_leaf, impurity < min_impurity_decrease, all samples same class.

### Complexity
- **Training:** $O(n_{\text{features}} \times n_{\text{samples}} \times n_{\text{samples}} \times \log n_{\text{samples}})$ naive; sklearn uses efficient sorting to achieve approximately $O(n_{\text{features}} \times n_{\text{samples}} \times \log n_{\text{samples}})$ per level via cached sort orders.
- **Inference:** $O(d)$ where $d$ = tree depth. Typically $O(\log n_{\text{samples}})$ for balanced trees.
- **Space:** $O(n_{\text{samples}})$ nodes worst case (every sample gets its own leaf).

### Feature Importance (MDI) Formula
$$\text{importance}(j) = \sum_{t : \text{splits on } j} \frac{n_t}{n} \cdot [H(Q_t) - \frac{n_l}{n_t} H(Q_l) - \frac{n_r}{n_t} H(Q_r)]$$

Normalized to sum to 1. Biased toward high-cardinality features; use permutation importance for unbiased estimates.

### The Connection to Ensembles
- **Random Forest:** Bagging (bootstrap sampling) of CART trees + random feature subsampling (`max_features='sqrt'`). Each tree is a full CART tree with high variance; averaging reduces variance.
- **GBM:** Trees as weak learners fit to pseudo-residuals. Each tree in XGBoost/LightGBM is a CART variant (with extra regularization). The `max_depth` of each tree in GBM should be much smaller (3–8) than in a standalone tree.

---

*End of research dossier. Word count: approximately 2,400 words (technical density is high — the actual content when expanded into prose for the chapter will reach 20,000+ tokens).*
