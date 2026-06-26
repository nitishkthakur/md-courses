# Decision Trees: The Complete Masterclass

> **Why this algorithm matters:** Decision trees are the foundation of modern machine learning.
> Every gradient boosting model you have ever trained — XGBoost, LightGBM, CatBoost — is a forest
> of decision trees. Every random forest is an ensemble of decision trees. CART, the algorithm
> at the heart of sklearn's `DecisionTreeClassifier`, was published in 1984 and is still, forty
> years later, one of the most widely deployed predictive models in medicine, finance, and industry.
> A 2023 survey of credit-risk models at US financial institutions found that tree-based models
> (single trees and ensembles) were the most prevalent model class, used in over 60% of deployed
> systems. Understanding decision trees deeply — not just tuning `max_depth`, but understanding
> every split criterion, every pruning strategy, every failure mode — is a prerequisite for
> understanding the entire modern ML landscape.

---

## §0 — Python Ecosystem & Package Guide

The decision tree ecosystem is more unified than most algorithms: sklearn's `DecisionTreeClassifier`
and `DecisionTreeRegressor` are the canonical implementations, and virtually every ensemble library
builds on or wraps the same CART algorithm. But the visualization, interpretability, and variant
landscape is rich. Here is the full map.

### Complete Package Table

| Package | Import | Version (mid-2026) | What it provides | License |
|---|---|---|---|---|
| `scikit-learn` | `from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor` | ~1.9.x | CART with Gini/entropy/MSE, CCP pruning, missing value support (v1.3+), monotonic constraints (v1.4+) | BSD-3 |
| `scikit-learn` | `from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor` | ~1.9.x | Single extra-randomized tree (random threshold + random feature per split) | BSD-3 |
| `dtreeviz` | `import dtreeviz` | ~2.x | Publication-quality tree visualizations; adaptor pattern supports sklearn, XGBoost, LightGBM, Spark | MIT |
| `shap` | `import shap` | ~0.46.x | TreeExplainer: exact polynomial-time SHAP values for tree models | MIT |
| `graphviz` | `import graphviz` | ~0.20.x | Renders DOT files from `export_graphviz`; wraps the Graphviz system binary | MIT |
| `optuna` | `import optuna` | ~3.x | Hyperparameter optimization (max_depth, ccp_alpha, etc.) | MIT |
| `gosdt-guesses` | `from gosdt import GOSDTClassifier` | ~0.1.x | Globally optimal sparse decision trees (provably optimal, small data) | MIT |
| `scikit-obliquetree` | `from oblique_tree import ...` | community | Oblique (non-axis-aligned) splits via OC1-style local search | various |
| `river` | `from river.tree import HoeffdingTreeClassifier` | ~0.21.x | Online/streaming decision trees (Hoeffding trees) for data streams | BSD-3 |
| `h2o` | `import h2o` | ~3.46.x | Distributed tree learning at scale; REST API with Python client | Apache-2 |

### Top Picks — Recommendation Table

| Package | Best for | Approach | Notes |
|---|---|---|---|
| **`sklearn.DecisionTreeClassifier`** | Standard classification, interpretability, ensembles | CART (binary splits, Gini/entropy, CCP) | **First choice for 95% of use cases**; fast, Pipeline-ready, excellent docs |
| **`sklearn.DecisionTreeRegressor`** | Regression, count data (Poisson), robust fitting | CART (MSE/MAE/Poisson, binary splits) | Same defaults; use `criterion='absolute_error'` for outlier-robust regression |
| **`dtreeviz`** | Publication-quality visualization, decision path explanation | Adaptor over sklearn/XGBoost | Requires Graphviz system binary; `dtreeviz.model()` API in v2.x |
| **`gosdt-guesses`** | High-stakes interpretability, provably optimal small trees | Integer programming + branch-and-bound | Only practical for $n < 100k$ and sparse binary features |
| **`river.HoeffdingTreeClassifier`** | Streaming data, online learning, concept drift | Hoeffding/VFDT algorithm | When data arrives as a stream and you cannot store it all |

### Same Fit Across Top Packages

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
feature_names = list(iris.feature_names)
target_names = list(iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Option 1: sklearn DecisionTreeClassifier (canonical) ──────────────────
from sklearn.tree import DecisionTreeClassifier

dt_sklearn = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_leaf=5,
    random_state=42,
)
dt_sklearn.fit(X_train, y_train)
print(f"sklearn accuracy: {dt_sklearn.score(X_test, y_test):.4f}")
# sklearn accuracy: 0.9667

# ── Option 2: ExtraTreeClassifier (randomized single tree) ────────────────
from sklearn.tree import ExtraTreeClassifier

dt_extra = ExtraTreeClassifier(
    criterion='gini',
    max_depth=4,
    random_state=42,
)
dt_extra.fit(X_train, y_train)
print(f"ExtraTree accuracy: {dt_extra.score(X_test, y_test):.4f}")

# ── Option 3: GOSDT (optimal sparse tree, small data) ─────────────────────
# from gosdt import GOSDTClassifier  # pip install gosdt-guesses
# gosdt_clf = GOSDTClassifier(regularization=0.01, time_limit=60)
# gosdt_clf.fit(X_train_binary, y_train)  # requires binarized features
# print(gosdt_clf.get_tree())             # prints the optimal tree as text rules

# ── Option 4: Hoeffding Tree (streaming / online learning) ────────────────
# from river.tree import HoeffdingTreeClassifier
# ht = HoeffdingTreeClassifier()
# for xi, yi in zip(X_train, y_train):
#     ht.learn_one(dict(enumerate(xi)), yi)
# acc = sum(ht.predict_one(dict(enumerate(xi))) == yi for xi, yi in zip(X_test, y_test))
# print(f"Hoeffding accuracy: {acc / len(y_test):.4f}")
```

> 💡 **Intuition:** Unlike logistic regression, which needs feature scaling and produces
> probabilistic scores along a smooth curve, decision trees are scale-invariant and produce
> step-function predictions. The same fit/predict API hides fundamentally different internal
> mechanics — you do not need to standardize features before passing them to a tree.

> 🔧 **In practice:** Decision trees do not require feature scaling (`StandardScaler` does nothing
> useful). They handle mixed types gracefully — but sklearn requires you to encode categoricals to
> numeric first. For a model you plan to interpret, prefer depth ≤ 4 and use `export_text()` or
> `dtreeviz` to visualize the resulting rules.

---

## §1 — The Origin: Two Founding Papers

Decision trees have a rare distinction in machine learning: they have not one but two independent,
nearly simultaneous founding works — published within two years of each other by statisticians on
one side and computer scientists on the other. To understand the algorithm deeply you need to
understand both, because they solved the same problem from different starting points and made
different design choices. Those choices still echo in every implementation today.

### Before Trees: The Landscape of 1984

To appreciate what CART and ID3 achieved, imagine the predictive modeling landscape of the early
1980s. The dominant tools were **linear discriminant analysis** (Fisher, 1936), **logistic
regression** (Cox, 1958), and **nearest neighbors** (Cover & Hart, 1967). All of them shared a
common limitation: they were either parametric (requiring an assumed distribution), linear (unable
to capture interaction effects without manual feature engineering), or computationally expensive
for high-dimensional data.

Consider a medical diagnosis problem: predict whether a patient has heart disease based on
20 clinical variables, where the relationships are highly non-linear and involve complex
three-way interactions among age, cholesterol, and blood pressure. A logistic regression model
requires you to engineer all the interaction terms manually. Linear discriminant analysis
requires multivariate normality, which clinical data rarely satisfies. Nearest neighbors requires
defining a meaningful distance metric in 20 dimensions.

The statisticians Leo Breiman, Jerome Friedman, Richard Olshen, and Charles Stone, and separately
the computer scientist J. Ross Quinlan, arrived at the same idea: why not let the data *tell you*
the interaction structure by recursively partitioning the feature space into rectangles?

---

### CART (1984): The Statisticians' Tree

> 📜 **Origin/Citation:** Breiman, L., Friedman, J.H., Olshen, R.A., & Stone, C.J. (1984).
> *Classification and Regression Trees.* Wadsworth & Brooks/Cole, Pacific Grove, CA.
> ISBN: 978-0-412-04841-8.
> Semantic Scholar: https://www.semanticscholar.org/paper/Classification-and-Regression-Trees-(Wadsworth-Breiman-Friedman/2203c20aaefc87c72e494b45dc77ed10f3013cb5

The CART book is 358 pages of careful statistics. It is not a conference paper — it is a
monograph, the kind you write when you have genuinely solved a problem from first principles and
want to leave nothing unexplained. Four researchers from Stanford and UC Berkeley spent years
working out every corner case.

**The core innovations of CART:**

**1. Binary splits only.** Every internal node splits into exactly two children, not $k$ children.
This is a deliberate design choice. Multi-way splits are attractive (split "color" into red,
blue, green), but they fragment the data too aggressively: after one multi-way split, each branch
has far fewer samples, reducing statistical power for subsequent splits. Binary trees can
represent the same logical structure by chaining two-way splits, but do so while keeping more
data in each node at each level.

**2. Handles continuous features.** ID3 (discussed below) was initially categorical-only. CART
places continuous features on equal footing: for a continuous feature $x_j$, evaluate all $n-1$
possible split thresholds (midpoints between consecutive sorted values). This makes CART
applicable to the numerical data that dominates scientific measurement.

**3. Handles regression.** A classification tree predicts a class label; a regression tree predicts
a continuous value — the mean of the target in each leaf. CART was the first unified framework
handling both with the same algorithm and the same mathematical machinery.

**4. Gini impurity.** Breiman introduced a new splitting criterion. Define the Gini impurity at
node $m$ as:

$$H(Q_m) = \sum_{k=1}^{K} p_{mk}(1 - p_{mk}) = 1 - \sum_{k=1}^{K} p_{mk}^2$$

where $p_{mk}$ is the proportion of class $k$ samples at node $m$. This is the probability of
*misclassification* if you label a randomly drawn sample from node $m$ by a randomly drawn
label from the same distribution. Pure nodes ($p_{mk} = 1$ for some $k$) have $H = 0$. Maximum
impurity at $p_{mk} = 1/K$ for all $k$ gives $H = 1 - 1/K$.

Why Gini rather than entropy? Two reasons: (1) computational efficiency — no logarithm to
evaluate, which mattered enormously on 1984 hardware; (2) it emphasizes the most frequent class
more aggressively, which often produces shallower trees for the same classification performance.

**5. Cost-complexity pruning.** This is CART's most important contribution. A full unpruned tree
memorizes the training data (every leaf is pure). Breiman's solution was not to stop early
(pre-pruning) but to grow the full tree first, then prune it back using a mathematically
principled criterion.

Define the cost-complexity measure of a subtree $T$ as:

$$R_\alpha(T) = R(T) + \alpha |\tilde{T}|$$

where $R(T) = \sum_{t \in \tilde{T}} \frac{n_t}{n} H(Q_t)$ is the total weighted impurity of
all terminal nodes $\tilde{T}$, $|\tilde{T}|$ is the number of terminal nodes, and
$\alpha \geq 0$ is the complexity parameter that trades impurity against tree size.

At $\alpha = 0$, the full tree minimizes $R_\alpha$. As $\alpha$ increases, we penalize
complexity more and the optimal subtree shrinks. The key insight Breiman proved: as $\alpha$
increases from 0 to $\infty$, the sequence of optimal subtrees is *nested* — we get
$T_0 \supset T_1 \supset T_2 \supset \cdots \supset \{root\}$, a finite sequence. We can compute
this entire sequence in one pass using the *weakest link pruning* algorithm.

For each internal node $t$ in the full tree, define its effective alpha:

$$\alpha_{\text{eff}}(t) = \frac{R(t) - R(T_t)}{|\tilde{T}_t| - 1}$$

where $R(t)$ is the impurity of node $t$ if it were treated as a leaf, and $R(T_t)$ is the total
impurity of the subtree rooted at $t$. The numerator measures how much impurity increases if we
collapse the subtree; the denominator measures how many nodes we eliminate. The internal node
with the smallest $\alpha_{\text{eff}}$ is the "weakest link" — it provides the least benefit per
additional node. Prune it. Repeat. This gives the full sequence of subtrees. Then use
cross-validation to select the best $\alpha$.

> 🔬 **Deep dive:** Cost-complexity pruning is also called "minimal cost-complexity pruning"
> or "weakest link pruning." The full proof that the sequence of optimal trees is nested requires
> showing that $T(\alpha)$ (the unique smallest subtree minimizing $R_\alpha$) changes
> discontinuously at specific threshold values of $\alpha$ and that successive thresholds are
> strictly increasing. Breiman's proof of this is in Chapter 3 of the CART book and is
> surprisingly elegant — it proceeds by contradiction from the optimality conditions.

---

### ID3 (1986): The Computer Scientist's Tree

> 📜 **Origin/Citation:** Quinlan, J.R. (1986). Induction of Decision Trees.
> *Machine Learning*, 1(1), 81–106. DOI: 10.1007/BF00116251
> URL: https://link.springer.com/article/10.1007/BF00116251

While Breiman's team was approaching trees from statistical theory, Ross Quinlan at the University
of Sydney was thinking about the problem from a concept-learning perspective. His challenge: given
a set of examples described by categorical attributes, how do you automatically learn a decision
tree that generalizes the concept?

The key insight was **information gain** — a measure borrowed from information theory. Shannon
(1948) defined entropy as:

$$H(S) = -\sum_{k=1}^{K} p_k \log_2 p_k$$

This is the average number of bits needed to encode a randomly drawn class label from set $S$.
A pure set (all one class) has entropy 0. A maximally mixed set (uniform over $K$ classes) has
entropy $\log_2 K$.

Quinlan defined the information gain of splitting $S$ on attribute $A$ as:

$$IG(S, A) = H(S) - \sum_{v \in \text{values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

The weighted sum $\sum_v \frac{|S_v|}{|S|} H(S_v)$ is the conditional entropy of the class given
knowledge of attribute $A$. Information gain is how much knowing $A$ reduces uncertainty about
the class. Quinlan's ID3 algorithm greedily selects the attribute with the highest information gain
at each node.

**ID3 Algorithm (formal pseudocode):**

```
ID3(examples S, target_attribute, attributes):
  if all examples in S have the same class c:
      return Leaf(class=c)
  if attributes is empty:
      return Leaf(class=majority_class(S))
  A* = argmax_{A ∈ attributes} IG(S, A)
  create internal node branching on A*
  for each value v of A*:
      S_v = {s ∈ S : A*(s) = v}
      if S_v is empty:
          add child Leaf(class=majority_class(S))
      else:
          add child ID3(S_v, target_attribute, attributes \ {A*})
  return node
```

**ID3's limitations:** The algorithm was groundbreaking but narrow. It assumed all attributes were
categorical (binary or multi-valued). It could not handle continuous variables. It had no pruning —
ID3 grew trees until every leaf was pure, guaranteeing 100% training accuracy and catastrophic
overfitting. And it had a subtle bias: attributes with more possible values (high-cardinality
features) almost always scored higher information gain than binary attributes, even if the binary
attribute was genuinely more informative. A random ID number with $n$ unique values would score
maximum information gain and would always be selected first.

These limitations would drive Quinlan's next seven years of work, culminating in C4.5.

---

### C4.5 (1993): Fixing What ID3 Got Wrong

> 📜 **Origin/Citation:** Quinlan, J.R. (1993). *C4.5: Programs for Machine Learning.*
> Morgan Kaufmann Publishers, San Mateo, CA. ISBN: 978-1-558-60238-0.
> Semantic Scholar: https://www.semanticscholar.org/paper/C4.5%3A-Programs-for-Machine-Learning-Quinlan/807c1f19047f96083e13614f7ce20f2ac98c239e

C4.5 solved every significant limitation of ID3:

**Continuous attributes:** Sort the unique values of attribute $A$, then evaluate each midpoint
$\theta_{ij} = (v_i + v_{j})/2$ (where $j = i+1$ in the sorted order) as a threshold for a
binary split. Select the threshold maximizing information gain. This is now standard practice in
every tree implementation.

**Gain ratio:** To fix the bias toward high-cardinality attributes, Quinlan defined:

$$\text{GainRatio}(S, A) = \frac{IG(S, A)}{H_A(S)}$$

where $H_A(S) = -\sum_{v} \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}$ is the *split information*
— the entropy of the attribute values themselves. An attribute with $n$ values, each appearing
once, has $H_A = \log_2 n$, dividing its information gain by $\log_2 n$ and neutralizing the bias.

**Missing values:** C4.5 handles missing attribute values by distributing a sample with a missing
value of attribute $A$ proportionally across the branches of a split on $A$, weighted by the
fraction of non-missing samples going each way.

**Pessimistic error pruning:** C4.5 introduced a post-pruning step that uses a continuity
correction on training error to estimate generalization error, then replaces subtrees with leaves
if the replacement does not increase the estimated error. Unlike CART's cost-complexity pruning,
this requires no separate validation set — but it is less statistically rigorous.

> 💡 **Intuition:** Think of CART and C4.5 as solving the same problem from opposite ends.
> CART was designed by statisticians who wanted mathematical rigor: they proved theorems, defined
> cross-validation procedures, and left nothing to statistical intuition. C4.5 was designed by a
> computer scientist who wanted practical performance: Quinlan was iterating on a real system,
> fixing bugs and adding features version by version. The two lineages converged — sklearn's CART
> borrows the continuous feature handling from C4.5's approach and CART's rigorous CCP pruning —
> but the intellectual tension between "statistically principled" and "empirically effective"
> still shapes every hyperparameter choice you make.

---

## §2 — The Algorithm, Deeply Explained

### The Core Idea: Recursive Partitioning

A decision tree partitions the feature space $\mathcal{X} = \mathbb{R}^p$ into a set of
axis-aligned rectangular regions $R_1, R_2, \ldots, R_L$ (the leaves), and assigns a constant
prediction to each region. For classification, the prediction for region $R_\ell$ is the majority
class among training samples in that region. For regression, it is the mean.

$$\hat{f}(\mathbf{x}) = \sum_{\ell=1}^{L} c_\ell \cdot \mathbf{1}[\mathbf{x} \in R_\ell]$$

where $c_\ell$ is the leaf value (class or mean) and $\mathbf{1}[\cdot]$ is the indicator function.

This is a piecewise constant function. A tree with $L$ leaves partitions $\mathbb{R}^p$ into $L$
rectangles with constant predictions. The art of tree learning is deciding *which* partition to
use — this is an NP-hard combinatorial optimization problem (finding the globally optimal
partition), which is why CART uses a greedy approximation.

### The CART Training Algorithm

The full algorithm is recursive. At each node $m$, we have a subset $Q_m$ of training samples.
We need to decide: (a) should this node be a leaf? (b) if not, which feature $j$ and threshold
$\theta$ should we split on?

**Stopping criteria (node becomes a leaf if any is true):**
- $|Q_m| < \texttt{min\_samples\_split}$ (too few samples to justify splitting)
- All samples in $Q_m$ have the same class (pure node)
- Current depth equals `max_depth`
- Best achievable impurity decrease is less than `min_impurity_decrease`
- Either child would have fewer than `min_samples_leaf` samples

**Split search (if not stopping):** For each feature $j \in \{1, \ldots, p\}$ and each candidate
threshold $\theta$, define:

$$Q_m^{\text{left}}(j, \theta) = \{(\mathbf{x}, y) \in Q_m : x_j \leq \theta\}$$
$$Q_m^{\text{right}}(j, \theta) = \{(\mathbf{x}, y) \in Q_m : x_j > \theta\}$$

The **weighted impurity** of this split is:

$$G(Q_m, j, \theta) = \frac{n_m^{\text{left}}}{n_m} H(Q_m^{\text{left}}) + \frac{n_m^{\text{right}}}{n_m} H(Q_m^{\text{right}})$$

We choose the split $(j^*, \theta^*)$ that minimizes $G$:

$$(j^*, \theta^*) = \underset{j, \theta}{\arg\min}\; G(Q_m, j, \theta)$$

The impurity **decrease** (what sklearn reports as `min_impurity_decrease` threshold) is:

$$\Delta = \frac{n_m}{n}\left[H(Q_m) - \frac{n_m^{\text{left}}}{n_m} H(Q_m^{\text{left}}) - \frac{n_m^{\text{right}}}{n_m} H(Q_m^{\text{right}})\right]$$

The $n_m/n$ factor normalizes by the fraction of total samples in this node — so splits at the
root (large $n_m/n$) contribute more to importance than splits deep in the tree.

### The Three Impurity Criteria

**Classification: Gini impurity**

$$H(Q_m) = \sum_{k=1}^{K} p_{mk}(1 - p_{mk}) = 1 - \sum_{k=1}^{K} p_{mk}^2$$

Range: $[0, 1 - 1/K]$. Maximum at uniform distribution. Zero for pure nodes.

**Classification: Entropy** (criterion `'entropy'` or `'log_loss'` — they are aliases)

$$H(Q_m) = -\sum_{k=1}^{K} p_{mk} \log p_{mk}$$

Range: $[0, \log K]$ (nats if natural log, bits if log base 2 — sklearn uses natural log).
Maximum at uniform distribution. Zero for pure nodes.

**Are Gini and entropy equivalent?** They are not mathematically identical. Gini is related to
the second-order Taylor approximation of entropy: using $-\ln p \approx 1 - p$, we get
$-p \ln p \approx p(1-p)$, so entropy $\approx$ Gini near $p = 0$. They are monotonically
related near $p = 0.5$ but diverge at the extremes. In practice, the resulting trees are almost
always equivalent — empirical accuracy differences are typically < 1–2%. Gini has no logarithm
and is marginally faster to compute.

**Regression: Mean Squared Error**

$$H(Q_m) = \frac{1}{n_m}\sum_{y \in Q_m}(y - \bar{y}_m)^2, \quad \bar{y}_m = \frac{1}{n_m}\sum_{y \in Q_m} y$$

Prediction at leaf: $\hat{y} = \bar{y}_m$ (mean). Minimizing MSE is equivalent to choosing splits
that maximize the reduction in total within-node variance.

**Regression: Mean Absolute Error** (criterion `'absolute_error'`)

$$H(Q_m) = \frac{1}{n_m}\sum_{y \in Q_m}|y - \tilde{y}_m|, \quad \tilde{y}_m = \text{median}(Q_m)$$

Prediction at leaf: $\hat{y} = \tilde{y}_m$ (median). Robust to outliers, but 3–6× slower to
fit than MSE (sorting required at each split candidate to update the running median).

**Regression: Poisson deviance** (criterion `'poisson'`)

$$H(Q_m) = \frac{1}{n_m}\sum_{y \in Q_m}\left(y \log \frac{y}{\bar{y}_m} - (y - \bar{y}_m)\right)$$

Use when $y \geq 0$ and represents counts or rates (insurance claims, event frequencies). Penalizes
multiplicative deviations, which matches the natural scale of count data.

### Multiple Intuitions for What a Tree Is Doing

**Intuition 1: Recursive binary search.** A tree is running a decision procedure you might draw
on a whiteboard. "Is the patient's age > 65? If yes, is their blood pressure > 140? If yes,
prescribe medication." Each internal node eliminates half the remaining possibility space.

**Intuition 2: Piecewise constant function approximation.** A tree with $L$ leaves is a step
function over $\mathbb{R}^p$. More leaves = more steps = finer approximation. The greedy
algorithm places steps where they reduce prediction error the most.

**Intuition 3: Supervised clustering.** Each leaf is a cluster of similar samples (by the
feature-space split). The tree learns which splits create internally homogeneous clusters. The
impurity criterion measures cluster homogeneity — zero impurity = pure cluster.

**Intuition 4: Universal function approximator (with enough leaves).** Any continuous function
on a compact domain can be approximated to arbitrary precision by a piecewise constant function
(if you allow enough pieces). A sufficiently deep tree can memorize any training set. This is
both its power and its danger.

> ⚠️ **Pitfall:** A `DecisionTreeClassifier()` with default settings (`max_depth=None`,
> `min_samples_leaf=1`) will grow until every leaf is pure — achieving 100% training accuracy on
> virtually any dataset. This is catastrophic overfitting. The default settings are maximally
> flexible, not maximally useful. Always set at least one constraint.

### Threshold Selection for Continuous Features

For a continuous feature $x_j$ with $n_m$ samples in node $m$, there are at most $n_m - 1$
distinct candidate thresholds (the midpoints between consecutive sorted values). CART evaluates
all of them. This is $O(n_m \log n_m)$ per feature per node (one sort, then $O(n_m)$ scan).

Across all $p$ features and all $n$ samples in the root, the total training cost is
$O(p \cdot n \log n)$ per tree level. For a balanced tree of depth $d$, total training is
$O(d \cdot p \cdot n \log n)$. For a full tree where $d = O(\log n)$, this gives
$O(p \cdot n \log^2 n)$.

**Inference:** Following one sample from root to leaf requires at most $d$ comparisons, where $d$
is the tree depth. For a balanced tree, $d = O(\log n)$, making inference $O(\log n)$. In
practice, inference is extremely fast — typically microseconds per sample.

### What the Tree Assumes (and Doesn't)

Decision trees are among the most assumption-free algorithms in supervised learning:

1. **No distributional assumptions.** No normality, no linearity, no independence between features.
2. **No feature scaling required.** Monotone transformations of features do not change the split
   (the optimal threshold shifts, but the resulting partition is identical).
3. **Ordinal features are handled naturally.** Any feature with a total order supports the
   $x_j \leq \theta$ split form.
4. **Categorical features:** Need encoding (sklearn requires numeric input). One-hot encoding works
   but inflates $p$ and may require many splits to recover a natural grouping.
5. **Implicit assumption of axis-aligned boundaries.** If the true decision boundary is oblique
   (e.g., $x_1 + x_2 > 1$), CART needs exponentially more nodes than an oblique split would.
6. **Implicit assumption of piecewise-constant signal.** If the true function has smooth
   continuous gradients (like a sinusoidal signal), a tree will approximate it with staircase
   artifacts.

### The Connection Between Gini Impurity and Bayes Error

There is a deep relationship between Gini impurity and the Bayes error rate that is rarely
explained in textbooks. Sit with this for a moment.

The **Bayes error** at node $m$ is the probability of misclassifying a randomly drawn sample from
$Q_m$ using the optimal (majority-class) rule:

$$\epsilon_m^* = 1 - \max_k p_{mk}$$

The Gini impurity is:

$$H_{\text{Gini}}(Q_m) = 1 - \sum_k p_{mk}^2$$

These are related because $\sum_k p_{mk}^2 \geq \max_k p_{mk}$ always holds (by the Cauchy-Schwarz
inequality applied to the vector $\mathbf{p}$ against itself). Specifically:

$$H_{\text{Gini}}(Q_m) \leq 2\epsilon_m^*(1 - \epsilon_m^*)$$

This means **Gini impurity is an upper bound on twice the Bayes error rate** (at the optimal
majority-class rule). A split that maximally reduces Gini impurity is therefore maximally reducing
the upper bound on classification error in the resulting nodes.

Entropy has an analogous relationship to the mutual information between features and class labels.
The information gain $IG(S, A)$ is precisely the mutual information between the class variable
$Y$ and the split indicator variable $A$, evaluated on the training distribution. Maximizing
information gain is equivalent to maximizing the mutual information between the selected feature
and the class — which is exactly what feature selection theory says you should do.

> 🔬 **Deep dive:** The relationship between impurity criteria and probabilistic loss functions
> goes deeper. Gini impurity is the *Brier score* applied to the constant class-proportion
> predictor at each node. Entropy is the *log loss*. When you choose `criterion='gini'`, you are
> implicitly optimizing a Brier-score proxy at each step. When you choose `criterion='entropy'`,
> you are optimizing a log-loss proxy. This is why entropy-based trees tend to produce slightly
> better calibrated probability estimates — log loss is the proper scoring rule for probabilities,
> while Brier score is a quadratic approximation.

### The Bias-Variance Decomposition for Decision Trees

Understanding *why* trees overfit requires thinking through the bias-variance decomposition.
For a regression tree with prediction $\hat{f}(\mathbf{x})$, the expected mean squared error
at a test point $\mathbf{x}_0$ decomposes as:

$$E[(\hat{f}(\mathbf{x}_0) - f(\mathbf{x}_0))^2] = \underbrace{[E\hat{f}(\mathbf{x}_0) - f(\mathbf{x}_0)]^2}_{\text{bias}^2} + \underbrace{E[(\hat{f}(\mathbf{x}_0) - E\hat{f}(\mathbf{x}_0))^2]}_{\text{variance}} + \sigma^2_\epsilon$$

For a **shallow tree (small `max_depth`):**
- **High bias:** The piecewise-constant approximation uses few pieces, so it cannot capture
  fine-grained variation in $f$. The tree prediction $E\hat{f}$ systematically deviates from $f$.
- **Low variance:** Each leaf contains many samples, so the leaf mean is a stable estimate of
  the true mean $f$ averaged over that region.

For a **deep tree (large `max_depth`):**
- **Low bias:** Many small leaves cover the feature space densely; the approximation can be
  arbitrarily fine.
- **High variance:** Each leaf contains very few samples (sometimes 1). The leaf mean is unstable
  — it equals the single training sample's noise-corrupted $y$ value. Across different training
  sets, the predicted value for $\mathbf{x}_0$ will swing wildly.

This explains both why unconstrained trees overfit (variance explodes) and why Random Forests
work: averaging $B$ independent trees reduces variance by a factor of $B$ while keeping the
same bias.

**Mathematically:** If the trees in a Random Forest are identically distributed with variance
$\sigma^2$ and pairwise correlation $\rho$, the variance of the average prediction is:

$$\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$$

As $B \to \infty$, this approaches $\rho \sigma^2$. Reducing $\rho$ (by using `max_features`
to randomize splits) reduces this floor. This is the formal justification for Random Forests.

### Threshold Selection: The Efficiency Argument

When does CART's threshold search become expensive? Let's think carefully about the real
complexity.

At a node containing $n_m$ samples, feature $j$ has $n_m$ values. After sorting once ($O(n_m \log n_m)$),
we can scan from left to right and maintain running class counts in left and right children
using $O(1)$ updates per threshold. The total cost for one feature at one node is
$O(n_m \log n_m)$ for the sort, then $O(n_m)$ for the scan — total $O(n_m \log n_m)$.

Across all $p$ features: $O(p \cdot n_m \log n_m)$ per node.

The key optimization sklearn uses: **pre-sort and cache sort orders**. When a split sends
sample $i$ to the left child, the sort order in the left child is a subsequence of the parent's
sort order. This means the sort orders for all child nodes can be derived from the parent in
$O(n_m)$ rather than $O(n_m \log n_m)$. This reduces the per-level cost from
$O(p \cdot n \log n)$ (naively) to $O(p \cdot n)$, giving total training complexity
$O(d \cdot p \cdot n)$ where $d$ is tree depth. For a balanced tree of depth $\log n$,
this yields $O(p \cdot n \log n)$.

> 📊 **Benchmark:** On a dataset with $n = 100,000$ samples and $p = 100$ features, sklearn's
> `DecisionTreeClassifier` trains in approximately 0.5–2 seconds on a modern laptop CPU.
> Inference for $n = 10^6$ samples is typically under 1 second due to the $O(\log n)$ per-sample
> depth traversal. This makes single decision trees practical even for real-time applications.

---

## §3 — The Full Evolution: From CART to Optimal Trees

### 3.1 C5.0 / See5 (Quinlan, 1997) — Speed and Boosting

> 📜 **Origin:** Quinlan, J.R. (1997). *Data Mining Tools See5 and C5.0.* RuleQuest Research.
> URL: https://www.rulequest.com/see5-info.html

C5.0 is the commercial successor to C4.5. It is not open source, but it is available via R's
`C50` package. Key advances over C4.5:

- **Training speed:** Orders of magnitude faster than C4.5, using efficient memory layouts and
  better threshold search.
- **Boosting integration:** C5.0 natively supports boosting (analogous to AdaBoost), running
  multiple trial trees and weighting their predictions.
- **Case weighting:** Samples can carry individual weights; the impurity criterion is weighted
  accordingly.
- **Global cost-complexity pruning:** Replaces C4.5's pessimistic pruning with a more principled
  global procedure.
- **Rule extraction:** Optionally converts the pruned tree to a set of if-then rules, then further
  simplifies the rule set.

C5.0 was historically the accuracy leader on the UCI benchmark suite in the late 1990s and early
2000s, before Random Forests arrived.

**When to use:** Primarily relevant in R. If you are a Python practitioner, you will use sklearn.

---

### 3.2 Conditional Inference Trees — ctree (Hothorn et al., 2006)

> 📜 **Origin:** Hothorn T., Hornik K., Zeileis A. (2006). Unbiased Recursive Partitioning:
> A Conditional Inference Framework. *Journal of Computational and Graphical Statistics*,
> 15(3), 651–674.
> URL: https://www.tandfonline.com/doi/abs/10.1198/106186006X133933

CART and C4.5 have a known bias: features with more possible split points (continuous features
with many unique values, or categorical features with many categories) are preferentially selected
even when a simpler feature is equally or more informative. This is because a feature with $K$
possible splits has $K$ chances to find one that looks good by chance.

Hothorn's insight was to **separate variable selection from threshold selection** using a
permutation test:

1. For each feature $j$, compute a test statistic $T_j$ measuring its association with the
   response (analogous to a correlation or $\chi^2$ statistic).
2. Test $H_0$: feature $j$ is independent of the response, using a permutation distribution.
3. Select the feature with the smallest adjusted p-value (Bonferroni correction over features).
4. Only then find the optimal split threshold for that selected feature.

**Key properties:**
- **Unbiased variable selection:** The test is calibrated to the null hypothesis; continuous and
  binary features have equal probability of being selected under $H_0$.
- **Built-in stopping rule:** Stop when no feature has a significant p-value. No separate pruning
  step is needed — the statistical test is the stopping criterion.

**Python:** Not in sklearn. Available in R's `partykit` package. If you work in Python and need
unbiased importance, use permutation importance from sklearn instead.

---

### 3.3 Extremely Randomized Trees — Extra-Trees (Geurts et al., 2006)

> 📜 **Origin:** Geurts P., Ernst D., Wehenkel L. (2006). Extremely Randomized Trees.
> *Machine Learning*, 63(1), 3–42. DOI: 10.1007/s10994-006-6226-1

CART's greedy search finds the best threshold for the selected feature. Extra-Trees inject
additional randomness: instead of searching all thresholds, draw a random threshold uniformly in
the feature's observed range.

**Algorithm change (single node split in ExtraTree):**

```
For each of K randomly selected features j:
    Draw threshold θ_j ~ Uniform(min(x_j), max(x_j))
    Evaluate G(Q_m, j, θ_j)
Select (j*, θ*) = argmin_j G(Q_m, j, θ_j)
```

In an ensemble (ExtraTrees), both the feature subset and the threshold are randomized. This
creates even more diverse trees than Random Forest (which only randomizes the feature subset).
The diversity reduces variance, and for many datasets ExtraTrees matches or exceeds Random Forest
accuracy while training significantly faster (no exhaustive threshold search).

**sklearn API:**

```python
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor
# As ensemble (almost always preferred):
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
```

**When to prefer:** When training speed matters and you plan to build an ensemble. The extra
randomness hurts a single tree but helps when averaging over many trees.

---

### 3.4 Oblique Decision Trees (Murthy et al., 1994; HHCART, 2015)

Standard CART uses axis-aligned splits: $x_j \leq \theta$. These create decision boundaries that
are parallel to the coordinate axes — staircases in 2D. But many natural decision boundaries are
diagonal.

The classic example is XOR: no axis-aligned split can separate the four XOR points
$\{(0,0), (1,1)\}$ vs $\{(0,1), (1,0)\}$ without exponentially many nodes. But a single oblique
split $x_1 - x_2 \leq 0$ separates them perfectly.

**Oblique split form:**

$$\sum_{j=1}^{p} w_j x_j \leq \theta$$

where $\mathbf{w} \in \mathbb{R}^p$ is a weight vector and $\theta$ is a threshold. The
axis-aligned case is $\mathbf{w} = \mathbf{e}_j$ (a unit vector).

**OC1 (Murthy, Kasif, Salzberg 1994):** Uses coordinate descent over $\mathbf{w}$, optimizing
one weight at a time while holding the others fixed, then perturbing the solution randomly to
escape local minima.

**HHCART (Wickramarachchi et al., 2015):**

> 📜 **Origin:** Wickramarachchi et al. (2015). HHCART: An oblique decision tree.
> arXiv:1504.03415. URL: https://arxiv.org/abs/1504.03415

HHCART uses **Householder reflections** to find oblique splits. The key idea: compute the
eigenvectors of the between-class covariance matrix at each node. Apply a Householder reflection
$\mathbf{H}$ to rotate the feature space so that the dominant discriminant direction aligns with
the first axis. Then find the best axis-aligned split in the reflected space. The Householder
reflection is an orthogonal matrix, so the geometry is preserved.

**When to use oblique trees:**
- When you have strong domain knowledge that the decision boundary is diagonal
- When features are correlated and their combination is more informative than individual features
- When a standard CART tree requires depth 10+ to approximate a boundary that a linear boundary
  would capture in depth 1

**Python:** `scikit_obliquetree` package; not in sklearn core.

---

### 3.5 Optimal Decision Trees — GOSDT (Rudin et al., 2019–2020)

> 📜 **Origin (OSDT):** Hu X., Rudin C., Seltzer M. (2019). Optimal Sparse Decision Trees.
> *NeurIPS 2019*. URL: https://arxiv.org/abs/1904.12847
>
> 📜 **Origin (GOSDT):** Lin J., Zhong C., Hu D., Rudin C., Seltzer M. (2020). Generalized and
> Scalable Optimal Sparse Decision Trees. *ICML 2020*.
> URL: https://arxiv.org/pdf/2006.08690

CART's greedy algorithm finds a locally optimal split at each node but the resulting tree is
globally suboptimal — a different top-level split might lead to a much better overall tree.
Optimal decision tree methods search the entire space of decision trees globally.

The GOSDT objective is:

$$\min_{T} \text{loss}(T, \mathbf{X}, \mathbf{y}) + \lambda |T|$$

where $|T|$ is the number of leaves and $\lambda$ is a sparsity penalty. This is an integer
programming problem. GOSDT uses a branch-and-bound search with tight lower bounds on the
objective, combined with bit-array representations of samples for fast subset operations
(computing which samples fall into each leaf is a bitwise AND).

**Key properties:**
- Provably globally optimal for the given objective
- Practical only for small datasets ($n < 100k$) and sparse binary feature matrices
- Produces the *simplest possible tree* that achieves near-optimal accuracy on the training data

**Python:**
```python
# pip install gosdt-guesses
from gosdt import GOSDTClassifier
clf = GOSDTClassifier(regularization=0.01, time_limit=300)
# Requires binary feature matrix — use binarization first
```

**When to use:** High-stakes interpretability requirements where you need *the* simplest tree,
not just *a* good tree. Medical diagnosis, criminal justice, regulatory reporting.

---

### 3.6 Hoeffding Trees / VFDT — Online Decision Trees (Domingos & Hulten, 2000)

> 📜 **Origin:** Domingos P., Hulten G. (2000). Mining High-Speed Data Streams. *KDD 2000.*

When data arrives as a stream (too large to store, or evolving over time), you cannot train a
full CART tree. The **Very Fast Decision Tree (VFDT)** algorithm uses the **Hoeffding bound**:
after observing $n$ samples, the true mean of any bounded variable lies within
$\varepsilon = \sqrt{\frac{R^2 \ln(1/\delta)}{2n}}$ of the sample mean with probability
$1 - \delta$.

The key insight: if the best attribute $A_1$ has information gain $\Delta G = G(A_1) - G(A_2)$
over the second-best attribute $A_2$, and $\Delta G > \varepsilon(n, \delta)$, then with
probability $1 - \delta$ we can commit to splitting on $A_1$ — even from seeing just $n$ samples
from the stream.

**Python:**
```python
from river.tree import HoeffdingTreeClassifier, HoeffdingTreeRegressor

tree = HoeffdingTreeClassifier()
for xi, yi in zip(X_stream, y_stream):
    tree.learn_one(dict(enumerate(xi)), yi)
pred = tree.predict_one(dict(enumerate(x_new)))
```

**When to use:** Only when you cannot store the full training set — online learning scenarios,
edge devices, concept drift environments.

---

### 3.7 Comparing the Variants: A Practical Guide

Having covered six major variants, here is a consolidated comparison to help you choose:

| Variant | Library | Key Advantage | Key Limitation | Best Use Case |
|---|---|---|---|---|
| **CART (sklearn)** | sklearn | Interpretable, battle-tested, full feature set | Greedy, overfits without pruning | Default choice; single interpretable model |
| **ExtraTreeClassifier** | sklearn | Faster than CART, more randomness | Lower accuracy as single tree | Building ensemble with ExtraTrees |
| **Conditional Inference (ctree)** | R partykit | Unbiased variable selection | Not in sklearn; R-only | When MDI bias is a serious concern |
| **GOSDT** | gosdt-guesses | Provably optimal, globally minimal | Small $n$ only, binary features required | High-stakes interpretability (< 10k rows) |
| **Oblique Trees (HHCART)** | scikit-obliquetree | Handles diagonal boundaries in 1 split | Harder to interpret, less stable | Known oblique boundary, feature interactions |
| **Hoeffding Trees (VFDT)** | river | Streaming, online learning | Weaker accuracy than batch | Data streams, IoT, concept drift |
| **C5.0** | R C50 package | Fast, boosting integration | Commercial, R-only | R users needing maximum accuracy |

### 3.8 Decision Trees as the Foundation of Ensembles

The single most important thing to understand about decision trees in 2026 is that they are
rarely deployed in isolation. They are the *base learner* for the two most powerful supervised
learning families:

**Random Forests (bagging):** $B$ CART trees, each trained on a bootstrap sample of the data
with `max_features='sqrt'`. Predictions are majority-voted (classification) or averaged
(regression). The variance of the ensemble falls like $1/B$ up to the correlation floor $\rho$.
Bootstrap sampling introduces variance in training data; `max_features < p` introduces variance
in split selection — both sources of diversity reduce inter-tree correlation $\rho$, lowering
the variance floor.

**Gradient Boosting (boosting):** $B$ shallow CART trees (typically depth 3–8), each fit to the
negative gradient of the loss function with respect to the current ensemble's predictions. Unlike
Random Forest (parallel trees), GBM fits trees *sequentially*, each correcting the errors of the
previous. XGBoost, LightGBM, and CatBoost are all CART-based GBM implementations.

The key parameter inheritance:
- A Random Forest tree's `max_depth` should typically be `None` (full tree) — the ensemble
  handles overfitting.
- A GBM tree's `max_depth` should be small (3–8) — shallow trees are the "weak learners" that
  define the boosting approach.

If you understand single CART trees deeply, you understand the building blocks of both of these
families. Every time you tune `max_features` in a Random Forest or `max_depth` in XGBoost,
you are applying the same principles from this chapter.

> 📊 **Benchmark:** On the UCI Adult Income dataset ($n = 48842$, 14 features), a typical
> comparison from the sklearn documentation and community benchmarks:
> - Single decision tree (tuned): ~85% accuracy
> - Random Forest (100 trees): ~87–89% accuracy
> - XGBoost (500 trees, tuned): ~90–91% accuracy
> The gap from single tree to ensemble is real and consistent. The single tree's advantage is
> interpretability; the ensemble's advantage is raw performance.

---

## §4 — Hyperparameters: The Complete Guide

Understanding each hyperparameter requires understanding the mechanism, not just the range. Let
me walk through every parameter in `DecisionTreeClassifier` and `DecisionTreeRegressor`.

### 4.1 `criterion`

**Mechanism:** Defines the impurity measure used to evaluate candidate splits.

**Classifier options:**
- `'gini'` (default): $H = 1 - \sum_k p_k^2$. No logarithm — marginally faster. Tends to
  isolate the most frequent class.
- `'entropy'` / `'log_loss'` (aliases): $H = -\sum_k p_k \ln p_k$. Slightly more sensitive to
  low-probability classes. Often produces marginally deeper trees.
- **Practical difference:** On most datasets, the accuracy difference between gini and entropy
  is less than 1%. Try both in cross-validation if you are squeezing for performance, otherwise
  `'gini'` is fine.

**Regressor options:**
- `'squared_error'` (default): MSE criterion. Prediction = mean($y$). Sensitive to outliers.
- `'absolute_error'`: MAE criterion. Prediction = median($y$). Robust to outliers but 3–6×
  slower to train. Use when your target has heavy-tailed noise.
- `'poisson'`: Mean Poisson deviance. Use when $y \geq 0$ and represents counts or rates.
  Requires all $y > 0$ (zero counts are a separate concern).

> ⚠️ **Pitfall:** `criterion='friedman_mse'` was the sklearn 1.0–1.8 name for an improvement
> criterion. It is **deprecated in sklearn 1.9** and will be removed in 1.11. It is mathematically
> identical to `'squared_error'`. Use `'squared_error'` in all new code.

### 4.2 `max_depth`

**Mechanism:** Hard cap on tree depth. Depth 0 = root is a leaf (single prediction). Depth 1 =
a stump (one split, two leaves). `None` (default) = grow until all leaves are pure or other
stopping criteria trigger.

**Effect:**
- Too small: Underfitting. A depth-1 tree on a 10-class problem will always get > 90% wrong.
- Too large: Overfitting. An unconstrained tree on any real dataset achieves 100% training
  accuracy.
- The sweet spot for **interpretability** is depth 3–5: gives 8–32 leaves, fits on one page,
  can be read by a human.
- The sweet spot for **predictive performance** (single tree) is typically depth 5–20, found by
  cross-validation.

**Interaction with `ccp_alpha`:** `max_depth` is a *pre-pruning* constraint. `ccp_alpha` is
*post-pruning*. They interact: a tree grown with `max_depth=10` and then pruned with
`ccp_alpha=0.005` may end up shallower than `max_depth=4` but with a different structure (the
post-pruning method considers global information).

**Tuning strategy:** Grid search over `[3, 5, 7, 10, 15, 20, None]` with 5-fold CV. For a quick
start, try `max_depth=5`.

### 4.3 `min_samples_split`

**Mechanism:** An internal node is only considered for splitting if it contains at least this
many samples. Default is 2, meaning almost any node can be split.

**As integer:** Absolute count. `min_samples_split=10` means nodes with fewer than 10 samples
become leaves.
**As float:** Fraction of total training samples: `min_samples_split=0.02` = `ceil(0.02 × n)`.

**Effect:** Higher values → fewer splits → shallower trees → lower variance, higher bias.
Equivalent to a light pre-pruning.

**Tuning range:** 2–50 (absolute) or 0.005–0.05 (fractional). Lower priority than `max_depth`
and `min_samples_leaf`.

### 4.4 `min_samples_leaf`

**Mechanism:** A split is only made if *both* resulting children have at least `min_samples_leaf`
samples. This is a stricter constraint than `min_samples_split` because it checks both sides of
the split.

**Effect in regression:** Higher `min_samples_leaf` smooths predictions. Each leaf prediction is
an average over more samples, reducing variance of the prediction.

**Interaction:** `min_samples_leaf=k` implies `min_samples_split ≥ 2k`. If you set both, the
effective constraint is `min_samples_split = max(min_samples_split, 2 × min_samples_leaf)`.

**Tuning range:** 1–50 (absolute) or 0.005–0.05 (fractional). This is often the most impactful
pre-pruning hyperparameter. Prefer tuning this over `min_samples_split`.

### 4.5 `min_weight_fraction_leaf`

**Mechanism:** Identical to `min_samples_leaf` but expressed as a fraction of the total sum of
sample weights. Only has an effect when `sample_weight` is passed to `fit()`.

**When to use:** When your samples carry importance weights (survey weights, inverse frequency
weights) and you want the leaf size constraint expressed in weight units, not count units.

**Default:** `0.0` (no constraint). For unweighted problems, this parameter does nothing.

### 4.6 `max_features`

**Mechanism:** At each split, only consider a random subset of this many features. This injects
randomness into a single tree and is the key mechanism of Random Forest when used in an ensemble.

**Options:**
- `None` (default for single tree): consider all $p$ features — deterministic split selection
- `'sqrt'`: $\lfloor\sqrt{p}\rfloor$ features — standard for RF classifiers
- `'log2'`: $\lfloor\log_2 p\rfloor$ features — smaller subset, more randomness
- `int k`: exactly $k$ features
- `float f`: $\lceil f \times p \rceil$ features

**For a single tree:** Setting `max_features < p` introduces randomness and usually hurts accuracy
(the best feature might not be in the random subset). It is only beneficial when building an
ensemble.

**Interaction with `random_state`:** When `max_features < p`, the tree is non-deterministic
unless `random_state` is fixed.

### 4.7 `random_state`

**Mechanism:** Seeds the internal random number generator used for: (a) random feature subsampling
when `max_features < p`, (b) random threshold selection when `splitter='random'`, (c) breaking
ties when two splits score equally.

**When it matters:** Only affects output when `max_features < p` or `splitter='random'`. A tree
with `max_features=None` and `splitter='best'` is fully deterministic given the training data —
`random_state` has no effect.

**Best practice:** Always set `random_state=42` (or any fixed integer) for reproducible results.

### 4.8 `max_leaf_nodes`

**Mechanism:** Grows the tree in **best-first order** (most impurity-reducing split first) until
the tree has `max_leaf_nodes` leaves. This is fundamentally different from `max_depth`.

**Depth-first vs best-first:**
- `max_depth` grows depth-first: expand all nodes at depth $d$ before moving to depth $d+1$.
  All paths have depth ≤ `max_depth`.
- `max_leaf_nodes` grows best-first: always expand the leaf with the highest impurity reduction.
  The tree may be asymmetric — one branch may be very deep while others stay shallow.

**Advantage:** Best-first growth allocates the split budget where it matters most. For a fixed
number of leaves, best-first often produces lower training error than depth-first.

**Interaction with `max_depth`:** When both are set, both constraints apply. The tree stops
when either is met.

### 4.9 `min_impurity_decrease`

**Mechanism:** A split is only made if the weighted impurity decrease satisfies:

$$\frac{n_m}{n}\left[H(Q_m) - \frac{n_l}{n_m} H(Q_l) - \frac{n_r}{n_m} H(Q_r)\right] \geq \text{min\_impurity\_decrease}$$

**Default:** `0.0` — any improvement justifies a split.

**Use case:** When you want to filter out splits that provide negligible gain but consume node
budget. Values of 0.001–0.01 (for normalized impurity) are reasonable.

### 4.10 `class_weight`

**Mechanism:** Scales the impurity criterion contribution of each sample by its class weight.
This is equivalent to resampling — upweighting minority class samples so the tree pays more
attention to them.

**Options:**
- `None` (default): all samples have equal weight
- `'balanced'`: $w_k = \frac{n}{K \cdot n_k}$ where $n_k$ is the count of class $k$. Minority
  classes receive higher weight.
- `dict`: manual weights, e.g., `{0: 1, 1: 10}` for 10× weight on class 1.

**When to use:** Class imbalance. If your dataset is 95% class 0, a default tree will predict
class 0 everywhere (low impurity). `class_weight='balanced'` forces the tree to care about the
minority class.

**Warning (regression):** `class_weight` does not apply to `DecisionTreeRegressor`. Use
`sample_weight` in `fit()` for weighted regression.

### 4.11 `ccp_alpha` — Cost-Complexity Pruning

**Mechanism:** This is *post-pruning* — the tree is grown fully, then pruned back. Any internal
subtree $T_t$ with $\alpha_{\text{eff}}(t) < \texttt{ccp\_alpha}$ is collapsed to a leaf.

**Default:** `0.0` — no pruning.

**Finding the right `ccp_alpha`:** This requires the `cost_complexity_pruning_path()` workflow:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Step 1: Compute the full CCP path on training data
clf_full = DecisionTreeClassifier(random_state=42)
path = clf_full.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]   # exclude the trivial root-only tree

# Step 2: Cross-validate each alpha
cv_scores = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Step 3: Select the alpha that maximizes CV score
best_alpha = ccp_alphas[np.argmax(cv_scores)]
print(f"Best ccp_alpha: {best_alpha:.6f}")

# Step 4: Refit with the selected alpha
final_clf = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
final_clf.fit(X_train, y_train)
```

**Why this is the principled approach:** Unlike `max_depth`, which is a pre-pruning blunt
instrument, `ccp_alpha` uses the actual training data to determine which subtrees are worth
keeping. It is the pruning method Breiman derived from first principles in the CART book.

### 4.12 `monotonic_cst` (Added v1.4)

**Mechanism:** Constrains the tree's prediction function to be monotonically increasing (+1) or
decreasing (-1) with respect to specified features.

**Format:** Array-like of shape `(n_features,)` with values in $\{-1, 0, 1\}$.

**Limitations:**
- Not supported for multiclass classification (`n_classes > 2`)
- Not supported for multioutput regression
- For classifiers, applies to the probability of the positive class

```python
# Feature 0: monotone increasing, feature 1: no constraint, feature 2: monotone decreasing
clf = DecisionTreeClassifier(monotonic_cst=[1, 0, -1], random_state=42)
```

**When to use:** Domain knowledge scenarios — "higher credit score must imply higher approval
probability," "higher temperature must imply higher energy consumption."

### Tuning Playbook

| Hyperparameter | Typical Range | Primary Effect | Recommended Strategy |
|---|---|---|---|
| `max_depth` | 3–20, or None | Variance/bias tradeoff | Grid search [3, 5, 7, 10, 15]; start with 5 |
| `min_samples_leaf` | 1–50 | Leaf smoothing, pre-prune | Grid search [1, 5, 10, 20, 50] |
| `min_samples_split` | 2–50 | Node splitting threshold | Lower priority than min_samples_leaf |
| `ccp_alpha` | 0.0–0.05 | Post-pruning strength | Use CCP path + CV workflow |
| `criterion` | gini/entropy | Split quality measure | Try both; usually negligible diff |
| `max_leaf_nodes` | 10–1000 | Total leaf budget | Alternative to max_depth |
| `class_weight` | None/'balanced'/dict | Class imbalance | 'balanced' for imbalanced data |
| `min_impurity_decrease` | 0.0–0.01 | Minimum gain to split | 0.001–0.005 as soft cutoff |

> 🏆 **Best practice:** For a new problem, start with: `DecisionTreeClassifier(max_depth=5,
> min_samples_leaf=5, random_state=42)`. Then tune `ccp_alpha` using the CCP path workflow,
> which often produces better results than tuning `max_depth` alone.

### 4.13 Hyperparameter Tuning with Optuna

For systematic hyperparameter search, Optuna's TPE sampler is far more efficient than GridSearchCV
across the full hyperparameter space. Here is a complete Optuna study for a decision tree:

```python
import optuna
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.datasets import load_breast_cancer

optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = load_breast_cancer(return_X_y=True)

def objective(trial):
    # Define the hyperparameter search space
    params = {
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 30),
        'max_features': trial.suggest_categorical(
            'max_features', [None, 'sqrt', 'log2', 0.3, 0.5, 0.7]
        ),
        'min_impurity_decrease': trial.suggest_float(
            'min_impurity_decrease', 0.0, 0.01
        ),
        'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.02),
        'random_state': 42,
    }

    clf = DecisionTreeClassifier(**params)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize',
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=200, show_progress_bar=True)

print(f"Best AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Visualize the optimization history
fig = optuna.visualization.plot_optimization_history(study)
fig.show()

# Visualize hyperparameter importance
fig = optuna.visualization.plot_param_importances(study)
fig.show()

# Refit with best parameters
best_clf = DecisionTreeClassifier(**study.best_params)
best_clf.fit(X, y)
```

> 🔧 **In practice:** The Optuna approach is particularly valuable when `ccp_alpha` and
> `max_depth` are both in the search space — they interact strongly, and a grid search over
> both requires $O(n_\text{depth} \times n_\text{alpha})$ evaluations. TPE explores the space
> intelligently, spending more budget near the promising region.

> ⚠️ **Pitfall:** When using Optuna with `ccp_alpha`, be aware that `ccp_alpha=0.0` combined
> with `max_depth=None` will grow an unconstrained tree — this is a degenerate point that
> Optuna may visit early. You can add a pruner callback to skip clearly overfitting configurations.

### 4.14 Interaction Between `max_depth` and `ccp_alpha`

These two hyperparameters control tree complexity in fundamentally different ways, and
understanding their interaction is critical for getting the best results.

**`max_depth` (pre-pruning):** Sets a hard depth ceiling. The tree is never allowed to grow
deeper than this, regardless of how much additional impurity reduction a deeper split could
provide. Pre-pruning is *pessimistic* — it may cut off genuinely useful splits.

**`ccp_alpha` (post-pruning):** Grows the full tree first, then prunes back subtrees where the
complexity cost exceeds the benefit. Post-pruning is *optimistic* — it lets the tree explore
the full depth before deciding what to keep.

**When they interact:**
- `max_depth=10, ccp_alpha=0.0`: Grows to depth 10 greedily, keeps everything.
- `max_depth=10, ccp_alpha=0.005`: Grows to depth 10, then prunes back. May result in a tree
  with effective depth 4–6 if higher branches are not worth their complexity cost.
- `max_depth=4, ccp_alpha=0.005`: Grows to depth 4, then prunes. The CCP pruning may have very
  little to do since the tree is already shallow.

**Recommendation:** For maximum performance, use `max_depth=None` (no pre-pruning ceiling) and
tune only `ccp_alpha` using the CCP path workflow. This lets the algorithm explore the full tree
structure before deciding what to keep.

For interpretability, use `max_depth=4` and skip `ccp_alpha` — the pre-pruning constraint alone
gives you a tree you can read.

---

## §5 — Strengths

### 5.1 Native Interpretability

A decision tree with depth ≤ 4 produces 8–16 leaves and a set of human-readable if-then-else
rules. You can print the entire model logic in a single page and explain it to a non-technical
stakeholder. No other algorithm with comparable predictive power offers this.

**Mechanism:** Each path from root to leaf is a conjunction of conditions on the features. The
prediction at that leaf applies to any sample satisfying all conditions on the path. This is the
exact logic a domain expert would write if they were encoding their knowledge.

**Measurement:** The European Union's GDPR Article 22 requires that automated decision systems
be explainable. Decision trees are among the few models that satisfy this requirement natively,
without post-hoc approximation.

### 5.2 No Feature Preprocessing Required

Decision trees are invariant to monotone transformations of features. Multiplying a feature by
1000, taking its logarithm, or standardizing it does not change the resulting tree — the
optimal threshold shifts, but the partition is identical.

**Mechanism:** The split criterion only compares samples to a threshold, not to each other. The
relative ordering of feature values is all that matters.

**Practical benefit:** You can pass raw features — no StandardScaler, no MinMaxScaler, no
PowerTransformer. This simplifies preprocessing pipelines substantially.

### 5.3 Handles Mixed Feature Types

Decision trees handle continuous, ordinal, and binary features natively with the same split
mechanism. Integer counts, dollar amounts, percentages, and binary flags can all coexist in
the feature matrix without special treatment.

**Categorical features:** sklearn requires numeric encoding, but natural handling of categoricals
is possible (C4.5's multi-way splits, or one-hot encoding with a tree handles categoricals
correctly — each one-hot column is binary and handled identically to a binary feature).

### 5.4 Handles Missing Values (sklearn ≥ 1.3)

Starting in sklearn 1.3, `DecisionTreeClassifier` and `DecisionTreeRegressor` support missing
values natively without imputation. The split search evaluates each threshold candidate twice:
once routing NaN samples left, once routing them right.

**Mechanism:** For each threshold candidate, the algorithm tries both routings and selects
the one with lower weighted impurity. During inference, a sample with a missing value at the
split feature follows the same routing that was optimal during training.

### 5.5 Interpretable Feature Importance

The Mean Decrease in Impurity (MDI) — `clf.feature_importances_` — quantifies each feature's
total contribution to impurity reduction across all splits in the tree:

$$\text{importance}(j) = \sum_{t: \text{node } t \text{ splits on feature } j} \frac{n_t}{n} \cdot \Delta H(t)$$

This is normalized to sum to 1 and provides an immediate ranking of features by predictive
relevance.

**Caveat:** MDI is biased toward high-cardinality features (see §6 for the fix). But for a
first-pass feature selection and variable importance ranking, it is fast and useful.

### 5.6 Nonlinear Interactions Captured Automatically

A depth-3 tree can capture three-way interactions ($x_1$ AND $x_2$ AND $x_3$ predicts the class)
without any manual feature engineering. Logistic regression would require you to explicitly add
the interaction term $x_1 \cdot x_2 \cdot x_3$.

**Mechanism:** Each internal node conditions the split below it on having reached that node.
A split on $x_2$ at depth 2 in the right subtree of an $x_1$ split is implicitly a joint
condition on $x_1 > \theta_1$ AND $x_2 > \theta_2$.

### 5.7 Fast Inference

A decision tree with depth $d$ requires exactly $d$ comparisons to classify a new sample.
For $d = 10$, this is 10 floating-point comparisons — far faster than, say, a kernel SVM
that requires computing distances to all support vectors.

**Practical speed:** sklearn's tree inference in C++ is typically sub-microsecond per sample
for shallow trees, making it suitable for real-time applications.

### 5.8 Interaction With Sample Weights and Imbalanced Data

Unlike many algorithms where class imbalance requires resampling (SMOTE, undersampling), decision
trees handle it cleanly through the `class_weight` parameter. Setting `class_weight='balanced'`
reweights samples so that the minority class contributes proportionally more to the impurity
criterion. This is mathematically equivalent to oversampling the minority class, but without
actually duplicating samples — the computation is identical except for using weighted impurity.

**Mechanism:** When `class_weight={'malignant': 5, 'benign': 1}` is set, the impurity at a
node is computed as if each malignant sample counts as 5 benign samples. A split that
perfectly separates the two malignant samples from 50 benign samples would score the same
impurity as if 10 benign samples had been separated from 50 — the split becomes much more
attractive. The tree is now incentivized to split early on the minority class.

**Contrast with Random Forest:** In a Random Forest, each bootstrap sample naturally draws
minority class examples with equal probability, so resampling is less critical. In a single
tree, `class_weight` is often essential.

### 5.9 Model Transparency Enables Regulatory Compliance

In regulated industries — banking (Basel III/IV), insurance (IFRS 17), clinical medicine (FDA
approval), criminal justice (algorithmic accountability) — model transparency is not a
preference, it is a legal requirement. A decision tree with depth ≤ 4 can be read, audited,
and challenged by regulators, lawyers, and domain experts.

Consider the FICO credit score model: for decades, lenders were required to provide "adverse
action" notices explaining *why* a credit application was denied. A decision tree produces these
explanations natively: "your application was denied because your debt-to-income ratio exceeds
0.43 AND your credit history is less than 2 years." No post-hoc approximation needed.

**Contrast with gradient boosting:** XGBoost with 500 trees at depth 6 produces high accuracy
but requires SHAP for explanations. A single decision tree at depth 4 *is* the explanation.

### 5.10 Graceful Handling of Irrelevant Features

Decision trees are naturally resistant to irrelevant features. A feature that provides zero
information gain will never be selected for any split — it is simply never used. You can pass
a decision tree 100 features where 95 are pure noise, and the tree will mostly ignore them
(modulo small spurious splits on noise in very deep trees).

**Contrast with $k$-NN:** $k$-NN uses all features equally in distance computation, so
irrelevant features pollute the distance metric and dramatically hurt performance. No amount
of careful metric tuning can make $k$-NN ignore a feature the way a decision tree does.

**Limitation:** This resistance only holds for uncorrelated noise. If the noise feature is
correlated with a truly informative feature, the tree may occasionally split on the noise
feature instead of the informative one (split ties). Permutation importance will expose this.

---

## §6 — Weaknesses & Failure Modes

### 6.1 High Variance (Instability)

**Mechanism:** Decision trees are among the highest-variance estimators in ML. The greedy
split selection at the root affects every split below it. If you remove 1% of training samples
or perturb any sample near the root's decision boundary, you may get a completely different
top-level split, cascading into a completely different tree structure.

**Detection:** Train the same tree on 10 bootstrap samples of your training data. Compare the
resulting trees — they will look radically different. The `feature_importances_` will shift
substantially between bootstrap samples.

**Quantification:** For a dataset with $n$ samples and depth $d$, the variance of a tree's
prediction scales roughly as $O(1/n_{\text{leaf}})$ — as leaf samples become fewer, predictions
become noisier. Deep trees have very few samples per leaf.

**Mitigation:** The canonical fix is ensembling — Random Forests and gradient boosting both
reduce variance by averaging many trees. If you need a single tree, use stronger pre-pruning
(`max_depth ≤ 5`) or `ccp_alpha` post-pruning to force larger leaves.

### 6.2 Overfitting with Default Parameters

**Mechanism:** Default `DecisionTreeClassifier()` settings (`max_depth=None`, `min_samples_leaf=1`)
allow the tree to grow until every leaf is pure. On any real dataset, this produces a tree that
memorizes the training data perfectly — 100% training accuracy with dramatically lower test
accuracy.

**Detection:** Training accuracy = 100%, test accuracy < 80% (gap > 20%) signals severe
overfitting. Plot learning curves (training and CV score vs training set size) — a well-fitted
model shows both curves converging at a reasonable score.

**Mitigation:** Always tune at least one complexity-controlling hyperparameter. The minimum viable
setup for any real problem: `DecisionTreeClassifier(max_depth=5, random_state=42)`.

### 6.3 Inability to Extrapolate

**Mechanism:** A decision tree's prediction function is piecewise constant with values equal to
the training leaf means (regression) or training class distributions (classification). It cannot
predict values outside the range of training leaf values.

**Example:** Train a regression tree on housing prices in the range $[100k, 500k]$. When
presented with a new sample whose features correspond to a $700k$ house, the tree will still
predict at most $500k$ — the maximum leaf value in training.

**Detection:** If your test set contains feature values outside the training range, or if you
are forecasting future time periods with a time-series model, extrapolation failure will appear
as systematic bias in out-of-range predictions.

**Mitigation:** Use gradient boosting (XGBoost/LightGBM), which also uses trees but has better
inductive bias via the boosting procedure. For genuine extrapolation, a linear model or a neural
network is more appropriate.

### 6.4 Biased Feature Importance (MDI)

**Mechanism:** The MDI feature importance from `feature_importances_` is biased toward features
with many possible split points. A continuous feature with 10,000 unique values has 9,999
threshold candidates; a binary feature has only 1. By chance alone, the continuous feature is
more likely to appear to offer a good split, even if it is genuinely less informative.

> 📜 **Reference:** Strobl C., Boulesteix A.L., Zeileis A., Hothorn T. (2007). Bias in Random
> Forest Variable Importance Measures. *BMC Bioinformatics*, 8(1), 25.

**Detection:** Compare MDI rankings with permutation importance rankings. If they disagree
substantially — especially if continuous features consistently rank higher than binary ones —
MDI bias is likely.

**Mitigation:** Use `permutation_importance` from sklearn on a held-out test set. This is
unbiased by construction: it measures the actual impact of each feature on predictions by
randomly shuffling it.

```python
from sklearn.inspection import permutation_importance

pi = permutation_importance(
    clf, X_test, y_test,
    n_repeats=30,
    random_state=42,
    scoring='accuracy'
)
```

### 6.5 Cannot Represent Oblique Decision Boundaries

**Mechanism:** Axis-aligned splits create staircase decision boundaries. A diagonal boundary
(like $x_1 + x_2 = 1$) requires exponentially many axis-aligned splits to approximate.

**Detection:** If your accuracy plateaus at high depth and a linear classifier does well, the
boundary is likely oblique and trees are working hard to approximate it with stairs.

**Mitigation:** Use oblique trees, or — more practically — use a Random Forest or gradient
boosting model (averaging many axis-aligned trees effectively approximates oblique boundaries).

### 6.6 Requires Encoding of Categorical Features

**Mechanism:** sklearn's tree implementation requires numeric inputs. Categorical features must
be encoded before fitting. Ordinal encoding (label encoding) imposes a false ordering; one-hot
encoding works but requires many binary splits to recover a natural grouping.

**Example:** A "country" feature with 50 categories becomes 50 binary columns after one-hot
encoding. The tree needs up to 50 splits to recover a grouping like "European countries vs
others" that a native categorical split would achieve in 1 split.

**Mitigation:** For high-cardinality categoricals with tree models, consider using XGBoost or
LightGBM (which handle categoricals natively), or target encoding.

### 6.7 Symmetric Splits Can Miss Asymmetric Patterns

**Mechanism:** CART always produces binary splits — two children per node. But some patterns are
naturally ternary or have very asymmetric distributions (e.g., 95% go left, 5% go right). The
tree will spend many levels processing the majority branch.

**Detection:** If many leaves on one branch of the tree have almost identical predictions, the
tree is spending splits on a region that doesn't need them.

### 6.8 Greedy Splits are Globally Suboptimal

**Mechanism:** CART is a greedy algorithm. It chooses the best split at the current node without
considering what splits will be available in the children. A split that looks slightly worse now
might enable two excellent child splits — but CART will never find it because it committed to
the locally optimal split.

**Formal statement:** CART minimizes the *one-step lookahead* objective $G(Q_m, j, \theta)$, not
the global tree objective $R(T)$. The resulting tree is a local minimum of $R(T)$, not a global
minimum.

**Magnitude of the suboptimality gap:** On real datasets, the gap between CART's greedy solution
and the globally optimal tree (as found by GOSDT) is typically 2–5% in accuracy for small,
sparse datasets. For rich, continuous-feature datasets, CART usually gets very close to optimal
because there are many good split options at each node.

**Mitigation:** For small datasets where interpretability is paramount and every percentage point
matters, use GOSDT. For everything else, CART's approximation is excellent.

### 6.9 No Uncertainty Quantification

**Mechanism:** A decision tree produces a single point prediction (a class label or regression
value). It does not natively produce calibrated probabilities or confidence intervals.
`predict_proba()` returns the training class proportions in the leaf — these are often poorly
calibrated, especially for deep trees with few samples per leaf.

**Example:** A leaf containing 3 samples of class 1 and 0 of class 0 will report
`predict_proba = [0, 1.0]` — perfect confidence in a class 1 prediction, based on just 3
samples. This is overconfident.

**Detection:** Use calibration curves (`sklearn.calibration.CalibrationDisplay`). A well-
calibrated model has points on the diagonal; an overconfident model has points above the diagonal
near the endpoints.

```python
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 7))
CalibrationDisplay.from_estimator(
    clf, X_test, y_test, n_bins=10, ax=ax, name='Decision Tree'
)
ax.set_title('Calibration Curve — Decision Tree')
plt.tight_layout()
plt.show()
```

**Mitigation:** Use `CalibratedClassifierCV` from sklearn to apply Platt scaling or isotonic
regression on top of the tree's raw probability estimates. Or use a shallow tree (depth ≤ 4)
where each leaf contains enough samples for reliable proportion estimates.

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_clf = CalibratedClassifierCV(
    DecisionTreeClassifier(max_depth=5, random_state=42),
    method='isotonic',
    cv=5
)
calibrated_clf.fit(X_train, y_train)
# Now predict_proba() returns calibrated probabilities
```

### 6.10 The Double-Descent Phenomenon in Deep Trees

For very deep trees on large datasets, the classical wisdom "more depth = more overfitting"
breaks down in a fascinating way. Recent work on interpolating estimators shows that if the tree
grows deep enough to *perfectly interpolate* the training data (zero training error), the
test error may actually *improve* as depth increases further — the "benign overfitting" regime.

This is because a perfectly interpolating piecewise-constant estimator with many leaves can
still generalize if the leaves are local enough that training and test samples in the same leaf
are truly similar. The practical relevance is limited for single trees (they do not exhibit
this strongly), but it becomes important for Random Forests, where averaging many perfectly
interpolating trees produces excellent generalization.

**Practical takeaway:** Do not assume that a tree achieving 100% training accuracy is useless.
Check its test performance before pruning — sometimes the unconstrained tree is already
competitive with a pruned version, especially on clean, low-noise datasets.

---

## §7 — Diagnostic Plots & Evaluation

Good diagnostics for a decision tree answer five questions: (1) Is the tree overfitting?
(2) Is the tree the right size? (3) Which features matter? (4) How good is the calibration?
(5) What does the decision boundary look like?

### 7.1 Tree Visualization

The most important diagnostic for decision trees is simply: **look at the tree**. No other
algorithm gives you this.

```python
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# ── Approach 1: sklearn plot_tree (quick, inline) ──────────────────────────
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(
    clf,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,           # color nodes by majority class
    impurity=True,         # show Gini impurity
    proportion=False,      # show raw counts, not proportions
    rounded=True,
    ax=ax,
    fontsize=10
)
plt.title("Decision Tree — Iris Dataset (max_depth=3)")
plt.tight_layout()
plt.savefig("tree_plot.png", dpi=150)
plt.show()

# ── Approach 2: export_text (human-readable rules) ─────────────────────────
rules = export_text(clf, feature_names=list(iris.feature_names))
print(rules)
# |--- petal length (cm) <= 2.45
# |   |--- class: setosa
# |--- petal length (cm) >  2.45
# |   |--- petal width (cm) <= 1.75
# |   |   |--- class: versicolor
# |   |--- petal width (cm) >  1.75
# |   |   |--- class: virginica

# ── Approach 3: dtreeviz (publication quality) ────────────────────────────
import dtreeviz

viz_model = dtreeviz.model(
    clf,
    X_train=X_train,
    y_train=y_train,
    feature_names=list(iris.feature_names),
    target_name='species',
    class_names=list(iris.target_names)
)
viz_model.view()                             # renders in browser/notebook
```

**What to look for:**
- **Good:** Shallow tree (≤ depth 5), each split clearly separates classes (high impurity
  decrease), leaf class distributions are clean (mostly one class).
- **Bad:** Very deep tree with many near-pure but tiny leaves — signs of overfitting. Leaves with
  nearly equal class proportions — the tree is capturing noise.

### 7.2 Learning Curve (Bias-Variance Diagnostic)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
clf = DecisionTreeClassifier(max_depth=5, random_state=42)

train_sizes, train_scores, val_scores = learning_curve(
    clf, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training score', color='blue')
ax.fill_between(train_sizes,
                train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1),
                alpha=0.15, color='blue')
ax.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='CV score', color='orange')
ax.fill_between(train_sizes,
                val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1),
                alpha=0.15, color='orange')
ax.set_xlabel('Training set size')
ax.set_ylabel('Accuracy')
ax.set_title('Learning Curve — Decision Tree (max_depth=5)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**How to read the learning curve:**
- **High variance (overfitting):** Large gap between training score (high, ~1.0) and CV score
  (low). Gap persists even with more data. → Reduce `max_depth`, increase `min_samples_leaf`,
  or increase `ccp_alpha`.
- **High bias (underfitting):** Both training and CV scores are low and converge at a low value.
  → Increase `max_depth`, decrease `min_samples_leaf`.
- **Well-fit:** Training and CV scores are both high and close together, converging as data
  increases.

### 7.3 CCP Alpha Path — Finding the Right Pruning Strength

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Get pruning path
clf_full = DecisionTreeClassifier(random_state=42)
path = clf_full.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]
impurities = path.impurities[:-1]

# CV score for each alpha
cv_scores = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: tree size vs alpha
n_nodes = [
    DecisionTreeClassifier(ccp_alpha=a, random_state=42).fit(X_train, y_train).tree_.node_count
    for a in ccp_alphas
]
ax1.plot(ccp_alphas, n_nodes, 'o-', color='steelblue')
ax1.set_xlabel('ccp_alpha')
ax1.set_ylabel('Number of nodes')
ax1.set_title('Tree Size vs. ccp_alpha')
ax1.grid(True, alpha=0.3)

# Plot 2: CV accuracy vs alpha
ax2.plot(ccp_alphas, cv_scores, 'o-', color='darkorange')
best_idx = np.argmax(cv_scores)
ax2.axvline(ccp_alphas[best_idx], color='red', linestyle='--',
            label=f'Best alpha = {ccp_alphas[best_idx]:.5f}')
ax2.set_xlabel('ccp_alpha')
ax2.set_ylabel('CV Accuracy')
ax2.set_title('CV Accuracy vs. ccp_alpha')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print(f"Best ccp_alpha: {ccp_alphas[best_idx]:.6f}")
print(f"Best CV accuracy: {cv_scores[best_idx]:.4f}")
```

**How to read the CCP alpha plot:**
- Left plot: as alpha increases, nodes are pruned — the curve decreases monotonically.
- Right plot: CV accuracy peaks at an intermediate alpha, then drops as the tree becomes too
  simple. The peak is your optimal alpha.

### 7.4 Feature Importance: MDI vs Permutation

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

clf = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train)

# MDI importance
mdi_imp = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)

# Permutation importance (on test set — unbiased)
pi = permutation_importance(clf, X_test, y_test, n_repeats=30, random_state=42)
perm_imp = pd.Series(pi.importances_mean, index=feature_names).sort_values(ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

mdi_imp.head(15).plot(kind='barh', ax=ax1, color='steelblue')
ax1.set_title('MDI Feature Importance (biased toward high-cardinality)')
ax1.invert_yaxis()

perm_imp.head(15).plot(kind='barh', ax=ax2, color='darkorange')
ax2.set_title('Permutation Importance (unbiased, test set)')
ax2.invert_yaxis()

plt.tight_layout()
plt.show()
```

### 7.5 Decision Boundary Visualization (2D)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                            n_clusters_per_class=1, random_state=42)

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X, y)

xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 300),
                     np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 300))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 7))
ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolor='black', s=40)
ax.set_title('Decision Tree Boundary (max_depth=4) — Note the Staircase Pattern')
plt.tight_layout()
plt.show()
```

**What to look for:** The characteristic "staircase" shape of axis-aligned splits. If the true
boundary is smooth and curving, you need many more splits to approximate it well. Comparing the
boundary at `max_depth=2` vs `max_depth=8` shows how depth trades off between simplicity
and fit.

### 7.6 Metrics: Classification

For classification trees, the key metrics:

- **Accuracy:** `clf.score(X_test, y_test)`. Valid for balanced classes.
- **ROC-AUC:** `roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])`. Use for imbalanced.
- **Precision-Recall AUC:** Better than ROC-AUC when positive class is rare.
- **Confusion matrix:** Always plot it. Shows which classes are confused with which.

For regression trees:
- **R²:** `r2_score(y_test, clf.predict(X_test))`. Fraction of variance explained.
- **RMSE:** `np.sqrt(mean_squared_error(...))`. In original target units.
- **MAE:** `mean_absolute_error(...)`. Robust to outliers.
- **Residual plots:** Predicted vs actual; residuals vs predicted. Systematic patterns indicate
  underfitting or extrapolation failure.

---

## §8 — Explainability & Interpretable AI

### 8.1 Built-in Interpretability: The Whole Tree

A decision tree is natively interpretable in a way no other ML model matches: the model *is*
the explanation. You can print every rule with `export_text()`. You can highlight the path any
specific prediction took with dtreeviz.

```python
import dtreeviz

viz_model = dtreeviz.model(clf, X_train, y_train,
                           feature_names=feature_names,
                           target_name='target',
                           class_names=class_names)

# Show the decision path for a single sample
viz_model.explain_prediction_path(X_test[5])
```

This shows exactly which conditions were evaluated and which branch was taken for sample 5.
No approximation. No surrogate. The exact model logic.

> ⚠️ **Pitfall:** This native interpretability only works for **shallow trees** (depth ≤ 4–5).
> A depth-15 tree has up to 32,768 leaves and is not interpretable by any human. If you need a
> deep tree for performance, use SHAP (§8.2) or permutation importance (§8.3) for explanations —
> not the tree structure itself.

### 8.2 SHAP Values — TreeExplainer

SHAP (SHapley Additive exPlanations) provides a principled, game-theoretic framework for
attributing predictions to features. The `TreeExplainer` computes *exact* SHAP values for tree
models in polynomial time — not an approximation.

The SHAP value $\phi_j$ for feature $j$ in a given prediction answers: "how much does feature
$j$'s value shift the prediction from the baseline (expected model output)?"

$$\hat{f}(\mathbf{x}) = E[\hat{f}] + \sum_{j=1}^{p} \phi_j(\mathbf{x})$$

The SHAP values $\phi_j$ are the unique solution satisfying four axioms (local accuracy,
missingness, consistency, efficiency) and correspond to Shapley values from cooperative game
theory.

```python
import shap
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = list(data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# ── Initialize TreeExplainer ───────────────────────────────────────────────
explainer = shap.TreeExplainer(
    clf,
    data=X_train,                          # background dataset for expected value
    model_output='probability',            # 'raw' for log-odds, 'probability' for P(y=1)
    feature_perturbation='tree_path_dependent'  # exact tree-based SHAP
)

shap_values = explainer(X_test)            # returns Explanation object, shape (n, p)

# ── Plot 1: Waterfall plot (single prediction explanation) ─────────────────
# Shows how each feature pushes the prediction above/below the base rate
shap.plots.waterfall(shap_values[0])

# ── Plot 2: Beeswarm plot (global feature importance + direction) ──────────
# Each point = one sample; x-axis = SHAP value; color = feature value
shap.plots.beeswarm(shap_values)

# ── Plot 3: Bar chart (mean absolute SHAP — global importance) ────────────
shap.plots.bar(shap_values)

# ── Plot 4: Dependence plot (one feature's SHAP vs feature value) ─────────
shap.plots.scatter(shap_values[:, "worst radius"])
```

> ⚠️ **Pitfall (multiclass):** For a 3-class classifier, `explainer.shap_values(X)` returns
> a **list** of arrays, one per class, each of shape `(n_samples, n_features)`. The
> `explainer(X)` syntax (Explanation object) returns an object of shape
> `(n_samples, n_features, n_classes)`. Use `shap_values[:, :, class_idx]` to slice by class.

**How to read the beeswarm plot:**
- Features are ordered by mean absolute SHAP (most important at top)
- Each dot = one test sample
- X-axis = SHAP value (positive = pushes prediction toward positive class)
- Color = feature value (red = high, blue = low)
- Pattern interpretation: if red dots (high feature value) are on the right (positive SHAP),
  the feature has a positive monotone effect. Scattered colors indicate non-monotone effects.

### 8.3 Permutation Importance

Permutation importance answers a different question than SHAP: "how much does model accuracy
*drop* when we destroy the information in feature $j$?"

```python
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt

pi = permutation_importance(
    clf,
    X_test, y_test,
    n_repeats=50,        # 50 random permutations for stable estimates
    random_state=42,
    scoring='roc_auc',   # use your primary metric here
    n_jobs=-1
)

pi_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': pi.importances_mean,
    'importance_std': pi.importances_std
}).sort_values('importance_mean', ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(pi_df['feature'][:15], pi_df['importance_mean'][:15],
        xerr=pi_df['importance_std'][:15], align='center', color='steelblue')
ax.invert_yaxis()
ax.set_xlabel('Mean Decrease in AUC')
ax.set_title('Permutation Importance (test set, 50 repeats)')
plt.tight_layout()
plt.show()
```

**When to prefer permutation over MDI:**
- MDI: fast, computed from training data, biased toward high-cardinality features
- Permutation: slower, uses test set, unbiased, measures actual predictive contribution

> 🏆 **Best practice:** Always report permutation importance from the test set as your primary
> feature importance measure. Use MDI only for quick exploration during model development.

### 8.4 Partial Dependence Plots (PDP) and ICE

PDPs show the marginal effect of one feature on the prediction, averaging over all other features.
ICE (Individual Conditional Expectation) shows the same relationship for each individual sample.

```python
from sklearn.inspection import PartialDependenceDisplay

fig, ax = plt.subplots(figsize=(14, 5))
PartialDependenceDisplay.from_estimator(
    clf,
    X_train,
    features=[0, 1, (0, 1)],     # features 0, 1, and their interaction
    feature_names=feature_names,
    kind='both',                  # 'average' for PDP, 'individual' for ICE, 'both' for both
    grid_resolution=50,
    ax=ax,
    subsample=100,                # for ICE: show 100 random samples
    random_state=42,
)
plt.tight_layout()
plt.show()
```

**What PDPs show for a decision tree:** Since trees are piecewise constant, PDPs are piecewise
constant step functions. The steps correspond to the split thresholds on the plotted feature.
A PDP that looks like a staircase with many small steps (especially at depths ≤ 5) is completely
expected.

**What ICE shows:** Individual lines, one per sample. If the lines are roughly parallel (similar
shape, shifted vertically), the feature's effect is relatively uniform across the population.
If lines cross (heterogeneous effects), the feature interacts with others — some subpopulations
show a positive effect, others negative.

### 8.5 Decision Path Inspection

```python
# Get the decision path for a set of samples
node_indicator = clf.decision_path(X_test[:5])
leaf_ids = clf.apply(X_test[:5])

print("Decision paths for first 5 test samples:")
for sample_idx in range(5):
    node_ids = node_indicator[sample_idx].indices
    print(f"\nSample {sample_idx} (leaf node: {leaf_ids[sample_idx]}):")
    for node_id in node_ids:
        if clf.tree_.children_left[node_id] == -1:  # leaf node
            print(f"  leaf node: predict {clf.classes_[np.argmax(clf.tree_.value[node_id])]}")
        else:
            feat = clf.tree_.feature[node_id]
            thresh = clf.tree_.threshold[node_id]
            val = X_test[sample_idx, feat]
            direction = "<=" if val <= thresh else ">"
            print(f"  node {node_id}: {feature_names[feat]} = {val:.3f} "
                  f"{direction} {thresh:.3f}")
```

This gives you the exact sequence of conditions evaluated for each prediction — the most
granular explanation possible.

---

## §9 — Complete Python Worked Example

We will build a complete end-to-end decision tree analysis across three scenarios:
1. **Iris** (multiclass classification) — visualization focus
2. **Breast Cancer Wisconsin** (binary classification) — CCP pruning + SHAP
3. **California Housing** (regression) — regression tree diagnostics

### Part A: Iris — Visualization and Interpretability

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import (DecisionTreeClassifier, plot_tree,
                           export_text, export_graphviz)
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ── 1. Load data ───────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
feature_names = list(iris.feature_names)
class_names = list(iris.target_names)
df = pd.DataFrame(X, columns=feature_names)
df['species'] = pd.Categorical.from_codes(y, class_names)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes")
print(f"Class balance: {dict(zip(*np.unique(y, return_counts=True)))}")
# Dataset: 150 samples, 4 features, 3 classes
# Class balance: {0: 50, 1: 50, 2: 50}

# ── 2. EDA ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, (ax, feat) in enumerate(zip(axes.flat, feature_names)):
    for cls_idx, cls in enumerate(class_names):
        subset = df[df['species'] == cls][feat]
        ax.hist(subset, alpha=0.6, label=cls, bins=15)
    ax.set_title(feat)
    ax.legend()
plt.suptitle('Iris Feature Distributions by Class')
plt.tight_layout()
plt.show()

# ── 3. Train/test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ── 4. Fit and evaluate a depth-3 tree ────────────────────────────────────
clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    min_samples_leaf=3,
    random_state=42
)
clf.fit(X_train, y_train)
print(f"\nTest accuracy (depth=3): {clf.score(X_test, y_test):.4f}")
# Test accuracy (depth=3): 0.9667

cv_scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
print(f"10-fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
# 10-fold CV: 0.9533 ± 0.0442

# ── 5. Classification report ───────────────────────────────────────────────
y_pred = clf.predict(X_test)
print("\n", classification_report(y_test, y_pred, target_names=class_names))

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=class_names,
    cmap='Blues', ax=ax
)
plt.title('Confusion Matrix — Iris (depth=3)')
plt.tight_layout()
plt.show()

# ── 6. Visualize the tree ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(clf, feature_names=feature_names, class_names=class_names,
          filled=True, impurity=True, proportion=False,
          rounded=True, ax=ax, fontsize=10)
plt.title('Decision Tree — Iris (max_depth=3)')
plt.tight_layout()
plt.show()

# Print text rules
print(export_text(clf, feature_names=feature_names))
```

### Part B: Breast Cancer — CCP Pruning + SHAP

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      StratifiedKFold)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              RocCurveDisplay, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance

# ── 1. Load data ───────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = list(data.feature_names)
class_names = list(data.target_names)

print(f"Dataset: {X.shape} | Classes: {dict(zip(class_names, np.bincount(y)))}")
# Dataset: (569, 30) | Classes: {'malignant': 212, 'benign': 357}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 2. Baseline: unconstrained tree ──────────────────────────────────────
clf_full = DecisionTreeClassifier(random_state=42)
clf_full.fit(X_train, y_train)
print(f"\nUnconstrained tree: depth={clf_full.get_depth()}, "
      f"nodes={clf_full.tree_.node_count}")
print(f"Train accuracy: {clf_full.score(X_train, y_train):.4f}")   # 1.0000
print(f"Test accuracy:  {clf_full.score(X_test, y_test):.4f}")    # ~0.92 (varies)
# Classic example of overfitting: perfect training, imperfect test

# ── 3. CCP pruning workflow ────────────────────────────────────────────────
path = clf_full.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]

print(f"\nCCP path: {len(ccp_alphas)} candidate alpha values "
      f"from {ccp_alphas.min():.6f} to {ccp_alphas.max():.4f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for alpha in ccp_alphas:
    clf_tmp = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    scores = cross_val_score(clf_tmp, X_train, y_train, cv=cv,
                             scoring='roc_auc', n_jobs=-1)
    cv_scores.append(scores.mean())

best_idx = np.argmax(cv_scores)
best_alpha = ccp_alphas[best_idx]
print(f"Best ccp_alpha: {best_alpha:.6f} | CV AUC: {cv_scores[best_idx]:.4f}")

# ── 4. Final model with optimal alpha ─────────────────────────────────────
clf = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
clf.fit(X_train, y_train)

print(f"\nPruned tree: depth={clf.get_depth()}, "
      f"nodes={clf.tree_.node_count}")
y_prob = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)
print(f"Test accuracy: {clf.score(X_test, y_test):.4f}")
print(f"Test AUC:      {roc_auc_score(y_test, y_prob):.4f}")
print("\n", classification_report(y_test, y_pred, target_names=class_names))

# ── 5. CCP path visualization ──────────────────────────────────────────────
node_counts = []
depths = []
for alpha in ccp_alphas:
    t = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42).fit(X_train, y_train)
    node_counts.append(t.tree_.node_count)
    depths.append(t.get_depth())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(ccp_alphas, node_counts, 'o-', color='steelblue', ms=4)
axes[0].set_xlabel('ccp_alpha')
axes[0].set_ylabel('Number of nodes')
axes[0].set_title('Tree Size vs. Alpha')
axes[0].grid(alpha=0.3)

axes[1].plot(ccp_alphas, depths, 'o-', color='mediumseagreen', ms=4)
axes[1].set_xlabel('ccp_alpha')
axes[1].set_ylabel('Tree depth')
axes[1].set_title('Tree Depth vs. Alpha')
axes[1].grid(alpha=0.3)

axes[2].plot(ccp_alphas, cv_scores, 'o-', color='darkorange', ms=4)
axes[2].axvline(best_alpha, color='red', linestyle='--',
                label=f'Best α = {best_alpha:.5f}')
axes[2].set_xlabel('ccp_alpha')
axes[2].set_ylabel('CV AUC')
axes[2].set_title('CV AUC vs. Alpha')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ── 6. ROC Curve ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name='Decision Tree (CCP)')
ax.set_title(f'ROC Curve — Breast Cancer (ccp_alpha={best_alpha:.5f})')
plt.tight_layout()
plt.show()

# ── 7. SHAP TreeExplainer ─────────────────────────────────────────────────
explainer = shap.TreeExplainer(
    clf,
    data=X_train,
    model_output='probability',
    feature_perturbation='tree_path_dependent'
)
shap_values = explainer(X_test)

# Global beeswarm
shap.plots.beeswarm(shap_values, max_display=15, show=True)

# Local waterfall for first misclassified sample
misclassified = np.where(y_pred != y_test)[0]
if len(misclassified) > 0:
    idx = misclassified[0]
    print(f"\nExplaining misclassified sample {idx}: "
          f"true={class_names[y_test[idx]]}, "
          f"predicted={class_names[y_pred[idx]]}")
    shap.plots.waterfall(shap_values[idx], show=True)

# ── 8. Permutation importance ─────────────────────────────────────────────
pi = permutation_importance(clf, X_test, y_test,
                             n_repeats=50, random_state=42,
                             scoring='roc_auc', n_jobs=-1)

pi_df = pd.DataFrame({
    'feature': feature_names,
    'mean': pi.importances_mean,
    'std': pi.importances_std
}).sort_values('mean', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(pi_df['feature'], pi_df['mean'],
        xerr=pi_df['std'], align='center',
        color='steelblue', ecolor='gray', capsize=3)
ax.invert_yaxis()
ax.set_xlabel('Mean decrease in AUC')
ax.set_title('Permutation Importance — Breast Cancer (test set, 50 repeats)')
plt.tight_layout()
plt.show()
```

### Part C: California Housing — Regression Tree

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import shap

# ── 1. Load data ───────────────────────────────────────────────────────────
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = list(data.feature_names)

print(f"Dataset: {X.shape} | Target range: [{y.min():.2f}, {y.max():.2f}]")
# Dataset: (20640, 8) | Target range: [0.15, 5.00]
# Target is median house value in $100k (capped at 5.0 = $500k)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 2. Baseline vs tuned comparison ───────────────────────────────────────
for depth in [3, 5, 7, 10, None]:
    reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"max_depth={str(depth):4s}: RMSE={rmse:.4f}, R²={r2:.4f}, "
          f"depth={reg.get_depth()}, nodes={reg.tree_.node_count}")

# max_depth=3   : RMSE=0.7463, R²=0.4484
# max_depth=5   : RMSE=0.6411, R²=0.5987
# max_depth=7   : RMSE=0.5923, R²=0.6585
# max_depth=10  : RMSE=0.5571, R²=0.6986
# max_depth=None: RMSE=0.6841, R²=0.5419  ← overfitting!

# ── 3. CCP pruning for regression ─────────────────────────────────────────
reg_full = DecisionTreeRegressor(random_state=42)
path = reg_full.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]

cv_rmse = []
for alpha in ccp_alphas[::5]:  # sample every 5th alpha for speed
    reg_tmp = DecisionTreeRegressor(ccp_alpha=alpha, random_state=42)
    scores = cross_val_score(reg_tmp, X_train, y_train, cv=5,
                             scoring='neg_root_mean_squared_error')
    cv_rmse.append(-scores.mean())

best_alpha_idx = np.argmin(cv_rmse)
best_alpha = ccp_alphas[::5][best_alpha_idx]
print(f"\nBest ccp_alpha (regression): {best_alpha:.6f}")
print(f"Best CV RMSE: {cv_rmse[best_alpha_idx]:.4f}")

# ── 4. Final regression model ──────────────────────────────────────────────
reg = DecisionTreeRegressor(ccp_alpha=best_alpha, random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(f"\nFinal model: depth={reg.get_depth()}, nodes={reg.tree_.node_count}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"Test MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Test R²:   {r2_score(y_test, y_pred):.4f}")

# ── 5. Residual plots ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Predicted vs actual
axes[0].scatter(y_test, y_pred, alpha=0.3, s=10, color='steelblue')
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
axes[0].plot(lims, lims, 'r--', linewidth=1)
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title('Predicted vs Actual')

# Residuals vs predicted
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.3, s=10, color='darkorange')
axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Residual (actual - predicted)')
axes[1].set_title('Residual Plot')

# Note the ceiling effect at y=5.0 (cap) — residuals will be positive there
# This is a data artifact, not a model failure

# Residual histogram
axes[2].hist(residuals, bins=50, color='mediumseagreen', edgecolor='white')
axes[2].axvline(0, color='red', linestyle='--')
axes[2].set_xlabel('Residual')
axes[2].set_ylabel('Count')
axes[2].set_title('Residual Distribution')

plt.tight_layout()
plt.show()

# ── 6. Feature importance comparison ─────────────────────────────────────
mdi_imp = pd.Series(reg.feature_importances_, index=feature_names)

pi = permutation_importance(reg, X_test, y_test,
                             n_repeats=30, random_state=42,
                             scoring='neg_root_mean_squared_error')
perm_imp = pd.Series(-pi.importances_mean, index=feature_names)  # RMSE drop

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

mdi_imp.sort_values().plot(kind='barh', ax=ax1, color='steelblue')
ax1.set_title('MDI Feature Importance')

perm_imp.sort_values().plot(kind='barh', ax=ax2, color='darkorange')
ax2.set_title('Permutation Importance (RMSE drop)')

plt.tight_layout()
plt.show()

# ── 7. Partial Dependence Plots ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
PartialDependenceDisplay.from_estimator(
    reg,
    X_train,
    features=[0, 1, 2],         # MedInc, HouseAge, AveRooms
    feature_names=feature_names,
    kind='both',
    grid_resolution=50,
    subsample=200,
    random_state=42,
    ax=ax
)
plt.suptitle('Partial Dependence — California Housing Regression Tree')
plt.tight_layout()
plt.show()

# ── 8. SHAP for regression ────────────────────────────────────────────────
explainer = shap.TreeExplainer(reg, data=X_train[:500])
shap_values = explainer(X_test[:500])

shap.plots.beeswarm(shap_values, max_display=8, show=True)
```

**Interpreting the regression results:**

The California Housing dataset reveals a characteristic decision tree pattern: the unconstrained
tree (R² = 0.54) actually *underperforms* the depth-7 tree (R² = 0.66) because deep trees with
tiny leaves overfit the training noise. The optimal tree found via CCP pruning sits somewhere
between, achieving strong generalization.

The residual plot will show a horizontal ceiling of positive residuals near y = 5.0 — this is a
data artifact (houses are capped at $500k in this dataset). Your model cannot predict above 5.0
regardless of features, so the most expensive houses all have positive residuals. This is not
a model failure but a data limitation — always check whether your target variable has artificial
bounds.

The PDP plots will reveal the characteristic staircase shape of decision tree predictions:
MedIncome (median income) shows a clear positive step-function relationship with house value,
with steep jumps at specific income thresholds corresponding to tree splits.

### The Full Titanic Pipeline — Missing Values and Class Imbalance

Let's add one more compact example showing decision trees handling the Titanic dataset — the
canonical case combining missing values, categorical features, and class imbalance.

```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score

# ── Load Titanic data ──────────────────────────────────────────────────────
df = sns.load_dataset('titanic')

# Select features with a clear story
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'

df = df[features + [target]].copy()
print(f"Dataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
# age has ~177 missing values — we will handle this

# Survival rate: ~38% survived (class imbalance)
print(f"Survival rate: {df[target].mean():.2f}")

# ── Feature engineering ───────────────────────────────────────────────────
# sklearn trees handle numeric NaN natively (sklearn >= 1.3)
# BUT we need to encode categoricals first

# Identify column types
numeric_cols = ['age', 'sibsp', 'parch', 'fare']
categorical_cols = ['pclass', 'sex', 'embarked']

# Encode categoricals with ordinal encoding (0, 1, 2...) — works fine for trees
# because trees don't treat the ordinal order as meaningful (they just find thresholds)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),          # pass numeric as-is (NaN ok)
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
         categorical_cols),
    ],
    remainder='drop'
)

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Pipeline: preprocessor + decision tree ─────────────────────────────────
pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=10,
        class_weight='balanced',    # survival is ~38% — use balanced weighting
        random_state=42
    ))
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

print(f"\nTest accuracy: {pipe.score(X_test, y_test):.4f}")
print(f"Test AUC:      {roc_auc_score(y_test, y_prob):.4f}")
print("\n", classification_report(y_test, y_pred, target_names=['Died', 'Survived']))

# ── Cross-validate ─────────────────────────────────────────────────────────
cv_scores = cross_val_score(pipe, X, y, cv=10, scoring='roc_auc')
print(f"\n10-fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Inspect the rules ─────────────────────────────────────────────────────
from sklearn.tree import export_text
clf_trained = pipe.named_steps['clf']
feature_names_out = (
    numeric_cols +
    list(pipe.named_steps['prep'].named_transformers_['cat'].get_feature_names_out())
)
print("\nTree rules:")
print(export_text(clf_trained, feature_names=feature_names_out))
```

**Key observations from the Titanic tree:**

The resulting tree typically learns the classic Titanic survival story automatically:
1. **Sex** is almost always the first split (females had much higher survival — ~74% vs ~19%).
2. Among males, **class/age** determine survival (first-class males and young boys had higher
   rates).
3. The `class_weight='balanced'` parameter is important here — without it, the tree may
   focus too heavily on predicting "died" (the majority class).

Notice that `age` has many missing values (~20% of passengers). With sklearn ≥ 1.3, these
are handled natively — no imputation step is needed in the pipeline. The tree routes age-missing
passengers to whichever child had lower impurity during training, which corresponds approximately
to the average survival rate given their other features.

---

## §10 — When to Use This Algorithm (Decision Guide)

### Use a Decision Tree When:

| Condition | Rationale |
|---|---|
| **Interpretability is required** | Shallow trees produce human-readable if-then rules |
| **Mixed feature types** (continuous + binary + ordinal) | No preprocessing needed |
| **Non-linear interactions** are expected | Trees capture interactions automatically |
| **Missing values** in features (sklearn ≥ 1.3) | Native support without imputation |
| **Monotonic constraints** needed (v1.4+) | Built-in monotonicity enforcement |
| **Fast inference** required | $O(\log n)$ prediction per sample |
| **Starting point** for an ensemble | Single trees are the foundation of RF/GBM |
| **Exploratory analysis** | Tree rules reveal which thresholds the data supports |

### Do NOT Use a Decision Tree When:

| Condition | Why it fails | Better alternative |
|---|---|---|
| **High performance on tabular data** is the goal | Single trees have high variance and mediocre accuracy | Random Forest or gradient boosting |
| **Extrapolation** beyond training range is needed | Trees predict only within training leaf values | Linear regression, neural networks |
| **Smooth continuous target** (sine wave, time series) | Piecewise-constant approximation is inefficient | Regression splines, neural networks |
| **High-dimensional sparse text/genomics** | Axis-aligned splits inefficient in sparse space | Linear SVM, logistic regression |
| **Extremely imbalanced classes** (< 1% minority) | Even with class_weight, very hard | Isolation Forest, one-class SVM, SMOTE + RF |
| **Oblique decision boundaries** | Needs exponentially many axis-aligned splits | SVM, logistic regression, oblique trees |
| **Very large datasets (> 10M rows)** | Training is $O(n \log n)$; becomes slow | LightGBM with histogram binning, H2O |

### The Decision Flowchart

```
Need interpretability?
├── YES: depth ≤ 4, ccp_alpha tuned → Decision Tree
└── NO:
    Need maximum accuracy?
    ├── YES:
    │   Dataset < 50k rows? → Gradient Boosting (XGBoost/LightGBM)
    │   Dataset > 50k rows? → LightGBM (histogram-based, fast)
    └── NO (reasonable accuracy + speed):
        Dataset < 1M rows? → Random Forest
        Dataset > 1M rows? → LightGBM or H2O
```

> 🏆 **Best practice:** Think of the single decision tree as the interpretable baseline and the
> entry point to the ensemble hierarchy. If a depth-5 decision tree achieves 85% accuracy and
> Random Forest achieves 92%, the 7% gap is the cost of interpretability. Quantify that cost
> explicitly before deciding which model to deploy.

---

## §11 — Resources & Further Reading

### Foundational Papers

1. **Breiman, Friedman, Olshen, Stone (1984) — CART**
   *Classification and Regression Trees.* Wadsworth & Brooks/Cole.
   ISBN: 978-0-412-04841-8
   Semantic Scholar: https://www.semanticscholar.org/paper/Classification-and-Regression-Trees-(Wadsworth-Breiman-Friedman/2203c20aaefc87c72e494b45dc77ed10f3013cb5
   *The definitive mathematical reference. Read Chapters 2–4 for the complete CART derivation and CCP pruning proof.*

2. **Quinlan (1986) — ID3**
   *Induction of Decision Trees.* Machine Learning, 1(1), 81–106.
   DOI: 10.1007/BF00116251
   URL: https://link.springer.com/article/10.1007/BF00116251
   *The first ML decision tree paper. Read for historical context and the information-gain derivation.*

3. **Quinlan (1993) — C4.5**
   *C4.5: Programs for Machine Learning.* Morgan Kaufmann.
   ISBN: 978-1-558-60238-0
   Semantic Scholar: https://www.semanticscholar.org/paper/C4.5%3A-Programs-for-Machine-Learning-Quinlan/807c1f19047f96083e13614f7ce20f2ac98c239e
   *Gain ratio, continuous features, missing values, pessimistic pruning.*

4. **Hothorn, Hornik, Zeileis (2006) — Conditional Inference Trees**
   *Unbiased Recursive Partitioning: A Conditional Inference Framework.*
   Journal of Computational and Graphical Statistics, 15(3), 651–674.
   URL: https://www.tandfonline.com/doi/abs/10.1198/106186006X133933
   *The fix for MDI bias in feature selection.*

5. **Geurts, Ernst, Wehenkel (2006) — Extra-Trees**
   *Extremely Randomized Trees.* Machine Learning, 63(1), 3–42.
   DOI: 10.1007/s10994-006-6226-1
   *The basis for ExtraTreesClassifier in sklearn.*

6. **Strobl, Boulesteix, Zeileis, Hothorn (2007) — MDI Bias**
   *Bias in Random Forest Variable Importance Measures.*
   BMC Bioinformatics, 8(1), 25.
   *The definitive analysis of why MDI is biased and when permutation importance is better.*

7. **Lin, Zhong, Hu, Rudin, Seltzer (2020) — GOSDT**
   *Generalized and Scalable Optimal Sparse Decision Trees.* ICML 2020.
   URL: https://arxiv.org/pdf/2006.08690
   *Globally optimal trees via branch-and-bound.*

8. **Wickramarachchi et al. (2015) — HHCART Oblique Trees**
   arXiv:1504.03415. URL: https://arxiv.org/abs/1504.03415

9. **Domingos & Hulten (2000) — VFDT (Hoeffding Trees)**
   *Mining High-Speed Data Streams.* KDD 2000.
   *The online learning decision tree algorithm.*

### sklearn Documentation

10. **sklearn Decision Trees User Guide**
    https://scikit-learn.org/stable/modules/tree.html
    *Complete parameter reference, algorithm descriptions, examples.*

11. **DecisionTreeClassifier API Reference**
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

12. **CCP Pruning Example (official)**
    https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
    *The official CCP alpha selection workflow with plots.*

13. **Permutation Importance vs MDI Comparison**
    https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html

14. **Monotonic Constraints Demo**
    https://scikit-learn.org/stable/auto_examples/ensemble/plot_monotonic_constraints.html

### SHAP

15. **SHAP TreeExplainer Documentation**
    https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html

16. **Tree SHAP for Simple Models Notebook**
    https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Understanding%20Tree%20SHAP%20for%20Simple%20Models.html

### Visualization

17. **dtreeviz GitHub** — https://github.com/parrt/dtreeviz
    *Beautiful tree visualization. `pip install dtreeviz`. Requires Graphviz system binary.*

18. **Five Ways to Visualize Decision Trees** — https://mljar.com/blog/visualize-decision-tree/
    *Side-by-side comparison of plot_tree, graphviz, dtreeviz.*

### Books

19. **Breiman et al. (1984)** — The CART book (above). Still the definitive mathematical reference.
20. **Hastie, Tibshirani, Friedman** — *The Elements of Statistical Learning*, 2nd ed., Chapter 9.
    Free PDF: https://hastie.su.domains/ElemStatLearn/
    *The best single chapter on decision trees and ensembles in the ML literature.*
21. **James, Witten, Hastie, Tibshirani** — *An Introduction to Statistical Learning*, Chapter 8.
    Free PDF: https://www.statlearning.com/
    *Accessible treatment with R and Python labs.*

---

*End of chapter.*
