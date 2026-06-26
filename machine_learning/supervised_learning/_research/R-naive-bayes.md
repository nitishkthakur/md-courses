# Research Dossier: Naive Bayes Classifiers
**Feed for:** `naive-bayes.md` (chapter file in the Supervised Learning Masterclass)
**Prepared by:** Research Agent
**Date:** 2026-06-25
**Status:** Ready for chapter author

---

## Scope — Which Algorithm File This Feeds

This dossier feeds the `naive-bayes.md` chapter. It covers all five sklearn Naive Bayes variants
(`GaussianNB`, `MultinomialNB`, `BernoulliNB`, `ComplementNB`, `CategoricalNB`), the full
theoretical derivation from Bayes' theorem, history from the 1960s through modern variants, the
zero-probability / calibration / independence-assumption breakdown problems, text classification
pipelines, out-of-core learning, and advanced variants (TAN, HNB, AODE, semi-naive Bayes).

---

## Verified Packages — Import Paths, Versions, Key API Methods

### Scikit-learn (primary — all five variants)

**Version confirmed in research:** 1.5.x (stable). Docs at https://scikit-learn.org/1.5/modules/naive_bayes.html

```python
from sklearn.naive_bayes import (
    GaussianNB,
    MultinomialNB,
    BernoulliNB,
    ComplementNB,
    CategoricalNB,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OrdinalEncoder
```

All five variants share the same sklearn estimator interface:

```python
# Universal interface
clf.fit(X, y, sample_weight=None)
clf.partial_fit(X, y, classes=None, sample_weight=None)   # ALL five support this
clf.predict(X)
clf.predict_proba(X)
clf.predict_log_proba(X)
clf.predict_joint_log_proba(X)  # new in sklearn 1.2
clf.score(X, y, sample_weight=None)
```

**Key version note:** `force_alpha` parameter was added in 1.2; its default changed from `False`
to `True` in sklearn 1.4. In 1.5.x `force_alpha=True` is the default for all four discrete NB
classifiers.

### NLTK (NaiveBayesClassifier — NLP workflows)

```python
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

# train_set: list of (feature_dict, label) tuples
classifier = NaiveBayesClassifier.train(train_set)
classifier.classify({"word_good": True, "word_bad": False})
classifier.show_most_informative_features(n=10)
classifier.prob_classify(features)   # returns probability distribution
```

NLTK's NaiveBayesClassifier works on feature dictionaries (not raw counts), uses Laplace
smoothing, and is tightly integrated with NLTK's tokenizers/corpora. Best for educational NLP
pipelines; sklearn is faster at scale.

### TextBlob (uses NLTK NaiveBayesClassifier internally)

```python
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from textblob.sentiments import NaiveBayesAnalyzer

# Custom classifier
train = [("Great product", "pos"), ("Terrible experience", "neg")]
cl = NaiveBayesClassifier(train)
cl.classify("I love it")           # returns class label
cl.prob_classify("I love it")      # returns prob distribution

# Built-in sentiment (pre-trained on movie reviews corpus)
blob = TextBlob("I love this", analyzer=NaiveBayesAnalyzer())
blob.sentiment   # Sentiment(classification='pos', p_pos=0.9, p_neg=0.1)
```

### Other relevant packages

| Package | Class/function | Notes |
|---------|---------------|-------|
| `river` | `river.naive_bayes.GaussianNB`, `MultinomialNB`, `BernoulliNB` | True online learning (single sample update) |
| `pomegranate` | `NaiveBayes` | GPU support via PyTorch backend; supports mixed distributions per feature |
| `weka` (via `python-weka-wrapper3`) | `NaiveBayes`, `NaiveBayesMultinomialText` | Java-based reference implementation |
| `h2o` | `H2OEstimator` via `h2o.estimators.naive_bayes.H2ONaiveBayesEstimator` | Distributed NB, handles missing values natively |

---

## The Original Algorithm — Paper Citation, Key Equations, Problem Solved

### Historical roots

Naive Bayes traces to Bayes' original theorem (1763, posthumously published), but its application
as a classifier in machine learning came much later. The earliest documented use as a text
classifier appeared in the late 1950s–1960s in information retrieval. The specific "naive Bayes
classifier" formulation for text gained prominence with:

- **Maron, M. E. (1961). "Automatic Indexing: An Experimental Inquiry."** *Journal of the ACM,
  8*(3), 404–417. — One of the earliest probabilistic document classifiers.
- **Duda, R. O., & Hart, P. E. (1973).** *Pattern Classification and Scene Analysis.* Wiley. —
  Formalized the generative classifier view.

The foundational theoretical analysis is:

> 📜 **Origin/Citation:**
> **Domingos, P., & Pazzani, M. (1997). "On the Optimality of the Simple Bayesian Classifier
> under Zero-One Loss."** *Machine Learning, 29*(2–3), 103–130.
> URL: https://gwern.net/doc/ai/1997-domingos.pdf
>
> This is the paper that definitively explained *why* naive Bayes works despite its manifestly
> false independence assumption.

The "empirical study" companion is:

> 📜 **Origin/Citation:**
> **Rish, I. (2001). "An Empirical Study of the Naive Bayes Classifier."** IJCAI 2001 Workshop
> on Empirical Methods in AI. URL: https://faculty.cc.gatech.edu/~isbell/reading/papers/Rish.pdf

### The Bayes Theorem Foundation

Given a feature vector $\mathbf{x} = (x_1, x_2, \ldots, x_p)$ and class labels $y \in \{c_1, \ldots, c_K\}$, Bayes' theorem gives:

$$P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y) \cdot P(y)}{P(\mathbf{x})}$$

where:
- $P(y)$ is the **prior** (class frequency in training data)
- $P(\mathbf{x} \mid y)$ is the **likelihood** (how probable is this feature vector given the class)
- $P(\mathbf{x})$ is the **evidence** (a normalizing constant, same for all classes)
- $P(y \mid \mathbf{x})$ is the **posterior** (what we want)

Since $P(\mathbf{x})$ is class-independent, the MAP decision rule becomes:

$$\hat{y} = \arg\max_{y} P(y) \cdot P(\mathbf{x} \mid y)$$

### The Naive Independence Assumption

The likelihood $P(\mathbf{x} \mid y) = P(x_1, x_2, \ldots, x_p \mid y)$ is intractable to estimate directly (the joint is exponentially large). The **naive** move is to assume conditional independence of all features given the class:

$$P(\mathbf{x} \mid y) = \prod_{i=1}^{p} P(x_i \mid y)$$

This reduces $\mathcal{O}(K \cdot 2^p)$ parameters to $\mathcal{O}(K \cdot p)$ — an exponential compression. The full decision rule:

$$\hat{y} = \arg\max_{y} P(y) \prod_{i=1}^{p} P(x_i \mid y)$$

In log space (to avoid underflow with many features):

$$\hat{y} = \arg\max_{y} \left[ \log P(y) + \sum_{i=1}^{p} \log P(x_i \mid y) \right]$$

Sklearn stores `class_log_prior_` and `feature_log_prob_` — prediction is a dot product plus bias, making NB formally a **linear classifier** in log-probability space.

### Why "Naive" and Why It Still Works

The independence assumption is "naive" because features in real data are almost never conditionally independent. In text, the words "New" and "York" co-occur far more than chance would predict. Yet NB achieves competitive accuracy.

**Domingos & Pazzani (1997) proved:** NB is optimal under zero-one loss if and only if the joint probability distribution induces the correct ordering of class posteriors — i.e., if $P(c_1 | \mathbf{x}) > P(c_2 | \mathbf{x})$ in truth, NB only needs to satisfy $\hat{P}(c_1 | \mathbf{x}) > \hat{P}(c_2 | \mathbf{x})$. Correct probability estimates are *not* required for correct classification. Two conditions favor this:
1. Feature correlations are **symmetric across classes** — errors cancel
2. One class has a strongly **dominant prior** — the ranking is hard to flip

This explains why NB excels at spam filtering (very skewed priors, repetitive patterns) and fails on finely-grained topics with similar feature distributions.

---

## Major Variants & Evolution — Each With Python Package

### 1. Gaussian Naive Bayes (sklearn `GaussianNB`)

**Problem solved:** Standard NB requires a discrete likelihood model. Continuous features must be discretized (lossy) or a continuous distribution must be assumed.

**Solution:** Assume each feature $x_i$ in class $c$ follows a univariate Gaussian:

$$P(x_i \mid y = c) = \frac{1}{\sqrt{2\pi\sigma_{ci}^2}} \exp\left(- \frac{(x_i - \mu_{ci})^2}{2\sigma_{ci}^2}\right)$$

Parameters estimated by MLE: $\mu_{ci} = \frac{1}{n_c}\sum_{j: y_j=c} x_{ij}$ and $\sigma_{ci}^2 = \frac{1}{n_c}\sum_{j: y_j=c}(x_{ij} - \mu_{ci})^2$.

**Decision boundary derivation:** Taking the log-ratio of posteriors for two classes with equal variance $\sigma^2$, the quadratic terms in $x_i^2$ cancel, yielding:

$$\log \frac{P(c_1|\mathbf{x})}{P(c_2|\mathbf{x})} = \sum_{i=1}^{p} \frac{\mu_{c_1,i} - \mu_{c_2,i}}{\sigma_i^2} x_i + \text{const} = \mathbf{w}^T \mathbf{x} + b$$

This is **linear** when per-feature variances are equal across classes (ties to LDA/logistic regression). When variances differ per class, the $x_i^2$ terms don't cancel → **quadratic boundary** (ties to QDA). GaussianNB thus interpolates between LDA and QDA depending on data.

**Origin:** Formalized in Duda & Hart (1973); widely used since the 1980s in pattern recognition.
**Package:** `sklearn.naive_bayes.GaussianNB`
**When to prefer:** Continuous features with approximately Gaussian marginals per class. Great baseline before trying SVM or LR.

### 2. Multinomial Naive Bayes (sklearn `MultinomialNB`)

**Problem solved:** Text classification with word counts. Features are non-negative integers (or real-valued frequencies like TF-IDF).

**Model:** Feature $x_i$ represents count of feature $i$ in a document. Class-conditional likelihood is multinomial:

$$P(\mathbf{x} \mid y = c) \propto \prod_{i=1}^{p} \theta_{ci}^{x_i}$$

where $\theta_{ci}$ is the probability of feature $i$ in class $c$. MLE with Laplace smoothing (alpha):

$$\hat{\theta}_{ci} = \frac{N_{ci} + \alpha}{N_c + \alpha \cdot p}$$

$N_{ci}$ = count of feature $i$ in all class-$c$ documents; $N_c$ = total token count in class $c$; $p$ = vocabulary size.

**Classification rule:**

$$\hat{y} = \arg\max_c \left[ \log P(c) + \sum_{i=1}^{p} x_i \log \hat{\theta}_{ci} \right]$$

This is literally a linear function of the count vector: $\mathbf{x}^T \log\hat{\boldsymbol{\theta}}_c + \log P(c)$.

**Origin:** McCallum & Nigam (1998) "A Comparison of Event Models for Naive Bayes Text Classification." AAAI Workshop. URL: https://www.cs.cmu.edu/~knigam/papers/multinomial-aaai98.pdf

**Package:** `sklearn.naive_bayes.MultinomialNB`
**When to prefer:** Word count or TF-IDF features, longer documents, larger vocabularies. The default choice for text classification.

### 3. Bernoulli Naive Bayes (sklearn `BernoulliNB`)

**Problem solved:** When documents are short and presence/absence of a word matters more than its frequency. Also useful for binary feature vectors (binarized images, one-hot encoded booleans).

**Model:** Each feature $x_i \in \{0, 1\}$ (binary). The key difference from MultinomialNB: BernoulliNB **explicitly models the absence** of features:

$$P(\mathbf{x} \mid y = c) = \prod_{i=1}^{p} P(i \mid c)^{x_i} \cdot (1 - P(i \mid c))^{1 - x_i}$$

This penalizes a class if a feature that is indicative of it is absent. MultinomialNB simply ignores zero-count features in the sum. This makes BernoulliNB better when **not seeing a word is informative** (e.g., spam filter: absence of "dear" is evidence against legitimate email).

The `binarize=0.0` parameter applies a threshold: features above threshold → 1, below → 0. Set `binarize=None` if input is already binary.

**Origin:** Same as MultinomialNB; the two-event-model comparison was formalized by McCallum & Nigam (1998). The "Bernoulli document model" traces to Robertson & Sparck Jones (1976) BM25 precursor work.

**Package:** `sklearn.naive_bayes.BernoulliNB`
**When to prefer:** Short texts, binary features, small vocabularies, when absence of a feature is informative. Underperforms MultinomialNB on long documents.

### 4. Complement Naive Bayes (sklearn `ComplementNB`) — Rennie et al. 2003

**Problem solved:** MultinomialNB suffers from **weight skew** on imbalanced datasets. When one class dominates the training set, its estimated $\theta$ parameters absorb more probability mass, biasing predictions toward the majority class.

**Innovation:** Instead of estimating $P(\text{feature} \mid \text{class } c)$, CNB estimates $P(\text{feature} \mid \text{complement of class } c)$ — all other classes:

$$\hat{\theta}_{ci} = \frac{\alpha_i + \sum_{j: y_j \neq c} d_{ij}}{\alpha + \sum_{j: y_j \neq c} \sum_{k} d_{kj}}$$

where $d_{ij}$ is the count of feature $i$ in document $j$. Weights: $w_{ci} = \log \hat{\theta}_{ci}$.

**Optional normalization (`norm=True`):**

$$w_{ci} \leftarrow \frac{w_{ci}}{\sum_j |w_{cj}|}$$

This handles variable document length better.

**Classification rule:** Assign to class with **smallest** complement weight (most poorly explained by everything-else):

$$\hat{c} = \arg\min_c \sum_i t_i \cdot w_{ci}$$

**Why it works:** CNB's parameter estimates are more uniform across classes, reducing the effect of imbalance. Rennie et al. showed CNB "regularly outperforms MNB, often by a considerable margin" on Reuters and 20 Newsgroups benchmarks.

> ⚠️ **Pitfall:** The default `norm=False` mirrors Mahout/Weka implementations but does NOT follow the full algorithm in the paper. Set `norm=True` if following the paper exactly or when document lengths vary greatly.

**Citation:**
> Rennie, J. D. M., Shih, L., Teevan, J., & Karger, D. R. (2003). "Tackling the Poor Assumptions of Naive Bayes Text Classifiers." *ICML 2003, Vol. 3*, pp. 616–623.
> URL: https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf

**Package:** `sklearn.naive_bayes.ComplementNB`
**When to prefer:** Imbalanced text classification datasets. Nearly always prefer over MultinomialNB when classes are unbalanced.

### 5. Categorical Naive Bayes (sklearn `CategoricalNB`) — sklearn 0.22+

**Problem solved:** Features are nominal categories (not ordinal, not counts, not continuous). Examples: blood type, country of origin, color.

**Model:** For feature $i$ with $n_i$ categories:

$$P(x_i = t \mid y = c) = \frac{N_{tic} + \alpha}{N_c + \alpha \cdot n_i}$$

where $N_{tic}$ = count of category $t$ for feature $i$ in class $c$; $N_c$ = samples in class $c$.

**Key requirement:** Categories must be non-negative integers $\{0, 1, \ldots, n_i - 1\}$. Use `OrdinalEncoder` for string categories before feeding to CategoricalNB.

```python
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
X_encoded = enc.fit_transform(X_categorical)
clf = CategoricalNB()
clf.fit(X_encoded, y)
```

The `min_categories` parameter ensures a minimum number of categories per feature, even if some weren't observed in training.

**Package:** `sklearn.naive_bayes.CategoricalNB`
**When to prefer:** Datasets with purely nominal features (no natural order). Do NOT use for ordinal data — CategoricalNB ignores ordering.

---

### Advanced Variants (Beyond sklearn)

#### Tree-Augmented Naive Bayes (TAN)

**Paper:** Friedman, N., Geiger, D., & Goldszmidt, M. (1997). "Bayesian Network Classifiers."
*Machine Learning, 29*, 131–163. DOI: 10.1023/A:1007465528199
URL: https://link.springer.com/article/10.1023/A:1007465528199

**Problem:** NB assumes all features are independent given the class. TAN relaxes this to allow each feature to have at most one other feature as a parent (in addition to the class).

**Algorithm:** Uses the **Chow-Liu algorithm** to find the maximum weight spanning tree over the feature dependency graph, where edge weights are conditional mutual information values $I(X_i; X_j \mid Y)$. Steps:
1. Compute $I(X_i; X_j \mid Y)$ for all feature pairs.
2. Build maximum spanning tree over these weights.
3. Root the tree and direct edges away from root.
4. Augment standard NB with these feature-feature dependencies.

**Complexity:** $\mathcal{O}(p^2 n)$ — manageable for moderate $p$. No search required (polynomial algorithm via Prim's or Kruskal's).

**Result:** TAN significantly outperforms NB on 21 benchmark datasets while maintaining computational tractability. Outperforms naive Bayes and Bayesian network approaches without search.

**Python:** No native sklearn implementation. Available in `pgmpy` (`pgmpy.models.BayesianNetwork` + Chow-Liu structure learning), WEKA (`TAN` classifier).

```python
from pgmpy.estimators import TreeSearch
from pgmpy.models import BayesianNetwork
est = TreeSearch(data, root_node="class_col")
dag = est.estimate(estimator_type="tan")
```

#### Hidden Naive Bayes (HNB)

**Paper:** Zheng, F., & Webb, G. I. (2005). "A Comparative Study of Semi-Naive Bayes Methods in Classification Learning." PAKDD 2005.
Also: Naive Bayes using Hidden Variables — AAAI 2005. URL: https://aaai.org/papers/00919-AAAI05-145-hidden-naive-bayes/

**Key idea:** For each feature $X_i$, create a hidden parent $H_i$ that is a weighted average of the influences of all other features $X_j$ on $X_i$. This captures one-dependence without explicit structure learning.

**When to use:** When you want semi-naive Bayes behavior but don't want to run structure learning (TAN). Not available in sklearn; research paper implementations only.

#### Averaged One-Dependence Estimators (AODE)

**Paper:** Webb, G. I., Boughton, J. R., & Wang, Z. (2005). "Not So Naive Bayes: Aggregating One-Dependence Estimators." *Machine Learning, 58*, 5–24.
URL: https://link.springer.com/article/10.1007/s10994-005-4258-6

**Key idea:** Average over all possible $p$ one-dependence classifiers, each using a different "super-parent" feature. More stable than choosing one parent (as in TAN). AODE ensemble:

$$P_{\text{AODE}}(y \mid \mathbf{x}) \propto \sum_{i: n(x_i) \geq m} P(y, x_i) \prod_{j \neq i} P(x_j \mid y, x_i)$$

where $n(x_i)$ is the count of feature value $x_i$ in training (only use super-parents seen enough times, controlled by threshold $m$).

**Complexity:** $\mathcal{O}(p^2 n)$ training. Substantially outperforms NB without structure search. Not available in sklearn; available in WEKA as `AODE`.

---

## Hyperparameter Reference — Verified Parameter Names and Defaults

### GaussianNB

```python
GaussianNB(
    priors=None,          # array-like (n_classes,) — if None, estimated from data as class freq
    var_smoothing=1e-9    # float — adds this fraction of max feature variance to all variances
)
```

| Parameter | Default | Range | Effect | Tuning |
|-----------|---------|-------|--------|--------|
| `priors` | `None` | array summing to 1 | Sets class prior probabilities | Set manually for known class imbalance; leave `None` to estimate from data |
| `var_smoothing` | `1e-9` | `[1e-12, 1e-1]` | Prevents zero variance (numeric stability) | Increase if features have very small variance; treat as regularization strength |

**Fitted attributes:** `theta_` (class-conditional means, shape `(n_classes, n_features)`), `var_` (class-conditional variances), `class_prior_` (prior probabilities), `epsilon_` (absolute smoothing added to variances).

**Tuning `var_smoothing`:**
```python
from sklearn.model_selection import GridSearchCV
param_grid = {"var_smoothing": np.logspace(-9, -1, 100)}
gs = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring="accuracy")
gs.fit(X_train, y_train)
```

### MultinomialNB

```python
MultinomialNB(
    alpha=1.0,            # float or array (n_features,) — Laplace smoothing; 1.0 = Laplace, <1 = Lidstone, 0 = no smoothing
    force_alpha=True,     # bool — if True, alpha used as-is even if alpha<1e-10
    fit_prior=True,       # bool — if False, uses uniform prior P(y)=1/K
    class_prior=None      # array-like (n_classes,) — manual prior override
)
```

| Parameter | Default | Range | Effect | Tuning |
|-----------|---------|-------|--------|--------|
| `alpha` | `1.0` | `[0, ∞)` | Additive smoothing for zero counts | Tune over `[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]`; often best between 0.1–1.0 |
| `force_alpha` | `True` | bool | Whether to allow very small alpha | Leave `True` unless debugging |
| `fit_prior` | `True` | bool | Learn class priors from data | Set `False` for class-balanced enforced prior; rarely helps |
| `class_prior` | `None` | array summing to 1 | Manual prior | Use when you have domain knowledge about true class distribution |

**Fitted attributes:** `feature_log_prob_` (shape `n_classes × n_features`), `class_log_prior_`, `feature_count_`, `class_count_`. The `coef_` and `intercept_` properties alias `feature_log_prob_` and `class_log_prior_` for linear model compatibility.

### BernoulliNB

```python
BernoulliNB(
    alpha=1.0,
    force_alpha=True,
    binarize=0.0,         # float or None — threshold for binarizing; None = already binary
    fit_prior=True,
    class_prior=None
)
```

Additional parameter:

| Parameter | Default | Range | Effect | Tuning |
|-----------|---------|-------|--------|--------|
| `binarize` | `0.0` | `float` or `None` | Threshold: features > binarize → 1 | Tune when input is already continuous (e.g., TF-IDF scores); `None` if input is binary |

**Tuning `binarize`:** If using TF-IDF input (real-valued) with BernoulliNB, tune the binarize threshold on the validation set or set equal to the median feature value.

### ComplementNB

```python
ComplementNB(
    alpha=1.0,
    force_alpha=True,
    fit_prior=True,       # Only used in edge case: single class
    class_prior=None,     # Not used (CNB ignores class priors)
    norm=False            # bool — second weight normalization per original paper
)
```

Note: `class_prior` is accepted but **not used** in CNB — CNB does not use class priors. `fit_prior` also has no effect in the multi-class case.

### CategoricalNB

```python
CategoricalNB(
    alpha=1.0,
    force_alpha=True,
    fit_prior=True,
    class_prior=None,
    min_categories=None   # int or array (n_features,) — minimum categories per feature
)
```

| Parameter | Default | Range | Effect | Tuning |
|-----------|---------|-------|--------|--------|
| `min_categories` | `None` | `int ≥ 1` | Ensures minimum number of categories seen; prevents index error on unseen categories at test time | Set to maximum number of categories per feature when test set may have unseen category values |

---

## Common Errors & Pitfalls

### 1. The Zero-Probability Problem (without smoothing)

Without smoothing (`alpha=0`), any feature-class combination with zero training count gives
$P(x_i \mid c) = 0$, which zeroes out the entire posterior. This can cause a test sample to have
probability 0 for ALL classes.

```python
# BAD: alpha=0 with rare features
clf = MultinomialNB(alpha=0, force_alpha=True)

# GOOD: always use smoothing
clf = MultinomialNB(alpha=1.0)  # Laplace smoothing
```

The fix: Laplace smoothing adds 1 (or `alpha`) to every count. The smoothed estimate is:
$\hat{\theta}_{ci} = (N_{ci} + \alpha) / (N_c + \alpha \cdot p)$.
For $\alpha = 1$: **Laplace smoothing**. For $0 < \alpha < 1$: **Lidstone smoothing**.

### 2. Numerical Underflow with Many Features

Multiplying thousands of small probabilities causes floating-point underflow to zero. Sklearn
internally works in **log space** (stores `feature_log_prob_`) and uses `logsumexp` for
normalization. Never implement NB from scratch using raw probability multiplication.

```python
# Underflow: NEVER do this
prob = np.prod([P(xi | c) for xi in x])   # → 0 for long documents

# CORRECT: use log probabilities (sklearn does this automatically)
log_prob = np.sum([np.log(P(xi | c)) for xi in x])
```

### 3. Calibration: NB is a Bad Probability Estimator

Despite its generative foundation, NB produces **severely miscalibrated** probabilities — they
cluster at 0 and 1. The root cause: multiplying $p$ feature likelihoods raises the uncertainty
to the $p$-th power, dramatically sharpening the posterior.

```python
# Calibration fix: use isotonic regression (requires ~1000+ samples)
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(
    GaussianNB(),
    method="isotonic",   # "sigmoid" for small datasets (<1000 samples)
    cv=5
)
calibrated_nb.fit(X_train, y_train)
calibrated_nb.predict_proba(X_test)   # now well-calibrated
```

**Diagnosis:** Plot calibration curve:
```python
from sklearn.calibration import CalibrationDisplay
CalibrationDisplay.from_estimator(GaussianNB(), X_test, y_test, n_bins=10)
```
A well-calibrated NB would be diagonal. Uncalibrated NB shows an S-curve bowed toward 0 and 1.

**Isotonic vs Sigmoid:** Use isotonic when >1000 calibration samples; sigmoid (Platt scaling)
for smaller calibration sets. Isotonic achieves lower Brier score but can overfit with few samples.

### 4. Feature Correlation Breaks the Independence Assumption Hard

When features are highly correlated (e.g., "loan" and "credit" in financial text, or correlated
numerical features), NB double-counts the evidence. This does NOT necessarily harm accuracy
(as Domingos & Pazzani showed) but DOES harm calibration.

**Detection:** Check pairwise correlations before using NB. If $|\text{corr}(x_i, x_j)| > 0.7$
for many pairs, consider GBM or LR instead.

**Mitigation options:**
- Feature selection to remove redundant features
- PCA before GaussianNB (defeats independence semantics but often helps)
- Use AODE or TAN variants instead of plain NB

### 5. GaussianNB: Non-Gaussian Features

GaussianNB assumes each feature follows a Gaussian within each class. Highly skewed features
(counts, incomes, durations) violate this.

```python
# Diagnostic: check for Gaussianity per class
for c in np.unique(y):
    from scipy.stats import shapiro
    stat, p = shapiro(X[y==c, feature_idx])
    print(f"Class {c}: Shapiro-Wilk p={p:.3f}")   # p<0.05 → non-Gaussian
```

**Fix:** Log-transform skewed features, or use a different distribution (BernoulliNB for binary,
MultinomialNB for counts). The `pomegranate` package allows per-feature distribution specification.

### 6. MultinomialNB with Negative Features (TF-IDF Sublinear Scaling)

MultinomialNB **requires non-negative features** (it's a count model). TF-IDF with `sublinear_tf=True` can still produce non-negative values, but certain preprocessing steps can introduce negatives.

```python
# GOOD: standard TF-IDF always non-negative
vec = TfidfVectorizer()
X_tfidf = vec.fit_transform(docs)   # all values ≥ 0 ✓

# BAD: centered/scaled features
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X_counts)   # negative values → error
MultinomialNB().fit(X_scaled, y)   # raises ValueError
```

Use `MinMaxScaler` instead of `StandardScaler` when you need to normalize before MultinomialNB.

### 7. partial_fit: Must Pass `classes` on First Call

```python
# WRONG: omitting classes on first partial_fit call
clf.partial_fit(X_chunk1, y_chunk1)   # raises error if y_chunk1 doesn't contain all classes

# CORRECT: pass all possible class labels on first call
all_classes = np.array([0, 1, 2])
clf.partial_fit(X_chunk1, y_chunk1, classes=all_classes)
clf.partial_fit(X_chunk2, y_chunk2)   # classes not needed on subsequent calls
```

### 8. CategoricalNB: Unseen Categories at Test Time

If test data contains a category value not seen in training, CategoricalNB will raise an error.
Fix with `min_categories` parameter:

```python
n_categories = [len(enc.categories_[i]) for i in range(X.shape[1])]
clf = CategoricalNB(min_categories=n_categories)
```

### 9. Text Pipeline: Vocabulary Mismatch in Streaming Settings

When using `partial_fit` for out-of-core text classification, the vocabulary must be fixed before
fitting (CountVectorizer does not support `partial_fit`; use `HashingVectorizer` instead):

```python
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# HashingVectorizer + MultinomialNB = truly online text classification
vec = HashingVectorizer(n_features=2**18, alternate_sign=False)
clf = MultinomialNB()

for X_batch, y_batch in stream_data():
    X_hashed = vec.transform(X_batch)
    clf.partial_fit(X_hashed, y_batch, classes=[0, 1])
```

Note: `alternate_sign=False` is critical — default `alternate_sign=True` produces negative values incompatible with MultinomialNB.

---

## Key Datasets for Examples

### 20 Newsgroups (text classification — primary NB showcase dataset)

```python
from sklearn.datasets import fetch_20newsgroups

# Full dataset (20 categories)
newsgroups_train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
newsgroups_test  = fetch_20newsgroups(subset="test",  remove=("headers", "footers", "quotes"))

# Subset for demo (4 categories)
categories = ["sci.space", "comp.graphics", "talk.religion.misc", "soc.religion.christian"]
train = fetch_20newsgroups(subset="train", categories=categories, remove=("headers",))
test  = fetch_20newsgroups(subset="test",  categories=categories, remove=("headers",))
```

18,846 documents, 20 categories. MultinomialNB + TF-IDF typically achieves ~85–88% accuracy on
full 20-way classification (without removing headers). Best standard NB benchmark.

### Iris (GaussianNB demonstration)

```python
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
# 4 continuous features, 3 classes, 150 samples — classic GaussianNB demo
```

### SMS Spam Collection (spam/ham classification)

Available via UCI ML Repository or directly:
```python
import pandas as pd
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
sms = pd.read_table(url, header=None, names=["label", "message"])
# BernoulliNB or MultinomialNB + CountVectorizer
```
~5,574 messages, ~87% ham / 13% spam (imbalanced → good case for ComplementNB).

### Wine Dataset (GaussianNB multi-class)

```python
from sklearn.datasets import load_wine
X, y = load_wine(return_X_y=True)
# 13 continuous features, 3 wine types, 178 samples
```

### Movie Reviews (BernoulliNB / NLTK demo)

```python
import nltk
nltk.download("movie_reviews")
from nltk.corpus import movie_reviews
# 2000 reviews, binary: pos/neg
```

---

## Curated Resources

### Original Papers
1. **Domingos & Pazzani (1997) — "On the Optimality of the Simple Bayesian Classifier"**
   URL: https://gwern.net/doc/ai/1997-domingos.pdf
   Why: The theoretical foundation — explains when NB is optimal despite violated assumptions.

2. **Rennie et al. (2003) — "Tackling the Poor Assumptions of Naive Bayes Text Classifiers"**
   URL: https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
   Why: CNB paper — essential for understanding ComplementNB.

3. **Friedman, Geiger & Goldszmidt (1997) — "Bayesian Network Classifiers"**
   URL: https://dang.cs.technion.ac.il/journal_papers/friedman1997Bayesian.pdf
   Why: The TAN paper — benchmark for NB variants in structured settings.

4. **Rish (2001) — "An Empirical Study of the Naive Bayes Classifier"**
   URL: https://faculty.cc.gatech.edu/~isbell/reading/papers/Rish.pdf
   Why: Systematic empirical analysis of when NB succeeds and fails.

5. **McCallum & Nigam (1998) — "A Comparison of Event Models for Naive Bayes Text Classification"**
   URL: https://www.cs.cmu.edu/~knigam/papers/multinomial-aaai98.pdf
   Why: The paper that distinguished MultinomialNB from BernoulliNB for text.

### Official Documentation
6. **sklearn Naive Bayes User Guide (1.5)**
   URL: https://scikit-learn.org/1.5/modules/naive_bayes.html
   Why: Verified formulas, parameters, and partial_fit behavior.

7. **sklearn GaussianNB API Reference (1.5)**
   URL: https://scikit-learn.org/1.5/modules/generated/sklearn.naive_bayes.GaussianNB.html

8. **sklearn MultinomialNB API Reference (1.5)**
   URL: https://scikit-learn.org/1.5/modules/generated/sklearn.naive_bayes.MultinomialNB.html

9. **sklearn Probability Calibration Guide**
   URL: https://scikit-learn.org/stable/modules/calibration.html
   Why: Explains exactly why NB is miscalibrated and how CalibratedClassifierCV fixes it.

### Tutorials and Explanations
10. **Paul Graham — "A Plan for Spam" (2002)**
    URL: https://paulgraham.com/spam.html
    Why: The essay that popularized Bayesian spam filtering; excellent historical context and motivation.

11. **VanderPlas — "In Depth: Naive Bayes Classification" (Python Data Science Handbook)**
    URL: https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html
    Why: Excellent intuitive treatment with code; decision boundary visualization.

12. **sklearn Text Feature Extraction + Grid Search Example (1.5)**
    URL: https://scikit-learn.org/1.5/auto_examples/model_selection/plot_grid_search_text_feature_extraction.html
    Why: Complete production-ready text pipeline with grid search.

13. **sklearn Calibration Plot Example**
    URL: https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration.html
    Why: Visual proof of NB miscalibration with Brier score comparison.

14. **NLTK Classify Howto**
    URL: https://www.nltk.org/howto/classify.html
    Why: Reference for NaiveBayesClassifier API.

15. **Wikipedia: Averaged One-Dependence Estimators**
    URL: https://en.wikipedia.org/wiki/Averaged_one-dependence_estimators
    Why: Good summary of AODE and semi-naive Bayes family.

---

## Hyperparameter Tuning Playbook

| Parameter | Variant | Typical Range | What It Controls | Tune With |
|-----------|---------|---------------|-----------------|-----------|
| `alpha` | MNB, BNB, CNB, CatNB | `[0.001, 5.0]` log-scale | Laplace/Lidstone smoothing strength | `GridSearchCV` with `np.logspace(-3, 1, 50)` |
| `var_smoothing` | GNB | `[1e-12, 0.1]` log-scale | Variance regularization | `np.logspace(-9, -1, 100)` |
| `binarize` | BNB | median of feature values | Threshold for continuous → binary | Tune or set to 0.0 for pre-binarized data |
| `fit_prior` | MNB, BNB, CatNB | `{True, False}` | Whether to use empirical class frequencies | Set `False` if classes severely imbalanced and you want uniform weighting |
| `norm` | CNB | `{True, False}` | Second normalization of weights | Set `True` if following the original Rennie et al. paper exactly |
| `min_categories` | CatNB | `int ≥ max_categories` | Prevents errors on unseen categories | Set to encoder's n_categories_per_feature |

### Text Pipeline Hyperparameters (interact with NB)

| Parameter | Location | Typical Range | Notes |
|-----------|----------|---------------|-------|
| `max_features` | `TfidfVectorizer` | `[5000, 100000]` | Vocabulary size cap; reduces noise |
| `ngram_range` | `TfidfVectorizer` | `(1,1)`, `(1,2)` | Bigrams improve accuracy but cost memory |
| `sublinear_tf` | `TfidfVectorizer` | `{True, False}` | Log-scaled TF reduces heavy hitter bias |
| `max_df` | `TfidfVectorizer` | `[0.5, 1.0]` | Remove high-frequency stopwords |
| `min_df` | `TfidfVectorizer` | `[1, 5, 10]` | Remove rare terms; reduces overfitting |

---

## Key Algorithmic Details the Writer Must Nail

### The Generative Model View

NB is a **generative** classifier: it models the joint $P(\mathbf{x}, y) = P(y) \cdot \prod_i P(x_i \mid y)$ and uses it to compute the posterior $P(y \mid \mathbf{x})$ via Bayes' theorem. Contrast with discriminative models (LR, SVM) that model $P(y \mid \mathbf{x})$ directly. Generative models can handle missing features naturally (just skip the missing $x_i$ term in the product). This is a rarely-mentioned strength worth covering.

### Computational Complexity

- **Training:** $\mathcal{O}(n \cdot p)$ — single pass over the data, just accumulating sufficient statistics (counts/means/variances). No iteration, no gradient, no matrix inversion.
- **Inference:** $\mathcal{O}(K \cdot p)$ — compute $K$ dot products of length $p$. Essentially free.
- **Memory:** $\mathcal{O}(K \cdot p)$ — stores one value per (class, feature) pair.
- Compare: LR requires $\mathcal{O}(n \cdot p \cdot \text{iterations})$ for training. NB wins decisively.

### Log-Sum-Exp Trick in Sklearn

sklearn uses log-domain computation throughout:
```python
# Internal predict_log_proba computation (conceptually):
joint_log_prob = feature_log_prob_ @ X.T + class_log_prior_[:, None]
# Normalize using logsumexp:
from scipy.special import logsumexp
log_proba = joint_log_prob - logsumexp(joint_log_prob, axis=0)
```

### Partial Fit for Out-of-Core Learning

All five sklearn NB variants support `partial_fit`. The sufficient statistics (counts, means,
variances) are updateable incrementally. For GaussianNB, the incremental mean and variance update
uses Welford's online algorithm internally.

**Performance tip:** Call `partial_fit` on the largest chunks that fit in RAM. Each `partial_fit`
call has fixed overhead proportional to $K \times p$, not the chunk size — so many tiny calls
are wasteful.

### MNB as a Linear Classifier

The decision rule for MultinomialNB (and BernoulliNB, CNB) is exactly linear in the feature
space. The model has:
- `coef_` = `feature_log_prob_` (shape `n_classes × n_features`) — the "weights"
- `intercept_` = `class_log_prior_` — the "biases"

Prediction: argmax of `X @ coef_.T + intercept_`. This is the same as any linear classifier.
This means NB is a **log-linear model** — equivalent to logistic regression with specific
parameter constraints derived from the generative model.

### Semi-Naive Bayes Taxonomy

The key insight for the section on variants:
- **NB:** All features independent given $y$ (zero edges in feature graph)
- **TAN:** Each feature has at most 1 feature parent (tree structure)
- **AODE:** Average over all $p$ one-parent models
- **HNB:** Hidden parent aggregates cross-feature dependencies
- **Bayesian Network Classifier:** General DAG among features (NB is a special case with no feature edges)

The tradeoff is: expressiveness ↑ → computation ↑ → variance ↓ but risk of overfitting ↑.

---

## Uncertainties — Items the Writer Must Double-Check

1. **CategoricalNB `partial_fit` support:** The sklearn 1.5 docs for CategoricalNB list `partial_fit` as a method, but the main Naive Bayes user guide mentions `partial_fit` for MultinomialNB, BernoulliNB, and GaussianNB only — it's unclear whether CategoricalNB's `partial_fit` is fully tested/recommended. **Verify in sklearn 1.5.x source or changelog.**

2. **`predict_joint_log_proba` availability:** This method appears in the latest sklearn docs but may not be in all sklearn 1.5.x patch versions. **Check exact version it was added (likely 1.2).**

3. **ComplementNB `norm=True` in original paper:** The sklearn docs explicitly state that `norm=False` (the default) mirrors Mahout/Weka but does NOT match the full Rennie et al. 2003 algorithm. The writer should verify whether the original paper's version (`norm=True`) consistently outperforms `norm=False` on real benchmarks — the sklearn docs are ambiguous about this.

4. **`force_alpha` history:** `force_alpha` was added in sklearn 1.2, and its default changed from `False` to `True` in 1.4. Content mentioning sklearn <1.4 behavior should note this version sensitivity. Verify exact version numbers in the changelog.

5. **NLTK NaiveBayesClassifier API stability:** NLTK's NaiveBayesClassifier API has been stable since ~3.x but the underlying internals changed between 3.5 and 3.8. The writer should test the snippet against current NLTK (3.8+).

6. **TextBlob NaiveBayesAnalyzer training corpus:** TextBlob's built-in NaiveBayesAnalyzer is trained on the NLTK movie reviews corpus. This corpus requires `nltk.download("movie_reviews")`. The writer should confirm this still works in mid-2026 TextBlob (0.19.x).

7. **pgmpy TAN API:** The pgmpy `TreeSearch` with `estimator_type="tan"` API shown above should be verified against current pgmpy docs — the API has been refactored in recent pgmpy releases.

8. **GaussianNB linear boundary equation:** The derivation that GaussianNB gives a linear boundary when per-class variances are equal is theoretically standard, but confirming the exact `w` vector formula vs. the sklearn implementation's `theta_` and `var_` attributes should be done with a test case. The boundary is linear only when `var_` values are equal across classes per feature.

9. **20 Newsgroups accuracy benchmarks:** The commonly cited ~85–88% MultinomialNB accuracy depends heavily on preprocessing (removing headers vs not, subsets vs full set). The writer should run actual benchmarks rather than citing specific numbers from memory.

10. **Brier score values from calibration example:** The specific Brier scores (0.104 uncalibrated, 0.084 isotonic) are from a synthetic dataset in the sklearn docs and should not be presented as universal benchmarks for NB calibration quality.
