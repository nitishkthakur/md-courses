# Naive Bayes: The Complete Masterclass

> **Why this algorithm matters:** In 2002, programmer and essayist Paul Graham published "A Plan
> for Spam" — a short essay describing how a simple probabilistic model, built on 200-year-old
> mathematics, could filter spam email better than any rule-based system then in production. Within
> months, Bayesian spam filters were deployed by every major email provider. The model was Naive
> Bayes. Today, Naive Bayes remains the fastest text classifier in widespread production use, the
> go-to baseline for any NLP task, and a genuinely illuminating algorithm to understand deeply —
> because its mixture of theoretical naivety and empirical resilience tells you something profound
> about what it takes for a classifier to be accurate.

---

## §0 — Python Ecosystem & Package Guide

Before the mathematics, let's orient ourselves in the ecosystem. Naive Bayes is one of the few
algorithms where the choice of *which variant* matters more than the choice of *which package* —
sklearn dominates for production use, but there are meaningful alternatives for NLP workflows,
online learning, and GPU-accelerated scenarios. Here is the full landscape.

### Complete Package Table

| Package | Import | Version (mid-2026) | Variants | License |
|---|---|---|---|---|
| `scikit-learn` | `from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB` | ~1.5.x | All five: GNB, MNB, BNB, CNB, CatNB; full `partial_fit` support | BSD-3 |
| `nltk` | `from nltk.classify import NaiveBayesClassifier` | ~3.9.x | BernoulliNB-style on feature dictionaries; built-in informative-feature inspection | Apache-2 |
| `textblob` | `from textblob.classifiers import NaiveBayesClassifier` | ~0.19.x | Wrapper over NLTK; pre-trained sentiment analyzer (movie reviews corpus) | MIT |
| `river` | `from river.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB` | ~0.21.x | True single-sample online learning; no batch required | BSD-3 |
| `pomegranate` | `from pomegranate.naive_bayes import NaiveBayes` | ~1.0.x | PyTorch backend; GPU support; mixed distributions per feature (Gaussian + Poisson + Bernoulli) | MIT |
| `h2o` | `from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator` | ~3.46.x | Distributed NB on large clusters; handles missing values natively | Apache-2 |
| `pgmpy` | `from pgmpy.estimators import TreeSearch` | ~0.1.26 | Tree-Augmented Naive Bayes (TAN) via Chow-Liu structure learning | MIT |
| `python-weka-wrapper3` | `weka.classifiers.Classifier("weka.classifiers.bayes.NaiveBayes")` | ~0.2.x | Java reference implementation of NB and TAN via WEKA | GPL-3 |

### Top Picks — Recommendation Table

| Package | Best for | Style | Notes |
|---|---|---|---|
| **`sklearn.MultinomialNB`** | Text classification, NLP pipelines | Batch + `partial_fit` | **Default choice for text** — integrates with `TfidfVectorizer`, `Pipeline`, `GridSearchCV` |
| **`sklearn.GaussianNB`** | Continuous numerical features, fast baseline | Batch + `partial_fit` | **Default choice for tabular continuous data** — nearly always the right NB starting point |
| **`sklearn.ComplementNB`** | Imbalanced text classification | Batch + `partial_fit` | Prefer over MultinomialNB whenever classes are unbalanced (usually they are) |
| **`sklearn.CategoricalNB`** | Purely nominal categorical features | Batch + `partial_fit` | Use with `OrdinalEncoder`; do not use for ordinal features |
| **`river.MultinomialNB`** | Streaming/real-time text classification | True online (one sample at a time) | When data arrives faster than you can batch |
| **`pomegranate.NaiveBayes`** | Mixed feature types or GPU acceleration | Batch | When features are not all the same type; significantly faster than sklearn on GPU |

### The Same Fit Across Top Packages

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Shared text data setup ────────────────────────────────────────────────────
categories = ["sci.space", "comp.graphics", "talk.religion.misc", "soc.religion.christian"]
train_data = fetch_20newsgroups(
    subset="train", categories=categories, remove=("headers", "footers", "quotes")
)
test_data = fetch_20newsgroups(
    subset="test",  categories=categories, remove=("headers", "footers", "quotes")
)

vec = TfidfVectorizer(max_features=20_000, sublinear_tf=True)
X_train = vec.fit_transform(train_data.data)
X_test  = vec.transform(test_data.data)
y_train, y_test = train_data.target, test_data.target

# ── Option 1: MultinomialNB (sklearn) — the workhorse ────────────────────────
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB(alpha=0.1)
mnb.fit(X_train, y_train)
print(f"MultinomialNB accuracy: {accuracy_score(y_test, mnb.predict(X_test)):.4f}")

# ── Option 2: ComplementNB (sklearn) — better under imbalance ────────────────
from sklearn.naive_bayes import ComplementNB

cnb = ComplementNB(alpha=0.1, norm=True)
cnb.fit(X_train, y_train)
print(f"ComplementNB accuracy:  {accuracy_score(y_test, cnb.predict(X_test)):.4f}")

# ── Option 3: BernoulliNB (sklearn) — presence/absence model ─────────────────
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

# BernoulliNB works on binary occurrence: use CountVectorizer, then binarize
count_vec = CountVectorizer(max_features=20_000, binary=True)
X_train_b = count_vec.fit_transform(train_data.data)
X_test_b  = count_vec.transform(test_data.data)

bnb = BernoulliNB(alpha=0.1)
bnb.fit(X_train_b, y_train)
print(f"BernoulliNB accuracy:   {accuracy_score(y_test_b := y_test, bnb.predict(X_test_b)):.4f}")

# ── Option 4: river (true online learning) ────────────────────────────────────
# river operates on feature dicts, one sample at a time
from river.naive_bayes import MultinomialNB as RiverMNB
from river.feature_extraction import BagOfWords

river_pipeline = BagOfWords() | RiverMNB(alpha=0.1)
for doc, label in zip(train_data.data, y_train):
    river_pipeline.learn_one(doc, label)

# Predict on test (stream simulation)
river_preds = [river_pipeline.predict_one(doc) for doc in test_data.data]
print(f"river.MultinomialNB:    {accuracy_score(y_test, river_preds):.4f}")

# ── Continuous features: GaussianNB on Iris ──────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

Xi, yi = load_iris(return_X_y=True)
from sklearn.model_selection import train_test_split as tts
Xi_tr, Xi_te, yi_tr, yi_te = tts(Xi, yi, test_size=0.2, random_state=42, stratify=yi)

gnb = GaussianNB()
gnb.fit(Xi_tr, yi_tr)
print(f"GaussianNB on iris:     {gnb.score(Xi_te, yi_te):.4f}")
```

> 🔧 **In practice:** For 95% of text classification problems, your starting point is
> `ComplementNB(alpha=0.1)` with `TfidfVectorizer(sublinear_tf=True)`. For continuous tabular
> features, start with `GaussianNB()` — zero hyperparameters to tune, fits in milliseconds,
> and gives you a calibrated performance floor before trying anything heavier.

> 📊 **Benchmark:** On the full 20 Newsgroups dataset (all 20 categories, headers/footers
> removed), a tuned `ComplementNB(alpha=0.05)` + `TfidfVectorizer(max_features=50000,
> sublinear_tf=True, ngram_range=(1,2))` pipeline achieves approximately 82–86% test accuracy.
> Logistic regression on the same features achieves 87–90%. This ~4% gap represents the
> practical cost of the independence assumption on this dataset — a reasonable price for the
> training speed advantage. Note: exact numbers depend on preprocessing details (vocabulary
> size, presence of headers, stop word removal). Always run your own benchmark on your data
> rather than citing these numbers directly.

> ⚠️ **Pitfall:** `river`'s `BagOfWords` tokenizer is not the same as sklearn's
> `TfidfVectorizer`. Direct accuracy comparisons require the same preprocessing. The numbers
> above are directional, not perfectly comparable.

---

## §1 — The Origin: The Paper That Started It All

### Thomas Bayes and the Posthumous Revolution (1763)

The story starts in an English country house sometime around 1755. A Presbyterian minister named
Thomas Bayes — whose only mathematical publication during his lifetime was a defense of Newton's
method of fluxions — wrote an essay he never submitted for publication. After his death in 1761,
his friend Richard Price found the manuscript, edited it, and sent it to the Royal Society. It
was published in the *Philosophical Transactions of the Royal Society of London* in 1763 under
the title "An Essay towards Solving a Problem in the Doctrine of Chances."

> 📜 **Origin/Citation:**
> **Bayes, T. (1763).** "An Essay towards Solving a Problem in the Doctrine of Chances."
> *Philosophical Transactions of the Royal Society of London, 53*, 370–418.
> URL: https://www.eis.mdx.ac.uk/staffpages/rvb/teaching/BUS3010/bayes1763.pdf

The problem Bayes was solving: given that an event has occurred a certain number of times in a
series of trials, what is the probability that the underlying parameter (the "chance") lies
within any given interval? In modern language, he was deriving the posterior distribution of a
binomial parameter — a first derivation of what we now call Bayesian inference. The result,
generalized by Laplace in 1812, became the theorem that underpins all of modern probabilistic
reasoning:

$$P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)}$$

Hypothesis $H$, evidence $E$: given that we observed the evidence, how probable is the
hypothesis? Two and a half centuries later, this is still the engine of Naive Bayes.

### From Probability Theory to Machine Learning (1950s–1970s)

The leap from Bayes' theorem to a classifier required connecting it to the problem of pattern
recognition. The earliest documented probabilistic text classifier appeared in:

> 📜 **Origin/Citation:**
> **Maron, M. E. (1961).** "Automatic Indexing: An Experimental Inquiry."
> *Journal of the ACM, 8*(3), 404–417. DOI: 10.1145/321075.321084

Maron's system indexed scientific abstracts by assigning each document a probability of
belonging to each subject category, using the word frequencies in the document and the
category-conditional word frequencies learned from training examples. The independence assumption
was made implicitly — the term "naive" had not yet been coined — and the core formula was
recognizably what we use today.

The formal generative model view was crystallized in the pattern recognition bible:

> 📜 **Origin/Citation:**
> **Duda, R. O., & Hart, P. E. (1973).** *Pattern Classification and Scene Analysis.* Wiley.

Duda and Hart placed the classifier firmly in the Bayesian decision theory framework, showed how
different distributional assumptions (Gaussian, multinomial) produced different variants, and
derived the linear and quadratic decision boundaries that fall out of the Gaussian case.

### Why Does It Work Despite Being Wrong? (Domingos & Pazzani, 1997)

The mystery that bothered researchers for decades: the naive independence assumption is almost
always false (words co-occur non-randomly, tabular features correlate), yet Naive Bayes achieves
competitive accuracy against much more sophisticated models. In 1997, Pedro Domingos and Michael
Pazzani provided the definitive theoretical answer:

> 📜 **Origin/Citation:**
> **Domingos, P., & Pazzani, M. (1997).** "On the Optimality of the Simple Bayesian Classifier
> under Zero-One Loss." *Machine Learning, 29*(2–3), 103–130.
> URL: https://gwern.net/doc/ai/1997-domingos.pdf

Their key result: Naive Bayes achieves optimal zero-one loss (i.e., correct classification, not
necessarily correct probabilities) if and only if the correct posterior ranking is preserved. The
NB estimator does not need to correctly estimate $P(y \mid \mathbf{x})$ — it only needs to
correctly *rank* the classes. Feature correlations that are symmetric across classes, or a
dominant prior probability for one class, both tend to preserve the correct ranking despite
incorrect probability magnitudes.

This is a beautiful result. It explains exactly when NB succeeds (skewed priors, symmetric
correlations, short texts) and when it fails (fine-grained classes with similar feature
distributions, highly asymmetric correlations). Keep this theorem in mind throughout the chapter.

### The Text Classification Renaissance (1990s)

Through the 1980s, NB was primarily a topic in the pattern recognition literature — applied
to medical diagnosis, fault detection, and sensor fusion. The explosion in digitized text
during the early 1990s brought a new application domain that turned out to be nearly perfectly
suited to NB's assumptions.

Two influential papers from this era established the text classification benchmark:

> 📜 **Origin/Citation:**
> **Lewis, D. D. (1992).** "Naive (Bayes) at Forty: The Independence Assumption in Information
> Retrieval." *ECML 1998 Invited Paper.*

> 📜 **Origin/Citation:**
> **Joachims, T. (1997).** "A Probabilistic Analysis of the Rocchio Algorithm with TFIDF for
> Text Categorization." *ICML 1997.*

The recurring finding was striking: despite its theoretical naivety, NB consistently ranked
among the top-performing classifiers on text categorization benchmarks when vocabulary features
were used. This puzzled researchers enough to motivate the Domingos & Pazzani (1997) theoretical
work that finally explained the phenomenon.

The reason text turns out to be a near-ideal domain for NB:
1. **Skewed priors:** In real document corpora, class frequencies are almost always highly
   skewed (most documents belong to a few dominant categories). Dominant priors help NB maintain
   correct class rankings.
2. **Many weak, approximately symmetric correlations:** Word co-occurrences create correlations,
   but they tend to be approximately symmetric across classes — the same pairs co-occur in both
   class-relevant and class-irrelevant contexts.
3. **High dimensionality per sample:** With 50,000 vocabulary features and only a few hundred
   words per document, most features are zero. NB's sparse computation (only present features
   contribute) is efficient and avoids the noise amplification that affects dense classifiers.

These conditions are rarely all true simultaneously for tabular data, which is why NB performs
so much more reliably on text than on general numerical datasets.

### The Spam Filtering Moment (Graham, 2002)

Theoretical optimality is one thing; cultural impact is another. In 2002, Paul Graham published
an essay that brought Naive Bayes out of academic papers and into every email inbox on the
planet:

> 📜 **Origin/Citation:**
> **Graham, P. (2002).** "A Plan for Spam."
> URL: https://paulgraham.com/spam.html

Graham's implementation was a Bayesian token frequency comparison between spam and ham corpora.
His key insight was to use the *most informative* tokens (those with extreme spam or ham
probability) rather than all tokens, and to combine them using a Bayesian update. The approach
worked strikingly well — Graham reported catching 99.5% of his spam with negligible false
positives. More importantly, the essay was widely read by engineers and triggered a wave of
Bayesian spam filter deployments across the industry.

Graham was not using sklearn's MultinomialNB exactly, but the mathematical foundations were
identical. His essay is worth reading not for the implementation details (which are outdated) but
for the clarity with which it motivates the probabilistic approach over rule-based methods.

What made Graham's approach work so well in practice was not just the math — it was the
observation that spam and ham occupy fundamentally different regions of token-probability space.
Spam contains very distinctive vocabulary (financial offers, pharmaceutical terms, urgency
signals) that is highly unlikely in legitimate email. The prior probability of spam was initially
low (5–15% of email in 2002), but NB correctly amplified the likelihood ratio for distinctly
spammy tokens to overcome the prior. This is Domingos & Pazzani's insight in action: the
priors were skewed enough and the vocabularies distinctive enough that the class ranking was
robustly preserved even with violated independence.

By 2004, Bayesian spam filters were integrated into SpamAssassin, Mozilla Thunderbird, and
Apple Mail. By 2006, they were being trained on billions of emails by major providers. NB had
moved from a textbook algorithm to critical infrastructure within two years of Graham's essay —
a remarkably short path from theory to deployment that few algorithms have matched.

---

## §2 — The Algorithm, Deeply Explained

### Bayes' Theorem for Classification

We want to classify a sample described by feature vector $\mathbf{x} = (x_1, x_2, \ldots, x_p)$
into one of $K$ classes $\{c_1, c_2, \ldots, c_K\}$. The Bayesian decision rule picks the class
that maximizes the posterior probability:

$$\hat{y} = \arg\max_{c \in \mathcal{C}} P(c \mid \mathbf{x})$$

Applying Bayes' theorem:

$$P(c \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid c) \cdot P(c)}{P(\mathbf{x})}$$

The denominator $P(\mathbf{x})$ is identical for all classes — it is simply a normalizing
constant. So the decision rule becomes:

$$\hat{y} = \arg\max_{c} \; P(c) \cdot P(\mathbf{x} \mid c)$$

The three components here have names:
- $P(c)$: the **class prior** — how often does class $c$ appear in our data?
- $P(\mathbf{x} \mid c)$: the **class-conditional likelihood** — if this is a class-$c$ example, how probable is this feature vector?
- $P(c \mid \mathbf{x})$: the **posterior** — what we actually want to maximize

The prior is easy: count the frequency of each class in the training set. The likelihood is
the hard part.

### The Exponential Scaling Problem

$P(\mathbf{x} \mid c)$ is a joint distribution over $p$ features. If each feature is binary, the
joint has $2^p$ possible values — you need $K \cdot 2^p$ parameters. With $p = 50$ features and
$K = 2$ classes, that is $2^{51} \approx 2 \times 10^{15}$ parameters to estimate from a finite
training set. This is completely intractable.

### The Naive Independence Assumption

The "naive" move is to assume that all features are **conditionally independent given the class**:

$$P(\mathbf{x} \mid c) = P(x_1, x_2, \ldots, x_p \mid c) \stackrel{\text{naive}}{=} \prod_{i=1}^{p} P(x_i \mid c)$$

This assumption is almost certainly false in practice — "New" and "York" are not conditionally
independent even given the topic "cities" — but it collapses the parameter count from $\mathcal{O}(K \cdot 2^p)$ to $\mathcal{O}(K \cdot p)$, making estimation trivial.

The full Naive Bayes decision rule becomes:

$$\hat{y} = \arg\max_{c} \left[ P(c) \cdot \prod_{i=1}^{p} P(x_i \mid c) \right]$$

### Working in Log Space

Multiplying $p$ probabilities causes numerical underflow to machine-precision zero when $p$ is
large (e.g., a vocabulary of 50,000 features). We convert to log space, turning products into
sums:

$$\hat{y} = \arg\max_{c} \left[ \log P(c) + \sum_{i=1}^{p} \log P(x_i \mid c) \right]$$

This is exactly what sklearn computes. The fitted attributes `class_log_prior_` (shape
`(K,)`) and `feature_log_prob_` (shape `(K, p)`) store these log quantities. Prediction is
a matrix multiplication plus a bias term — a linear classifier in log-probability space.

> 💡 **Intuition:** Naive Bayes as a linear classifier: The decision rule `argmax(X @ coef_.T + intercept_)` is identical in structure to logistic regression. The difference is *how the weights are estimated* — NB derives them from counting, LR finds them by gradient descent on the cross-entropy loss. This is why sklearn exposes `coef_` and `intercept_` as aliases for NB estimators: they are formally linear models.

### The Log-Sum-Exp Trick and Numerical Stability

Working in log space solves the underflow problem, but computing probabilities from log-scores
introduces a new challenge: the normalizing constant $P(\mathbf{x}) = \sum_c P(c) P(\mathbf{x} \mid c)$.
In log space, this requires computing $\log \sum_c \exp(s_c)$ where $s_c$ is the unnormalized
log-posterior for class $c$.

If $s_c$ is very large or very negative (as it often is with high-dimensional features), direct
exponentiation will overflow or underflow. The **log-sum-exp trick** stabilizes this:

$$\log \sum_c \exp(s_c) = \max_c(s_c) + \log \sum_c \exp(s_c - \max_c(s_c))$$

By subtracting the maximum score before exponentiating, we ensure the largest term is $\exp(0) = 1$
and all other terms are $\leq 1$. sklearn uses `scipy.special.logsumexp` internally. Here is the
exact conceptual computation:

```python
import numpy as np
from scipy.special import logsumexp

# For a single test sample x:
# feature_log_prob_: (K, p) array
# class_log_prior_:  (K,) array

# Unnormalized log-posteriors (one per class)
joint_log_probs = feature_log_prob_ @ x + class_log_prior_   # shape (K,)

# Normalize using log-sum-exp
log_evidence = logsumexp(joint_log_probs)
log_posteriors = joint_log_probs - log_evidence   # now log-probabilities

# Predict class
predicted_class = np.argmax(log_posteriors)
posteriors = np.exp(log_posteriors)   # actual probabilities
```

For a batch of $n$ test samples, this becomes a matrix multiplication:
`joint = feature_log_prob_ @ X.T + class_log_prior_[:, None]` — sklearn uses exactly this.

---

### Naive Bayes as a Log-Linear Model: The Exponential Family Connection

> 🔬 **Deep dive:** This section explores the formal connection between NB and logistic regression
> that reveals why they often achieve similar accuracy with very different training procedures.

Multinomial NB's decision rule is:

$$\hat{c} = \arg\max_c \left[ \log P(c) + \sum_i x_i \log \hat{\theta}_{ci} \right]$$

This is a **linear function** of $\mathbf{x}$: $\hat{c} = \arg\max_c (\mathbf{w}_c^T \mathbf{x} + b_c)$
where $w_{ci} = \log \hat{\theta}_{ci}$ and $b_c = \log P(c)$.

Logistic regression for $K$ classes has the exact same structure:

$$\hat{c} = \arg\max_c (\mathbf{w}_c^T \mathbf{x} + b_c)$$

The only difference is *how* $\mathbf{w}_c$ is estimated:
- **NB:** $w_{ci} = \log \hat{\theta}_{ci}$ — direct MLE from counts, with specific constraints
  (the $\hat{\theta}_{ci}$ must form a valid probability distribution)
- **LR:** $\mathbf{w}_c$ maximizes the conditional log-likelihood $\log P(y \mid \mathbf{x})$
  via gradient descent, with no distributional constraints

This connection was formalized by Ng & Jordan (2002) in a landmark paper:

> 📜 **Origin/Citation:**
> **Ng, A. Y., & Jordan, M. I. (2002).** "On Discriminative vs. Generative Classifiers: A
> comparison of logistic regression and naive Bayes." *NIPS 2002.*
> URL: https://papers.nips.cc/paper/2002/file/aaebdb8bb6b0e73f6c3c54a0ab0c6415-Paper.pdf

Their key theoretical findings:
1. NB and LR have the same decision boundary *in the limit of infinite data* when the NB
   generative model is correctly specified.
2. NB converges to its (possibly suboptimal) solution with $\mathcal{O}(\log p)$ samples.
   LR converges to its (optimal) solution with $\mathcal{O}(p)$ samples.
3. Therefore, NB asymptotically underperforms LR — but for small training sets ($n \ll p$),
   NB's faster convergence can make it more accurate in practice.

> 💡 **Intuition:** NB is a biased-but-low-variance estimator. LR is unbiased-but-high-variance.
> With few samples, bias is acceptable and variance kills you — NB wins. With many samples, bias
> compounds and variance shrinks — LR wins. This crossover typically occurs somewhere around
> $n \approx 30p$ to $100p$.

This explains the empirical pattern practitioners observe: on small training sets, NB is
surprisingly competitive with or better than LR. On large datasets with enough data to fully
specify the LR model, LR consistently wins.

**The NB-LR weight constraints:** NB's generative model imposes specific constraints on the
weight structure. For MultinomialNB, $w_{ci} = \log \hat{\theta}_{ci}$ where the
$\hat{\theta}_{ci}$ form a probability simplex. These constraints are exactly right when the
generative model is correct, but constrain the model when the independence assumption is violated.
LR relaxes these constraints, allowing any weight vector that fits the data — at the cost of
requiring more data to estimate reliably.

---

### Training: Sufficient Statistics

NB training requires only one pass over the data, accumulating sufficient statistics:
- **GaussianNB:** class-conditional means $\mu_{ci}$ and variances $\sigma_{ci}^2$
- **MultinomialNB / BernoulliNB / ComplementNB:** class-conditional feature counts $N_{ci}$
- **CategoricalNB:** class-conditional category counts $N_{tic}$

There is no optimization objective, no gradient computation, no iteration. This makes NB
uniquely amenable to online (incremental) learning via `partial_fit`.

**Computational complexity:**
- Training: $\mathcal{O}(n \cdot p)$ — single pass, accumulating counts
- Inference: $\mathcal{O}(K \cdot p)$ — $K$ dot products, essentially free
- Memory: $\mathcal{O}(K \cdot p)$ — one parameter per (class, feature) pair

Compare to logistic regression: $\mathcal{O}(n \cdot p \cdot T)$ training ($T$ = iterations).
For large $n$, NB wins decisively in wall-clock time.

**Why are sufficient statistics sufficient?** For each NB variant, the sufficient statistics
fully characterize the maximum likelihood estimates. No additional information from the data
is needed. For MultinomialNB, the sufficient statistics are just two arrays: the total count
of each feature in each class ($N_{ci}$), and the total token count per class ($N_c$).

For GaussianNB, the online updates use Welford's algorithm to compute running mean and variance
without storing all data points:

$$\mu_n = \mu_{n-1} + \frac{x_n - \mu_{n-1}}{n}, \qquad
M_n = M_{n-1} + (x_n - \mu_{n-1})(x_n - \mu_n), \qquad
\sigma_n^2 = M_n / n$$

This is $\mathcal{O}(1)$ per new sample — exact and numerically stable.

### The Four Likelihoods (The Core Design Choice)

The choice of likelihood model for $P(x_i \mid c)$ is the fundamental design decision in NB.
Different choices produce the four main sklearn variants:

**Gaussian:** $P(x_i \mid c) = \mathcal{N}(x_i; \mu_{ci}, \sigma_{ci}^2)$
For continuous real-valued features assumed to be Gaussian within each class.

**Multinomial:** $P(\mathbf{x} \mid c) \propto \prod_i \theta_{ci}^{x_i}$
For non-negative count features (word counts, term frequencies).

**Bernoulli:** $P(\mathbf{x} \mid c) = \prod_i \theta_{ci}^{x_i}(1-\theta_{ci})^{1-x_i}$
For binary presence/absence features. Crucially, this explicitly penalizes absent features.

**Categorical:** $P(x_i = t \mid c) = \theta_{tic}$ for nominal category $t$
For discrete nominal features with $n_i$ possible values.

We will derive each in detail in §3.

### A Full Derivation: MNB from Scratch

Let's derive the MultinomialNB decision rule step by step to make everything concrete. Suppose
we are classifying a text document $d$ into one of $K$ topic categories. We represent $d$ as a
bag-of-words vector $\mathbf{x} = (x_1, x_2, \ldots, x_p)$ where $x_i$ is the count of word
$i$ in the document.

**Step 1: Model the class-conditional distribution**

For class $c$, we model the likelihood of generating a document with word counts $\mathbf{x}$
as a multinomial distribution. Think of it as: the document is generated by drawing $N = \sum_i x_i$
words independently from a vocabulary, where word $i$ is drawn with probability $\theta_{ci}$:

$$P(\mathbf{x} \mid c) = \frac{N!}{\prod_i x_i!} \prod_{i=1}^{p} \theta_{ci}^{x_i}$$

The multinomial coefficient $\frac{N!}{\prod_i x_i!}$ is identical for all classes (same document,
same counts) so it cancels in the argmax. What remains is the kernel:

$$P(\mathbf{x} \mid c) \propto \prod_{i=1}^{p} \theta_{ci}^{x_i}$$

**Step 2: Estimate $\theta_{ci}$ by MLE with Laplace smoothing**

Maximum likelihood: $\hat{\theta}_{ci}^{\text{MLE}} = N_{ci} / N_c$ where $N_{ci}$ = total count
of word $i$ across all class-$c$ training documents, $N_c = \sum_i N_{ci}$ = total tokens in
class $c$.

Without smoothing, if word $i$ never appears in any class-$c$ document, $\hat{\theta}_{ci} = 0$,
and the product becomes 0 for any document containing word $i$. This destroys the posterior.

Laplace smoothing adds pseudo-counts $\alpha$ to every word count:

$$\hat{\theta}_{ci} = \frac{N_{ci} + \alpha}{N_c + \alpha \cdot p}$$

The denominator ensures $\sum_i \hat{\theta}_{ci} = 1$ — the smoothed parameters still form a
valid probability distribution.

**Step 3: Write the log-posterior**

$$\log P(c \mid \mathbf{x}) \propto \log P(c) + \sum_{i=1}^{p} x_i \log \hat{\theta}_{ci}$$

Notice: only words present in the document ($x_i > 0$) contribute. Zero-count words contribute
$0 \cdot \log \hat{\theta}_{ci} = 0$. This is why MNB is efficient on sparse text vectors.

**Step 4: Make the prediction**

$$\hat{c} = \arg\max_{c \in \{1, \ldots, K\}} \left[ \underbrace{\log \hat{P}(c)}_{\text{class\_log\_prior\_}[c]} + \underbrace{\mathbf{x}^T \cdot \log \hat{\boldsymbol{\theta}}_c}_{\text{X @ feature\_log\_prob\_[c]}} \right]$$

That's it. Two matrix operations and an argmax.

> 📊 **Benchmark:** On the 20 Newsgroups dataset (18,846 documents, 20 classes), a well-tuned
> MultinomialNB + TF-IDF pipeline (with bigrams, sublinear_tf=True, alpha≈0.05–0.1) achieves
> approximately 83–86% test accuracy when headers/footers are removed. With headers present
> (which contain strong category signals like newsgroup names), accuracy can reach 95%+. Always
> verify which preprocessing your benchmark used.

---

### What Data Assumptions Does NB Make?

Every algorithm makes assumptions about the data-generating process. Here are NB's, stated
precisely:

**Assumption 1: Conditional independence.** Given the class label $y$, all features $x_1, \ldots, x_p$
are mutually independent. This is almost never true in practice but is the core computational
shortcut.

**Assumption 2 (GaussianNB): Gaussian marginals.** Within each class, each feature follows a
univariate Gaussian distribution. This rules out multimodal features, discrete features, and
heavy-tailed distributions.

**Assumption 3 (MultinomialNB): Multinomial generation.** Documents are generated by sampling
tokens i.i.d. from a class-specific multinomial distribution. This ignores word order, sentence
structure, and all linguistic context. The famous example: "dog bites man" and "man bites dog"
have identical representations.

**Assumption 4: Training distribution = test distribution.** Like all supervised methods, NB
assumes the feature distribution is the same at test time as at training time. Concept drift
(topics shift, spam tactics evolve, product vocabulary changes) will degrade performance over
time, requiring model updates via `partial_fit`.

Understanding when these assumptions are *approximately* satisfied is the key skill in deciding
whether to use NB. The first assumption is robustly violated (always) but rarely fatal. The
second is more fragile. The third is acceptable for topic classification but fatal for tasks
requiring word-order understanding.

---

### The Generative Model View

NB is a **generative classifier**: it models the full joint distribution $P(\mathbf{x}, y)$,
and uses it to derive the posterior by Bayes' theorem. This is in contrast to **discriminative
classifiers** like logistic regression, which model $P(y \mid \mathbf{x})$ directly and never
estimate the data distribution.

The generative nature has a practical consequence that is rarely mentioned: NB handles missing
features gracefully. If feature $x_j$ is missing for a test sample, simply remove its term from
the log-sum. The prediction still makes sense — you are just using the remaining evidence. A
discriminative model like logistic regression has no principled way to handle missing features
at inference time without imputation.

> ⚠️ **Pitfall:** NB produces severely miscalibrated probabilities. Because it multiplies $p$
> likelihoods, the posterior gets pushed toward 0 or 1 — the model is pathologically
> overconfident. If you need reliable probability estimates (for risk scoring, decision thresholds,
> or downstream models), always wrap NB with `CalibratedClassifierCV`. We cover this in detail
> in §7.

---

## §3 — The Full Evolution: From Gaussian NB to TAN

The five sklearn variants are not arbitrary choices — each one solved a specific limitation of
the one before. Here is the complete lineage, with the exact algorithmic change each introduced.

### 3.1 Gaussian Naive Bayes (GNB) — Duda & Hart, 1973

**The problem solved:** How do you apply NB to continuous real-valued features? The original NB
formulation assumed discrete features (present/absent, count). Real data (measurements, sensor
readings, physiological variables) is continuous.

**The solution:** Assume each feature $x_i$ follows a univariate Gaussian distribution within
each class:

$$P(x_i \mid y = c) = \frac{1}{\sqrt{2\pi\sigma_{ci}^2}} \exp\left(-\frac{(x_i - \mu_{ci})^2}{2\sigma_{ci}^2}\right)$$

Parameters are estimated by maximum likelihood — simply compute the sample mean and variance of
each feature within each class:

$$\hat{\mu}_{ci} = \frac{1}{n_c} \sum_{j:\, y_j = c} x_{ij}, \qquad
\hat{\sigma}_{ci}^2 = \frac{1}{n_c} \sum_{j:\, y_j = c} (x_{ij} - \hat{\mu}_{ci})^2$$

**The decision boundary geometry:** Taking the log posterior ratio for two classes and simplifying:

$$\log \frac{P(c_1 \mid \mathbf{x})}{P(c_2 \mid \mathbf{x})} = \log \frac{P(c_1)}{P(c_2)} - \sum_i \frac{(x_i - \mu_{c_1 i})^2}{2\sigma_{c_1 i}^2} + \sum_i \frac{(x_i - \mu_{c_2 i})^2}{2\sigma_{c_2 i}^2}$$

When per-feature variances are equal across classes ($\sigma_{c_1 i}^2 = \sigma_{c_2 i}^2$), the
$x_i^2$ terms cancel and the boundary becomes **linear** — equivalent to Linear Discriminant
Analysis (LDA). When variances differ across classes, the $x_i^2$ terms survive and the boundary
is **quadratic** — equivalent to Quadratic Discriminant Analysis (QDA). GaussianNB naturally
interpolates between LDA and QDA depending on the data.

**The `var_smoothing` parameter** adds a fraction of the largest observed variance to all
per-class feature variances before computing likelihoods. This prevents divisions by zero when a
feature has zero variance in some class (which happens on small datasets).

**When to prefer GNB:** Continuous numerical features with roughly Gaussian marginals per class.
Medical measurements, biometric data, sensor arrays. Use as your first classifier before trying
SVM or logistic regression — it fits in under a millisecond and gives you a principled baseline.

**sklearn:** `sklearn.naive_bayes.GaussianNB`

```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB(var_smoothing=1e-9)   # default
gnb.fit(X_train, y_train)
# Fitted attributes:
# gnb.theta_          # (n_classes, n_features) — per-class means
# gnb.var_            # (n_classes, n_features) — per-class variances (smoothed)
# gnb.class_prior_    # (n_classes,) — class frequencies
```

---

### 3.2 Multinomial Naive Bayes (MNB) — McCallum & Nigam, 1998

**The problem solved:** Text classification. Documents are represented as word count vectors —
non-negative integers. GaussianNB is completely wrong for this; word counts are not Gaussian.

**The model:** Treat each document as a bag of words. Feature $x_i$ is the count of word $i$ in
the document. The class-conditional likelihood is modeled as a **multinomial distribution**:

$$P(\mathbf{x} \mid c) \propto \prod_{i=1}^{p} \theta_{ci}^{x_i}$$

where $\theta_{ci}$ is the probability of word $i$ in class $c$.

**The zero-count problem and Laplace smoothing:** If word $i$ never appears in any class-$c$
training document, $\hat{\theta}_{ci} = 0$ and the product collapses to zero for any document
containing word $i$. Fix: Laplace (or Lidstone) smoothing:

$$\hat{\theta}_{ci} = \frac{N_{ci} + \alpha}{N_c + \alpha \cdot p}$$

where $N_{ci}$ = total count of word $i$ across all class-$c$ documents, $N_c$ = total token
count in class $c$, $p$ = vocabulary size, $\alpha$ = smoothing parameter. Setting $\alpha = 1$
gives **Laplace smoothing**. Setting $0 < \alpha < 1$ gives **Lidstone smoothing**.

**Classification rule in log space:**

$$\hat{c} = \arg\max_c \left[ \log P(c) + \sum_{i=1}^{p} x_i \log \hat{\theta}_{ci} \right]$$

This is a dot product: $\mathbf{x}^T \mathbf{w}_c + b_c$ where $w_{ci} = \log \hat{\theta}_{ci}$
and $b_c = \log P(c)$. sklearn exposes `coef_` and `intercept_` as aliases.

> 📜 **Origin/Citation:**
> **McCallum, A., & Nigam, K. (1998).** "A Comparison of Event Models for Naive Bayes Text
> Classification." *AAAI-98 Workshop on Learning for Text Categorization.*
> URL: https://www.cs.cmu.edu/~knigam/papers/multinomial-aaai98.pdf

**When to prefer MNB:** Word count or TF-IDF features, medium-to-long documents. The default
choice for any text NB pipeline.

---

### 3.3 Bernoulli Naive Bayes (BNB) — Robertson & Sparck Jones lineage, 1970s

**The problem solved:** For short texts and small vocabularies, what matters is not how often a
word appears but whether it appears at all. MNB ignores zero-count features entirely. But
absence can be informative: the absence of "Viagra" is evidence against spam.

**The key difference from MNB:** The Bernoulli likelihood explicitly models both presence and
absence:

$$P(\mathbf{x} \mid c) = \prod_{i=1}^{p} P(i \mid c)^{x_i} \cdot (1 - P(i \mid c))^{1 - x_i}$$

For a feature absent from the test document ($x_i = 0$), MNB contributes $\theta_{ci}^0 = 1$
(no effect). BNB contributes $(1 - \theta_{ci})^1 = 1 - \theta_{ci}$ — a genuine penalty if
the class typically uses this feature.

For long documents with large vocabularies, this becomes noise — most words are absent from any
given document, creating a flood of weak penalties. For short texts and small vocabularies, this
signal is genuinely useful.

**When to prefer BNB:** Short texts (tweets, headlines), small vocabularies, binary feature
vectors generally. Underperforms MNB on long documents.

---

### 3.4 Complement Naive Bayes (CNB) — Rennie et al., 2003

**The problem solved:** MNB is biased toward majority classes. The majority class accumulates
more token counts, producing systematically higher $\hat{\theta}_{ci}$ values, skewing predictions.

**The innovation:** Estimate $P(\text{word}_i \mid \overline{c})$ — the probability of word $i$
in all documents that are *not* class $c$ (the complement):

$$\hat{\theta}_{ci} = \frac{\alpha_i + \sum_{j:\, y_j \neq c} d_{ij}}{\alpha + \sum_{j:\, y_j \neq c} \sum_k d_{kj}}$$

Prediction assigns the class with the **smallest** complement weight sum:

$$\hat{c} = \arg\min_c \sum_i x_i \cdot \log \hat{\theta}_{ci}$$

Because each class's estimate draws from a much larger pool (all other classes combined),
the imbalance effect is substantially reduced.

> 📜 **Origin/Citation:**
> **Rennie, J. D. M., Shih, L., Teevan, J., & Karger, D. R. (2003).** "Tackling the Poor
> Assumptions of Naive Bayes Text Classifiers." *ICML 2003, Vol. 3*, 616–623.
> URL: https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf

> 🏆 **Best practice:** For text classification, default to `ComplementNB(alpha=0.1, norm=True)`
> unless you have confirmed balanced classes. CNB "regularly outperforms MNB, often by a
> considerable margin" on standard benchmarks.

---

### 3.5 Categorical Naive Bayes (CatNB) — sklearn 0.22+, 2019

**The problem solved:** Tabular datasets with nominal categorical features (blood type, country
of origin, product category). GaussianNB treats them as continuous; MultinomialNB treats them
as counts — both are wrong.

**The model:** For feature $i$ with $n_i$ possible categories, and test value $x_i = t$:

$$P(x_i = t \mid y = c) = \frac{N_{tic} + \alpha}{N_c + \alpha \cdot n_i}$$

Requires `OrdinalEncoder` preprocessing. Set `min_categories` to prevent errors on unseen
categories at test time.

**When to prefer CatNB:** Purely nominal features. Do not use for ordinal features — CatNB
treats all categories as exchangeable.

---

### 3.6 Tree-Augmented Naive Bayes (TAN) — Friedman, Geiger & Goldszmidt, 1997

**The problem solved:** Relax the full independence assumption by allowing each feature to depend
on at most one other feature, without exploding computational complexity.

**The innovation:** Use the **Chow-Liu algorithm** to find the maximum-weight spanning tree over
the feature graph, where edge weights are conditional mutual information:

$$I(X_i; X_j \mid Y) = \sum_{c} \sum_{x_i, x_j} P(x_i, x_j, c) \log \frac{P(x_i, x_j \mid c)}{P(x_i \mid c) P(x_j \mid c)}$$

Steps:
1. Compute $I(X_i; X_j \mid Y)$ for all $\binom{p}{2}$ feature pairs — $\mathcal{O}(p^2 n)$
2. Build maximum weight spanning tree (Kruskal's or Prim's)
3. Root and direct edges; augment NB CPTs with feature-feature dependencies

Result: each feature has at most one other feature parent. TAN significantly outperforms plain
NB on 21 of 28 UCI benchmark datasets.

> 📜 **Origin/Citation:**
> **Friedman, N., Geiger, D., & Goldszmidt, M. (1997).** "Bayesian Network Classifiers."
> *Machine Learning, 29*, 131–163. DOI: 10.1023/A:1007465528199

**Python:** `pgmpy.estimators.TreeSearch` with `estimator_type="tan"` — verify API against
current pgmpy docs as it has changed across releases.

---

### 3.7 Averaged One-Dependence Estimators (AODE) — Webb et al., 2005

**The problem solved:** TAN picks one tree structure; instability across roots introduces
variance. Average over all $p$ possible super-parent one-dependence classifiers instead:

$$P_{\text{AODE}}(y \mid \mathbf{x}) \propto \sum_{i:\, n(x_i) \geq m} P(y, x_i) \prod_{j \neq i} P(x_j \mid y, x_i)$$

More stable than TAN; substantially outperforms NB. Available in WEKA; no maintained sklearn
implementation as of mid-2026.

> 📜 **Origin/Citation:**
> **Webb, G. I., Boughton, J. R., & Wang, Z. (2005).** "Not So Naive Bayes: Aggregating
> One-Dependence Estimators." *Machine Learning, 58*, 5–24.

---

### 3.8 The Semi-Naive Bayes Taxonomy: Where Each Variant Sits

It helps to see all the NB variants in a unified framework. The core question is: how many
feature-feature dependencies are we willing to model?

| Model | Feature dependencies allowed | Complexity | Where to use |
|---|---|---|---|
| **Naive Bayes** | None (all features independent given $y$) | $\mathcal{O}(Kp)$ params | Default everywhere — fast, interpretable |
| **TAN** | Each feature has at most 1 other feature parent | $\mathcal{O}(Kp^2)$ training | When you suspect strong pairwise dependencies and have tabular data |
| **AODE** | All one-parent models averaged | $\mathcal{O}(Kp^2)$ inference | When TAN is too sensitive to tree root choice |
| **Bayesian Network Classifier** | Any DAG structure (searched) | $\mathcal{O}(K \cdot 2^p)$ worst case | Only for small $p$ (<20 features); research/academic contexts |
| **Logistic Regression** | All feature interactions implicitly captured via weights | $\mathcal{O}(Kp \cdot T)$ training | When independence is too strong a constraint and $n$ is large enough |

The progression from NB to LR is a continuum: as you relax the independence assumption more
and more, you move along the spectrum from NB toward LR (or GBM at the extreme). Each step
requires more data to estimate the additional parameters. This is the fundamental tradeoff
in choosing your classifier.

> 💡 **Intuition:** Think of NB and LR as two extremes of a generative-discriminative spectrum.
> NB says: "I know the exact mechanism that generated the data (joint distribution), and I
> use it to classify." LR says: "I don't know the mechanism — I just find the best decision
> boundary given the data." NB's assumptions help it learn fast with few examples. LR's
> flexibility helps it be accurate with many examples. For text, this crossover occurs around
> a few thousand training examples per class.

---

## §4 — Hyperparameters: The Complete Guide

### 4.1 `var_smoothing` (GaussianNB only)

**What it controls:** Adds `var_smoothing × max(var_)` to all per-class feature variances before
computing Gaussian likelihoods. Prevents division by zero when a feature has identical values
within a class; acts as a regularizer.

**Default:** `1e-9`

**Mechanism:** Effective variance: $\tilde{\sigma}_{ci}^2 = \sigma_{ci}^2 + \varepsilon \cdot \max_{c', i'} \sigma_{c'i'}^2$.
The `epsilon_` attribute stores the actual additive value applied.

**Too small:** Zero-variance features → log-probability of $-\infty$, zeroing out the feature.
**Too large:** Classes see effectively the same wide Gaussian; predictions collapse to the prior.

**Tuning:**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
import numpy as np

param_grid = {"var_smoothing": np.logspace(-9, -1, 100)}
gs = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring="accuracy")
gs.fit(X_train, y_train)
print(f"Best var_smoothing: {gs.best_params_['var_smoothing']:.2e}")
```

**Typical range:** `[1e-9, 1e-1]` on a log scale. Usually best near the default unless features
have very small variance or the dataset is tiny.

---

### 4.2 `alpha` (MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB)

**What it controls:** Additive smoothing. Prevents zero-probability estimates for feature-class
combinations not observed in training.

**Default:** `1.0` (Laplace smoothing)

**Mechanism:**
$$\hat{\theta}_{ci} = \frac{N_{ci} + \alpha}{N_c + \alpha \cdot p}$$

- `alpha = 0`: MLE — zero-count features produce $\log(0) = -\infty$
- `alpha = 1`: Laplace smoothing
- `0 < alpha < 1`: Lidstone smoothing — often better in practice
- `alpha > 1`: Pushes class-conditional distributions toward uniform

**Too small:** Zero-count features cause $-\infty$. `force_alpha=True` (default in sklearn 1.4+)
prevents using alpha below `1e-10` without explicit override.
**Too large:** Estimates dominated by pseudo-counts, not actual data; classification degrades
toward the prior.

**Tuning with Optuna:**
```python
import optuna
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
    model = MultinomialNB(alpha=alpha)
    return cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)
print(f"Best alpha: {study.best_params['alpha']:.4f}")
```

**Rule of thumb:** For TF-IDF text features, `alpha` between 0.01 and 0.5 typically outperforms
the default of 1.0. Start with `alpha=0.1`.

> ⚠️ **Pitfall:** `force_alpha` defaulted to `False` before sklearn 1.4 and to `True` in 1.4+.
> If you specify very small alpha values and recently upgraded sklearn, verify the behavior
> hasn't changed. # verify in current docs

---

### 4.3 `fit_prior` (MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB)

**What it controls:** Whether to learn class prior probabilities from training data frequency.
If `False`, uses uniform priors $P(c) = 1/K$.

**Default:** `True`

**When to set `False`:** When training class distribution is artificial (oversampled/undersampled
for training). With uniform priors, the model relies entirely on the likelihood.

> 💡 **Intuition:** `fit_prior=False` is a maximum-entropy prior: "Let the features decide."
> Use this when you have applied SMOTE or undersampling.

---

### 4.4 `class_prior` (GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB)

**What it controls:** Manually set prior probabilities, overriding both estimated and uniform.

**Default:** `None`

**When to use:** When the deployment base rate differs from training composition. If spam is
2% of real traffic but training was 50/50 balanced, set `class_prior=[0.98, 0.02]`.

> 🔧 **In practice:** In production fraud/spam classification, always explicitly set
> `class_prior` to match the deployment base rate. This makes the prior assumption transparent
> and auditable, unlike threshold tuning which achieves the same effect implicitly.

---

### 4.5 `binarize` (BernoulliNB only)

**What it controls:** Threshold for binarizing continuous inputs. Features above `binarize` → 1;
below → 0. Set `binarize=None` if input is already binary.

**Default:** `0.0`

**Tuning:** For TF-IDF inputs, tune on validation set. A value near the median feature value
often works well. Pre-binarize with `sklearn.preprocessing.Binarizer` for interpretability.

---

### 4.6 `norm` (ComplementNB only)

**What it controls:** Whether to L1-normalize each class's weight vector.

**Default:** `False` (matches Mahout/Weka)

**When to set `True`:** When document lengths vary greatly; when following Rennie et al. (2003)
exactly.

---

### 4.7 Tuning Playbook Summary

| Parameter | Variant | Typical Range | Effect | Tune with |
|---|---|---|---|---|
| `var_smoothing` | GNB | `[1e-9, 1e-1]` log | Variance regularization | `GridSearchCV` with `np.logspace(-9, -1, 100)` |
| `alpha` | MNB, BNB, CNB, CatNB | `[0.001, 5.0]` log | Smoothing strength | Optuna / `GridSearchCV` |
| `fit_prior` | MNB, BNB, CNB, CatNB | `{True, False}` | Use empirical class freqs | `False` when ratios are artificial |
| `class_prior` | GNB, MNB, BNB, CatNB | array summing to 1 | Manual prior | Set to deployment base rate |
| `binarize` | BNB | median feature value | Continuous → binary threshold | Tune on validation set |
| `norm` | CNB | `{True, False}` | L1 weight normalization | `True` for variable-length docs |
| `min_categories` | CatNB | `≥ max categories` | Prevent unseen-category errors | Set from encoder's `n_categories_` |

**Text pipeline parameters** (often matter more than NB-specific params):

| Parameter | Location | Typical Range | Notes |
|---|---|---|---|
| `max_features` | `TfidfVectorizer` | `[5000, 100000]` | Vocabulary cap; removes noisy rare words |
| `ngram_range` | `TfidfVectorizer` | `(1,1)`, `(1,2)` | Bigrams improve accuracy, cost memory |
| `sublinear_tf` | `TfidfVectorizer` | `{True, False}` | Log-scales TF; usually improves NB |
| `max_df` | `TfidfVectorizer` | `[0.5, 1.0]` | Removes high-frequency stopwords |
| `min_df` | `TfidfVectorizer` | `[1, 5, 10]` | Removes rare terms |

### 4.8 The `predict_joint_log_proba` Method (sklearn 1.2+)

sklearn 1.2 added `predict_joint_log_proba(X)`, which returns the **unnormalized** log-posterior
(joint log-probability $\log P(\mathbf{x}, c)$) for each class, before the `logsumexp`
normalization step. This is distinct from `predict_log_proba`, which returns the normalized
log-posteriors.

```python
# verify in current sklearn 1.5.x docs — added in 1.2
joint_log_probs = mnb.predict_joint_log_proba(X_test)   # shape (n_samples, n_classes)
log_posteriors  = mnb.predict_log_proba(X_test)          # normalized

# Verify: normalized = joint - logsumexp(joint)
from scipy.special import logsumexp
manual_normalized = joint_log_probs - logsumexp(joint_log_probs, axis=1, keepdims=True)
assert np.allclose(log_posteriors, manual_normalized, atol=1e-6)
```

The joint log-probabilities are useful when you want to compare across models (a sample with
lower joint log-probability across all classes may be an outlier that the model is uncertain
about) or when implementing custom scoring functions that depend on the unnormalized evidence.

> ⚠️ **Version note:** `predict_joint_log_proba` was added in sklearn 1.2. If you are using an
> older sklearn version, compute it manually as `feature_log_prob_ @ X.T + class_log_prior_[:, None]`.
> # verify in current docs

---

## §5 — Strengths

### 5.1 Training Speed: $\mathcal{O}(n \cdot p)$ with a Tiny Constant

NB requires a single pass over the training data. No optimization loop, no gradient computation,
no convergence criterion. A `MultinomialNB` on 18,000 documents with a 50,000-word vocabulary
trains in under 200 milliseconds on a laptop. Logistic regression on the same data requires
several seconds of LBFGS iterations. Random forests: minutes.

In pipelines with millions of short documents, NB's training speed enables real-time model
updates — retrain after every batch, adapt immediately to distribution shift.

**The mechanism:** Counting and averaging are sufficient statistics. $N_{ci}$ for all $(c, i)$
pairs is everything you need. These statistics are additive and support exact incremental updates.

---

### 5.2 Out-of-Core and Online Learning via `partial_fit`

All five sklearn NB variants support `partial_fit` — incremental updates from mini-batches.
NB scales to arbitrarily large datasets without loading all data into memory simultaneously.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

vec = HashingVectorizer(n_features=2**18, alternate_sign=False)
clf = MultinomialNB(alpha=0.1)
all_classes = np.array([0, 1, 2, 3])

for X_batch, y_batch in stream_data():
    X_hashed = vec.transform(X_batch)
    clf.partial_fit(X_hashed, y_batch, classes=all_classes)
    all_classes = None   # only pass on first call
```

For GaussianNB, mean and variance updates use Welford's online algorithm — numerically stable,
$\mathcal{O}(1)$ per sample.

> ⚠️ **Pitfall:** `TfidfVectorizer` does NOT support `partial_fit`. For streaming text, use
> `HashingVectorizer` with `alternate_sign=False` (default `True` produces negatives
> incompatible with MultinomialNB).

---

### 5.3 Handles High-Dimensional Sparse Data Gracefully

Text feature spaces have 10,000–1,000,000 dimensions; NB never needs to compute pairwise
interactions, invert covariance matrices, or solve linear systems. Parameter count is
$\mathcal{O}(K \cdot p)$ — linear in dimensionality.

Contrast: LDA requires estimating a $p \times p$ covariance matrix (infeasible for $p = 50{,}000$);
SVM kernel matrices scale as $\mathcal{O}(n^2)$.

For extremely high-dimensional sparse text, NB's memory footprint is also unbeatable. A
MultinomialNB model with 20 classes and a 100,000-word vocabulary stores exactly
$20 \times 100{,}000 = 2{,}000{,}000$ floating-point parameters for `feature_log_prob_` plus
20 for `class_log_prior_`. At 8 bytes per float64, this is approximately 16 MB total. A random
forest with 100 trees on the same data might require hundreds of megabytes. In memory-
constrained deployment environments (edge devices, mobile, serverless functions with tight
memory limits), NB is often the only viable classifier.

---

### 5.4 Works Well with Small Training Sets

NB estimates each parameter independently. With Laplace smoothing as a prior over unseen values,
NB produces reasonable estimates even with dozens of training examples per class. Valuable in
cold-start scenarios: new product categories, low-resource language tagging.

---

### 5.5 Handles Missing Features Gracefully (Generative Model Advantage)

A missing feature $x_j$ can be excluded from the log-sum at inference time. No imputation
required. Discriminative models like logistic regression have no principled way to handle
missing features without imputation.

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
from scipy.stats import norm as scipy_norm

gnb = GaussianNB().fit(X_train, y_train)

x_raw = np.array([1.2, np.nan, 0.8, 3.1])
observed = ~np.isnan(x_raw)

log_post = gnb.class_log_prior_.copy()
for i in np.where(observed)[0]:
    log_post += scipy_norm.logpdf(
        x_raw[i], gnb.theta_[:, i], np.sqrt(gnb.var_[:, i])
    )
pred_class = np.argmax(log_post)
```

---

### 5.6 Calibration is Fixable

Despite producing overconfident raw probabilities, NB's probabilities are fixable with
`CalibratedClassifierCV`. After calibration, NB can serve as a fast probability estimator with
competitive Brier scores. This is in contrast to some complex models where calibration is harder
to achieve reliably.

---

### 5.7 The "Good Enough" Baseline That Rarely Fails

There is a deeper, more practical reason to always run NB before anything else: it is a
**nearly unbreakable sanity check** on your data pipeline.

When NB achieves 50% accuracy on a 10-class problem (random chance level), it almost always
means your data pipeline is broken: features are computed on the wrong split, labels are
shuffled, there is a preprocessing bug, or the feature extraction is producing garbage. NB's
simplicity means there are almost no model-specific failure modes — if NB fails, the problem
is almost certainly in the data, not the model.

When NB achieves 70–80% accuracy and logistic regression achieves 75–85%, you know two things:
(1) the data contains genuine signal; (2) the remaining gap is due to feature correlations and
interactions that NB cannot model. This tells you exactly what to invest in: better feature
engineering or a more expressive model.

> 🏆 **Best practice:** Always run `ComplementNB(alpha=0.1)` with `TfidfVectorizer` as your
> first text classifier and `GaussianNB()` as your first tabular classifier, before fitting
> anything else. This takes under 10 lines of code and 5 seconds of compute. If it fails
> unexpectedly, debug the pipeline. If it succeeds at 70%+, you have a production-ready baseline
> and a bound on how much complex models can improve over it.

---

### 5.8 Native Multi-Class Classification

NB naturally extends to $K > 2$ classes without any modification or one-vs-rest wrapping.
The decision rule is already over all $K$ classes simultaneously:

$$\hat{y} = \arg\max_{c \in \{c_1, \ldots, c_K\}} \left[ \log P(c) + \sum_{i=1}^p \log P(x_i \mid c) \right]$$

Compare to SVMs, which require $K(K-1)/2$ binary classifiers for $K$-class problems
(one-vs-one) or $K$ classifiers (one-vs-rest). For the 20-class 20 Newsgroups task, NB trains
as a single model in 200ms; linear SVM requires training 20 binary classifiers with multiple
iterations of dual coordinate descent each.

---

## §6 — Weaknesses & Failure Modes

### 6.1 The Independence Assumption: When It Breaks Hard

**Highly correlated features:** "loan" and "credit" in financial text co-occur in 90% of
documents. NB double-counts their evidence — both features give the same signal but NB treats
them as independent. This catastrophically distorts probability estimates (though accuracy may
be unaffected per Domingos & Pazzani).

**Asymmetric correlations:** If "New York" co-occurs disproportionately in sports documents but
is anti-correlated with financial text features, NB's independence assumption can flip the class
ranking entirely.

**Detection:**
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

corr_matrix = pd.DataFrame(X_train).corr().abs()
n_high_corr = (corr_matrix > 0.7).values.sum() - len(corr_matrix)
print(f"Feature pairs with |corr| > 0.7: {n_high_corr // 2}")

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, vmax=1, center=0.5, cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout(); plt.show()
```

**Mitigation:** Feature selection; PCA before GaussianNB; TAN/AODE variants; switch to logistic
regression, which handles correlations correctly.

---

### 6.2 Pathologically Overconfident Probabilities

Because NB multiplies $p$ likelihoods, the posterior clusters near 0 and 1 for large $p$.
Raw probabilities of 0.9999 or 0.0001 are meaningless as probability estimates, even when the
true posterior is 0.6 vs. 0.4. This makes NB unsuitable for: threshold selection based on
calibrated probabilities, expected value maximization, or as a feature in a stacking ensemble.

**Mitigation:**
```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB

calibrated_nb = CalibratedClassifierCV(
    MultinomialNB(alpha=0.1),
    method="isotonic",   # use "sigmoid" for <1000 calibration samples
    cv=5
)
calibrated_nb.fit(X_train, y_train)
```

---

### 6.3 GaussianNB: Non-Gaussian Features

Real-world continuous features are often skewed (income, counts, durations). GaussianNB produces
incorrect likelihoods for these features.

**Detection:** Shapiro-Wilk test per class per feature. Log-transform skewed features as a fix.
For mixed distributions, use `pomegranate.NaiveBayes` with per-feature distribution specification.

---

### 6.4 Feature Type Mismatch

- **MultinomialNB:** requires non-negative features. StandardScaler → negatives → `ValueError`.
- **BernoulliNB:** expects binary; raw TF-IDF silently binarized at `binarize` threshold.
- **CategoricalNB:** requires integer-encoded categories; float features cause incorrect indexing.

> 🏆 **Best practice:** Assert feature type preconditions before fitting:
> ```python
> assert (X_train >= 0).all(), "MultinomialNB requires non-negative features"
> ```

---

### 6.5 Cannot Learn Feature Interactions

NB is blind to interaction effects: "not good" is not the negation of "good" to NB, just two
separate features. Bigrams via `ngram_range=(1,2)` partially address this in text, but this is
preprocessing, not model expressiveness. For interaction-heavy problems, gradient boosting or
neural models will substantially outperform NB.

---

### 6.6 Concept Drift: NB Forgets Old Data Unless Explicitly Updated

NB's `partial_fit` supports online updates, but it accumulates all evidence with equal weight.
Early training examples from a different distribution (old spam tactics, old product categories)
have the same weight as recent examples. In drifting environments, historical data is not just
useless — it is actively harmful.

**The issue:** If spam evolves from Nigerian prince emails to phishing for cryptocurrency
wallets, the historical "Nigerian prince" features still have high log-probability for the spam
class and will continue to influence predictions indefinitely.

**Mitigation strategies:**
- **Periodic full retraining:** Discard the old model and refit from scratch on a recent time
  window (sliding window approach). For NB this is trivially fast.
- **Exponential forgetting:** Multiply the accumulated counts by a decay factor $\lambda < 1$
  before incorporating new data: `clf.feature_count_ *= lambda_`. This is not supported
  natively in sklearn but is easy to implement manually.

```python
import numpy as np
from sklearn.naive_bayes import MultinomialNB

# Manual decay for concept drift adaptation
def decay_and_update(clf, X_new, y_new, decay=0.95):
    """Apply exponential forgetting before updating with new data."""
    clf.feature_count_ *= decay
    clf.class_count_   *= decay
    clf.partial_fit(X_new, y_new)
    return clf
```

- **`river` adaptive NB:** The `river` library provides adaptive NB variants that handle
  concept drift more gracefully in true online settings.

---

### 6.7 Class Prior Mismatch Between Training and Deployment

This is one of the most common NB deployment bugs. Your training set was 50/50 balanced (you
undersampled the majority class for training). NB learns a prior of $P(\text{spam}) = 0.5$.
In production, the true prior is $P(\text{spam}) = 0.02$. Every prediction is biased toward
spam — false positive rate is enormous.

**Detection:**
```python
# Check what prior NB learned
print("Learned class priors:", model.class_prior_)
print("True deployment priors (from ops dashboard): [0.98, 0.02]")

# If they differ significantly, fix it:
model_corrected = MultinomialNB(alpha=0.1)
model_corrected.fit(X_train, y_train)
model_corrected.class_log_prior_ = np.log([0.98, 0.02])   # Override the prior
```

**Or use `class_prior` at construction time:**
```python
model = MultinomialNB(alpha=0.1, class_prior=[0.98, 0.02])
model.fit(X_train, y_train)
```

> ⚠️ **Pitfall:** After calling `.fit()`, modifying `class_log_prior_` directly works in
> sklearn 1.5.x but is fragile across versions. The robust solution is to set `class_prior`
> at construction time and refit.

---

## §7 — Diagnostic Plots & Evaluation

NB classification is evaluated like any other classifier, but two diagnostics are especially
important: the calibration curve (because NB probabilities are severely miscalibrated) and the
top-feature log-probability plot (because NB is interpretable and you should verify it learned
sensible patterns). Here is the complete diagnostic toolkit with runnable code.

### 7.1 Calibration Curve

The calibration curve is the single most important diagnostic for NB. Plot it before using any
NB probabilities for decision-making.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ── Setup: binary classification (sci.space vs. talk.religion.misc) ──────────
train_data = fetch_20newsgroups(
    subset="train",
    categories=["sci.space", "talk.religion.misc"],
    remove=("headers", "footers", "quotes"),
    random_state=42,
)
test_data = fetch_20newsgroups(
    subset="test",
    categories=["sci.space", "talk.religion.misc"],
    remove=("headers", "footers", "quotes"),
    random_state=42,
)

vec = TfidfVectorizer(max_features=20_000, sublinear_tf=True)
X_train = vec.fit_transform(train_data.data)
X_test  = vec.transform(test_data.data)
y_train, y_test = train_data.target, test_data.target

# ── Uncalibrated NB ───────────────────────────────────────────────────────────
mnb = MultinomialNB(alpha=0.1)
mnb.fit(X_train, y_train)

# ── Calibrated NB (isotonic regression) ──────────────────────────────────────
cal_mnb_iso = CalibratedClassifierCV(MultinomialNB(alpha=0.1), method="isotonic", cv=5)
cal_mnb_iso.fit(X_train, y_train)

# ── Calibrated NB (Platt scaling / sigmoid) ──────────────────────────────────
cal_mnb_sig = CalibratedClassifierCV(MultinomialNB(alpha=0.1), method="sigmoid", cv=5)
cal_mnb_sig.fit(X_train, y_train)

# ── Plot calibration curves ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

for model, name, color in [
    (mnb,          "Uncalibrated MNB",   "tab:blue"),
    (cal_mnb_iso,  "Isotonic calibrated", "tab:orange"),
    (cal_mnb_sig,  "Sigmoid calibrated",  "tab:green"),
]:
    CalibrationDisplay.from_estimator(
        model, X_test, y_test,
        n_bins=10, ax=ax, name=name, color=color
    )

ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Curve: Naive Bayes (Binary Text Classification)")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
```

**How to read it:** A perfectly calibrated model lies on the diagonal. Raw NB shows a
characteristic S-curve bowed toward 0 and 1 — probabilities are clustered at the extremes.
After isotonic calibration, the curve should closely follow the diagonal. If it still deviates,
you may need more calibration data or a different method.

> 💡 **Intuition:** The S-shaped miscalibration is a direct consequence of the independence
> assumption. Multiplying $p$ independent likelihoods is equivalent to raising the true
> posterior to the power of $p$ (approximately). For large $p$, this saturates at 0 or 1.

---

### 7.2 Confusion Matrix

```python
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Use the best model (calibrated MNB or raw MNB for classification)
y_pred = mnb.predict(X_test)

# ── Confusion matrix (absolute counts) ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=test_data.target_names)
disp.plot(ax=axes[0], colorbar=False)
axes[0].set_title("Confusion Matrix (counts)")

# ── Normalized confusion matrix (rates) ───────────────────────────────────────
cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
disp_norm = ConfusionMatrixDisplay(cm_norm, display_labels=test_data.target_names)
disp_norm.plot(ax=axes[1], colorbar=False)
axes[1].set_title("Confusion Matrix (recall per class)")

plt.tight_layout()
plt.show()

# ── Classification report ─────────────────────────────────────────────────────
print(classification_report(y_test, y_pred, target_names=test_data.target_names))
```

**How to read it:** The normalized matrix shows per-class recall (what fraction of true
class-$c$ examples are correctly classified). For NB, class imbalance is a common cause of
asymmetric recall — this is exactly what ComplementNB fixes.

---

### 7.3 Top Feature Log-Probabilities (NB's Built-In Explainability)

For MultinomialNB and ComplementNB, the `feature_log_prob_` matrix tells you exactly which
features the model associates most strongly with each class. This is one of NB's most
practical strengths — you can read directly what the model learned.

```python
import pandas as pd

# ── Top features per class for MultinomialNB ──────────────────────────────────
def top_features_per_class(model, vectorizer, n_top=15):
    feature_names = np.array(vectorizer.get_feature_names_out())
    rows = []
    for i, class_name in enumerate(model.classes_):
        log_probs = model.feature_log_prob_[i]
        top_indices = np.argsort(log_probs)[-n_top:][::-1]
        for rank, idx in enumerate(top_indices):
            rows.append({
                "class": class_name,
                "rank": rank + 1,
                "feature": feature_names[idx],
                "log_prob": log_probs[idx],
            })
    return pd.DataFrame(rows)

df_top = top_features_per_class(mnb, vec, n_top=15)

# ── Visualize ─────────────────────────────────────────────────────────────────
n_classes = len(mnb.classes_)
fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 6), sharey=False)

for i, (class_idx, group) in enumerate(df_top.groupby("class")):
    ax = axes[i] if n_classes > 1 else axes
    ax.barh(group["feature"][::-1], group["log_prob"][::-1], color="steelblue")
    ax.set_title(f"Class: {train_data.target_names[class_idx]}")
    ax.set_xlabel("Log P(feature | class)")
    ax.tick_params(axis="y", labelsize=9)

plt.suptitle("Top 15 Features per Class (MultinomialNB)", fontsize=14)
plt.tight_layout()
plt.show()
```

**How to read it:** Higher (less negative) log probability means the feature is more common
in that class. For spam classification, you expect to see "click", "free", "offer" near the
top of the spam class; "regards", "attached", "meeting" near the top of ham.

> 🔧 **In practice:** This plot is invaluable for catching data leakage. If a feature that
> should not be informative (like an email header field, a timestamp format, a database ID)
> appears in the top features, your preprocessing pipeline has a leak.

---

### 7.4 Learning Curve

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    Pipeline([
        ("vec", TfidfVectorizer(max_features=20_000, sublinear_tf=True)),
        ("clf", MultinomialNB(alpha=0.1)),
    ]),
    train_data.data, y_train,
    train_sizes=np.linspace(0.05, 1.0, 15),
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_sizes, train_mean, "o-", color="tab:blue",   label="Training score")
ax.plot(train_sizes, val_mean,   "o-", color="tab:orange", label="Validation score")
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="tab:blue")
ax.fill_between(train_sizes, val_mean   - val_std,   val_mean   + val_std,   alpha=0.2, color="tab:orange")
ax.set_xlabel("Training set size")
ax.set_ylabel("Accuracy")
ax.set_title("Learning Curve — MultinomialNB + TF-IDF")
ax.legend()
plt.tight_layout()
plt.show()
```

**How to read it:** A well-behaved NB learning curve shows validation accuracy rapidly
converging as training size increases, with very little gap between training and validation
scores (NB rarely overfits). If the gap is large, check for data leakage or consider reducing
vocabulary size with `max_features`.

---

### 7.5 ROC Curve and AUC (Binary Case)

```python
from sklearn.metrics import RocCurveDisplay, roc_auc_score

fig, ax = plt.subplots(figsize=(7, 6))

for model, name, color in [
    (mnb,         "Uncalibrated MNB",   "tab:blue"),
    (cal_mnb_iso, "Isotonic calibrated", "tab:orange"),
]:
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name, color=color)

ax.set_title("ROC Curve — Naive Bayes (binary)")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.tight_layout()
plt.show()

# AUC is independent of calibration — uncalibrated NB should match calibrated NB on AUC
auc_raw = roc_auc_score(y_test, mnb.predict_proba(X_test)[:, 1])
auc_cal = roc_auc_score(y_test, cal_mnb_iso.predict_proba(X_test)[:, 1])
print(f"Uncalibrated AUC: {auc_raw:.4f}")
print(f"Calibrated AUC:   {auc_cal:.4f}")
```

> 💡 **Intuition:** AUC measures the ranking quality of the probabilities — are higher
> predicted scores more likely to be positive? Calibration does not change the ranking, only
> the magnitude. So AUC should be nearly identical before and after calibration. If it changes
> substantially, something else changed in the calibration pipeline (model refitting).

---

### 7.6 Brier Score Decomposition

The Brier score is the mean squared error between predicted probabilities and true labels.
Lower is better. It directly penalizes miscalibrated overconfident predictions.

```python
from sklearn.metrics import brier_score_loss

prob_raw = mnb.predict_proba(X_test)[:, 1]
prob_cal = cal_mnb_iso.predict_proba(X_test)[:, 1]

brier_raw = brier_score_loss(y_test, prob_raw)
brier_cal = brier_score_loss(y_test, prob_cal)

print(f"Brier score (uncalibrated): {brier_raw:.4f}")
print(f"Brier score (calibrated):   {brier_cal:.4f}")
print(f"Improvement: {(brier_raw - brier_cal) / brier_raw * 100:.1f}%")
```

Calibration typically reduces the Brier score for NB by 20–40% on text classification tasks.

---

### 7.7 Precision-Recall Curve (For Imbalanced Classes)

When classes are severely imbalanced (spam detection: 2% positive rate; medical screening:
0.1% positive rate), the ROC curve is overly optimistic — the large true-negative pool
inflates the AUC. The **Precision-Recall curve** is more informative in these settings.

```python
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score

fig, ax = plt.subplots(figsize=(7, 6))

for model, name, color in [
    (mnb,         "Uncalibrated MNB",    "tab:blue"),
    (cal_mnb_iso, "Isotonic calibrated", "tab:orange"),
]:
    PrecisionRecallDisplay.from_estimator(
        model, X_test, y_test, ax=ax, name=name, color=color
    )

baseline = y_test.mean()   # fraction of positives = no-skill line
ax.axhline(y=baseline, linestyle="--", color="gray", alpha=0.7,
           label=f"No-skill baseline (prevalence={baseline:.2f})")
ax.set_title("Precision-Recall Curve")
ax.legend()
plt.tight_layout()
plt.show()

ap_raw = average_precision_score(y_test, mnb.predict_proba(X_test)[:, 1])
ap_cal = average_precision_score(y_test, cal_mnb_iso.predict_proba(X_test)[:, 1])
print(f"Average Precision (uncalibrated): {ap_raw:.4f}")
print(f"Average Precision (calibrated):   {ap_cal:.4f}")
```

> 💡 **Intuition:** Average Precision (AP, also called AUC-PR) summarizes the PR curve as the
> area under it. Unlike ROC-AUC, AP is sensitive to class imbalance — a model that never
> predicts positive gets AP ≈ prevalence, not 0.5. AP is the right summary metric when you care
> about finding the positives (spam, fraud, disease) more than about correct negatives.

> 🔧 **In practice:** For spam filtering or fraud detection, use AP as your primary offline
> evaluation metric, not ROC-AUC. At deployment, calibrate the model and set the operating
> threshold based on your precision/recall tradeoff requirements (e.g., "at most 1% false
> positive rate" or "at least 90% recall").

---

### 7.8 Validation Curve: Finding Optimal Alpha

```python
from sklearn.model_selection import validation_curve

alphas = np.logspace(-3, 1, 30)

train_scores, val_scores = validation_curve(
    Pipeline([
        ("vec", TfidfVectorizer(max_features=20_000, sublinear_tf=True)),
        ("clf", MultinomialNB()),
    ]),
    train_data.data, y_train,
    param_name="clf__alpha",
    param_range=alphas,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(alphas, train_mean, "o-", color="tab:blue",   label="Training score")
ax.semilogx(alphas, val_mean,   "o-", color="tab:orange", label="Validation score")
ax.fill_between(alphas, val_mean - val_std, val_mean + val_std, alpha=0.2, color="tab:orange")

best_alpha = alphas[np.argmax(val_mean)]
ax.axvline(x=best_alpha, color="red", linestyle="--",
           label=f"Best alpha = {best_alpha:.4f}")

ax.set_xlabel("alpha (log scale)")
ax.set_ylabel("Accuracy")
ax.set_title("Validation Curve — MultinomialNB alpha")
ax.legend()
plt.tight_layout()
plt.show()

print(f"Best alpha: {best_alpha:.4f}, Best CV accuracy: {val_mean.max():.4f}")
```

**How to read it:** As `alpha` increases from very small (near 0) toward 1, the validation
curve typically rises sharply (smoothing prevents zeros) then plateaus and slowly declines
(too much smoothing kills discrimination). The peak is your best `alpha`. For TF-IDF text,
this peak is usually between 0.01 and 0.5.

---

## §8 — Explainability & Interpretable AI

NB is one of the most naturally interpretable classifiers in machine learning. The model
parameters have direct probabilistic meaning: `feature_log_prob_[c, i]` is the log probability
of feature $i$ given class $c$. No black-box needed.

### 8.1 Built-In: Feature Log-Probabilities (The Native Explanation)

The cleanest way to explain a Naive Bayes prediction is to show the log-probability contribution
of each feature to the final score:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

def explain_prediction(model, vectorizer, document, true_label=None):
    """
    Show which features contributed most to the NB prediction for a single document.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    x = vectorizer.transform([document])

    # Get non-zero features in this document
    nonzero_features = x.nonzero()[1]
    counts = np.array(x[0, nonzero_features]).flatten()

    # Score contribution: count * log_prob per class
    scores_per_class = {}
    for c_idx, c_name in enumerate(model.classes_):
        log_probs = model.feature_log_prob_[c_idx, nonzero_features]
        contributions = counts * log_probs   # weighted log-prob per feature
        scores_per_class[c_name] = {
            "features": feature_names[nonzero_features],
            "contributions": contributions,
            "prior": model.class_log_prior_[c_idx],
            "total": model.class_log_prior_[c_idx] + contributions.sum(),
        }

    predicted_class = model.predict(x)[0]
    predicted_proba = model.predict_proba(x)[0]

    print(f"Prediction: class {predicted_class} "
          f"({model.classes_[predicted_class] if hasattr(model.classes_[0], '__int__') else predicted_class})")
    if true_label is not None:
        print(f"True label: {true_label}")
    print(f"Predicted probabilities: {dict(zip(model.classes_, predicted_proba.round(4)))}")
    print()

    # Show top contributing features for predicted class
    c_idx = predicted_class
    contribs = scores_per_class[c_idx]["contributions"]
    features = scores_per_class[c_idx]["features"]

    top_k = min(15, len(features))
    top_idx = np.argsort(np.abs(contribs))[-top_k:][::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["steelblue" if contribs[i] > 0 else "tomato" for i in top_idx]
    ax.barh(features[top_idx][::-1], contribs[top_idx][::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Log-probability contribution (count × log P(feature|class))")
    ax.set_title(f"Feature contributions for predicted class {predicted_class}")
    plt.tight_layout()
    plt.show()

    return scores_per_class

# Usage
explain_prediction(mnb, vec, "astronauts returned from the space station orbit")
```

> 💡 **Intuition:** This is not an approximation — it is the exact computation that NB
> performs internally. Each bar in the chart represents exactly how much that feature's
> presence (and count) moved the log-posterior for the predicted class. The sum of all bars
> plus the prior log-probability gives the unnormalized log-posterior.

---

### 8.2 SHAP Values with KernelExplainer

NB does not have a dedicated SHAP explainer (like TreeExplainer for tree models or
LinearExplainer for linear models). The closest built-in SHAP approach is
`shap.LinearExplainer`, which can be applied because NB (for text variants) is formally a
linear model in log-probability space.

```python
import shap
import numpy as np
from sklearn.naive_bayes import MultinomialNB

# ── Method 1: LinearExplainer (fast, exact for linear models) ─────────────────
# MultinomialNB's predict_log_proba is linear: X @ coef_.T + intercept_
# For binary classification, use class 1 log-probability

# Convert sparse to dense for SHAP (feasible for moderate p)
X_train_dense = X_train.toarray()
X_test_dense  = X_test.toarray()

# Background: use a sample of training data (SHAP LinearExplainer needs dense)
background = shap.sample(X_train_dense, 100)

# Wrap model as a probability function for positive class
def predict_proba_pos(X):
    return mnb.predict_proba(X)[:, 1]

explainer = shap.KernelExplainer(predict_proba_pos, background)

# Explain a small test subset (KernelExplainer is slow for large datasets)
X_explain = X_test_dense[:50]
shap_values = explainer.shap_values(X_explain, nsamples=200)

# ── Summary plot (global feature importance) ──────────────────────────────────
feature_names = vec.get_feature_names_out()
shap.summary_plot(shap_values, X_explain, feature_names=feature_names, max_display=20)
```

> ⚠️ **Pitfall:** `KernelExplainer` is model-agnostic and slow — it runs $\mathcal{O}(n_{\text{samples}} \times p)$ model evaluations per explained sample. For text with 20,000 features, explaining 50 samples with `nsamples=200` takes a few minutes. For production explanations, use the built-in feature log-probability approach from §8.1 — it is exact and instant.

```python
# ── Method 2: LinearExplainer (if treating MNB as a linear model directly) ────
# This is an approximation — it treats NB as a standard linear model
# and may not accurately reflect the multiplicative (log-additive) structure

# Only appropriate for binary classification
from sklearn.linear_model import LogisticRegression

# NB coef_ and intercept_ aliases make it compatible with LinearExplainer
explainer_linear = shap.LinearExplainer(
    mnb,
    X_train,
    feature_perturbation="correlation_dependent"  # verify in shap 0.46.x docs
)
shap_values_linear = explainer_linear.shap_values(X_test[:20])
shap.summary_plot(shap_values_linear, X_test[:20].toarray(),
                  feature_names=feature_names, max_display=15)
```

> 🔧 **In practice:** For NB explainability, the built-in `feature_log_prob_` inspection
> (§8.1) is almost always preferable to SHAP. It is exact, instantaneous, and directly
> reflects the model's computation. Use SHAP only when you need a unified comparison with
> other model types in a multi-model explanation framework.

---

### 8.3 LIME (Local Interpretable Model-agnostic Explanations)

LIME fits a local linear model around each prediction using perturbed samples. It is fully
model-agnostic and works well with NB because NB's decision boundary is locally linear.

```python
from lime.lime_text import LimeTextExplainer  # pip install lime

# Define prediction function that returns probability arrays
def predict_fn(texts):
    X = vec.transform(texts)
    return mnb.predict_proba(X)

lime_explainer = LimeTextExplainer(
    class_names=[str(c) for c in mnb.classes_],
    random_state=42,
)

# Explain one document
doc_idx = 5
explanation = lime_explainer.explain_instance(
    test_data.data[doc_idx],
    predict_fn,
    num_features=15,
    num_samples=500,
)

print(f"True class: {test_data.target_names[y_test[doc_idx]]}")
print(f"Predicted:  {test_data.target_names[mnb.predict(X_test[doc_idx])[0]]}")
explanation.show_in_notebook(text=True)  # in Jupyter; use .as_pyplot_figure() otherwise
```

For NB, LIME explanations closely mirror the feature log-probability explanation from §8.1 —
they should agree on the most influential features. If they disagree substantially, it may
indicate non-linearity in the calibration wrapper or a very borderline classification.

---

### 8.4 Global Interpretability: Feature Odds Ratios

For binary classification, you can compute the log-odds ratio of each feature — how much more
(or less) likely is a feature in class 1 versus class 0:

```python
import pandas as pd

# Log-odds ratio: log P(feature | class 1) - log P(feature | class 0)
# Positive: feature favors class 1; negative: favors class 0
log_odds = mnb.feature_log_prob_[1] - mnb.feature_log_prob_[0]
feature_names = vec.get_feature_names_out()

odds_df = pd.DataFrame({
    "feature": feature_names,
    "log_odds": log_odds,
}).sort_values("log_odds", ascending=False)

print("Top features for class 1 (positive log-odds):")
print(odds_df.head(20).to_string(index=False))

print("\nTop features for class 0 (negative log-odds):")
print(odds_df.tail(20).sort_values("log_odds").to_string(index=False))

# ── Visualize ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

top20 = odds_df.head(20)
bot20 = odds_df.tail(20).sort_values("log_odds")

axes[0].barh(top20["feature"][::-1], top20["log_odds"][::-1], color="steelblue")
axes[0].set_title(f"Top 20 features → {test_data.target_names[1]}")
axes[0].set_xlabel("Log-odds ratio")

axes[1].barh(bot20["feature"], bot20["log_odds"], color="tomato")
axes[1].set_title(f"Top 20 features → {test_data.target_names[0]}")
axes[1].set_xlabel("Log-odds ratio")

plt.tight_layout()
plt.show()
```

---

### 8.5 Caveats: When NB Explanations Can Be Misleading

1. **Correlated features:** If "New" and "York" are both high log-probability features for a
   sports class, the explanation double-counts their contribution. The model is not actually
   identifying two independent signals — it is counting the same "New York" twice.

2. **Calibrated wrapper:** After `CalibratedClassifierCV`, `predict_proba` is computed by the
   calibration model (isotonic regression or sigmoid), not the base NB. The `feature_log_prob_`
   attributes still reflect the underlying NB, but SHAP values computed on the calibrated
   probabilities may differ from those computed on raw NB outputs.

3. **Global ≠ local:** The top features from `feature_log_prob_` are *global* averages.
   For a specific document, only its present features matter — use §8.1's per-document
   contribution analysis for local explanations.

---

### 8.6 Information-Theoretic Interpretation of NB Weights

There is a beautiful information-theoretic reading of NB feature weights that deepens the
intuition considerably. For binary classification with MultinomialNB, the log-odds weight:

$$w_{ci} = \log \frac{\hat{\theta}_{1i}}{\hat{\theta}_{0i}} = \log \hat{\theta}_{1i} - \log \hat{\theta}_{0i}$$

is precisely the **pointwise mutual information (PMI)** between feature $i$ and class $c = 1$,
up to a normalization constant:

$$\text{PMI}(x_i; c=1) = \log \frac{P(x_i, c=1)}{P(x_i) P(c=1)}$$

This means NB's "most informative features" — those with highest |$w_{ci}$| — are exactly the
features with the highest mutual information with the class label. You can rank features by
their NB weights as a proxy for feature selection based on mutual information:

```python
import numpy as np
import pandas as pd

def nb_feature_importance(model, feature_names, class_idx=1, top_k=20):
    """
    Extract features ranked by |log-odds ratio| — equivalent to mutual information proxy.
    For binary classification: positive log-odds → feature favors class 1.
    """
    if len(model.classes_) == 2:
        # Binary: log-odds ratio
        log_odds = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        df = pd.DataFrame({
            "feature": feature_names,
            "log_odds": log_odds,
            "abs_log_odds": np.abs(log_odds),
        }).sort_values("abs_log_odds", ascending=False)
    else:
        # Multi-class: max absolute log-prob deviation from mean
        mean_log_prob = model.feature_log_prob_.mean(axis=0)
        max_dev = np.max(np.abs(model.feature_log_prob_ - mean_log_prob), axis=0)
        df = pd.DataFrame({
            "feature": feature_names,
            "max_deviation": max_dev,
        }).sort_values("max_deviation", ascending=False)

    return df.head(top_k)

# Usage with MultinomialNB on text data
importance_df = nb_feature_importance(mnb, vec.get_feature_names_out(), top_k=20)
print("Most informative features (by NB log-odds):")
print(importance_df.to_string(index=False))
```

This is the same computation as NLTK's `show_most_informative_features()`, which explicitly
reports the ratio $P(\text{feature} \mid \text{class 1}) / P(\text{feature} \mid \text{class 0})$
for binary NB classifiers.

> 💡 **Intuition:** The NB weights are directly interpretable as "how much more likely is this
> feature in class 1 vs. class 0?" A log-odds of 3.0 means the feature is $e^3 \approx 20\times$
> more common in class 1 than class 0. A log-odds of $-2.0$ means it is $e^2 \approx 7\times$
> more common in class 0. This is one of the cleanest natural-language interpretations of any
> machine learning model.

---

### 8.7 Permutation Importance for GaussianNB

For GaussianNB on tabular data, `feature_log_prob_` is not available (it stores `theta_` and
`var_` instead). The equivalent interpretability tool is permutation importance:

```python
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np

gnb = GaussianNB()
gnb.fit(X_train, y_train)

result = permutation_importance(
    gnb, X_test, y_test,
    n_repeats=30,
    random_state=42,
    scoring="accuracy",
    n_jobs=-1,
)

# Sort by importance mean
sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots(figsize=(9, 6))
ax.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(feature_names)[sorted_idx],
)
ax.set_xlabel("Accuracy decrease when feature is permuted")
ax.set_title("GaussianNB Permutation Importance")
ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
plt.tight_layout()
plt.show()
```

> ⚠️ **Pitfall:** Permutation importance can be misleading when features are correlated.
> If two features are correlated, permuting one may not decrease accuracy because the other
> still provides the same information. Both features may show low importance even if both are
> individually useful. This is the same problem that affects all permutation-based importance
> methods, not specific to NB.

---

## §9 — Complete Python Worked Examples

Two end-to-end examples: MultinomialNB on 20 Newsgroups (the canonical NB text benchmark),
and GaussianNB on Iris (the continuous-feature baseline showcase).

### 9.1 MultinomialNB on 20 Newsgroups — Full Text Classification Pipeline

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, brier_score_loss, accuracy_score
)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 1. Data Loading ────────────────────────────────────────────────────────────
print("Loading 20 Newsgroups...")

# Use all 20 categories for the full benchmark
train_raw = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes"),
    random_state=42,
)
test_raw = fetch_20newsgroups(
    subset="test",
    remove=("headers", "footers", "quotes"),
    random_state=42,
)

print(f"Train: {len(train_raw.data)} documents, {len(train_raw.target_names)} classes")
print(f"Test:  {len(test_raw.data)} documents")

# EDA: class distribution
class_counts = pd.Series(train_raw.target).value_counts().sort_index()
class_names  = [train_raw.target_names[i] for i in class_counts.index]

fig, ax = plt.subplots(figsize=(12, 5))
ax.barh(class_names, class_counts.values, color="steelblue")
ax.set_xlabel("Number of documents")
ax.set_title("Class Distribution — 20 Newsgroups Train Set")
plt.tight_layout()
plt.show()

# Document length distribution
doc_lengths = [len(doc.split()) for doc in train_raw.data]
print(f"Median document length: {np.median(doc_lengths):.0f} words")
print(f"Mean document length: {np.mean(doc_lengths):.0f} words")

# ── 2. Preprocessing Pipeline ─────────────────────────────────────────────────
# TF-IDF is the standard featurization for MultinomialNB text classification
# sublinear_tf=True: replace TF with 1+log(TF) — reduces heavy hitter bias

pipeline_mnb = Pipeline([
    ("vec", TfidfVectorizer(
        max_features=50_000,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),   # unigrams + bigrams
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z]{2,}\b",  # alphabetic tokens only
    )),
    ("clf", MultinomialNB(alpha=0.1)),
])

pipeline_cnb = Pipeline([
    ("vec", TfidfVectorizer(
        max_features=50_000,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z]{2,}\b",
    )),
    ("clf", ComplementNB(alpha=0.1, norm=True)),
])

# ── 3. Train / Validate ────────────────────────────────────────────────────────
print("\nFitting MultinomialNB pipeline...")
pipeline_mnb.fit(train_raw.data, train_raw.target)

print("Fitting ComplementNB pipeline...")
pipeline_cnb.fit(train_raw.data, train_raw.target)

y_pred_mnb = pipeline_mnb.predict(test_raw.data)
y_pred_cnb = pipeline_cnb.predict(test_raw.data)

acc_mnb = accuracy_score(test_raw.target, y_pred_mnb)
acc_cnb = accuracy_score(test_raw.target, y_pred_cnb)

print(f"\nMultinomialNB accuracy:  {acc_mnb:.4f}")
print(f"ComplementNB accuracy:   {acc_cnb:.4f}")

# ── 4. Hyperparameter Tuning with Optuna ─────────────────────────────────────
print("\nRunning Optuna hyperparameter search for ComplementNB pipeline...")

def objective_cnb(trial):
    alpha      = trial.suggest_float("alpha", 1e-3, 5.0, log=True)
    max_feat   = trial.suggest_int("max_features", 10_000, 100_000, step=10_000)
    min_df     = trial.suggest_int("min_df", 1, 5)
    sublinear  = trial.suggest_categorical("sublinear_tf", [True, False])
    use_bigram = trial.suggest_categorical("use_bigrams", [True, False])
    norm       = trial.suggest_categorical("norm", [True, False])

    pipe = Pipeline([
        ("vec", TfidfVectorizer(
            max_features=max_feat,
            sublinear_tf=sublinear,
            min_df=min_df,
            ngram_range=(1, 2) if use_bigram else (1, 1),
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z]{2,}\b",
        )),
        ("clf", ComplementNB(alpha=alpha, norm=norm)),
    ])
    return cross_val_score(
        pipe, train_raw.data, train_raw.target, cv=3, scoring="accuracy", n_jobs=-1
    ).mean()

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective_cnb, n_trials=30, show_progress_bar=True)

print(f"\nBest params: {study.best_params}")
print(f"Best CV accuracy: {study.best_value:.4f}")

# Refit best model on full training set
best_params = study.best_params
best_pipeline = Pipeline([
    ("vec", TfidfVectorizer(
        max_features=best_params["max_features"],
        sublinear_tf=best_params["sublinear_tf"],
        min_df=best_params["min_df"],
        ngram_range=(1, 2) if best_params["use_bigrams"] else (1, 1),
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z]{2,}\b",
    )),
    ("clf", ComplementNB(alpha=best_params["alpha"], norm=best_params["norm"])),
])
best_pipeline.fit(train_raw.data, train_raw.target)
y_pred_best = best_pipeline.predict(test_raw.data)
print(f"Tuned ComplementNB accuracy: {accuracy_score(test_raw.target, y_pred_best):.4f}")

# ── 5. Diagnostic Plots ────────────────────────────────────────────────────────
# Confusion matrix
cm = confusion_matrix(test_raw.target, y_pred_best)
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=test_raw.target_names,
    yticklabels=test_raw.target_names,
    ax=ax,
)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Confusion Matrix — Tuned ComplementNB (20 Newsgroups)")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

print("\nClassification report (tuned ComplementNB):")
print(classification_report(test_raw.target, y_pred_best, target_names=test_raw.target_names))

# ── 6. Top Features per Class ─────────────────────────────────────────────────
clf = best_pipeline.named_steps["clf"]
vec = best_pipeline.named_steps["vec"]
feature_names = np.array(vec.get_feature_names_out())

# Show top 10 features for a few interesting classes
interesting_classes = [0, 5, 10, 15, 19]   # a sample of classes
fig, axes = plt.subplots(1, len(interesting_classes), figsize=(18, 6), sharey=False)

for ax_idx, class_idx in enumerate(interesting_classes):
    if class_idx >= len(clf.classes_):
        continue
    log_probs = clf.feature_log_prob_[class_idx]
    top10_idx = np.argsort(log_probs)[-10:][::-1]
    ax = axes[ax_idx]
    ax.barh(feature_names[top10_idx][::-1], log_probs[top10_idx][::-1], color="steelblue")
    ax.set_title(f"{test_raw.target_names[class_idx][:20]}", fontsize=9)
    ax.set_xlabel("log P(f|c)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)

plt.suptitle("Top 10 Features per Class", fontsize=12)
plt.tight_layout()
plt.show()
```

---

### 9.2 GaussianNB on Iris — Continuous Features with Full Diagnostics

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
)
from sklearn.preprocessing import LabelEncoder

# ── 1. Load and Explore ────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names   = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["species"] = [class_names[i] for i in y]

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes")
print(df.groupby("species").describe().round(2).to_string())

# ── Check Gaussianity ────────────────────────────────────────────────────────
print("\nShapiro-Wilk normality test per feature per class:")
for class_idx, class_name in enumerate(class_names):
    for feat_idx, feat_name in enumerate(feature_names):
        x_class = X[y == class_idx, feat_idx]
        stat, p = shapiro(x_class)
        status = "OK" if p >= 0.05 else "NON-GAUSSIAN"
        print(f"  {class_name} / {feat_name}: p={p:.3f} [{status}]")

# ── Pairplot: feature distributions ──────────────────────────────────────────
g = sns.pairplot(df, hue="species", diag_kind="kde", plot_kws={"alpha": 0.6})
g.fig.suptitle("Iris Dataset — Feature Distributions by Class", y=1.02)
plt.tight_layout()
plt.show()

# ── 2. Train / Test Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Fit and Evaluate ────────────────────────────────────────────────────────
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print(f"\nGaussianNB accuracy: {gnb.score(X_test, y_test):.4f}")
print(f"Cross-validated (5-fold): {cross_val_score(gnb, X, y, cv=5).mean():.4f} ± "
      f"{cross_val_score(gnb, X, y, cv=5).std():.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ── 4. Tune var_smoothing ─────────────────────────────────────────────────────
param_grid = {"var_smoothing": np.logspace(-9, -1, 100)}
gs = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
gs.fit(X_train, y_train)
print(f"\nBest var_smoothing: {gs.best_params_['var_smoothing']:.2e}")
print(f"Best CV accuracy:   {gs.best_score_:.4f}")

gnb_tuned = gs.best_estimator_
y_pred_tuned = gnb_tuned.predict(X_test)

# ── 5. Fitted Parameters ───────────────────────────────────────────────────────
print("\nLearned class priors:", gnb_tuned.class_prior_.round(4))
print("\nPer-class means (theta_):")
print(pd.DataFrame(gnb_tuned.theta_, index=class_names, columns=feature_names).round(3))
print("\nPer-class variances (var_):")
print(pd.DataFrame(gnb_tuned.var_,   index=class_names, columns=feature_names).round(3))

# ── 6. Decision Boundary Visualization (2D slice) ─────────────────────────────
# Use first two features for 2D visualization
feat_x, feat_y = 0, 1   # sepal length, sepal width
h = 0.02  # mesh step size

x_min = X[:, feat_x].min() - 0.5
x_max = X[:, feat_x].max() + 0.5
y_min = X[:, feat_y].min() - 0.5
y_max = X[:, feat_y].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Fix other features at their mean values
X_mesh = np.tile(X.mean(axis=0), (xx.ravel().shape[0], 1))
X_mesh[:, feat_x] = xx.ravel()
X_mesh[:, feat_y] = yy.ravel()

Z = gnb_tuned.predict(X_mesh).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(9, 7))
ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
scatter = ax.scatter(
    X[:, feat_x], X[:, feat_y], c=y,
    cmap="RdYlBu", edgecolors="black", linewidths=0.5, s=50
)
ax.set_xlabel(feature_names[feat_x])
ax.set_ylabel(feature_names[feat_y])
ax.set_title("GaussianNB Decision Boundary (2D projection, other features at mean)")
handles, _ = scatter.legend_elements()
ax.legend(handles, class_names)
plt.tight_layout()
plt.show()

# ── 7. Calibration ────────────────────────────────────────────────────────────
# GaussianNB is often better calibrated than MultinomialNB, but still check
cal_gnb = CalibratedClassifierCV(GaussianNB(var_smoothing=gs.best_params_["var_smoothing"]),
                                 method="isotonic", cv=5)
cal_gnb.fit(X_train, y_train)

# For multi-class calibration, plot per-class (one-vs-rest)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (class_name, ax) in enumerate(zip(class_names, axes)):
    y_test_bin = (y_test == i).astype(int)

    for model, name, color in [
        (gnb_tuned, "Uncalibrated GNB", "tab:blue"),
        (cal_gnb,   "Isotonic calibrated", "tab:orange"),
    ]:
        proba = model.predict_proba(X_test)[:, i]
        CalibrationDisplay.from_predictions(
            y_test_bin, proba, n_bins=8, ax=ax, name=name, color=color
        )
    ax.set_title(f"Calibration: {class_name}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)

plt.suptitle("Calibration Curves — GaussianNB (Iris, one-vs-rest per class)", fontsize=12)
plt.tight_layout()
plt.show()

print("\nGaussianNB on Iris — Worked example complete.")
print(f"Final test accuracy (tuned): {accuracy_score(y_test, y_pred_tuned):.4f}")
```

---

## §10 — When to Use This Algorithm (Decision Guide)

### The Core Decision Flowchart

```
Start here: What are your features?
│
├─ Continuous real-valued features
│   └─ Are they approximately Gaussian per class?
│       ├─ YES: GaussianNB — fast baseline before SVM or LR
│       └─ NO (skewed, heavy-tailed): log-transform first, or use pomegranate
│                                      with a different distribution
│
├─ Text / word count / TF-IDF features (non-negative counts or frequencies)
│   ├─ Are classes balanced?
│   │   ├─ YES: MultinomialNB(alpha=0.1) + TfidfVectorizer(sublinear_tf=True)
│   │   └─ NO (imbalanced): ComplementNB(alpha=0.1, norm=True) — always prefer
│   │
│   ├─ Short texts (tweets, headlines) or small vocabulary (<5000 words)?
│   │   └─ BernoulliNB(alpha=0.1) — word presence/absence matters more
│   │
│   └─ Feature correlations are strong (e.g., bigrams matter)?
│       └─ Add ngram_range=(1,2) or switch to LogisticRegression
│
├─ Binary presence/absence features (booleans, one-hot encoded)
│   └─ BernoulliNB(binarize=None)
│
└─ Nominal categorical features (country, product category, blood type)
    └─ CategoricalNB with OrdinalEncoder preprocessing
```

### The Quick Decision Table

| Scenario | Recommended Variant | Notes |
|---|---|---|
| Text classification, balanced classes | `MultinomialNB(alpha=0.1)` | Workhorse for NLP |
| Text classification, imbalanced classes | `ComplementNB(alpha=0.1, norm=True)` | Almost always better than MNB |
| Short text, binary vocabulary | `BernoulliNB(alpha=0.1)` | Absence of words is signal |
| Continuous tabular features | `GaussianNB(var_smoothing=1e-9)` | Fast baseline |
| Pure nominal categorical features | `CategoricalNB + OrdinalEncoder` | Don't use for ordinal data |
| Need calibrated probabilities | Any NB + `CalibratedClassifierCV` | Always calibrate |
| Streaming / online learning | `MultinomialNB.partial_fit` + `HashingVectorizer` | True online learning |
| Mixed feature types per feature | `pomegranate.NaiveBayes` | GPU + mixed distributions |
| Strong feature dependencies known | `pgmpy` TAN | Relaxes independence |

### When NOT to Use Naive Bayes

| Signal | Why NB Fails | Better Choice |
|---|---|---|
| Features are highly correlated (many feature pairs with \|corr\| > 0.7) | Double-counts correlated evidence; probability estimates are meaningless | Logistic Regression, which handles correlations |
| You need reliable probability scores (Brier score < 0.1 on test set) | NB probabilities are overconfident even after calibration if correlations are severe | Gradient Boosted Trees with `calibrate_estimator=True` |
| Feature interactions are the main signal (negation, context-dependence) | NB cannot learn "not good" ≠ "good" | Gradient Boosting, Neural Networks |
| $n$ is large (millions) and $p$ is moderate (hundreds) | NB is always fast, but discriminative models (LR, GBM) will outperform it here | Logistic Regression, LightGBM |
| Continuous features with non-Gaussian distributions and heavy tails | GaussianNB likelihood is wrong | Random Forest, GBM |
| Multi-label classification | NB does not natively support multi-label | `ClassifierChain(MultinomialNB())` from sklearn |

### Where NB Definitively Wins

1. **First text classification baseline:** Faster and less buggy to set up than anything else.
   NB + TF-IDF is the "sanity check" that any more complex model must beat.

2. **Streaming text with millions of documents:** HashingVectorizer + MultinomialNB.partial_fit
   is the only truly scalable online text classifier in the sklearn ecosystem.

3. **Cold start / few-shot classification:** With 50 training examples per class, NB is more
   reliable than logistic regression (which may not converge) or GBM (which may overfit).

4. **Spam / fraud detection with known base rates:** Setting `class_prior` to the real-world
   base rate and using ComplementNB gives a strong, interpretable production classifier.

5. **When you need to explain every prediction in plain language:** NB's feature log-probability
   attribution is exact, instant, and human-readable. No post-hoc approximation needed.

---

### Checklist Before Deploying NB to Production

Before shipping a Naive Bayes classifier, run through this checklist:

**Data preconditions:**
- [ ] Features match the variant's requirements (non-negative for MNB, integers for CatNB, etc.)
- [ ] No data leakage: features were computed on the training split, not the full dataset
- [ ] `partial_fit` first call includes `classes=all_possible_classes`
- [ ] For text pipelines: `HashingVectorizer` used instead of `TfidfVectorizer` if online learning

**Modeling:**
- [ ] `alpha` tuned on a held-out validation set (not left at default 1.0)
- [ ] `class_prior` set to match deployment base rate (not training set frequency)
- [ ] Calibration applied: `CalibratedClassifierCV` with isotonic or sigmoid regression
- [ ] Calibration curve plotted and verified to be near-diagonal

**Evaluation:**
- [ ] Calibration curve plotted (§7.1)
- [ ] Confusion matrix inspected for systematic class confusions (§7.2)
- [ ] Top features per class reviewed for sanity (§7.3) — check for data leakage signals
- [ ] Learning curve confirms validation accuracy is not plateaued at low n (§7.4)
- [ ] AP (average precision) computed if classes are imbalanced (§7.7)

**Monitoring:**
- [ ] Model retraining cadence defined (weekly, monthly, or triggered by accuracy drop)
- [ ] Drift monitoring in place: track input feature distribution over time
- [ ] Fallback defined if model degrades (rule-based backup, prior-only model)

---

## §11 — Resources & Further Reading

### The Original Papers (Essential)

1. **Domingos, P., & Pazzani, M. (1997).** "On the Optimality of the Simple Bayesian Classifier
   under Zero-One Loss." *Machine Learning, 29*(2–3), 103–130.
   URL: https://gwern.net/doc/ai/1997-domingos.pdf
   *Why read it:* The definitive theoretical explanation of why NB works despite the violated
   independence assumption. Answers the question every practitioner asks.

2. **McCallum, A., & Nigam, K. (1998).** "A Comparison of Event Models for Naive Bayes Text
   Classification." *AAAI-98 Workshop on Learning for Text Categorization.*
   URL: https://www.cs.cmu.edu/~knigam/papers/multinomial-aaai98.pdf
   *Why read it:* The paper that distinguished MultinomialNB from BernoulliNB. Essential for
   understanding which text NB variant to use.

3. **Rennie, J. D. M., Shih, L., Teevan, J., & Karger, D. R. (2003).** "Tackling the Poor
   Assumptions of Naive Bayes Text Classifiers." *ICML 2003, Vol. 3*, 616–623.
   URL: https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
   *Why read it:* The ComplementNB paper. Explains the weight-skew problem and its fix.

4. **Friedman, N., Geiger, D., & Goldszmidt, M. (1997).** "Bayesian Network Classifiers."
   *Machine Learning, 29*, 131–163. DOI: 10.1023/A:1007465528199
   URL: https://link.springer.com/article/10.1023/A:1007465528199
   *Why read it:* The TAN paper. Shows how to relax the independence assumption tractably.

5. **Rish, I. (2001).** "An Empirical Study of the Naive Bayes Classifier." *IJCAI 2001
   Workshop on Empirical Methods in AI.*
   URL: https://faculty.cc.gatech.edu/~isbell/reading/papers/Rish.pdf
   *Why read it:* Systematic analysis of when NB succeeds and fails on real datasets.

6. **Webb, G. I., Boughton, J. R., & Wang, Z. (2005).** "Not So Naive Bayes: Aggregating
   One-Dependence Estimators." *Machine Learning, 58*, 5–24.
   URL: https://link.springer.com/article/10.1007/s10994-005-4258-6
   *Why read it:* The AODE paper — best performing semi-naive Bayes variant.

7. **Maron, M. E. (1961).** "Automatic Indexing: An Experimental Inquiry." *Journal of the
   ACM, 8*(3), 404–417. DOI: 10.1145/321075.321084
   *Why read it:* Historical origin — one of the earliest probabilistic document classifiers.

### Official Documentation

8. **sklearn Naive Bayes User Guide (1.5)**
   URL: https://scikit-learn.org/1.5/modules/naive_bayes.html
   *Why read it:* Verified formulas for all five variants, `partial_fit` behavior, known
   limitations, and code examples.

9. **sklearn Probability Calibration Guide**
   URL: https://scikit-learn.org/stable/modules/calibration.html
   *Why read it:* Explains exactly why NB is miscalibrated and how `CalibratedClassifierCV`
   fixes it. Includes Brier score comparisons and calibration curve interpretation.

10. **sklearn Text Feature Extraction + Grid Search Example (1.5)**
    URL: https://scikit-learn.org/1.5/auto_examples/model_selection/plot_grid_search_text_feature_extraction.html
    *Why read it:* Complete production-ready text classification pipeline with grid search,
    combining TfidfVectorizer hyperparameters with NB hyperparameters.

### Books (Specific Chapters)

11. **Mitchell, T. (1997).** *Machine Learning.* McGraw-Hill. **Chapter 6 (Bayesian Learning).**
    URL: http://www.cs.cmu.edu/~tom/mlbook.html
    *Why read it:* Classic textbook treatment of Bayesian classifiers. Derives the naive Bayes
    algorithm from first principles with clear worked examples.

12. **Bishop, C. (2006).** *Pattern Recognition and Machine Learning.* Springer. **Section 4.2
    (Probabilistic generative models).**
    URL: https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/
    *Why read it:* Rigorous treatment of generative vs. discriminative classifiers. Places NB
    in the context of Gaussian generative models, LDA, and logistic regression.

13. **VanderPlas, J. (2016).** *Python Data Science Handbook.* O'Reilly. **Chapter 5.5
    (In-Depth: Naive Bayes Classification).**
    URL: https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html
    *Why read it:* Intuitive treatment with decision boundary visualizations. Excellent for
    building geometric intuition before the algebraic derivation.

14. **Jurafsky, D., & Martin, J. H. (2023).** *Speech and Language Processing* (3rd ed.).
    **Chapter 4 (Naive Bayes, Text Classification, and Sentiment).**
    URL: https://web.stanford.edu/~jurafsky/slp3/ (free PDF)
    *Why read it:* The standard NLP textbook treatment. Excellent coverage of bag-of-words,
    Laplace smoothing, and the sentiment analysis use case.

### Tutorials and Blog Posts

15. **Graham, P. (2002).** "A Plan for Spam."
    URL: https://paulgraham.com/spam.html
    *Why read it:* The essay that brought Bayesian spam filtering to the world. Brilliant
    motivation for why probabilistic methods beat hand-crafted rules.

16. **VanderPlas, J.** "In Depth: Naive Bayes Classification" (Python Data Science Handbook).
    URL: https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html
    *Why read it:* Best visual explanation of GaussianNB decision boundaries.

17. **sklearn Calibration Curve Example**
    URL: https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration.html
    *Why read it:* Visual demonstration of NB miscalibration with Brier score comparisons.

18. **NLTK Classify HOWTO**
    URL: https://www.nltk.org/howto/classify.html
    *Why read it:* Reference for NLTK's `NaiveBayesClassifier` API, including
    `show_most_informative_features`.

### Package Documentation

19. **sklearn GaussianNB API Reference (1.5)**
    URL: https://scikit-learn.org/1.5/modules/generated/sklearn.naive_bayes.GaussianNB.html

20. **sklearn MultinomialNB API Reference (1.5)**
    URL: https://scikit-learn.org/1.5/modules/generated/sklearn.naive_bayes.MultinomialNB.html

21. **sklearn ComplementNB API Reference (1.5)**
    URL: https://scikit-learn.org/1.5/modules/generated/sklearn.naive_bayes.ComplementNB.html

22. **river NB Documentation**
    URL: https://riverml.xyz/latest/api/naive-bayes/GaussianNB/
    *Why read it:* Reference for true online (single-sample) NB implementation.

23. **pomegranate Documentation**
    URL: https://pomegranate.readthedocs.io/en/latest/
    *Why read it:* GPU-accelerated NB with mixed per-feature distributions.

---

*End of chapter. Estimated word count: ~14,600 words (~19,500 tokens).*
*Datasets used: 20 Newsgroups (sklearn.datasets.fetch_20newsgroups), Iris (sklearn.datasets.load_iris)*

*API uncertainties flagged:*
- *`CategoricalNB.partial_fit` — listed in sklearn 1.5 API but limited documentation on recommended usage; verify in sklearn changelog.*
- *`predict_joint_log_proba` — added in sklearn 1.2; confirm availability in your specific 1.5.x patch version.*
- *`ComplementNB norm=True` performance vs. `norm=False` — sklearn docs are ambiguous; original paper favors `norm=True`.*
- *`force_alpha` default change from `False` (< sklearn 1.4) to `True` (≥ 1.4) — may affect behavior if upgrading from older sklearn.*
- *`pgmpy TreeSearch(estimator_type="tan")` — API has been refactored in recent pgmpy releases; verify against current pgmpy docs before use.*
- *`shap.LinearExplainer` with `feature_perturbation="correlation_dependent"` — verify parameter name in shap 0.46.x docs.*





