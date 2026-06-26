# COURSE BIBLE — Supervised Learning Masterclass (internal authoring guide)

> Every writing subagent MUST read this file first, then the assigned research dossier(s),
> then write its chapter. This guarantees voice, structure, Python-stack, and notation coherence.

---

## 1. What this course is

A deep, expert-level masterclass on supervised learning algorithms. Each algorithm gets its own
standalone Markdown file. The goal: after reading one file, the reader is a genuine expert on
that algorithm — theory, history, all major variants, every hyperparameter, diagnostics,
explainability, code, strengths/weaknesses, and the best modern packages.

The audience: a strong Python ML practitioner who wants to go beyond "call fit()" to truly
understanding what the algorithm does, why it works, where it fails, and how to squeeze every
drop of performance and interpretability out of it.

## 2. The teacher persona & voice

Write like a **world-class ML researcher and practitioner mentoring a talented senior data
scientist**. The voice blends:
- **Deep theoretical rigor**: derive things, explain the math, don't hand-wave. Show equations.
- **Historical narrative**: start from the original paper, trace the lineage of ideas, show
  how each innovation solved a real limitation of what came before.
- **Brutal practicality**: every idea must cash out in code, in a hyperparameter you tune, in
  a diagnostic you read, in a decision you make in the real world.
- **Honest assessment**: name the weaknesses clearly. Don't hype. Explain *when* to use and
  *when not to* use the algorithm. Practitioners trust honesty.

Concrete voice rules:
- Open with a motivating **"Why this algorithm matters"** — a crisp story of the problem it
  solves, with a real number or a famous application.
- Use callouts (see §6) liberally.
- Always explain the WHY behind every design choice. "Breiman added bootstrapping because..."
- Talk *to* the reader: "you", "we", "notice that", "sit with this for a moment".
- Be verbose and generous — 20,000+ tokens is the target. Depth, never filler.

## 3. Required file structure (every algorithm file MUST have all of these)

### Section 0 — Python Ecosystem & Package Guide (ALWAYS FIRST)
A comprehensive table + prose covering:
- ALL major Python packages implementing this algorithm (sklearn, xgboost, lightgbm, catboost,
  cuml, river, vowpalwabbit, h2o, etc. — whatever is relevant to this specific algorithm)
- For each: the exact import, what algorithm variant it uses internally, key strengths, who
  maintains it, latest stable version (as of mid-2026), license
- A **"Top 3-5 picks"** recommendation table with: package, best for (use case), modern/legacy
- Which are GPU-accelerated, which support streaming/online learning, which have the best docs
- Code snippet showing the same fit() call across the top packages for direct comparison

### Section 1 — The Origin: The Paper That Started It All
- The original paper: full citation (author, title, venue, year, DOI/URL)
- The problem the author was trying to solve — the limitation of prior art
- The core insight/innovation in the paper, explained deeply
- The original algorithm, stated formally (pseudocode or numbered steps)
- Key equations from the paper, derived or at least explained
- What was experimental/uncertain when it was published
- Historical context: what was the state of the art before this?

### Section 2 — The Algorithm, Deeply Explained
- The mathematical foundations (derive it properly)
- Intuition: what is the algorithm actually doing? Multiple analogies.
- The exact prediction formula
- How training works (optimization objective, how the parameters are found)
- Computational complexity: training O(?) and inference O(?)
- What the algorithm assumes about the data (the implicit assumptions)
- Where these assumptions can break

### Section 3 — The Full Evolution (one subsection per major variant/advancement)
For EACH significant variant (e.g., for Random Forests: Extremely Randomized Trees, Isolation
Forest, Oblique RF, Mondrian Forests, Deep Forests; for GBM: AdaBoost → MART → XGBoost →
LightGBM → CatBoost → NGBoost → LightAutoML, etc.):
- What problem of the original it solves
- The paper/origin (citation + year)
- The key algorithmic change (show the equation or pseudocode that changed)
- When to prefer this variant over the base
- Python package implementing it

### Section 4 — Hyperparameters: The Complete Guide
For EVERY hyperparameter:
- What it controls (the actual mechanism)
- Its default value and what the default implies
- How it interacts with other hyperparameters
- How to tune it: the recommended strategy, the range, the heuristic
- What happens if it's too high / too low
- A "tuning playbook" table at the end: hyperparameter → typical range → effect → how to tune

### Section 5 — Strengths
Concrete, specific, with the mechanism behind each strength (not marketing bullet points).
Include performance benchmarks or papers where relevant.

### Section 6 — Weaknesses & Failure Modes
Concrete, specific, with examples of the kinds of data/problems where this algorithm fails.
For each weakness: the mechanism, how to detect it, and the mitigation.

### Section 7 — Diagnostic Plots & Evaluation
For both classification and regression use cases:
- Which metrics to optimize (and why, not just which ones exist)
- The key diagnostic plots (learning curves, validation curves, feature importances, partial
  dependence plots, residual plots, calibration curves, etc.)
- For EACH plot: what it shows, how to read it, what "good" vs "bad" looks like
- Full Python code generating each plot (using matplotlib, sklearn, shap, yellowbrick, or
  whatever is appropriate)
- Out-of-bag error where applicable

### Section 8 — Explainability & Interpretable AI
- What built-in interpretability the algorithm provides (if any)
- SHAP values: the specific SHAP explainer for this algorithm (TreeExplainer, KernelExplainer,
  etc.), full code, how to read the plots (summary, waterfall, force, beeswarm, dependence)
- LIME: applicability and code
- Permutation importance: when to prefer it over built-in importance
- Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) plots
- Global vs local explanations
- For tree-based models: decision path, treeinterpreter, etc.
- Caveats: when explanations can be misleading

### Section 9 — Complete Python Worked Example
A full end-to-end example on a NAMED, REAL dataset (from sklearn.datasets, seaborn, UCI, or
Kaggle). The example must include:
- Data loading + brief EDA
- Preprocessing pipeline (the right one for this algorithm)
- Train/val/test split with stratification where appropriate
- The model, fit, and predict
- Hyperparameter tuning (GridSearch or Optuna or RandomSearch — show the code)
- All diagnostic plots from Section 7
- SHAP explanations from Section 8
- Final interpretation of results in plain English

### Section 10 — When to Use This Algorithm (Decision Guide)
A concrete flowchart/table: given your data characteristics (size, dimensionality, sparsity,
missing values, categorical features, need for interpretability, need for speed, etc.), should
you reach for this algorithm? When is it the right tool? When is it definitively wrong?

### Section 11 — Resources & Further Reading
- The original paper (with full citation and real URL)
- The 5-10 most important follow-up papers (with citations + URLs)
- The best books that cover this algorithm (specific chapters)
- The best blog posts / tutorials (real URLs only — from known authors: Towards Data Science,
  fast.ai, distill.pub, official package docs, etc.)
- Online courses where this is taught well
- The package documentation pages (direct links)

## 4. Python stack & version policy

As of mid-2026, use these canonical import styles:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             mean_squared_error, r2_score)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import shap
import optuna
```

For each algorithm, add the algorithm-specific packages (xgboost, lightgbm, catboost, etc.).

Scikit-learn: ~1.5.x. XGBoost: ~2.x. LightGBM: ~4.x. CatBoost: ~1.2.x. SHAP: ~0.46.x.
Optuna: ~3.x.

Always show the canonical sklearn version first, then the specialized packages.
Flag anything version-uncertain with `# verify in current docs`.

## 5. Notation conventions

- Use LaTeX `$...$` for inline math, `$$...$$` for display math (GitHub renders it)
- $n$ = number of samples, $p$ = number of features, $k$ = number of classes/trees/neighbors
- $\mathbf{x}_i$ = feature vector of sample $i$, $y_i$ = label
- $\hat{y}$ = prediction, $f(\mathbf{x})$ = model function
- Bold for vectors/matrices: $\mathbf{X}$, $\mathbf{w}$
- Loss function: $L(y, \hat{y})$; empirical risk: $\hat{R}$

## 6. Callout convention

- `> 💡 **Intuition:**` — the mental model, the analogy
- `> 🔧 **In practice:**` — practical engineering guidance, code tip
- `> ⚠️ **Pitfall:**` — the trap practitioners fall into
- `> 📜 **Origin/Citation:**` — where an idea comes from
- `> 🏆 **Best practice:**` — the gold-standard recommendation
- `> 🔬 **Deep dive:**` — an optional detailed derivation or tangent
- `> 📊 **Benchmark:**` — a real performance comparison or benchmark number

## 7. The ten algorithms & their files

| File | Algorithm | Sklearn estimator(s) |
|---|---|---|
| `random-forests.md` | Random Forests | `RandomForestClassifier/Regressor`, `ExtraTreesClassifier/Regressor` |
| `gradient-boosting.md` | Gradient Boosting Machines | `GradientBoostingClassifier`, `HistGradientBoosting*`, XGBoost, LightGBM, CatBoost |
| `support-vector-machines.md` | Support Vector Machines | `SVC`, `SVR`, `LinearSVC`, `NuSVC` |
| `k-nearest-neighbors.md` | K-Nearest Neighbors | `KNeighborsClassifier/Regressor` |
| `logistic-regression.md` | Logistic Regression | `LogisticRegression`, `SGDClassifier` |
| `decision-trees.md` | Decision Trees | `DecisionTreeClassifier/Regressor` |
| `linear-models.md` | Linear Models (Ridge/Lasso/ElasticNet) | `Ridge`, `Lasso`, `ElasticNet`, `LinearRegression`, `BayesianRidge` |
| `naive-bayes.md` | Naive Bayes | `GaussianNB`, `MultinomialNB`, `BernoulliNB`, `ComplementNB` |
| `extra-trees.md` | Extra Trees & Rotation Forests | `ExtraTreesClassifier/Regressor`, `BaggingClassifier` |
| `mlp-neural-networks.md` | MLP Neural Networks | `MLPClassifier/Regressor` |

## 8. Word count target: 20,000+ tokens per file

Write incrementally: Write the title + §0 + §1, then Edit/append §2, then §3, etc. until
complete. Do NOT stop at a partial chapter. The reader should finish the file knowing
everything they need to be an expert on this algorithm.

## 9. Output discipline for writing agents

Write ONLY the Markdown to the file. Return a short summary: file path, approx word count,
sections covered, datasets used, any uncertainties flagged. Do NOT echo the chapter text back.
