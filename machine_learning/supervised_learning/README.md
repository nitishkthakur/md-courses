# Supervised Learning Masterclass

A deep-dive course covering the core supervised learning algorithms used in production data science and machine learning engineering. Each file pairs rigorous mathematical foundations with practical sklearn-first code, covering both classical intuition and modern implementation details. The course targets practitioners who want to understand *why* an algorithm works, not just call `.fit()`.

---

## Course Index

| File | Algorithm | Description |
|------|-----------|-------------|
| [decision-trees.md](decision-trees.md) | Decision Trees | CART/ID3/C4.5 split criteria, pruning strategies, and tree visualization |
| [random-forests.md](random-forests.md) | Random Forests | Breiman 2001 ensemble of trees with bagging and random feature selection |
| [extra-trees.md](extra-trees.md) | Extra Trees | Extremely randomized split thresholds and head-to-head comparison with Random Forests |
| [gradient-boosting.md](gradient-boosting.md) | Gradient Boosting Machines | Sequential ensemble lineage: AdaBoost → MART → XGBoost → LightGBM → CatBoost |
| [linear-models.md](linear-models.md) | Linear Models | OLS, Ridge, Lasso, ElasticNet, and BayesianRidge — regularization paths and solver selection |
| [logistic-regression.md](logistic-regression.md) | Logistic Regression | Log-odds linear model, 6 solvers, regularization strategies, and probability calibration |
| [support-vector-machines.md](support-vector-machines.md) | Support Vector Machines | Maximum-margin hyperplane, kernel trick, and support vector regression (SVR) |
| [k-nearest-neighbors.md](k-nearest-neighbors.md) | K-Nearest Neighbors | Instance-based learning, distance metrics, KD-tree/ball-tree indexing, and approximate NN |
| [naive-bayes.md](naive-bayes.md) | Naive Bayes | Gaussian/Multinomial/Bernoulli/Complement variants, Bayes theorem, and Laplace smoothing |
| [mlp-neural-networks.md](mlp-neural-networks.md) | MLP Neural Networks | Backpropagation, activation functions, optimizers, and sklearn vs PyTorch tradeoffs |

---

## How to Use This Course

Each file is a self-contained reference — read them in any order based on what you need. There are no prerequisites between files; a chapter on SVMs does not assume you have read the chapter on logistic regression. Jump directly to the algorithm you are working with, use the table of contents within each file to navigate to a specific concept (theory, implementation, tuning, or diagnostics), and return when the next algorithm comes up in your work.

If you are new to supervised learning entirely, a natural reading order is: Decision Trees → Random Forests → Gradient Boosting → Logistic Regression → Linear Models → then any remaining chapters in any sequence.

---

## Python Stack

All code in this course targets the following library versions, reflecting the mid-2026 ecosystem:

| Library | Version |
|---------|---------|
| scikit-learn | 1.5.x |
| XGBoost | 2.x |
| LightGBM | 4.x |
| CatBoost | 1.2.x |
| SHAP | 0.46.x |
| Optuna | 3.x |

Code is written for Python 3.11+ and assumes a standard scientific Python environment (NumPy, pandas, matplotlib). Where PyTorch appears (MLP chapter), it is treated as an optional comparison rather than a core dependency.
