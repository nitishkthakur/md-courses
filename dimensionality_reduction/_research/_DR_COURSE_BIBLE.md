# COURSE BIBLE — Dimensionality Reduction Masterclass (internal authoring guide)

> Every writing subagent MUST read this file first, then the assigned research dossier,
> then write its chapter. This guarantees voice, structure, Python-stack, and notation coherence.

---

## 1. What this course is

A deep, expert-level masterclass on dimensionality reduction algorithms. Each algorithm (or
closely-related family) gets its own standalone Markdown file. The goal: after reading one file,
the reader is a genuine expert on that algorithm — theory, history, all major variants, every
hyperparameter, diagnostics, real-world applications (including innovative industry applications
found via current research), code, strengths/weaknesses, and the best modern packages.

Audience: a strong Python ML practitioner who wants to go beyond `fit_transform()` to truly
understand what each DR algorithm does geometrically, why it works, where it fails, how to tune
it, and how to choose between algorithms.

---

## 2. The teacher persona & voice

Write like a **world-class ML researcher and practitioner mentoring a talented senior data
scientist**:
- **Deep theoretical rigor**: derive things, show the math, don't hand-wave. Show equations.
- **Historical narrative**: start from the original paper, trace the lineage of ideas.
- **Brutal practicality**: every idea must cash out in code, a hyperparameter, a diagnostic, a
  decision in the real world.
- **Honest assessment**: name the weaknesses clearly. Explain when to use and when NOT to use.

Voice rules:
- Open with "Why this algorithm matters" — a crisp story with a real number or famous application.
- Use callouts (see §6) liberally.
- Always explain the WHY behind every design choice.
- Talk TO the reader: "you", "we", "notice that", "sit with this for a moment".
- Be verbose and generous — 18,000+ tokens is the target. Depth, never filler.

---

## 3. Required file structure (every algorithm file MUST have all of these)

### §0 — Python Ecosystem & Package Guide (ALWAYS FIRST)
- ALL major Python packages for this algorithm
- For each: exact import, what variant it uses internally, key strengths, maintainer, version
  (mid-2026), license
- A **Top 3-5 picks** recommendation table: package → best for → modern/legacy
- GPU-accelerated options, streaming options
- Code snippet showing the same fit_transform() call across top packages for direct comparison

### §1 — The Origin: The Paper That Started It All
- Full citation (author, title, venue, year, DOI/URL)
- The problem the author was solving — limitation of prior art
- The core insight/innovation, explained deeply
- Original algorithm, stated formally (pseudocode or numbered steps)
- Key equations derived or explained
- Historical context: what came before?

### §2 — The Algorithm, Deeply Explained
- Mathematical foundations (derive properly)
- Geometric intuition: what is the algorithm doing in the data space?
- Multiple analogies
- The exact transform formula
- How fitting works (optimization objective, how parameters are found)
- Computational complexity: fitting O(?) and transform O(?)
- Implicit assumptions about the data
- Where these assumptions break

### §3 — The Full Evolution (one subsection per major variant/advancement)
For EACH significant variant:
- What problem of the original it solves
- Paper/origin (citation + year)
- Key algorithmic change (equation or pseudocode)
- When to prefer this variant
- Python package implementing it

### §4 — Hyperparameters: The Complete Guide
For EVERY hyperparameter:
- What it controls (the actual mechanism)
- Default value and what the default implies
- How it interacts with other hyperparameters
- Tuning strategy: range, heuristic, search space
- What happens if too high / too low
- A **Tuning Playbook** table at the end

### §5 — Strengths
Concrete, specific, with the mechanism behind each strength.

### §6 — Weaknesses & Failure Modes
Concrete, with examples of data/problems where this algorithm fails.
For each: mechanism, how to detect, mitigation.

### §7 — What I Think About When Fitting
**THIS IS A KEY SECTION.** Write this as a practitioner's mental checklist:
- Before fitting: what do I check about the data?
- During fitting: what signals do I monitor?
- After fitting: what diagnostics do I run?
- Decision tree: if I see X in the output, I do Y
- Common "smell tests" for whether the embedding is meaningful
This section should read like the internal monologue of an expert.

### §8 — Diagnostic Plots & Evaluation
- How to evaluate a dimensionality reduction (reconstruction error, trustworthiness,
  continuity, neighborhood preservation, intrinsic dimensionality)
- The key diagnostic plots for this specific algorithm
- For EACH plot: what it shows, how to read it, what "good" vs "bad" looks like
- Full Python code generating each plot
- Scree plots, explained variance, perplexity sensitivity plots, etc. — whatever is relevant

### §9 — Innovative Industry Applications
**THIS IS A KEY SECTION.** Research and document:
- 5-8 real-world applications of this algorithm in industry
- Include at least 2-3 "innovative" or non-obvious applications (not just "data visualization")
- For each: the domain, what problem it solved, why THIS algorithm was the right choice
- Real company examples where possible (from papers, blog posts, public case studies)
- Code sketch showing how you'd set it up for each application type

### §10 — Complete Python Worked Example
A full end-to-end example on a NAMED, REAL dataset (sklearn.datasets, seaborn, UCI, or Kaggle):
- Data loading + brief EDA (what is the dimensionality problem here?)
- Preprocessing pipeline
- Fitting the DR algorithm
- Hyperparameter tuning (show the code — Optuna or grid search)
- All diagnostic plots from §8
- Visualization of the embedding (2D or 3D scatter, colored by label/cluster/feature)
- Interpretation of results in plain English: what did we learn from the embedding?

### §11 — When to Use This Algorithm (Decision Guide)
- Concrete flowchart/table: given your problem characteristics, should you use this?
- Comparison with the closest alternatives: when does algorithm A beat algorithm B?
- "Red flags" that mean you should NOT use this algorithm

### §12 — Top Papers to Study
**THIS IS A KEY SECTION.** The 5-10 must-read papers for mastery:
- The original paper
- The most important follow-up papers (with why they matter)
- Any critical critique/analysis papers
- For EACH: full citation, DOI/URL, and a 2-3 sentence annotation explaining what you'll
  learn from it and why it's worth reading
- Ranked: "Start here" → "Read next" → "Deep mastery"

### §13 — Resources & Further Reading
- Best books (specific chapters)
- Best blog posts / tutorials (real URLs — distill.pub, official docs, well-known authors)
- Online courses where this is taught well
- Package documentation pages (direct links)

---

## 4. Python stack & version policy (mid-2026)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD, NMF, FastICA, KernelPCA, SparsePCA
from sklearn.decomposition import IncrementalPCA, MiniBatchNMF, MiniBatchSparsePCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding, MDS
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import umap                        # umap-learn package
from openTSNE import TSNE as openTSNE   # openTSNE package (faster t-SNE)
import plotly.express as px        # interactive embeddings
import shap
import optuna
```

- scikit-learn: ~1.5.x
- umap-learn: ~0.5.x
- openTSNE: ~1.0.x
- SHAP: ~0.46.x
- Optuna: ~3.x

Flag anything version-uncertain with `# verify in current docs`.

---

## 5. Notation conventions

- $n$ = number of samples, $p$ = original dimensionality, $d$ = target/reduced dimensionality
- $\mathbf{X} \in \mathbb{R}^{n \times p}$ = data matrix
- $\mathbf{Z} \in \mathbb{R}^{n \times d}$ = embedding/projection matrix
- $\mathbf{W}$ = weight/loading matrix
- Bold lowercase = vectors: $\mathbf{x}_i$, $\mathbf{z}_i$
- Use LaTeX `$...$` inline, `$$...$$` display

---

## 6. Callout convention

- `> 💡 **Intuition:**` — the geometric mental model, the analogy
- `> 🔧 **In practice:**` — practical engineering guidance, code tip
- `> ⚠️ **Pitfall:**` — the trap practitioners fall into
- `> 📜 **Origin/Citation:**` — where an idea comes from
- `> 🏆 **Best practice:**` — the gold-standard recommendation
- `> 🔬 **Deep dive:**` — optional detailed derivation
- `> 📊 **Benchmark:**` — real performance comparison
- `> 🏭 **Industry application:**` — real-world use case

---

## 7. The ten algorithm files

| File | Algorithm family | Key sklearn / specialized packages |
|---|---|---|
| `pca.md` | PCA + Randomized PCA + Incremental PCA + Kernel PCA + Sparse PCA | `sklearn.decomposition.PCA`, `KernelPCA`, `SparsePCA`, `IncrementalPCA` |
| `lda.md` | Linear Discriminant Analysis | `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` |
| `nmf.md` | Non-negative Matrix Factorization | `sklearn.decomposition.NMF`, `MiniBatchNMF` |
| `ica.md` | Independent Component Analysis | `sklearn.decomposition.FastICA` |
| `tsne.md` | t-SNE + variants | `sklearn.manifold.TSNE`, `openTSNE`, parametric t-SNE |
| `umap.md` | UMAP + variants | `umap-learn`, parametric UMAP, supervised UMAP |
| `manifold-methods.md` | Isomap, LLE, Spectral Embedding, MDS, PHATE | `sklearn.manifold`, `phate` |
| `clustering-as-dr.md` | Clustering-based DR (k-means, GMM, PHATE, prototype distances) | `sklearn.cluster`, `sklearn.mixture` |
| `autoencoder-dr.md` | Autoencoder + VAE + contrastive DR | `torch`, `sklearn.neural_network` |
| `random-projections.md` | Johnson-Lindenstrauss, sparse RP, Gaussian RP | `sklearn.random_projection` |

---

## 8. The Overview File

The course MUST have an `overview.md` file (separate from README) that:
1. Explains what dimensionality reduction is and why it matters
2. Divides all algorithms into **Workhorse** vs **Advanced/Specialized** categories
3. Ranks workhorse algorithms by difficulty (easiest first):
   1. PCA (and variants)
   2. LDA
   3. NMF
   4. ICA
   5. t-SNE
   6. UMAP
4. Provides a **quick-reference decision table**: given your problem, which algorithm?
5. Explains the curse of dimensionality
6. Covers the taxonomy: linear vs nonlinear, global vs local, supervised vs unsupervised
7. Covers evaluation: how do you know if a DR is good?

---

## 9. Word count target: 18,000+ tokens per file

Write incrementally: Write §0 + §1 first, then Edit/append §2, then §3, etc.
Do NOT stop at a partial chapter. 

---

## 10. Output discipline for writing agents

Write ONLY the Markdown to the file. Return a short summary: file path, approx word count,
sections covered, datasets used, any uncertainties flagged. Do NOT echo the chapter text back.
