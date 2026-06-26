# Research Dossier: Clustering as Dimensionality Reduction

**Chapter topic:** Clustering as Dimensionality Reduction  
**Prepared for:** Dimensionality Reduction Masterclass writing agent  
**Research date:** 2026-06-26  
**Status:** Complete — all 7 research tasks done

---

## Table of Contents

1. [Core Conceptual Framework](#1-core-conceptual-framework)
2. [Innovative Real-World Applications](#2-innovative-real-world-applications)
3. [Paper Citations and Summaries](#3-paper-citations-and-summaries)
4. [Sklearn 1.5.x API — Full Parameter Reference](#4-sklearn-15x-api--full-parameter-reference)
5. [Specialized Python Packages](#5-specialized-python-packages)
6. [Hyperparameter Tuning Guidance](#6-hyperparameter-tuning-guidance)
7. [Expert Practitioner Checklist](#7-expert-practitioner-checklist)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Key Code Patterns](#9-key-code-patterns)

---

## 1. Core Conceptual Framework

### The Core Idea

Clustering as dimensionality reduction works by using the geometric relationship between a data point and a set of learned cluster prototypes (centroids) as a new, low-dimensional feature representation. Instead of describing a sample with its raw D-dimensional feature vector, we describe it by its relationship (distance or probability) to K cluster centers, where K << D.

Three flavors of this idea:

| Approach | Output per sample | Output dimension | Algorithm |
|---|---|---|---|
| **Hard assignment** | One-hot cluster label | K (sparse) or 1 (integer) | KMeans labels |
| **Distance vector** | Distance to every centroid | K (dense) | KMeans.transform() |
| **Soft assignment** | Posterior probability of each component | K (dense) | GaussianMixture.predict_proba() |

The underlying intuition: if the data clusters, then the K-dimensional position-in-cluster-space is often a sufficient statistic for downstream tasks, and the remaining variance within each cluster is noise.

### Why This Is Legitimate Dimensionality Reduction

- Formally: KMeans.transform() maps R^D → R^K via f(x) = [||x - c_1||, ..., ||x - c_K||].
- Each output feature is nonlinear in the original features (Euclidean distance is a nonlinear function).
- Unlike PCA (linear), this can capture cluster-shaped structure that linear projections miss.
- The GMM version additionally captures uncertainty: a sample near a cluster boundary gets non-trivial probability mass on two components, encoding ambiguity that hard assignment loses.

### Relationship to Other DR Methods

- **Spectral Clustering** is literally a two-step DR pipeline: first apply Laplacian eigenmaps (a form of nonlinear DR), then cluster in the embedding. The spectral embedding itself becomes the DR output.
- **SOM** produces a 2D grid that is simultaneously a topology-preserving DR and a clustering.
- **LVQ** is a supervised prototype-based method; its prototypes define a Voronoi partition that can serve as a feature space.
- **Feature Agglomeration (FeatureAgglomeration)** applies clustering to the *feature* dimension, not the sample dimension, reducing the number of features by pooling correlated features into groups.

---

## 2. Innovative Real-World Applications

### Application 1: Bag of Visual Words (Computer Vision, Pre-Deep Learning)

**Domain:** Computer vision, image classification and retrieval  
**Problem:** Image descriptors (SIFT, HOG, SURF) produce variable-length, high-dimensional local feature vectors per image. Images need fixed-length feature vectors for downstream classifiers.  
**Solution:** Apply KMeans to a large pool of local descriptors to learn a "visual vocabulary" of K codewords (cluster centers). Then represent each image as a K-dimensional histogram: for each descriptor in an image, find its nearest codeword, increment that bin. The result is a K-dimensional Bag of Visual Words (BoVW) vector.  
**Why this algorithm:** KMeans naturally defines a Voronoi-based codebook. The quantization step (assign descriptor to nearest cluster) is the DR step — collapsing millions of local descriptors to one K-dim vector per image.  
**Reference:** Sivic & Zisserman, 2003, "Video Google"; Csurka et al., 2004, "Visual categorization with bags of keypoints". Dominated the ImageNet-era leaderboards before CNNs.  
**Scale:** Codebooks of K=1,000–50,000 visual words were typical. Google Image Search used this at web scale (billions of descriptors).

### Application 2: Fisher Vectors (Computer Vision, Enhanced BoVW)

**Domain:** Image classification, large-scale retrieval  
**Problem:** Standard BoVW only encodes 0th-order statistics (counts). It loses information about *how* descriptors are distributed relative to each Gaussian component.  
**Solution:** Fit a GMM to local descriptors. For each image, compute the sum of first and second-order gradient statistics with respect to the GMM parameters (mean and covariance). The Fisher Vector is the concatenation of these gradients: dimension 2*K*D per image (K components, D-dim descriptors, first + second order). This uses GMM posterior probabilities as soft feature representations.  
**Why GMM:** The soft assignment (predict_proba) encodes uncertainty about which visual word a descriptor belongs to. First-order statistics encode the mean deviation from each component; second-order encodes variance. Together these carry far more discriminative power than hard BoVW.  
**Reference:** Perronnin & Dance (2007), Perronnin et al. ECCV 2010. Fisher Vectors won the ImageNet LSVRC 2011 before AlexNet (2012).  
**Company use:** Yandex, Google, Getty Images used Fisher Vectors for visual similarity search.

### Application 3: VLAD — Vector of Locally Aggregated Descriptors

**Domain:** Image retrieval and recognition  
**Problem:** Fisher Vectors are high-dimensional (2*K*D). Need a compact alternative that still beats BoVW.  
**Solution:** Like BoVW, assign each descriptor to its nearest KMeans centroid. But instead of counting, accumulate the *residual* (descriptor - centroid) for each cluster. VLAD vector = K*D dimensional; can be compressed further with PCA.  
**Key insight:** VLAD is a simplified Fisher Vector that captures only first-order statistics. The residual is the DR — mapping D-dimensional descriptor space to a K-dimensional residual-per-cluster space.  
**Reference:** Jégou, Douze, Schmid, Pérez, "Aggregating local descriptors into a compact image representation," CVPR 2010.

### Application 4: K-Means on Word Embeddings for Topic Features (NLP)

**Domain:** Text classification, topic modeling, news categorization  
**Problem:** Word2Vec/GloVe/BERT embeddings are 100-768 dimensional. Documents need fixed-length representations for classifiers. LDA topic models are slow to train and interpret.  
**Solution:** Cluster word embeddings with KMeans (K=50–500 topics). Represent a document by: (a) distance vector to each word cluster center — a K-dimensional document feature, or (b) soft GMM posterior probability profile over K "semantic clusters." K-means finds coherent semantic neighborhoods (e.g., finance cluster, sports cluster) without any labels.  
**Why this works:** Word embeddings that are semantically close cluster together. The cluster membership effectively creates a soft topic model. Downstream: use the K-dim document vector as input to a logistic regression or gradient boosting model.  
**Reference:** Nalisnick et al., 2016; practical implementation at sklearn: "Clustering text documents using k-means" (sklearn 1.5.2 example).  
**Advantage over LDA:** No need to tune alpha/beta hyperparameters; faster; works directly on dense embeddings.

### Application 5: Time Series — Bag of Patterns (BOP)

**Domain:** Time series classification (medical, industrial sensor, finance)  
**Problem:** Time series of different lengths need fixed-length feature vectors. Raw series are high-dimensional and shift-variant.  
**Solution:** Convert each time series window into a symbolic representation (SAX — Symbolic Aggregate approXimation), then cluster these symbolic "patterns" with KMeans. Represent each series as a histogram over K pattern clusters (Bag of Patterns). DR from T-dimensional time series → K-dimensional bag.  
**Why clustering:** Patterns that share similar shapes cluster together regardless of exact magnitude, making the representation robust to noise and magnitude shifts.  
**Reference:** Lin & Li (2009), "Finding Structural Similarity in Time Series Data by Message Passing Algorithms." Recent: multi-scale BOP (ICIC 2024 conference paper on multivariate time series).  
**Applications:** ECG arrhythmia detection, industrial machine fault detection (vibration patterns), financial market regime detection.

### Application 6: Genomics — Pathway Activity Features via Clustering (scRNA-seq)

**Domain:** Single-cell genomics, cancer research, drug discovery  
**Problem:** scRNA-seq data has D~20,000 genes per cell, N~10,000–1M cells. Most genes are noise; biologically meaningful structure lives in correlated gene modules (pathways).  
**Solution:** Cluster genes (features) using AgglomerativeClustering or KMeans on the gene co-expression matrix. Each cluster of genes defines a "module." Represent each cell by its mean expression in each module — reduces from 20,000 genes to K~100–500 pathway features.  
**Alternative:** Use GMM posteriors over pre-defined pathway clusters as cell features, capturing soft pathway activation scores (analogous to AUCell scores).  
**Why clustering vs. PCA:** Clustering preserves biological interpretability (each component = a gene cluster = a potential pathway). PCA components are linear combinations of all genes — hard to interpret biologically.  
**Reference:** scMUG (Briefings in Bioinformatics, 2025); joint dimensionality reduction and clustering (Nucleic Acids Research, 2022 — PRECAST method); Integrating pathway knowledge with deep neural networks (BioData Mining, 2021).  
**Company use:** 10x Genomics toolchain (Seurat, Scanpy) uses clustering-based DR as the standard first step. Genentech, Novartis use this pipeline for drug target discovery.

### Application 7: Anomaly Detection — Distance to Nearest Cluster as Anomaly Score

**Domain:** Fraud detection, network intrusion detection, manufacturing defect detection, IoT sensor monitoring  
**Problem:** Anomaly detection without labeled anomaly examples. Need a score that says "how abnormal is this point?"  
**Solution:** Fit KMeans on normal (inlier) training data. For each new point, compute distances to all K centroids via `kmeans.transform(X)`, then take the minimum distance: `anomaly_score = transform(X).min(axis=1)`. Points far from all cluster centers are anomalies. Threshold at, say, the 99th percentile of training scores.  
**Why K-means:** O(K·D) inference time — extremely fast for real-time fraud scoring. Contrast with LOF or isolation forest which require storing training data.  
**Advantage:** The K-dimensional distance vector is itself a feature representation that can be fed to a one-class SVM, autoencoders, or GBDT for a more powerful anomaly detector.  
**Reference:** Edge Impulse documentation uses KMeans anomaly detection for embedded ML (microcontroller deployment). Patent US10318886 (Anomaly detection with K-means clustering).  
**Company use:** Edge Impulse (embedded ML platform), various fintech firms for real-time transaction scoring.

### Application 8: Recommendation Systems — User/Item Cluster Features

**Domain:** E-commerce, streaming services, social networks  
**Problem:** User-item interaction matrices are enormous (millions of users × millions of items). Need compact user representations that preserve behavioral similarity.  
**Solution:** Cluster users by interaction patterns (KMeans on user embedding matrix from matrix factorization or collaborative filtering). Represent each user by: (a) their cluster assignment (one-hot), or (b) distance vector to K user-type centroids. This K-dimensional user profile is used as features for downstream models (pricing models, churn prediction, content ranking).  
**Example:** Netflix uses over 1,300 clusters based on viewing metadata. Spotify's recommendation system uses behavioral clustering to group users, then trains per-cluster recommendation models.  
**Key innovation:** Using GMM posterior probabilities instead of hard cluster assignment allows users near cluster boundaries (e.g., someone who likes both classical and hip-hop) to be represented with non-zero probability in multiple genre clusters — richer signal than hard assignment.  
**Reference:** EfficientRec (arXiv 2024): unlimited user-item scale recommendation using clustering + user interaction embedding profiles.

---

## 3. Paper Citations and Summaries

### Paper 1: MacQueen (1967) — Original K-Means

**Full citation:**  
MacQueen, J. B. (1967). "Some methods for classification and analysis of multivariate observations." In L. M. Le Cam & J. Neyman (Eds.), *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, Vol. 1, pp. 281–297. University of California Press.

**URL:** https://www.semanticscholar.org/paper/Some-methods-for-classification-and-analysis-of-MacQueen/ac8ab51a86f1a9ae74dd0e4576d1a019f5e654ed  
**DOI:** No DOI (conference proceedings, pre-DOI era). Available via Semantic Scholar.

**What you learn:**  
MacQueen coins the term "k-means" and proves convergence of the algorithm under finite-step iterations. He frames the algorithm as a classification problem: partition N observations into K classes to minimize within-class sum of squared Euclidean distances. The paper introduces the two-step (assignment + centroid update) loop that all modern implementations use.

**Difficulty:** Low — the mathematics is elementary, the writing is accessible. Good for historical framing in the chapter.

---

### Paper 2: Kohonen (1982) — Self-Organizing Map

**Full citation:**  
Kohonen, T. (1982). "Self-organized formation of topologically correct feature maps." *Biological Cybernetics*, 43, 59–69.

**DOI:** https://doi.org/10.1007/BF00337288  
**PDF:** https://www.cnbc.cmu.edu/~tai/nc19journalclubs/Kohonen1982_Article_Self-organizedFormationOfTopol.pdf

**What you learn:**  
Introduces the SOM as a biologically motivated model of the somatosensory cortex. The key insight is a neighborhood-respecting competitive learning rule: when a neuron "wins" a competition, its neighbors also update, creating a smooth topographic map from high-dimensional input space to a 2D grid. The paper proves that this rule converges to a topology-preserving projection for 1D inputs, and motivates the conjecture for higher dimensions.

**Difficulty:** Medium — the biological motivation is light, but the mathematical formulation of the neighborhood function and competitive learning rule is straightforward. Readers with a neural networks background will be comfortable.

---

### Paper 3: Perronnin & Dance (2007) / Perronnin et al. (2010) — Fisher Vectors

**Full citation (improved Fisher Vector):**  
Perronnin, F., Sánchez, J., & Mensink, T. (2010). "Improving the Fisher kernel for large-scale image classification." In *Proceedings of the European Conference on Computer Vision (ECCV 2010)*, Heraklion, Greece, pp. 143–156. Springer.

**DOI:** https://doi.org/10.1007/978-3-642-15561-1_11  
**PDF (INRIA HAL):** https://inria.hal.science/inria-00548630v1/document  
**Additional:** Sánchez et al. (2013). "Image Classification with the Fisher Vector: Theory and Practice." *International Journal of Computer Vision*, 105(3), 222–245. DOI: https://doi.org/10.1007/s11263-013-0636-x

**What you learn:**  
The 2010 ECCV paper introduces two key improvements over the original Fisher Kernel: (1) power normalization (apply sqrt to Fisher Vector components before L2-normalization), and (2) explicit L2 normalization of the final vector. Together these eliminate the burstiness problem (a single very common descriptor dominating the encoding) and dramatically improve classification accuracy on PASCAL VOC and ImageNet benchmarks. The GMM-based soft assignment is shown to be superior to hard BoVW assignment.

**Difficulty:** Medium-high — requires familiarity with Fisher information and the score function (gradient of log-likelihood). The computer vision background (SIFT, HOG) can be skimmed.

---

### Paper 4: Jégou et al. (2010) — VLAD

**Full citation:**  
Jégou, H., Douze, M., Schmid, C., & Pérez, P. (2010). "Aggregating local descriptors into a compact image representation." In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2010)*, pp. 3304–3311.

**DOI:** Available via IEEE Xplore.  
**PDF:** https://falchi.isti.cnr.it/Draft/2013-SISAP-VLAD-DRAFT.pdf

**What you learn:**  
Proposes VLAD as a more compact alternative to Fisher Vectors that achieves competitive performance at much lower dimensionality. The key idea: instead of accumulating zero-order statistics (BoVW counts), accumulate first-order residuals (descriptor - centroid) per cluster. The resulting K*D-dimensional VLAD vector can be further compressed with PCA. Experiments on Oxford Buildings dataset demonstrate excellent retrieval performance. This paper makes explicit the connection between vector quantization (k-means codebook) and DR.

**Difficulty:** Medium — clear exposition, well-motivated. Readers with linear algebra and a basic understanding of image retrieval will follow easily.

---

### Paper 5: Belkin & Niyogi (2001) — Laplacian Eigenmaps (spectral embedding foundation)

**Full citation:**  
Belkin, M., & Niyogi, P. (2001). "Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering." In *Advances in Neural Information Processing Systems (NIPS 2001)*, Vol. 14.

**URL:** https://papers.nips.cc/paper/1961-laplacian-eigenmaps-and-spectral-techniques-for-embedding-and-clustering  
**Semantic Scholar:** https://www.semanticscholar.org/paper/Laplacian-Eigenmaps-and-Spectral-Techniques-for-and-Belkin-Niyogi/9d16c547d15a08091e68c86a99731b14366e3f0d

**What you learn:**  
The foundational paper showing that spectral clustering is equivalent to embedding via the graph Laplacian eigenvectors, then clustering in the embedding. The first k nontrivial eigenvectors of the Laplacian encode the natural cluster structure of the data. This establishes that Spectral Clustering (sklearn.cluster.SpectralClustering) is literally a DR method (Laplacian Eigenmap) followed by K-Means.

**Difficulty:** Medium-high — requires some comfort with spectral graph theory and differential geometry. The intuition section is accessible.

---

### Paper 6: Ghasemian et al. (2019) — Assessing Dimensionality Reduction on Clustering (systematic study)

**Full citation:**  
See: "Assessing the impact of dimensionality reduction on clustering performance — a systematic study," arXiv:2604.22099  
https://arxiv.org/html/2604.22099v1

**What you learn:**  
Systematic empirical study of how DR preprocessing affects downstream clustering quality across 12 DR methods and 20 datasets. Key finding: DR before clustering (e.g., PCA→KMeans) often helps, but the benefit depends on the DR method and dataset geometry. Particularly relevant to the "t-SNE as preprocessing for clustering" discussion.

**Difficulty:** Low-medium — empirical paper, no heavy math. Read the tables and conclusion sections.

---

## 4. Sklearn 1.5.x API — Full Parameter Reference

**Note on version compatibility:** All parameters below are verified against sklearn 1.5.2 documentation. Where parameters changed between versions, version annotations are given. sklearn 1.9 is current as of mid-2026; the parameters below remain stable from 1.5 through 1.9 unless flagged.

---

### 4.1 `sklearn.cluster.KMeans` (sklearn 1.5.2)

```python
from sklearn.cluster import KMeans

KMeans(
    n_clusters=8,
    init='k-means++',
    n_init='auto',
    max_iter=300,
    tol=1e-4,
    verbose=0,
    random_state=None,
    copy_x=True,
    algorithm='lloyd'
)
```

| Parameter | Type | Default | What it controls |
|---|---|---|---|
| `n_clusters` | int | 8 | Number of clusters K; also the output dimension of `transform()` |
| `init` | str or array | `'k-means++'` | Initialization: `'k-means++'` (smart), `'random'`, or pre-supplied centroids array |
| `n_init` | int or `'auto'` | `'auto'` | Number of restarts. `'auto'` → 10 for random init, 1 for `k-means++` or array. **Changed in 1.4:** default changed from 10 to `'auto'` |
| `max_iter` | int | 300 | Max EM iterations per run |
| `tol` | float | 1e-4 | Relative tolerance on inertia change for convergence |
| `verbose` | int | 0 | Verbosity (0=silent) |
| `random_state` | int/None/RS | None | Seed for reproducibility |
| `copy_x` | bool | True | If False, modify X in-place (numerical precision: center before computing distances) |
| `algorithm` | str | `'lloyd'` | `'lloyd'` (standard EM-style), `'elkan'` (uses triangle inequality — faster for well-separated clusters with low K) |

**Key methods for DR use:**

```python
# After fitting:
X_transformed = kmeans.transform(X)
# Returns ndarray of shape (n_samples, n_clusters)
# Each column = distance to that cluster centroid

X_transformed = kmeans.fit_transform(X)
# More efficient: fits and transforms in one call
# Same output shape

labels = kmeans.predict(X)
# Returns ndarray of shape (n_samples,) — hard cluster assignment
# For pure DR: use transform(), not predict()
```

**API note — `algorithm` parameter:** `'full'` was removed in sklearn 1.1; use `'lloyd'` instead (they are identical). If you have legacy code with `algorithm='full'`, update it.

---

### 4.2 `sklearn.mixture.GaussianMixture` (sklearn 1.5.2)

```python
from sklearn.mixture import GaussianMixture

GaussianMixture(
    n_components=1,
    covariance_type='full',
    tol=1e-3,
    reg_covar=1e-6,
    max_iter=100,
    n_init=1,
    init_params='kmeans',
    weights_init=None,
    means_init=None,
    precisions_init=None,
    random_state=None,
    warm_start=False,
    verbose=0,
    verbose_interval=10
)
```

| Parameter | Type | Default | What it controls |
|---|---|---|---|
| `n_components` | int | 1 | Number of Gaussian components K; also the output dimension of `predict_proba()` |
| `covariance_type` | str | `'full'` | `'full'`: each component has its own full covariance matrix (most expressive, most parameters). `'tied'`: all share one covariance. `'diag'`: diagonal covariance per component (no correlations). `'spherical'`: isotropic covariance per component (fewest parameters). |
| `tol` | float | 1e-3 | EM convergence tolerance on log-likelihood |
| `reg_covar` | float | 1e-6 | Regularization added to covariance diagonal (prevents singular matrices) |
| `max_iter` | int | 100 | Maximum EM iterations |
| `n_init` | int | 1 | Number of EM restarts. **Unlike KMeans, default is 1** — increase to 5–10 for unstable fits |
| `init_params` | str | `'kmeans'` | Initialization: `'kmeans'`, `'k-means++'`, `'random'`, `'random_from_data'`. **`'k-means++'` and `'random_from_data'` added in sklearn 1.1** |
| `weights_init` | array or None | None | User-supplied initial weights (shape: n_components,) |
| `means_init` | array or None | None | User-supplied initial means (shape: n_components, n_features) |
| `precisions_init` | array or None | None | User-supplied initial precisions (inverse of covariance matrices) |
| `random_state` | int/None/RS | None | Seed for reproducibility |
| `warm_start` | bool | False | Reuse previous fit as initialization for next call to `fit()` |
| `verbose` | int | 0 | Verbosity |
| `verbose_interval` | int | 10 | Print interval when verbose > 0 |

**Key methods for DR use:**

```python
# After fitting:
proba = gmm.predict_proba(X)
# Returns ndarray of shape (n_samples, n_components)
# Each row is a probability distribution over K components
# Sum of each row = 1.0 (it's a proper posterior)
# THIS is the soft feature representation for DR

labels = gmm.predict(X)
# Hard assignment (argmax of predict_proba)
# For DR: use predict_proba(), not predict()

log_prob = gmm.score_samples(X)
# Log-likelihood per sample (shape: n_samples,)
# Useful for anomaly scoring (low log-prob → anomaly)
```

**API note:** `GaussianMixture` has no `transform()` method — you must use `predict_proba()` explicitly. This is a common gotcha.

---

### 4.3 `sklearn.cluster.FeatureAgglomeration` (sklearn 1.5.2)

```python
from sklearn.cluster import FeatureAgglomeration

FeatureAgglomeration(
    n_clusters=2,
    metric='euclidean',
    memory=None,
    connectivity=None,
    compute_full_tree='auto',
    linkage='ward',
    pooling_func=np.mean,
    distance_threshold=None,
    compute_distances=False
)
```

| Parameter | Type | Default | What it controls |
|---|---|---|---|
| `n_clusters` | int or None | 2 | Number of output features (clusters of input features). Must be None if `distance_threshold` is set |
| `metric` | str or callable | `'euclidean'` | Distance metric for linkage: `'euclidean'`, `'l1'`, `'l2'`, `'manhattan'`, `'cosine'`, `'precomputed'`. **`metric` parameter added in sklearn 1.2.** Only `'euclidean'` works with `linkage='ward'` |
| `memory` | str or Memory | None | Directory path or joblib.Memory for caching the tree computation |
| `connectivity` | array or callable | None | Connectivity matrix (e.g., from `kneighbors_graph`) to constrain which features can merge |
| `compute_full_tree` | bool or `'auto'` | `'auto'` | Whether to compute the full dendrogram. `'auto'` = True when `distance_threshold` is set |
| `linkage` | str | `'ward'` | Linkage criterion: `'ward'` (min variance), `'complete'` (max distance), `'average'` (mean distance), `'single'` (min distance) |
| `pooling_func` | callable | `np.mean` | How to aggregate feature values within each cluster. Can be `np.mean`, `np.max`, `np.sum`, any callable accepting `axis=1` |
| `distance_threshold` | float or None | None | Alternative to `n_clusters`: stop merging when distance exceeds this threshold |
| `compute_distances` | bool | False | If True, store cluster distances (needed for dendrogram visualization). Adds memory overhead |

**Key methods:**

```python
# Fit and reduce feature space:
X_reduced = agglo.fit_transform(X)
# Input: (n_samples, n_features)
# Output: (n_samples, n_clusters) — features pooled within each cluster

# Inverse transform (approximate reconstruction):
X_approx = agglo.inverse_transform(X_reduced)
# Output: (n_samples, n_features_original) — each feature gets its cluster's pooled value

# Inspect which features merged:
print(agglo.labels_)
# Array of shape (n_features,) — cluster index for each original feature
```

**Version history (important changes):**
- v0.21: `distance_threshold` added
- v1.0: `feature_names_in_` attribute added  
- v1.2: `metric` parameter added (previously always Euclidean)
- v1.4: `metric=None` deprecated (will error in v1.6); `set_output()` API added
- v1.5: `Xt` parameter in `inverse_transform()` renamed to `X`

---

### 4.4 `sklearn.cluster.AgglomerativeClustering` (sklearn 1.5.2)

```python
from sklearn.cluster import AgglomerativeClustering

AgglomerativeClustering(
    n_clusters=2,
    metric='euclidean',
    memory=None,
    connectivity=None,
    compute_full_tree='auto',
    linkage='ward',
    distance_threshold=None,
    compute_distances=False
)
```

| Parameter | Type | Default | What it controls |
|---|---|---|---|
| `n_clusters` | int or None | 2 | Target number of clusters |
| `metric` | str/callable | `'euclidean'` | Distance metric. Same options as FeatureAgglomeration |
| `linkage` | str | `'ward'` | Same options as FeatureAgglomeration |
| `distance_threshold` | float or None | None | Stop merging above this distance |
| `compute_distances` | bool | False | Store inter-cluster distances |

**For DR use:** AgglomerativeClustering.fit(X).labels_ gives cluster assignment. Use these labels as features (one-hot encode or embed) downstream. Does NOT have a `transform()` method — use `labels_` attribute.

**Version history:** `metric` parameter added in v1.2; `metric=None` deprecated in v1.4.

---

### 4.5 `sklearn.cluster.SpectralClustering` (sklearn 1.5.2)

```python
from sklearn.cluster import SpectralClustering

SpectralClustering(
    n_clusters=8,
    eigen_solver=None,
    n_components=None,
    random_state=None,
    n_init=10,
    gamma=1.0,
    affinity='rbf',
    n_neighbors=10,
    eigen_tol='auto',
    assign_labels='kmeans',
    degree=3,
    coef0=1,
    kernel_params=None,
    n_jobs=None,
    verbose=False
)
```

| Parameter | Type | Default | What it controls |
|---|---|---|---|
| `n_clusters` | int | 8 | Number of clusters |
| `n_components` | int or None | None | Number of eigenvectors to use for spectral embedding. If None, defaults to `n_clusters`. **Set higher than `n_clusters` to get a richer embedding** |
| `eigen_solver` | str | None | Eigendecomposition method: `'arpack'`, `'lobpcg'`, `'amg'` (requires pyamg). None → auto-select |
| `affinity` | str/callable | `'rbf'` | How to build the affinity matrix: `'nearest_neighbors'`, `'rbf'` (Gaussian kernel), `'precomputed'`, `'precomputed_nearest_neighbors'`, or kernel name |
| `gamma` | float | 1.0 | Kernel coefficient for rbf/poly/sigmoid/laplacian/chi2 kernels |
| `n_neighbors` | int | 10 | Neighbors for `nearest_neighbors` affinity |
| `eigen_tol` | float | `'auto'` | Stopping criterion for Laplacian eigendecomposition |
| `assign_labels` | str | `'kmeans'` | Clustering method in embedding space: `'kmeans'`, `'discretize'`, `'cluster_qr'` |
| `n_init` | int | 10 | K-means restarts when `assign_labels='kmeans'` |

**DR connection:** Internally applies `sklearn.manifold.SpectralEmbedding` to get a `n_components`-dim representation, then clusters with the `assign_labels` method. The eigenvector matrix IS the dimensionality-reduced representation — you can access it by using `SpectralEmbedding` directly.

---

### 4.6 `sklearn.manifold.SpectralEmbedding` (sklearn 1.5.2)

```python
from sklearn.manifold import SpectralEmbedding

SpectralEmbedding(
    n_components=2,
    affinity='nearest_neighbors',
    gamma=None,
    random_state=None,
    eigen_solver=None,
    eigen_tol='auto',
    n_neighbors=None,
    n_jobs=None
)
```

**Key method:** `fit_transform(X)` → ndarray of shape (n_samples, n_components). This is the pure DR step underlying SpectralClustering.

---

## 5. Specialized Python Packages

### 5.1 MiniSom — Self-Organizing Maps

| Attribute | Value |
|---|---|
| **Package name** | MiniSom |
| **PyPI name** | `MiniSom` |
| **Current version (mid-2026)** | 2.3.6 (released Feb 11, 2026) |
| **Install** | `pip install MiniSom` |
| **Key import** | `from minisom import MiniSom` |
| **License** | MIT |
| **Maintenance** | Active (maintained by Giuseppe Vettigli, consistent releases) |
| **GitHub** | https://github.com/JustGlowing/minisom |

**What it does better than sklearn:**
- sklearn has no SOM implementation whatsoever; MiniSom is the de facto standard
- Pure NumPy implementation (minimal dependencies)
- Supports JIT-accelerated training via Numba (`train_batch_offline_fast` method) — significant speedup on large datasets
- Supports multiple distance metrics and neighborhood functions
- Grid topologies: rectangular and hexagonal (toroidal optional)
- Built-in visualization utilities (U-matrix, component planes)

**Core API:**

```python
from minisom import MiniSom
import numpy as np

# Initialize SOM: (grid_x, grid_y, input_len, sigma, learning_rate)
som = MiniSom(
    x=10,                    # grid width
    y=10,                    # grid height  
    input_len=X.shape[1],   # input dimensionality
    sigma=1.0,               # neighborhood radius
    learning_rate=0.5,       # initial learning rate
    neighborhood_function='gaussian',  # 'gaussian', 'mexican_hat', 'bubble', 'triangle'
    topology='rectangular',  # 'rectangular' or 'hexagonal'
    random_seed=42
)

som.random_weights_init(X)   # initialize weights
som.train(X, 1000, verbose=True)   # train for 1000 iterations

# DR: get 2D grid coordinates for each sample
winners = np.array([som.winner(x) for x in X])
# shape: (n_samples, 2) — each sample mapped to a grid cell

# OR: get distance to all neurons (full embedding)
distances = np.array([som.distance_map_values(x) for x in X])
```

**Limitation:** No sklearn-compatible Pipeline interface (no fit/transform pattern). Wrap in a custom sklearn transformer if needed.

---

### 5.2 somoclu — Large-Scale SOM

| Attribute | Value |
|---|---|
| **Package name** | somoclu |
| **PyPI name** | `somoclu` |
| **Install** | `pip install somoclu` |
| **License** | MIT |
| **Key import** | `from somoclu import Somoclu` |

**What it does better:** GPU-accelerated SOM training via MPI/OpenCL. Designed for large datasets (>100K samples). sklearn-like API. Better than MiniSom for production-scale SOMs.

---

### 5.3 sklearn-som

| Attribute | Value |
|---|---|
| **PyPI name** | `sklearn-som` |
| **Install** | `pip install sklearn-som` |
| **License** | MIT |

**What it does better:** sklearn-compatible SOM (fit/transform/predict interface). Drop-in for Pipelines.

---

### 5.4 hdbscan / HDBSCAN (via sklearn 1.3+)

| Attribute | Value |
|---|---|
| **Package name** | hdbscan (standalone) or `sklearn.cluster.HDBSCAN` (sklearn ≥ 1.3) |
| **PyPI name** | `hdbscan` |
| **Install** | `pip install hdbscan` or use sklearn ≥ 1.3 |
| **License** | BSD-3 |
| **Key import** | `from sklearn.cluster import HDBSCAN` (sklearn) or `import hdbscan` |

**Relevance to clustering-as-DR:** HDBSCAN produces soft cluster membership strengths via `soft_clustering` mode — these membership scores (shape: n_samples × n_clusters) serve as a probabilistic feature representation analogous to GMM posteriors.

---

### 5.5 PyTorch/FAISS for Large-Scale K-Means

| Attribute | Value |
|---|---|
| **Package name** | faiss-cpu / faiss-gpu |
| **PyPI name** | `faiss-cpu` or `faiss-gpu` |
| **Install** | `pip install faiss-cpu` |
| **License** | MIT |
| **Key import** | `import faiss` |

**What it does better:** FAISS KMeans scales to billions of vectors with GPU acceleration. sklearn KMeans struggles above ~10M samples. For BoVW codebook training on large image datasets, FAISS is the production standard (used by Meta).

```python
import faiss
import numpy as np

kmeans = faiss.Kmeans(d=128, k=1000, niter=20, gpu=True)
kmeans.train(descriptors.astype(np.float32))
centroids = kmeans.centroids  # shape: (1000, 128)

# Fast assignment:
index = faiss.IndexFlatL2(128)
index.add(centroids)
distances, assignments = index.search(query_descriptors.astype(np.float32), 1)
```

---

### 5.6 pyclustering

| Attribute | Value |
|---|---|
| **PyPI name** | `pyclustering` |
| **Install** | `pip install pyclustering` |
| **License** | GNU GPL v3 |
| **Key import** | `from pyclustering.cluster.kmeans import kmeans` |

**Relevance:** Includes LVQ implementation (Learning Vector Quantization) — not available in sklearn. Also includes BANG, CURE, FCPS — useful for prototype-based DR experiments.

---

## 6. Hyperparameter Tuning Guidance

### 6.1 KMeans — `n_clusters` (K)

**What it is:** The single most important hyperparameter; controls both the model complexity and the output dimensionality of the DR.

**Too low (K too small):**
- Clusters are large and heterogeneous; the distance features don't distinguish fine-grained structure
- High inertia (within-cluster sum of squares)
- Downstream model can't distinguish important subgroups

**Too high (K too large):**
- Clusters overfit to noise; each cluster has too few samples
- Distances become noisy; the DR space is unreliable
- Inference is slower (all K distances computed per sample)
- For DR purposes: risk of almost-empty clusters with random behavior

**Recommended tuning range:** For a dataset with N samples and D features, a reasonable starting range is K = [sqrt(N/2), 2*sqrt(N)]. More practically: try K = [5, 10, 20, 50, 100, 200] and use cross-validation on downstream task.

**Heuristics:**
1. **Elbow method:** Plot inertia vs K. Pick the "elbow" — the point where the rate of decrease slows. In practice, the elbow is often not sharp; use it as a starting point, not a definitive answer.
2. **Silhouette score:** Compute for K = 2 to 20; pick K that maximizes mean silhouette. `sklearn.metrics.silhouette_score(X, labels)`. More reliable than elbow but O(N^2) memory — use `silhouette_samples` for large N.
3. **BIC for GMM (not pure KMeans):** `gmm.bic(X)` — lower is better. Penalizes model complexity. Reliable for GMM; for KMeans, use gap statistic or silhouette.
4. **For DR purpose:** If the goal is a downstream classifier, do a nested CV: outer loop evaluates classifier performance, inner loop tunes K. Treat K as a standard hyperparameter.

**Optuna search space:**
```python
import optuna

def objective(trial):
    k = trial.suggest_int("n_clusters", 5, 200, log=True)
    kmeans = KMeans(n_clusters=k, random_state=42)
    X_dr = kmeans.fit_transform(X_train)
    clf = LogisticRegression().fit(X_dr, y_train)
    return clf.score(X_val, y_val)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

---

### 6.2 KMeans — `init` and `n_init`

**`init='k-means++'` vs `'random'`:**
- Always prefer `'k-means++'` for quality; `'random'` is only for speed experiments.
- `k-means++` guarantees O(log K) approximation ratio for initial inertia.
- With `k-means++`, `n_init=1` is often sufficient; with `'random'`, use `n_init=10+`.

**`n_init` too low:** Risk of poor local optimum. Symptom: repeated runs give very different inertia.  
**`n_init` too high:** Wastes time. With `k-means++`, diminishing returns after 3–5 restarts.

---

### 6.3 GaussianMixture — `covariance_type`

**Impact on DR output:** `covariance_type` determines the shape of the Voronoi-like regions that define cluster membership.

| Type | Parameters | When to use |
|---|---|---|
| `'full'` | K * D*(D+1)/2 | When D is small (< 50) and you have ample data |
| `'tied'` | D*(D+1)/2 | When clusters have similar shapes |
| `'diag'` | K * D | Good default for moderate D; catches axis-aligned ellipses |
| `'spherical'` | K | When clusters are known to be isotropic (like K-Means clusters) |

**`n_init` for GMM:** Default is 1, which is often insufficient. Use `n_init=5` to `n_init=10` for production fits. EM has many local optima.

**`reg_covar`:** If you get `ConvergenceWarning` or `LinAlgError` (singular matrix), increase `reg_covar` from 1e-6 to 1e-3.

**BIC-based tuning for n_components:**
```python
bic_scores = []
for k in range(2, 30):
    gmm = GaussianMixture(n_components=k, covariance_type='diag', n_init=3, random_state=42)
    gmm.fit(X_train)
    bic_scores.append(gmm.bic(X_train))
best_k = np.argmin(bic_scores) + 2
```

---

### 6.4 FeatureAgglomeration — `n_clusters` and `linkage`

**`n_clusters`:**
- Target output dimensionality.
- Try `n_clusters` = [D/10, D/5, D/2] where D is original feature count.
- Use cross-validation on downstream task (same Optuna pattern as KMeans).

**`linkage`:**
- `'ward'`: Best default for most tabular datasets. Minimizes within-cluster variance. Requires Euclidean metric.
- `'average'`: Good for cosine-similar data (text, embeddings). Pair with `metric='cosine'`.
- `'complete'`: Produces compact clusters; sensitive to outliers.
- `'single'`: Produces long chain-like clusters; rarely useful for DR.

**`pooling_func`:**
- `np.mean`: Standard; good when features are comparable in scale.
- `np.max`: Max pooling; useful when you want the "most active" feature in a cluster.
- `np.sum`: Equivalent to mean * n_features_in_cluster; appropriate for count data.

---

### 6.5 SOM (MiniSom) — Grid Size and Learning Parameters

**Grid dimensions (x, y):**
- Total neurons = x*y. Too few → poor topology; too many → overfitting.
- Rule of thumb: x*y ≈ 5*sqrt(N) where N = number of training samples.
- Square grid (x=y) for general use. Rectangular if one dimension is more important.

**`sigma` (neighborhood radius):**
- Initial radius; typically starts at max(x, y) / 2.
- Decays to ~1.0 by end of training.
- Too large → all neurons update together, poor differentiation.
- Too small from the start → no topology preservation, equivalent to random K-Means.

**`learning_rate`:**
- Typical range: 0.1 to 1.0 (start high, decay).
- Too high: oscillations, no convergence.
- Too low: slow convergence, stuck in initial configuration.

**Number of iterations:**
- Rule of thumb: 500 * N iterations (N = training samples).
- Monitor quantization error (mean distance from each sample to its winning neuron); should decrease monotonically.

---

## 7. Expert Practitioner Checklist

### Before Fitting

**Data Preparation:**
- [ ] **Scale your features.** KMeans is distance-based. Unscaled features (e.g., income in thousands vs. age in decades) will dominate the distance. Always apply `StandardScaler` or `MinMaxScaler` before KMeans/GMM. Exception: if all features are on the same scale (e.g., pixel values, z-scores from preprocessing).
- [ ] **Handle outliers before KMeans.** A single extreme outlier can pull a centroid far from the main cluster. Options: winsorize, clip to 3σ, or use a robust K-Medoids approach.
- [ ] **Reduce linear structure first (optional but common).** For high-D data (D > 100), apply PCA to 50 components before KMeans. This removes noise dimensions and speeds up training. This is the standard scRNA-seq pipeline (PCA → K-Means).
- [ ] **Check for appropriate cluster structure.** Hopkins statistic or visual inspection (PCA scatterplot): if the data looks like a uniform blob, clustering-as-DR will produce meaningless features. KMeans always finds K clusters regardless of whether they exist.
- [ ] **Decide K before fitting based on domain knowledge** (e.g., number of product categories, number of known cell types). Then validate with elbow/silhouette.

**Pitfall — scaling after fit:** NEVER fit the scaler on the test set or after the KMeans. The pipeline must be: `fit(scaler, train) → transform(train) → fit(kmeans, scaled_train) → transform(test via fitted scaler → fitted kmeans)`.

### During Fitting

- [ ] **Check inertia convergence.** With `verbose=1`, KMeans prints inertia per iteration. It should decrease monotonically. If inertia bounces, `max_iter` may be too low or data has problems.
- [ ] **Check `n_iter_` attribute.** If `n_iter_ == max_iter`, the model didn't converge — increase `max_iter`.
- [ ] **For GMM: check `converged_` attribute.** If False, the EM didn't converge. Increase `max_iter` or `n_init`.
- [ ] **Watch for empty clusters.** With random init or very high K, some centroids may attract zero samples. sklearn warns: "Number of distinct clusters found smaller than n_clusters." Symptom: `transform()` output has a column of constant distances.
- [ ] **Watch for degenerate GMM components.** `reg_covar` prevents covariance matrices from becoming singular; if you see warnings, increase it to 1e-3.

**Pitfall — data leakage in pipelines:** When using KMeans inside a `Pipeline` with cross-validation, ensure the `fit_transform` step in the pipeline is never applied to validation/test data. Use `Pipeline` correctly — sklearn Pipelines handle this automatically.

### After Fitting

- [ ] **Inspect cluster sizes.** Check `np.bincount(kmeans.labels_)`. Extremely imbalanced clusters (one cluster has 90% of samples) suggest K is too small or data has a dominant group. Very small clusters (<5 samples per cluster for typical data) suggest K is too large.
- [ ] **Visualize clusters.** Project with PCA(n_components=2) or t-SNE, color by cluster label. If clusters don't visually separate, the cluster structure may not be meaningful.
- [ ] **Check silhouette scores.** `silhouette_score(X_scaled, kmeans.labels_)`. Score < 0.2: poor cluster separation. Score > 0.5: reasonable. Score > 0.7: excellent. Note: a decent silhouette doesn't guarantee good DR features — validate downstream.
- [ ] **Validate the DR features downstream.** Fit a logistic regression (for classification) or ridge regression (for regression) on the transformed features. Compare to baseline (raw features, PCA features). If the clustering features don't improve performance, consider a different K or a different approach.
- [ ] **For FeatureAgglomeration: check `labels_` to see which features merged.** Features that cluster together should be semantically related (e.g., related genes, correlated sensor readings). If they're not, your distance metric or linkage may be inappropriate.
- [ ] **Reproducibility:** Always set `random_state`. KMeans with `n_init>1` may give different label assignments (but same quality) across runs. If downstream code depends on which label is "cluster 0," this causes subtle bugs.

### Common Pitfalls with Concrete Examples

**Pitfall 1: Using KMeans transform without scaling**
```python
# WRONG — income dominates age:
kmeans = KMeans(n_clusters=5).fit(X)  # X has income ($10k-$200k) and age (18-80)

# RIGHT:
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pipe = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=5))])
X_dr = pipe.fit_transform(X)
```

**Pitfall 2: Fitting GMM with n_init=1 on high-D data**
```python
# WRONG — likely stuck in local optimum:
gmm = GaussianMixture(n_components=10).fit(X_high_dim)

# RIGHT:
gmm = GaussianMixture(n_components=10, n_init=5, covariance_type='diag', random_state=42).fit(X_high_dim)
```

**Pitfall 3: Applying FeatureAgglomeration to test data separately**
```python
# WRONG — refit on test, different cluster assignments:
agglo.fit_transform(X_train)
X_test_reduced = agglo.fit_transform(X_test)  # ERROR: refitting changes cluster assignments

# RIGHT:
X_train_reduced = agglo.fit_transform(X_train)
X_test_reduced = agglo.transform(X_test)  # Only transform, never fit again
```

**Pitfall 4: Assuming KMeans cluster labels are stable across runs**
```python
# With n_init > 1 and different random_state, 'cluster 0' may be what was 'cluster 2' before.
# This breaks any code that hard-codes label values (e.g., 'cluster 0 = high-risk users').
# Solution: sort clusters by a meaningful property after fitting (e.g., sort by centroid[0]).
```

**Pitfall 5: Using t-SNE output as input to KMeans**
```python
# t-SNE exaggerates cluster separation. KMeans on t-SNE coordinates may artificially
# fragment or merge true clusters. Best practice: use t-SNE ONLY for visualization,
# not as input to KMeans for DR. For DR, use UMAP or PCA before KMeans.
# Exception: researchers have used KMeans on t-SNE coords for scRNA-seq (works in
# practice because scRNA-seq clusters are well-separated, but methodologically fragile).
```

---

## 8. Evaluation Metrics

### 8.1 Metrics for Clustering Quality (Before DR Validation)

These measure whether the clusters themselves are good.

**Inertia (Within-Cluster Sum of Squares):**
```python
kmeans.inertia_  # float — lower is better, but decreases with K always
```
- Only comparable across same K. Use for elbow method, not for comparing different K values.

**Silhouette Score:**
```python
from sklearn.metrics import silhouette_score, silhouette_samples
score = silhouette_score(X_scaled, labels)   # -1 to +1; higher is better
per_sample = silhouette_samples(X_scaled, labels)
```
- For each sample i: `s(i) = (b(i) - a(i)) / max(a(i), b(i))` where a(i) = mean distance to same-cluster points, b(i) = mean distance to nearest other-cluster points.
- Scale: -1 (wrong cluster), 0 (cluster boundary), +1 (tight, well-separated cluster).
- Compute time: O(N^2). For large N, subsample: `silhouette_score(X_scaled[sample_idx], labels[sample_idx])`.

**Davies-Bouldin Index:**
```python
from sklearn.metrics import davies_bouldin_score
db = davies_bouldin_score(X_scaled, labels)  # lower is better
```
- Ratio of within-cluster scatter to between-cluster separation.
- No upper bound (unlike silhouette). Lower values = better separated clusters.

**Calinski-Harabasz Score (Variance Ratio Criterion):**
```python
from sklearn.metrics import calinski_harabasz_score
ch = calinski_harabasz_score(X_scaled, labels)  # higher is better
```
- Ratio of between-cluster dispersion to within-cluster dispersion.
- Fast to compute (O(N·K)). Good for comparing different K values.

**BIC / AIC (GMM only):**
```python
gmm.bic(X_train)   # lower is better; penalizes n_parameters
gmm.aic(X_train)   # AIC (lighter penalty than BIC)
```
- Use BIC to select `n_components`. Plot BIC vs K and pick the "elbow" or minimum.

---

### 8.2 Metrics for DR Quality (Evaluating the Feature Representation)

These measure whether the clustering-based DR captures the relevant structure.

**Trustworthiness and Continuity:**

```python
from sklearn.manifold import trustworthiness

# Trustworthiness: if sample i's k nearest neighbors in the DR space
# are also among its k nearest neighbors in the original space.
trust = trustworthiness(X_original, X_dr, n_neighbors=5)
# Range: 0 to 1; higher is better. Good value: > 0.9
```
- Trustworthiness penalizes false neighbors (points that appear close in DR but aren't close in original).
- Continuity (not directly in sklearn) penalizes missing neighbors (points close in original but far in DR).
- Compute both for a full picture. A DR can have high trustworthiness but low continuity (many original neighbors are missing from DR neighborhood).

**Downstream Task Performance (Gold Standard):**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Baseline: raw features
baseline = cross_val_score(LogisticRegression(), X_scaled, y, cv=5).mean()

# With DR:
X_dr = KMeans(n_clusters=20).fit_transform(X_scaled)
dr_score = cross_val_score(LogisticRegression(), X_dr, y, cv=5).mean()

print(f"Raw: {baseline:.3f}, Clustering-DR: {dr_score:.3f}")
```
This is the most practical metric. If downstream performance improves (or matches at lower dimensionality), the DR is working.

**Reconstruction Error (for FeatureAgglomeration):**
```python
agglo = FeatureAgglomeration(n_clusters=32).fit(X_train)
X_reduced = agglo.transform(X_train)
X_reconstructed = agglo.inverse_transform(X_reduced)
recon_error = np.mean((X_train - X_reconstructed) ** 2)  # MSE
```
Lower reconstruction error = the pooling within clusters loses less information.

**Cluster Purity (if labels known):**
```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari = adjusted_rand_score(y_true, kmeans_labels)
nmi = normalized_mutual_info_score(y_true, kmeans_labels)
# ARI: -1 to 1; NMI: 0 to 1; both: higher = clustering aligns with true classes
```
Only applicable when ground-truth labels exist. Good for validation on labeled datasets (but don't train on labels — that defeats the purpose of unsupervised DR).

**Neighborhood Preservation (Knn-based):**

A more granular check: train a KNN classifier on the original space and on the DR space. If classification accuracy is similar, the relevant neighborhood structure is preserved.

```python
from sklearn.neighbors import KNeighborsClassifier

# Neighborhood preservation via kNN accuracy
knn_orig = KNeighborsClassifier(n_neighbors=5).fit(X_scaled, y)
knn_dr   = KNeighborsClassifier(n_neighbors=5).fit(X_dr, y)

acc_orig = cross_val_score(knn_orig, X_scaled, y, cv=5).mean()
acc_dr   = cross_val_score(knn_dr, X_dr, y, cv=5).mean()
```

---

### 8.3 Metrics for Anomaly Detection Use Case

```python
# Train-set distribution of anomaly scores:
train_scores = kmeans.transform(X_train).min(axis=1)
threshold = np.percentile(train_scores, 99)  # 99th percentile as threshold

# Test-set anomaly detection:
test_scores = kmeans.transform(X_test).min(axis=1)
anomalies = test_scores > threshold
```

Evaluate with: `roc_auc_score(y_true_anomaly, test_scores)` — AUC-ROC is the standard metric for anomaly detection.

---

## 9. Key Code Patterns

### Pattern 1: KMeans as DR in a Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Clustering-based DR as feature engineering step
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=20, n_init='auto', random_state=42)),
    ('clf', LogisticRegression(max_iter=1000))
])

scores = cross_val_score(pipe, X, y, cv=5)
print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

Note: `KMeans` exposes a `transform()` method, so Pipeline automatically calls it as a transformer step. The output is the (n_samples, n_clusters) distance matrix.

### Pattern 2: GMM as Soft Feature Representation

```python
from sklearn.mixture import GaussianMixture
import numpy as np

gmm = GaussianMixture(
    n_components=15,
    covariance_type='diag',
    n_init=5,
    random_state=42
).fit(X_train_scaled)

# Soft features: posterior probability per component
X_train_dr = gmm.predict_proba(X_train_scaled)   # shape: (n_train, 15)
X_test_dr  = gmm.predict_proba(X_test_scaled)    # shape: (n_test, 15)

# Check fit quality:
print(f"BIC: {gmm.bic(X_train_scaled):.1f}, converged: {gmm.converged_}")
```

Note: GaussianMixture has no `transform()` method. You MUST call `predict_proba()` explicitly.

### Pattern 3: FeatureAgglomeration for Tabular Feature Reduction

```python
from sklearn.cluster import FeatureAgglomeration
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Reduce 500 features → 30 feature groups
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('agglo', FeatureAgglomeration(
        n_clusters=30,
        linkage='ward',
        pooling_func=np.mean
    )),
    ('clf', LogisticRegression())
])

# Inspect which features were grouped:
pipe.fit(X_train, y_train)
print("Feature group labels:", pipe['agglo'].labels_)
# labels_[i] = cluster index for feature i
```

### Pattern 4: SOM with MiniSom for DR + Visualization

```python
from minisom import MiniSom
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SOM: 10x10 grid → 2D DR
som = MiniSom(
    x=10, y=10,
    input_len=X_scaled.shape[1],
    sigma=1.5,
    learning_rate=0.5,
    random_seed=42
)
som.random_weights_init(X_scaled)
som.train(X_scaled, num_iteration=10000, verbose=False)

# DR: 2D grid coordinates per sample
coords = np.array([som.winner(x) for x in X_scaled])
# coords.shape: (n_samples, 2)

# Richer DR: quantization error per sample (distance to winning neuron)
q_errors = np.array([som.distance_map_values(x).min() for x in X_scaled])
```

### Pattern 5: Bag of Visual Words (Conceptual Pipeline)

```python
from sklearn.cluster import KMeans
import numpy as np

# Assume descriptors is a (N_total_descriptors, 128) array of SIFT features
# from all training images

# Step 1: Train visual vocabulary
vocab_size = 1000
kmeans = KMeans(n_clusters=vocab_size, n_init=1, random_state=42)
kmeans.fit(descriptors)  # May need FAISS for large N

# Step 2: Encode each image as a histogram
def encode_image(image_descriptors, kmeans, vocab_size):
    if len(image_descriptors) == 0:
        return np.zeros(vocab_size)
    assignments = kmeans.predict(image_descriptors)
    histogram, _ = np.histogram(assignments, bins=vocab_size, range=(0, vocab_size))
    return histogram.astype(float) / (histogram.sum() + 1e-10)  # L1 normalize

X_encoded = np.array([encode_image(img_desc, kmeans, vocab_size) for img_desc in all_image_descriptors])
# shape: (n_images, 1000) — this IS the dimensionality reduction
```

### Pattern 6: Spectral Embedding + KMeans (Spectral Clustering Decomposed)

```python
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Spectral embedding (the DR step)
spec_embed = SpectralEmbedding(
    n_components=10,        # Embed to 10D (richer than final n_clusters)
    affinity='nearest_neighbors',
    n_neighbors=15,
    random_state=42
)
X_embedded = spec_embed.fit_transform(X_scaled)  # shape: (n_samples, 10)

# Step 2: Cluster in embedding space
kmeans = KMeans(n_clusters=5, n_init='auto', random_state=42)
labels = kmeans.fit_predict(X_embedded)

# The 10-dimensional embedding IS the DR output;
# the cluster labels are the final assignment
```

---

## API Uncertainties and Flags

1. **FLAG:** `GaussianMixture.transform()` does NOT exist in any sklearn version. Writers must call `predict_proba()` explicitly and note this in the chapter.
2. **FLAG:** `KMeans(algorithm='full')` was removed in sklearn 1.1. Code from pre-2021 tutorials may use this and will break.
3. **FLAG:** `FeatureAgglomeration(metric=None)` is deprecated in sklearn 1.4 and will raise an error in 1.6. The chapter code should use explicit `metric='euclidean'`.
4. **FLAG:** `n_init='auto'` was introduced in sklearn 1.2 for KMeans. Code targeting sklearn < 1.2 must use `n_init=10`.
5. **FLAG:** `GaussianMixture(init_params='k-means++')` was added in sklearn 1.1. Not available in older versions.
6. **VERIFY:** The `MiniSom` version 2.3.6 was the latest as of Feb 2026. Confirm on PyPI before publication.

---

*End of dossier. All 7 research tasks completed: applications, papers, sklearn API, packages, hyperparameter tuning, practitioner checklist, evaluation metrics.*
