# Meta-DR: Meta-Learning for Dimensionality Reduction Recommendation

A meta-learning framework for the automated selection of dimensionality reduction (DR) techniques in high-dimensional data scenarios.

---

## Introduction

High-dimensional data often degrades the performance of machine learning algorithms due to sparsity, noise, distance concentration, and overfitting. **Dimensionality reduction** simplifies data and can improve efficiency and generalization, but choosing the best DR technique for each dataset is difficult and costly. **Meta-learning (MtL)** addresses this by learning from past experience: it uses dataset characteristics (meta-features) and historical performance of DR algorithms to recommend suitable techniques for new datasets without exhaustive evaluation. This repository implements **Meta-DR**, a framework that learns the mapping between dataset properties and DR performance rankings and recommends techniques for unseen data.

---

## Objective and Motivation

The **Algorithm Selection Problem**—choosing the best algorithm for a given dataset—is a central bottleneck in ML. For dimensionality reduction, the effectiveness of each technique depends on the data; no single method is best for all cases. Manually testing many DR methods is time-consuming and computationally expensive.

**Meta-DR** aims to:
- **Automate** the selection of DR techniques using meta-learning.
- **Reduce** the need for repeated evaluation of all candidates on each new dataset.
- **Learn** from prior experiments: dataset meta-features and performance rankings are used to train a meta-model that predicts a ranking of DR techniques for new datasets.

The main idea is that datasets with similar meta-feature profiles tend to benefit from similar DR techniques; the meta-learner exploits this to generalize and recommend effectively.

---

## Methodology

The framework has two phases:

1. **Construction (offline):** For a set of datasets, extract meta-features, apply each DR technique, evaluate classification performance (e.g. F1 with KNN), and build a meta-dataset of (meta-features, DR performance rankings). A meta-learner (e.g. Random Forest) is trained to predict rankings from meta-features.
2. **Recommendation (online):** For a new dataset, extract the same meta-features and use the trained meta-model to predict a ranking of DR techniques—without running all DR algorithms.

The figure below summarizes the workflow.

![Meta-DR operational workflow](images/proposedFramework.pdf)

*Figure 1: Operational workflow of Meta-DR (construction and recommendation phases).*

---

## Results

Experiments used **68 high-dimensional datasets** (OpenML), **10 DR techniques** (PCA, Kernel PCA, LDA, t-SNE, LLE, Truncated SVD, Incremental PCA, Random Trees Embedding, SelectKBest, Spectral Embedding), and meta-features from **General**, **Statistical**, and **Information-theoretic** categories.

### Dataset and meta-feature analysis

![Dataset summary](images/dataset_summary.pdf)

*Figure 2: Summary statistics of the 68 datasets (instances, features, classes).*

![Meta-feature correlation](images/correlation_heatmap.pdf)

*Figure 3: Correlation among meta-features (original set).*

![Meta-feature correlation reduced](images/correlation_heatmap_reduced.pdf)

*Figure 4: Correlation after removing highly correlated meta-features.*

![Feature importance](images/feature_importance.pdf)

*Figure 5: Meta-feature importance from the Random Forest meta-learner.*

### Ranking prediction (Spearman correlation)

The meta-model was evaluated with **Spearman rank correlation (SRC)** between predicted and observed DR rankings. Meta-DR outperformed non-learning baselines (mean and median ranking predictors).

![Mean SRC](images/src_distributionbarIC.pdf)

*Figure 6: Mean Spearman rank correlation for Meta-DR variants and baselines.*

![SRC distribution](images/src_distributionboxplot.pdf)

*Figure 7: Distribution of SRC across datasets.*

### Classification performance

Using the **top-ranked** DR technique recommended by Meta-DR per dataset led to higher average classification performance than fixed DR choices and baselines. Recommending a **small subset** (e.g. best of 2 or 3) further improved robustness.

![Average performance](images/Average-PerformanceBarIC_3_short.pdf)

*Figure 8: Average F1-score of Meta-DR variants, individual DR techniques, and baselines.*

![Performance box plots](images/Average-Performancebox.pdf)

*Figure 9: Distribution of classification performance (box plots).*

### Critical difference diagram

![CD diagram](images/cd_diagram.pdf)

*Figure 10: Critical difference diagram. Methods connected by a horizontal line are not statistically different (Nemenyi test). Meta-DR variants appear among the top-ranked approaches.*

---

## Conclusion

Meta-DR shows that **meta-learning is an effective and viable strategy** for recommending dimensionality reduction techniques. The framework:

- **Outperforms** simple baselines in terms of rank correlation (predicted vs. observed rankings) and base-level classification performance.
- **Shifts cost** to an offline construction phase; recommendation for new datasets is a low-cost inference step.
- **Benefits** from dataset characterization via meta-features, with statistical and information-theoretic meta-features being particularly informative.

Results support the use of data-driven, meta-learning-based selection of DR techniques in high-dimensional and big data scenarios, reducing the need for manual trial-and-error while achieving competitive performance. Future work may include multiple base classifiers, more DR methods (e.g. deep learning-based), and integration into full AutoML pipelines.

---

## Repository structure and usage

- **`Meta_DR.ipynb`**: Main notebook with the full pipeline (meta-feature extraction, DR evaluation, meta-dataset construction, meta-learner training, and evaluation).
- **`data/`**: Processed data (baseline classification, DR results, meta-features, predictions).
- **`images/`**: Figures used in the paper and in this README.

### Running on your PC

1. **Clone the repo** and open the project folder (e.g. `meta-dr/`).
2. **Optional but recommended:** create a virtual environment so dependencies stay isolated:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # or:  .venv\Scripts\activate   # Windows
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Open and run the notebook** from the **project root** (so paths like `data/...` work). Use Jupyter, VS Code, or any environment that runs `.ipynb` files.

The first cell installs `openml`, `pymfe`, and `mlxtend` if you are on Google Colab; when running locally, those packages are provided by `requirements.txt`. The notebook detects local runs and skips Google Drive mount, so no `.venv` or Colab is required—but using a `.venv` is recommended to avoid conflicts with other Python projects.
