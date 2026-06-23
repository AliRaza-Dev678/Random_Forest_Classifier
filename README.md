# 🌳 Random Forest Classifier — Iris Dataset

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-red?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> A supervised machine learning project implementing a **Random Forest Classifier** on the classic **Iris Dataset** — exploring ensemble learning, decision tree aggregation, feature importance, and comprehensive model evaluation.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [What is Random Forest?](#-what-is-random-forest)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Key Concepts Covered](#-key-concepts-covered)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🔍 Overview

This project applies the **Random Forest** ensemble learning algorithm to classify iris flowers into three species — *Setosa*, *Versicolor*, and *Virginica* — based on their sepal and petal measurements. By combining the predictions of multiple decision trees, Random Forest achieves higher accuracy and better generalization than any single tree alone. The full pipeline — from EDA to feature importance analysis — is implemented in a Jupyter Notebook.

---

## 🧠 What is Random Forest?

**Random Forest** is an ensemble learning method that builds a large collection of decision trees during training and outputs the **majority vote** (for classification) across all trees.

```
Final Prediction = Majority Vote of N Decision Trees
```

Each tree is trained on a **random bootstrap sample** of the data, and at every split, only a **random subset of features** is considered. This double injection of randomness makes the trees decorrelated and the ensemble robust.

| Property | Description |
|---|---|
| **Ensemble Method** | Combines many weak learners into one strong learner |
| **Bagging** | Each tree is trained on a random sample with replacement |
| **Feature Randomness** | Each split considers only a random subset of features |
| **Voting** | Final class = class predicted by the most trees |
| **Feature Importance** | Measures how much each feature reduces impurity across trees |

**Why Random Forest outperforms a single Decision Tree:**
- Reduces **overfitting** by averaging many trees
- More **robust to noise** in the data
- Naturally provides **feature importance** scores
- Works well **out-of-the-box** with minimal tuning

---

## 📊 Dataset

The **Iris Dataset** is one of the most well-known datasets in machine learning, introduced by statistician Ronald Fisher in 1936.

| Property | Details |
|---|---|
| **Samples** | 150 flower instances |
| **Features** | 4 numerical measurements |
| **Classes** | 3 species (50 samples each) |
| **Missing Values** | None |
| **Source** | Built into `sklearn.datasets` |

### Features

| Feature | Description |
|---|---|
| `sepal length (cm)` | Length of the sepal |
| `sepal width (cm)` | Width of the sepal |
| `petal length (cm)` | Length of the petal |
| `petal width (cm)` | Width of the petal |

### Target Classes

| Label | Species |
|---|---|
| 0 | *Iris Setosa* |
| 1 | *Iris Versicolor* |
| 2 | *Iris Virginica* |

---

## ⚙️ Methodology

### 1. Data Loading & Exploration
- Load the Iris dataset directly from `sklearn.datasets`
- Inspect shape, class distribution, and descriptive statistics
- Verify no missing values across all features

### 2. Exploratory Data Analysis (EDA)
- **Pairplots** to visualize pairwise feature relationships colored by class
- **Box plots** to examine feature spread and outliers per species
- **Correlation heatmap** to identify highly correlated features

### 3. Data Preprocessing
- Split the dataset into **training (80%) and test (20%) sets** with stratified sampling
- Apply **StandardScaler** for feature normalization

### 4. Model Training
- Train a **Random Forest Classifier** using scikit-learn
- Configure `n_estimators` (number of trees) and other key hyperparameters
- Each tree is grown on a bootstrap sample using random feature subsets at each split

### 5. Hyperparameter Exploration
- Tune key parameters:
  - `n_estimators` — number of trees in the forest
  - `max_depth` — maximum depth of each tree
  - `min_samples_split` — minimum samples required to split a node
  - `max_features` — number of features considered at each split

### 6. Feature Importance
- Extract and visualize **feature importance scores** from the trained forest
- Identify which measurements (sepal/petal dimensions) drive classification most

### 7. Model Evaluation
- Evaluate with **Accuracy Score**, **Classification Report**, and **Confusion Matrix**
- Visualize confusion matrix as a labeled heatmap

---

## 📚 Key Concepts Covered

| Concept | Description |
|---|---|
| **Ensemble Learning** | Combining multiple models for stronger predictions |
| **Bagging (Bootstrap Aggregating)** | Training each tree on a random sample with replacement |
| **Decision Tree** | Base learner that splits data on feature thresholds |
| **Gini Impurity** | Measure of node purity used to find optimal splits |
| **Feature Importance** | Contribution of each feature to reducing impurity across all trees |
| **Out-of-Bag (OOB) Error** | Built-in validation using samples not in a tree's bootstrap sample |
| **Majority Voting** | Final prediction aggregated from all individual tree predictions |
| **Overfitting Prevention** | Achieved through decorrelated trees via bagging and feature randomness |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **Jupyter Notebook** | Interactive development environment |
| **Pandas** | Data manipulation and tabular display |
| **NumPy** | Numerical computations |
| **Matplotlib** | Plots and visualizations |
| **Seaborn** | Statistical plots and heatmaps |
| **scikit-learn** | Dataset, Random Forest model, preprocessing, and evaluation |

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ and pip installed.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/AliRaza-Dev678/Random_Forest_Classifier.git
cd Random_Forest_Classifier

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the Notebook

```bash
jupyter notebook Random_Forest_Tree.ipynb
```

> ✅ No external dataset download needed — the Iris dataset is built directly into scikit-learn.

---

## 📁 Project Structure

```
Random_Forest_Classifier/
│
├── Random_Forest_Tree.ipynb    # Main notebook — full pipeline
└── README.md                   # Project documentation
```

---

## 📈 Model Performance

Random Forest consistently achieves near-perfect accuracy on the Iris dataset, significantly outperforming a single Decision Tree due to ensemble averaging.

| Metric | Expected Range |
|---|---|
| **Accuracy** | ~96 – 100% |
| **Precision (macro avg)** | ~96 – 100% |
| **Recall (macro avg)** | ~96 – 100% |
| **F1-Score (macro avg)** | ~96 – 100% |

**Feature Importance (typical ranking):**

| Rank | Feature | Importance |
|---|---|---|
| 1st | `petal length (cm)` | Highest |
| 2nd | `petal width (cm)` | High |
| 3rd | `sepal length (cm)` | Moderate |
| 4th | `sepal width (cm)` | Lowest |

- *Iris Setosa* is **linearly separable** and classified perfectly by nearly every tree
- *Iris Versicolor* and *Iris Virginica* have slight feature overlap, but ensemble voting resolves most ambiguities

> Run the notebook for exact metric outputs and feature importance bar charts.

---

## 🔭 Future Improvements

- Apply **GridSearchCV** or **RandomizedSearchCV** for systematic hyperparameter tuning
- Visualize **individual decision trees** within the forest using `plot_tree`
- Enable **Out-of-Bag (OOB) scoring** (`oob_score=True`) as a free validation metric
- Compare Random Forest against **SVM**, **Naive Bayes**, **KNN**, and **Gradient Boosting**
- Apply **cross-validation (k-fold)** for more robust accuracy estimates
- Test on real-world datasets beyond Iris (e.g., breast cancer, wine quality)
- Export the trained model with **joblib** for reuse and deployment

---

## 👤 Author

**Ali Raza**

[![GitHub](https://img.shields.io/badge/GitHub-AliRaza--Dev678-black?style=flat-square&logo=github)](https://github.com/AliRaza-Dev678)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it.

---

> ⭐ If you found this project useful, consider giving it a star on GitHub!
