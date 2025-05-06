## 📁 Project 1: 🧠 Binary Classification — Marketplace Product Matching

**Date:** January 25, 2024 – February 15, 2024  
**Company:** [Samokat.Tech](https://samokat.tech) (Remote)  
**Tech Stack:** Python 3.10, Pandas, NumPy, Scikit-learn, LightGBM, Matplotlib, Seaborn, Jupyter

---

### 📦 Project Summary

This project addresses a real-world challenge from **Samokat.Tech**, a tech-driven, real-time grocery delivery platform.

I developed the **final stage of a product matching pipeline** tasked with determining whether a seller’s offer matched a catalog product.

This was a **binary classification** problem, involving both structured tabular data and high-dimensional **image/text embeddings**.

---

### 🧾 About Samokat.Tech

Samokat.Tech powers one of Russia’s most innovative grocery delivery services. Their fully automated, digitized supply chain helped them grow from:

- 📦 **1.6M orders/month in 2020**
- 🚀 **92M orders in H1 2023**

This growth was made possible by advanced ML, robotics, and scalable IT infrastructure.

---

### 🧠 Problem Formulation

**Objective:** Predict whether a pair of products — one from a seller, one from the internal catalog — are the same item.

- 🔍 **Match:** 1  
- ❌ **No Match:** 0

Each product pair contained:

- Tabular attributes
- Image embeddings
- Text embeddings

**Evaluation Metric:** F1-Score

---

### 📊 Dataset Overview

Data provided by **Megamarket**, included:

#### 📁 Files

- `train.csv` — Labeled training data  
- `test.csv` — Unlabeled test set  

#### 🔑 Key Columns

| Column                   | Description                                  |
|--------------------------|----------------------------------------------|
| `offer_depersonalised`, `goods_depersonalised` | IDs of the seller offer and catalog item |
| `attrs+title_score`      | Pre-scored similarity score                  |
| `sum_length`             | Combined name/attribute lengths              |
| `offer_price`, `item_price` | Item prices                              |
| `target`                 | Match label (only in train)                  |

#### 🔎 Embeddings

- `goods_image_vectors`, `offer_image_vectors`
- `goods_title_vectors`, `offer_title_vectors`

---

### 🔨 Tech Highlights

| Category         | Tools / Techniques                            |
|------------------|------------------------------------------------|
| Language         | Python 3.10                                    |
| ML Libraries     | LightGBM, Scikit-learn                         |
| Data Handling    | Pandas, NumPy                                  |
| Visualization    | Matplotlib, Seaborn                            |
| Embedding Work   | PCA, Distance Metrics                          |
| Development      | Jupyter Notebooks                              |

---

### 🚀 Workflow Highlights

- 🧹 Preprocessed and cleaned tabular and embedding data  
- 🔍 Engineered features from embeddings (distances, PCA)  
- 📉 Applied PCA to reduce dimensionality  
- 🧠 Trained LightGBM model with early stopping  
- 🛠️ Tuned iterations to avoid overfitting  
- 📈 Optimized for better F1 with reduced compute cost  

---

### 🏆 Final Results

| Model                          | F1 Score | Notes                                 |
|--------------------------------|----------|----------------------------------------|
| LightGBM (optimized, 300 iters)| 0.9206   | ✅ Best performance, fast & generalizes well |
| LightGBM (non-optimized, 1500 iters) | 0.9193 | ❌ Overfit and slower                   |

✅ **Final choice:** LightGBM (300 iterations) — faster and more reliable.

---

### 📈 Recommendation

For production use, I recommend integrating **Optuna** for automatic hyperparameter optimization to further boost accuracy and training efficiency.
