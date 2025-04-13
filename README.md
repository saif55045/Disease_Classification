# 🧠 Disease Classification with KNN & Logistic Regression

This project explores machine learning techniques for classifying diseases using **K-Nearest Neighbors (KNN)** and **Logistic Regression**, comparing the effectiveness of **TF-IDF** vs **One-Hot Encoding** for medical feature extraction.

---


---

## ✅ Tasks Breakdown

### 📌 Task 1: TF-IDF Feature Extraction
- Parsed stringified lists into usable lists.
- Applied `TfidfVectorizer` to **Risk Factors**, **Symptoms**, **Signs**, and **Subtypes**.
- Combined into a single sparse matrix.
- Compared TF-IDF with one-hot matrix (sparsity, shape, feature count).

### 🎯 Task 2: Dimensionality Reduction
- Applied **PCA** to One-Hot and **Truncated SVD** to TF-IDF.
- Visualized results in 2D using Matplotlib.
- Clustered diseases into categories: cardiovascular, neurological, respiratory, metabolic, infectious.

### 🧪 Task 3: Model Training & Evaluation
- Trained **KNN models** (k = 3, 5, 7) using:
  - Euclidean
  - Manhattan
  - Cosine distance
- Trained **Logistic Regression** models.
- Evaluated with **5-fold cross-validation**.
- Reported Accuracy, Precision, Recall, F1-Score.
- Compared results across:
  - Encoding methods
  - Model types
  - Distance metrics

### 🧠 Task 4: Critical Analysis
- Analyzed when and why TF-IDF may outperform One-Hot.
- Discussed clinical relevance of clusters.
- Described limitations of each method.

---

## 🚀 Streamlit Web App

A simple web application to:
- Input disease features
- Select encoding type (TF-IDF or One-Hot)
- Choose K and distance metric
- Get prediction on disease category

To run locally:

```bash
pip install streamlit pandas scikit-learn joblib scipy
streamlit run knn_streamlit_app.py
```
# 📊 Results Summary


| Model               | Encoding | K   | Distance    | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----|-------------|----------|-----------|--------|----------|
| KNN                | One-Hot  | 3   | Cosine      | 0.48     | 0.267     | 0.425  | 0.313    |
| KNN                | One-Hot  | 3   | Euclidean   | 0.44     | 0.256     | 0.400  | 0.278    |
| KNN                | One-Hot  | 3   | Manhattan   | 0.44     | 0.256     | 0.400  | 0.278    |
| KNN                | One-Hot  | 5   | Cosine      | 0.44     | 0.267     | 0.400  | 0.295    |
| KNN                | One-Hot  | 5   | Euclidean   | 0.36     | 0.163     | 0.325  | 0.192    |
| KNN                | One-Hot  | 5   | Manhattan   | 0.36     | 0.163     | 0.325  | 0.192    |
| KNN                | One-Hot  | 7   | Cosine      | 0.44     | 0.296     | 0.375  | 0.302    |
| KNN                | One-Hot  | 7   | Euclidean   | 0.36     | 0.180     | 0.300  | 0.185    |
| KNN                | One-Hot  | 7   | Manhattan   | 0.36     | 0.180     | 0.300  | 0.185    |
| KNN                | TF-IDF   | 3   | Cosine      | 0.52     | 0.317     | 0.450  | 0.348    |
| KNN                | TF-IDF   | 3   | Euclidean   | 0.40     | 0.225     | 0.350  | 0.250    |
| KNN                | TF-IDF   | 3   | Manhattan   | 0.32     | 0.080     | 0.250  | 0.119    |
| KNN                | TF-IDF   | 5   | Cosine      | 0.56     | 0.370     | 0.500  | 0.410    |
| KNN                | TF-IDF   | 5   | Euclidean   | 0.44     | 0.248     | 0.375  | 0.270    |
| KNN                | TF-IDF   | 5   | Manhattan   | 0.32     | 0.080     | 0.250  | 0.119    |
| KNN                | TF-IDF   | 7   | Cosine      | 0.64     | 0.445     | 0.600  | 0.494    |
| KNN                | TF-IDF   | 7   | Euclidean   | 0.36     | 0.147     | 0.300  | 0.187    |
| KNN                | TF-IDF   | 7   | Manhattan   | 0.32     | 0.080     | 0.250  | 0.119    |
| Logistic Regression| One-Hot  | -   | -           | 0.44     | 0.296     | 0.400  | 0.285    |
| Logistic Regression| TF-IDF   | -   | -           | 0.44     | 0.220     | 0.400  | 0.265    |

---

# 📘 Blog & LinkedIn Post

**📝 Medium Blog:**  
*How Machine Learning Can Classify Diseases: TF-IDF vs One-Hot* → [link here]

**🔗 LinkedIn Post:**  
[link here]

---

# 📌 Dependencies

- Python 3.8+
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `joblib`
- `streamlit`

---

# 💬 Author

**[Saif Ullah]**  
*Data Science & Machine Learning Enthusiast*  
[LinkedIn](https://www.linkedin.com/in/saif-ullah-5ba1b1140/) | [GitHub](https://github.com/saif55045)
