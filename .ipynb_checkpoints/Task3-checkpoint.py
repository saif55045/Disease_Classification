# Re-run after kernel reset
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.base import BaseEstimator, ClassifierMixin

# Load TF-IDF matrix
tfidf_matrix = load_npz("tfidf_combined_matrix.npz")

# Load one-hot encoded matrix
onehot_df = pd.read_csv("encoded_output2.csv")
onehot_matrix = onehot_df.drop(columns=["Disease"]).values
labels = onehot_df["Disease"].values

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Custom KNN classifier to support Manhattan and Cosine with sparse matrices
class CustomKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=3, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        dists = pairwise_distances(X, self.X_train, metric=self.metric)
        neighbors = np.argsort(dists, axis=1)[:, :self.n_neighbors]
        pred_labels = []
        for i in range(neighbors.shape[0]):
            neighbor_labels = self.y_train[neighbors[i]]
            counts = np.bincount(neighbor_labels)
            pred_labels.append(np.argmax(counts))
        return np.array(pred_labels)

# Metrics function
def evaluate_model(X, y, model, name, cv_splits=5):
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=skf)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y, y_pred, average='weighted', zero_division=0),
        "F1-Score": f1_score(y, y_pred, average='weighted', zero_division=0)
    }

# Configurations
k_values = [1]
metrics = ['euclidean', 'manhattan', 'cosine']
results = []

# Evaluate KNN on both matrices
for k in k_values:
    for metric in metrics:
        knn_model_tfidf = CustomKNN(n_neighbors=k, metric=metric)
        result = evaluate_model(tfidf_matrix, y, knn_model_tfidf, f"KNN (TF-IDF) k={k}, {metric}")
        results.append(result)

        knn_model_onehot = CustomKNN(n_neighbors=k, metric=metric)
        result = evaluate_model(onehot_matrix, y, knn_model_onehot, f"KNN (One-Hot) k={k}, {metric}")
        results.append(result)

# Evaluate Logistic Regression
logreg_tfidf = LogisticRegression(max_iter=1000)
results.append(evaluate_model(tfidf_matrix, y, logreg_tfidf, "Logistic Regression (TF-IDF)"))

logreg_onehot = LogisticRegression(max_iter=1000)
results.append(evaluate_model(onehot_matrix, y, logreg_onehot, "Logistic Regression (One-Hot)"))

# Final results
results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values(by="F1-Score", ascending=False)
results_df_sorted.reset_index(drop=True, inplace=True)
results_df_sorted.head(10)  # Show top 10 for readability
