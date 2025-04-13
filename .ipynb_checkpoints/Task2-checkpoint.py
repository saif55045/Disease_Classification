import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import matplotlib.pyplot as plt
from scipy.sparse import load_npz


# Load datasets
df = pd.read_csv("disease_features.csv")
encoded_df = pd.read_csv("encoded_output2.csv")
# Load the saved TF-IDF matrix
tfidf_combined = load_npz("tfidf_combined_matrix.npz")


# Prepare one-hot matrix
encoded_matrix = encoded_df.drop(columns=["Disease"]).values

# Dimensionality Reduction
pca_onehot = PCA(n_components=3)
pca_onehot_result = pca_onehot.fit_transform(encoded_matrix)
pca_variance = pca_onehot.explained_variance_ratio_

svd_tfidf = TruncatedSVD(n_components=3)
svd_tfidf_result = svd_tfidf.fit_transform(tfidf_combined)
svd_variance = svd_tfidf.explained_variance_ratio_

# Create mock disease categories for coloring
categories = ['cardiovascular', 'neurological', 'respiratory', 'metabolic', 'infectious']
category_labels = [categories[i % len(categories)] for i in range(len(encoded_df))]
le = LabelEncoder()
color_labels = le.fit_transform(category_labels)

# Plotting function
def plot_2d(data, title, labels):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', s=80, edgecolor='k')
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.colorbar(scatter, ticks=range(len(set(labels))), label='Category')
    plt.show()

# Plot results
plot_2d(pca_onehot_result, "PCA (One-Hot Encoded Matrix)", color_labels)
plot_2d(svd_tfidf_result, "Truncated SVD (TF-IDF Matrix)", color_labels)

# Print variance ratios
print("PCA (One-Hot) Explained Variance Ratio:", pca_variance)
print("Truncated SVD (TF-IDF) Explained Variance Ratio:", svd_variance)
