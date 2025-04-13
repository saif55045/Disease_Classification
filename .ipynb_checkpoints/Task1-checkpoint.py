import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

from scipy.sparse import save_npz
import joblib




# Step 1: Load the disease features dataset
df = pd.read_csv("disease_features.csv")

# Step 2: Parse and convert lists into strings
def parse_and_flatten_column(column):
    return df[column].apply(lambda x: " ".join(ast.literal_eval(x)) if pd.notna(x) and x.strip() else "")

df["Risk Factors Clean"] = parse_and_flatten_column("Risk Factors")
df["Symptoms Clean"] = parse_and_flatten_column("Symptoms")
df["Signs Clean"] = parse_and_flatten_column("Signs")
df["Subtypes Clean"] = parse_and_flatten_column("Subtypes")  # ✅ Now including Subtypes

# Step 3 & 4: TF-IDF vectorization for each column
vectorizer_risk = TfidfVectorizer()
vectorizer_symptoms = TfidfVectorizer()
vectorizer_signs = TfidfVectorizer()
vectorizer_subtypes = TfidfVectorizer()

tfidf_risk = vectorizer_risk.fit_transform(df["Risk Factors Clean"])
tfidf_symptoms = vectorizer_symptoms.fit_transform(df["Symptoms Clean"])
tfidf_signs = vectorizer_signs.fit_transform(df["Signs Clean"])
tfidf_subtypes = vectorizer_subtypes.fit_transform(df["Subtypes Clean"])  # ✅

# Step 5: Combine all TF-IDF matrices into one
tfidf_combined = hstack([tfidf_risk, tfidf_symptoms, tfidf_signs, tfidf_subtypes])

# Step 6: Compare with the one-hot encoded matrix
encoded_df = pd.read_csv("encoded_output2.csv")
encoded_matrix = encoded_df.drop(columns=["Disease"]).values

# Step 7: Summary comparison
comparison = {
    "TF-IDF Matrix Shape": tfidf_combined.shape,
    "One-Hot Matrix Shape": encoded_matrix.shape,
    "TF-IDF Sparsity (%)": 100.0 * (1.0 - tfidf_combined.nnz / (tfidf_combined.shape[0] * tfidf_combined.shape[1])),
    "One-Hot Sparsity (%)": 100.0 * (1.0 - (encoded_matrix != 0).sum() / encoded_matrix.size),
    "TF-IDF Unique Features": tfidf_combined.shape[1],
    "One-Hot Unique Features": encoded_matrix.shape[1],
}

# Save TF-IDF matrix
save_npz("tfidf_combined_matrix.npz", tfidf_combined)

# Save the vectorizers if needed later
joblib.dump(vectorizer_risk, "vectorizer_risk.pkl")
joblib.dump(vectorizer_symptoms, "vectorizer_symptoms.pkl")
joblib.dump(vectorizer_signs, "vectorizer_signs.pkl")
joblib.dump(vectorizer_subtypes, "vectorizer_subtypes.pkl")

# Display results
for key, value in comparison.items():
    print(f"{key}: {value}")
