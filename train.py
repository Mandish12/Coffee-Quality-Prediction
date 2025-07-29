import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
df = pd.read_csv("Coffee_Qlty.csv")

# Select key columns and drop nulls
cupping_cols = ["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", 
                "Balance", "Sweetness", "Clean.Cup", "Uniformity", "Moisture"]
df = df.dropna(subset=cupping_cols)

# Compute average score and classify
df["QualityScore"] = df[cupping_cols[:-1]].mean(axis=1)
df["QualityLabel"] = df["QualityScore"].apply(lambda score:
    "Excellent" if score >= 8.5 else "Good" if score >= 7.5 else "Average")

# Use only first 10 rows and 10 columns
df_sample = df[cupping_cols + ["QualityLabel"]].head(10)

# Features and label
X = df_sample.drop(columns=["QualityLabel"])
y = df_sample["QualityLabel"]

# Print feature means
print("Feature Means:")
print(X.mean())

# Train-test split (8 train, 2 test due to only 10 samples)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train_scaled, y_train)

# Predict and display results
y_pred = clf.predict(X_test_scaled)
print("\nPredicted Labels:", y_pred)
print("Actual Labels   :", y_test.values)
