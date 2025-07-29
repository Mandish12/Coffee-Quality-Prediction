print("🧪 Script started!")

import os
print("📁 Current directory:", os.getcwd())
print("📄 Files in folder:", os.listdir())

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

filename = "Coffee_Qlty.csv"
print(f"🔍 Looking for dataset file: {filename}")
if not os.path.exists(filename):
    print(f"❌ File '{filename}' not found in {os.getcwd()}")
    exit()
else:
    print("✅ File found. Loading dataset...")

df = pd.read_csv(filename)

cupping_cols = ["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance", "Sweetness", "Clean.Cup", "Uniformity"]
df = df.dropna(subset=cupping_cols)
print(f"📊 Rows after dropping null cupping values: {len(df)}")

df["QualityScore"] = df[cupping_cols].mean(axis=1)

def classify_quality(score):
    if score >= 8.5:
        return "Excellent"
    elif score >= 7.5:
        return "Good"
    else:
        return "Average"

df["QualityLabel"] = df["QualityScore"].apply(classify_quality)
print("\n📈 Label distribution:\n", df["QualityLabel"].value_counts())

features = cupping_cols + ["Moisture", "Processing.Method", "Country.of.Origin", "Continent.of.Origin"]
df = df.dropna(subset=features)

X = df[features]
y = df["QualityLabel"]

processing_methods = sorted(df["Processing.Method"].dropna().unique())
countries = sorted(df["Country.of.Origin"].dropna().unique())
continents = sorted(df["Continent.of.Origin"].dropna().unique())

X = pd.get_dummies(X, columns=["Processing.Method", "Country.of.Origin", "Continent.of.Origin"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# CROSS-VALIDATION ANALYSIS
# ================================
print("\n" + "="*50)
print("🔄 CROSS-VALIDATION ANALYSIS")
print("="*50)

# Test different k values with cross-validation
k_values = [3, 5, 7, 9, 11, 15, 21]
cv_scores = {}
cv_stds = {}

# Use StratifiedKFold to maintain class distribution
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("🧪 Testing different k values with 5-fold cross-validation...")
print("-" * 60)

for k in k_values:
    clf = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
    cv_scores[k] = scores.mean()
    cv_stds[k] = scores.std()
    print(f"k={k:2d}: Accuracy = {scores.mean():.4f} (±{scores.std():.4f}) | Scores: {scores}")

# Find best k
best_k = max(cv_scores, key=cv_scores.get)
print(f"\n🏆 Best k value: {best_k} with accuracy: {cv_scores[best_k]:.4f} (±{cv_stds[best_k]:.4f})")

# Plot cross-validation results
plt.figure(figsize=(10, 6))
k_vals = list(cv_scores.keys())
mean_scores = list(cv_scores.values())
std_scores = list(cv_stds.values())

plt.errorbar(k_vals, mean_scores, yerr=std_scores, marker='o', capsize=5, capthick=2)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('KNN Cross-Validation Performance vs k Value')
plt.grid(True, alpha=0.3)
plt.xticks(k_vals)
for i, (k, score) in enumerate(zip(k_vals, mean_scores)):
    plt.annotate(f'{score:.3f}', (k, score), textcoords="offset points", xytext=(0,10), ha='center')
plt.tight_layout()
plt.show()

# ================================
# DETAILED CROSS-VALIDATION WITH BEST K
# ================================
print(f"\n🔍 Detailed cross-validation analysis with k={best_k}:")
print("-" * 50)

best_clf = KNeighborsClassifier(n_neighbors=best_k)

# Perform cross-validation with detailed metrics
cv_accuracies = []
cv_precisions = []
cv_recalls = []
cv_f1s = []

fold_num = 1
for train_idx, val_idx in cv.split(X_scaled, y):
    X_train_cv, X_val_cv = X_scaled[train_idx], X_scaled[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
    
    best_clf.fit(X_train_cv, y_train_cv)
    y_pred_cv = best_clf.predict(X_val_cv)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val_cv, y_pred_cv)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val_cv, y_pred_cv, average='weighted')
    
    cv_accuracies.append(accuracy)
    cv_precisions.append(precision)
    cv_recalls.append(recall)
    cv_f1s.append(f1)
    
    print(f"Fold {fold_num}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    fold_num += 1

print("\n📊 Cross-Validation Summary:")
print(f"Mean Accuracy:  {np.mean(cv_accuracies):.4f} (±{np.std(cv_accuracies):.4f})")
print(f"Mean Precision: {np.mean(cv_precisions):.4f} (±{np.std(cv_precisions):.4f})")
print(f"Mean Recall:    {np.mean(cv_recalls):.4f} (±{np.std(cv_recalls):.4f})")
print(f"Mean F1-Score:  {np.mean(cv_f1s):.4f} (±{np.std(cv_f1s):.4f})")

# ================================
# FINAL MODEL TRAINING AND EVALUATION
# ================================
print("\n" + "="*50)
print("🎯 FINAL MODEL TRAINING & EVALUATION")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Training samples: {len(X_train)}")
print(f"📊 Testing samples: {len(X_test)}")

# Train final model with best k
final_clf = KNeighborsClassifier(n_neighbors=best_k)
final_clf.fit(X_train, y_train)
print(f"\n✅ Final model trained successfully with k={best_k}!")

y_pred = final_clf.predict(X_test)

print("\n🔮 Sample Predictions:")
print("Predicted:", y_pred[:10])
print("Actual   :", y_test.values[:10])

accuracy = final_clf.score(X_test, y_test)
print(f"\n🎯 Final Model Accuracy: {accuracy:.4f}")

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

print("🔥 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=final_clf.classes_, yticklabels=final_clf.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (k={best_k})")
plt.tight_layout()
plt.show()

# ================================
# COMPARISON: CV vs HOLDOUT
# ================================
print("\n" + "="*50)
print("⚖️  CROSS-VALIDATION vs HOLDOUT COMPARISON")
print("="*50)
print(f"Cross-Validation Accuracy: {np.mean(cv_accuracies):.4f} (±{np.std(cv_accuracies):.4f})")
print(f"Holdout Test Accuracy:     {accuracy:.4f}")
print(f"Difference:                {abs(np.mean(cv_accuracies) - accuracy):.4f}")

if abs(np.mean(cv_accuracies) - accuracy) < 0.02:
    print("✅ Good agreement between CV and holdout - model appears stable")
else:
    print("⚠️  Large difference suggests potential overfitting or data variance issues")

# ================================
# INTERACTIVE PREDICTION SYSTEM
# ================================
def get_float_input(prompt, min_val=None, max_val=None):
    while True:
        try:
            val = float(input(prompt))
            if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
                print(f"Please enter a value between {min_val} and {max_val}.")
                continue
            return val
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_choice_input(prompt, choices):
    choices_str = ", ".join(choices)
    while True:
        val = input(f"{prompt} ({choices_str}): ").strip()
        if val in choices:
            return val
        else:
            print("Invalid choice. Please select from the options.")

def cli_predict():
    print("\n--- Enter new coffee sample details to predict quality ---")
    
    input_data = {}
    for feature in cupping_cols:
        input_data[feature] = get_float_input(f"{feature} (0-10): ", 0, 10)
    input_data["Moisture"] = get_float_input("Moisture (%): ", 0, 100)
    
    input_data["Processing.Method"] = get_choice_input("Processing Method", processing_methods)
    input_data["Country.of.Origin"] = get_choice_input("Country of Origin", countries)
    input_data["Continent.of.Origin"] = get_choice_input("Continent of Origin", continents)
    
    df_input = pd.DataFrame([input_data])
    df_input = pd.get_dummies(df_input, columns=["Processing.Method", "Country.of.Origin", "Continent.of.Origin"])
    
    for col in X.columns:
        if col not in df_input.columns:
            df_input[col] = 0
    
    df_input = df_input[X.columns]
    input_scaled = scaler.transform(df_input)
    
    prediction = final_clf.predict(input_scaled)
    print(f"\n🧪 Predicted Quality Label: {prediction[0]} (using optimized k={best_k})")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🎮 INTERACTIVE PREDICTION SYSTEM")
    print("="*50)
    while True:
        cli_predict()
        cont = input("\nDo you want to predict another sample? (y/n): ").strip().lower()
        if cont != 'y':
            print("👋 Goodbye!")
            break