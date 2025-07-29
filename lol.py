print("🎨 Coffee Quality Analysis - Comprehensive Visualizations")
print("=" * 60)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_recall_fscore_support, roc_curve, auc, 
                           precision_recall_curve, f1_score)
from sklearn.inspection import permutation_importance
from scipy import stats
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ================================
# DATA LOADING AND PREPROCESSING
# ================================
print("📁 Loading and preprocessing data...")

filename = "Coffee_Qlty.csv"
if not os.path.exists(filename):
    print(f"❌ File '{filename}' not found!")
    exit()

df = pd.read_csv(filename)
print(f"📊 Original dataset shape: {df.shape}")

# Define cupping columns
cupping_cols = ["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance", "Sweetness", "Clean.Cup", "Uniformity"]
df = df.dropna(subset=cupping_cols)
print(f"📊 Rows after dropping null cupping values: {len(df)}")

# Create quality score and labels
df["QualityScore"] = df[cupping_cols].mean(axis=1)

def classify_quality(score):
    if score >= 8.5:
        return "Excellent"
    elif score >= 7.5:
        return "Good"
    else:
        return "Average"

df["QualityLabel"] = df["QualityScore"].apply(classify_quality)

# Prepare features
features = cupping_cols + ["Moisture", "Processing.Method", "Country.of.Origin", "Continent.of.Origin"]
df = df.dropna(subset=features)

X = df[features]
y = df["QualityLabel"]

# One-hot encoding
X_encoded = pd.get_dummies(X, columns=["Processing.Method", "Country.of.Origin", "Continent.of.Origin"])

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Data preprocessing completed!")

# ================================
# MODEL TRAINING
# ================================
print("\n🤖 Training models...")

# KNN Model (from your original code)
knn = KNeighborsClassifier(n_neighbors=7)  # Using optimal k from your analysis
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_pred_proba = knn.predict_proba(X_test)

# Logistic Regression
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)
log_pred_proba = log_reg.predict_proba(X_test)

# Random Forest (for feature importance)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("✅ Models trained successfully!")

# ================================
# VISUALIZATION FUNCTIONS
# ================================

def create_figure_grid(rows, cols, figsize=(20, 15)):
    """Create a figure with subplots"""
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    elif cols == 1:
        axes = [[ax] for ax in axes]
    else:
        axes = axes.flatten() if rows * cols > 1 else [axes]
    return fig, axes

# ================================
# 1. DATASET OVERVIEW VISUALIZATIONS
# ================================
print("\n📊 Creating dataset overview visualizations...")

fig, axes = create_figure_grid(2, 3, (18, 12))

# Quality label distribution
axes[0].pie(df['QualityLabel'].value_counts().values, 
           labels=df['QualityLabel'].value_counts().index, 
           autopct='%1.1f%%', startangle=90)
axes[0].set_title('Quality Label Distribution', fontsize=14, fontweight='bold')

# Quality score distribution
axes[1].hist(df['QualityScore'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[1].set_xlabel('Quality Score')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Quality Score Distribution', fontsize=14, fontweight='bold')
axes[1].axvline(df['QualityScore'].mean(), color='red', linestyle='--', label=f'Mean: {df["QualityScore"].mean():.2f}')
axes[1].legend()

# Top countries by count
top_countries = df['Country.of.Origin'].value_counts().head(10)
axes[2].barh(range(len(top_countries)), top_countries.values)
axes[2].set_yticks(range(len(top_countries)))
axes[2].set_yticklabels(top_countries.index)
axes[2].set_xlabel('Count')
axes[2].set_title('Top 10 Countries by Coffee Samples', fontsize=14, fontweight='bold')

# Processing methods distribution
processing_counts = df['Processing.Method'].value_counts()
axes[3].bar(range(len(processing_counts)), processing_counts.values)
axes[3].set_xticks(range(len(processing_counts)))
axes[3].set_xticklabels(processing_counts.index, rotation=45, ha='right')
axes[3].set_ylabel('Count')
axes[3].set_title('Processing Methods Distribution', fontsize=14, fontweight='bold')

# Continent distribution
continent_counts = df['Continent.of.Origin'].value_counts()
axes[4].pie(continent_counts.values, labels=continent_counts.index, autopct='%1.1f%%', startangle=90)
axes[4].set_title('Continent Distribution', fontsize=14, fontweight='bold')

# Moisture content vs Quality
axes[5].scatter(df['Moisture'], df['QualityScore'], alpha=0.6, c=df['QualityScore'], cmap='viridis')
axes[5].set_xlabel('Moisture Content')
axes[5].set_ylabel('Quality Score')
axes[5].set_title('Moisture vs Quality Score', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# ================================
# 2. CORRELATION ANALYSIS
# ================================
print("\n🔗 Creating correlation analysis...")

fig, axes = create_figure_grid(1, 2, (16, 6))

# Correlation heatmap for cupping scores
corr_matrix = df[cupping_cols + ['QualityScore']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, ax=axes[0], fmt='.2f')
axes[0].set_title('Cupping Scores Correlation Matrix', fontsize=14, fontweight='bold')

# Feature correlation with quality score
quality_corr = df[cupping_cols + ['Moisture']].corrwith(df['QualityScore']).sort_values(ascending=False)
axes[1].barh(range(len(quality_corr)), quality_corr.values)
axes[1].set_yticks(range(len(quality_corr)))
axes[1].set_yticklabels(quality_corr.index)
axes[1].set_xlabel('Correlation with Quality Score')
axes[1].set_title('Feature Correlation with Quality', fontsize=14, fontweight='bold')
axes[1].axvline(0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()

# ================================
# 3. MODEL PERFORMANCE VISUALIZATIONS
# ================================
print("\n📈 Creating model performance visualizations...")

# Calculate metrics for all models
models = {'KNN': (knn_pred, knn_pred_proba), 'Logistic Regression': (log_pred, log_pred_proba), 'Random Forest': (rf_pred, None)}
model_metrics = {}

for name, (pred, pred_proba) in models.items():
    accuracy = accuracy_score(y_test, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average='weighted')
    model_metrics[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}

# Performance comparison
fig, axes = create_figure_grid(2, 2, (15, 12))

metrics_df = pd.DataFrame(model_metrics).T
metrics_df.plot(kind='bar', ax=axes[0], rot=45)
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Score')
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Confusion matrices
for i, (model_name, (pred, _)) in enumerate(models.items()):
    if i < 2:  # Show first two models
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i+1])
        axes[i+1].set_title(f'{model_name} Confusion Matrix', fontsize=12, fontweight='bold')
        axes[i+1].set_xlabel('Predicted')
        axes[i+1].set_ylabel('Actual')

# F1-Score by class
f1_by_class = {}
for name, (pred, _) in models.items():
    _, _, f1_scores, _ = precision_recall_fscore_support(y_test, pred, average=None)
    f1_by_class[name] = dict(zip(np.unique(y_test), f1_scores))

f1_df = pd.DataFrame(f1_by_class)
f1_df.plot(kind='bar', ax=axes[3], rot=45)
axes[3].set_title('F1-Score by Quality Class', fontsize=14, fontweight='bold')
axes[3].set_ylabel('F1-Score')
axes[3].legend()

plt.tight_layout()
plt.show()

# ================================
# 4. ROC CURVES AND PRECISION-RECALL CURVES
# ================================
print("\n📊 Creating ROC and Precision-Recall curves...")

# For multiclass ROC, we need to binarize the output
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle

y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_bin.shape[1]

fig, axes = create_figure_grid(1, 2, (15, 6))

# ROC Curves for KNN (multiclass)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], knn_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=color, lw=2,
                label=f'ROC curve class {np.unique(y)[i]} (AUC = {roc_auc:.2f})')

axes[0].plot([0, 1], [0, 1], 'k--', lw=2)
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves - KNN Model', fontsize=14, fontweight='bold')
axes[0].legend(loc="lower right")

# Precision-Recall Curves for KNN
for i, color in zip(range(n_classes), colors):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], knn_pred_proba[:, i])
    axes[1].plot(recall, precision, color=color, lw=2,
                label=f'PR curve class {np.unique(y)[i]}')

axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curves - KNN Model', fontsize=14, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.show()

# ================================
# 5. FEATURE IMPORTANCE ANALYSIS
# ================================
print("\n🔍 Creating feature importance analysis...")

fig, axes = create_figure_grid(2, 2, (18, 12))

# Random Forest Feature Importance
rf_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).head(15)

axes[0].barh(range(len(rf_importance)), rf_importance['importance'])
axes[0].set_yticks(range(len(rf_importance)))
axes[0].set_yticklabels(rf_importance['feature'])
axes[0].set_xlabel('Importance')
axes[0].set_title('Top 15 Features - Random Forest Importance', fontsize=12, fontweight='bold')

# Logistic Regression Coefficients (for binary classes, we'll show average)
log_coef = np.abs(log_reg.coef_).mean(axis=0)
log_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': log_coef
}).sort_values('importance', ascending=False).head(15)

axes[1].barh(range(len(log_importance)), log_importance['importance'])
axes[1].set_yticks(range(len(log_importance)))
axes[1].set_yticklabels(log_importance['feature'])
axes[1].set_xlabel('Average |Coefficient|')
axes[1].set_title('Top 15 Features - Logistic Regression Coefficients', fontsize=12, fontweight='bold')

# Permutation Importance for KNN
perm_importance = permutation_importance(knn, X_test, y_test, n_repeats=10, random_state=42)
perm_df = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False).head(15)

axes[2].barh(range(len(perm_df)), perm_df['importance'])
axes[2].set_yticks(range(len(perm_df)))
axes[2].set_yticklabels(perm_df['feature'])
axes[2].set_xlabel('Permutation Importance')
axes[2].set_title('Top 15 Features - Permutation Importance (KNN)', fontsize=12, fontweight='bold')

# Cupping scores importance comparison
cupping_importance = rf_importance[rf_importance['feature'].isin(cupping_cols)]
if len(cupping_importance) > 0:
    axes[3].bar(range(len(cupping_importance)), cupping_importance['importance'])
    axes[3].set_xticks(range(len(cupping_importance)))
    axes[3].set_xticklabels(cupping_importance['feature'], rotation=45, ha='right')
    axes[3].set_ylabel('Importance')
    axes[3].set_title('Cupping Scores Importance', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# ================================
# 6. CROSS-VALIDATION ANALYSIS
# ================================
print("\n🔄 Creating cross-validation analysis...")

# Perform cross-validation for different models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

models_cv = {'KNN': KNeighborsClassifier(n_neighbors=7),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)}

for name, model in models_cv.items():
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    cv_results[name] = scores

fig, axes = create_figure_grid(1, 2, (15, 6))

# Box plot of CV scores
cv_df = pd.DataFrame(cv_results)
cv_df.boxplot(ax=axes[0])
axes[0].set_title('Cross-Validation Accuracy Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Accuracy')
axes[0].tick_params(axis='x', rotation=45)

# Mean CV scores with error bars
means = [cv_results[model].mean() for model in cv_results]
stds = [cv_results[model].std() for model in cv_results]
axes[1].bar(range(len(means)), means, yerr=stds, capsize=5, alpha=0.7)
axes[1].set_xticks(range(len(means)))
axes[1].set_xticklabels(cv_results.keys(), rotation=45, ha='right')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Mean CV Accuracy with Standard Deviation', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# ================================
# 7. TOP-N ACCURACY ANALYSIS
# ================================
print("\n🎯 Creating Top-N accuracy analysis...")

def calculate_top_n_accuracy(y_true, y_proba, n=2):
    """Calculate top-n accuracy"""
    top_n_pred = np.argsort(y_proba, axis=1)[:, -n:]
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)
    
    correct = 0
    for i, true_label in enumerate(y_true_encoded):
        if true_label in top_n_pred[i]:
            correct += 1
    return correct / len(y_true)

# Calculate top-n accuracies
n_values = [1, 2, 3]
top_n_results = {}

for model_name, (_, pred_proba) in [('KNN', (knn_pred, knn_pred_proba)), 
                                   ('Logistic Regression', (log_pred, log_pred_proba))]:
    if pred_proba is not None:
        top_n_results[model_name] = [calculate_top_n_accuracy(y_test, pred_proba, n) for n in n_values]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for model_name, accuracies in top_n_results.items():
    ax.plot(n_values, accuracies, marker='o', linewidth=2, markersize=8, label=model_name)

ax.set_xlabel('Top-N')
ax.set_ylabel('Accuracy')
ax.set_title('Top-N Accuracy Analysis', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(n_values)

plt.tight_layout()
plt.show()

# ================================
# 8. WORDCLOUD VISUALIZATION
# ================================
print("\n☁️ Creating word cloud visualization...")

# Create word clouds for different quality levels
fig, axes = create_figure_grid(1, 3, (18, 6))

quality_levels = df['QualityLabel'].unique()
for i, quality in enumerate(quality_levels):
    # Get countries for this quality level
    countries_quality = df[df['QualityLabel'] == quality]['Country.of.Origin'].value_counts()
    
    if len(countries_quality) > 0:
        # Create word cloud
        wordcloud = WordCloud(width=400, height=300, background_color='white', 
                            max_words=50, colormap='viridis').generate_from_frequencies(countries_quality.to_dict())
        
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].axis('off')
        axes[i].set_title(f'{quality} Quality Coffee\nCountries', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# ================================
# 9. RELATIONSHIP ANALYSIS
# ================================
print("\n🔗 Creating relationship analysis...")

fig, axes = create_figure_grid(2, 2, (15, 12))

# Quality score vs Cupping attributes scatter matrix (sample)
selected_features = ['Aroma', 'Flavor', 'Aftertaste', 'Body']
for i, feature in enumerate(selected_features):
    scatter = axes[i].scatter(df[feature], df['QualityScore'], 
                             c=df['QualityScore'], cmap='viridis', alpha=0.6)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Quality Score')
    axes[i].set_title(f'{feature} vs Quality Score', fontsize=12, fontweight='bold')
    
    # Add correlation coefficient
    corr_coef = df[feature].corr(df['QualityScore'])
    axes[i].text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=axes[i].transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# ================================
# 10. NORMALIZED CUMULATIVE GAIN (DCG-like analysis)
# ================================
print("\n📊 Creating quality distribution analysis...")

fig, axes = create_figure_grid(1, 2, (15, 6))

# Quality score distribution by continent
continent_quality = df.groupby('Continent.of.Origin')['QualityScore'].apply(list).to_dict()
axes[0].boxplot(continent_quality.values(), labels=continent_quality.keys())
axes[0].set_title('Quality Score Distribution by Continent', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Quality Score')
axes[0].tick_params(axis='x', rotation=45)

# Processing method vs quality
processing_quality = df.groupby('Processing.Method')['QualityScore'].mean().sort_values(ascending=False)
axes[1].bar(range(len(processing_quality)), processing_quality.values)
axes[1].set_xticks(range(len(processing_quality)))
axes[1].set_xticklabels(processing_quality.index, rotation=45, ha='right')
axes[1].set_ylabel('Average Quality Score')
axes[1].set_title('Average Quality Score by Processing Method', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# ================================
# 11. MODEL COMPARISON SUMMARY
# ================================
print("\n📊 Final Model Comparison Summary")
print("=" * 50)

summary_data = []
for name, (pred, pred_proba) in models.items():
    accuracy = accuracy_score(y_test, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average='weighted')
    cv_mean = cv_results[name].mean() if name in cv_results else np.nan
    cv_std = cv_results[name].std() if name in cv_results else np.nan
    
    summary_data.append({
        'Model': name,
        'Test Accuracy': f'{accuracy:.4f}',
        'Precision': f'{precision:.4f}',
        'Recall': f'{recall:.4f}',
        'F1-Score': f'{f1:.4f}',
        'CV Mean': f'{cv_mean:.4f}',
        'CV Std': f'{cv_std:.4f}'
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n✅ All visualizations completed successfully!")
print("📊 Summary of created visualizations:")
print("1. Dataset Overview (distribution, countries, processing methods)")
print("2. Correlation Analysis (heatmap, feature correlations)")
print("3. Model Performance Comparison (accuracy, precision, recall, F1)")
print("4. ROC and Precision-Recall Curves")
print("5. Feature Importance Analysis (RF, LogReg, Permutation)")
print("6. Cross-Validation Analysis")
print("7. Top-N Accuracy Analysis")
print("8. Word Cloud Visualization")
print("9. Feature Relationships Analysis")
print("10. Quality Distribution Analysis")