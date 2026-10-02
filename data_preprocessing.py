"""
Task 1: Data Cleaning & Preprocessing (Titanic Dataset)
Elevate Labs - AI & ML Internship

Steps:
 1. Import dataset and explore basic info (nulls, dtypes)
 2. Handle missing values (median / mode imputation, drop sparse column)
 3. Encode categorical features (label + one-hot)
 4. Standardize / normalize numerical features
 5. Visualize outliers with boxplots and remove them (IQR method)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sns.set_theme(style="whitegrid")
os.makedirs("screenshots", exist_ok=True)

# ---------------------------------------------------------------- 1. EXPLORE
print("=" * 60, "\n1. LOADING & EXPLORING DATA\n", "=" * 60)
df = pd.read_csv("Titanic-Dataset.csv")
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nInfo:")
df.info()
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nMissing % :\n", (df.isnull().mean() * 100).round(2))
print("\nSummary statistics:\n", df.describe())
print("\nDuplicate rows:", df.duplicated().sum())

# Visualise missing values
plt.figure(figsize=(9, 5))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
plt.title("Missing Values Heatmap (yellow = missing)")
plt.tight_layout()
plt.savefig("screenshots/01_missing_values_heatmap.png", dpi=120)
plt.close()

# ------------------------------------------------------- 2. MISSING VALUES
print("\n", "=" * 60, "\n2. HANDLING MISSING VALUES\n", "=" * 60)
df_clean = df.copy()

# Cabin: ~77% missing -> too sparse to impute; keep a useful flag then drop
df_clean["Has_Cabin"] = df_clean["Cabin"].notnull().astype(int)
df_clean.drop(columns=["Cabin"], inplace=True)

# Age: numeric & skewed -> median imputation
age_median = df_clean["Age"].median()
df_clean["Age"] = df_clean["Age"].fillna(age_median)
print(f"Age filled with median = {age_median}")

# Embarked: categorical -> mode imputation
embarked_mode = df_clean["Embarked"].mode()[0]
df_clean["Embarked"] = df_clean["Embarked"].fillna(embarked_mode)
print(f"Embarked filled with mode = {embarked_mode}")

# Drop identifier / high-cardinality text columns that carry no model signal
df_clean.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)

print("\nMissing values after cleaning:\n", df_clean.isnull().sum())

# ----------------------------------------------------------- 3. ENCODING
print("\n", "=" * 60, "\n3. ENCODING CATEGORICAL FEATURES\n", "=" * 60)
# Sex is binary -> label encoding
df_clean["Sex"] = df_clean["Sex"].map({"male": 0, "female": 1})
# Embarked is nominal (3 classes, no order) -> one-hot encoding
df_clean = pd.get_dummies(df_clean, columns=["Embarked"], prefix="Embarked", dtype=int)
print("Columns after encoding:", list(df_clean.columns))
print(df_clean.head())

# ----------------------------------------------- 4. OUTLIERS (visualise + remove)
# Done BEFORE scaling so boxplots are in original, interpretable units.
print("\n", "=" * 60, "\n4. OUTLIER DETECTION & REMOVAL (IQR)\n", "=" * 60)
num_cols = ["Age", "Fare", "SibSp", "Parch"]

fig, axes = plt.subplots(1, len(num_cols), figsize=(14, 4))
for ax, col in zip(axes, num_cols):
    sns.boxplot(y=df_clean[col], ax=ax, color="skyblue")
    ax.set_title(col)
fig.suptitle("Boxplots BEFORE Outlier Removal")
plt.tight_layout()
plt.savefig("screenshots/02_boxplots_before.png", dpi=120)
plt.close()

# Only Age and Fare are continuous; SibSp/Parch are small counts so
# removing their "outliers" would discard legitimate large families.
outlier_cols = ["Age", "Fare"]
before = len(df_clean)
mask = pd.Series(True, index=df_clean.index)
for col in outlier_cols:
    q1, q3 = df_clean[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df_clean[col] < low) | (df_clean[col] > high)).sum()
    print(f"{col}: Q1={q1:.2f} Q3={q3:.2f} IQR={iqr:.2f} "
          f"bounds=[{low:.2f}, {high:.2f}] -> {n_out} outliers")
    mask &= df_clean[col].between(low, high)
df_no_out = df_clean[mask].reset_index(drop=True)
print(f"Rows: {before} -> {len(df_no_out)} (removed {before - len(df_no_out)})")

fig, axes = plt.subplots(1, len(num_cols), figsize=(14, 4))
for ax, col in zip(axes, num_cols):
    sns.boxplot(y=df_no_out[col], ax=ax, color="lightgreen")
    ax.set_title(col)
fig.suptitle("Boxplots AFTER Outlier Removal")
plt.tight_layout()
plt.savefig("screenshots/03_boxplots_after.png", dpi=120)
plt.close()

# --------------------------------------------------------- 5. SCALING
print("\n", "=" * 60, "\n5. FEATURE SCALING\n", "=" * 60)
df_scaled = df_no_out.copy()
scale_cols = ["Age", "Fare", "SibSp", "Parch"]

# Standardization (z-score): mean 0, std 1
df_scaled[scale_cols] = StandardScaler().fit_transform(df_no_out[scale_cols])
print("Standardized (mean ~0, std ~1):\n",
      df_scaled[scale_cols].agg(["mean", "std"]).round(3))

# Min-Max normalization shown for comparison (0-1 range)
df_minmax = df_no_out.copy()
df_minmax[scale_cols] = MinMaxScaler().fit_transform(df_no_out[scale_cols])
print("\nMin-Max normalized (min 0, max 1):\n",
      df_minmax[scale_cols].agg(["min", "max"]).round(3))

# Distribution comparison plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df_no_out["Fare"], kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Fare - Original")
sns.histplot(df_scaled["Fare"], kde=True, ax=axes[1], color="orange")
axes[1].set_title("Fare - Standardized")
sns.histplot(df_minmax["Fare"], kde=True, ax=axes[2], color="green")
axes[2].set_title("Fare - Min-Max")
plt.tight_layout()
plt.savefig("screenshots/04_scaling_comparison.png", dpi=120)
plt.close()

# Correlation heatmap of final data
plt.figure(figsize=(9, 7))
sns.heatmap(df_scaled.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (Processed Data)")
plt.tight_layout()
plt.savefig("screenshots/05_correlation_heatmap.png", dpi=120)
plt.close()

# --------------------------------------------------------------- SAVE
df_scaled.to_csv("Titanic_cleaned_preprocessed.csv", index=False)
print("\nFinal shape:", df_scaled.shape)
print("Saved -> Titanic_cleaned_preprocessed.csv")
print(df_scaled.head())
