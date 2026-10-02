# Task 1: Data Cleaning & Preprocessing (Titanic Dataset)

**Elevate Labs – AI & ML Internship**

## Objective
Learn how to clean and prepare raw data for machine learning.

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Files
| File | Description |
|---|---|
| `data_preprocessing.py` | Full preprocessing pipeline (run: `python data_preprocessing.py`) |
| `Titanic-Dataset.csv` | Raw dataset (891 rows, 12 columns) |
| `Titanic_cleaned_preprocessed.csv` | Final cleaned dataset (721 rows, 11 columns) |
| `output_log.txt` | Console output of the run |
| `screenshots/` | Plots: missing-value heatmap, boxplots before/after, scaling comparison, correlation heatmap |

## What I Did
1. **Explored the data** – checked shape, dtypes, nulls, summary stats, duplicates (0).
   `Age` had 19.87% missing, `Cabin` 77.10%, `Embarked` 0.22%.
2. **Handled missing values**
   - `Age` → median (28.0), since the distribution is skewed.
   - `Embarked` → mode (`S`).
   - `Cabin` → too sparse to impute; converted to a `Has_Cabin` flag (0/1) and dropped the column.
   - Dropped `PassengerId`, `Name`, `Ticket` (identifiers / no predictive signal as raw text).
3. **Encoded categoricals**
   - `Sex` → label encoding (male=0, female=1).
   - `Embarked` → one-hot encoding (`Embarked_C/Q/S`).
4. **Visualized and removed outliers** – boxplots on `Age`, `Fare`, `SibSp`, `Parch`; applied the IQR (1.5×) rule on `Age` and `Fare` only (`SibSp`/`Parch` are small counts; large families are legitimate). Rows: 891 → 721 (170 removed).
5. **Scaled features** – `StandardScaler` (z-score) on `Age`, `Fare`, `SibSp`, `Parch` for the final dataset; Min-Max normalization shown for comparison.

## Key Results
- No missing values remain.
- All features are numeric.
- Scaled columns have mean ≈ 0 and std ≈ 1.

## Notes / Design Decisions
- Outliers were removed *before* scaling so boxplots are in interpretable units and extreme values don't distort the scaler's mean/std.
- Removing 170 rows (~19%) is aggressive. Many high `Fare` values are real first-class passengers, so for actual modelling you may prefer capping (winsorizing) or a log transform. Removal was done here because the task explicitly asks for it.
- For a real train/test pipeline, fit imputers/scalers on the training set only to avoid data leakage.

---

## Interview Questions & Answers

**1. What are the different types of missing data?**
- **MCAR (Missing Completely At Random):** missingness is unrelated to any data (e.g., a sensor randomly fails).
- **MAR (Missing At Random):** missingness depends on *observed* variables (e.g., older passengers' ages recorded less often given their class).
- **MNAR (Missing Not At Random):** missingness depends on the missing value itself (e.g., high earners not reporting income). `Cabin` in Titanic is arguably MNAR – lower-class passengers often had no recorded cabin.

**2. How do you handle categorical variables?**
Convert them to numbers: label/ordinal encoding (ordered or binary categories), one-hot encoding (nominal, low cardinality), target/frequency/hash encoding (high cardinality), or use models that handle categories natively (CatBoost, LightGBM).

**3. Normalization vs standardization?**
- **Normalization (Min-Max):** rescales to a fixed range, usually [0, 1]: `(x − min)/(max − min)`. Sensitive to outliers. Good for KNN, neural nets, image data.
- **Standardization (Z-score):** `(x − mean)/std` → mean 0, std 1; no fixed range, less outlier-sensitive. Good for linear/logistic regression, SVM, PCA.

**4. How do you detect outliers?**
Boxplots / IQR rule (outside Q1−1.5·IQR, Q3+1.5·IQR), Z-score (|z| > 3), histograms/scatter plots, and model-based methods like Isolation Forest, Local Outlier Factor, DBSCAN. Always check with domain knowledge.

**5. Why is preprocessing important in ML?**
Real data is messy: nulls, inconsistent formats, mixed scales, text categories, noise. Most algorithms can't accept nulls or strings, and unscaled features can dominate distance/gradient-based models. "Garbage in, garbage out" – good preprocessing improves accuracy, stability, and training speed.

**6. One-hot vs label encoding?**
- **Label encoding** maps each category to an integer (0,1,2…). Compact, but implies an order – fine for ordinal data/tree models, misleading for nominal data in linear models.
- **One-hot encoding** creates one binary column per category. No false ordering, but increases dimensionality (problematic for high-cardinality features).

**7. How do you handle data imbalance?**
Resampling (random oversampling, undersampling, **SMOTE**), class weights / cost-sensitive learning, ensemble methods (Balanced Random Forest, XGBoost `scale_pos_weight`), threshold tuning, and using proper metrics (Precision, Recall, F1, PR-AUC) instead of accuracy. Resample only the training split.

**8. Can preprocessing affect model accuracy?**
Yes, significantly. Good imputation, encoding, and scaling can raise performance and speed convergence; poor choices (data leakage, removing too many rows, wrong encoding, scaling before splitting) can reduce accuracy or give falsely optimistic results.

---
*Submitted as part of Elevate Labs AI & ML Internship – Task 1.*
