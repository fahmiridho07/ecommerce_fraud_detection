# Data Preprocessing Best Practices for Fraud Detection

**Executive Summary:** Effective fraud detection hinges on careful data preparation. Recent studies emphasize **cleaning** (drop irrelevant IDs, fix types), **missing-value handling** (impute or flag missing data), **outlier treatment** (log‐transform or cap extreme amounts) and **feature engineering** (extract time-based and aggregate features).  **Class imbalance** is typically addressed via oversampling (e.g. SMOTE) or cost-sensitive training, always ensuring that test data remain realistically imbalanced.  Categorical variables are encoded (e.g. label or one-hot) so models can use them, and numerical features are often scaled (e.g. standard or robust scaling, especially after log transforms).  Critically, ensure **no data leakage**: split data *before* any imputation or selection and use pipelines to apply transformations learned on training data only.  The recommendations below synthesize recent IEEE‐level research into an actionable pipeline. 

## Literature Survey and Synthesis

We reviewed recent IEEE/Q1–Q2 studies on fraud detection that detail preprocessing pipelines. Table 1 summarizes key findings:

| **Study** | **Dataset** | **Models** | **Preprocessing Steps** | **Metrics** | **Key Findings** |
|:---|:---|:---|:---|:---|:---|
| *Almalki et al., 2025 (arXiv)* | IEEE-CIS Fraud (590k transactions; 3.5% fraud) | Stacked XGBoost, LightGBM, CatBoost | Drop IDs; merge tables; impute missings (categorical→mode, numeric→median); label-encode categories; **SMOTE** oversample minority (train only) | Accuracy 99%, AUC-ROC 0.99 | Stacked ensemble + XAI achieved very high detection rates. Balancing via SMOTE on training improved minority recall. |
| *Moradi et al., 2025 (Preprint)* | IEEE-CIS (590k; 3.5% fraud) | RF, XGBoost, LightGBM, CatBoost, Logistic, KNN, MLP (ensemble) | Remove features >95% missing and zero-variance; treat missings by creating “missing” category for sparse categorical and median-imputing numeric (per class); add binary indicators for heavily missing fields; log-transform transaction amount; derive 80 new features (time components, rolling aggregates, target/frequency encodings); apply SMOTE/Borderline-SMOTE/ADASYN. | AUC-ROC 0.918; AUC-PR 0.891 | Careful elimination of bad features and rich feature engineering (temporal and aggregate) substantially improved performance. Oversampling consistently boosted fraud detection rates. |
| *Shi et al., 2023 (TKDE)* | Medicare Claims (large healthcare transactions) | Cost-Sensitive Classifier (post-FPCA) | Group data into patient-time trajectories; extract features via **Functional PCA** on temporal covariates; incorporate distributional FPCA; apply **random undersampling** of majority class; use cost-sensitive learning for imbalanced outcomes. | Cost-aware metrics (balanced cost reduction) | Combining temporal feature extraction with cost-sensitive modeling and undersampling improved detection, yielding significant cost savings. Emphasizes addressing imbalance via specialized strategies. |

These and other studies converge on several best practices.  For example, **missing values** are often handled by dropping nearly-empty features and imputing remaining gaps (mode for categorical, median for continuous).  Categorical columns are encoded (one-hot, ordinal or hashing) to numeric values.  Skewed numeric fields like transaction amount are log‐transformed to stabilize variance.  Many pipelines create new features – especially **temporal features** (hour of day, day of week, time deltas) – and customer or card aggregations (e.g. rolling sums/means, frequency counts). Class imbalance is a central concern: nearly all models either oversample fraud cases (SMOTE or variants) or use cost-sensitive loss functions.  Importantly, evaluation is typically on the original imbalanced test set to reflect real-world performance.  Cross-validation is stratified by class or time to avoid leakage.

**Table 1:** Comparison of recent fraud-detection studies (datasets, models, preprocessing, metrics, findings).

## Recommended Preprocessing Workflow

Based on the literature, we recommend the following **step-by-step pipeline** (see diagram below). Each step is motivated by evidence or widespread practice:

```mermaid
flowchart LR
    raw[Raw Transaction Data] --> clean[Data Cleaning\n(remove IDs, duplicates)]
    clean --> missing[Handle Missing Data\n(drop >95% missing; impute median/mode; add missing flags)]
    missing --> outlier[Outlier Treatment\n(log-transform amounts; cap extreme values)]
    outlier --> feateng[Feature Engineering\n(time features, aggregates, interactions)]
    feateng --> encode[Encoding\n(categoricals\u2192numeric via one-hot/label)]
    encode --> scale[Scaling\n(Standardize or normalize numeric features)]
    scale --> balance[Class Imbalance Handling\n(SMOTE/ADASYN or cost-sensitive learning)]
    balance --> split[Train/Test Split\n(stratified or time-aware split)]
    split --> cv[Cross-Validation\n(stratified K-fold or time-series CV)]
    cv --> model[Model Training & Evaluation]
```

- **Data Cleaning:** Drop unique identifiers or indexes (e.g. `TransactionID`), correct data types, and remove duplicates or irrelevant columns.  
- **Missing-Value Handling:** Remove features with extremely high missing rates (e.g. >95%). For remaining missing values, *impute* using statistics: use the *mode* (most frequent) for categorical fields and the *median* for numerical fields.  When a field has substantial missingness (<95%), consider adding a “missing” category (or a binary indicator) to capture missingness patterns.  This preserves information about missing data, which can itself be a fraud signal.  
- **Outlier Treatment and Transformations:** Monetary features (e.g. transaction amount) are often highly skewed. Apply a log or similar transform to stabilize variance.  For extreme outliers, winsorizing or capping (e.g. to the 99th percentile) can prevent undue influence on models.  Robust scaling (e.g. `RobustScaler` in scikit-learn) can also mitigate outliers when normalization is needed.  
- **Feature Engineering:** Derive new features that capture fraud patterns.  Decompose timestamps into cyclic components (hour, day of week, month) and, if applicable, create time-difference features (e.g. time since last transaction).  Build *behavioral* features such as counts or sums of transactions over sliding windows (e.g. past 24 hours) for each customer or card.  Log-transformed amounts can be ranked per user or per time window.  Encode high-cardinality categories by frequency or target encoding to capture empirical probabilities. Overall, combine domain knowledge (e.g. velocity features) with automated generation.  
- **Encoding Categorical Variables:** Convert nominal categories to numerical form.  If using tree-based models, simple *ordinal encoding* (assigning integer codes) often suffices. For linear models or neural nets, use **one-hot encoding** (or binary/hashing encodings for very high cardinality).  Ensure consistent mappings by fitting encoders on the training data only to avoid leakage.  
- **Feature Scaling/Normalization:** Most algorithms benefit from scaled numeric features. Apply *standardization* (zero mean, unit variance) or *min-max scaling* to continuous features.  After log-transforming skewed features, standard scaling typically works well. If outliers remain, use robust methods. Always **fit scalers on the training set** and then apply to test data to prevent leakage.  
- **Class Imbalance Handling:** Fraud datasets are highly imbalanced (often <5% fraud). In training, use resampling to balance classes.  **SMOTE** (Synthetic Minority Oversampling) and its variants (Borderline-SMOTE, ADASYN) are widely used; these create synthetic fraud examples.  Alternatively, one can randomly **undersample** the majority class if data are plentiful.  Modern boosters (XGBoost, LightGBM) also allow class-weighting (e.g. `scale_pos_weight`) or cost-sensitive losses. Critically, *apply resampling only on the training folds* – the validation/test set should remain at the true fraud rate for honest evaluation.  
- **Train/Test Splitting:** If timestamps are present and fraud patterns may drift over time, split by time (e.g. train on older data, test on newer) to avoid future leakage. Otherwise, use a random split that **stratifies** on the fraud label to preserve class ratio.  For grouped data (e.g. multiple transactions per user), ensure that all records of an entity fall entirely in train or test to prevent information leakage across folds.  
- **Cross-Validation:** Use **stratified K-fold CV** to ensure each fold has similar fraud rates. If modeling temporal data, use time-series-aware splitting (rolling windows or nested CV with time blocks). Always use pipelines (e.g. `sklearn.Pipeline`) so that imputation, scaling, and resampling are fitted on each train fold and applied to its test fold, preventing leakage.  
- **Reproducibility:** Fix random seeds, document library versions, and consider containerization. Use clear pipelines (e.g. scikit-learn `Pipeline` or `ColumnTransformer`) for all steps so that preprocessing is tied to model training. This guards against “test data snooping”.

## Implementation Snippets (Python/pseudocode)

Key steps in code might look as follows:

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold

# Split data (stratified)
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)

# Impute missing values
num_cols = X_train.select_dtypes(include='number').columns
cat_cols = X_train.select_dtypes(include='object').columns
imp_num = SimpleImputer(strategy='median')
imp_cat = SimpleImputer(strategy='most_frequent')
X_train[num_cols] = imp_num.fit_transform(X_train[num_cols])
X_train[cat_cols] = imp_cat.fit_transform(X_train[cat_cols])
X_test[num_cols]  = imp_num.transform(X_test[num_cols])
X_test[cat_cols]  = imp_cat.transform(X_test[cat_cols])

# Encode categoricals (label encoding example)
encoder = OrdinalEncoder()
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_test[cat_cols]  = encoder.transform(X_test[cat_cols])

# Log-transform skewed features (e.g. transaction amount)
X_train['TransactionAmt_log'] = np.log1p(X_train['TransactionAmt'])
X_test['TransactionAmt_log']  = np.log1p(X_test['TransactionAmt'])

# Scale numeric features
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

# Balance the training set with SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {sorted(Counter(y_train_bal).items())}")

# (Now train model on X_train_bal, evaluate on X_test as is.)
```

To integrate with cross-validation safely, one would use `Pipeline` or apply these steps inside each fold. For example, using `StratifiedKFold`:

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc')
print("AUC scores:", scores)
```

This ensures that each fold’s training data is imputed and scaled without leaking information from the validation fold.

## Experiments and Ablation Studies

To validate and fine-tune preprocessing choices, the following experiments are recommended:

- **Imputation strategies:** Compare *median* vs *mean* vs *KNN/MICE* imputation on numeric features, and *mode* vs adding a “missing” category for categoricals. Measure impact on recall/AUC.
- **Outlier handling:** Test performance with/without log-transform or winsorizing large transaction amounts. For example, compare model AUC on raw amounts versus log1p(amount).
- **Feature engineering ablation:** Starting from a baseline model, incrementally add feature groups (time features, rolling aggregates, interaction encodings) to quantify each contribution. This isolates which engineered features drive gains.
- **Class-balancing methods:** Compare no resampling vs SMOTE vs undersampling vs cost-sensitive weighting. Evaluate metrics (especially recall and precision on the fraud class) to see trade-offs.
- **Cross-validation strategy:** If data are time-stamped, compare random stratified CV vs forward-chaining/time-series splits to check for temporal leakage.
- **Pipeline reproducibility:** Verify that using sklearn `Pipeline` yields identical results as manual stepwise processing, ensuring no leakage. For example, run the same CV with and without the pipeline structure to check consistency.
- **Encoding schemes:** If many categories exist, experiment with one-hot vs ordinal vs target encoding. For tree models, check if simple ordinal encoding suffices; for linear models, one-hot may be better.
- **Scaling vs no scaling:** For tree models scaling is often unnecessary, but for distance-based or neural models it’s crucial. Compare with/without standardization for non-tree algorithms.
- **Threshold tuning:** Especially in fraud, the decision threshold can be tuned for optimal F1 or cost-sensitive criteria. Perform an experiment sweeping thresholds to maximize business metrics.

These studies can be done systematically (e.g. grid-search over preprocessing options) to identify the most robust choices for a given fraud dataset.

## Assumptions and Limitations

- We assume a transaction-style fraud dataset (mixed numeric/categorical fields, including amounts, timestamps, user/card IDs) similar to benchmarks like the IEEE-CIS set. Exact steps may vary if, for example, only aggregated data or image data were available. 
- If the dataset has unique characteristics (e.g. text fields, network graphs), additional preprocessing (NLP embedding, graph features) would be needed. Our pipeline focuses on typical tabular transactional data.
- We assume a highly imbalanced binary fraud label. If fraud is multi-class or the imbalance is moderate, some balancing steps might be adjusted.
- The recommendations aim for machine learning algorithms (tree ensembles, neural nets, etc.). If using entirely rule-based or anomaly detection methods, some steps (like SMOTE) may not apply.

## Sources

- Moradi *et al.*, 2025 (Preprint) – detailed IEEE-CIS case study, includes systematic removal of sparse features, advanced feature engineering, and oversampling methods.  
- Almalki *et al.*, 2025 (ArXiv) – credit-card fraud stacking model; discusses drop-ID, median/mode imputation, label encoding, and SMOTE balancing.  
- Shi *et al.*, 2023 (IEEE TKDE) – insurance fraud detection with temporal FPCA features; uses random undersampling and cost-sensitive learning to handle imbalance.  
- NVIDIA Financial Fraud Docs – industry guidance on preprocessing fraud data (log transforms for skew, encoding, time features).  
- Code & Tutorial (scikit-learn) – best practices to avoid data leakage with pipelines.  
- Feng *et al.*, 2023 (MDPI) – empirical study showing mode imputation improves merged credit-card fraud datasets.  

Each step above is grounded in these recent studies and standard ML practices. By following this structured pipeline—cleaning and imputing data, thoughtfully encoding features, scaling as needed, carefully handling imbalance, and rigorously splitting data—you can maximize model performance while avoiding common pitfalls (like leakage). 

