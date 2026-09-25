# Module 2: Analytics Pipeline

Run `python3 analytics/pipeline.py` from the repository root. This is the only
`sns.load_dataset('titanic')` call. It immediately writes the committed offline
fallback `analytics/titanic.csv`; all EDA and modeling continue from that one
loaded DataFrame.

## Cleaning and EDA

Missing rates: `age` 19.865%, `embarked` 0.224%, `deck` 77.217%, and
`embark_town` 0.224%. Under the threshold rule, the two `embarked` rows are
dropped; age is median-imputed for EDA. Deck is dropped because its 77.217%
missingness makes imputation unreliable. `embark_town` duplicates embarkation
information and is dropped. The modeling pipeline independently fits its age
imputer on training data only.

Age has 65 IQR outliers and fare has 114. Fare is right-skewed: mean 32.10 is
above median 14.45, which is above the mode 8.05. The z-score check saved by
the pipeline reports approximate zero means and unit standard deviations for
both age and fare.

Female survival is higher than male survival, and first-class survival exceeds
third-class survival. The combined sex/class chart shows these effects persist
within classes. The fare-by-survival chart shows survivors generally paid more,
although the fare distribution has substantial high-value outliers. The age
charts show a broad passenger age distribution rather than a single narrow
survivor profile.

The correlation heatmap uses exactly survived, pclass, age, sibsp, parch, and
fare. Its strongest absolute pairs are pclass/fare (0.548) and sibsp/parch
(0.415). Higher-class travel is strongly associated with higher fares, while
passengers traveling with siblings/spouses often also traveled with parents or
children.

## Modeling

The survived class balance is 61.75% not survived and 38.25% survived, so the
train/test split is stratified to preserve this target mix. Imputation, one-hot
encoding, and scaling are all inside a `ColumnTransformer`/`Pipeline` fitted
only on training data.

| Classifier | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.809 | 0.783 | 0.691 | 0.734 | 0.861 |
| Decision Tree | 0.764 | 0.760 | 0.559 | 0.644 | 0.837 |
| Random Forest | 0.803 | 0.762 | 0.706 | 0.733 | 0.824 |

SMOTE had the highest F1 (0.735) among baseline (0.734), class-weighted
(0.734), and SMOTE variants, while class weighting produced the highest recall
(0.750). Random Forest grid search selected 300 estimators, no max depth, and
all features, with OOB score 0.816. The fare regression produced MAE 21.14,
RMSE 41.75, R2 0.347, and adjusted R2 0.312; its residual plot shows widening
spread at higher predicted fares, consistent with heteroscedasticity.

I would deploy Logistic Regression for this baseline because it has the highest
accuracy (0.809), F1 is effectively tied with the alternatives, and it has the
best AUC (0.861). It is also simpler to explain and operate than the Random
Forest. The saved `best_survival_pipeline.joblib` contains preprocessing plus
the tuned Random Forest estimator and reloads with identical raw-input
predictions, as demonstrated by the pipeline.

Generated charts, confusion matrices, ROC curve, comparison CSVs, and the full
machine-readable report are in `analytics/artifacts/`.
