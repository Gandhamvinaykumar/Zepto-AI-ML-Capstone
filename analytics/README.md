# Analytics

This module explores the Titanic data, cleans it, trains several models, and saves the best pipeline. The saved pipeline is tested again on raw rows with `reload_pipeline_check.py`.

## Setup

```bash
pip install -r requirements.txt
python analytics/02_modeling.py
python analytics/reload_pipeline_check.py
```

## Main choices

- Missing values under 5% are removed; between 5% and 30% are imputed.
- The `deck` column is removed because it is missing in roughly 77.2% of rows, which makes imputation unreliable.
- `age` is median-imputed, while `embarked`/`embark_town` use the mode.
- The post-outcome `alive` column is excluded from features because it directly leaks the `survived` target.
- The data is split with stratification before modeling.
- The selected pipeline is saved with `joblib.dump` and accepts raw input at prediction time.
- Decision trees are bounded with `max_depth`; Random Forest tuning searches only depths 4, 6, and 8 with bounded leaf sizes to reduce overfitting.
- The imbalance comparison uses SMOTE when `imbalanced-learn` loads correctly. If Windows blocks the required scikit-learn DLL, the script uses a labeled balanced-logistic fallback so the rest of the analysis can still run.

## Model summary

The classifiers are Logistic Regression, Decision Tree, and Random Forest. The script reports accuracy, precision, recall, F1, and ROC-AUC for classification, along with MAE, RMSE, R2, and adjusted R2 for fare regression.
