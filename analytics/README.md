# Analytics

This module explores the Titanic data, cleans it, trains several models, and saves the best pipeline. The saved pipeline is tested again on raw rows with `reload_pipeline_check.py`.

## Setup

```bash
pip install -r requirements.txt
python analytics/02_modeling.py
python analytics/reload_pipeline_check.py
```

## Main choices

- Missing values under 5% are removed row-wise. `embarked` and `embark_town` each have 0.224% missing values, so the two affected rows are dropped. `age` has 19.865% missing values, so it is filled with the median. `deck` has 77.217% missing values, so it is dropped rather than imputed because that much missingness would make the values unreliable.
- The post-outcome `alive` column is excluded from features because it directly leaks the `survived` target.
- The EDA script writes the cleaned data to `clean_titanic.csv`, and modeling reads that same file rather than loading Titanic again. The data is then split with stratification before modeling. The training split is about 61.7% not survived and 38.3% survived, so stratification keeps that balance similar in train and test.
- The modeling `ColumnTransformer` fits the median/most-frequent imputers, one-hot encoder, and scaler on the training split through the pipeline. The test split is only transformed.
- The selected pipeline is saved with `joblib.dump` and accepts raw input at prediction time.
- The EDA script saves `analytics/boxplots.png` along with the other charts.
- Decision trees are bounded with `max_depth`; Random Forest tuning searches only depths 4, 6, and 8 with bounded leaf sizes to reduce overfitting.
- The imbalance comparison uses SMOTE when `imbalanced-learn` loads correctly. If Windows blocks the required scikit-learn DLL, the script uses a labeled balanced-logistic fallback so the rest of the analysis can still run.

## Model summary

The classifiers are Logistic Regression, Decision Tree, and Random Forest. The script reports accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and ROC curves for classification, along with MAE, RMSE, R2, and adjusted R2 for fare regression.

## EDA results and interpretation

The cleaned dataset has 889 rows after dropping the two rows with the small `embarked`/`embark_town` gaps. The IQR rule found 65 age outliers and 114 fare outliers. Fare has mean 32.097, median 14.454, and mode 8.050; because mean > median > mode, the fare distribution is right-skewed.

The survival rates were:

- By sex: female 0.740, male 0.189.
- By passenger class: first 0.626, second 0.473, third 0.242.
- By sex and class: female first class 0.967, female second class 0.921, female third class 0.500; male first class 0.369, male second class 0.157, male third class 0.135.

The two strongest absolute correlations were fare with pclass (`-0.548`) and parch with sibsp (`0.415`). The negative fare/pclass relationship means first-class passengers generally paid more, while the positive parch/sibsp relationship shows that family size variables tend to move together. The heatmap is saved as `charts/corr_heatmap.png`.

The EDA charts are saved in `analytics/charts/`. Standalone copies are also available as `analytics/histograms.png`, `analytics/boxplots.png`, and `analytics/scatterplot.png`:

1. **Age histogram:** Most passengers were young adults, with fewer passengers at the oldest ages. This shows why the median is a reasonable value for the missing ages.
2. **Age boxplot:** The upper tail contains several older-age outliers under the IQR rule. They are reported but retained because age is a meaningful passenger characteristic.
3. **Fare histogram:** Most fares are low, while a small number of expensive tickets create a long right tail. This agrees with the mean being higher than the median and mode.
4. **Fare boxplot:** The fare boxplot makes the high-price outliers easy to see. Those values are kept because they may reflect real differences in passenger class.
5. **Survival by sex:** Female passengers survived at a much higher rate than male passengers in this dataset. Sex is therefore an important variable for explaining the outcome.
6. **Survival by class:** Survival decreases from first class to third class. This suggests that passenger class captured differences in location, access, and travel conditions.
7. **Age by sex boxplot:** The age distributions overlap, but the two groups have different centers and spreads. Age alone does not explain survival as strongly as sex or class.
8. **Fare versus age scatter plot:** The points show that expensive fares occur across several ages, while survival labels are mixed. Fare is useful in combination with other features rather than as a single rule.
9. **Correlation heatmap:** The heatmap highlights the strong negative association between fare and class and the positive association between sibling/spouse count and parent/child count. It also shows that no single numeric feature has a near-perfect relationship with survival.

The z-score check is saved in the EDA output: age and fare both have transformed means of approximately 0 and standard deviations of 1. The transformed columns are also stored as `age_zscore` and `fare_zscore` in `clean_titanic.csv`; they are removed before modeling because this is an EDA check, not part of the model's input features.

## Modeling results

All three classifiers use the same stratified split. The comparison table is saved in `model_metrics.csv`, and the separate classification/regression table is saved in `model_comparison.csv`.

| Classifier | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.815 | 0.797 | 0.691 | 0.740 | 0.869 |
| Decision Tree | 0.815 | 0.787 | 0.706 | 0.744 | 0.815 |
| Random Forest | 0.792 | 0.772 | 0.647 | 0.704 | 0.833 |

The confusion matrices are saved in `confusion_matrices.png`, and the three ROC curves are saved in `roc_curves.png`. The Decision Tree visualization is saved in `decision_tree.png` with transformed feature names and class names.

For imbalance handling, the baseline F1 was 0.740, class weighting produced 0.779, and SMOTE produced 0.758. Class weighting worked best on this split because it improved recall without reducing precision as much as SMOTE. SMOTE is inside the imbalanced-learn pipeline after preprocessing, so it is fitted only on the training fold.

Random Forest tuning selected `n_estimators=100`, `max_depth=6`, `max_features='sqrt'`, and `min_samples_leaf=2`, with an OOB score of about 0.812. The bounded depth and leaf-size settings reduce the chance of fitting noise in this small dataset.

The fare regression produced MAE 18.394, RMSE 41.358, R2 0.359, and adjusted R2 0.268. The residual plot is saved as `residual_plot.png`; its spread is reasonably random around zero, so there is no strong visible heteroscedasticity pattern, although the model does not explain all fare variation.

The decision tree is the selected classifier by holdout F1 at 0.744, narrowly ahead of Logistic Regression at 0.740. Logistic Regression has the best AUC at 0.869, so it ranks the classes better across thresholds. I would deploy the Decision Tree if the priority is the chosen F1 score and straightforward explanation, while keeping Logistic Regression as a strong alternative for probability ranking. Neither model should be treated as perfect; the realistic scores and the leakage removal make this comparison more trustworthy.
