from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
CHART_DIR = BASE_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)

# Load the source data and keep a local copy for the next script.
df = sns.load_dataset("titanic")
print("Raw data shape:", df.shape)
print(df.info())
print(df.describe())

df.to_csv(BASE_DIR / "titanic.csv", index=False)

df = pd.read_csv(BASE_DIR / "titanic.csv")
print("\nMissing value percentages:")
print(df.isna().mean().mul(100).sort_values(ascending=False))

# `deck` is mostly missing, so it is left out. The smaller gaps are filled below.

df_clean = df.drop(columns=["deck"]).copy()
df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())
df_clean["embarked"] = df_clean["embarked"].fillna(df_clean["embarked"].mode()[0])
df_clean["embark_town"] = df_clean["embark_town"].fillna(df_clean["embark_town"].mode()[0])

print("\nAge missing count after fill:", df_clean["age"].isna().sum())
print("Embarked missing count after fill:", df_clean["embarked"].isna().sum())
print("Embark town missing count after fill:", df_clean["embark_town"].isna().sum())

# Count outliers with the IQR rule.
for col in ["age", "fare"]:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((df_clean[col] < lower) | (df_clean[col] > upper)).sum())
    print(f"{col} outliers using IQR rule: {count}")

fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode().iloc[0]
print(f"Fare mean={fare_mean:.3f}, median={fare_median:.3f}, mode={fare_mode:.3f}")
print("Fare distribution is right-skewed because mean > median > mode.")

# Bivariate survival rates.
survival_by_sex = df_clean.groupby("sex")["survived"].mean()
survival_by_pclass = df_clean.groupby("pclass")["survived"].mean()
survival_by_sex_pclass = df_clean.groupby(["sex", "pclass"])["survived"].mean().unstack()
print("\nSurvival by sex:")
print(survival_by_sex)
print("\nSurvival by pclass:")
print(survival_by_pclass)
print("\nSurvival by sex and pclass:")
print(survival_by_sex_pclass)

# Compare the main numeric columns.
correlation_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
correlation_matrix = df_clean[correlation_columns].corr()
print("\nCorrelation matrix:")
print(correlation_matrix)

corr_abs = correlation_matrix.where(~np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)).stack().abs()
# Ignore the diagonal and sort the remaining pairs.
strongest_pairs = corr_abs[corr_abs.index.get_level_values(0) != corr_abs.index.get_level_values(1)].sort_values(ascending=False)
print("\nTop two absolute off-diagonal correlations:")
print(strongest_pairs.head(2))

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Titanic correlation heatmap")
plt.tight_layout()
plt.savefig(CHART_DIR / "corr_heatmap.png", dpi=200)
plt.close()

# Single-variable plots.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(df_clean["age"], bins=25, kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Age distribution")

sns.boxplot(x=df_clean["age"], ax=axes[0, 1])
axes[0, 1].set_title("Age boxplot")

sns.histplot(df_clean["fare"], bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Fare distribution")

sns.boxplot(x=df_clean["fare"], ax=axes[1, 1])
axes[1, 1].set_title("Fare boxplot")

plt.tight_layout()
plt.savefig(CHART_DIR / "age_fare_distribution.png", dpi=200)
plt.close()

# A few comparisons between variables.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.barplot(data=df_clean, x="sex", y="survived", estimator="mean", ax=axes[0, 0])
axes[0, 0].set_title("Survival rate by sex")

sns.barplot(data=df_clean, x="pclass", y="survived", estimator="mean", ax=axes[0, 1])
axes[0, 1].set_title("Survival rate by passenger class")

sns.boxplot(data=df_clean, x="sex", y="age", ax=axes[1, 0])
axes[1, 0].set_title("Age by sex")

sns.scatterplot(data=df_clean, x="fare", y="age", hue="survived", alpha=0.7, ax=axes[1, 1])
axes[1, 1].set_title("Fare vs age by survival")

plt.tight_layout()
plt.savefig(CHART_DIR / "multivariate_story.png", dpi=200)
plt.close()

# Check the standardized age and fare values.
for col in ["age", "fare"]:
    z_col = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std(ddof=0)
    print(f"{col} z-score mean={z_col.mean():.6f}, std={z_col.std(ddof=0):.6f}")

# Save the cleaned data for modeling.
df_clean.to_csv(BASE_DIR / "clean_titanic.csv", index=False)

print("\nEDA complete. Charts saved to:", CHART_DIR)
