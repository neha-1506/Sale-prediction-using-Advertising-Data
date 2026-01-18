

# Case Study: Sales Prediction Using Multiple Linear Regression (Advertising Data)

---

## 1. Introduction

In today’s competitive business environment, organizations invest heavily in advertising through multiple channels such as television, radio, and newspapers. While advertising helps increase brand awareness and sales, improper allocation of advertising budgets can lead to unnecessary expenses and low returns on investment.

To address this challenge, businesses rely on **data-driven models** to understand how different advertising channels collectively influence sales. In this case study, **Multiple Linear Regression** is used to predict the **increase in sales** based on advertising expenditures across multiple media channels. Unlike simple regression, multiple regression allows the simultaneous analysis of **more than one independent variable**, making it ideal for this problem.

---

## 2. Case Study Background

The dataset contains advertising budget data collected over **200 months**. For each month, spending on **TV, Radio, and Newspaper advertisements** is recorded along with the corresponding increase in sales. The data is measured in **thousands of dollars**.

The goal is to determine:

* How each advertising channel affects sales
* Which channel provides the highest return
* How to predict sales given a combination of advertising investments

---

## 3. Problem Statement

To build a **Multiple Linear Regression model** that predicts the **increase in sales** based on advertising expenditure on TV, Radio, and Newspaper, and to derive meaningful insights that help optimize advertising budget allocation.

---

## 4. Objectives

* Implement a Multiple Linear Regression model
* Predict sales increase using multiple advertising channels
* Evaluate model performance using R², MAE, and RMSE
* Identify the most influential advertising medium

---

## 5. Dataset Description

| Feature   | Description                              |
| --------- | ---------------------------------------- |
| TV        | TV advertising budget (in $1000s)        |
| Radio     | Radio advertising budget (in $1000s)     |
| Newspaper | Newspaper advertising budget (in $1000s) |
| Sales     | Increase in sales (in $1000s)            |

---

## 6. Methodology

1. Data loading and understanding
2. Data cleaning and validation
3. Exploratory Data Analysis (EDA)
4. Correlation analysis
5. Feature-target separation
6. Train-test split
7. Model building using Multiple Linear Regression
8. Model evaluation and interpretation

---

# 7. Code Commentary with Detailed Explanation

---

## Step 1: Importing Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
```

### Explanation:

* **pandas**: data handling
* **numpy**: numerical operations
* **matplotlib & seaborn**: data visualization
* Suppressing warnings for clean output

---

## Step 2: Loading the Dataset

```python
data = pd.read_csv("Advertising.csv", index_col=0)
data.head()
```

### Explanation:

* Loads the advertising dataset.
* First column is treated as index.
* Displays initial rows to verify data.

---

## Step 3: Dataset Inspection

```python
data.info()
```

### Explanation:

* Confirms data types and non-null values.
* Ensures all variables are numerical and usable.

---

## Step 4: Checking for Missing Values

```python
data.isnull().sum()
```

### Explanation:

* Confirms dataset completeness.
* No missing values means no imputation required.

---

## Step 5: Descriptive Statistics

```python
data.describe()
```

### Explanation:

* Provides statistical overview of advertising budgets and sales.
* Helps identify data spread and scale differences.

---

## Step 6: Exploratory Data Analysis (EDA)

```python
sns.pairplot(data)
plt.show()
```

### Explanation:

* Visualizes relationships between advertising channels and sales.
* Indicates linear relationships, justifying regression use.

---

## Step 7: Correlation Analysis

```python
corr = data.corr(method="pearson")

plt.figure(figsize=(8,5))
sns.heatmap(corr, annot=True, vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()
```

### Explanation:

* Measures strength of linear relationships.
* Identifies most influential advertising channels.
* Helps detect multicollinearity.

---

## Step 8: Defining Independent and Dependent Variables

```python
X = data.drop("Sales", axis=1)
y = data["Sales"]
```

### Explanation:

* `X`: TV, Radio, Newspaper budgets
* `y`: Sales increase (target variable)

---

## Step 9: Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Explanation:

* Splits data into training and testing sets.
* Prevents overfitting and allows fair evaluation.

---

## Step 10: Building the Multiple Linear Regression Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

### Explanation:

* Trains a Multiple Linear Regression model.
* Learns combined effect of all advertising channels on sales.

---

## Step 11: Interpreting Model Coefficients

```python
coeff_df = pd.DataFrame({
    "Advertising Channel": X.columns,
    "Coefficient": model.coef_
})

coeff_df
```

### Explanation:

* Shows impact of each channel while keeping others constant.
* Higher coefficient → greater contribution to sales.
* Useful for budget allocation decisions.

---

## Step 12: Making Predictions

```python
y_pred = model.predict(X_test)
```

### Explanation:

* Predicts sales increase for unseen data.
* Used to evaluate model accuracy.

---

## Step 13: Model Evaluation

```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R² Score:", r2)
print("MAE:", mae)
print("RMSE:", rmse)
```

### Explanation:

* **R²**: proportion of variance explained by the model
* **MAE**: average absolute prediction error
* **RMSE**: penalizes large errors more heavily

---

## 8. Key Business Insights

* TV advertising has the strongest influence on sales.
* Radio advertising contributes moderately.
* Newspaper advertising has minimal impact.
* Optimal budget allocation should prioritize TV and Radio.

---

## 9. Conclusion

This case study demonstrates the effective use of **Multiple Linear Regression** to analyze and predict sales based on advertising expenditures across multiple channels. The model not only delivers accurate predictions but also provides actionable insights to optimize advertising investments and improve business profitability.

---


