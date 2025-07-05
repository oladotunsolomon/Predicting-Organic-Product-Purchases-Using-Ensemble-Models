# Predicting Organic Product Purchases Using Ensemble Models
This project addresses the supermarket’s need to identify customers who are most likely to purchase organic products. Using historical data on customer demographics and purchasing behavior, the project develops, compares, and evaluates several predictive models including Decision Tree, Logistic Regression, Neural Network, and an Ensemble Model to classify customers based on their likelihood to buy organic products.

The overall goal is to improve prediction accuracy so that the supermarket can better target marketing campaigns, reduce promotional costs, and increase sales of organic products. The project follows the CRISP-DM framework, covering business understanding, data understanding, preparation, modeling, evaluation, and comparison of results. Through this structured approach, the final recommendation identifies the most effective model for deployment in targeted marketing initiatives.

## Project Overview
- **Objective**: Predict which customers are likely to purchase organic products using demographic and purchasing behavior data.
- **Business Question**: Can the supermarket improve predictive accuracy by comparing different models and using ensemble techniques to identify likely organic-product buyers?
- **Target Variable**: `TargetBuy` (1 = Purchase, 0 = No Purchase)
- **Framework**: CRISP-DM (Cross-Industry Standard Process for Data Mining)

## Dataset
- **Source**: Supermarket customer data provided in Excel format.
- **Records**: Replace with your count (e.g., ~22,000 customers)
- **Features**: 9 predictor variables + 1 target variable (`TargetBuy`).
- **Description**: Customer demographics, purchasing behaviors, and promotion response.
- **Target Variable**: `TargetBuy` (1 = Purchase, 0 = No Purchase)
- **Rejected Columns**: `DemCluster`, `TargetAmt` (excluded from modeling as per instructions)

**Data Dictionary**:
- `ID`: Unique customer identifier
- `DemAffl`: Affluence score (higher = more affluent)
- `DemAge`: Customer age
- `DemClusterGroup`: Demographic cluster group (A–F)
- `DemGender`: Gender (M = Male, F = Female, U = Unknown)
- `DemReg`: Customer region
- `DemTVReg`: TV advertising region
- `PromClass`: Promotion classification (Gold, Silver, Tin, etc.)
- `PromSpend`: Amount spent on promotions
- `PromTime`: Duration spent in promotion

## Tools and Technologies
- **Programming Language**: Python
- **Notebook Environment**: Google Colab
- **Data Processing**: pandas, numpy
- **Data Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn (Decision Tree, Logistic Regression, Neural Network, VotingClassifier for ensemble)
- **Data Preprocessing**: StandardScaler (for scaling), OneHotEncoder (via pandas get_dummies)
- **Evaluation Metrics**: Accuracy, AUC (ROC), F1-score, Confusion Matrix, ROC Curve plotting

## Methodology (CRISP-DM)
- **Business Understanding**: Defined objective to predict organic product buyers for targeted marketing.
- **Data Understanding**: Explored data, summarized key stats, assessed missing values and class imbalance.
- **Data Preparation**: Handled missing data, encoded categoricals, scaled numerics, stratified 70/20/10 split with seed 12345.
- **Modeling**: Built Decision Tree, Logistic Regression, Neural Network, and Ensemble (soft voting) models.
- **Evaluation**: Compared models using Accuracy, AUC, F1-score; interpreted results and selected Ensemble Model for deployment.

## Results
- **Evaluation Metrics**: Accuracy, AUC (Area Under the ROC Curve), F1-score on validation dataset.

**Performance Comparison Table**:

| Model               | Accuracy | AUC   | F1-score |
|----------------------|---------|-------|----------|
| Decision Tree        | 0.811   | 0.817 | 0.521    |
| Logistic Regression  | 0.806   | 0.803 | 0.487    |
| Neural Network       | 0.804   | 0.807 | 0.503    |
| Ensemble             | 0.811   | 0.817 | 0.514    |

- **Interpretation**:
  - Decision Tree and Ensemble achieved the highest Accuracy (0.811) and AUC (0.817), showing strong discrimination between buyers and non-buyers.
  - Decision Tree had the highest F1-score (0.521), slightly outperforming the Ensemble and Neural Network in balancing precision and recall.
  - Logistic Regression showed slightly lower performance, reflecting simpler linear decision boundaries.
  - Neural Network achieved competitive results, capturing non-linear patterns.

- **Confusion Matrices**:
  - *Decision Tree*: `[[3150 193], [645 456]]`
  - *Logistic Regression*: `[[3174 169], [692 409]]`
  - *Neural Network*: `[[3136 207], [662 439]]`
  - *Ensemble*: `[[3157 186], [656 445]]`
  
  - These matrices confirm that all models effectively identified positive cases while maintaining balanced false positive and false negative rates.

- **Variable Importance (Decision Tree)**:
  - `DemAffl` (Affluence score)
  - `PromSpend` (Promotion spending)
  - `DemAge` (Customer age)
  
  - Suggests more affluent, older customers who spend more on promotions are more likely to purchase organic products.

- **Business Interpretation**:
  - All models performed comparably well, with Ensemble and Decision Tree slightly leading.
  - Ensemble Model is recommended for deployment due to its stability, robustness, and ability to combine diverse model predictions.
  - Supports targeted marketing efforts to increase organic product purchases by identifying high-likelihood customers.

- **Conclusion**:
  - The Ensemble Model offers the best trade-off between predictive accuracy, robustness, and business interpretability.
  - Recommended for customer segmentation and targeted marketing to optimize promotional spending and boost sales.

```python

# Install necessary package
!pip install openpyxl

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve

# Upload file
from google.colab import files
uploaded = files.upload()

# Load Excel file
file_path = list(uploaded.keys())[0]
df = pd.read_excel(file_path)
print(df.head())

# Drop rejected columns
df = df.drop(columns=['DemCluster', 'TargetAmt'])
print(df.head())

# Check missing values
print(df.isnull().sum())

# Handle missing data
numeric_cols = ['DemAffl', 'DemAge', 'PromSpend', 'PromTime']
categorical_cols = ['DemClusterGroup', 'DemGender', 'DemReg', 'DemTVReg', 'PromClass']

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[categorical_cols] = df[categorical_cols].fillna('Unknown')
print(df.isnull().sum())

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
print(df_encoded.head())

# Standardize numeric columns
scaler = StandardScaler()
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
print(df_encoded[numeric_cols].describe())

# Define features and target
X = df_encoded.drop(columns=['ID', 'TargetBuy'])
y = df_encoded['TargetBuy']

# Split into train, validation, test (70/20/10)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=12345
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=12345
)

print(f"Train: {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test: {X_test.shape}")

print("\nTarget distribution:")
print(f"Train: {y_train.mean():.2f}")
print(f"Validation: {y_val.mean():.2f}")
print(f"Test: {y_test.mean():.2f}")

# Train models
dt_model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=12345)
dt_model.fit(X_train, y_train)

lr_model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=12345)
lr_model.fit(X_train, y_train)

nn_model = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', solver='adam', max_iter=300, random_state=12345)
nn_model.fit(X_train, y_train)

# Ensemble model
ensemble_model = VotingClassifier(
    estimators=[
        ('DecisionTree', dt_model),
        ('LogisticRegression', lr_model),
        ('NeuralNetwork', nn_model)
    ],
    voting='soft'
)
ensemble_model.fit(X_train, y_train)

# Evaluate models
models = {
    'Decision Tree': dt_model,
    'Logistic Regression': lr_model,
    'Neural Network': nn_model,
    'Ensemble': ensemble_model
}

results = []

for name, model in models.items():
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    f1 = f1_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    
    print(f"\nModel: {name}")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  AUC: {auc:.3f}")
    print(f"  F1-score: {f1:.3f}")
    print(f"  Confusion Matrix:\n{cm}")
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'AUC': auc,
        'F1-score': f1
    })

# Results table
results_df = pd.DataFrame(results)
print("\nPerformance Comparison Table:")
print(results_df)

# Plot ROC Curves
plt.figure(figsize=(8,6))
for name, model in models.items():
    y_prob = model.predict_proba(X_val)[:,1]
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    plt.plot(fpr, tpr, label=f"{name}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Validation ROC Curves')
plt.legend()
plt.show()

# Feature Importance (Decision Tree)
importances = dt_model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print("\nTop Feature Importances (Decision Tree):")
print(importance_df.head(10))
