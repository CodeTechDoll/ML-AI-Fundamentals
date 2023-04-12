# Real-World Example: Loan Approval Prediction

Imagine you're a machine learning engineer working for a financial institution. Your task is to build a model that predicts whether a customer's loan application will be approved or denied based on various features such as credit score, income, employment status, and loan amount. Logistic regression is a suitable choice for this binary classification problem.

## Core Concepts

### Sigmoid function

After gathering and preprocessing the data, you train a logistic regression model. The model uses the sigmoid function to convert the output of the linear combination of features into a probability value between 0 and 1. This probability represents the likelihood of a loan being approved.

Mathematically, the sigmoid function is defined as:

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

### Decision boundary

You need to decide on a threshold to classify the loan applications as approved or denied. By default, you use a decision boundary of 0.5, meaning that loan applications with a probability greater than 0.5 are approved, and the rest are denied. However, you can adjust the threshold to make the model more conservative or liberal in approving loans, depending on the business requirements.

## Practical Tips and Considerations

### Feature scaling

Before training the model, you notice that the features have different scales (e.g., income in thousands of dollars, credit score between 300-850). To improve the model's performance and training time, you apply standardization or normalization to the dataset.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Multicollinearity

After analyzing the dataset, you find that some features are highly correlated, such as the loan amount and the applicant's income. This multicollinearity can lead to unstable parameter estimates and reduce model interpretability. You decide to use techniques like variance inflation factor (VIF) to identify and remove multicollinear features.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
```

### Regularization

To prevent overfitting and improve the model's generalization, you incorporate L1 (Lasso) or L2 (Ridge) regularization into your logistic regression model. This regularization adds a penalty term to the loss function, discouraging overly complex models.

```python
from sklearn.linear_model import LogisticRegression

# L1 regularization
model_lasso = LogisticRegression(penalty='l1', solver='liblinear')
model_lasso.fit(X_train_scaled, y_train)

# L2 regularization
model_ridge = LogisticRegression(penalty='l2', solver='lbfgs')
model_ridge.fit(X_train_scaled, y_train)
```

### Multiclass Problems

If your task involves predicting multiple loan risk categories (e.g., low risk, medium risk, high risk) instead of a binary outcome, you can extend logistic regression to handle multiclass classification using techniques like *one-vs-rest* (OvR) or *one-vs-one* (OvO).

```python
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

ovr_classifier = OneVsRestClassifier(LogisticRegression())
ovr_classifier.fit(X_train_scaled, y_train_multiclass)
```

### Model Evaluation

After training the model, you evaluate its performance using various metrics such as accuracy, precision, recall, F1-score, and area under the *receiver operating characteristic* (ROC) curve. Depending on the business requirements, you might prioritize different metrics. For example, if the financial institution wants to minimize the number of loans given to customers who will default, you might focus on maximizing precision.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC AUC:", roc_auc)
```
