# Solutions

Below are the solutions for the small tasks mentioned in the "Building Blocks" section:

1. **Visualize data**

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

plt.scatter(X, y)
plt.xlabel("Input Feature")
plt.ylabel("Target Variable")
plt.title("Scatter Plot")
plt.show()
```

2. **Check assumptions**

Refer to [this response](https://www.reddit.com/r/statistics/comments/29roo8/how_to_test_the_assumptions_of_linear_regression/) for a comprehensive guide on testing linear regression assumptions.

3. **Train a model**

```python

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
```

4. **Experiment with optimization techniques**

In scikit-learn, you can use the `LinearRegression` class for normal equations and `SGDRegressor` with `loss='squared_loss'` for gradient descent.

5. **Evaluate the model**

```python

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)
```

6. **Plot residuals**

```python

residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
```

7. **Regularization**

For Lasso (L1) regularization, use `Lasso` class from scikit-learn:

```python

from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X_train, y_train)
```

For Ridge (L2) regularization, use `Ridge` class from scikit-learn:

```python

from sklearn.linear_model import Ridge

model = Ridge(alpha=0.1)
model.fit(X_train, y_train)
```

8. **Feature scaling**

For standardization and normalization, use `StandardScaler` and `MinMaxScaler` from scikit-learn:

```python

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# or

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

9. **Feature selection**

Use `SelectKBest` and `f_regression` from scikit-learn for feature selection:

```python

from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k=2)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

10. **Cross-validation**

Use `cross_val_score` from scikit-learn for cross-validation:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("R-squared:", np.mean(scores))
```
