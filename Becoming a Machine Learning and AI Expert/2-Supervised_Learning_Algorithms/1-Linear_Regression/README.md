# Linear Regression

Linear regression is a simple supervised learning algorithm used for predicting a continuous target variable based on one or more input features. It is a fundamental technique in machine learning and serves as a good starting point for understanding more complex models.

## Assumptions

Linear regression makes the following assumptions about the data:

1. **Linearity**: There is a linear relationship between the input features and the target variable.
2. **Independence**: The input features are independent of each other (i.e., there is no multicollinearity).
3. **Homoscedasticity**: The variance of the errors is constant across all levels of the input features.
4. **Normality**: The errors are normally distributed.

It is important to check these assumptions before using linear regression, as violations can lead to inaccurate or unreliable predictions.

## Training and Evaluation

Training a linear regression model involves finding the coefficients that minimize the sum of squared errors between the actual and predicted target values. This can be done using various optimization techniques, such as gradient descent or normal equations.

To evaluate the performance of a linear regression model, you can use evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), or R-squared.

## Training

Training a linear regression model involves finding the coefficients that minimize the sum of squared errors between the actual and predicted target values. The linear regression model can be represented as:

$$
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon
$$

where:

- $y$ is the target variable
- $x_i$ are the input features
- $\beta_i$ are the coefficients to be estimated
- $\epsilon$ is the error term

The objective is to find the coefficients $\beta_i$ that minimize the cost function, which is the sum of squared errors:

$$
J(\beta) = \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$

where:

- $m$ is the number of observations
- $y_i$ is the actual target value
- $\hat{y}_i$ is the predicted target value

This optimization problem can be solved using various techniques, such as gradient descent or normal equations.

### Gradient Descent

Gradient descent is an iterative optimization algorithm that updates the coefficients by moving in the direction of the steepest decrease of the cost function. The coefficients are updated using the following formula:

$$
\beta_j := \beta_j - \alpha \frac{\partial J(\beta)}{\partial \beta_j}
$$

where:

- $\alpha$ is the learning rate
- $\frac{\partial J(\beta)}{\partial \beta_j}$ is the partial derivative of the cost function with respect to $\beta_j$

### Normal Equations

Normal equations provide a closed-form solution to the linear regression problem by finding the coefficients that minimize the cost function directly. The solution can be computed using the following formula:

$$
\beta = (X^TX)^{-1}X^Ty
$$

where:

- $X$ is the matrix of input features
- $y$ is the target variable

## Evaluation

To evaluate the performance of a linear regression model, you can use evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), or R-squared.

### Mean Absolute Error (MAE)

Mean Absolute Error is the average of the absolute differences between the actual and predicted target values. It is defined as:

$$
\text{MAE} = \frac{1}{m}\sum_{i=1}^{m} |y_i - \hat{y}_i|
$$

MAE is easy to interpret, as it gives the average error in the same unit as the target variable.

### Mean Squared Error (MSE)

Mean Squared Error is the average of the squared differences between the actual and predicted target values. It is defined as:

$$
\text{MSE} = \frac{1}{m}\sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$

MSE emphasizes larger errors, as the differences are squared before being averaged. It is widely used in optimization and has a clear relationship to the cost function used in linear regression.

$$
\text{MSE} = \frac{1}{m}\sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$

MSE emphasizes larger errors, as the differences are squared before being averaged. It is widely used in optimization and has a clear relationship to the cost function used in linear regression.

### Root Mean Squared Error (RMSE)

Root Mean Squared Error is the square root of the MSE. It is defined as:

$$
\text{RMSE} = \sqrt{\frac{1}{m}\sum_{i=1}^{m} (y_i - \hat{y}_i)^2}
$$

RMSE is in the same unit as the target variable, which makes it more interpretable than the MSE. It is sensitive to outliers and large errors.
R-squared

R-squared, also known as the coefficient of determination, measures the proportion of the variance in the target variable that is predictable from the input features. It is defined as:

$$
R^2 = 1 - \frac{\sum_{i=1}^{m} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{m} (y_i - \bar{y})^2}
$$
where:

- $\bar{y}$ is the mean of the target variable

R-squared ranges from 0 to 1, with higher values indicating better model performance. It is a widely used metric for evaluating the goodness of fit of a regression model.

## Simple Real World Example

Suppose you have data on the number of hours studied by students and their corresponding exam scores. You could use a linear regression model to predict a student's exam score based on the number of hours they studied. The input feature (hours studied) and the target variable (exam score) are expected to have a linear relationship. By training and evaluating the linear regression model, you can determine how well the model captures this relationship and how accurate its predictions are.

## Comparisons in Use Cases

Linear regression is well-suited for problems where the relationship between input features and the target variable is linear. It is simple to implement and computationally efficient, making it a good choice for small datasets and as a baseline model. However, linear regression is limited by its assumptions and may not perform well on complex datasets with non-linear relationships or where the assumptions are not met. In such cases, more advanced models like decision trees, support vector machines, or neural networks may be more appropriate.

## Exercise

Implement a linear regression model using scikit-learn and TensorFlow.

### Linear Regression using scikit-learn

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate sample data
X = np.random.rand(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print("R-squared:", score)
```

### Linear Regression using TensorFlow

```python
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Generate sample data
X = np.random.rand(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(10)

# Define the linear regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss='mean_squared_error')

# Train the model
model.fit(train_dataset, epochs=100, verbose=0)

# Evaluate the model
y_pred = model.predict(X_test)
mse = tf.keras.losses.mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse.numpy())
```
