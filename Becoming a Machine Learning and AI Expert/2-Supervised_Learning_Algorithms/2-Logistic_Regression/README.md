# Logistic Regression

Logistic Regression is a supervised learning algorithm used for binary classification problems, where the goal is to predict one of two possible outcomes. It is an extension of linear regression, with the main difference being the use of a logistic function (sigmoid function) to transform the linear regression output into a probability value, which is then thresholded to make a binary classification decision.

## Applications

Logistic regression is widely used in various domains, including but not limited to:

1. Medical diagnosis: Predicting the presence or absence of a disease based on patient data.
2. Spam detection: Classifying emails as spam or not spam based on their content.
3. Customer churn prediction: Predicting if a customer will stop using a service or product.
4. Credit risk assessment: Predicting if a customer will default on a loan or not.

## Training and Evaluation

Training a logistic regression model involves finding the coefficients that maximize the likelihood of observing the given data. This is typically achieved by minimizing the cross-entropy loss, which is a measure of the difference between the predicted probabilities and the actual class labels. The optimization can be done using various techniques, such as gradient descent or more advanced optimization algorithms like L-BFGS or Newton-Raphson.

To evaluate the performance of a logistic regression model, you can use evaluation metrics such as accuracy, precision, recall, F1-score, or area under the receiver operating characteristic (ROC) curve.

## Training

The logistic regression model can be represented as:

$$
P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n)}}
$$

where:

- $P(y=1|x)$ is the probability of the target variable being 1 (positive class) given the input features $x$
- $x_i$ are the input features
- $\beta_i$ are the coefficients to be estimated

The objective is to find the coefficients $\beta_i$ that maximize the likelihood of observing the given data. This optimization problem can be solved using various techniques, such as gradient descent or more advanced optimization algorithms like L-BFGS or Newton-Raphson.

## Evaluation

To evaluate the performance of a logistic regression model, you can use evaluation metrics such as accuracy, precision, recall, F1-score, or area under the receiver operating characteristic (ROC) curve.

## Exercise

Implement a logistic regression model using scikit-learn and TensorFlow.

### Logistic Regression using scikit-learn

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

### Logistic Regression using TensorFlow

```python
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create logistic regression model using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(2,), activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy:", accuracy)
```
