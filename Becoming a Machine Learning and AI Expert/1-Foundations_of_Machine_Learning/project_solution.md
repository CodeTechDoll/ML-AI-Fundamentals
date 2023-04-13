# Machine Learning Mini-Project Solution

## Part 1: Implement Supervised and Unsupervised Learning Algorithms

### Task 1.1: Supervised Learning - Logistic Regression

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test)

# Evaluate the model's performance using accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

### Task 1.2: Unsupervised Learning - K-Means Clustering

```python
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get cluster assignments
cluster_assignments = kmeans.labels_

# Calculate the adjusted Rand index
ari = adjusted_rand_score(y, cluster_assignments)
print("Adjusted Rand index:", ari)
```

## Part 2: Feature Selection and Training/Test Sets
### Task 2.1: Feature Selection

```python
# Train a logistic regression model using only two features
X_two_features = X[:, :2]
X_train_2f, X_test_2f, y_train_2f, y_test_2f = train_test_split(X_two_features, y, test_size=0.2, random_state=42)

lr_2f = LogisticRegression(max_iter=200, random_state=42)
lr_2f.fit(X_train_2f, y_train_2f)

# Evaluate the model's performance
y_pred_2f = lr_2f.predict(X_test_2f)
accuracy_2f = accuracy_score(y_test_2f, y_pred_2f)
print("Accuracy with two features:", accuracy_2f)

# The model's performance is lower with only two features, suggesting that the other features contain valuable information.
```

### Task 2.2: Training and Test Sets

```python
# Split the Iris dataset into different training and test set sizes
split_ratios = [(0.5, 0.5), (0.7, 0.3), (0.9, 0.1)]

for train_size, test_size in split_ratios:
    X_train_ratio, X_test_ratio, y_train_ratio, y_test_ratio = train_test_split(X, y, test_size=test_size, random_state=42)

    lr_ratio = LogisticRegression(max_iter=200, random_state=42)
    lr_ratio.fit(X_train_ratio, y_train_ratio)
    y_pred_ratio = lr_ratio.predict(X_test_ratio)
    accuracy_ratio = accuracy_score(y_test_ratio, y_pred_ratio)

    print(f"Accuracy with {train_size * 100}/{test_size * 100} split:", accuracy_ratio)

# The model's performance generally improves with larger training set sizes, indicating the importance of having sufficient training data.
```

## Part 3: Bias, Variance, and Evaluation Metrics

### Task 3.1: Bias and Variance

```python