# Types of Machine Learning

Machine learning can be broadly categorized into three main types: supervised learning, unsupervised learning, and reinforcement learning. Each type has a different approach to learning from data.

## Supervised Learning

In supervised learning, the algorithm is provided with a labeled dataset, which contains input-output pairs. The goal is to learn a mapping function that can predict the output given a new input. Examples of supervised learning algorithms include linear regression, logistic regression, and support vector machines.

### Key Lessons

- Supervised learning requires a labeled dataset.
- The algorithm learns to predict outputs based on input data.
- Commonly used for classification and regression tasks.

## Unsupervised Learning

Unsupervised learning deals with datasets that do not have labeled outputs. The objective is to discover underlying patterns, structures, or relationships in the data. Clustering and dimensionality reduction are common techniques used in unsupervised learning. Examples of unsupervised learning algorithms include K-means clustering and principal component analysis (PCA).

### Key Lessons

- Unsupervised learning works with unlabeled data.
- The algorithm aims to discover patterns or relationships in the data.
- Commonly used for clustering and dimensionality reduction tasks.

## Reinforcement Learning

Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives feedback in the form of rewards or penalties and learns to maximize the cumulative reward over time. Examples of reinforcement learning algorithms include Q-learning and Deep Q Networks (DQN).

### Key Lessons

- Reinforcement learning involves decision-making through interaction with an environment.
- The agent learns by receiving rewards or penalties.
- Commonly used for control, robotics, and game playing tasks.

## Exercise

Implement basic supervised and unsupervised learning algorithms using scikit-learn.

### Supervised Learning Example: Linear Regression

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

### Unsupervised Learning Example: K-Means Clustering

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
data, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)

# Get cluster assignments
labels = kmeans.labels_

# Print the centroids
print("Cluster centroids:\n", kmeans.cluster_centers_)
```