# Decision Trees and Random Forests

**Decision trees** are a type of machine learning algorithm used for both classification and regression tasks. They work by recursively splitting the input space into regions and then making predictions based on the majority vote or average value of the samples within each region.

The splitting is done by selecting features and thresholds that maximize the reduction in impurity, such as Gini impurity or entropy for classification and mean squared error for regression.

**Random forests** are an ensemble learning technique that constructs multiple decision trees and combines their outputs.

The idea behind random forests is that each decision tree is trained on a random subset of the training data, and each split is chosen from a random subset of features. This introduces diversity among the trees, which reduces overfitting and improves generalization. The final prediction is obtained by averaging the predictions of all the trees in the case of regression, or by majority vote in the case of classification.

## Exercise

Implement decision trees and random forests using scikit-learn.

### Decision Tree

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

# Train random forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
