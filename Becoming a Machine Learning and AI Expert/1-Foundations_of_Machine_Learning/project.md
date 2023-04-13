# Machine Learning Mini-Project

In this mini-project, you will apply the concepts learned from the previous lessons to build and evaluate machine learning models. The project is divided into three parts:

1. Implement and compare supervised and unsupervised learning algorithms.
2. Explore feature selection and understand the importance of training and test sets.
3. Analyze the bias-variance trade-off and evaluate your models using appropriate metrics.

## Part 1: Implement Supervised and Unsupervised Learning Algorithms

In this part, you will implement a supervised learning algorithm (logistic regression) and an unsupervised learning algorithm (K-means clustering) using the Iris dataset.

### Task 1.1: Supervised Learning - Logistic Regression

1. Load the Iris dataset and split it into training and test sets.
2. Train a logistic regression model on the training data.
3. Evaluate the model's performance on the test data using accuracy as the evaluation metric.

### Task 1.2: Unsupervised Learning - K-Means Clustering

1. Load the Iris dataset and remove the labels.
2. Apply K-means clustering to the dataset with the number of clusters set to 3.
3. Compare the clustering results with the true labels. Calculate the adjusted Rand index to measure the similarity between the true labels and the clustering assignments.

## Part 2: Feature Selection and Training/Test Sets

In this part, you will explore the importance of feature selection and understand the role of training and test sets.

### Task 2.1: Feature Selection

1. Train a logistic regression model using only two features of the Iris dataset.
2. Compare the model's performance to the one trained with all features. What do you observe?

### Task 2.2: Training and Test Sets

1. Split the Iris dataset into different training and test set sizes (e.g., 50/50, 70/30, 90/10).
2. Train a logistic regression model on each training set and evaluate its performance on the corresponding test set.
3. Discuss the impact of different training and test set sizes on model performance.

## Part 3: Bias, Variance, and Evaluation Metrics

In this part, you will analyze the bias-variance trade-off and evaluate your models using appropriate metrics.

### Task 3.1: Bias and Variance

1. Train logistic regression models with different levels of regularization (e.g., L1 or L2 regularization with varying strength).
2. Evaluate the models' performance on the test set and discuss the impact of regularization on bias and variance.

### Task 3.2: Evaluation Metrics

1. Evaluate your logistic regression models from Task 1.1 and Task 2.1 using precision, recall, and F1-score.
2. Discuss the differences between these metrics and explain which one might be more appropriate for the Iris dataset.

