# Bias, Variance, and Evaluation Metrics

In machine learning, it is essential to understand the concepts of bias, variance, and evaluation metrics to build accurate and efficient models. These concepts help you diagnose and address issues related to model performance.

## Bias

Bias refers to the error introduced by approximating a real-world problem using a simplified model. In other words, it represents the difference between the model's predictions and the true values. High bias occurs when the model makes strong assumptions about the data and is unable to capture the underlying patterns, leading to underfitting.

### Key Lessons

- Bias is the error caused by a simplified model.
- High bias leads to underfitting.
- To reduce bias, use more complex models or incorporate additional features.

## Variance

Variance refers to the error introduced by a model's sensitivity to small fluctuations in the training data. A model with high variance is likely to overfit the training data, meaning it captures noise and random patterns that are not representative of the overall data distribution.

### Key Lessons

- Variance is the error caused by a model's sensitivity to training data fluctuations.
- High variance leads to overfitting.
- To reduce variance, use simpler models, increase training data size, or apply regularization techniques.

## Evaluation Metrics

Evaluation metrics are quantitative measures used to assess the performance of machine learning models. They help you understand how well your model is performing and guide your model selection process. Different metrics are appropriate for different types of problems, such as classification or regression tasks.

### Common Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, F1-score, Area Under the ROC Curve (AUC-ROC)
- **Regression**: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared

## Quiz

Test your understanding of bias, variance, and common evaluation metrics.

1. What is the primary cause of bias in a machine learning model?

   a. Overfitting
   b. Simplified model assumptions
   c. Sensitivity to training data fluctuations
   d. Inadequate training data size

2. How can you reduce variance in a machine learning model?

   a. Use more complex models
   b. Increase the size of the training data
   c. Decrease the number of features
   d. Use non-linear models

3. Which of the following evaluation metrics is most appropriate for a binary classification problem?

   a. Mean Absolute Error (MAE)
   b. Mean Squared Error (MSE)
   c. F1-score
   d. R-squared

### Answers

1. b. Simplified model assumptions
2. b. Increase the size of the training data
3. c. F1-score
