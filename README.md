# Telecom Customer Churn Prediction Using Machine Learning

## Project Overview

This project focuses on analyzing customer behaviour and predicting customer churn in the Telecommunications industry using machine learning techniques. Customer churn refers to the situation where customers discontinue using a company's services. The project uses a telecom customer dataset to identify patterns and factors associated with customer churn, helping understand customer retention and potential business challenges. It implements and compares five supervised machine learning algorithms: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM) to determine the most suitable model for predicting customer churn.

## Objectives

1. **Data Analysis:** Explore customer information and identify patterns in churn behaviour through Exploratory Data Analysis (EDA).
2. **Data Preprocessing:** Clean the dataset by removing duplicate records, handling missing values, eliminating unnecessary columns, and preparing features for model training.
3. **Churn Pattern Identification:** Analyze the relationship between customer complaints, tariff plans, subscription length, customer value, and churn.
4. **Machine Learning Implementation:** Train five different classification algorithms to predict whether a customer is likely to leave the telecom service.
5. **Model Evaluation and Comparison:** Evaluate model performance using Accuracy, Precision, Recall, F1-score, and Balanced Accuracy.
6. **Best Model Selection:** Compare the performance of all five algorithms and identify the best-performing model based on its F1-score.
7. **Performance Visualization:** Generate visualizations, including churn distribution, customer behaviour analysis, model performance comparison, and confusion matrices.

## Technologies Used

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn
- **Machine Learning:** Logistic Regression, Decision Tree, Random Forest, KNN, SVM
- **Development Environment:** Visual Studio Code

## Key Features

- Analysis of customer churn distribution and behavioural patterns.
- Comparison of five different machine learning classification algorithms.
- Data cleaning, missing-value handling, and feature scaling.
- Evaluation using multiple performance metrics rather than accuracy alone.
- Confusion matrix and classification reports for model evaluation.
- Identification of the best-performing model using F1-score.
- Eight visualizations for understanding customer behaviour and model performance.
- Comparison against a majority-class baseline to assess model effectiveness.

## Model Evaluation

The five machine learning algorithms are evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall prediction correctness. |
| Precision | Measures how many predicted churn cases are actually churn cases. |
| Recall | Measures how many actual churn cases are correctly identified. |
| F1-score | Balances precision and recall. |
| Balanced Accuracy | Accounts for performance across both target classes. |

## Project Outcome

The project provides a comparative analysis of five machine learning algorithms for telecom customer churn prediction. Instead of relying solely on accuracy, it considers multiple evaluation metrics to obtain a more balanced understanding of model performance. The algorithm achieving the highest F1-score is selected as the best-performing model on the test dataset. The findings can help understand customer behaviour and identify patterns associated with customer retention.

## Future Improvements

- Hyperparameter tuning to improve model performance.
- Cross-validation for more reliable evaluation.
- Feature importance analysis to identify the strongest churn indicators.
- Development of an interactive dashboard for customer churn analysis.
- Deployment of the selected model as a web-based prediction application.
