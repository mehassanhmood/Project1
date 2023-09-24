# Project Title: Healthcare Stroke Prediction
## Introduction:
This project focuses on predicting stroke risk using healthcare data. It covers data preprocessing, feature selection, model training, and evaluation. The primary goal is to develop a model that can accurately predict stroke risk based on various features.

Dependencies
Before running the project, ensure you have the following Python libraries and dependencies installed:

pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
imbalanced-learn (imblearn)
joblib
You can install these libraries using pip:

bash
Copy code
pip install pandas numpy scikit-learn xgboost matplotlib seaborn imbalanced-learn joblib
Data Import
The project begins by importing the dataset from the 'healthcare-dataset-stroke-data.csv' file. The dataset is explored to understand its structure, data types, and missing values.

Data Preprocessing
Handling Missing Values
Missing values in the 'bmi' column are imputed with different values depending on the 'stroke' column's value. If 'stroke' is 1, the missing 'bmi' values are imputed with the mean of 'bmi' for stroke patients. If 'stroke' is 0, the missing 'bmi' values are imputed with the mean of 'bmi' for non-stroke individuals.

Encoding Categorical Features
Categorical features are one-hot encoded using the OneHotEncoder from scikit-learn. The encoded features are concatenated with the original dataset.

Handling Imbalanced Data
To address class imbalance, random oversampling is performed on the minority class ('stroke' = 1) to match the majority class ('stroke' = 0). The oversampled dataset is then split into training and testing sets.

Model Training
Without GridSearchCV
Several machine learning models are trained without hyperparameter tuning. The following models are included:

Random Forest
Support Vector Machine (SVM)
Logistic Regression
Decision Tree
Stochastic Gradient Descent (SGD)
With GridSearchCV
The same machine learning models are trained again, but this time with hyperparameter tuning using GridSearchCV. The hyperparameter grids for each model are defined.

Feature Selection
Feature selection techniques are applied to identify the most important features. The following methods are used:

Chi-squared feature selection
Recursive Feature Elimination (RFE)
Logistic Regression-based feature selection
Random Forest-based feature selection
The top 5 features selected by each method are used for model training.

Model Evaluation
Model performance is evaluated in terms of accuracy, precision, and recall. Precision and recall are calculated for each model, and the results are visualized using bar plots.

Conclusion
This project demonstrates the process of building a stroke prediction model using healthcare data. It highlights the importance of handling missing values, encoding categorical features, addressing class imbalance, and selecting relevant features. The use of hyperparameter tuning with GridSearchCV improves model performance.
