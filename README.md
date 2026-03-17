# 100DaysOfML
Documenting my #100DaysOfMLChallenge where I learn and implement machine learning concepts daily.

## Day 1 – Data Preprocessing

### Goal
Learn the basic steps required to prepare raw data before applying a machine learning model.

### Concepts Learned
- Importing required Python libraries
- Loading dataset using pandas
- Separating independent variables (X) and dependent variable (Y)
- Handling missing values using SimpleImputer
- Encoding categorical data using LabelEncoder and OneHotEncoder
- Splitting dataset into training and testing sets
- Applying feature scaling using StandardScaler

### Key Idea
Machine learning models cannot work directly with raw data.  
Data preprocessing converts messy data into a clean and structured format so that models can learn from it effectively.

### Files
```
Day1-DataPreprocessing/
│
├── data.csv
└── preprocessing.py
```

---

## Day 2 – Simple Linear Regression

### Goal
Build a simple linear regression model to predict student scores based on the number of hours studied.

### Concepts Learned
- Understanding independent variable (X) and dependent variable (Y)
- Splitting dataset into training and testing sets
- Training a model using LinearRegression from scikit-learn
- Predicting results for test data
- Visualizing the regression line using matplotlib

### Key Idea
Simple Linear Regression finds the best fitting straight line between two variables.

Equation:

y = b0 + b1x

Where:
- **x** = independent variable (hours studied)  
- **y** = predicted value (student score)  
- **b0** = intercept  
- **b1** = slope of the regression line  

### Files
```
Day2-SimpleLinearRegression/
│
└── visualization
     └── simple_linear_regression_testset.png
     └── simple_linear_regression_trainingset.png
├── studentscores.csv
└── simple_linear_regression.py
```

---

## Day 3 – Multiple Linear Regression

### Goal
Build a Multiple Linear Regression model to predict startup profit using multiple independent variables such as R&D Spend, Administration cost, Marketing Spend, and State.

### Concepts Learned
- Understanding Multiple Linear Regression
- Working with datasets having multiple independent variables
- Encoding categorical data (State column) using OneHotEncoder
- Avoiding Dummy Variable Trap
- Splitting dataset into training and testing sets
- Training a Multiple Linear Regression model using LinearRegression
- Predicting results for test data
- Comparing Actual vs Predicted values
- Evaluating model performance using R² Score

### Key Idea
Multiple Linear Regression is used when the prediction depends on more than one independent variable.

Equation:

y = b0 + b1x1 + b2x2 + b3x3 + ... + bnxn

Where:

- **x1**, **x2**, **x3** = independent variables (R&D Spend, Administration, Marketing Spend, State)
- **y** = predicted value (Startup Profit)
- **b0** = intercept
- **b1**, **b2**, **b3** = coefficients that represent the impact of each feature on the prediction

In this project, the model learns the relationship between startup investments and the resulting profit.

### Files
```
Day3-MultipleLinearRegression/
│
├── 50_Startups.csv
└── multiple_linear_regression.py
```

---

## Day 4 – Logistic Regression

### Goal
Build a Logistic Regression model to predict whether a user will purchase a product based on their Age and Estimated Salary.

### Concepts Learned
- Understanding Classification problems
- Understanding Logistic Regression
- Difference between Linear Regression and Logistic Regression
- Splitting dataset into training and testing sets
- Feature Scaling using StandardScaler
- Training a Logistic Regression model using LogisticRegression
- Predicting results for test data
- Evaluating the model using Confusion Matrix
- Evaluating performance using Accuracy, Precision, Recall and F1 Score
- Visualizing the decision boundary

### Key Idea
Logistic Regression is used for **classification problems**, where the output variable is categorical (for example: Yes/No, True/False, 0/1).

Instead of predicting a continuous value like Linear Regression, Logistic Regression predicts the **probability of a class**.

The output probability is calculated using the **Sigmoid Function**.

Equation:

P = 1 / (1 + e^(-z))

Where:

z = b0 + b1x1 + b2x2

- **x1** = Age
- **x2** = Estimated Salary
- **P** = Probability that the user will purchase the product

If the probability is **greater than 0.5**, the model predicts:

1 → User will purchase

If the probability is **less than 0.5**, the model predicts:

0 → User will not purchase

The model also creates a **decision boundary**, which separates the two classes in the feature space.

### Files
```
Day4-LogisticRegression/
│
└── visualization
     └── logistic_regression_testset.png
     └── logistic_regression_trainingset.py
├── Social_Network_Ads.csv
└── logistic_regression.py
```

---

## Day 5 – K-Nearest Neighbors (KNN)

### Goal
Build a K-Nearest Neighbors (KNN) model to predict whether a user will purchase a product based on their Age and Estimated Salary.

### Concepts Learned
- Understanding K-Nearest Neighbors (KNN) algorithm
- Difference between parametric and non-parametric models
- Distance-based learning approach
- Using Euclidean distance to find nearest neighbors
- Splitting dataset into training and testing sets
- Feature Scaling using StandardScaler
- Training a KNN model using KNeighborsClassifier
- Predicting results for test data
- Evaluating the model using Confusion Matrix
- Evaluating performance using Accuracy, Precision, Recall and F1 Score
- Visualizing non-linear decision boundaries
- Effect of different values of K (overfitting vs underfitting)

### Key Idea
K-Nearest Neighbors (KNN) is a **classification algorithm** that classifies a data point based on the majority class of its nearest neighbors.

Unlike Logistic Regression, KNN does not learn an explicit equation. Instead, it stores the training data and makes predictions based on similarity (distance).

The most commonly used distance metric is **Euclidean Distance**.

Equation:

distance = √((x1 - x2)² + (y1 - y2)²)

Where:

- **x1, y1** = coordinates of a new data point
- **x2, y2** = coordinates of existing data points

Working:

- Choose a value of **K** (number of neighbors)
- Find the **K nearest points** to the new data point
- Perform **majority voting**
- Assign the class with the highest votes

Example:

If K = 5 and among nearest neighbors:
- 3 belong to class 1 (Purchased)
- 2 belong to class 0 (Not Purchased)

Then prediction → **1 (Purchased)**

The decision boundary in KNN is **non-linear (zig-zag/curved)** because it depends on the local distribution of data points.

### Files
```
Day5-KNearestNeighbors (KNN)/
│
├── visualization
│     ├── knn_testsetset.png
│     └── knn_trainingset.png
├── Social_Network_Ads.csv
└── knn.py
```

---
