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
├── studentscores.csv
└── simple_linear_regression.py
```

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
