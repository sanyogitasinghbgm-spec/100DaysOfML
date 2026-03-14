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
