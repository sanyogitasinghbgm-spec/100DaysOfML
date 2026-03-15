# Importing libraries
from h11 import Data
from matplotlib.style import available
import pandas as pd
import numpy as np

# Importing dataset
dataset = pd.read_csv("50_Startups.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [3])],
    remainder="passthrough"
)

X = ct.fit_transform(X)

# Avoiding dummy variable trap
X = X[:, 1:]

# Splitting dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Training the model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting test results
y_pred = regressor.predict(X_test)

# Comparing Actual vs Predicted
comparison = pd.DataFrame({
    "Actual Profit": y_test,
    "Predicted Profit": y_pred
})

print(comparison)

# Model Accuracy (R2 Score)
from sklearn.metrics import r2_score

score = r2_score(y_test, y_pred)
print("R2 Score:", score)

# Regression Coefficients
print("Intercept:", regressor.intercept_)  # b₀ value  Agar sab features 0 ho jaye, to profit approx 49032 hoga
print("Coefficients:", regressor.coef_)    # profit effect in each starstup case


# Model basically ye equation learn karta hai:
# y = b_0 + b_1 x_1 + b_2 x_2 + b_3 x_3 + \dots + b_n x_n
# Where:
# y → Profit
# x₁ → R&D Spend
# x₂ → Administration
# x₃ → Marketing Spend
# b₀ → intercept
# b₁,b₂,b₃ → coefficients
# Matlab model learn karta hai profit ka formula.


# Dataset
#    ↓
# Data preprocessing
#    ↓
# Categorical encoding
#    ↓
# Train Test Split
#    ↓
# Model Training
#    ↓
# Prediction
#    ↓
# Evaluation


# one hot encoding k wajah se erfect multicollinearity aa rhi thi 
# iska mtlb ek column baaki do se already predict ho raha hai.  NewYork = 1 − California − Florida
# X = X[:, 1:] isliye hum ek col state ka hata dete hai, taki multicollinearity na aaye.  Ab bhi State ki information available hai.


# Ab bhi State ki information available hai.

# Example:

# Florida	NewYork	State
# 0	0	California
# 1	0	Florida
# 0	1	NewYork

# Isliye prediction par koi loss nahi hota.