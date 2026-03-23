#importing libraries
import numpy as np
import pandas as pd

#importing dataset
dataset = pd.read_csv('data.csv')

print(dataset) #check

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[: , 3].values

#taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy="mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(dataset) #check

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

#creating a dummy variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])],
    remainder='passthrough'
)
X = ct.fit_transform(X)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#check
print(X) #prints the X structure, input
print(Y) #prints the Y structure, output
print(X_train.shape) #training rows -- 8, features -- 5
print(X_test.shape)  #testing rows -- 2, features -- 5
print(X_train)

# X → input features   [Country, Age, Salary]
# Y → output label     Purchased
# encoding → text ko numbers me convert
# arrays → fast computation for ml model


# Before encoding:
# Country   Age   Salary
# France    44    72000
# Spain     27    48000
# Germany   30    54000

# After encoding:
# France Germany Spain Age Salary
# 1      0       0     44 72000    
# 0      0       1     27 48000
# 0      1       0     30 54000


# One-Hot Encoding ka ide
# Country_France
# Country_Germany
# Country_Spain

# Ab agar row me France hai:
# France  → 1 0 0
# Meaning:
# Country_France  = 1
# Country_Germany = 0
# Country_Spain   = 0