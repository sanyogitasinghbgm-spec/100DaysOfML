#DATA PREPROCESSING
# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Load dataset
dataset = pd.read_csv('student_scores.csv')

# Step 3: Split independent and dependent variables
X = dataset.iloc[:, :1].values  #X = input feature -- hours studied
Y = dataset.iloc[:, 1].values   #Y = target / output -- scores predicted

# Step 4: Split dataset into training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0    #75% → training, 25% → testing #7 → training, 3 → testing
)

##DAY2---LINEAR REGRESSION
# Step 5: Train Simple Linear Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train) #model training

# Step 6: Predicting the test set results
Y_pred = regressor.predict(X_test)  #predict the scores 

# Step 7: Visualize the training set results
plt.scatter(X_train, Y_train, color='red')                  #red dots = real data
plt.plot(X_train, regressor.predict(X_train), color='blue') #blue line = regression line
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours studied")
plt.ylabel("Score")
plt.show()


# Step 8: Visualize the test set results
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Hours vs Scores (Test set)")
plt.xlabel("Hours studied")
plt.ylabel("Score")
plt.show()

#same window me comparison
# plt.scatter(X_train, Y_train, color='blue')
# plt.scatter(X_test, Y_test, color='red')
# plt.plot(X_train, regressor.predict(X_train), color='black')
# plt.title("Training vs Test")
# plt.show()

#blue  → training
# red   → test
# black → regression line

print("Actual:", Y_test)
print("Predicted:", Y_pred)

# Accuracy measure
from sklearn.metrics import r2_score
print("R² Score:", r2_score(Y_test, Y_pred)) #0.93
#means --> Model 93% variance explain kar raha hai