# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Independent variables
X = dataset.iloc[:, [2, 3]].values  # all rows and 2nd and 3rd column (Age and Estimated Salary) ko X me store kar diya

# Dependent variable
y = dataset.iloc[:, 4].values       # all rows and 4th column (Purchased) ko y me store kar diya

# Splitting dataset into training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# 75% data → training
# 25% data → testing

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)

classifier.fit(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)  # printing confusion matrix
print("Accuracy:", classifier.score(X_test, y_test)) #accuracy is 100% because of small dataset and good separation between classes, maybe overfitting

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Visualising the Training set results

from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(
    np.arange(start=X_set[:,0].min() - 1, stop=X_set[:,0].max() + 1, step=0.01),
    np.arange(start=X_set[:,1].min() - 1, stop=X_set[:,1].max() + 1, step=0.01)
)

plt.contourf(
    X1,
    X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        color=ListedColormap(('red', 'green'))(i),
        label=j
    )

plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# Visualising the Test set results

X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(
    np.arange(start=X_set[:,0].min() - 1, stop=X_set[:,0].max() + 1, step=0.01),
    np.arange(start=X_set[:,1].min() - 1, stop=X_set[:,1].max() + 1, step=0.01)
)

plt.contourf(
    X1,
    X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        color=ListedColormap(('red', 'green'))(i),
        label=j
    )

plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



# 0 UserID
# 1 Gender
# 2 Age
# 3 EstimatedSalary
# 4 Purchased

# X = Age + Salary

# Y = Purchased 
# 0 → Not Purchased
# 1 → Purchased


# Confusion Matrix
# |          | Predicted 0 | Predicted 1 |
# | -------- | ----------- | ----------- |
# | Actual 0 | 65          | 3           |
# | Actual 1 | 8           | 24          |

# in graph --- Black Line (Decision BoundarLinear regression:

# Linear Regression
# output = number

# Logistic regression:
# output = probability between 0 and 1

# Logistic Regression is ued for class classification and uses logistic function or sigmoid function
# Formula: P = 1 / (1 + e^-z)
# Output always between 0 and 1

# Real Life Ex: 
# Young + Low Salary → Not buy
# Older + High Salary → Buy

#classification_report gives us precision, recall, f1-score and accuracy of the model. It is a more detailed report than just accuracy.

# | Metric    | Meaning                        | Result |
# | --------- | ------------------------------ | ------ |
# | Precision | predicted buyers kitne correct | 1.00   |
# | Recall    | actual buyers kitne detect     | 1.00   |
# | F1 Score  | balance of precision & recall  | 1.00   |
# | Accuracy  | total correct predictions      | 1.00   |
