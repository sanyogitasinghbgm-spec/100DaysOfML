# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Independent variables (input/ features)
X = dataset.iloc[:, [2, 3]].values  # all rows and 2nd and 3rd column (Age and Estimated Salary) ko X me store kar diya

# Dependent variables (output/ target)
Y = dataset.iloc[:, 4].values       # all rows and 4th column (Purchased) ko Y me store kar diya 

# Splitting dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # model seekhta nhi hai bass data store hota hai
X_test = sc.transform(X_test)

# KNN Model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
classifier.fit(X_train, Y_train)

# Prediction
Y_pred = classifier.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

print("Accuracy: ", classifier.score(X_test, Y_test))
print(classification_report(Y_test, Y_pred))


#Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train 

X1, X2 = np.meshgrid(
    np.arange(X_set[:,0].min() - 1, X_set[:, 0].max() + 1, 0.01),
    np.arange(X_set[:,1].min() - 1, X_set[:, 1].max() + 1, 0.01)
)

plt.contourf(
    X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75, cmap=ListedColormap(('red', 'green'))           
)

for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(
        X_set[Y_set == j, 0],
        X_set[Y_set == j, 1],
        color=ListedColormap(('red', 'green'))(i),
        label=j
    )

plt.title('KNN (Training set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Visualising the Test set results

from matplotlib.colors import ListedColormap

X_set, Y_set = X_test, Y_test

X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
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

for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(
        X_set[Y_set == j, 0],
        X_set[Y_set == j, 1],
        color=ListedColormap(('red', 'green'))(i),
        label=j
    )

plt.title('KNN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



# Try different k values and see the effect on accuracy and decision boundary   
# | K value | Behavior                |
# | ------- | ----------------------- |
# | K = 1   | overfitting (too noisy) |
# | K = 5   | balanced                |
# | K = 10  | smooth boundary         |

# can use elbow method to choose best k value

# Logistic Regression me:
# 👉 straight line boundary

# KNN me:
# 👉 zig-zag / curved boundary 

# Kyuki:
# decision depends on nearby points


# ⚔️ Comparison: Linear vs Logistic vs KNN
# 🔵 1. Linear Regression
# Output → number
# Example: predict salary = 50000
# 👉 continuous values

# 🟢 2. Logistic Regression
# # Output → probability → class
# Boundary → straight line
# 👉 fast + simple, based on sigmoid or logistic function equation

# 🔴 3. KNN
# Output → class
# No equation
# # Boundary → irregular / zig-zag
# 👉 based on distance and majority voting in class



# | Algorithm           | Best For              |
# | ------------------- | --------------------- |
# | Linear Regression   | numeric prediction    |
# | Logistic Regression | simple classification |
# | KNN                 | complex patterns      |


# to determin k = 1, k= 3 or k = 5 id good for thi samll datset...we can do
# for k in [1, 3, 5]:
#     model = KNeighborsClassifier(n_neighbors=k)
#     model.fit(X_train, Y_train)
#     print("K =", k, "Accuracy =", model.score(X_test, Y_test))