# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Dataset
dataset = pd.read_csv('loan_data.csv')

# Features
X = dataset.iloc[:,[0,1]].values

# Tareget
Y = dataset.iloc[:,3].values

# Splitting Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.25, random_state=0)

# Training Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)

# Prediction
Y_pred = classifier.predict(X_test)

# Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
print("Accuracy: ", accuracy_score(Y_test, Y_pred))

# Visualization (Training Set)
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train

X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min()-1, stop=X_set[:,0].max()+1,step=1),
    np.arange(start=X_set[:, 1].min()-1000, stop=X_set[:, 1].max()+1000, step=1000)
)
plt.contourf(
    X1, X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap = ListedColormap(('red' ,'green'))
)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(
        X_set[Y_set == j ,0],
        X_set[Y_set == j, 1],
        color = ListedColormap(('red' ,'green'))(i),
        label = j
    )

plt.title('Random Forest (Training set)')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()


# Visualization (Test set)
X_set, Y_set = X_test, Y_test

X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=1),
    np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=100)
)

plt.contourf(
    X1, X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap = ListedColormap(('red' ,'green'))
)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(
        X_set[Y_set == j ,0],
        X_set[Y_set == j, 1],
        color = ListedColormap(('red' ,'green'))(i),
        label = j
    )

plt.title('Random Forest (Test set)')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()

# Ye Decision Tree ka upgraded version hai
# Ek tree nahi → 100 trees (n_estimators=100) ban rahe hai
# Har tree apna prediction deta hai → final answer = majority vote

# Boundary:
# Decision Tree → box-type (rectangles)
# Random Forest → multiple boxes combine → smoother but still blocky

# So:
# Decision Tree → ek bada box
# Random Forest → chote chote boxes ka group