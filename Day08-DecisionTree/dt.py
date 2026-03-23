import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import dataset
dataset = pd.read_csv('loan_data.csv')

# Features (Age, Income)
X = dataset.iloc[:, [0,1]].values

# Target (Loan_Approved)
Y = dataset.iloc[:, 3].values

# Splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X ,Y, test_size=0.25, random_state=0)

# Training Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)  # "entropy" is being used here !
classifier.fit(X_train, Y_train)

# Prediction
Y_pred = classifier.predict(X_test)

# Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
print("Accuracy: ",accuracy_score(Y_test, Y_pred))

# Visualization (Training set)
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train

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

plt.title('Decision Tree (Training set)')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()


# Visualization (Test set)
from matplotlib.colors import ListedColormap
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

plt.title('Decision Tree (Test set)')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()

from sklearn.tree import export_graphviz
import graphviz

# Feature names (important for labels)
feature_names = ['Age', 'Income']

# Export tree
dot_data = export_graphviz(
    classifier,
    out_file=None,
    feature_names=feature_names,
    class_names=['Not Approved', 'Approved'],
    filled=True,
    rounded=True,
    special_characters=True
)

# Create graph
graph = graphviz.Source(dot_data)

# Save as file
from sklearn.tree import export_graphviz
import graphviz

# Feature names (important for labels)
feature_names = ['Age', 'Income']

# Export tree
dot_data = export_graphviz(
    classifier,
    out_file=None,
    feature_names=feature_names,
    class_names=['Not Approved', 'Approved'],
    filled=True,
    rounded=True,
    special_characters=True
)

# Create graph
graph = graphviz.Source(dot_data)

# Save as file
from sklearn.tree import export_graphviz
import graphviz

# Feature names (important for labels)
feature_names = ['Age', 'Income']

# Export tree
dot_data = export_graphviz(
    classifier,
    out_file=None,
    feature_names=feature_names,
    class_names=['Not Approved', 'Approved'],
    filled=True,
    rounded=True,
    special_characters=True
)

# Create graph
graph = graphviz.Source(dot_data)

# Save as file
graph.format = 'png'
graph.render("decision_tree")

# Entropy -- measure of randomness
# Entropy=−∑pi​log2​(pi​) -- data kitna mixed hai
# DECISION TREE MAGIC
# IF Income > 50000 → Approved
# ELSE → Not Approved

# | Model         | Boundary |
# | ------------- | -------- |
# | Logistic      | straight |
# | KNN           | zig-zag  |
# | SVM           | optimal  |
# | Decision Tree | box-type |

# RECTANGULAR BOUNDARY / BOX-TYPE -- decision tree boundary
