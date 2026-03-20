# Importing Libraries
import numpy as np
import pandas as pd

# Import dataset
dataset = pd.read_csv('spam_detection.csv')

# Features
X = dataset.iloc[:, :-1].values  # all rows and last col chhorke sab features

# Target
Y = dataset.iloc[:, -1].values   # Spam(0 or 1)

# Splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler # sab same range me a aa jaye
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training Naive Bayes
# can use both bernoulli and gaussian naive bayes as the dataset contains both continuous and binary data types

from sklearn.naive_bayes import GaussianNB # data normal distribution follow krta hai
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, Y_train)

# Prediction
Y_pred = classifier.predict(X_test)

# Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix
print("Confusion Matrix:\n", cm)
print("Accuracy: ",accuracy_score(Y_test, Y_pred))
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

# Naive Bayes caluculates
# P(spam | Features) using BAYES THEOREM
# P(A|B) = ( P(B|A) * P(A) ) / P(B)
# "given features -> spam hone ki probability kitni hai?"

# Naive Bayes 3 types...
# (i)   GaussianNB -- continuous data(numbers) like age,salry,height,marks,etc
#                  -- data normal distribution follow krta hai (bell curve)
#                  -- Student marks -> pass/fail predict krna

# (ii)  BernoulliNB -- Binary data(0/1)
#                   -- Email has "offer" -> yes/no
#                   -- feature ya to present hai ya nhi

# (iii) MultinomialNB -- count/frequency data
#                     -- frequency of features matter
#                     -- Text classification (NLP) --> "free" 3times, "offer" 5times
