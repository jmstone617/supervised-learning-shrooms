import os
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

import matplotlib as mpl
mpl.use('TkAgg')

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mushrooms.csv")
mushrooms = pd.read_csv(filepath, header=0)

print("#############################")
print(mushrooms.head())
print("#############################")
print(mushrooms.describe())

le = preprocessing.LabelEncoder()
for col in mushrooms.columns:
    mushrooms[col] = le.fit_transform(mushrooms[col])

print("#############################")
print(mushrooms.head())

predictors = [x for x in mushrooms.columns if x != "class"]

algs = {
    "GaussianNB": GaussianNB(),
    "Linear Regression": LinearRegression(normalize=True),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Extra Trees": ExtraTreesClassifier(n_estimators=5),
    "Support Vector Machine": SVC()
}

train_x, test_x, train_y, test_y = train_test_split(mushrooms[predictors], mushrooms["class"],
test_size=0.2, random_state=36)

for name, alg in algs.items():
    alg.fit(train_x, train_y)
    score = alg.score(test_x, test_y)
    print(name + ": " + str(score))

forest = algs["Extra Trees"]
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
columns = list(mushrooms.columns.values)
for f in range(train_x.shape[1]):
    print("%d. %s (%f)" % (f + 1, columns[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(train_x.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(train_x.shape[1]), indices)
plt.xlim([-1, train_x.shape[1]])
plt.show()
