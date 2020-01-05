# fruit classification problem using a labeled dataset for apple, mandarin, orange and lemon

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

fruit = pd.read_table('fruit_data_with_colors.txt')

# splitting the data into train and test data sets 75% and 25%

X = fruit[['height', 'width', 'mass', 'color_score']]
y = fruit['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# 3d scatterplot

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')
plt.show()

# create classifier object:

knn = KNeighborsClassifier(n_neighbors=5)

# train the classifier

knn.fit(X_train, y_train)

# see the accuracy

knn.score(X_test, y_test)
