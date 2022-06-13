from cProfile import label
import enum
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import classes
from perceptron import Perceptron, plot_decision_regions


"""
1. sepal length in cm 
2. sepal width in cm 
3. petal length in cm 
4. petal width in cm 
5. class: -- Iris Setosa -- Iris Versicolour -- Iris Virginica
"""




# Load the dataset using the URL:
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(s, header = None, encoding='utf-8')

# print(df.tail())

# For the purpose of this implementation, we consider the first 100 class labels that correspond to the 50 iris-setosa and 50 iris-versicolor.

#select iris setosa and versicolor
y = df.iloc[0:100, 4 ].values
y = np.where(y == 'Iris-setosa', 0,1)

#select sepal length and petal length
X = df.iloc[0:100, [0,2]].values

#visualize the 2 classes using a scatter plot:
plt.scatter(X[:50, 0], X[:50,1], color = 'red', marker = 'o', label = 'Setosa')
plt.scatter(X[50:100, 0], X[50:100,1], color = 'blue', marker = 's', label = 'Versicolor')
plt.xlabel('Sepal Length in cm')
plt.ylabel('Petal Length in cm')
plt.legend(loc = 'upper left')
plt.show()

####Train the perceptron algorithm on the iris data subset####

"""Plot the misclassification error for each epoch to check whether the algorithm converged and found a decision boundary.
    NOTE: The number of misclassification errors and the number of updates is the same, since the perceptron weights and bias are updated each time
    it misclassifies an example. 
"""

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)

plt.plot(range(1,len(ppn.errors_) + 1),
ppn.errors_, marker = 'o')

plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

# ###Function for visualizing the decision boundary###

# from matplotlib.colors import ListedColormap

# def plot_decision_regulation(X,y, classifier, resolution = 0.2):
#     #setup marker generator and colour map
#     markers = ('o', 's', '^', '<')
#     colors = ('red', 'blue', 'green', 'gray', 'cyan')
#     cmap = ListedColormap(colors[:len(np.unique(y))])

#     # plot the decision surface
#     x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
#     x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
#                             np.arange(x2_min,x2_max, resolution))
#     lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#     lab = lab.reshape(xx1.shape)
#     plt.contour(xx1,xx2,lab,alpha=0.3, cmap = cmap)
#     plt.xlim(xx1.min(), xx1.max())
#     plt.xlim(xx2.min(), xx2.max())

#     # plot class examples
#     for idx, cl in enumerate(np.unique(y)):
#         plt.scatter(x=X[y == cl, 0],
#                     y=X[y == cl, 1],
#                     alpha=0.8,
#                     c = colors[idx],
#                     marker=markers[idx],
#                     label = f'Class {cl}',
#                     edgecolors='black')


# plot_decision_regulation(X,y, classifier=ppn)
# plt.xlabel('Sepal Length in cm')
# plt.ylabel('Peral Length in cm')
# plt.legend(loc = 'upper left')
# plt.show()


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()