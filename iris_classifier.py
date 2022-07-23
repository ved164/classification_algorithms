import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import classes
from perceptron import Perceptron, plot_decision_regions
from adaline import AdalineGD


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

# ppn = Perceptron(eta=0.1, n_iter=10)
# ppn.fit(X,y)

# plt.plot(range(1,len(ppn.errors_) + 1),
# ppn.errors_, marker = 'o')

# plt.xlabel('Epochs')
# plt.ylabel('Number of updates')
# plt.show()

# plot_decision_regions(X, y, classifier=ppn)
# plt.xlabel('Sepal length [cm]')
# plt.ylabel('Petal length [cm]')
# plt.legend(loc='upper left')
# plt.show()





# Plotting Adaline 

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X, y)
ax[0].plot(range(1, len(ada1.losses_) + 1), np.log10(ada1.losses_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title('Adaline - Learning rate 0.1')

ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.losses_) + 1), ada2.losses_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Mean squared error')
ax[1].set_title('Adaline - Learning rate 0.0001')

# plt.savefig('images/02_11.png', dpi=300)
plt.show()