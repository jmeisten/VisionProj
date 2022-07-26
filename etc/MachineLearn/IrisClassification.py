import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import mglearn
from IPython.display import display

# In this intro example we have to create a model to identifiy iris' (flowers) based on petal dim  sepal dim
# GOAL:
# BUILD AN ML MODEL TO PREDICT TYPE OF IRIS FLOWERS WHEN PRESENTED AN IMAGE WHEN GIVEN A DATASET. MUST CLASSIFY AS PROPER SPECIES.

# Encapsulates algorithm used to build model from training data (face rec?)
knn = KNeighborsClassifier(n_neighbors=1)

# Load prebuilt model for detection iris' 
iris_dataset = load_iris()

# ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
print ("Keys from Iris dataset: \n{}".format(iris_dataset.keys()))

# Prints the dataset description 
print(iris_dataset['DESCR'] + "\n...")
# print("Target names: {}".format(iris_dataset['target_names']))
# print("Feature names: {}".format(iris_dataset['feature_names']))
# print("fraame: {}".format(iris_dataset['frame']))
# print("target: {}".format(iris_dataset['target']))
# print("data: {}".format(iris_dataset['data']))

# Load data and create a testing set of ~25% for the model
# X = data y = label

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

print("X train: {}".format(X_train.shape))
print("X testy: {}".format(X_test.shape))
print("y train: {}".format(y_train.shape))
print("y test: {}".format(y_test.shape))

# Create datafram from data in x train
# Label columns using the labels from feature names
irisDataFrame = pd.DataFrame(X_train,columns=iris_dataset.feature_names)
# Create a scatter matrix from dataframe coloured by y_train
pd.plotting.scatter_matrix(irisDataFrame,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins': 20},s=60,alpha=.8,cmap=mglearn.cm3)


# Using K nearest neighbor method
# This takes the training data and stores it. Modifieing its own instance
knn.fit(X_train,y_train)

# Now we can model. Lets create a fake iris [Speal l, Sepal W, Petal L, Petal W}]
# All data comparisons must be done with np arrays
X_new = np.array([[5,2.9,1,.02]])

# Now we do the prediction
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted name: {}".format(iris_dataset['target_names'][prediction]))

# Cool it got setosa. Is that right? Lets eval the model
# We know the labels corresponding to X_test so lets test that entire data set to check accuracy
y_pred = knn.predict(X_test)
print("Test set preictions: \n {}".format(y_pred))

# Mean score of the data to 2 decimal
# np.mean(y_pred == y_test) averages a compraison array that would look like
# (True, False, False, True, True, ....) with True=1 and False=0
print("Test set score : {:.2F}".format(np.mean(y_pred == y_test)))

# RESULT
# You just made your very first supervised learning model``