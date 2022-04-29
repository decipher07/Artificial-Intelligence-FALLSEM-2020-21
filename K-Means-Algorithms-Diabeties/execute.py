# kmeans_clustering.py

# Brianna Drew
# March 27, 2021
# ID: #0622446
# Lab #9

# import required libraries and modules
import sklearn.datasets
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy

data = sklearn.datasets.load_diabetes() # load diabetes dataset
numData = data['data'] # get data excluding class labels
scaledData = preprocessing.scale(numData) # scale the data

kmeans = KMeans(n_clusters = 2, max_iter = 300) # create kMeans clustering
kmeans.fit(scaledData) # apply kMeans clustering to scaled data
predictions = kmeans.predict(scaledData) # get classification predictions resulting from the kMeans clustering

# create and show scatter plot (using third and fourth variables)
plt.scatter(scaledData[:, 2], scaledData[:, 3], c = predictions)
plt.title("Diabetes kMeans Clustering")
plt.xlabel("Body Mass Index (BMI)")
plt.ylabel("Average Blood Pressure")
plt.show()