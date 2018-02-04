# Required Python Packages
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import neighbors, datasets
from sklearn.datasets import make_classification


def KNN_with_LDE():
    """ Load Raw Data """
    print("loading data ...")
    # Load the Training Data
    songs_data = pd.read_csv("train_data.csv", header=None);
    # Load the True Labels
    songs_labels = pd.read_csv("train_labels.csv", header=None);
    val_x = pd.read_csv("test_data.csv", header=None);

    """ Pre Process Data """
    print("preprocessing data ...")
    songs_data_min, songs_data_max = np.min(songs_data, 0), np.max(songs_data, 0)
    songs_data = (songs_data - songs_data_min) / (songs_data_max - songs_data_min)


    """ Split training data into training data and test data for cross validation """
    print("partitioning data ...")
    train_x, test_x, train_y, test_y = train_test_split(songs_data, songs_labels, train_size=0.8, random_state=0);

    """ Perform feature reduction"""
    print("performing feature reduction ...")
    train_x_pca = PCA(6)
    train_x_pca.fit(train_x)
    train_x_reduced = train_x_pca.transform(train_x)
    test_x_reduced = train_x_pca.transform(test_x)
    val_x_reduced = train_x_pca.transform(val_x)

    """ Build the KNN model"""
    print("building model...")
    n_neighbors = 6
    weights = 'uniform'
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights)
    clf.fit(train_x_reduced, np.asarray(train_y).ravel())

    """ Output accuracy results for Train and Test partitions"""
    print("Train Accuracy :: ", metrics.accuracy_score(train_y, clf.predict(train_x_reduced)));
    print("Test Accuracy ::  ", metrics.accuracy_score(test_y, clf.predict(test_x_reduced)));

    """ Export the classification results to for the Validation set to a csv file """
    test_data_results_knn = clf.predict(val_x_reduced)
    df = pd.DataFrame(test_data_results_knn);
    df.to_csv('songs_data_reduced.csv', index=False, header=False);

if __name__ == "__main__":
    KNN_with_LDE();