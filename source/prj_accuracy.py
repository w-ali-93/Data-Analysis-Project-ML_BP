# Required Python Packages
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)


def feature_reduction(dataset, n_components, batch_size, whiten):
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size, whiten=True)
    Xn = ipca.fit_transform(dataset)
    return Xn


if __name__ == "__main__":
    """ Load Raw data """
    print("loading data ...")
    #Load the Training Data
    data_path_x = "train_data.csv";
    songs_data = pd.read_csv("train_data.csv", header=None);
    #Load the True Labels
    data_path_y = "train_labels.csv";
    songs_labels = pd.read_csv("train_labels.csv", header=None);
    #Load the Test Data
    data_path_z = "test_data.csv";
    val_x = pd.read_csv("test_data.csv", header=None);

    """ Pre process data"""
    print("preprocessing data ...")
    # scaler = preprocessing.StandardScaler().fit(train_x)
    # scaler
    # scaler.mean_
    # scaler.scale_
    # scaler.transform(train_x)
    # scaler.transform(test_x)
    songs_data_min, songs_data_max = np.min(songs_data, 0), np.max(songs_data, 0)
    X = (songs_data - songs_data_min) / (songs_data_max - songs_data_min)

    """ Partition data"""
    print("partitioning data ...")
    train_x, test_x, train_y, test_y = train_test_split(songs_data, songs_labels, train_size=0.8, random_state=0);

    """ Perform feature reduction"""
    #Perform PCA on the Train Partition """
    print("performing feature reduction ...")
    train_x_pca = PCA(264)
    train_x_pca.fit(train_x)
    #Transform The Training Partition with the PCA"""
    train_x_reduced = train_x_pca.transform(train_x)
    test_x_reduced = train_x_pca.transform(test_x)
    val_x_reduced = train_x_pca.transform(val_x)

    if 0:
        """ Apply MLR"""
        #apply multinomial logistic regression
        print("applying MLR ...")
        mul_lr = linear_model.LogisticRegression(multi_class='multinomial', max_iter=1000, solver='newton-cg').fit(train_x, train_y.values.ravel());
        print("**************")
        print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)));
        print("Multinomial Logistic regression Test Accuracy ::  ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)));
        print("**************")

        """ Predict the validation data using MLR and export predictions to CSV"""
        # Classify the test data using multinomial logistic regression
        test_data_results_mlr = mul_lr.predict(val_x)
        df = pd.DataFrame(test_data_results_mlr);
        df.to_csv('test_data_results_mlr.csv', index=False, header=False);

    if 0:
        """ Apply OVR"""
        #apply one vs rest classification
        print("applying OVR ...")
        classif = OneVsRestClassifier(SVC(C=0.7, cache_size=800, class_weight=None, coef0=0.0,
            decision_function_shape = 'ovr', degree = 3, kernel = 'poly',
            max_iter = 1000, probability = False, random_state = None, shrinking = True,
            tol = 0.001, verbose = False))
        classif.fit(train_x, train_y.values.ravel())
        print("**************")
        print("One vs rest classification Train Accuracy :: ", metrics.accuracy_score(train_y, classif.predict(train_x)));
        print("One vs rest classification Test Accuracy ::  ", metrics.accuracy_score(test_y, classif.predict(test_x)));
        print("**************")

        """ Predict the validation data using OVR and export predictions to CSV"""
        test_data_results_ovr = classif.predict(val_x)
        df = pd.DataFrame(test_data_results_ovr);
        df.to_csv('test_data_results_ovr.csv', index=False, header=False);

    if 1:
        n_neighbors = 10
        print("computing modified LLE embedding ...")
        prep = manifold.LocallyLinearEmbedding(n_neighbors, n_components=9,
                                              method='modified')
        train_x_mlle = prep.fit_transform(train_x)
        test_x_mlle = prep.fit_transform(test_x)
        val_x_mlle = prep.fit_transform(val_x)
        print("Done. Reconstruction error: %g" % prep.reconstruction_error_)

        # now apply multinomial logistic regression
        print("applying MLR ...")
        mul_lr = linear_model.LogisticRegression(multi_class='multinomial', max_iter=1000, solver='newton-cg').fit(
            train_x_mlle, train_y.values.ravel());
        print("**************")
        print("Multinomial Logistic regression w/ LLE Train Accuracy :: ",
              metrics.accuracy_score(train_y, mul_lr.predict(train_x_mlle)));
        print("Multinomial Logistic regression w/ LLE Test Accuracy ::  ",
              metrics.accuracy_score(test_y, mul_lr.predict(test_x_mlle)));
        print("**************")

        """ Predict the validation data using MLR and export predictions to CSV"""
        # Classify the test data using multinomial logistic regression
        test_data_results_mlr = mul_lr.predict(val_x_mlle)
        df = pd.DataFrame(test_data_results_mlr);
        df.to_csv('test_data_results_mlr_lle.csv', index=False, header=False);