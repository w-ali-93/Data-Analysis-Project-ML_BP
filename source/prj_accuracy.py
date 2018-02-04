# Required Python Packages
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import neighbors
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)


if __name__ == "__main__":
    """ Load Raw data """
    print("loading data ...")
    #Load the Training Data
    data_path_x = "train_data.csv";
    songs_data = pd.read_csv("train_data.csv", header=0);
    #Load the True Labels
    data_path_y = "train_labels.csv";
    songs_labels = pd.read_csv("train_labels.csv", header=0);
    #Load the Test Data
    data_path_z = "test_data.csv";
    val_x = pd.read_csv("test_data.csv", header=0);

    """ Pre process data"""
    print("preprocessing data ...")
    scaler = preprocessing.StandardScaler().fit(songs_data)
    scaler
    scaler.mean_
    scaler.scale_
    scaler.transform(songs_data)
    scaler.transform(val_x)

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
        #http://www.astroml.org/book_figures/chapter7/fig_S_manifold_PCA.html
        #http://scikit-learn.org/stable/modules/manifold.html
        """ Apply KNN with Modified LDE"""
        print("applying Modified LDE ...")
        n_neighbors = 22
        print("computing modified LLE embedding ...")
        prep = manifold.LocallyLinearEmbedding(n_neighbors, n_components=21, method='modified')
        train_x_mlle = prep.fit_transform(train_x)
        test_x_mlle = prep.fit_transform(test_x)
        val_x_mlle = prep.fit_transform(val_x)
        print("Done. Reconstruction error: %g" % prep.reconstruction_error_)

        """ Build the KNN model"""
        print("building model...")
        n_neighbors = 18
        weights = 'uniform'
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights)
        clf.fit(train_x_mlle, np.asarray(train_y).ravel())

        """ Output accuracy results for Train and Test partitions"""
        print("Train Accuracy :: ", metrics.accuracy_score(train_y, clf.predict(train_x_mlle)));
        print("Test Accuracy ::  ", metrics.accuracy_score(test_y, clf.predict(test_x_mlle)));

        """ Export the classification results to for the Validation set to a csv file """
        test_data_results_knn = clf.predict(val_x_mlle)
        df = pd.DataFrame(test_data_results_knn);
        df.to_csv('songs_data_reduced.csv', index=False, header=False);

    if 1:
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
        classif = linear_model.LogisticRegression(multi_class='ovr', max_iter=500, solver='newton-cg')
        classif.fit(train_x, train_y.values.ravel())
        print("**************")
        print("One vs rest classification Train Accuracy :: ", metrics.accuracy_score(train_y, classif.predict(train_x)));
        print("One vs rest classification Test Accuracy ::  ", metrics.accuracy_score(test_y, classif.predict(test_x)));
        print("**************")

        """ Predict the validation data using OVR and export predictions to CSV"""
        test_data_results_ovr = classif.predict(val_x)
        df = pd.DataFrame(test_data_results_ovr);
        df.to_csv('test_data_results_ovr.csv', index=False, header=False);