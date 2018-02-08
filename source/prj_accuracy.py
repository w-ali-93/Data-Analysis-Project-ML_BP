# Required Python Packages
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import neighbors
from sklearn import manifold


if __name__ == "__main__":
    """Load data"""
    print("loading data ...")
    #Load the Training Data
    train_data = pd.read_csv("train_data.csv", header=0);
    #Load the True Labels
    train_labels = pd.read_csv("train_labels.csv", header=0);
    #Load the Test Data Used For Generating Predictions That Will Be Validated By Kaggle Checker
    test_data = pd.read_csv("test_data.csv", header=0);

    """Preprocess data"""
    print("preprocessing data ...")
    scaler = preprocessing.StandardScaler().fit(train_data)
    scaler
    scaler.mean_
    scaler.scale_
    scaler.transform(train_data)
    scaler.transform(test_data)

    """Partition data"""
    print("partitioning data ...")
    train_x, test_x, train_y, test_y = train_test_split(train_data, train_labels, train_size=0.8, random_state=0);

    """Perform feature reduction"""
    # #Perform PCA on the Train Partition
    # print("performing feature reduction ...")
    # train_x_pca = PCA(264)
    # train_x_pca.fit(train_x)
    # #Transform The Training Partition with the PCA
    # train_x = train_x_pca.transform(train_x)
    # test_x = train_x_pca.transform(test_x)
    # test_data = train_x_pca.transform(test_data)

    if 0:
        #http://www.astroml.org/book_figures/chapter7/fig_S_manifold_PCA.html
        #http://scikit-learn.org/stable/modules/manifold.html
        """Apply KNN with LDE"""
        print("applying LDE ...")
        n_neighbors = 22
        print("computing modified LLE embedding ...")
        prep = manifold.LocallyLinearEmbedding(n_neighbors, n_components=21, method='modified')
        train_x = prep.fit_transform(train_x)
        test_x = prep.fit_transform(test_x)
        test_data = prep.fit_transform(test_data)
        print("Done. Reconstruction error: %g" % prep.reconstruction_error_)

        print("applying KNN ...")
        n_neighbors = 18
        weights = 'uniform'
        classif1 = neighbors.KNeighborsClassifier(n_neighbors, weights)
        classif1.fit(train_x, np.asarray(train_y).ravel())

        """Output accuracy results for Train and Test partitions"""
        print("KNN Train Accuracy :: ", metrics.accuracy_score(train_y, classif1.predict(train_x)));
        print("KNN Test Accuracy ::  ", metrics.accuracy_score(test_y, classif1.predict(test_x)));

        """Predict the validation data and export predictions to CSV"""
        test_data_results_knn = classif1.predict(test_data)
        df = pd.DataFrame(test_data_results_knn);
        df.to_csv('test_data_results_knn.csv', index=False, header=False);

    if 0:
        """Apply MLR"""
        #apply multinomial logistic regression
        print("applying MLR ...")
        classif2 = linear_model.LogisticRegression(multi_class='multinomial', max_iter=1000, solver='newton-cg');
        classif2.fit(train_x, train_y.values.ravel())

        print("Multinomial Logistic Regression Train Accuracy :: ", metrics.accuracy_score(train_y, classif2.predict(train_x)));
        print("Multinomial Logistic Regression Test Accuracy ::  ", metrics.accuracy_score(test_y, classif2.predict(test_x)));

        """Predict the validation data and export predictions to CSV"""
        test_data_results_mlr = classif2.predict(test_data)
        df = pd.DataFrame(test_data_results_mlr);
        df.to_csv('test_data_results_mlr.csv', index=False, header=False);

    if 1:
        """Apply OVR"""
        #apply one vs rest classification
        print("applying OVR ...")
        classif3 = linear_model.LogisticRegression(multi_class='ovr', max_iter=500, solver='newton-cg')
        classif3.fit(train_x, train_y.values.ravel())

        print("One vs Rest Classification Train Accuracy :: ", metrics.accuracy_score(train_y, classif3.predict(train_x)));
        print("One vs Rest Classification Test Accuracy ::  ", metrics.accuracy_score(test_y, classif3.predict(test_x)));

        """Predict the validation data and export predictions to CSV"""
        test_data_results_ovr = classif3.predict(test_data)
        df = pd.DataFrame(test_data_results_ovr);
        df.to_csv('test_data_results_ovr.csv', index=False, header=False);