from sklearn.model_selection import RepeatedKFold, cross_validate, permutation_test_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, make_scorer, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

__author__ = "Siba Mohsen"

class Classification:

    """
    computes the accuracy, F1-Score, MCC and ROC_auc scores for each of the three machine learning models:
    SVM, RF and MLP.
    """

    def __init__(self, x, y):
        self.X = x
        self.y = y

    models = {'Support Vector Machine': SVC(kernel='linear'),
              'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=2),
              'Multi Layer Perceptron': MLPClassifier(activation='logistic',
                                                      solver='lbfgs',
                                                      alpha=1e-5,
                                                      hidden_layer_sizes=(5, 2),
                                                      random_state=1)}

    scoring = {'accuracy': make_scorer(accuracy_score),
               'f1': make_scorer(f1_score),
               'MCC': make_scorer(matthews_corrcoef),
               'roc_auc': make_scorer(roc_auc_score)}
    # NOTE: MCC will warn because of a floating point error! np.seterr

    kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)

    def predict(self):
        """
        This function is responsible for splitting/training the data using the repeated k-fold cross validation.
        After preprocessing, machine learning models are employed to predict some metrics.
        The dictionaries defined above represent the models and the scoring metrics.
        Using the function cross_validate the algorithm iterates through the models and print the metrics in an
        appealing way.
        :return: metrics of each machine learning model.
        """
        results = {}
        for key, model in self.models.items():
            results.update({key: cross_validate(estimator=model,
                                                X=self.X,
                                                y=self.y,
                                                cv=self.kfold,
                                                scoring=self.scoring)})

        for k, prediction in results.items():
            print("***************************************************************************************************")
            print("                                Metric's Scores of {}".format(k))
            print("***************************************************************************************************")
            for kk, output in prediction.items():
                print(kk, "\n", output)
                print(kk, "\n Mean of predictions: ", np.mean(output),
                      "\n Standard deviation of predictions ", np.std(output), "\n")

    def permutation_predict(self):
        """
        This function returns the predicted labels of each Encoding after permutation of the labels using the
        function permutation_test_score().
        :return:
        """

        for key, model in self.models.items():
            print("********************************************{}*****************************************".format(key))
            for k, scorer in self.scoring.items():
                score, permutation_score, pvalue = permutation_test_score(estimator=model,
                                                                          X=self.X,
                                                                          y=self.y,
                                                                          cv=self.kfold,
                                                                          scoring=scorer)
                print("****{}****".format(k))
                print("Score: ", score,
                      "\n",
                      "Permutation Score: ", np.mean(permutation_score),
                      "\n",
                      "p-value: ", pvalue)
