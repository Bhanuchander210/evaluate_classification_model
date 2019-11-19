import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Test options and evaluation metric
validation_size = 0.3
seed = 7
scoring = 'accuracy'
kfold = 10


def get_model():
    return SVC(kernel='linear', probability=True, verbose=False)


def get_logging():
    logging.basicConfig(filename='logs.out', format='%(asctime)-15s : %(filename)s:%(lineno)s : %(funcName)s() : %(message)s', filemode='a',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    return logging


def get_models():
    models = list()
    # models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
    models.append(('SVM_LINEAR', SVC(kernel='linear', probability=True)))
    return models


class ClassifierData:

    def __init__(self, model_path, input_path, target_path):
        self.model_path = model_path
        self.input_path = input_path
        self.target_path = target_path

    def get_input_data(self):
        return np.load(self.input_path)

    def get_target_data(self):
        return np.load(self.target_path)
