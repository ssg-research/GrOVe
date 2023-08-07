# Authors: Asim Waheed, Vasisht Duddu
# Copyright 2020 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier

class ThresholdClassifier:
    def fit(self, train_X, train_Y):
        if train_X.ndim == 2:
            train_X = train_X.reshape(-1)

        assert len(train_X) == len(train_Y), "Train X should be 1-Dimensional"
        # Get precision/recall for different threshold values
        min_thresh = 1
        max_thresh = 25
        num = (max_thresh - min_thresh)*10 + 1
        thresholds = np.linspace(min_thresh, max_thresh, num=num)

        best_fscore = 0
        for threshold in thresholds:
            predictions = np.where(train_X <= threshold, 1, 0)
            fscore = f1_score(train_Y, predictions)
            if fscore > best_fscore:
                best_threshold = threshold
                best_fscore = fscore

        print("Best Threshold:", best_threshold)

        self.threshold_ = best_threshold

    def predict(self, train_X):
        if train_X.ndim == 2:
            train_X = train_X.reshape(-1)

        return np.where(train_X <= self.threshold_, 1, 0)

def train_model(train_X, train_Y):
    return train_mlp(train_X, train_Y)

def train_mlp(train_X, train_Y):
    parameters = {'hidden_layer_sizes':[(64, 64), (128, 128)], 
                    'activation':['tanh', 'relu']}

    mlp = MLPClassifier(max_iter=300)

    # clf=mlp
    clf = GridSearchCV(mlp, parameters, 
						cv=10, scoring='accuracy', 
						return_train_score=True,
						n_jobs=4)

    clf.fit(train_X, train_Y)

    accuracy = evaluate(clf, train_X, train_Y)

    print("Best parameters:", clf.best_params_)

    # return clf, accuracy
    return clf.best_estimator_, accuracy

def train_threshold_clf(train_X, train_Y):

    clf = ThresholdClassifier()

    clf.fit(train_X, train_Y)

    evaluate(clf, train_X, train_Y)

    return clf

def train_svm(train_X, train_Y):
    parameters = {'kernel':('poly' , 'rbf'), 'nu':[0.01, 0.1, 0.5, 1]}

    svm = OneClassSVM()

    # clf=svm
    clf = GridSearchCV(svm, parameters, 
						cv=10, scoring='accuracy', 
						return_train_score=True,
						n_jobs=4)

    clf.fit(train_X, train_Y)

    evaluate(clf, train_X, train_Y)

    print("Best parameters:", clf.best_params_)

    # return clf
    return clf.best_estimator_

def evaluate(clf, X, Y_true):
    Y_preds = clf.predict(X)

    accuracy = accuracy_score(Y_true, Y_preds)

    print("Accuracy: ", accuracy)

    conf_mat = confusion_matrix(Y_true, Y_preds) 

    print("Confusion Matrix:")
    print(conf_mat)

    return accuracy