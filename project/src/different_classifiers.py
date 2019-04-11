import cv2
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def classifier_knn(train_data, test_data, train_label, test_label, neigh=3):

    clf = KNeighborsClassifier(neigh)
    print("Training KNN")
    clf.fit(train_data, train_label)
    pred_label = clf.predict(test_data)
    print("Accuracy = ",accuracy_score(test_label,pred_label))
    print("Precision = ",precision_score(test_label,pred_label))
    print("Recall = ",recall_score(test_label,pred_label))
    return pred_label


def classifier_gaussianNB(train_data, test_data, train_label, test_label):
    clf = GaussianNB()
    print("Training Gaussian")
    clf.fit(train_data, train_label)
    pred_label = clf.predict(test_data)
    print("Accuracy = ",accuracy_score(test_label,pred_label))
    print("Precision = ",precision_score(test_label,pred_label))
    print("Recall = ",recall_score(test_label,pred_label))
    return pred_label


def classifier_svm(train_data, test_data, train_label, test_label):
    clf = SVC()
    print("Training SVM")
    clf.fit(train_data, train_label)
    pred_label = clf.predict(test_data)
    print("Accuracy = ",accuracy_score(test_label,pred_label))
    print("Precision = ",precision_score(test_label,pred_label))
    print("Recall = ",recall_score(test_label,pred_label))
    return pred_label


def classifier_decisionTree(train_data, test_data, train_label, test_label):
    clf = DecisionTreeClassifier()
    print("Training Decision Trees")
    clf.fit(train_data, train_label)
    pred_label = clf.predict(test_data)
    print("Accuracy = ",accuracy_score(test_label,pred_label))
    print("Precision = ",precision_score(test_label,pred_label))
    print("Recall = ",recall_score(test_label,pred_label))
    return pred_label


def classifier_BernoulliNB(train_data, test_data, train_label, test_label):
    print("Training BernoulliNB")
    clf = BernoulliNB(alpha = 1)
    clf.fit(train_data,train_label)
    pred_label = clf.predict(test_data)
    print("Accuracy = ",accuracy_score(test_label,pred_label))
    print("Precision = ",precision_score(test_label,pred_label))
    print("Recall = ",recall_score(test_label,pred_label))

print("Loading datasets...")
Xs = pickle.load(open('../data/X.pkl', 'rb'))
ys = pickle.load(open('../data/Y.pkl', 'rb'))
print("Done.")

X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size = 0.33,random_state = 42)
classifier_gaussianNB(X_train,X_test,y_train,y_test)
classifier_svm(X_train,X_test,y_train,y_test)
classifier_knn(X_train,X_test,y_train,y_test)
classifier_BernoulliNB(X_train,X_test,y_train,y_test)