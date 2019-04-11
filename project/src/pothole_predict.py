import cv2
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import glob
import sys

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from histogram import gradient, magnitude_orientation, hog
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from canny import canny 


def classifier_gaussianNB(train_data, test_data, train_label, test_label):
    clf = GaussianNB()
    print("Training Gaussian")
    clf.fit(train_data, train_label)
    pred_label = clf.predict(test_data)
    # print(clf.predict(pred_test))
    print("Accuracy\n",accuracy_score(test_label,pred_label))
    print("Precision\n",precision_score(test_label,pred_label))
    print("Recall\n",recall_score(test_label,pred_label))
    return clf

def read_image():
    if len(sys.argv) == 2:
        image = cv2.imread(sys.argv[1],0)
        image_rgb = cv2.imread(sys.argv[1])
        image = cv2.resize(image,(200,200))
        image_rgb = cv2.resize(image_rgb,(200,200))
    else:
        print("No file input\n")
    return image,image_rgb

def compute_hog(img):
    gx, gy = gradient(img)
    mag, ori = magnitude_orientation(gx, gy)
    im1 = img
    h = hog(im1, cell_size=(4, 4), cells_per_block=(1, 1), nbins=8)
    c = h.ravel()
    return c

def localize(img2):
    edges = canny(img2)
    edges = np.array(edges * 255, dtype = np.uint8)

    image1, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    out = cv2.drawContours(img2, contours, -1, (0,250,0),1)

    try: hierarchy = hierarchy[0]
    except: hierarchy = []

    height, width, _ = img2.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours, hierarchy):
        (x,y,w,h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        if w > 80 and h > 80:
            cv2.rectangle(img2, (x,y), (x+w,y+h), (255, 0, 0), 2)

    if max_x - min_x > 0 and max_y - min_y > 0:
        cv2.rectangle(img2, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
    
    cv2.imshow("Show",img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Loading the datasets
print("Loading datasets...")
Xs = pickle.load(open('../data/X.pkl', 'rb'))
ys = pickle.load(open('../data/Y.pkl', 'rb'))
print("Done.")

print(sys.argv[1])
X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size = 0.33,random_state = 42)
clf = classifier_gaussianNB(X_train,X_test,y_train,y_test)
img, img_rgb = read_image()
hog = compute_hog(img)
value = clf.predict(hog.reshape(1,-1))
if value[0] == 1:
    print("Yes. Pothole detected\n")
    localize(img_rgb)
else:
    print("No Pothole detected\n")
        