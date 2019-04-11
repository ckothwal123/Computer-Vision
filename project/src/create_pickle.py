from histogram import gradient, magnitude_orientation, hog
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import pickle

X_pothole = np.zeros((78,20000))
Y_pothole = np.ones((78))


images_clean = glob.glob('../data/clean/*.jpg')
i = 0
for fname in images_clean:
    i+=1

X_no_pothole = np.zeros((i,20000))
Y_no_pothole = np.zeros((i))

k = 0
print("Clean images\n")
for fname in images_clean :
    
    img = cv2.imread(fname,0)
    img = cv2.resize(img,(200,200))
    gx, gy = gradient(img)
    mag, ori = magnitude_orientation(gx, gy)
    im1 = img
    h = hog(im1, cell_size=(4, 4), cells_per_block=(1, 1), nbins=8)
    c = h.ravel()
    X_no_pothole[k] = c
    k +=1
    print("Image ",k)


images_pothole = glob.glob('../data/pothole/*.jpg')
j = 0 
for fname in images_pothole:
    j+=1

p=0
print("Pothole image begins\n")
for fname in images_pothole :
    
    img = cv2.imread(fname,0)
    img = cv2.resize(img,(200,200))
    gx, gy = gradient(img)
    mag, ori = magnitude_orientation(gx, gy)
    im1 = img
    h = hog(im1, cell_size=(4, 4), cells_per_block=(1, 1), nbins=8)
    d = h.ravel()
    # print(j)
    X_pothole[p] = d
    p +=1
    print(p)
    

X_all_true = np.concatenate((X_pothole, X_no_pothole), axis=0)
Y_all_true = np.concatenate((Y_pothole,Y_no_pothole),axis=0)
print(X_all_true)
print(Y_all_true)
pickle.dump(X_all_true,open("../data/X.pkl",'wb'))
pickle.dump(Y_all_true,open("../data/Y.pkl",'wb'))