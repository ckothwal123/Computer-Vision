from histogram import gradient, magnitude_orientation, hog
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

def read_image():
    if len(sys.argv) == 2:
        image = cv2.imread(sys.argv[1],0)
        image = cv2.resize(image,(200,200))
    else:
        print("No file input\n")
    return image

# img = octagon()
img = read_image()
img = cv2.resize(img,(200,200))
gx, gy = gradient(img)
mag, ori = magnitude_orientation(gx, gy)

# Show gradient and magnitude
plt.figure()
plt.title('gradients and magnitude')
plt.subplot(141)
plt.imshow(img, cmap=plt.cm.Greys_r)
plt.subplot(142)
plt.imshow(gx, cmap=plt.cm.Greys_r)
plt.subplot(143)
plt.imshow(gy, cmap=plt.cm.Greys_r)
plt.subplot(144)
plt.imshow(mag, cmap=plt.cm.Greys_r)


# Show the orientation deducted from gradient
plt.figure()
plt.title('orientations')
plt.imshow(ori)
plt.pcolor(ori)
plt.colorbar()

plt.show()
