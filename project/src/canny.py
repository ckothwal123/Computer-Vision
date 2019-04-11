import numpy as np
import cv2


# Non-maximal supression
def NonMaxSupWithInterpol(Gmag, deg, Gx, Gy):
    NMS = np.zeros(Gmag.shape)
    
    # Ignoring the first and the last column
    for i in range(1, int(Gmag.shape[0]) - 1):
        for j in range(1, int(Gmag.shape[1]) - 1):
            if((deg[i,j] >= 0 and deg[i,j] <= 45) or (deg[i,j] < -135 and deg[i,j] >= -180)):
                bot = np.array([Gmag[i,j+1], Gmag[i+1,j+1]])
                top = np.array([Gmag[i,j-1], Gmag[i-1,j-1]])
                alpha = np.absolute(Gy[i,j]/Gmag[i,j])
                if (Gmag[i,j] >= ((bot[1]-bot[0])*alpha+bot[0]) and Gmag[i,j] >= ((top[1]-top[0])*alpha+top[0])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((deg[i,j] > 45 and deg[i,j] <= 90) or (deg[i,j] < -90 and deg[i,j] >= -135)):
                bot = np.array([Gmag[i+1,j] ,Gmag[i+1,j+1]])
                top = np.array([Gmag[i-1,j] ,Gmag[i-1,j-1]])
                alpha = np.absolute(Gx[i,j]/Gmag[i,j])
                if (Gmag[i,j] >= ((bot[1]-bot[0])*alpha+bot[0]) and Gmag[i,j] >= ((top[1]-top[0])*alpha+top[0])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((deg[i,j] > 90 and deg[i,j] <= 135) or (deg[i,j] < -45 and deg[i,j] >= -90)):
                bot = np.array([Gmag[i+1,j] ,Gmag[i+1,j-1]])
                top = np.array([Gmag[i-1,j] ,Gmag[i-1,j+1]])
                alpha = np.absolute(Gx[i,j]/Gmag[i,j])
                if (Gmag[i,j] >= ((bot[1]-bot[0])*alpha+bot[0]) and Gmag[i,j] >= ((top[1]-top[0])*alpha+top[0])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((deg[i,j] > 135 and deg[i,j] <= 180) or (deg[i,j] < 0 and deg[i,j] >= -45)):
                bot = np.array([Gmag[i,j-1] ,Gmag[i+1,j-1]])
                top = np.array([Gmag[i,j+1] ,Gmag[i-1,j+1]])
                alpha = np.absolute(Gy[i,j]/Gmag[i,j])
                if (Gmag[i,j] >= ((bot[1]-bot[0])*alpha+bot[0]) and Gmag[i,j] >= ((top[1]-top[0])*alpha+top[0])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
    
    return NMS




# Hysterisis
def hysterisis_threshold(img):
    highThresholdRatio = 0.8  
    lowThresholdRatio = 0.3 
    hyster = np.copy(img)
    h = int(hyster.shape[0])
    w = int(hyster.shape[1])
    ht = np.max(hyster) * highThresholdRatio
    lt = ht * lowThresholdRatio    
    x = 0.1
    oldx=0
    
    # The while loop is used so that the loop will keep executing till the number of strong edges do not change, i.e all weak edges connected to strong edges have been found
    while(oldx != x):
        oldx = x
        for i in range(1,h-1):
            for j in range(1,w-1):
                if(hyster[i,j] > ht):
                    hyster[i,j] = 1
                elif(hyster[i,j] < lt):
                    hyster[i,j] = 0
                else:
                    if((hyster[i-1,j-1] > ht) or 
                        (hyster[i-1,j] > ht) or
                        (hyster[i-1,j+1] > ht) or
                        (hyster[i,j-1] > ht) or
                        (hyster[i,j+1] > ht) or
                        (hyster[i+1,j-1] > ht) or
                        (hyster[i+1,j] > ht) or
                        (hyster[i+1,j+1] > ht)):
                        hyster[i,j] = 1
        x = np.sum(hyster == 1)
    
    hyster = (hyster == 1) * hyster # This is done to remove/clean all the weak edges which are not connected to strong edges
    
    return hyster




def canny(image):
   
    # use the image given in the argument
    image = image

    # resige image as the program just to sure its 200*200
    image = cv2.resize(image,(200,200))
    
    # Convert color image to grayscale to help extraction of edges and plot it
    image_gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    
    # Blur the grayscale image so that only important edges are extracted and the noisy ones ignored
    image_gray_blurred = cv2.GaussianBlur(image_gray,(5,5),1.4)
   
    # Apply Sobel Filter in X direction
    # gx = SobelFilter(image_gray_blurred, 'x')
    Gx = np.array([[-1,0,+1], [-2,0,+2],  [-1,0,+1]])
    gx = cv2.filter2D(image_gray_blurred,-1,Gx)
    gx = gx/np.max(gx)
  
    # Apply Sobel Filter in Y direction
    # gy = SobelFilter(image_gray_blurred, 'y')
    Gy = np.array([[-1,-2,-1], [0,0,0], [+1,+2,+1]])
    gy = cv2.filter2D(image_gray_blurred,-1,Gy)
    gy = gy/np.max(gy)
  
    # Calculate the magnitude of the gradients obtained
    Mag = np.hypot(gx,gy)
    Mag = Mag/np.max(Mag)

    # Calculate direction of the gradients
    Gradient = np.degrees(np.arctan2(gy,gx))
                
    # Get the Non-Max Suppressed output
    NMS = NonMaxSupWithInterpol(Mag, Gradient, gx, gy)
    NMS = NMS/np.max(NMS)
   
    # The output of canny edge detection 
    final = hysterisis_threshold(NMS)
    # cv2.imshow('Output', final)
    # cv2.waitKey(0)

    return final