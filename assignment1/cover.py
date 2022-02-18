"""
assignment1 Visual Computing ss21
Marie Scharhag
Aufgabe 4
"""
import numpy as np
import cv2

iLeft = cv2.imread('cover_left.JPG', 0)
iRight = cv2.imread('cover_right.JPG', 0)
iTop = cv2.imread('cover_top.JPG', 0)

#create new images
luminance = np.ones(iLeft.shape,np.uint8)*255
bias = np.ones(iLeft.shape,np.uint8)*255
brightness = np.ones(iLeft.shape,np.uint8)*255

#calculate images pixel by pixel
for y in range(iLeft.shape[0]):
    for x in range(iLeft.shape[1]):
        luminance[y,x] = (iLeft.item(y,x)+iTop.item(y,x)+iRight.item(y,x))/3
        bias[y,x] = iLeft.item(y,x) - iRight.item(y,x)
        brightness[y,x] = iTop.item(y,x)-((iLeft.item(y,x) + iRight.item(y,x))/2)

#show images
cv2.imshow('Image_luminance',luminance)
cv2.imshow('Image_bias',bias)
cv2.imshow('Image_brightness',brightness)
cv2.waitKey(0)
cv2.destroyAllWindows()

#safe images
cv2.imwrite('Image_luminance.jpg',luminance)
cv2.imwrite('Image_bias.jpg',bias)
cv2.imwrite('Image_brightness.jpg',brightness)