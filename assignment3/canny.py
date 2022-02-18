import cv2
import numpy as np

img = cv2.imread('src.png',0)

sobelx = cv2.Sobel(img, cv2.CV_8U, 1, 0)
sobely = cv2.Sobel(img, cv2.CV_8U, 0, 1)
length = cv2.magnitude(np.float32(sobelx), np.float32(sobely))

canny = cv2.Canny(img,100,100)

cv2.imshow('Canny',canny)
cv2.imshow('Length',np.uint8(length))

cv2.waitKey(0)
cv2.destroyAllWindows()