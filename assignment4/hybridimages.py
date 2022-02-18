import cv2
import numpy as np

img1 = cv2.imread('images/Marylin.png')
img2 = cv2.imread('images/John.png')

img1I = np.fft.fftshift(np.fft.fft2(img1))
img2I = np.fft.fftshift(np.fft.fft2(img2))
img1G = cv2.GaussianBlur(img1,(25,25),0)/255
img2G = img2/255 - (cv2.GaussianBlur(img2,(25,25),0)/255)+0.5

eins = np.correlate(img1,img1G)
zwei = np.correlate(img2,img2G)
# eins = np.fft.ifft2(np.fft.ifftshift(img1I*img1G))
# zwei = np.fft.ifft2(np.fft.ifftshift(img2I*img2G))

# hybrid = (img1I*img1G)+(img2I*img2G)
hybrid = eins + zwei


cv2.imshow('Image1',img1G)
cv2.imshow('Image2',img2G)
cv2.imshow('Hybrid',hybrid)
cv2.waitKey(0)
cv2.destroyAllWindows()

