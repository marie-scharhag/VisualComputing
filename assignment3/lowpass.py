import cv2

def change(self):
    pass

img = cv2.imread('src.png',1)

cv2.namedWindow('Blur')

minS = 0
maxS = 30

cv2.createTrackbar('Gaussian','Blur',minS,maxS,change)
cv2.createTrackbar('Box','Blur',minS,maxS,change)
cv2.createTrackbar('Bilateral','Blur',minS,maxS,change)
cv2.createTrackbar('Median','Blur',minS,maxS,change)

while(True):
    gaus = cv2.getTrackbarPos('Gaussian','Blur')
    gaus = 2 * gaus - 1
    box = cv2.getTrackbarPos('Box','Blur')
    bil = cv2.getTrackbarPos('Bilateral','Blur')
    medi = cv2.getTrackbarPos('Median','Blur')
    medi = 2 * medi - 1

    if medi > 0:
        cv2.imshow('Blur',cv2.medianBlur(img,medi))
    elif gaus > 0:
        cv2.imshow('Blur',cv2.GaussianBlur(img,(gaus,gaus),0))
    elif bil > 0:
        cv2.imshow('Blur',cv2.bilateralFilter(img,bil,100,100))
    elif box > 0:
        cv2.imshow('Blur',cv2.boxFilter(img,-1,(box,box)))
    else:
        cv2.imshow('Blur', img)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()