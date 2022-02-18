import cv2
import numpy as np

level = 6
horse = cv2.imread('images/horse.png')
zebra = cv2.imread('images/zebra.png')

rows, cols, dpt = horse.shape
directBlend = np.hstack((zebra[:,:int(cols/2)], horse[:,int(cols/2):]))

def CalcGaussPyramid(img):
    gausPyramid = [img]
    for i in range(level):
        img = cv2.pyrDown(img)
        gausPyramid.append(img)

    return gausPyramid

def CalcLaplacianPyramid(gausPyramid):
    laplacePyramid = [gausPyramid[level-1]]
    for i in range(level-1, 0, -1):
        g = cv2.pyrUp(gausPyramid[i])
        l = cv2.subtract(gausPyramid[i-1],g)
        laplacePyramid.append(l)

    return laplacePyramid

horseGaus = CalcGaussPyramid(horse)
horseLaplacian = CalcLaplacianPyramid(horseGaus)
zebraGaus = CalcGaussPyramid(zebra)
zebraLaplacian = CalcLaplacianPyramid(zebraGaus)

merged = []
for levelH, levelZ in zip(horseLaplacian,zebraLaplacian):
    rows,cols,dep = levelH.shape
    stack = np.hstack((levelZ[:,0:int(cols/2)],levelH[:,int(cols/2):]))
    merged.append(stack)

pyramidBlend = merged[0]
for i in range(1,level):
    pyramidBlend = cv2.pyrUp(pyramidBlend)
    pyramidBlend = cv2.add(pyramidBlend, merged[i])

cv2.imshow('Direct Blending', directBlend)
cv2.imshow('Pyramid Blending', pyramidBlend)
cv2.waitKey(0)
cv2.destroyAllWindows()