"""
assignment2 Visual Computing ss21
Marie Scharhag
Aufgabe 6
"""
import numpy as np
import cv2 as cv2

data = {}
#get matrix and distortion coeffients from data.txt (aus Werten von Aufgabe davor)
with open('data.txt') as lines:
    for line in lines.readlines():
        l = line.replace("[", "").replace("]", "").split(',')
        matrix = np.array([x.split() for x in l[1].split(";")], dtype=float)
        distortion = np.array(l[2].split(), dtype=float)
        data.update({'calibrationImagesCheckerboard/'+l[0]: [matrix, distortion]})

for img, values in data.items():
    print(img)
    cv2.namedWindow(img + "_original", cv2.WINDOW_NORMAL)
    cv2.namedWindow(img + "_new", cv2.WINDOW_NORMAL)
    #read Image
    orig = cv2.imread(img, 1)
    #undistort image
    undist = cv2.undistort(orig, values[0], values[1])
    #resize Windows
    undist = cv2.resize(undist, (0, 0), fx=0.2, fy=0.2)
    orig = cv2.resize(orig, (0, 0), fx=0.2, fy=0.2)
    #show new and original image
    cv2.imshow(img + "_original", orig)
    cv2.imshow(img + "_new", undist)
    cv2.moveWindow(img+ "_new",800, 0)
    cv2.waitKey(500)
    cv2.destroyAllWindows()


