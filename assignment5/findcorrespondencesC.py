'''
Exercise 2.3
find correspondences with ORB and BFMatcher
'''
import cv2

image1 = cv2.imread('images/church_left.png')
image2 = cv2.imread('images/church_right.png')

gray_img1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=2000)
kp1, des1 = orb.detectAndCompute(gray_img1, None)
kp2, des2 = orb.detectAndCompute(gray_img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

match_img = cv2.drawMatches(image1, kp1, image2, kp2, matches[:50], None)
cv2.imshow('Matches', match_img)
cv2.waitKey()