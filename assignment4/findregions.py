import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("images/koreanSigns.png", 0)

#template waehlen
region = cv2.selectROI(img, False)
template = img[int(region[1]):int(region[1])+int(region[3]), int(region[0]):int(region[0])+int(region[2])]
width = template.shape[0]
height = template.shape[1]

# result = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
# print(result)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
# print(min_val)
# print(max_val)
# print(min_loc)
# print(max_loc)
# # threshold = (min_val + 1e-6) * 1.5
# matches = np.where(result >= 0.8)
# for match in zip(*matches[::-1]): cv2.rectangle(img, match, (match[0] + width, match[1] + height), (0, 255, 0), 2)

result = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
matches = np.where(result >= 0.8)
for match in zip(*matches[::-1]): cv2.rectangle(img, match, (match[0] + width, match[1] + height), (0, 255, 0), 2)


cv2.imshow("matches", img)

cv2.waitKey(0)
cv2.destroyAllWindows()