import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import cv2
import copy


img_path_left = 'images/church_left.png'
img_path_right = 'images/church_right.png'

cv_left = cv2.imread(img_path_left, 1)
cv_right = cv2.imread(img_path_right, 1)

cv_img = np.concatenate((cv_left, cv_right), axis=1)

height, width, channels = cv_img.shape
cv_img_left = cv_img[:, :int(width / 2)]
cv_img_right = cv_img[:, int(width / 2):]


def goodFeaturesCorners(srcs_img,template_area,color):
    detected_img = copy.copy(srcs_img)

    gray = cv2.cvtColor(detected_img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.goodFeaturesToTrack(gray, 15, 0.05, 10)
    corners = np.int0(corners)

    templates = []
    corns = []
    for corner in corners:
        x, y = corner.ravel()
        corns.append([x,y])
        temp = srcs_img[y - template_area:y + template_area, x - template_area:x + template_area]
        height, width = temp.shape[:2]
        if height == template_area * 2 and width == template_area * 2:
            templates.append(temp)
        cv2.circle(detected_img, (x, y), 4, color, 1)

    return detected_img, np.array(templates), corns


def harrisCorners(srcs_img, template_area, color):
    detected_img = copy.copy(srcs_img)

    gray = cv2.cvtColor(detected_img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)


    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    templates = []
    for corner in corners:
        x = int(corner[0])
        y = int(corner[1])
        temp = srcs_img[y - template_area:y + template_area, x - template_area:x + template_area]
        height, width = temp.shape[:2]
        if height == template_area * 2 and width == template_area * 2:
            templates.append(temp)
        cv2.circle(detected_img, (x, y), 4, color, 1)

    return detected_img, np.array(templates), corners


def match_corners(*args):
    if cornerDet.get() == 'HCD':
        detection_left, templates_left, corners_left = harrisCorners(cv_img_left, 30, (255, 0, 0))
        detection_right, templates_right, corners_right = harrisCorners(cv_img_right, 30, (0, 255, 0))
    elif cornerDet.get() == 'GFTT':
        detection_left, templates_left, corners_left = goodFeaturesCorners(cv_img_left, 30, (255, 0, 0))
        detection_right, templates_right, corners_right = goodFeaturesCorners(cv_img_right, 30, (0, 255, 0))


    points_left = []
    points_right = []

    for a in range(0, len(templates_left)):
        for b in range(0, len(templates_right)):
            if stvar.get() == "SSD":
                res = cv2.matchTemplate(templates_left[a], templates_right[b], cv2.TM_SQDIFF)
            if stvar.get() == "NCC":
                res = cv2.matchTemplate(templates_left[a], templates_right[b],cv2.TM_CCORR_NORMED)

            if res[0][0] > 0.95:
                points_left.append((corners_left[a][0], corners_left[a][1]))
                points_right.append((corners_right[b][0], corners_right[b][1]))

    stitched = np.concatenate((detection_left, detection_right), axis=1)

    for i in range(len(points_left)):
        cv2.line(stitched, (int(points_left[i][0]),int(points_left[i][1])), (int(points_right[i][0] + (width / 2)), int(points_right[i][1])), (255, 255, 0),1)

    new_img = ImageTk.PhotoImage(Image.fromarray(stitched))
    label.configure(image=new_img)
    label.image = new_img

if __name__ == '__main__':
    # GUI
    root = Tk()
    root.title("Find Correspondences")

    img = ImageTk.PhotoImage(Image.fromarray(cv_img))
    label = Label(root, image=img)
    label.pack()

    frame = Frame(root, relief=RAISED, borderwidth=1)

    frame.pack(fill="both", expand="yes")

    stvar = tk.StringVar()
    stvar.set("NCC")
    filterOption = stvar.option = tk.OptionMenu(frame, stvar, "SSD", "NCC", command=match_corners)
    filterOption.pack(side=LEFT, padx=5, pady=5)

    cornerDet = tk.StringVar()
    cornerDet.set("HCD")
    filterOption2 = cornerDet.option = tk.OptionMenu(frame, cornerDet, "HCD", "GFTT", command=match_corners)
    filterOption2.pack(side=LEFT, padx=5, pady=5)

    root.bind("<Return>", match_corners)
    root.mainloop()