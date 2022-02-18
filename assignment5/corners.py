from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import cv2

img_path = cv2.imread('images/girl.png', 1)
cv_img = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)

def rotateFkt(r_img):
    angle = rotate_value.get()
    rows, cols, a = r_img.shape
    M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, 1)
    return cv2.warpAffine(r_img, M, (cols, rows))

def scaleFkt(s_img):
    factor = scale_value.get()
    if factor > 0:
        return cv2.resize(s_img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    else:
        return s_img

def gaussian(g_img):
    xy = int(pgbf_kernelsize_value.get())
    if xy == 0:
        return g_img

    if xy > 0 and xy % 2 == 0:
        xy += 1
        pgbf_kernelsize_value.set(xy)

    return cv2.GaussianBlur(g_img, (xy, xy), 0)


def detect_corner(arg):
    rotated = rotateFkt(cv_img)
    scaled = scaleFkt(rotated)
    blurred = gaussian(scaled)
    h_img = blurred

    gray = cv2.cvtColor(h_img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    blockSize = nsize_value.get()
    k = free_value.get()
    ksize = int(sfsize_value.get())

    if ksize % 2 == 0:
        ksize += 1
        sfsize_value.set(ksize)

    dst = cv2.cornerHarris(gray, blockSize, ksize, k)
    dst = cv2.dilate(dst, None)

    threshold = threshold_value.get()
    h_img[dst > threshold * dst.max()] = [0, 0, 255]

    new_img = ImageTk.PhotoImage(Image.fromarray(h_img))
    label.configure(image=new_img)
    label.image = new_img


if __name__ == '__main__':
    root = Tk()
    root.title("Harris Corner Detection")

    img = ImageTk.PhotoImage(Image.fromarray(cv_img))
    label = Label(root, image=img)
    label.pack()

    frame = Frame(root, relief=RAISED, borderwidth=1)

    frame.pack(fill="both", expand="yes")

    # Slider for rotating and scaling the image
    sr_frame = Frame(root, borderwidth=1)
    sr_frame.pack(fill="both", expand="yes")

    rotate = Label(sr_frame, text="rotation: ")
    rotate.pack(side=LEFT, padx=5, pady=5)
    rotate_value = Scale(sr_frame, from_=-180, to=180, orient=HORIZONTAL, length=200, command=detect_corner)
    rotate_value.pack(side=LEFT, padx=5, pady=5)

    scale = Label(sr_frame, text="scale: ")
    scale.pack(side=LEFT, padx=5, pady=5)
    scale_value = Scale(sr_frame, from_=0.001, to=2, resolution=0.001, orient=HORIZONTAL, length=200, command=detect_corner)
    scale_value.pack(side=LEFT, padx=5, pady=5)
    scale_value.set(1)

    # Slider for kernel size if preliminary gaussian blur filter
    pgbf_frame = Frame(root, borderwidth=1)
    pgbf_frame.pack(fill="both", expand="yes")

    pgbf_kernelsize = Label(pgbf_frame, text="kernel size (gaussian filter): ")
    pgbf_kernelsize.pack(side=LEFT, padx=5, pady=5)
    pgbf_kernelsize_value = Scale(pgbf_frame, from_=0, to=100, orient=HORIZONTAL, length=200, command=detect_corner)
    pgbf_kernelsize_value.pack(side=LEFT, padx=5, pady=5)
    pgbf_kernelsize_value.set(2)

    # Slider for neighborhood size and kernel size of sobel filter
    nssf_frame = Frame(root, borderwidth=1)
    nssf_frame.pack(fill="both", expand="yes")

    nsize = Label(nssf_frame, text="neighborhood size: ")
    nsize.pack(side=LEFT, padx=5, pady=5)
    nsize_value = Scale(nssf_frame, from_=1, to=25, orient=HORIZONTAL, length=200, command=detect_corner)
    nsize_value.pack(side=LEFT, padx=5, pady=5)
    nsize_value.set(2)

    sfsize = Label(nssf_frame, text="kernel size (sobel filter): ")
    sfsize.pack(side=LEFT, padx=5, pady=5)
    sfsize_value = Scale(nssf_frame, from_=0, to=31, orient=HORIZONTAL, length=200, command=detect_corner)
    sfsize_value.pack(side=LEFT, padx=5, pady=5)
    sfsize_value.set(3)

    # Slider for Harris detector free paramenter
    free_frame = Frame(root, borderwidth=1)
    free_frame.pack(fill="both", expand="yes")

    free = Label(free_frame, text="free parameter: ")
    free.pack(side=LEFT, padx=5, pady=5)
    free_value = Scale(free_frame, from_=0, to=0.25, resolution=0.001, orient=HORIZONTAL, length=200, command=detect_corner)
    free_value.pack(side=LEFT, padx=5, pady=5)
    free_value.set(0.04)

    # Slider threshold for the maxima in the response map denoting corners
    threshold_frame = Frame(root, borderwidth=1)
    threshold_frame.pack(fill="both", expand="yes")

    threshold = Label(threshold_frame, text="threshold: ")
    threshold.pack(side=LEFT, padx=5, pady=5)
    threshold_value = Scale(threshold_frame, from_=0, to=0.1, resolution=0.001, orient=HORIZONTAL, length=200, command=detect_corner)
    threshold_value.pack(side=LEFT, padx=5, pady=5)
    threshold_value.set(0.01)

    root.bind("<Return>", detect_corner)
    root.mainloop()