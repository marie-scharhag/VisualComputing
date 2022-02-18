import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

f = open("covtest.dat", 'r')
v = f.read().split()
values = [float(i) for i in v]
print(values)

m = int(len(values)/2)
array = np.array(values).reshape(m,2)
print(array)

X,Y = [],[]
for x in array:
    X.append(x[0])
    Y.append(x[1])

covmatrix = np.cov(X,Y)
print(covmatrix)

mean = np.mean(array)
eigenvalues, eigenvectors = np.linalg.eig(covmatrix)

#find and sort eigenvectors and eigenvalues
order = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:,order]

#angle to rotate ellipse
vx,vy = eigenvectors[:,0][0],eigenvectors[:,0][1]
print(vx, vy)
theta = np.arctan2(vy,vx)

width, height = 2 * np.sqrt(eigenvalues)

meanX = np.mean(X)
meanY = np.mean(Y)

ell = Ellipse(xy=(meanX,meanY),width=width,height=height, angle=np.degrees(theta), color='red')

plt.figure()
ax = plt.gca()
ax.add_patch(ell)

print(mean)
print('______________________________________________________')
print(eigenvectors)
print('______________________________________________________')
print(eigenvalues)




# image = Image.new('RGB',(50,50),color=(255,255,255))
# draw = ImageDraw.Draw(image)
# for a in array:
#     draw.point((a[0]+25,a[1]+25),fill='blue')
# image = image.transpose(Image.FLIP_TOP_BOTTOM)
# image.show()
print('______________________________________________________')
# sortIndex = np.argsort(eigenvalues)[::-1]
# sortEigValue = eigenvalues[sortIndex]
# sortEigVectors = eigenvectors[:,sortIndex]
#
# subsetEigenvector = sortEigVectors[:,0:2]
#
# ced = np.dot(subsetEigenvector.transpose(),mean.transpose()).transpose()
# print(ced)

for a in array:
    plt.plot(a[0],a[1],'bx')

plt.show()