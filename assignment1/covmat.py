"""
assignment1 Visual Computing ss21
Marie Scharhag
Aufgabe 3
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

#read data
f = open("covtest.dat", 'r')
v = f.read().split()
values = [float(i) for i in v]

#convert to numpy array
m = int(len(values)/2)
array = np.array(values).reshape(m,2)

#get x and y values
X,Y = [],[]
for x in array:
    X.append(x[0])
    Y.append(x[1])

#calculate covmatrix
covmatrix = np.cov(X,Y)
print(covmatrix)

#calculate mean, eigenvectors, eigenvalues
mean = np.mean(array)
eigenvalues, eigenvectors = np.linalg.eig(covmatrix)

#find and sort eigenvectors and eigenvalues
order = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:,order]
print(eigenvectors)

#angle to rotate ellipse
vx,vy = eigenvectors[:,0][0],eigenvectors[:,0][1]
theta = np.arctan2(vy,vx)

#width and height of the ellipse
width, height = 2 * np.sqrt(eigenvalues)

#middlepoint of the ellipse
meanX = np.mean(X)
meanY = np.mean(Y)

ell = Ellipse(xy=(meanX,meanY),width=width,height=height, angle=np.degrees(theta), color='red')

#plot Ellipse
plt.figure()
ax = plt.gca()
ax.add_patch(ell)

#plot points
for a in array:
    plt.plot(a[0],a[1],'bx')

plt.show()