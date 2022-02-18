import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

# Create Matplotlib figure
fig, ax = plt.subplots()
ax.set(ylim=(-2,2))
title = 'Fitting sin(x) with a thrid order polynomial'
ax.set(xlabel='x', ylabel='y', title='title')
ax.add_artist(ax.legend())
ims = []

# Create random input and output data
x = np.linspace(-math.pi,math.pi,2000)
y = np.sin(x)

# randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

# set learning rate
learning_rate = 1e-6

for t in range(2000):
    # compute predicted y
    y_pred = a + b*x + c*x**2 + d*x**3

    # compute and print loss and save animation image
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)
        im = ax.plot(x, y, color='b', label='polynom')
        im = ax.plot(x, y_pred, color='k', label='polynom')
        ims.append((im))

    # backprop to compute gradients of a, b, c,d loss
    grad_y_pred = 2.0*(y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x**2).sum()
    grad_d = (grad_y_pred * x**3).sum()

    #update weights

    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

# show results
print(f'Result: y = {a} + {b}x + {c}x^2 + {d}x^3')
ani = animation.ArtistAnimation(fig,ims)
plt.draw()
plt.show()





