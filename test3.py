import matplotlib.pyplot as plt
import numpy as np

with plt.ion():
    y = np.random.random([10,1])
    plt.plot(y)
    for i in range(50):
        y = np.random.random([10,1])
        plt.plot(y)
        plt.draw()
        plt.pause(0.020)
        plt.clf() # to clear last drawing before drawing onto it.