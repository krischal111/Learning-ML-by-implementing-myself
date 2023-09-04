from matplotlib import pyplot as plt
from IPython.display import clear_output
import numpy as np

with plt.ion():
    for i in range(50):
        y = np.random.random([10,1])
        plt.plot(y)
        clear_output(wait=True)
        # print("Something")
        print("I am somßßething")
        plt.draw()
        plt.pause(0.02)
        plt.clf()