# Importing modules

import numpy as np
import matplotlib.pyplot as plt
from time import sleep

print("Modules successfully imported")

# Inputting datas
x = np.array([5, 15, 25, 35, 45, 55], dtype=float).reshape((-1,1))
y = np.array([5, 20, 14, 32, 22, 38], dtype=float)

# Very simple linear regression (y = a x)
def simple_linear_regression(model:float, x:float) -> float:
    return model * x

# %matplotlib widget

def get_line(model:float, x):
    minx = np.min(x)
    maxx = np.max(x)
    linex = np.array([minx, maxx])
    liney = np.array(simple_linear_regression(model, linex))
    return linex, liney

# Visualize old function (individual images)
def visualize_current(model:float, x, y):
    plt.plot(x,y, 'x')
    linex = np.array([np.min(x), np.max(x)])
    liney = np.array([simple_linear_regression(model, np.min(x)), simple_linear_regression(model, np.max(x))])
    plt.plot(linex, liney)

# Visualize new function (same image)
def ax_visualize_current(line1, model:float, x):
    xx, yy = get_line(model, x)
    line1.set_xdata(xx)
    line1.set_ydata(yy)

model = 5

with plt.ion():
    # Figures and subplot
    figure = plt.figure()
    ax = figure.add_subplot(111)

    # Lines in the plot
    xx, yy = get_line(model, x)
    line1, = ax.plot(xx, yy, label="Predicted Model")
    line2 = ax.scatter(x,y, label="Training Datas")

    # Setting titles and lables on the axes
    plt.title("The datas")
    plt.xlabel("X-values")
    plt.ylabel("Y-values")
    plt.legend()


    # visualize_current(model, x, y)
    # plt.show()
    for _ in range(7):
        ax_visualize_current(line1, model, x)
        figure.canvas.draw()
        figure.canvas.flush_events()
        model *= 0.75
        sleep(0.3)