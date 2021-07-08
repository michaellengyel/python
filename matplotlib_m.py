import matplotlib.pyplot as plt
import numpy as np
import random

from matplotlib.animation import FuncAnimation
from itertools import count


# Reference:
# https://www.youtube.com/watch?v=UO98lJQ3QGI
# https://www.youtube.com/watch?v=nKxLfUrkLE8
# https://www.youtube.com/watch?v=zZZ_RCwp49g
# https://www.youtube.com/watch?v=Ercd-Ip5PfQ


def simple_plot():
    print("Plotting simple plot:")

    x = [1, 2, 4, 3, 4, 5, 4, 3, 4, 2, 3]

    plt.plot(x)

    plt.xlabel("Natural Numbers")
    plt.ylabel("Data")
    plt.title("Simple Plot")

    plt.show()


def multi_plot():
    print("Multi Data Plot")

    x = np.array([0, 1, 3, 4, 6, 9, 12, 22, 23, 27])

    y_1 = np.array([2, 3, 3, 4, 2, 5, 3, 4, 3, 3])
    y_2 = np.array([3, 4, 5, 6, 5, 7, 9, 14, 15, 12])
    y_3 = np.random.randint(10, size=10)
    y_4 = np.random.randint(2, size=10)

    plt.plot(x, y_1, color='k', linestyle=None, marker=None, label="Data 1")
    plt.plot(x, y_2, color='c', linestyle='--', marker='o', label="Data 2")
    plt.plot(x, y_3, color='r', linestyle='-.', marker='.', linewidth='3', label="Data 3")
    plt.plot(x, y_4, color='#5a7d9a', linestyle='', marker='D', label="Data 4")

    plt.xlabel("Random Incremental Numbers")
    plt.ylabel("Data")
    plt.title("Multi Data Plot")
    plt.legend()

    plt.grid(True)
    plt.tight_layout()

    plt.show()


def style_plot():
    print(plt.style.available)
    plt.style.use('dark_background')
    # plt.xkcd() # Don't. Just don't.

    x = np.arange(100)
    y = np.arange(100) ** 2

    plt.plot(x, y)

    plt.xlabel("Random Incremental Numbers")
    plt.ylabel("Data")
    plt.title("Multi Data Plot")
    plt.legend()

    plt.grid(True)

    # plt.savefig("plot.png")
    plt.show()


def style_line():
    plt.style.use('classic')

    x = np.arange(100)
    y = np.arange(100) ** 2

    plt.plot(x, y, marker='D', mfc='green', mec='red', ms='6', mew='1', markevery=6)

    plt.xlabel("Random Incremental Numbers")
    plt.ylabel("Data")
    plt.title("Style Data Plot")
    plt.legend()

    plt.show()


def bar_plot():
    plt.rcdefaults()  # Reset plot style to None

    width = 0.25

    x = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    y_1 = np.array([3212, 3423, 2524, 5345, 7234, 5345, 5345, 2546, 5434, 6434])
    y_2 = np.array([1234, 2342, 7464, 6543, 3554, 6453, 6463, 7345, 2435, 5346])

    x_index = np.arange(len(x))

    plt.bar(x_index, y_1, width=width, label="Data 1")
    plt.bar(x_index + width, y_2, width=width, label="Data 2")

    plt.xlabel("Age")
    plt.ylabel("Weight")
    plt.title("Bar Plot")
    plt.legend()
    plt.xticks(ticks=x_index, labels=x)  # Ensures that the x labels are correct
    plt.grid(True)

    plt.show()


def fills_plot():
    x = np.arange(10)

    y_1 = np.array([2, 3, 5, 6, 7, 9, 12, 15, 17, 20])
    y_2 = np.array([6, 7, 6, 2, 4, 5, 21, 22, 24, 27])

    plt.plot(x, y_1, color='k', linestyle=None, marker=None, label="Expense")
    plt.plot(x, y_2, color='c', linestyle='--', marker='o', label="Income")

    plt.fill_between(x, y_1, y_2, where=(y_1 <= y_2), interpolate=True, color='green', alpha=0.25, label="Profit")
    plt.fill_between(x, y_1, y_2, where=(y_1 > y_2), interpolate=True, color='red', alpha=0.25, label="Loss")

    plt.xlabel("x-range")
    plt.ylabel("y-range")
    plt.title("Fill Area Between Lines Plot")
    plt.legend()

    plt.show()


def histogram_plot():
    mu, sigma = 40, 7
    y = np.random.normal(mu, sigma, 1000)
    y_sigma_calculated = np.average(y)

    plt.hist(y, bins=33)

    plt.axvline(y_sigma_calculated, color='red', label='Average(mu)')

    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.title("Histogram of gaussian data with mu=40 and sigma=7")
    plt.legend()

    plt.show()


def scatter_plot():
    x_mu, x_sigma = 47, 12
    y_mu, y_sigma = 23, 4

    x = np.random.normal(x_mu, x_sigma, 10000)
    y = np.random.normal(y_mu, y_sigma, 10000)

    plt.scatter(x, y, s=1, c='green')

    plt.axvline(np.average(x), color='red', label='x average(mu)')
    plt.axhline(np.average(y), color='blue', label='y average(mu)')

    plt.xlabel("x-data")
    plt.ylabel("y-data")
    plt.title("Scatter plot of gaussian data")
    plt.legend()

    plt.show()


def animation_plot():
    x = []
    y = []

    index = count()

    mu, sigma = 13, 3

    def animate(i):
        x.append(next(index))
        y.append(np.random.normal(mu, sigma, 1))

        if i > 100:
            del x[0]
            del y[0]

        plt.cla()  # clear axis

        plt.plot(x, y, label='Data')

        plt.legend(loc="upper left")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Gaussian data")

    animation = FuncAnimation(plt.gcf(), animate, interval=10)

    plt.show()


def animation_plot_scatter():
    x = []
    y = []

    mu_x, sigma_x = 28, 4
    mu_y, sigma_y = 36, 2

    def animate(i):

        x.append(np.random.normal(mu_x, sigma_x, 1))
        y.append(np.random.normal(mu_y, sigma_y, 1))

        if len(x) > 10:
            del x[0]
            del y[0]

        plt.cla()  # clear axis

        plt.axvline(np.average(x), color='red', label='x average(mu)')
        plt.axhline(np.average(y), color='blue', label='y average(mu)')

        plt.scatter(x, y, s=1, c='red', label='Data')

        plt.legend(loc="upper left")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Gaussian data")

        plt.xlim([0, 50])
        plt.ylim([0, 50])

    animation = FuncAnimation(plt.gcf(), animate, interval=10)

    plt.show()


def main():
    simple_plot()
    multi_plot()
    style_plot()
    style_line()
    bar_plot()
    fills_plot()
    histogram_plot()
    scatter_plot()
    animation_plot()
    animation_plot_scatter()


if __name__ == '__main__':
    main()
