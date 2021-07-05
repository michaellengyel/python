import matplotlib.pyplot as plt
import numpy as np


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
    y = np.arange(100)**2

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
    y = np.arange(100)**2

    plt.plot(x, y, marker='D', mfc='green', mec='red', ms='6', mew='1', markevery=6)

    plt.xlabel("Random Incremental Numbers")
    plt.ylabel("Data")
    plt.title("Style Data Plot")
    plt.legend()

    plt.show()


def main():

    simple_plot()
    multi_plot()
    style_plot()
    style_line()


if __name__ == '__main__':
    main()
