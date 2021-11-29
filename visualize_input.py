import matplotlib.pyplot as plt
import numpy as np
import lib

x, y, names = lib.get_data()

def calculate_distribution(y):
    dis = np.zeros((11), np.int64)

    for value in y:
        dis[value - 1] = dis[value - 1] + 1

    return dis

plt.xlabel('Rating')
plt.ylabel('Number of reviews')
distribution = calculate_distribution(y)
bar_range = np.array(range(len(distribution)))
plt.bar(bar_range - 0.35 / 2, distribution, width=0.35,
        label='Input Data', tick_label=range(len(distribution)))

plt.legend()
plt.savefig(f'input_data_distribution.jpg')
plt.clf