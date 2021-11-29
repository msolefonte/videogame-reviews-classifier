import numpy as np


def calculate_distribution(y):
    dis = np.zeros((11), np.int)

    for value in y:
        dis[value - 1] = dis[value - 1] + 1

    return dis
