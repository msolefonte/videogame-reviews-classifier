import numpy as np


def calculate_distribution(y):
    dis = np.zeros(11, np.int)

    for value in y:
        dis[int((value - 1)/10)] = dis[int((value - 1)/10)] + 1

    return dis
