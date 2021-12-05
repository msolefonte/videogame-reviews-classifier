import numpy as np


def calculate_distribution(y):
    dis = np.zeros(5, int)

    xticklables = [
        '0-19', '20-49', '50-74', '75-89', '90-100'
    ]

    for value in y:
        if (value - 1) >= 90:
            tag = 4
        elif (value - 1) >= 75:
            tag = 3
        elif (value - 1) >= 50:
            tag = 2
        elif (value - 1) >= 20:
            tag = 1
        else:
            tag = 0
        dis[tag] = dis[tag] + 1

    return dis, xticklables
