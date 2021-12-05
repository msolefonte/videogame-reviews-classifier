import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from lib.loader import get_data
from lib.utils.common import calculate_distribution

# Formatting

plt.rc('font', size=14)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=12)
plt.rcParams['figure.constrained_layout.use'] = True


x, y, names = get_data()
folds = 5
kf = KFold(folds)


def average(sum):
    return sum / folds


def run(neighbours):
    knn_neighbours = KNeighborsRegressor(
        n_neighbors=neighbours, weights='distance')

    mean_absolute_error_train = 0
    score_train = 0
    mean_absolute_error_test = 0
    score_test = 0
    for train_idx, test_idx in kf.split(x):
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]

        knn_neighbours.fit(x_train, y_train)

        y_pred = knn_neighbours.predict(x_train)
        # print(y_pred)
        mean_absolute_error_train = mean_absolute_error_train + \
            mean_absolute_error(y_pred, y_train)
        score_train = score_train + \
            knn_neighbours.score(x_train, y_train)

        y_pred = knn_neighbours.predict(x_test)

        mean_absolute_error_test = mean_absolute_error_test + \
            mean_absolute_error(y_pred, y_test)
        score_test = score_train + knn_neighbours.score(x_test, y_test)

    score_train = average(score_train)
    score_test = average(score_test)
    mean_absolute_error_test = average(mean_absolute_error_test)
    mean_absolute_error_train = average(mean_absolute_error_train)

    print(
        f'Training accuracy={score_train} \n Test Accuracy={score_test}')
    print(
        f'Mean Absolute Error Training={mean_absolute_error_train} \n Mean Absolute Error Test={mean_absolute_error_test}')

    plt.xlabel('score')
    plt.ylabel('Number of reviews')

    distribution, xticklables = calculate_distribution(y_pred)
    bar_range = np.array(range(len(distribution)))
    plt.bar(bar_range-0.35/2, distribution, width=0.35,
            label='Predicted', tick_label=range(len(distribution)))

    distribution, xticklables = calculate_distribution(y_test)
    plt.bar(bar_range+0.35/2, distribution, width=0.35,
            label='Real', tick_label=range(len(distribution)))

    plt.legend()
    plt.savefig(f'images/knn_{neighbours}.jpg')
    plt.clf()

    print(score_train, score_test,
          mean_absolute_error_train, mean_absolute_error_test)
    return score_train, score_test, mean_absolute_error_train, mean_absolute_error_test


def main():
    neighbours = [10, 20, 30, 50, 65, 70]
    score_train = [0.0] * len(neighbours)
    score_test = [0.0] * len(neighbours)
    mean_absolute_error_train = [0.0] * len(neighbours)
    mean_absolute_error_test = [0.0] * len(neighbours)

    for i in range(len(neighbours)):
        score_train[i], score_test[i], mean_absolute_error_train[i], mean_absolute_error_test[i] = run(
            neighbours[i])

    plt.xlabel('Neighbours')
    plt.ylabel('mean absolute error')
    plt.plot(neighbours, mean_absolute_error_train, label='train')
    plt.plot(neighbours, mean_absolute_error_test, label='test')
    plt.legend()
    plt.savefig('images/knn_vs_mean_average_error.jpg')
    plt.clf()

    plt.xlabel('Neighbours')
    plt.ylabel('Score')
    plt.plot(neighbours, score_train, label='train')
    plt.plot(neighbours, score_test, label='test')
    plt.legend()
    plt.savefig('images/knn_vs_score.jpg')
    plt.clf()


if __name__ == "__main__":
    main()
