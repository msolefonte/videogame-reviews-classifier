import pickle
import numpy as np
import matplotlib.pyplot as plt
from lib.loader import get_data
from lib.utils.common import calculate_distribution
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# Formatting

plt.rc('font', size=14)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=12)
plt.rcParams['figure.constrained_layout.use'] = True


x, y, names = get_data()
kFold = KFold()


def run_model(num=100, draw_plot=False):
    model = RandomForestRegressor(n_estimators=num, n_jobs=-1)

    mean_average_error_train = 0
    mean_average_error_test = 0

    for train_idx, test_idx in kFold.split(x):
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]

        model.fit(x_train, y_train)

        y_pred = model.predict(x_train)
        mean_average_error_train = mean_average_error_train + \
            mean_absolute_error(y_pred, y_train)

        y_pred = model.predict(x_test)
        mean_average_error_test = mean_average_error_test + \
            mean_absolute_error(y_pred, y_test)

    mean_average_error_train = mean_average_error_train / 5
    mean_average_error_test = mean_average_error_test / 5

    if draw_plot:
        plt.xlabel('score')
        plt.ylabel('number of reviews')

        distribution, xticklables = calculate_distribution(y_pred)
        bar_range = np.array(range(len(distribution)))
        plt.bar(bar_range-0.35/2, distribution, width=0.35, label='Real')

        distribution, xticklables = calculate_distribution(y_test)
        bar_range = np.array(range(len(distribution)))
        plt.bar(bar_range+0.35/2, distribution, width=0.35, label='Prediction')

        plt.xticks(bar_range, xticklables)
        plt.legend()

        plt.savefig('images/rf_num=%g.jpg' % num)
        plt.clf()

    return mean_average_error_train, mean_average_error_test


def main():
    number = range(1, 101, 10)

    mean_average_error_train = np.zeros_like(number, np.float)
    mean_average_error_test = np.zeros_like(number, np.float)

    for i in range(len(number)):
        mean_average_error_train[i], mean_average_error_test[i] = \
            run_model(number[i])

    plt.xlabel('num of trees')
    plt.ylabel('mean_average_error')
    plt.plot(number, mean_average_error_train, label='train')
    plt.plot(number, mean_average_error_test, label='test')
    plt.legend()
    plt.savefig('images/rf_num_vs_mean_average_error.jpg')
    plt.clf()

    run_model(draw_plot=True)


if __name__ == "__main__":
    main()
