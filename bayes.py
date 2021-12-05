from math import pi
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import save
from lib.loader import get_data
from lib.utils.common import calculate_distribution
from sklearn.linear_model import BayesianRidge
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


def run_model(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6, draw_plot=False):
    model = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2,
                          lambda_1=lambda_1, lambda_2=lambda_2)

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

        plt.savefig('images/bayes.jpg')
        plt.clf()


    return mean_average_error_train, mean_average_error_test


def main():
    # alpha 1
    alpha_1 = np.linspace(0, 100)
    mean_average_error_train = np.zeros_like(alpha_1)
    mean_average_error_test = np.zeros_like(alpha_1)

    for i in range(len(alpha_1)):
        mean_average_error_train[i], mean_average_error_test[i] = \
            run_model(alpha_1=alpha_1[i])

    plt.xlabel('alpha_1')
    plt.ylabel('mean average error')
    plt.plot(alpha_1, mean_average_error_train, label='train')
    plt.plot(alpha_1, mean_average_error_test, label='test')
    plt.legend()
    plt.savefig('images/bayes_alpha_1_vs_mean_average_error.jpg')
    plt.clf()

    # alpha 2
    alpha_2 = np.linspace(0, 100)
    mean_average_error_train = np.zeros_like(alpha_2)
    mean_average_error_test = np.zeros_like(alpha_2)

    for i in range(len(alpha_2)):
        mean_average_error_train[i], mean_average_error_test[i] = \
            run_model(alpha_2=alpha_2[i])

    plt.xlabel('alpha_2')
    plt.ylabel('mean average error')
    plt.plot(alpha_2, mean_average_error_train, label='train')
    plt.plot(alpha_2, mean_average_error_test, label='test')
    plt.legend()
    plt.savefig('images/bayes_alpha_2_vs_mean_average_error.jpg')
    plt.clf()

    # lambda 1
    lambda_1 = np.linspace(0, 100)
    mean_average_error_train = np.zeros_like(lambda_1)
    mean_average_error_test = np.zeros_like(lambda_1)

    for i in range(len(lambda_1)):
        mean_average_error_train[i], mean_average_error_test[i] = \
            run_model(lambda_1[i])

    plt.xlabel('lambda_1')
    plt.ylabel('mean average error')
    plt.plot(lambda_1, mean_average_error_train, label='train')
    plt.plot(lambda_1, mean_average_error_test, label='test')
    plt.legend()
    plt.savefig('images/bayes_lambda_1_vs_mean_average_error.jpg')
    plt.clf()

    # lambda 2
    lambda_2 = np.linspace(0, 100)
    mean_average_error_train = np.zeros_like(lambda_2)
    mean_average_error_test = np.zeros_like(lambda_2)

    for i in range(len(lambda_2)):
        mean_average_error_train[i], mean_average_error_test[i] = \
            run_model(lambda_2[i])

    plt.xlabel('lambda_2')
    plt.ylabel('mean average error')
    plt.plot(lambda_2, mean_average_error_train, label='train')
    plt.plot(lambda_2, mean_average_error_test, label='test')
    plt.legend()
    plt.savefig('images/bayes_lambda_2_vs_mean_average_error.jpg')
    plt.clf()

    run_model(draw_plot=True)


if __name__ == "__main__":
    main()
