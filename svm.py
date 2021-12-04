import numpy as np
import matplotlib.pyplot as plt
from lib.loader import get_data
from lib.utils.common import calculate_distribution
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

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


def run_model(c, draw_plot=False):
    model = SVR(C=c)

    mean_average_error_train = 0
    mean_average_error_test = 0

    for train_idx, test_idx in kFold.split(x):
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]

        model.fit(x_train, y_train)

        y_pred = model.predict(x_train)
        mean_average_error_train = mean_average_error_train + mean_absolute_error(y_pred, y_train)

        y_pred = model.predict(x_test)
        mean_average_error_test = mean_average_error_test + mean_absolute_error(y_pred, y_test)

    mean_average_error_train = mean_average_error_train / 5
    mean_average_error_test = mean_average_error_test / 5

    if draw_plot:
        plt.xlabel('score')
        plt.ylabel('count of score')

        distribution = calculate_distribution(y_pred)
        bar_range = np.array(range(len(distribution)))
        plt.bar(bar_range-0.35/2, distribution, width=0.35,
                label='train', tick_label=range(len(distribution)))

        distribution = calculate_distribution(y_test)
        plt.bar(bar_range+0.35/2, distribution, width=0.35,
                label='test', tick_label=range(len(distribution)))

        plt.legend()
        plt.savefig('images/svm_c=%g.jpg' % c)
        plt.clf()

    print(c, mean_average_error_train, mean_average_error_test)
    return mean_average_error_train, mean_average_error_test


def main():
    c_range = [0.001, 0.01, 0.03, 0.05, 0.08, 0.1, 0.3, 0.5, 0.8, 1]

    mean_average_error_train = np.zeros_like(c_range)
    mean_average_error_test = np.zeros_like(c_range)

    for i in range(len(c_range)):
        mean_average_error_train[i], mean_average_error_test[i] = run_model(c_range[i], True)

    plt.xlabel('c')
    plt.ylabel('mean average error')
    plt.plot(c_range, mean_average_error_train, label='train')
    plt.plot(c_range, mean_average_error_test, label='test')
    plt.legend()
    plt.savefig('images/svm_c_vs_mean_average_error.jpg')
    plt.clf()


if __name__ == "__main__":
    main()
