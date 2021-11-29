import numpy as np
import matplotlib.pyplot as plt
from lib.loader import get_data
from lib.utils.common import calculate_distribution
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.svm import LinearSVC

x, y, names = get_data()
kFold = KFold()


def run_model(c, draw_plot=False):
    alpha = 1 / (2 * c)
    model = LinearSVC(alpha=alpha)

    mean_average_error_train = 0
    mean_average_error_test = 0
    accuracy_train = 0
    accuracy_test = 0

    for train_idx, test_idx in kFold.split(x):
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]

        model.fit(x_train, y_train)

        y_pred = model.predict(x_train)
        mean_average_error_train = mean_average_error_train + mean_absolute_error(y_pred, y_train)
        accuracy_train = accuracy_train + accuracy_score(y_pred, y_train)

        y_pred = model.predict(x_test)
        mean_average_error_test = mean_average_error_test + mean_absolute_error(y_pred, y_test)
        accuracy_test = accuracy_test + accuracy_score(y_pred, y_test)

    mean_average_error_train = mean_average_error_train / 5
    mean_accuracy_train = accuracy_train / 5

    mean_average_error_test = mean_average_error_test / 5
    mean_accuracy_test = accuracy_test / 5

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
        plt.savefig('images/svm_c=%g.jpg' % alpha)
        plt.clf()

    return mean_average_error_train, mean_accuracy_train, mean_average_error_test, mean_accuracy_test


def main():
    c_range = [0.001, 0.003, 0.005, 0.0008, 0.01, 0.03, 0.05, 0.08, 0.1, 0.3, 0.5, 0.8, 1, 3, 5, 8, 10, 30, 50, 80, 100]
    for c in c_range:
        run_model(c, True)

    mean_average_error_train = np.zeros_like(c_range)
    mean_average_error_test = np.zeros_like(c_range)
    accuracy_train = np.zeros_like(c_range)
    accuracy_test = np.zeros_like(c_range)

    for i in range(len(c_range)):
        mean_average_error_train[i], accuracy_train[i], mean_average_error_test[i], accuracy_test[i] = \
            run_model(c_range[i])

    plt.xlabel('c')
    plt.ylabel('mean average error')
    plt.plot(c_range, mean_average_error_train, label='train')
    plt.plot(c_range, mean_average_error_test, label='test')
    plt.legend()
    plt.savefig('images/svm_vs_c_mean_average_error.jpg')
    plt.clf()

    plt.xlabel('c')
    plt.ylabel('acc')
    plt.plot(c_range, accuracy_train, label='train')
    plt.plot(c_range, accuracy_test, label='test')
    plt.legend()
    plt.savefig('images/svm_c_vs_acc.jpg')
    plt.clf()


if __name__ == "__main__":
    main()
