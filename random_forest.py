import numpy as np
import matplotlib.pyplot as plt
from lib.loader import get_data
from lib.utils.common import calculate_distribution
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_absolute_error

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


def run_model(num, draw_plot=False):
    model = RandomForestClassifier(n_estimators=num, n_jobs=-1)

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
        plt.savefig('images/rf_num=%g.jpg' % num)
        plt.clf()

    return mean_average_error_train, mean_accuracy_train, mean_average_error_test, mean_accuracy_test


def main():
    run_model(10, True)
    run_model(20, True)
    run_model(50, True)
    run_model(100, True)

    number = range(1, 101, 10)

    mean_average_error_train = np.zeros_like(number, np.float)
    mean_average_error_test = np.zeros_like(number, np.float)
    accuracy_train = np.zeros_like(number, np.float)
    accuracy_test = np.zeros_like(number, np.float)

    for i in range(len(number)):
        mean_average_error_train[i], accuracy_train[i], mean_average_error_test[i], accuracy_test[i] = \
            run_model(number[i])

    plt.xlabel('num of trees')
    plt.ylabel('mean_average_error')
    plt.plot(number, mean_average_error_train, label='train')
    plt.plot(number, mean_average_error_test, label='test')
    plt.legend()
    plt.savefig('images/rf_num_vs_mean_average_error.jpg')
    plt.clf()

    plt.xlabel('num of trees')
    plt.ylabel('acc')
    plt.plot(number, accuracy_train, label='train')
    plt.plot(number, accuracy_test, label='test')
    plt.legend()
    plt.savefig('images/rf_num_vs_acc.jpg')
    plt.clf()


if __name__ == "__main__":
    main()
