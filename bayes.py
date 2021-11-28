import lib
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_absolute_error

x, y, names = lib.get_data()
kf = KFold()


def calculate_distribution(y):
    dis = np.zeros((11), np.int)

    for value in y:
        dis[value - 1] = dis[value - 1] + 1

    return dis


def run(alpha, draw):
    model = MultinomialNB(alpha=alpha)

    mae_train = 0
    acc_train = 0
    mae_test = 0
    acc_test = 0
    for train_idx, test_idx in kf.split(x):
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]

        model.fit(x_train, y_train)

        y_pred = model.predict(x_train)
        mae_train = mae_train + mean_absolute_error(y_pred, y_train)
        acc_train = acc_train + accuracy_score(y_pred, y_train)

        y_pred = model.predict(x_test)
        mae_test = mae_test + mean_absolute_error(y_pred, y_test)
        acc_test = acc_test + accuracy_score(y_pred, y_test)

    mae_train = mae_train / 5
    acc_train = acc_train / 5

    mae_test = mae_test / 5
    acc_test = acc_test / 5

    if not draw:
        return mae_train, acc_train, mae_test, acc_test

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
    plt.savefig('bayes_alpha=%g.jpg' % alpha)
    plt.clf()

    return mae_train, acc_train, mae_test, acc_test


def main():
    run(0, True)
    run(0.02, True)
    run(0.5, True)
    run(1, True)
    run(10, True)
    run(100, True)

    alpha = np.linspace(0, 100)
    mae_train = np.zeros_like(alpha)
    acc_train = np.zeros_like(alpha)
    mae_test = np.zeros_like(alpha)
    acc_test = np.zeros_like(alpha)
    for i in range(len(alpha)):
        mae_train[i], acc_train[i], mae_test[i], acc_test[i] = run(
            alpha[i], False)

    plt.xlabel('alpha')
    plt.ylabel('mae')
    plt.plot(alpha, mae_train, label='train')
    plt.plot(alpha, mae_test, label='test')
    plt.legend()
    plt.savefig('bayes_alpha_vs_mae.jpg')
    plt.clf()

    plt.xlabel('alpha')
    plt.ylabel('acc')
    plt.plot(alpha, acc_train, label='train')
    plt.plot(alpha, acc_test, label='test')
    plt.legend()
    plt.savefig('bayes_alpha_vs_acc.jpg')
    plt.clf()


if __name__ == "__main__":
    main()
