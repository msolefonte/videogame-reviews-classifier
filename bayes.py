import lib
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

x, y, names = lib.get_data()
kf = KFold()


def calculate_distribution(y):
    dis = np.zeros((11), np.int)

    for value in y:
        dis[value - 1] = dis[value - 1] + 1

    return dis


def run(alpha, draw):
    model = MultinomialNB(alpha=alpha)

    mae = 0
    for train_idx, test_idx in kf.split(x):
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        mae = mae + np.mean(np.abs(y_pred-y_test))

    mae = mae / 5
    print('mae: ', mae)

    if not draw:
        return mae

    plt.xlabel('score')
    plt.ylabel('count of score')

    distribution = calculate_distribution(y_pred)
    bar_range = np.array(range(len(distribution)))
    plt.bar(bar_range-0.35/2, distribution, width=0.35,
            label='prediction', tick_label=range(len(distribution)))

    distribution = calculate_distribution(y_test)
    plt.bar(bar_range+0.35/2, distribution, width=0.35,
            label='validation', tick_label=range(len(distribution)))

    plt.legend()
    plt.savefig('bayes_alpha=%g.jpg' % alpha)
    plt.clf()

    return mae


def main():
    run(0, True)
    run(0.02, True)
    run(0.5, True)
    run(1, True)

    alpha = np.linspace(0, 1)
    mae = np.zeros_like(alpha)
    for i in range(len(alpha)):
        mae[i] = run(alpha[i], False)

    plt.xlabel('alpha')
    plt.ylabel('mae')

    plt.plot(alpha, mae, label='mae')

    plt.legend()
    plt.savefig('bayes_kfold.jpg')
    plt.clf()


if __name__ == "__main__":
    main()
