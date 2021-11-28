import lib
from sklearn.ensemble import RandomForestClassifier
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


def run(num, draw):
    model = RandomForestClassifier(n_estimators=num, n_jobs=-1)

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
    plt.savefig('rf_num=%g.jpg' % num)
    plt.clf()

    return mae


def main():
    run(10, True)
    run(20, True)
    run(50, True)
    run(100, True)

    number = range(1, 101, 10)

    mae = np.zeros_like(number)
    for i in range(len(number)):
        mae[i] = run(mae[i], False)

    plt.xlabel('number')
    plt.ylabel('mae')

    plt.plot(number, mae, label='mae')

    plt.legend()
    plt.savefig('rf_kfold.jpg')
    plt.clf()


if __name__ == "__main__":
    main()
