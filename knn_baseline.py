import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import lib

x, y, names = lib.get_data()
folds = 5
kf = KFold(folds)


def average(sum):
    return sum / folds


def calculate_distribution(y):
    dis = np.zeros((11), np.int)

    for value in y:
        dis[value - 1] = dis[value - 1] + 1

    return dis


def run(neighbours):
    knn_neighbours = KNeighborsClassifier(n_neighbors=neighbours)

    mean_absolute_error_train = 0
    accuracy_train = 0
    mean_absolute_error_test = 0
    accuracy_test = 0
    for train_idx, test_idx in kf.split(x):
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]

        knn_neighbours.fit(x_train, y_train)

        y_pred = knn_neighbours.predict(x_train)
        print(y_pred)
        mean_absolute_error_train = mean_absolute_error_train + mean_absolute_error(y_pred, y_train)
        accuracy_train = accuracy_train + accuracy_score(y_pred, y_train)

        y_pred = knn_neighbours.predict(x_test)
        print(y_pred)

        mean_absolute_error_test = mean_absolute_error_test + mean_absolute_error(y_pred, y_test)
        accuracy_test = accuracy_test + accuracy_score(y_pred, y_test)

    accuracy_train = average(accuracy_train)
    accuracy_test = average(accuracy_test)
    mean_absolute_error_test= average(mean_absolute_error_test)
    mean_absolute_error_test= average(mean_absolute_error_train)

    print(f'Training accuracy={accuracy_train} \n Test Accuracy={accuracy_test}')
    print(f'Mean Square Error Training={mean_absolute_error_train} \n Mean Square Error Test={mean_absolute_error_test}')


    plt.xlabel('score')
    plt.ylabel('Number of reviews')
    
    distribution = calculate_distribution(y_pred)
    bar_range = np.array(range(len(distribution)))
    plt.bar(bar_range - 0.35 / 2, distribution, width=0.35,
            label='train', tick_label=range(len(distribution)))
    
    distribution = calculate_distribution(y_test)
    plt.bar(bar_range + 0.35 / 2, distribution, width=0.35,
            label='test', tick_label=range(len(distribution)))
    
    plt.legend()
    plt.savefig(f'knn_{neighbours}.jpg')
    plt.clf()

    print(accuracy_train, accuracy_test)
    return accuracy_train, accuracy_test


def main():
    neighbours = [10, 20, 30, 50, 70]
    accuracy_train = [0.0]*len(neighbours)
    accuracy_test = [0.0]*len(neighbours)
    
    for i in range(len(neighbours)):
        accuracy_train[i], accuracy_test[i] = run(
            neighbours[i])

    plt.xlabel('Neighbours')
    plt.ylabel('Accuracy')
    plt.plot(neighbours, accuracy_train, label='train')
    plt.plot(neighbours, accuracy_test, label='test')
    plt.legend()
    plt.savefig('knn_vs_accuracy.jpg')
    plt.clf()


if __name__ == "__main__":
    main()
