from lib.loader import get_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_absolute_error

x, y, names = get_data()
kFold = KFold()


def run_model():
    model = LogisticRegression()

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

    return mean_average_error_train, mean_average_error_test


def main():
    mean_average_error_train, mean_average_error_test = run_model()

    print('mean_average_error_train:', mean_average_error_train)
    print('mean_average_error_test:', mean_average_error_test)


if __name__ == "__main__":
    main()
