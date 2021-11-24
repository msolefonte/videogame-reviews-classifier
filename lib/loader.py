import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

def get_stop_words():
    stop_words_file_path = '../data/stop_words.txt'
    stop_words_absolute_path = os.path.join(
        os.path.dirname(__file__), stop_words_file_path)

    stop_words = []

    stop_words_file = open(stop_words_absolute_path, 'r')
    for line in stop_words_file:
        stop_words.append(line[:-2])

    return stop_words


def get_data():
    data_file_path = '../data/dataset.ml'
    data_absolute_path = os.path.join(
        os.path.dirname(__file__), data_file_path)
    data_file = open(data_absolute_path, 'rb')

    dataset = pickle.load(data_file)

    x = np.array(dataset.iloc[:, 1])
    y = np.array(dataset.iloc[:, 0])

    # TODO Stemming (?)
    vectorizer = CountVectorizer(stop_words=get_stop_words())
    x_train_counts = vectorizer.fit_transform(x)

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    return x_train_tfidf.toarray(), y, vectorizer.get_feature_names_out()


def get_train_test_split_data():
    x, y, names = get_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    return x_train, x_test, y_train, y_test, names

if __name__ == "__main__":
    x, y, name = get_data()
    print(x.shape)
    print(y.shape)
