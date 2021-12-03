import numpy as np
import pickle
import os
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split


def preprocess_word(word, stemmer):
    if word.isdigit():
        return 'NUM'
    return stemmer.stem(word)


def get_data():
    snowball_stemmer = SnowballStemmer('english')

    data_file_path = '../data/dataset.ml'
    data_absolute_path = os.path.join(
        os.path.dirname(__file__), data_file_path)
    data_file = open(data_absolute_path, 'rb')

    dataset = pickle.load(data_file)

    x = np.array(dataset.iloc[:, 1])
    y = np.array(dataset.iloc[:, 0])
    
    vectorizer = CountVectorizer()
    preprocessed_x = [' '.join([preprocess_word(word, snowball_stemmer) for word in review.split(' ')]) for review in x]
    x_train_counts = vectorizer.fit_transform(preprocessed_x)
    
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    
    return x_train_tfidf.toarray(), y, vectorizer.get_feature_names()

def get_data_for_tokenization():
    snowball_stemmer = SnowballStemmer('english')

    data_file_path = '../data/dataset.ml'
    data_absolute_path = os.path.join(
        os.path.dirname(__file__), data_file_path)
    data_file = open(data_absolute_path, 'rb')

    dataset = pickle.load(data_file)

    x = np.array(dataset.iloc[:, 1])
    y = np.array(dataset.iloc[:, 0])
    preprocessed_x = [' '.join([preprocess_word(word, snowball_stemmer) for word in review.split(' ')]) for review in x]
    
    return x,y


def get_train_test_split_data():
    x, y, names = get_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    return x_train, x_test, y_train, y_test, names


def main():
    x, y, name = get_data()
    print(x.shape)
    print(y.shape)
    # print(name)


if __name__ == "__main__":
    main()
