import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,  TfidfTransformer


def get_data():
    data_file_path = '../data/reviews.csv'
    data_absolute_path = os.path.join(os.path.dirname(__file__), data_file_path)

    df = pd.read_csv(data_absolute_path, sep='\t', header=0)

    # TODO Filter our non english reviews
    # https://towardsdatascience.com/4-python-libraries-to-detect-english-and-non-english-language-c82ad3efd430

    x = np.array(df.iloc[:, 1])
    y = np.array(df.iloc[:, 0])

    # TODO Use custom stop words. Maybe use that in the doc / compare
    # TODO Stemming (?)
    vectorizer = CountVectorizer(stop_words='english')
    x_train_counts = vectorizer.fit_transform(x)

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    return x_train_counts, vectorizer.get_feature_names_out(), y


get_data()
