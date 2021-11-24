import os
import pandas as pd
import pickle
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector


def load_dataset():
    data_file_path = '../data/reviews.csv'
    data_absolute_path = os.path.join(
        os.path.dirname(__file__), data_file_path)

    return pd.read_csv(data_absolute_path, sep='\t', header=0)


def remove_non_english_reviews(dataset):
    nlp = spacy.load("en_core_web_lg")
    Language.factory("language_detector", func=lambda nlp,
                     name: LanguageDetector())
    nlp.add_pipe('language_detector', last=True)

    def english_filer(entry):
        nlp_document = nlp(entry.review)
        detected_language = nlp_document._.language

        return detected_language['language'] == 'en' and detected_language['score'] > 0.99

    return dataset[dataset.apply(english_filer, axis=1)]


def preprocess_data():
    data_file_path = '../data/dataset.ml'
    data_absolute_path = os.path.join(
        os.path.dirname(__file__), data_file_path)

    dataset = load_dataset()
    print(dataset)
    # english_dataset = remove_non_english_reviews(dataset)
    #
    # data_file = open(data_absolute_path, 'wb')
    # pickle.dump(english_dataset, data_file)


if __name__ == "__main__":
    preprocess_data()
