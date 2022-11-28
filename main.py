import math
import string
from math import e
from typing import List
import os
import json
import nltk
import re

import numpy as np
from nltk.lm import Vocabulary
from sklearn.naive_bayes import MultinomialNB

from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

data_dir = '../../data'
train_dir = f'{data_dir}/aclImdb/train'
test_dir = f'{data_dir}/aclImdb/test'
neg_train_dir = f'{train_dir}/neg'
pos_train_dir = f'{train_dir}/pos'

pos_test_dir = f'{test_dir}/pos'
neg_test_dir = f'{test_dir}/neg'

regex = re.compile('[^a-zA-Z]')


def get_rating(fp: str) -> int:
    try:
        return int(fp.split('_')[1].split('.')[0])
    except (ValueError, IndexError) as e:
        raise Exception(f"Couldn't extract rating from filepath: '{fp}'") from e


def load_reviews(fps: List[str], dir_: str, max_n: int = -1) -> List[str]:
    fps_cut = fps
    if max_n != -1:
        fps_cut = fps[:max_n]
    reviews = []
    for fp in fps_cut:
        with open(dir_ + '/' + fp, 'r') as f:
            reviews.append(f.read())

    return reviews


def try_make_float(x):
    try:
        return float(x)
    except ValueError:
        return 0.


def sigmoid(x):
    return (20 / (1 + e ** (-0.05 * x))) - 10


def m(x):
    y = 2
    return -((y * x) - (y / 2)) ** 2 + 1


def clean(x):
    return regex.sub(' ', x).lower()


ps = nltk.PorterStemmer()
sw = set(stopwords.words())
html_tag = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')


def make_tfidf(reviews, vocab):
    # Build TF matrix
    tf = np.ndarray(shape=(len(reviews), len(vocab)))
    idf = []
    for i, review in enumerate(reviews):
        clean_review = re.sub(html_tag, '', review)
        tokens = nltk.word_tokenize(clean_review)
        stems = [ps.stem(token, to_lowercase=True) for token in tokens if token not in sw]

        tf[i] = [stems.count(stem) for stem in sorted(vocab)]

    n_docs = len(reviews)
    # Calculate Inverse Document Frequencies
    for i, word in enumerate(sorted(vocab)):
        if word not in vocab or word == '<UNK>':
            idf.append(0)
            continue
        docs_with_word = 0
        for row in tf:
            if row[i] > 0:
                docs_with_word += 1

        if docs_with_word == 0:
            idf.append(0)
            continue

        x = n_docs / docs_with_word
        assert x >= 1, word

        idf.append(math.log(x))
    return tf * idf


def main():
    print("Finding reviews...")
    neg_fps = [fp for fp in os.listdir(neg_train_dir) if fp.endswith('.txt')]
    pos_fps = [fp for fp in os.listdir(pos_train_dir) if fp.endswith('.txt')]
    test_pos_fps = [fp for fp in os.listdir(pos_test_dir) if fp.endswith('.txt')]
    test_neg_fps = [fp for fp in os.listdir(neg_test_dir) if fp.endswith('.txt')]

    print("Loading reviews...")
    pos_reviews = load_reviews(pos_fps, pos_train_dir, 1000)
    neg_reviews = load_reviews(neg_fps, neg_train_dir, 1000)
    test_pos_reviews = load_reviews(test_pos_fps, pos_test_dir, 100)
    test_neg_reviews = load_reviews(test_neg_fps, neg_test_dir, 100)
    # print(f"{test_pos_reviews[0]=}")

    print("Extracting ratings...")
    pos_ratings = [get_rating(fp) for fp in pos_fps][:len(pos_reviews)]
    neg_ratings = [get_rating(fp) for fp in neg_fps][:len(neg_reviews)]
    test_pos_ratings = [get_rating(fp) for fp in test_pos_fps][:len(test_pos_reviews)]
    test_neg_ratings = [get_rating(fp) for fp in test_neg_fps][:len(test_neg_reviews)]

    reviews = pos_reviews + neg_reviews
    test_reviews = test_pos_reviews + test_neg_reviews

    ratings = pos_ratings + neg_ratings
    test_ratings = test_pos_ratings + test_neg_ratings

    print("Collecting vocab...")
    # https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
    all_words = []
    for i, review in enumerate(reviews):
        # Remove HTML tags
        clean_review = re.sub(html_tag, '', review)
        tokens = nltk.word_tokenize(clean_review)
        stems = [ps.stem(token, to_lowercase=True) for token in tokens if token not in sw]

        # remove all punctuation
        stems = [x for x in ["".join(c for c in s if c not in string.punctuation) for s in stems] if x]
        all_words += stems

    vocab = Vocabulary(all_words, unk_cutoff=5)

    print("Making TFIDF for training data...")
    tfidf = make_tfidf(reviews, vocab)
    print("Making TFIDF for testing data...")
    tfidf_test = make_tfidf(test_reviews, vocab)
    mnb = MultinomialNB()
    sentiment = ["pos" if rating > 5 else "neg" for rating in ratings]

    print("Fitting model...")
    model = mnb.fit(tfidf, sentiment)

    print("Testing model...")
    results = mnb.predict(tfidf_test)

    n_correct = 0
    for i, rating in enumerate(test_ratings):
        actual = "pos" if rating > 5 else "neg"
        predict = results[i]
        if actual == predict:
            n_correct += 1

    print(f"{n_correct} correct out of {len(results)}. ({n_correct/len(results)})")


if __name__ == "__main__":
    main()
