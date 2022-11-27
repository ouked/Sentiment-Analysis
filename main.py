from math import e
from typing import List
import os
import json
import nltk
import re

nltk.download('punkt')

train_dir = 'data/aclImdb/train'
test_dir = 'data/aclImdb/test'
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


def load_reviews(fps: List[str], dir_: str) -> List[str]:
    reviews = []
    for fp in fps:
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


def make_er(train_data, ratings, use_cache=False):
    if use_cache:
        with open('results.txt', 'r') as f:
            return json.load(f)

    vocab_ratings = {}
    vocab_occurrences = {}
    porter = nltk.PorterStemmer()
    n_reviews = len(train_data)
    for i, rev in enumerate(train_data):
        words = nltk.word_tokenize(clean(rev))
        stems = [porter.stem(word) for word in words]
        n_words = len(words)
        # Convert (1 to 10) to (-5 to 4)
        rating = ratings[i] - 6
        rel_rating = rating / n_words
        for word in stems:
            vocab_ratings[word] = vocab_ratings.get(word, 0) + rel_rating
            vocab_occurrences[word] = vocab_occurrences.get(word, 0) + 1
        if i % 1000 == 0:
            print(f"{i}/{n_reviews}...")
    vocab_ratings = {word: sum_ / vocab_occurrences[word] for word, sum_ in vocab_ratings.items()}
    with open('results.txt', 'w') as f:
        f.write(json.dumps(vocab_ratings))

    return vocab_ratings


def main():
    n_reviews = 6000
    with open("data/aclImdb/imdb.vocab", 'r') as f:
        vocab = f.read().split('\n')

    vocab_index = {word: i for i, word in enumerate(vocab)}

    with open("data/aclImdb/imdbEr.txt", 'r') as f:
        expected_ratings = list(map(try_make_float, f.read().split('\n')))

    print("Finding reviews...")
    neg_fps = [fp for fp in os.listdir(neg_train_dir) if fp.endswith('.txt')]
    pos_fps = [fp for fp in os.listdir(pos_train_dir) if fp.endswith('.txt')]
    test_pos_fps = [fp for fp in os.listdir(pos_test_dir) if fp.endswith('.txt')]
    test_neg_fps = [fp for fp in os.listdir(neg_test_dir) if fp.endswith('.txt')]

    print("Loading reviews...")
    pos_reviews = load_reviews(pos_fps, pos_train_dir)[:n_reviews//2]
    neg_reviews = load_reviews(neg_fps, neg_train_dir)[:n_reviews//2]
    test_pos_reviews = load_reviews(test_pos_fps, pos_test_dir)
    test_neg_reviews = load_reviews(test_neg_fps, neg_test_dir)
    print(f"{test_pos_reviews[0]=}")

    print("Extracting ratings...")
    pos_ratings = [get_rating(fp) for fp in pos_fps][:n_reviews//2]
    neg_ratings = [get_rating(fp) for fp in neg_fps][:n_reviews//2]
    test_pos_ratings = [get_rating(fp) for fp in test_pos_fps]

    print("Generating ERs...")
    # expected_ratings_dict = make_er(pos_reviews + neg_reviews, pos_ratings + neg_ratings)
    expected_ratings_dict = make_er([], [], use_cache=True)

    def evaluate():
        porter = nltk.PorterStemmer()
        n_positive = 0
        n_reviews = len(test_neg_reviews)
        for i, rev in enumerate(test_neg_reviews):
            words = [word.lower() for word in nltk.word_tokenize(clean(rev)) if word not in '.,\'"']
            stems = [porter.stem(word) for word in words]
            sum_expected_rating = 0
            n_words = len(words)
            for j, word in enumerate(stems):
                expected_rating = expected_ratings_dict.get(word, 0)
                sum_expected_rating += expected_rating

            if sum_expected_rating > 0:
                # print(n_positive, i)
                n_positive += 1

            if i % 1000 == 0:
                print(f"{i}/{n_reviews}...")
                print(f"Accuracy: {n_positive}/{len(test_neg_reviews)} ({n_positive / len(test_neg_reviews)})...")

        print("Done evaluating.")
        print(f"Accuracy: {n_positive}/{len(test_neg_reviews)} ({n_positive / len(test_neg_reviews)})")

    print("Evaluating...")
    evaluate()


if __name__ == "__main__":
    main()
