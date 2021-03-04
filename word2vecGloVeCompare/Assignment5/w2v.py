#!/usr/bin/python3

import os
import json
import argparse
import string
import gensim
import time
import shutil
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
UNSUP_DIR = os.path.join(DATA_DIR, "unsup")
W2V_MODEL_FILE = os.path.join(os.path.dirname(__file__), "w2v_embedding.bin")
TRAIN_CORPUS_FILE = os.path.join(os.path.dirname(__file__), "train_corpus.bin")
TEST_CORPUS_FILE = os.path.join(os.path.dirname(__file__), "test_corpus.bin")

N_ITEMS = 25000
N_FEATURES = 300
MIN_WORD_CNT = 40
DOWNSAMPLING = 1E-3
WIN_SZ = 10
N_WRKS = 4

def load_data():
    if os.path.isdir(UNSUP_DIR):
        shutil.rmtree(UNSUP_DIR)
    data = {}
    data["train"] = {}
    load_train = load_files(TRAIN_DIR, load_content=True, encoding="utf-8")
    data["train"]["data"], data["train"]["target"] = load_train.data, load_train.target

    data["test"] = {}
    load_test = load_files(TEST_DIR, load_content=True, encoding="utf-8")
    test_data, test_target = load_test.data, load_test.target
    data["test"]["data"], data["test"]["target"] = load_test.data, load_test.target

    print("========= DATA LOADED =========")
    return data

def prepare_corpus(train, data_type):
    corpus = []
    for review in train["data"][:N_ITEMS]:
        tokens = word_tokenize(review)
        tokens = [w.lower() for w in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        words = [w for w in words if not w in stop_words]
        corpus.append(words)

    if data_type == "train":
        with open(TRAIN_CORPUS_FILE, "w") as f:
            json.dump(corpus, f)
    elif data_type == "test":
        with open(TEST_CORPUS_FILE, "w") as f:
            json.dump(corpus, f)

    print("========= CORPUS CREATED =========")
    return corpus


def train_w2v_model(corpus):
    model = gensim.models.Word2Vec(sentences=corpus, size=N_FEATURES,
                                   window=WIN_SZ, workers=N_WRKS,
                                   min_count=MIN_WORD_CNT,
                                   sample=DOWNSAMPLING)
    vocab = list(model.wv.vocab)

    model.save(W2V_MODEL_FILE)

    print("========= EMBEDDING SAVED =========")
    print("\tw2v model file : {}".format(W2V_MODEL_FILE))
    print("\tvocab size : {}".format(len(model.wv.vocab)))
    return model


def load_w2v_model_file(filename):
    print("========= LOADING W2V PRETRAINED MODEL =========")
    model = gensim.models.Word2Vec.load(W2V_MODEL_FILE)
    print("\tvocab size : {}".format(len(model.wv.vocab)))
    return model


def vec_from_review(review_words, model):
    feature_vec = np.zeros((N_FEATURES), dtype="float32")
    word_cnt = 0

    idx2word_set = set(model.wv.index2word)
    for word in review_words:
        if word in idx2word_set:
            word_cnt += 1
            feature_vec += model.wv[word]

    feature_vec /= word_cnt
    return feature_vec


def create_review_vectors(corpus, w2v_model):
    idx = 0
    feature_vec = np.zeros((len(corpus), N_FEATURES), dtype="float32")

    for review in corpus:
        feature_vec[idx] = vec_from_review(review, model)
        idx = idx + 1

    return feature_vec


def get_vectorized_data(train_corpus, test_corpus, w2v_model):
    print("========= CONVERTING DATA INTO VECTOR =========")
    train_vec = create_review_vectors(train_corpus, w2v_model)
    test_vec = create_review_vectors(test_corpus, w2v_model)

    if "numpy.ndarray" not in str(type(train_vec)):
        train_vec = train_vec.toarray()
        test_vec = test_vec.toarray()

    return train_vec, test_vec


@ignore_warnings(category=ConvergenceWarning)
def exec_logistic_regression(train_vec, test_vec, data):
    print("========= LOGISTIC REGRESSION =========")
    logisticReg = LogisticRegression()
    logisticReg.fit(train_vec, data["train"]["target"][:N_ITEMS])
    predicts = logisticReg.predict(test_vec)
    score = logisticReg.score(test_vec, data["test"]["target"][:N_ITEMS])
    print("Accuracy : {}".format(score))
    cm = metrics.confusion_matrix(data["test"]["target"][:N_ITEMS], predicts)
    print("Confusion Matrix : \n{}".format(cm))


@ignore_warnings(category=ConvergenceWarning)
def exec_deep_neural_net(train_vec, test_vec, data):
    print("========= DENSE NEURAL NETWORK =========")
    mlp = MLPClassifier(hidden_layer_sizes=(5, 2))
    mlp.fit(train_vec, data["train"]["target"][:N_ITEMS])
    predicts = mlp.predict(test_vec)
    score = mlp.score(test_vec, data["test"]["target"][:N_ITEMS])
    print("Accuracy : {}".format(score))
    cm = metrics.confusion_matrix(data["test"]["target"][:N_ITEMS], predicts)
    print("Confusion Matrix : \n{}".format(cm))

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w2v_trained", action="store_true", help="To create embeddings from corpus")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argument_parser()
    data = load_data()

    train_corpus = test_corpus = None
    if args.w2v_trained:
        with open(TRAIN_CORPUS_FILE, "r") as f:
            train_corpus = json.load(f)

        with open(TEST_CORPUS_FILE, "r") as f:
            test_corpus = json.load(f)

        model = load_w2v_model_file(W2V_MODEL_FILE)
    else:
        train_corpus = prepare_corpus(data["train"], "train")
        test_corpus = prepare_corpus(data["test"], "test")
        model = train_w2v_model(train_corpus)

    train_vec, test_vec = get_vectorized_data(train_corpus, test_corpus, model)

    exec_logistic_regression(train_vec, test_vec, data)
    exec_deep_neural_net(train_vec, test_vec, data)
