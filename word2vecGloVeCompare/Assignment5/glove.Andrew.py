#!/usr/bin/python3

import os
import json
import argparse
import string
import gensim
import time
import shutil
from nltk import corpus
import numpy as np
import sys
import hashlib

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

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
GLOVE_TXT_FILE = "glove.6B.300d.txt"
GLOVE_MODEL_FILE = os.path.join(os.path.dirname(__file__), "glove_embedding.bin")
TRAIN_CORPUS_FILE = os.path.join(os.path.dirname(__file__), "train_corpus.bin")
TEST_CORPUS_FILE = os.path.join(os.path.dirname(__file__), "test_corpus.bin")

N_ITEMS = 25000
N_FEATURES = 300
MIN_WORD_CNT = 40
DOWNSAMPLING = 1E-3
WIN_SZ = 10
N_WRKS = 4

# Pre-computed glove files values.
pretrain_num_lines = {"glove.840B.300d.txt": 2196017, "glove.42B.300d.txt":1917494}

pretrain_checksum = {
"glove.6B.300d.txt":"b78f53fb56ec1ce9edc367d2e6186ba4",
"glove.twitter.27B.50d.txt":"6e8369db39aa3ea5f7cf06c1f3745b06",
"glove.42B.300d.txt":"01fcdb413b93691a7a26180525a12d6e",
"glove.6B.50d.txt":"0fac3659c38a4c0e9432fe603de60b12",
"glove.6B.100d.txt":"dd7f3ad906768166883176d69cc028de",
"glove.twitter.27B.25d.txt":"f38598c6654cba5e6d0cef9bb833bdb1",
"glove.6B.200d.txt":"49fa83e4a287c42c6921f296a458eb80",
"glove.840B.300d.txt":"eec7d467bccfa914726b51aac484d43a",
"glove.twitter.27B.100d.txt":"ccbdddec6b9610196dd2e187635fee63",
"glove.twitter.27B.200d.txt":"e44cdc3e10806b5137055eeb08850569",
}

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

def build_gensim_glove_file():
    # Input: GloVe Model File
    # More models can be downloaded from http://nlp.stanford.edu/projects/glove/
    glove_file=os.path.join(DATA_DIR, GLOVE_TXT_FILE)
    _, tokens, dimensions, _ = GLOVE_TXT_FILE.split('.')
    num_lines = check_num_lines_in_glove(GLOVE_TXT_FILE)
    dims = int(dimensions[:-1])

    # Output: Gensim Model text format.
    gensim_file=GLOVE_MODEL_FILE
    gensim_first_line = "{} {}".format(num_lines, dims)

    # Prepends the line.
    if sys.platform == "linux" or sys.platform == "linux2":
        prepend_line(glove_file, gensim_file, gensim_first_line)
    else:
        prepend_slow(glove_file, gensim_file, gensim_first_line)
    return gensim_file

def train_glove_model(corpus):
    # model = gensim.models.Word2Vec(sentences=corpus, size=N_FEATURES,
    #                                window=WIN_SZ, workers=N_WRKS,
    #                                min_count=MIN_WORD_CNT,
    #                                sample=DOWNSAMPLING)
    # vocab = list(model.wv.vocab)

    model = gensim.models.KeyedVectors.load_word2vec_format(build_gensim_glove_file(),binary=False)
    #model.save(W2V_MODEL_FILE)
    print("========= EMBEDDING SAVED =========")
    print("\tglove model file : {}".format(GLOVE_MODEL_FILE))
    print("\tvocab size : {}".format(len(model.wv.vocab)))
    return model

def load_glove_model_file(filename):
    print("========= LOADING W2V PRETRAINED MODEL =========")
    model = gensim.models.KeyedVectors.load_word2vec_format(GLOVE_MODEL_FILE,binary=False)
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
    parser.add_argument("--corpus_trained", action="store_true", help="To create embeddings from corpus")
    parser.add_argument("--glove_trained", action="store_true", help="To create embeddings from corpus")
    args = parser.parse_args()
    return args

def prepend_line(infile, outfile, line):
	""" 
	Function use to prepend lines using bash utilities in Linux. 
	(source: http://stackoverflow.com/a/10850588/610569)
	"""
	with open(infile, 'r', encoding='utf-8') as old:
		with open(outfile, 'w', encoding='utf-8') as new:
			new.write(str(line) + "\n")
			shutil.copyfileobj(old, new)

def prepend_slow(infile, outfile, line):
	"""
	Slower way to prepend the line by re-creating the inputfile.
	"""
	with open(infile, 'r', encoding='utf-8') as fin:
		with open(outfile, 'w', encoding='utf-8') as fout:
			fout.write(line + "\n")
			for line in fin:
				fout.write(line)

def checksum(filename):
	"""
	This is to verify the file checksum is the same as the glove files we use to
	pre-computed the no. of lines in the glove file(s).
	"""
	BLOCKSIZE = 65536
	hasher = hashlib.md5()
	with open(filename, 'rb') as afile:
		buf = afile.read(BLOCKSIZE)
		while len(buf) > 0:
			hasher.update(buf)
			buf = afile.read(BLOCKSIZE)
	return hasher.hexdigest()

def check_num_lines_in_glove(filename, check_checksum=False):
    if check_checksum:
        assert checksum(filename) == pretrain_checksum[filename]
    if filename.startswith('glove.6B.'):
        return 400000
    elif filename.startswith('glove.twitter.27B.'):
        return 1193514
    else:
        return pretrain_num_lines[filename]



# Demo: Loads the newly created glove_model.txt into gensim API.
#model=gensim.models.Word2Vec.load_word2vec_format(build_gensim_glove_file(),binary=False) #GloVe Model

#print (model.most_similar(positive=['australia'], topn=10))
#print (model.similarity('woman', 'man'))

if __name__ == "__main__":
    args = argument_parser()
    data = load_data()

    train_corpus = test_corpus = None
    if args.corpus_trained:
        with open(TRAIN_CORPUS_FILE, "r") as f:
            train_corpus = json.load(f)

        with open(TEST_CORPUS_FILE, "r") as f:
            test_corpus = json.load(f)
    else:
        train_corpus = prepare_corpus(data["train"], "train")
        test_corpus = prepare_corpus(data["test"], "test")

    if args.glove_trained:
        model = load_glove_model_file(GLOVE_MODEL_FILE)
    else:
        model = train_glove_model(train_corpus)

    train_vec, test_vec = get_vectorized_data(train_corpus, test_corpus, model)

    exec_logistic_regression(train_vec, test_vec, data)
    exec_deep_neural_net(train_vec, test_vec, data)
