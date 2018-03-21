from __future__ import print_function, unicode_literals
import os
import six
import numpy as np
from functools import reduce

from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler

from base.document import Document
from config import EMBEDDING_SIZE, WORD2VEC_WORKERS, MIN_WORD_COUNT, \
    WORD2VEC_CONTEXT
# from utils import get_documents, save_to_disk


def train_word2vec(doc_directory, vec_dim=EMBEDDING_SIZE):
    """
    Train the Word2Vec object iteratively, loading stuff to memory one by one.
    :param doc_directory: directory with the documents
    :param vec_dim: the dimensionality of the vector that's being built

    :return: Word2Vec object
    """
    class SentenceIterator(object):
        def __init__(self, dirname):
            self.dirname = dirname

        def __iter__(self):
            files = {filename[:-4] for filename in os.listdir(self.dirname)}
            for doc_id, fname in enumerate(files):
                d = Document(doc_id, os.path.join(
                    self.dirname, fname + '.txt'))
                for sentence in d.read_sentences():
                    yield sentence

    # Initialize and train the model
    model = Word2Vec(
        SentenceIterator(doc_directory),
        workers=WORD2VEC_WORKERS,
        size=vec_dim,
        min_count=MIN_WORD_COUNT,
        window=WORD2VEC_CONTEXT,
    )

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    return model
