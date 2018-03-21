from __future__ import division

try:
    import cPickle as pickle
except ImportError:
    import pickle

import io
import os
import random
from collections import Counter, defaultdict

from base.document import Document


def get_documents(data_dir, shuffle=False):
    """
    Extract documents from *.txt files in a given directory
    :param data_dir: path to the directory with .txt files
    :param shuffle: flag whether to return the documents
    in a shuffled vs sorted order

    :return: generator or a list of Document objects
    """
    files = list({filename[:-4] for filename in os.listdir(data_dir)})
    files.sort()
    if shuffle:
        random.shuffle(files)

    generator = (Document(doc_id, os.path.join(data_dir, f + '.txt'))
                 for doc_id, f in enumerate(files))
    return list(generator)


def get_all_answers(data_dir, filtered_by=None):
    """
    Extract ground truth answers from *.lab files in a given directory
    :param data_dir: path to the directory with .lab files
    :param filtered_by: whether to filter the answers.

    :return: dictionary of the form e.g. {'101231': set('lab1', 'lab2') etc.}
    """
    answers = dict()

    files = {filename[:-4] for filename in os.listdir(data_dir)}
    for f in files:
        answers[f] = get_answers_for_doc(f + '.txt',
                                         data_dir,
                                         filtered_by=filtered_by)

    return answers


def get_answers_for_doc(doc_name, data_dir, filtered_by=None):
    """
    Read ground_truth answers from a .lab file corresponding to the doc_name
    :param doc_name: the name of the document, should end with .txt
    :param data_dir: directory in which the documents and answer files are
    :param filtered_by: whether to filter the answers.

    :return: set of unicodes containing answers for this particular document
    """
    filename = os.path.join(data_dir, doc_name[:-4] + '.lab')

    if not os.path.exists(filename):
        raise ValueError("Answer file " + filename + " does not exist")

    with io.open(filename, 'r') as f:
        answers = {line.rstrip('\n') for line in f}

    if filtered_by:
        answers = {kw for kw in answers if kw in filtered_by}

    return answers


def calculate_label_distribution(data_dir, filtered_by=None):
    """
    Calculate the distribution of labels in a directory. Function can be used
    to find the most frequent and not used labels, so that the target
    vocabulary can be trimmed accordingly.
    :param data_dir: directory path with the .lab files
    :param filtered_by: a set of labels that defines the vocabulary

    :return: list of KV pairs of the form (14, ['lab1', 'lab2']), which means
             that both lab1 and lab2 were labels in 14 documents
    """
    answers = [kw for v in get_all_answers(data_dir, filtered_by=filtered_by).values()
               for kw in v]
    counts = Counter(answers)

    histogram = defaultdict(list)
    for kw, cnt in counts.items():
        histogram[cnt].append(kw)

    return histogram
