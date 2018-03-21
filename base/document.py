from __future__ import print_function, unicode_literals

import io
import os
import nltk
import string

from deepcut.deepcut import tokenize

# from nltk.tokenize import WordPunctTokenizer, sent_tokenize, word_tokenize

# nltk.download('punkt', quiet=True)  # make sure it's downloaded before using


class Document(object):
    """ Class representing a document that the keywords are extracted from """

    def __init__(self, doc_id, filepath, tokenizer_model=None, text=None):
        self.doc_id = doc_id
        self.tokenizer_model = tokenizer_model
        if text:  # is a text to process
            self.text = text
            self.filename = None
            self.filepath = None
        else:  # is a path to a file
            if not os.path.exists(filepath):
                raise ValueError("The file " + filepath + " doesn't exist")

            self.filepath = filepath
            self.filename = os.path.basename(filepath)

            with io.open(filepath, 'r', encoding='utf-8') as f:
                print("read text!", filepath)
                self.text = f.read()
                # clean text might be needed here!

    def __str__(self):
        return self.text

    def get_all_words(self):
        """ Return all words tokenized by Tokenizer """
        # Tokenizer return [[word, word, word]] so that require to extract only idx 0
        print("self.text", self.text)
        return [w for w in self.tokenizer_model.predict(sentence=self.text)[0]]

    def read_sentences(self):
        """ Return all readable sentences """
        return self.text
