r"""
Corpora and Vector Spaces
=========================

Demonstrates transforming text into a vector space representation.

Also introduces corpus streaming and persistence to disk in various formats.
"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

###############################################################################

#
documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]

###############################################################################

from pprint import pprint  # pretty-printer
from collections import defaultdict

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in documents
]

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [
    [token for token in text if frequency[token] > 1]
    for text in texts
]

pprint(texts)

###############################################################################


from gensim import corpora
dictionary = corpora.Dictionary(texts)
dictionary.save('gensim/temp/deerwester.dict')  # store the dictionary, for future reference
print(dictionary)

###############################################################################
print(dictionary.token2id)

###############################################################################
# To actually convert tokenized documents to vectors:

new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)  # the word "interaction" does not appear in the dictionary and is ignored

###############################################################################
# The function :func:`doc2bow` simply counts the number of occurrences of
# each distinct word, converts the word to its integer word id
# and returns the result as a sparse vector. The sparse vector ``[(0, 1), (1, 1)]``
# therefore reads: in the document `"Human computer interaction"`, the words `computer`
# (id 0) and `human` (id 1) appear once; the other ten dictionary words appear (implicitly) zero times.

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('gensim/temp/deerwester.mm', corpus)  # store to disk, for later use
print(corpus)

###############################################################################
#
from smart_open import open  # for transparently opening remote files


class MyCorpus(object):
    def __iter__(self):
        for line in open('https://radimrehurek.com/gensim/mycorpus.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())

###############################################################################

corpus_memory_friendly = MyCorpus()  # doesn't load the corpus into memory!
print(corpus_memory_friendly)

###############################################################################

for vector in corpus_memory_friendly:  # load one vector into memory at a time
    print(vector)

###############################################################################

from six import iteritems
# collect statistics about all tokens
dictionary = corpora.Dictionary(line.lower().split() for line in open('https://radimrehurek.com/gensim/mycorpus.txt'))
# remove stop words and words that appear only once
stop_ids = [
    dictionary.token2id[stopword]
    for stopword in stoplist
    if stopword in dictionary.token2id
]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed
print(dictionary)

###############################################################################

corpus = [[(1, 0.5)], []]  # make one document empty, for the heck of it

corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus)

###############################################################################


corpora.SvmLightCorpus.serialize('/tmp/corpus.svmlight', corpus)
corpora.BleiCorpus.serialize('/tmp/corpus.lda-c', corpus)
corpora.LowCorpus.serialize('/tmp/corpus.low', corpus)


###############################################################################

corpus = corpora.MmCorpus('/tmp/corpus.mm')

###############################################################################

print(corpus)

###############################################################################
print(list(corpus))  # calling list() will convert any sequence to a plain Python list

###############################################################################
for doc in corpus:
    print(doc)

###############################################################################

corpora.BleiCorpus.serialize('/tmp/corpus.lda-c', corpus)

###############################################################################

import gensim
import numpy as np
numpy_matrix = np.random.randint(10, size=[5, 2])  # random matrix as an example
corpus = gensim.matutils.Dense2Corpus(numpy_matrix)
# numpy_matrix = gensim.matutils.corpus2dense(corpus, num_terms=number_of_corpus_features)

###############################################################################

import scipy.sparse
scipy_sparse_matrix = scipy.sparse.random(5, 2)  # random sparse matrix as example
corpus = gensim.matutils.Sparse2Corpus(scipy_sparse_matrix)
scipy_csc_matrix = gensim.matutils.corpus2csc(corpus)

###############################################################################

