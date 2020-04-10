'''
A word embedding is a dense vector representation. Each word is represented by a point in the
embedding space and they are learnt and moved around in the vector space based on the context of
word around it. Simply, WE is allows words with similar meaning to be clustered together.

1. Gensim: Open Source Library in NLP for topic modeling. gensim provides a class Word2Vec to work with the below.
2. word2vec: Algorithm for learning Embeddings from a corpus of text. In a general sense these algorithms
look at fixed window of words for each target thus finding the context aka meaning it carries with itself.
Word2vec requires a large amount of text.
    a. CBOW: Continuous Bag of Words
    b. skip n-grams:

    Parameters for the constructor of class Word2Vec
        1. size: no. of dimensions
        2. window:
        3. min_count: least frequency to be considered
        4. workers: count of threads to train the model
        5. sg: Algo choice 0 -> CBOW | 1 -> skip grams

3. Similarity is determined by the cosine distance between two word vectors.
'''

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from matplotlib import pyplot

# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]

# train model
model = Word2Vec(sentences, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['sentence'])
# save model
model.wv.save_word2vec_format('R&D/model.txt', binary=False)
model.wv.save_word2vec_format('R&D/model.bin') # both behave similar
model.save('R&D/model.bin')
# load model
new_model = Word2Vec.load('R&D/model.bin')
print(new_model)

# Visualize word embeddings
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# Scatterplot
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0]+.0005, result[i, 1]+.0005))
pyplot.show()

# load the google word2vec model
filename = 'data/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)
print(result)

vector = model['this']
vector.shape

vectors = [model[x] for x in "This is some text I am processing with Spacy".split(' ')]
# GloVe: Global Vectors for Word Representations






########################################################### Creating Custom Word Embeddings
'''
Custom training embeddings on a specific domain increases performance than
pre-trained models. The gensim module provides API for [word2vec, doc2vec, fastText].
The main challange would be to collect a dataset.
Training model in gensim requires the input data to be a list of sentences, with each 
sentence being a list of words. So the steps would follow :
1. Data cleaning and make in a format of list of sentence.
2. Phrase detection using gensim parser.
'''

import os
import sys
import re
from gensim.models.phrases import Phraser, Phrases
TEXT_DATA_DIR = 'data/20_newsgroups'

# Newsgroups data is split between many files and folders.
# Directory stucture 20_newsgroup/<newsgroup label>/<post ID>
texts = []         # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []        # list of label ids
label_text = []    # list of label texts
# Go through each directory
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            # News groups posts are named as numbers, with no extensions.
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                f = open(fpath, encoding='latin-1')
                t = f.read() # t is a string format
                i = t.find('\n\n')  # skip header in file (starts with two newlines.)
                if 0 < i: # i has the index of \n\n in the complete string
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)
                label_text.append(name)
print('Found %s texts.' % len(texts))

sentences = []
# Go through each text in turn
for ii in range(len(texts)):
    sentences = [re.sub(pattern=r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]',
                        repl='',
                        string=x
                       ).strip().split(' ') for x in texts[ii].split('\n')
                      if not x.endswith('writes:')]
    sentences = [x for x in sentences if x != ['']]
    texts[ii] = sentences


# concatenate all sentences from all texts into a single list of sentences
all_sentences = []
for text in texts:
    all_sentences += text

# Phrase Detection
# Give some common terms that can be ignored in phrase detection
# For example, 'state_of_affairs' will be detected because 'of' is provided here:
common_terms = ["of", "with", "without", "and", "or", "the", "a"]
# Create the relevant phrases from the list of sentences:
phrases = Phrases(all_sentences, common_terms=common_terms)
# The Phraser object is used from now on to transform sentences
bigram = Phraser(phrases)
# Applying the Phraser to transform our sentences is simply
all_sentences = list(bigram[all_sentences])


model = Word2Vec(all_sentences,
                 min_count=3,   # Ignore words that appear less than this
                 size=200,      # Dimensionality of word embeddings
                 workers=2,     # Number of processors (parallelisation)
                 window=5,      # Context window for words during training
                 iter=30)       # Number of epochs training over corpus

print(len(model.wv.vocab))
print(model.most_similar('orange'))

