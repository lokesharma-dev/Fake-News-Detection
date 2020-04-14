########################################################### Creating Custom Word Embeddings
'''
Custom training embeddings on a specific domain increases performance than
pre-trained models. The gensim module provides API for [word2vec, doc2vec, fastText].
The main challenge would be to collect a dataset.
Training model in gensim requires the input data to be a list of sentences, with each
sentence being a list of words. So the steps would follow :
1. Data cleaning and make in a format of list of sentence.
2. Phrase detection using gensim parser.
'''
import os
import re
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from gensim.models import Word2Vec


JSON_DATA_DIR = 'data/dataverse_Nela_Gt.2019/nela-eng-2019/'

df = pd.read_json('data/dataverse_Nela_Gt.2019/nela-eng-2019/21stcenturywire.json', orient = 'columns')
labels = pd.read_csv('data/dataverse_Nela_Gt.2019/labels.csv')


for name in sorted(os.listdir(JSON_DATA_DIR)):
    path = os.path.join(JSON_DATA_DIR, name)
    content = pd.read_json(path)['content']

text = []
for row in df['content']:
    text.append(row.lower())

all_sentences = []
for str in text:
    str = str.split('\n\n')
    for sentence in str:
        sentence = ''.join(sentence)
        sentence = [re.sub(pattern=r'[^A-Za-z]',
                            repl='',
                            string=x) for x in sentence.replace('.',' ').split(' ')]

        sentence = [token for token in sentence if token!='']
        if len(sentence)>0:
            all_sentences.append(sentence)


# remove common words and tokenize
stoplist = set('a the is'.split())

all_sentences = [[word for word in sentence if word not in stoplist and len(word)>1] for sentence in all_sentences]

# Build the custom model
model = Word2Vec(all_sentences,
                 min_count=5,   # Ignore words that appear less than this
                 size=300,      # Dimensionality of word embeddings
                 workers=4,     # Number of processors (parallelisation)
                 window=5,      # Context window for words during training
                 iter=30)       # Number of epochs training over corpus

for i, word in enumerate(model.wv.vocab):
    if i == 25:
        break
    print(word)

print(len(model.wv.vocab))

##################################################### Reduce Dimension for Visual purposes.
def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    vectors = [] # positions in vector space
    labels = [] # keep track of words to label our data again later
    for word in model.wv.vocab:
        vectors.append(model.wv[word])
        labels.append(word)

    # convert both lists into numpy vectors for reduction
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    # reduce using t-SNE
    vectors = np.asarray(vectors)
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels

x_vals, y_vals, labels = reduce_dimensions(model)


##################################################### Matplotlib
def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))
    plt.show()

plot_with_matplotlib(x_vals, y_vals, labels)

print(model.most_similar('confronted'))