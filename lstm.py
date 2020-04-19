import re
import time
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Basic Example

x = ['sun rises from east',
        'sun sets in west',
        'sun sets in north',
        'sun rises in south'
        'earth revolves around sun',
        'solar system has nine planets',
        'sun is longer in summers',
        'earth has long summers',
        'winters have less sun',
        'sun and sunlight mean the same',
     'sun sets in east']
labels = [1,1,0,0,1,1,1,1,1,0]
labels = np.array(labels)

docs = [clean_text(row) for row in x]

MAX_VOCAB_SIZE = 100
EMBEDDING_DIM = 20
MAX_DOC_LENGTH = 10

# Tokenize & pad sequences
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(docs)
encoded_docs = tokenizer.texts_to_sequences(docs)
word_index = tokenizer.word_index
print('Vocabulary size :', len(word_index))
sequences = pad_sequences(encoded_docs, padding='post', maxlen=MAX_DOC_LENGTH)
print('Shape of data tensor:', sequences.shape)
print('Shape of label tensor', labels.shape)

# Word Embeddings : the dimension are chosen in a experimental way have abstract meanings. They have nothing to do with corpus size.
# larger dimension will capture more information but harder to use.

model = Sequential()
model.add(Embedding(len(word_index)+1, EMBEDDING_DIM, input_length=MAX_DOC_LENGTH))
model.add(LSTM(units=64))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # only compilation
history = model.fit(sequences[:-1], labels[:-1], epochs=10, batch_size=10, validation_split=0.2)


model.predict([1,2,4,0,0,0,0,0,0,0])


# Clean the texts
def clean_text(text, remove_stopwords=True):
    output = ""
    text = str(text).replace(r'http[\w:/\.]+', '') # removing urls
    text = str(text).replace(r'[^\.\w\s]', '') # removing everything but characters and punctuation
    text = str(text).replace(r'\.\.+', '.') # replace multiple periods with a single one
    text = str(text).replace(r'\.', ' . ') # replace periods with a single one
    text = str(text).replace(r'\s\s++', ' ') # replace multiple whitespace with one
    text = str(text).replace(r'\n', '') # removing line break
    text = re.sub(r'[^\w\s]', '', text.lower()) # lower texts
    if remove_stopwords:
        text = text.split(" ")
        for word in text:
            if word not in stopwords.words('english'):
                output = output + " " + word
    return output











# '''
# Sequence classification :
# is a predictive algorithm where you have a sequence of inputs over space or time
# abd the goal is to categorize the sequence.
#
# '''
#
# # LSTM with Dropout for sequence classification in the IMDB dataset
# import numpy
# from keras.datasets import imdb
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout
# from keras.layers.embeddings import Embedding
# from keras.preprocessing import sequence
# # fix random seed for reproducibility
# numpy.random.seed(7)
# # load the dataset but only keep the top n words, zero the rest
# top_words = 5000
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# # truncate and pad input sequences
# max_review_length = 500
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# # create the model
# embedding_vecor_length = 32
# model = Sequential()
# model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# model.add(Dropout(0.2))
# model.add(LSTM(100))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# model.fit(X_train, y_train, epochs=1, batch_size=500)
# # Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))
#
#
#
# # create another model
# embedding_vecor_length = 32
# model = Sequential()
# model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# model.fit(X_train, y_train, epochs=3, batch_size=64)
# # Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))