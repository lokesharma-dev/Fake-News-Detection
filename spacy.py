# Import packages
import pandas as pd
import numpy as np
import random
import spacy
import time
import re


from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential


'''
Text Preprocessing using spaCy 
1. Sentence detection / Tokenization
2. Stemming / Lemmatization
3. Stopwords
4. POS tagging
5. Punctuations and Noise removal
'''

class Spacy(object):

    def __init__(self):
        # python -m spacy download en_core_web_lg
        self.nlp = spacy.load("en_core_web_lg")
        pass

    def deNoise(self, text):
        text = re.sub(r'[“”""]', '', text) # removes quotes
        text = text.replace("'s", '')
        text = re.sub(r'[-]', ' ', text) # helps in splitting doc into sentences
        text = re.sub(r'http[\w:/\.]+', '', text) # removing urls
        text = re.sub(r'[^\.\w\s]', '', text) # removing everything but characters and punctuation
        text = re.sub(r'\.', '.', text) # replace periods with a single one
        text = re.sub(r'\n', ' ', text) # removing line break
        text = re.sub(r'[^\w\s]', '', text.lower())
        text = re.sub(r'\s\s+', ' ', text)  # replace multiple whitespace with one
        return text

    def stopWords(self, text):
        tokens = ""
        spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
        text = text.split(" ")
        for word in text:
            if word not in spacy_stopwords:
                tokens = tokens + " " + word
        return tokens

    def lemmatize(self, tokens):
        lemma_token = ""
        tokens_object = self.nlp(tokens)
        # lemma_token = [token.lemma_ for token in tokens_object]
        # lemma_token = ''.join(lemma_token) # converts list to string
        for token in tokens_object:
            lemma_token = lemma_token + " " + token.lemma_
        lemma_token = re.sub(r'\s\s+', ' ', lemma_token)  # replace multiple whitespace with one
        lemma_token = lemma_token.strip() # removes trailing whitespaces
        return  lemma_token

    def set_custom_boundaries(self, doc):
        for token in doc[:-1]:
            if token.text == '--':
                doc[token.i+1].is_sent_start = True
        return doc

    def sentence_detect(self, text):
        self.nlp.add_pipe(self.set_custom_boundaries, before='parser')
        doc = self.nlp(text)
        sentences = list(doc.sents)
        for sentence in sentences:
            print(sentence)

    def tokenize(self, text):
        doc = self.nlp(text)
        print([token.text for token in doc])

    def orchestrate(self, text):
        return self.lemmatize(self.stopWords(self.deNoise(text)))


class Embedding(object):

    def __init__(self):
        # Parameters
        self.MAX_VOCAB_SIZE = 1000000  # maximum no of unique words
        self.MAX_DOC_LENGTH = 500  # maximum no of words in each sentence
        self.EMBEDDING_DIM = 300  # Embeddings dimension from Glove directory
        self.GLOVE_DIR = 'models/glove.6B/glove.6B.' + str(self.EMBEDDING_DIM) + 'd.txt'

    def tokenize_padding(self, docs):
        # Tokenize & pad sequences
        tokenizer = Tokenizer(num_words=self.MAX_VOCAB_SIZE, oov_token='-EOS-')
        tokenizer.fit_on_texts(docs)
        encoded_docs = tokenizer.texts_to_sequences(docs)
        word_index = tokenizer.word_index
        print('Vocabulary size :', len(word_index))
        sequences = pad_sequences(encoded_docs, padding='post', maxlen=self.MAX_DOC_LENGTH)
        return [word_index, sequences]

    def load_glove(self):
        embeddings_index = {}
        f = open(self.GLOVE_DIR, encoding='utf-8')
        print('Loading Glove from: ', self.GLOVE_DIR, '...', end='')
        for line in f:
            values = line.split()
            word = values[0]
            embeddings_index[word] = np.asarray(values[1:], dtype='float32')
        f.close()
        print('Found %s word vectors.' % len(embeddings_index))
        print('\nDone.\nProcedding with Embedded Matrix...', end='')
        return embeddings_index

    def embedding_matrix(self, word_index, embeddings_index):
        # Create an embedding matrix
        # first create a matrix of zeros, this is our embedding matrix
        embeddings_matrix = np.zeros((len(word_index) + 1, self.EMBEDDING_DIM))
        # embeddings_matrix = np.random.random(((20568),EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embeddings_matrix[i] = embedding_vector
            else:
                # doesn't exist, assign a random vector
                embeddings_matrix[i] = np.random.random(self.EMBEDDING_DIM)
        print('\nCompleted')
        return embeddings_matrix


if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv('GBVAT/data/processed_datasets/celebrityDataset.csv')

    # Feature Engineering
    df.nunique()
    df.isna().sum()
    df.Subject.fillna('', inplace=True)

    x = df.Subject + " " + df.Content
    y = pd.Series([0 if row == 'Fake' else 1 for row in df.Label])  # Series is 1D array but with same dtype

    S = Spacy()
    start = time.time()
    docs = [S.orchestrate(row) for row in x]
    end = time.time()
    print("Cleaning the document took {} seconds".format(round(end - start)))

    E = Embedding()
    sequences = E.tokenize_padding(docs)
    word_index = sequences[0]
    sequences = sequences[1]
    print('Shape of data tensor:', sequences.shape)
    print('Shape of label tensor', y.shape)

    # Shuffle data random before splitting
    indices = np.arange(sequences.shape[0])
    random.Random(1).shuffle(indices)
    data = sequences[indices]
    labels = y[indices]

    # save processed files as npy
    # np.save('GBVAT/data/npy/spacy/data.npy', data)
    # np.save('GBVAT/data/npy/spacy/label.npy', labels)
    #
    # np.save('GBVAT/data/npy/nltk/data.npy', data)
    # np.save('GBVAT/data/npy/nltk/label.npy', labels)
    #
    #
    #
    #
    # TEST_SPLIT = 0.2
    # EMBEDDING_DIM = 300
    # MAX_DOC_LENGTH = 500
    #
    # # Split into test set
    # num_test_samples = int(TEST_SPLIT * data.shape[0])
    # x_train = data[:-num_test_samples]
    # y_train = labels[:-num_test_samples]
    # x_test = data[-num_test_samples:]
    # y_test = labels[-num_test_samples:]
    #
    # # model.add(Embedding(input_dim=len(word_index)+1, output_dim=EMBEDDING_DIM,
    # #                     embeddings_initializer = Constant(embeddings_matrix),
    # #                     input_length = MAX_DOC_LENGTH,
    # #                     trainable=True,
    # #                     mask_zero=True))
    #
    # temp = x_train[0]
    #
    #
    # # Develop DNN
    # model = Sequential()
    # model.add(Embedding(input_dim=len(word_index)+1,
    #                     output_dim=EMBEDDING_DIM,
    #                     input_length=MAX_DOC_LENGTH,
    #                     trainable=False))
    #
    # model.add(LSTM(units=128))
    # model.add(Dense(1, activation='sigmoid'))
    # model.summary()
    #
    # # Train the model
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # only compilation
    # history = model.fit(x_train, y_train, epochs=3, batch_size=40, validation_split=0.2)
    # # evaluating model
    # score, acc = model.evaluate(x_test, y_test, batch_size=10)
    # print('Test score:', score)
    # print('Test accuracy:', acc)

    # Currently not using Glove embedding values rather going with random values
    # start = time.time()
    # embeddings_index = E.load_glove()
    # embeddings_matrix = E.embedding_matrix(word_index, embeddings_index)
    # end = time.time()
    # print("Loading Glove and creating Embedding matrix took {} seconds".format(round(end - start)))


'''
To Dos
1. In Spacy an extra s occurs with -PRON- eg: she's
2. Set a dynamic range for MAX_DOC_LENGTH to best optimize. eg : Median length of all docs
'''
