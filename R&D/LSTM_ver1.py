import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding,LSTM
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from tensorflow.keras import regularizers, initializers, optimizers, callbacks

df = pd.read_csv('data/practise/fake_or_real_news.csv')
df.head()
df.info()

# Extract relevant features.
x = df['title'] + " " + df['text']
y = pd.get_dummies(df['label'])
y = np.array(y)

# Parameters
MAX_NB_WORDS = 100000 # max number of words for tokenizer
MAX_SEQUENCE_LENGTH = 1000 # max length of each sentences, including padding
VALIDATION_SPLIT = 0.2 # 20% data for validation
EMBEDDING_DIM = 100 # dimensions
GLOVE_DIR = 'models/glove.6B/glove.6B.' + str(EMBEDDING_DIM) + 'd.txt'

# Data Cleaning
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
            if '\n' in word: # Manual intervention
                word = word.replace('\n', '')
            if word not in stopwords.words('english'):
                output = output + " " + word
    return output

texts = []
for row in x:
    texts.append(clean_text(row))

# Tokenize your data
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Vocabulary size:', len(word_index))

# Add padding to make it uniform
data = pad_sequences(sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', y.shape)

# Shuffle your data randomly
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = y[indices]

# Split into validation set
num_validation_samples = int(VALIDATION_SPLIT*data.shape[0])
x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
print('Number of entries in each category:')
print('Training: ', y_train.sum(axis=0))
print('Validation: ',y_val.sum(axis=0))

# Word Embeddings : the dimension are chosen in a experimental way have abstract meanings. They have nothing to do with corpus size.
# larger dimension will capture more information but harder to use.

embeddings_index = {}
f = open(GLOVE_DIR, encoding='utf-8')
print('Loading Glove from: ', GLOVE_DIR, '...', end='')
for line in f:
    values = line.split()
    word = values[0]
    embeddings_index[word] = np.asarray(values[1:], dtype='float32')
f.close()
print('\nDone.\n Procedding with Embedded Matrix...', end='')

embeddings_matrix = np.random.random(((len(word_index)+1),EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector
print('\nCompleted')

# LSTM Model initialization
model = Sequential()
model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
model.add(Embedding(len(word_index)+1, EMBEDDING_DIM,
                    weights = [embeddings_matrix],
                    input_length = MAX_SEQUENCE_LENGTH,
                    trainable = False,
                    name = 'embeddings'))

model.add(LSTM(60, return_sequences=True, name='lstm_layer'))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.1))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='sigmoid'))

# Train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # only compilation
history = model.fit(x_train, y_train, epochs=10, batch_size=128,
                    validation_data=(x_val, y_val))

# Model Evaluation
import matplotlib.pyplot as plt
loss = history.history[‘loss’]
val_loss = history.history[‘val_loss’]
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, label=’Training loss’)
plt.plot(epochs, val_loss, label=’Validation loss’)
plt.title(‘Training and validation loss’)
plt.xlabel(‘Epochs’)
plt.ylabel(‘Loss’)
plt.legend()
plt.show()

accuracy = history.history[‘acc’]
val_accuracy = history.history[‘val_acc’]
plt.plot(epochs, accuracy, label=’Training accuracy’)
plt.plot(epochs, val_accuracy, label=’Validation accuracy’)
plt.title(‘Training and validation accuracy’)
plt.ylabel(‘Accuracy’)
plt.xlabel(‘Epochs’)
plt.legend()
plt.show()

random_num = np.random.randint(0, 100)
test_data = x[random_num]
test_label = y[random_num]
clean_test_data = clean_text(test_data)
test_tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
test_tokenizer.fit_on_texts(clean_test_data)
test_sequences = tokenizer.texts_to_sequences(clean_test_data)
word_index = test_tokenizer.word_index
test_data_padded = pad_sequences(test_sequences, padding = ‘post’, maxlen = MAX_SEQUENCE_LENGTH)

prediction = model.predict(test_data_padded)
prediction[random_num].argsort()[-len(prediction[random_num]):]
