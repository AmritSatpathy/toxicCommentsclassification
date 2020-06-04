import numpy as np, pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import GlobalMaxPool1D,Bidirectional
from keras.models import Model
from sklearn import model_selection
import gensim.models.keyedvectors as word2vec
import gc

print("Reading from csv")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
embed_size=0

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]

print("Making vectors")
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


def loadEmbeddingMatrix(typeToLoad):

    if (typeToLoad == "word2vec"):
        word2vecDict = word2vec.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
        embed_size = 300


    embeddings_index = dict()
    for word in word2vecDict.wv.vocab:
            embeddings_index[word] = word2vecDict.word_vec(word)
    print(typeToLoad)
    print('Loaded %s word vectors ' % len(embeddings_index))

    gc.collect()

    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    nb_words = len(tokenizer.word_index)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    gc.collect()

    embeddedCount = 0
    for word, i in tokenizer.word_index.items():
        i -= 1
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            embeddedCount += 1
    print('total embedded:', embeddedCount, 'common words')

    del (embeddings_index)
    gc.collect()

    return embedding_matrix

print("Loading Embedding Matrix")
embedding_matrix = loadEmbeddingMatrix('word2vec')

print("Building model")
inp = Input(shape=(maxlen, ))

x = Embedding(len(tokenizer.word_index), embedding_matrix.shape[1],weights=[embedding_matrix],trainable=False)(inp)
x = Bidirectional(LSTM(60, return_sequences=True,name='lstm_layer',dropout=0.1,recurrent_dropout=0.1))(x)

x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


X_train, X_valid, Y_train, Y_valid = model_selection.train_test_split(X_t, y, test_size=0.1)
print("Printing Model Summary")
model.summary()

batch_size = 10
epochs = 4
print("Fitting model")
model.fit(X_train,Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, Y_valid),verbose=2)
score, acc = model.evaluate(X_valid, Y_valid, batch_size=batch_size)
print('Value of loss function of model is:', score)
print('Validation accuracy is:', acc)


