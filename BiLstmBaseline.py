import pandas as pd
from keras.models import Model
from sklearn import model_selection
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

max_features = 20000
maxlen = 100

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train_val = train.sample(frac=1)

list_sentences_train = train["comment_text"].fillna("NULL").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("NULL").values

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

def get_model():
    embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

model = get_model()

X_train, X_valid, Y_train, Y_valid = model_selection.train_test_split(X_t, y, test_size=0.1)

model.summary()
batch_size = 10
epochs = 3

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, Y_valid),verbose=2)

score, acc = model.evaluate(X_valid, Y_valid, batch_size=batch_size)
print('Value of loss function of model is:', score)
print('Validation accuracy is:', acc)

