import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn import model_selection
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import GlobalMaxPool1D
from keras.models import Model

print("Reading from csv")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.isnull().any(),test.isnull().any()

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]


print("making vectors")
max_word = 20000
tokenizer = Tokenizer(num_words=max_word)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

print("Building model")
inp = Input(shape=(maxlen, ))

embed_size = 128
x = Embedding(max_word, embed_size)(inp)

x = LSTM(60, return_sequences=True,name='lstm_layer')(x)

x = GlobalMaxPool1D()(x)

x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)


model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

X_train, X_valid, Y_train, Y_valid = model_selection.train_test_split(X_t, y, test_size=0.1)

print("Model summary:")
model.summary()
print("Fitting model")
batch_size = 10
epochs = 1


model.fit(X_train,Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, Y_valid),verbose=2)
print("Model summary:")
model.summary()
score, acc = model.evaluate(X_valid, Y_valid, batch_size=batch_size)
print('Value of loss function of model is:', score)
print('Validation accuracy is:', acc)

y_test = model.predict(X_te)

sample_submission = pd.read_csv("test_labels.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv("lstm_submi.csv", index=False)