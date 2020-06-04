import time
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import os, numpy as np, pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Conv1D, GRU
from keras.layers import Bidirectional
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import warnings
warnings.filterwarnings("ignore")

start_time = time.time()
np.random.seed(32)
os.environ["OMP_NUM_THREADS"] = "4"

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=2)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch + 1, score))


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
embedding_path = "crawl-300d-2M.vec"

embed_size = 300
max_features = 130000
max_len = 220

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
train["comment_text"].fillna("no comment")
test["comment_text"].fillna("no comment")
X_train, X_valid, Y_train, Y_valid = train_test_split(train, y, test_size=0.1)

raw_text_train = X_train["comment_text"].str.lower()
raw_text_valid = X_valid["comment_text"].str.lower()
raw_text_test = test["comment_text"].str.lower()

tk = Tokenizer(num_words=max_features, lower=True)
tk.fit_on_texts(raw_text_train)
X_train["comment_seq"] = tk.texts_to_sequences(raw_text_train)
X_valid["comment_seq"] = tk.texts_to_sequences(raw_text_valid)
test["comment_seq"] = tk.texts_to_sequences(raw_text_test)

X_train = pad_sequences(X_train.comment_seq, maxlen=max_len)
X_valid = pad_sequences(X_valid.comment_seq, maxlen=max_len)
test = pad_sequences(test.comment_seq, maxlen=max_len)


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path,encoding='utf8'))

word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


file_path = "best_model.hdf5"
check_point = ModelCheckpoint(file_path, monitor="val_loss", verbose=1,save_best_only=True, mode="min")
ra_val = RocAucEvaluation(validation_data=(X_valid, Y_valid), interval=1)
early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)


def build_model(lr=0.0, lr_d=0.0, units=0, dr=0.0):
    inp = Input(shape=(max_len,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x1 = SpatialDropout1D(dr)(x)

    x = Bidirectional(GRU(units, return_sequences=True))(x1)
    x = Conv1D(int(units / 2), kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)

    y = Bidirectional(LSTM(units, return_sequences=True))(x1)
    y = Conv1D(int(units / 2), kernel_size=2, padding="valid", kernel_initializer="he_uniform")(y)

    avg_pool1 = GlobalAveragePooling1D()(x)
    max_pool1 = GlobalMaxPooling1D()(x)

    avg_pool2 = GlobalAveragePooling1D()(y)
    max_pool2 = GlobalMaxPooling1D()(y)

    x = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])

    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    history = model.fit(X_train, Y_train, batch_size=128, epochs=3, validation_data=(X_valid, Y_valid),verbose=2, callbacks=[ra_val, check_point, early_stop])
    model = load_model(file_path)
    return model


model = build_model(lr=1e-3, lr_d=0, units=112, dr=0.2)
pred = model.predict(test, batch_size=1024, verbose=1)

print("printing model summary")
model.summary()

submission = pd.read_csv("sample_submission.csv")
submission[list_classes] = (pred)
submission.to_csv("submi_fasttext.csv", index=False)
print("[{}] Completed!".format(time.time() - start_time))