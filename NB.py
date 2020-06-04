
import pandas as pd
from sklearn.pipeline import make_union
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import gc

print('Reading csv')
train = pd.read_csv("train.csv").fillna('unknown')
test = pd.read_csv("test.csv").fillna('unknown')

class_names = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
y = train[class_names]


train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

print('Making tfidf vectors')
word_vectorizer = TfidfVectorizer(ngram_range =(1,3),
                             min_df=3, max_df=0.9,
                             strip_accents='unicode',
                             stop_words = 'english',
                             analyzer = 'word',
                             use_idf=1,
                             smooth_idf=1,
                             sublinear_tf=1 )

char_vectorizer = TfidfVectorizer(ngram_range =(1,4),
                                 min_df=3, max_df=0.9,
                                 strip_accents='unicode',
                                 analyzer = 'char',
                                 stop_words = 'english',
                                 use_idf=1,
                                 smooth_idf=1,
                                 sublinear_tf=1,
                                 max_features=50000)

vectorizer = make_union(word_vectorizer, char_vectorizer)
vectorizer.fit(all_text)

train_matrix =vectorizer.transform(train['comment_text'])
test_matrix = vectorizer.transform(test['comment_text'])

val_score=[]
def cross_validation(model,y_train):
    score = cross_val_score(model,train_matrix,y_train,scoring='accuracy',cv=5)
    val_score.append(score.mean())

print("Building Naive Bayes Model")
model = MultinomialNB()

for clas in class_names:

    print(clas)
    cross_validation(model,train[clas])
    train_target = train[clas]
    model.fit(train_matrix,train_target)

    predictions = model.predict(train_matrix)
    print('\nAccuracy Score\n',accuracy_score(y[clas], predictions))
    print('\nConfusion matrix\n',confusion_matrix(y[clas], predictions))
    print(classification_report(y[clas], predictions))


