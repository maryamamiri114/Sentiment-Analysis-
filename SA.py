from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from keras.initializers import Constant
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,GRU,LSTM,Embedding
from keras.layers import SpatialDropout1D,Dropout,Bidirectional,Conv1D,GlobalMaxPooling1D,MaxPooling1D,Flatten
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from numpy.random import seed
seed(1)
from sklearn.svm import SVC

train = pd.read_excel('s.db.xlsx')
test=pd.read_excel('Evaluation.xlsx')


x_train = train["review-RAW"].astype(str)
y_train = train["Sent"].astype(str)
x_test = test["review"].astype(str)
#y_test = test["Sent"].astype(str)

train["review-RAW"] = train["review-RAW"].astype(str)
test["review"]= test["review"].astype(str)

#Tokenization of text
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')

def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean,'', text)

train["html"]= train["review-RAW"].apply(clean_html)
test["html"]=test["review"].apply(clean_html)

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


train["imp"]= train["html"].apply(denoise_text)
test["imp"]=test["html"].apply(denoise_text)

def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text
#Apply function on review column
train["char"]= train["imp"].apply(remove_special_characters)
test["char"]=test["imp"].apply(remove_special_characters)

def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text
#Apply function on review column
train["stem"]=train["char"].apply(simple_stemmer)
test["stem"]=test["char"].apply(simple_stemmer)

stop=set(stopwords.words('english'))

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


train["stop"]=train["stem"].apply(remove_stopwords)
test["stop"]=test["stem"].apply(remove_stopwords)



X_train = train["stop"]
X_test = test["stop"]


from sklearn.pipeline import Pipeline


pipe2 = Pipeline([('tfidf', TfidfVectorizer()), ('clf', SVC(kernel="linear", probability=True))])

pipe2.fit(X_train, y = y_train)

y_pred_SVM= pipe2.predict(X_test)

print(type(y_pred_SVM))

df = pd.DataFrame(y_pred_SVM)

df.to_excel("y_pred_SVMeval.xlsx")

y_pred_SVM2 = pipe2.predict_proba(X_test)


df = pd.DataFrame(y_pred_SVM2)

df.to_excel("y_pred_SVM2eval.xlsx")

print("done1")


corpus = []
for text in X_train:
    words = [word.lower() for word in word_tokenize(text)]
    corpus.append(words)
for text in X_test:
    words = [word.lower() for word in word_tokenize(text)]
    corpus.append(words)

num_words = len(corpus)

tokenizer = Tokenizer(num_words)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=128, truncating='post', padding='post')

X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=128, truncating='post', padding='post')

word_index = tokenizer.word_index


embedding = {}
with open("glove.twitter.27B.100d.txt") as file:
    for line in file:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], 'float32')
        embedding[word] = vectors
file.close()

embedding_matrix = np.zeros((num_words, 100))
for word, i in word_index.items():
    if i > 1000:
        continue
    vector = embedding.get(word)
    if vector is not None:
        embedding_matrix[i] = vector


le = LabelEncoder()
y_train = le.fit_transform(y_train)


max_features = 13000
max_words = 50
batch_size = 128
epochs = 3
num_classes=1

X_train,X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)

model5_CNN= Sequential()
model5_CNN.add(Embedding(input_dim=num_words, output_dim=100, embeddings_initializer=Constant(embedding_matrix)))
model5_CNN.add(Dropout(0.2))
model5_CNN.add(Conv1D(1028,kernel_size=3,padding='same',activation='relu', kernel_regularizer=l2(0.02)))
model5_CNN.add(GlobalMaxPooling1D())
model5_CNN.add(Dense(128,activation='relu', kernel_regularizer=l2(0.02)))
model5_CNN.add(Dropout(0.2))
model5_CNN.add(Dense(num_classes,activation='sigmoid', kernel_regularizer=l2(0.02)))
model5_CNN.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model5_CNN.summary()


history = model5_CNN.fit(X_train, y_train, epochs=15, batch_size=512, validation_data=(X_val,y_val))

print(history)


y_pred_CNN = model5_CNN.predict_classes(X_test)

print(type(y_pred_CNN))

df = pd.DataFrame(y_pred_CNN)

df.to_excel("y_pred_CNNeval.xlsx")

y_pred_CNN2 = model5_CNN.predict(X_test)


df = pd.DataFrame(y_pred_CNN2)

df.to_excel("y_pred_CNN2eval.xlsx")

print("done2")


model3_LSTM= Sequential()
model3_LSTM.add(Embedding(input_dim=num_words, output_dim=100, embeddings_initializer=Constant(embedding_matrix)))
model3_LSTM.add(Bidirectional(LSTM(286,dropout=0.2, kernel_regularizer=l2(0.0001))))
#model3_LSTM.add(LSTM(64,dropout=0.4,return_sequences=False, kernel_regularizer=l2(0.02)))
model3_LSTM.add(Dense(128,activation='relu', kernel_regularizer=l2(0.0001)))
model3_LSTM.add(Dropout(0.2))
model3_LSTM.add(Dense(64,activation='relu', kernel_regularizer=l2(0.0001)))
model3_LSTM.add(Dense(num_classes,activation='sigmoid', kernel_regularizer=l2(0.0001)))

model3_LSTM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model3_LSTM.summary()


history=model3_LSTM.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=3, batch_size=batch_size)

print(history)

y_pred_LSTM = model3_LSTM.predict_classes(X_test)

print(type(y_pred_LSTM))

df = pd.DataFrame(y_pred_LSTM)

df.to_excel("y_pred_LSTMeval.xlsx")

y_pred_LSTM2 = model3_LSTM.predict(X_test)

df = pd.DataFrame(y_pred_LSTM2)

df.to_excel("y_pred_LSTM2eval.xlsx")

print("done3")


models1 = {'model1':y_pred_SVM2,'model2':y_pred_CNN2,'model3':y_pred_LSTM2}

sub_all1=pd.DataFrame([models1])
pred_mode1=sub_all1.agg('mean',axis=1)

df = pd.DataFrame(pred_mode1)
df.to_excel("pred_mode1.xlsx")

print("done6")

models2 = {'model1':y_pred_SVM,'model2':y_pred_CNN,'model3':y_pred_LSTM}

sub_all2=pd.DataFrame([models2])
pred_mode2=sub_all2.agg('mode',axis=1)

df = pd.DataFrame(pred_mode2)
df.to_excel("pred_mode2.xlsx")

print("done5")


