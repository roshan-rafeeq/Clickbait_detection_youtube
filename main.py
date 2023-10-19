import pandas as pd
import nltk
from nltk.corpus import stopwords
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
import streamlit as st

st.title("YouTube Title Classification")

df = pd.read_csv("./Dataset/clickbait.csv")
st.write(df)
# print(df)

st.write("Start Stopwords Removal")
nltk.download("stopwords")
stop = stopwords.words('english')
st.write(stop)
# print(stop)

df['title_without_stopwords'] = df['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in(stop)]))
st.write("Finish Stopwords Removal")
# print(df['title_without_stopwords'])
st.write("Title after the stopwords has been removed")
st.write(df["title_without_stopwords"])

le = LabelEncoder()
le.fit(df['label'])
df['label_encoded'] = le.transform(df['label'])

title = df.title_without_stopwords
y = df.label_encoded

pp_method = st.sidebar.selectbox("Select Preprocessing Methods: ", ("Bag of Words", "TF-IDF"))

st.write(f"Start {pp_method} Preprocessing")
if(pp_method == "TF-IDF"):
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(title)
    X.toarray()    
st.write(f"Finish {pp_method} Preprocessing")

st.write("Start Data Splitting (80% Training Data, 10% Test Data, 10% Validation Data)")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# x_tfidf_test, x_tfidf_val, y_tfidf_test, y_tfidf_val = train_test_split(x_tfidf_test, y_tfidf_test, test_size = 0.5, random_state = 42)
st.write("Finish Splitting Data")

st.write("Start Model Training")
svm = SVC(kernel = 'linear', probability = True)
svm.fit(X_train, y_train)
st.write("Finish Model Training")

st.write("Start Model Prediction and Evaluation Scoring")
y_pred = svm.predict(X_test)
accuracy_kernel_tfidf = accuracy_score(y_test, y_pred)
precision_kernel_tfidf = precision_score(y_test, y_pred)
recall_kernel_tfidf = recall_score(y_test, y_pred)
f1_kernel_tfidf = f1_score(y_test, y_pred)
st.write("Finish Model Prediction and Evaluation Scoring")

st.write(f"Accuracy: ", accuracy_kernel_tfidf)
st.write(f"Precision: ", precision_kernel_tfidf)
st.write(f"Recall: ", recall_kernel_tfidf)
st.write(f"f1: ", f1_kernel_tfidf)
    

    
    



