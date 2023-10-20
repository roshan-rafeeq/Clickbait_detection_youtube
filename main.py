import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Example data for training the model
st.title("YouTube Clickbait Title Prediction")

df = pd.read_csv("./Dataset/clickbait.csv")
st.write(df)

st.write("Start Stopwords Removal")
nltk.download("stopwords")
stop = stopwords.words('english')
st.write(stop)
df['title_without_stopwords'] = df['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in(stop)]))
st.write("Finish Stopwords Removal")
# print(df['title_without_stopwords'])
st.write("Title after the stopwords has been removed")
st.write(df.title_without_stopwords)

bow = CountVectorizer()
tfidf = TfidfVectorizer()
# Convert the text data into a matrix of token counts
pp_method = st.sidebar.selectbox("Select Your Preprocessing Mehtod:", ("Bag of Words", "TF-IDF"))
def pp_method_pick(pp_method, data):
    st.write(f"{pp_method} has been selected for the preprocessing method")
    if(pp_method == "Bag of Words"):
        X = bow.fit_transform(data)
    else:
        X = tfidf.fit_transform(data)
    return X

st.write("Start Preprocessing")
X = df.title_without_stopwords
X = pp_method_pick(pp_method, X)
y = df.label
st.write("Finished Preprocessing")


# Split the data into training and testing sets
st.write("Start Data Splitting (80% Training, 20% Testing)")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.write("Finish Data Splitting (80% Training, 20% Testing)")

# Train a logistic regression model
st.write("Start Model Training")
model = SVC(kernel = "linear")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
st.write("Finish Model Training")

st.write("Start Model Prediction and Evaluation Scoring")
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
st.write("Finish Model Prediction and Evaluation Scoring")

st.write(f"Accuracy: ", acc)
st.write(f"Precision: ", prec)
st.write(f"Recall: ", rec)
st.write(f"f1: ", f1)

# Example input string for prediction
input_string = st.text_area("YouTube Video Title: ", "Type Your Title Here")
st.write(input_string)

# Preprocess the input string
def input_method_pick(pp_method, data):
    if pp_method == "Bag of Words":
        X = bow.transform([data])
# input_vector = input_method_pick(pp_method, input_string)
if(pp_method == "Bag of Words"):
    input_vector = bow.transform([input_string])
else:
    input_vector = tfidf.transform([input_string])
# Predict the class
prediction = model.predict(input_vector)
st.write("The prediction for the inputed YouTube Title is:")
if prediction == 1:
    st.title("CLICKBAIT")
else:
    st.title("NON-CLICKBAIT")
st.write("The prediction for the input string is:", prediction[0])
