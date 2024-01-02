import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
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

clf_method = st.sidebar.selectbox("Select Your Classifier: ", ("SVM", "Naive-Bayes", "Random Forest"))
model = None
def clf_method_pick(clf_method):
    model = None  # Initialize model to None
    if clf_method == "SVM":
        function = st.sidebar.selectbox("Select the Function: ", ("Kernel", "Gamma"))
        if function == "Kernel":
            Kernel = st.sidebar.selectbox("Select the Kernel parameter: ", ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'))
            model = SVC(kernel=Kernel)
        else:
            Gamma = st.sidebar.number_input("Gamma Value: ")
            st.sidebar.write(Gamma)
            model = SVC(gamma=Gamma)
    elif clf_method == "Naive-Bayes":
        nb_method = st.sidebar.selectbox("Select the Naive-Bayes Model: ", ("Multinomial", "Bernoulli"))
        if nb_method == "Multinomial":
            Alpha = st.sidebar.number_input("Alpha Value: ")
            st.sidebar.write(Alpha)
            model = MultinomialNB(alpha=Alpha)
        else:
            Alpha = st.sidebar.number_input("Alpha Value: ")
            st.sidebar.write(Alpha)
            model = BernoulliNB(alpha=Alpha)
    else:
        n_estimators = st.sidebar.slider("Number of Trees: ", min_value=50, max_value=200, step=50)
        max_depth = st.sidebar.slider("Maximum Tree Depth: ", min_value=1, max_value=30, step=1)
        min_samples_split = st.sidebar.slider("Minimum Samples Split: ", min_value=2, max_value=20, step=1)
        min_samples_leaf = st.sidebar.slider("Minimum Samples Leaf: ", min_value=1, max_value=10, step=1)
        max_features = st.sidebar.selectbox("Maximum Features: ", ['sqrt', 'log2'])

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features
        )
    return model

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
model = clf_method_pick(clf_method)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
st.write("Finish Model Training")

st.write("Start Model Prediction and Evaluation Scoring")   
st.write("Finish Model Prediction and Evaluation Scoring")
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
