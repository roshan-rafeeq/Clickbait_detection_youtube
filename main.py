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

import pytube

import google.generativeai as genai



# Configure GenerativeAI with the gemini-pro model for sentiment analysis

genai.configure(api_key="AIzaSyCEFozD1fG-vwC37HxCAE8UYS8amhzsuw0")

model_gemini = genai.GenerativeModel(model_name="gemini-pro")



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

model_ml = None

def clf_method_pick(clf_method):

    model_ml = None  # Initialize model to None

    if clf_method == "SVM":

        function = st.sidebar.selectbox("Select the Function: ", ("Kernel", "Gamma"))

        if function == "Kernel":

            Kernel = st.sidebar.selectbox("Select the Kernel parameter: ", ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'))

            model_ml = SVC(kernel=Kernel)

        else:

            Gamma = st.sidebar.number_input("Gamma Value: ")

            st.sidebar.write(Gamma)

            model_ml = SVC(gamma=Gamma)

    elif clf_method == "Naive-Bayes":

        nb_method = st.sidebar.selectbox("Select the Naive-Bayes Model: ", ("Multinomial", "Bernoulli"))

        if nb_method == "Multinomial":

            Alpha = st.sidebar.number_input("Alpha Value: ")

            st.sidebar.write(Alpha)

            model_ml = MultinomialNB(alpha=Alpha)

        else:

            Alpha = st.sidebar.number_input("Alpha Value: ")

            st.sidebar.write(Alpha)

            model_ml = BernoulliNB(alpha=Alpha)

    else:

        n_estimators = st.sidebar.slider("Number of Trees: ", min_value=50, max_value=200, step=50)

        max_depth = st.sidebar.slider("Maximum Tree Depth: ", min_value=1, max_value=30, step=1)

        min_samples_split = st.sidebar.slider("Minimum Samples Split: ", min_value=2, max_value=20, step=1)

        min_samples_leaf = st.sidebar.slider("Minimum Samples Leaf: ", min_value=1, max_value=10, step=1)

        max_features = st.sidebar.selectbox("Maximum Features: ", ['sqrt', 'log2'])



        model_ml = RandomForestClassifier(

            n_estimators=n_estimators,

            max_depth=max_depth,

            min_samples_split=min_samples_split,

            min_samples_leaf=min_samples_leaf,

            max_features=max_features

        )

    return model_ml



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

model_ml = clf_method_pick(clf_method)

model_ml.fit(X_train, y_train)

y_pred_ml = model_ml.predict(X_test)

st.write("Finish Model Training")



st.write("Start Model Prediction and Evaluation Scoring for Machine Learning Model")

accuracy_ml = accuracy_score(y_test, y_pred_ml)

precision_ml = precision_score(y_test, y_pred_ml, pos_label=1)

recall_ml = recall_score(y_test, y_pred_ml, pos_label=1)

f1_ml = f1_score(y_test, y_pred_ml, pos_label=1)



st.write(f"Accuracy: {accuracy_ml * 100:.2f}%")

st.write(f"Precision: {precision_ml * 100:.2f}%")

st.write(f"Recall: {recall_ml * 100:.2f}%")

st.write(f"F1-Score: {f1_ml * 100:.2f}%")

st.write("Finish Model Prediction and Evaluation Scoring for Machine Learning Model")



# Input YouTube URL

input_url = st.text_input("YouTube Video URL: ", "Enter YouTube Video URL")


def get_thumbnail_url(url):
        yt = pytube.YouTube(url)
        return yt.thumbnail_url
if input_url:

    yt = pytube.YouTube(input_url)

    input_title = yt.title

    input_description = yt.description if yt.description else ""  # Handle None description
    thumb = get_thumbnail_url(input_url)
    
    




    if input_description:

        text_for_analysis = f"{input_title}\n{input_description}"

    else:

        text_for_analysis = input_title



    # Preprocess the input data

    def input_method_pick(pp_method, title, description):

        input_data = title + " " + description

        if pp_method == "Bag of Words":

            X = bow.transform([input_data])

        else:

            X = tfidf.transform([input_data])

        return X



    input_vector = input_method_pick(pp_method, input_title, input_description)



    # Predict using machine learning model

    prediction_ml = model_ml.predict(input_vector)



    # Predict using Gemini AI model

    prompt = f"Analyze the following text: {text_for_analysis}\n   - Is the title or description likely clickbait reply with yes or no? Why or why not? Reason:"

    response_gemini = model_gemini.generate_content(prompt)



    if response_gemini:

        result = response_gemini.text.strip().lower()

        reason = response_gemini.text.strip().split("Reason:")[-1].strip()

        if "yes" in result:

            gemini_prediction=1

        else:

            gemini_prediction=0

    else:

        st.write("Unable to analyze the video.")



    # Combine predictions using logical OR operation

    final_prediction = gemini_prediction and prediction_ml[0]



         

    st.write("Final Prediction:")

    if final_prediction == 0:

        st.title("NON-CLICKBAIT")
        if thumb:
            st.image(thumb, caption="YouTube Thumbnail")
        else:
            st.warning("Thumbnail not found for the given video ID.")

    else:
        if thumb:
            st.image(thumb, caption="YouTube Thumbnail")
        else:
            st.warning("Thumbnail not found for the given video ID.")
        st.title("CLICKBAIT")

        st.write(reason)

    st.write("The final prediction is:",final_prediction)
