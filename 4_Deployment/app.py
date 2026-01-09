import streamlit as st
import pickle

st.set_page_config(page_title="Financial News Sentiment Analyzer")

st.title("Real-Time Financial News Sentiment Analyzer")

text = st.text_input("Enter a financial news headline:")

if text:
    model = pickle.load(open("model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

    transformed = tfidf.transform([text.lower()])
    prediction = model.predict(transformed)[0]

    st.subheader("Predicted Sentiment:")
    st.success(prediction)
