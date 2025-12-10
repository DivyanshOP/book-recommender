import os
import pickle

import pandas as pd
import streamlit as st

from similarity_generator import main as generate_similarity


@st.cache_data
def load_data():

    if not os.path.exists("similarity_scores.pkl"):
        generate_similarity()

    with open("book_list.pkl", "rb") as f:
        df = pickle.load(f)

    with open("similarity_scores.pkl", "rb") as f:
        simi = pickle.load(f)

    return df, simi


df, simi = load_data()

st.title("Book Recommender System")
st.markdown(
    "Select a book you like, and I'll recommend 5 similar ones "
    "based on **plot**, **era**, and **rating**."
)

book_list = df["title"].values
selected_book = st.selectbox("Type or select a book from the dropdown", book_list)


def recommender(title):
    k = df[df["title"] == title].index[0]
    distances = sorted(
        list(enumerate(simi[k])),
        reverse=True,
        key=lambda x: x[1],
    )
    # top 5, skipping the book itself at index 0
    return distances[1:6]


if st.button("Show Recommendations"):
    indicess = recommender(selected_book)
    st.subheader("You might also like:")

    cols = st.columns(5)

    for col, (idx, score) in zip(cols, indicess):
        with col:
            book_title = df.iloc[idx].title
            book_year = int(df.iloc[idx].published_year)
            book_rating = df.iloc[idx].average_rating
            book_thumbnail = df.iloc[idx].thumbnail

            st.image(book_thumbnail)
            st.markdown(f"**{book_title}**")
            st.text(f"Year: {book_year} | Rating: {book_rating}")
