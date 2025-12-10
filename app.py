import streamlit as st
import pandas as pd
import pickle 
@st.cache_data
def load_data():
    with open('book_list.pkl','rb') as f:
        df = pickle.load(f)
    with open('similarity_scores.pkl','rb') as f:
        simi = pickle.load(f)
    return df,simi

df,simi = load_data()
st.title("Book Recommender System")
st.markdown("Select a book you like, and I'll recommend 5 similar ones based on **plot**, **era**, and **rating**.")
book_list = df['title'].values
selected_book = st.selectbox("Type or select a book from the dropdown",book_list)
def recommender(title):
    
    k = df[df['title']==title].index[0]
    distances = sorted(list(enumerate(simi[k])),reverse=True,key = lambda x: x[1])
    return distances[1:6]
if st.button('Show Recommendations'):
    indicess = recommender(selected_book)
    st.subheader("You might also like:")
    col1, col2, col3, col4, col5 = st.columns(5)
    recommended_books = []
    for i in indicess[1:6]:
        
        book_title = df.iloc[i[0]].title 
        book_year = int(df.iloc[i[0]].published_year)
        book_rating = df.iloc[i[0]].average_rating
        book_thumbnail = df.iloc[i[0]].thumbnail
        
        
        st.image(book_thumbnail)
        st.markdown(f"**{book_title}**")
        st.text(f"Year: {book_year} | Rating: {book_rating}")
        st.markdown("---")