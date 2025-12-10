import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
def space_fixer(text):
    L=[]
    for i in text:
        L.append(i.replace(" ",""))
        return L
def main():
    books = pd.read_csv("Datasets/data.csv")
    books = books[['title','authors','categories','thumbnail','description','published_year','average_rating','ratings_count']]
    books.dropna(inplace=True)
    
        
    books['authors']= books['authors'].apply(lambda x: x.split(";"))
    
    books['authors']=books['authors'].apply(space_fixer)
    books['categories']=books['categories'].apply(lambda x : x.replace(" ",""))
    books['authors']=books['authors'].apply(lambda x: " ".join(x))
    books['tags2'] = books['authors']+ " " + books['categories'] + " "+ books['description']
    final_data = books[['title','tags2','published_year','average_rating','thumbnail']]
    vectorizer = TfidfVectorizer(max_features=3000,stop_words="english")
    y = vectorizer.fit_transform(final_data['tags2'])
    y = y.toarray()
    similarity = cosine_similarity(y)
    scaler = MinMaxScaler()

    final_data[['year_scaled', 'rating_scaled']] = scaler.fit_transform(final_data[['published_year', 'average_rating']])
    final_data = final_data.reset_index(drop=True)
    similarity2= 1/(1+euclidean_distances(final_data['rating_scaled'].to_numpy().reshape(-1,1)))
    similarity3 = 1/(1+euclidean_distances(final_data['year_scaled'].to_numpy().reshape(-1,1)))
    final_similarity = 0.7*similarity + 0.2*similarity2 + 0.1*similarity3
    with open("similarity_scores.pkl", "wb") as f:
        pickle.dump(final_similarity, f)

    print("similarity_scores.pkl generated.")

if __name__ == "__main__":
    main()
