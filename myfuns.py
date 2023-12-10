import pandas as pd
import numpy as np
import requests

# Define the URL for movie data
myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"
response = requests.get(myurl)
movie_lines = response.text.split('\n')
movie_data = [line.split("::") for line in movie_lines if line]
movies = pd.DataFrame(movie_data, columns=['MovieID', 'Title', 'Genres'])
movies['MovieID'] = movies['MovieID'].astype(int)


url2 = "https://liangfgithub.github.io/MovieData/ratings.dat?raw=true"
ratings = pd.read_csv(url2, sep='::', engine = 'python', header=None)
ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
ratings['MovieID'] = ratings['MovieID'].astype(int)

url3="https://drive.google.com/uc?export=download&id=1NvQOFbcRHcz3GikOFHefUE56xQvhMzKn"
S30 = pd.read_csv(url3,index_col=0)
names = S30.columns

url4="https://drive.google.com/uc?export=download&id=1tKUtZAx8sjq0yrEwAG6DqT-sqvtk49PV"
genre_rec_lookup = pd.read_csv(url4,index_col=0)


genres = list(
    sorted(set([genre for genres in movies.Genres.unique() for genre in genres.split("|")]))
)


movies2id = {}

for id in movies['MovieID'].unique():
  movies2id[int(id)] = movies[movies['MovieID'] == id].iloc[0]['Title']

id2movies = {}

for id in movies['MovieID'].unique():
  id2movies[int(id)] = movies[movies['MovieID'] == id].iloc[0]




def get_displayed_movies():
    return movies.head(100)

def get_recommended_movies(new_user_ratings):
    return movies.head(10)

def get_popular_movies(genre: str):
    if genre == genres[1]:
        return movies.head(10)
    else: 
        return movies[10:20]
    

def genreRecommendation(genre, n=10): # n top picks of genre
   return genre_rec_lookup[genre_rec_lookup['Genre'] == genre]
     


def myIBCF(newuser):

  n=len(names)

  # Code Block for handling passed dictionary of ratings in Dash, as opposed to full 3706, vector of ratings

  newuser_ = np.zeros((n,))
  names_=[]
  for na in names:
     names_.append(int(na[1:]))
  names_vec = pd.DataFrame(data=newuser_,index=names_)
  for k,v in newuser.items():
     names_vec.loc[k] = v

  newuser = np.array(names_vec)

  #newuser = np.nan_to_num(newuser, nan=0.0)

  movie_preds = np.zeros((n,))

  for l in range(n):

    rating = newuser[l]

    if (rating > 0):
      movie_preds[l] = 0
      continue

    else:
      #print("MOVIE", l)
      non_nan_indxs = np.where(S30.iloc[l] > 0)[0]  #~S30.iloc[l].is_na()
      
      #print("num non nan", len(non_nan_indxs))

      if (non_nan_indxs.shape[0] < 1):
         movie_preds[l] = 0
         continue
      
      #print("NONNANS", non_nan_indxs)

      denom=0
      num=0

      for i in range(non_nan_indxs.shape[0]):

        idx = non_nan_indxs[i]
        num += newuser[idx] * S30.iloc[l][idx]

        if (newuser[idx] > 0):
          denom += S30.iloc[l][idx]

      if (denom == 0):
        movie_preds[l] = 0
        continue

      movie_preds[l] = num / denom

  top_preds = np.argsort(movie_preds)[-10:][::-1]
  top_movies = names[top_preds]
  top_movies_ = [int(t[1:]) for t in top_movies]
  moviz = movies[movies['MovieID'].isin(top_movies_)]

  return moviz

