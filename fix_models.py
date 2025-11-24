# fix_models.py
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor

print("Creating proper models...")

# Load data
movies = pd.read_csv('data/movies.csv').head(1000)
ratings = pd.read_csv('data/ratings.csv')
ratings = ratings[ratings['movieId'].isin(movies['movieId'])].head(20000)

print(f"Using: {len(movies)} movies, {len(ratings)} ratings")

# Create proper models
class SimpleRecommender:
    def __init__(self):
        self.movies_with_stats = None
        self.cosine_sim = None
        self.knn = None
        self.tfidf = None
    
    def get_similar_movies(self, movie_title, top_n=10):
        if self.movies_with_stats is None or movie_title not in self.movies_with_stats['title'].values:
            return None
        idx = self.movies_with_stats[self.movies_with_stats['title'] == movie_title].index[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        return self.movies_with_stats.iloc[movie_indices]
    
    def get_popular_movies(self, top_n=10):
        if self.movies_with_stats is None:
            return pd.DataFrame()
        return self.movies_with_stats.sort_values('weighted_score', ascending=False).head(top_n)

# Create recommender
recommender = SimpleRecommender()

# 1. Create movies with stats
movie_stats = ratings.groupby('movieId').agg({'rating': ['count', 'mean']}).round(3)
movie_stats.columns = ['rating_count', 'rating_mean']
recommender.movies_with_stats = movies.merge(movie_stats, on='movieId', how='left')
recommender.movies_with_stats['rating_count'] = recommender.movies_with_stats['rating_count'].fillna(0)
recommender.movies_with_stats['rating_mean'] = recommender.movies_with_stats['rating_mean'].fillna(0)
recommender.movies_with_stats['weighted_score'] = recommender.movies_with_stats['rating_mean'] * np.log1p(recommender.movies_with_stats['rating_count'])

# 2. Create similarity matrix
recommender.tfidf = TfidfVectorizer(stop_words='english', max_features=500)
tfidf_matrix = recommender.tfidf.fit_transform(recommender.movies_with_stats['genres'])
recommender.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float16)

# 3. Create KNN model
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
recommender.knn = NearestNeighbors(n_neighbors=10, metric='cosine')
recommender.knn.fit(user_movie_matrix)

# 4. Create simple Random Forest
X = ratings[['userId', 'movieId']]
y = ratings['rating']
rf_model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
rf_model.fit(X, y)

# Save proper models
os.makedirs('models', exist_ok=True)

with open('models/full_recommender.pkl', 'wb') as f:
    pickle.dump(recommender, f)
print("✓ full_recommender.pkl")

with open('models/cosine_similarity.pkl', 'wb') as f:
    pickle.dump(recommender.cosine_sim, f)
print("✓ cosine_similarity.pkl")

with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(recommender.tfidf, f)
print("✓ tfidf_vectorizer.pkl")

with open('models/knn_model.pkl', 'wb') as f:
    pickle.dump(recommender.knn, f)
print("✓ knn_model.pkl")

with open('models/movies_with_stats.pkl', 'wb') as f:
    pickle.dump(recommender.movies_with_stats, f)
print("✓ movies_with_stats.pkl")

with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("✓ random_forest_model.pkl")

print("✅ All models created properly!")