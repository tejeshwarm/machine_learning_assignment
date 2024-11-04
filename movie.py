import pandas as pd
import numpy as np
import random

# Step 1: Create the dataset with the updated Tamil movies
data = {
    "movie_title": [
        "Master", 
        "Nayakan", 
        "Vikram", 
        "GOAT", 
        "Anbe Sivam",
        "Enthiran", 
        "24", 
        "Jigarthanda", 
        "Jailer", 
        "Indru Netru Naalai", 
        "Visaaranai", 
        "Ratsasan", 
        "Asuran", 
        "Leo", 
        "Mersal", 
        "Theri", 
        "Kaithi"
    ],
    "genre": [
        "action", 
        "crime", 
        "action", 
        "drama", 
        "drama",
        "sci-fi", 
        "sci-fi", 
        "drama", 
        "action", 
        "sci-fi", 
        "thriller", 
        "thriller", 
        "drama", 
        "action", 
        "action", 
        "action", 
        "action"
    ]
}

# Create the DataFrame
df = pd.DataFrame(data)

# Step 2: Initialize Q-learning variables
n_movies = len(df)
n_genres = len(df['genre'].unique())

# Initialize Q-table with zeros
Q = np.zeros((n_movies, n_genres))

# Define parameters
learning_rate = 0.1
discount_factor = 0.9
exploration_prob = 1.0
exploration_decay = 0.99
min_exploration_prob = 0.1
n_episodes = 30

# To track movies already recommended for each genre
recommended_movies = {genre: set() for genre in df['genre'].unique()}

# Step 3: Define the function to recommend movies
def recommend_movie(genre_index):
    available_movies = [
        i for i in range(n_movies) if df['genre'][i] == df['genre'].unique()[genre_index] and i not in recommended_movies[df['genre'].unique()[genre_index]]
    ]
    
    # Reset if all movies in this genre have been recommended
    if not available_movies:
        recommended_movies[df['genre'].unique()[genre_index]] = set()
        available_movies = [i for i in range(n_movies) if df['genre'][i] == df['genre'].unique()[genre_index]]
    
    # Get the movie index with the highest Q-value
    movie_index = max(available_movies, key=lambda x: Q[x, genre_index])
    return df.iloc[movie_index]['movie_title'], movie_index

# Function to update the Q-table
def update_q_table(movie_index, genre_index, reward):
    best_future_q = np.max(Q[movie_index])
    Q[movie_index, genre_index] += learning_rate * (reward + discount_factor * best_future_q - Q[movie_index, genre_index])

# Step 4: Train the agent
for episode in range(n_episodes):
    genre_index = random.randint(0, n_genres - 1)
    movie_title, movie_index = recommend_movie(genre_index)

    print(f"Recommended Movie for genre '{df['genre'].unique()[genre_index]}': {movie_title}")
    user_feedback = int(input("Rate the movie (1-5): "))  # 1 is bad, 5 is excellent

    # Normalize feedback to -2 (bad) to +2 (excellent)
    reward = user_feedback - 3
    update_q_table(movie_index, genre_index, reward)

    # Track the recommended movie for the genre
    recommended_movies[df['genre'].unique()[genre_index]].add(movie_index)

    # Decay exploration probability
    exploration_prob = max(min_exploration_prob, exploration_prob * exploration_decay)

# Step 5: Final recommendations after training
print("\nFinal movie recommendations based on learned preferences:")
for genre_index in range(n_genres):
    movie_title, _ = recommend_movie(genre_index)
    print(f"For genre '{df['genre'].unique()[genre_index]}', recommended movie: {movie_title}")
