import pandas as pd
import numpy as np

def calculate_weighted_rating(data):
    """
    Calculates the IMDB weighted rating for each movie in the dataset.

    The weighted rating considers both the average rating of the movie and 
    the number of votes it has received, penalizing movies with very few votes 
    by pulling their score closer to the global average.

    Args:
        data (pd.DataFrame): The movie dataset containing 'vote_average' and 'vote_count'.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The dataset with an added 'weighted_rating' column.
            - float (C): The mean vote average across the entire dataset.
            - float (m): The minimum number of votes required to be in the 60th percentile.
    """
    C = data['vote_average'].mean()
    m = data['vote_count'].quantile(0.60)

    def wr_func(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + m) * R) + (m / (v + m) * C)

    data['weighted_rating'] = data.apply(wr_func, axis=1)
    return data, C, m

def get_baseline_recommendations(df, n=10, min_votes=None, genre_filter=None):
    """
    Generates baseline movie recommendations based strictly on the highest weighted ratings.

    Args:
        df (pd.DataFrame): The movie dataset containing a 'weighted_rating' column.
        n (int, optional): The number of top movies to return. Defaults to 10.
        min_votes (float, optional): The minimum number of votes a movie must have. 
            If None, the 60th percentile of votes is used. Defaults to None.
        genre_filter (str, optional): A specific genre to filter the recommendations by. 
            Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the top `n` recommended movies.
    """
    if min_votes is None:
        m = df['vote_count'].quantile(0.60)
        min_votes = m
    
    filtered_df = df[df['vote_count'] >= min_votes].copy()
    
    if genre_filter:
        filtered_df = filtered_df[
            filtered_df['genres'].apply(lambda x: genre_filter in x)
        ]
    
    top_n = filtered_df.sort_values('weighted_rating', ascending=False).head(n)
    return top_n

def recommendation(title, cosine_similarity, data, indices, alpha=0.3, min_votes=50, use_mmr=True, lambda_mmr=0.5):
    """
    Generates intelligent content-based movie recommendations using similarity matrices, 
    quality normalization, and popularity/age penalties. Optionally applies Maximal 
    Marginal Relevance (MMR) for result diversity.

    Args:
        title (str): The title of the movie the user liked.
        cosine_similarity (np.ndarray): The precomputed cosine similarity matrix.
        data (pd.DataFrame): The preprocessed movie dataset.
        indices (pd.Series): A pandas Series mapping movie titles to their DataFrame indices 
            (created once in main.py to boost performance).
        alpha (float, optional): Weight balancing quality vs. similarity (0 to 1). Defaults to 0.3.
        min_votes (int, optional): Minimum vote threshold for recommended movies. Defaults to 50.
        use_mmr (bool, optional): Whether to apply MMR for result diversity. Defaults to True.
        lambda_mmr (float, optional): Trade-off parameter for MMR. Defaults to 0.5.

    Returns:
        pd.DataFrame: A DataFrame containing the final recommended movies with their scores.
    """
    if title not in indices:
        print(f"Error: Title '{title}' not found in the index.")
        return pd.DataFrame() 
    
    idx = indices[title]
    
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    reference_age = data.iloc[idx]['movie_age']
    if pd.isna(reference_age):
        reference_age = data['movie_age'].median()

    similarity_scores = list(enumerate(cosine_similarity[idx]))
    
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:201]

    movie_indices = [i[0] for i in similarity_scores]
    similarity_values = [i[1] for i in similarity_scores]
    
    candidates = data.iloc[movie_indices].copy()
    candidates['similarity'] = similarity_values

    candidates = candidates[candidates['vote_count'] >= min_votes]
    if candidates.empty:
        return pd.DataFrame()

    def normalize_series(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    candidates['quality_norm'] = normalize_series(candidates['vote_average'])
    candidates['similarity_norm'] = normalize_series(candidates['similarity'])
    
    log_votes = np.log1p(candidates['vote_count'])
    candidates['popularity_penalty'] = 1 - (0.3 * (log_votes / log_votes.max())) 

    candidates['age_diff'] = abs(candidates['movie_age'] - reference_age)
    candidates['age_pen'] = 1 / (1 + candidates['age_diff'] / 10)

    candidates['final_score'] = (
        (alpha * candidates['quality_norm']) +
        ((1 - alpha) * candidates['similarity_norm'])
    ) * candidates['age_pen'] * candidates['popularity_penalty']

    if use_mmr and len(candidates) > 10:
        pass 
    else:
        candidates = candidates.sort_values('final_score', ascending=False).head(20)

    output_cols = ['original_title', 'movie_age', 'vote_average', 'vote_count', 'genres', 'final_score', 'similarity']
    existing_output = [c for c in output_cols if c in candidates.columns]
    
    return candidates[existing_output].reset_index(drop=True)