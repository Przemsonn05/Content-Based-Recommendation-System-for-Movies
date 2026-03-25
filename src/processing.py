import pandas as pd
import numpy as np
import ast
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler, normalize
from scipy.sparse import csr_matrix, hstack
from datetime import datetime

def parse_json_columns(data):
    """
    Cleans and formats metadata columns for vectorization (TF-IDF/Count).
    
    This function assumes that the JSON columns have already been decoded and 
    converted into lists of strings during the data loading phase. It focuses 
    on trimming lists and normalizing text (lowercase, removing spaces).

    Args:
        data (pd.DataFrame): The movie dataset with list-formatted metadata columns.

    Returns:
        pd.DataFrame: The dataset with cleaned and formatted text columns.
    """
    if 'cast' in data.columns:
        data['cast'] = data['cast'].apply(lambda x: x[:5] if isinstance(x, list) else [])

    if 'crew' in data.columns:
        data = data.drop('crew', axis=1)

    def clean_list(x):
        if isinstance(x, list): 
            return [str(i).lower().replace(" ", "") for i in x]
        if isinstance(x, str): 
            return x.lower().replace(" ", "")
        return ""

    for feature in ['cast', 'keywords', 'director', 'genres']:
        if feature in data.columns:
            data[feature] = data[feature].apply(clean_list)
            
    return data

def add_engineered_features(data):
    """
    Creates new engineered features for the recommendation models.

    This function calculates movie age, applies logarithmic transformations to 
    highly skewed numerical columns (popularity, vote count), creates a hybrid 
    score combining rating and popularity, and handles missing runtime/overview values.

    Args:
        data (pd.DataFrame): The preprocessed movie dataset.

    Returns:
        pd.DataFrame: The dataset enriched with engineered features.
    """
    current_year = datetime.now().year
    data['movie_age'] = current_year - data['release_year']
    data['log_movie_age'] = np.log1p(data['movie_age'].fillna(0))
    data['log_vote_count'] = np.log1p(data['vote_count'])
    data['log_popularity'] = np.log1p(data['popularity'])

    data.loc[data['runtime'] <= 15, 'runtime'] = np.nan
    data['runtime'] = data['runtime'].fillna(data['runtime'].median())

    scaler = MinMaxScaler()
    data[['popularity_scaled', 'vote_average_scaled']] = scaler.fit_transform(
        data[['popularity', 'vote_average']]
    )

    data['hybrid_score'] = (
        0.6 * data['vote_average_scaled'] +
        0.4 * data['popularity_scaled']
    )

    C = data['vote_average'].mean()
    m = data['vote_count'].quantile(0.60)
    
    def WR(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + m) * R) + (m / (v + m) * C)

    data['score'] = data.apply(WR, axis=1)
    scaler_score = MinMaxScaler()
    data['scaled_score'] = scaler_score.fit_transform(data[['score']])

    data['overview'] = data.apply(
        lambda x: str(x['overview']) + ' ' + str(x['tagline']) if len(str(x['overview'])) < 100 else str(x['overview']), 
        axis=1
    )
    
    return data

def build_matrices(data):
    """
    Constructs the final combined feature matrix for the content-based recommendation model.

    This function creates a metadata "soup" from genres, keywords, cast, and director, 
    applies CountVectorizer to the soup, applies TfidfVectorizer to the movie overviews, 
    and combines them with scaled numerical features into a single sparse matrix.

    Args:
        data (pd.DataFrame): The enriched movie dataset.

    Returns:
        tuple: A tuple containing:
            - scipy.sparse.csr_matrix: The combined, normalized feature matrix.
            - pd.DataFrame: The dataset containing the newly created 'soup' column.
    """
    all_genres = [g for genres in data['genres'] for g in genres if isinstance(genres, list)]
    genre_counts = Counter(all_genres)
    total_movies = len(data)
    genre_weights = {g: np.log(total_movies / (c + 1)) for g, c in genre_counts.items()}
    max_weight = max(genre_weights.values()) if genre_weights else 1.0

    def cook_soup(row):
        keywords = ' '.join(row['keywords']) if isinstance(row['keywords'], list) else ''
        cast = ' '.join(row['cast']) if isinstance(row['cast'], list) else ''
        director = (str(row['director']) + ' ') * 3 if pd.notna(row['director']) else ''
        
        soup = (keywords * 2) + ' ' + cast + ' ' + director
        if isinstance(row['genres'], list):
            for genre in row['genres']:
                weight = genre_weights.get(genre, 1.0)
                repeats = min(int((weight / max_weight) * 4) + 2, 5)
                soup += (genre + ' ') * repeats
        return soup

    data['soup'] = data.apply(cook_soup, axis=1).fillna('')

    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000, min_df=2, max_df=0.8)
    
    tfidf_matrix = tfidf.fit_transform(data['overview'].fillna(''))

    count = CountVectorizer(stop_words='english', max_features=3000, min_df=2)
    count_matrix = count.fit_transform(data['soup'])

    cols_to_scale = ['log_movie_age', 'runtime', 'vote_average']
    num_features = data[cols_to_scale].copy()
    
    for col in num_features.columns:
        num_features[col] = num_features[col].fillna(num_features[col].median())

    num_features_scaled = MinMaxScaler().fit_transform(num_features)
    num_features_sparse = csr_matrix(num_features_scaled)
        
    combined_matrix = normalize(hstack([
        tfidf_matrix * 2.5, 
        count_matrix * 1.5, 
        num_features_sparse * 0.2
    ]))
    
    return combined_matrix, data