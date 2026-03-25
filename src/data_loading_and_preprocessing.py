import pandas as pd
import os
import ast

def load_and_merge_data(movies_path, credits_path, ratings_path=None, movies_from_rate_path=None):
    """
    Loads movie and credit datasets from specified file paths, merges them, 
    and performs initial data cleaning and preprocessing.

    This function handles missing values, corrects specific erroneous movie entries, 
    parses JSON-like string columns into Python lists/dictionaries, extracts key personnel 
    (like the director and top 5 cast members), and formats financial and date columns.

    Args:
        movies_path (str): File path to the main movies CSV dataset.
        credits_path (str): File path to the movie credits CSV dataset.
        ratings_path (str, optional): File path to the user ratings CSV dataset. Defaults to None.
        movies_from_rate_path (str, optional): File path to the secondary movies CSV dataset 
            associated with ratings. Defaults to None.

    Returns:
        pd.DataFrame: A merged and preprocessed pandas DataFrame containing the cleaned movie data.

    Raises:
        FileNotFoundError: If the files specified by `movies_path` or `credits_path` do not exist.
    """
    if not os.path.exists(movies_path) or not os.path.exists(credits_path):
        raise FileNotFoundError(f"Files not found: {movies_path} or {credits_path}")

    print("Loading data...")
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    ratings = pd.read_csv(ratings_path) if ratings_path else None
    movies_from_rate = pd.read_csv(movies_from_rate_path) if movies_from_rate_path else None

    print("Merging data...")
    data = movies.merge(credits, left_on='id', right_on='movie_id', how='inner')
    
    if 'title_x' in data.columns and 'title_y' in data.columns:
        data = data.drop(['title_y'], axis=1)
        data = data.rename(columns={'title_x': 'title'})

    if 'movie_id' in data.columns and 'id' in data.columns:
        data = data.drop(['movie_id'], axis=1)

    data = data.drop(['homepage'], axis=1, errors='ignore') 
    
    data['tagline'] = data['tagline'].fillna('')
    data['overview'] = data['overview'].fillna('')

    data.loc[data['original_title'] == 'Food Chains', 'overview'] = (
        "Food Chains is a powerful documentary exposing the dark realities of the US agricultural industry. "
        "It highlights the systemic exploitation of farmworkers and follows the Coalition of Immokalee Workers in Florida. "
        "These intrepid tomato pickers challenge multinational supermarket giants, demanding fair wages and humane "
        "working conditions through their innovative Fair Food Program"
    )

    data.loc[data['original_title'] == 'Chiamatemi Francesco - Il Papa della gente', 'overview'] = (
        "Call Me Francesco - The People's Pope is a biographical drama chronicling the life of Jorge Mario Bergoglio. "
        "The film traces his journey from his early years in Buenos Aires, navigating the dangerous era of Argentina's "
        "military dictatorship, to his profound spiritual evolution and ultimate election as Pope Francis"
    )

    data.loc[data['original_title'] == 'To Be Frank, Sinatra at 100', 'overview'] = (
        "To Be Frank: Sinatra at 100 is a retrospective documentary celebrating the legendary crooner's centenary. "
        "Directed by Simon Napier-Bell, it chronicles Frank Sinatra's remarkable life—from his New Jersey roots to "
        "his iconic music and film career—exploring his personal loyalties, political allegiances, and enduring legacy "
        "through intimate interviews and archival footage"
    )

    data.loc[data['original_title'] == 'America Is Still the Place', 'release_date'] = '2014-11-01'
    data.loc[data['original_title'] == 'Chiamatemi Francesco - Il Papa della gente', 'runtime'] = 98.0
    data.loc[data['original_title'] == 'To Be Frank, Sinatra at 100', 'runtime'] = 81.0

    dict_columns = [
        'genres', 'keywords', 'production_companies',
        'production_countries', 'spoken_languages', 'cast', 'crew'
    ]

    def parse(x):
        try:
            if isinstance(x, list): return x
            if pd.isna(x) or x == '': return []
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []

    for column in dict_columns:
        if column in data.columns:
            data[column] = data[column].apply(parse)

    def get_boss(x):
        if isinstance(x, list):
            for i in x:
                if isinstance(i, dict) and i.get('job') == 'Director':
                    return i.get('name', '')
        return ''

    def get_5_cast(x, n=5):
        if isinstance(x, list):
            return [i.get('name') for i in x[:n] if isinstance(i, dict)]
        return []

    if 'crew' in data.columns:
        data['director'] = data['crew'].apply(get_boss)
    if 'cast' in data.columns:
        data['actors_top5'] = data['cast'].apply(get_5_cast)

    def get_list_names(x):
        if isinstance(x, list):
            return [i.get('name') for i in x if isinstance(i, dict) and 'name' in i]
        return []
    
    for column in dict_columns:
        if column in data.columns:
            data[column] = data[column].apply(get_list_names)

    if 'budget' in data.columns:
        data['budget_formatted'] = data['budget'].apply(lambda x: f"{x:,}")
    if 'revenue' in data.columns:
        data['revenue_formatted'] = data['revenue'].apply(lambda x: f"{x:,}")
        
    data.drop(['budget', 'revenue'], axis=1, inplace=True, errors='ignore')

    data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
    data['release_year'] = data['release_date'].dt.year.astype('Int64')

    return data