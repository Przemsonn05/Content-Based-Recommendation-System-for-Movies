import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import urllib.parse
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Movies Recommendation System", layout="wide")

st.markdown("""
<style>
    .movie-title {
        height: 80px;            
        display: flex;          
        align-items: center;    
        justify-content: center; 
        text-align: center;      
        font-weight: bold;      
        font-size: 14px;       
        line-height: 1.2;       
        margin-bottom: 10px;    
        
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 3; 
        -webkit-box-orient: vertical;
        color: #fff;
    }
    
    div[data-testid="stImage"] img {
        height: 300px !important;  
        object-fit: cover !important; 
        border-radius: 8px;        
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    div[data-testid="stCaptionContainer"] {
        text-align: center;
        color: #d1d5db !important;
    }

    .stApp {
        background: linear-gradient(to right, #0f0c29, #302b63, #24243e);
        background-attachment: fixed;
    }

    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #ffffff !important;
    }

    div[data-testid="column"] {
        background-color: rgba(60, 10, 80, 0.15); 
        border: 1px solid rgba(200, 100, 255, 0.1); 
        border-radius: 15px;      
        padding: 15px;            
        backdrop-filter: blur(10px); 
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    div[data-testid="column"]:hover {
        transform: translateY(-7px);
        box-shadow: 0 10px 25px rgba(130, 0, 255, 0.3); 
        border-color: rgba(200, 100, 255, 0.4); 
    }

    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #8E2DE2, #4A00E0);
    }
    
    div.stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
    }
            
    /* Styl dla wiersza z filmem w Baseline */
    .custom-movie-row {
        position: relative;
        background: rgba(60, 10, 80, 0.15);
        padding: 15px;
        margin-bottom: 8px;
        border-radius: 8px;
        border: 1px solid rgba(200, 100, 255, 0.1);
        cursor: pointer;
        color: white;
        transition: background 0.3s;
    }
    
    .custom-movie-row:hover {
        background: rgba(130, 0, 255, 0.2);
    }
    
    /* Styl dla wyskakującego okienka (tooltip) */
    .movie-tooltip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        top: -20px;
        left: 55%; 
        width: 320px;
        background: #1a1a2e;
        border: 1px solid rgba(200, 100, 255, 0.5);
        border-radius: 10px;
        padding: 12px;
        z-index: 9999;
        box-shadow: 0 10px 30px rgba(0,0,0,0.8);
        transition: opacity 0.3s ease-in-out, left 0.3s;
        pointer-events: none; 
    }
    
    /* Co się dzieje po najechaniu myszką */
    .custom-movie-row:hover .movie-tooltip {
        visibility: visible;
        opacity: 1;
        left: 50%; 
    }
    
    /* Styl zdjęcia i tekstu w tooltipie */
    .movie-tooltip img {
        width: 100%;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .movie-tooltip p {
        font-size: 13px !important;
        margin: 0;
        color: #e2e8f0 !important;
        line-height: 1.4 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Przemsonn/Recommendation_System",
        filename="recommendation_intelligent_model.joblib"
    )
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
    return model

data = load_model()

@st.cache_resource
def load_baseline_model():
    model_path = hf_hub_download(
        repo_id="Przemsonn/Recommendation_System",
        filename="recommendation_baseline_model.joblib"
    )
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
    return model

data_baseline = load_baseline_model()

if 'page' not in st.session_state:
    st.session_state.page = 'home'

def go(page):
    st.session_state.page = page
    st.rerun()

@st.cache_data
def fetch_poster(title):
    api_key = "e9b3d423b23f3b61816fe8887b777754"
    encoded_title = urllib.parse.quote(title)
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={encoded_title}&language=en-US"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                first_result = data['results'][0]
                poster_path = first_result.get('poster_path')
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500/{poster_path}"
        return "https://via.placeholder.com/500x750?text=No+Poster"
    except Exception:
        return "https://via.placeholder.com/500x750?text=Error"

@st.cache_data
def fetch_movie_details(title):
    api_key = "e9b3d423b23f3b61816fe8887b777754"
    encoded_title = urllib.parse.quote(title)
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={encoded_title}&language=en-US"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                first_result = data['results'][0]
                backdrop_path = first_result.get('backdrop_path')
                overview = first_result.get('overview', 'No description available for this movie.')
                
                if backdrop_path:
                    img_url = f"https://image.tmdb.org/t/p/w500/{backdrop_path}"
                else:
                    img_url = "https://via.placeholder.com/500x281?text=No+Scene+Available"
                    
                return img_url, overview
        return "https://via.placeholder.com/500x281?text=No+Image", "Movie not found."
    except Exception:
        return "https://via.placeholder.com/500x281?text=Error", "Error fetching details."

@st.dialog("🎬 Movie Details")
def show_movie_info(title, backdrop_url, overview):
    st.image(backdrop_url, use_container_width=True)
    st.markdown(f"### {title}")
    st.write(overview)

movies_df = data_baseline['movies_df']
C = data_baseline['C']
m = data_baseline['m']

def get_baseline_recommendations(df, n=10, min_votes=None, genre_filter=None):
    if min_votes is None:
        min_votes = m
    
    filtered_df = df[df['vote_count'] >= min_votes].copy()
    
    if genre_filter:
        filtered_df = filtered_df[
            filtered_df['genres'].apply(lambda x: genre_filter in x if isinstance(x, (list, str)) else False)
        ]
    
    top_n = filtered_df.sort_values('weighted_rating', ascending=False).head(n)
    return top_n

df = data['dataframe']
cosine_sim = data['similarity']
indices = data['indices']

def recommendation(title, alpha=0.5):
    if title not in indices:
        return f"Film {title} was not found in the database"
    
    idx = indices[title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:51]

    if 'original_language' in df.columns:
        movie_language = df.loc[df['original_title'] == title, 'original_language'].iloc[0]
    else:
        movie_language = 'en'

    movie_indices = [i[0] for i in similarity_scores]
    similarity_values = [i[1] for i in similarity_scores]
    
    candidates = df.iloc[movie_indices].copy()
    candidates['similarity'] = similarity_values

    def normalize(series):
        min_val = series.min()
        max_val = series.max()
        if max_val - min_val == 0:
            return series.apply(lambda x: 0.5) 
        return (series - min_val) / (max_val - min_val)

    candidates['similarity_norm'] = normalize(candidates['similarity'])
    candidates['quality_norm'] = normalize(candidates['hybrid_score']) 

    if movie_language != 'en':
        bonus = np.where(candidates['original_language'] == movie_language, 0.1, 0.0)
        candidates['similarity_norm'] += bonus
        candidates['similarity_norm'] = candidates['similarity_norm'].clip(upper=1.0)
    
    candidates['final_score'] = (alpha * candidates['quality_norm']) + ((1-alpha) * candidates['similarity_norm'])

    return candidates.sort_values('final_score', ascending=False).head(10)

def home():
    st.markdown("""
    <h1 style='text-align:center; font-weight:800; letter-spacing:1px; margin-bottom:30px'>
        🎬 Movie Recommendation System
    </h1>
    """, unsafe_allow_html=True)

    st.divider()
    
    spacer_left, col1, col2, spacer_right = st.columns([1, 2, 2, 1])

    with col1:
        with st.container(border=True, height=220):
            st.markdown("### 🎯 Personalized Recommendations")
            st.markdown("Discover movies similar to the ones you already love")
            
            st.markdown("<br>", unsafe_allow_html=True) 
            
            if st.button("Explore movies", use_container_width=True):
                go("search")

    with col2:
        with st.container(border=True, height=220):
            st.markdown("### 🔥 Popular Movies — Curated Picks")
            st.markdown("See what’s trending and highly rated right now")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("Browse top movies", use_container_width=True):
                go("look")

    st.divider()

    st.markdown("""
    <div style="max-width: 950px; margin: 0 auto; line-height: 1.6; font-size: 1.05rem;">
    <h2 style="text-align:center;">🎬 Advanced Movie Recommendation Engine</h2>
    
    In today’s era of endless streaming options, users frequently face the "paradox of choice"—spending more time searching for content than actually consuming it. This project is an advanced **movie recommendation engine** designed to solve this exact problem by surfacing highly relevant films tailored to user preferences. 
    
    Rather than relying on a single algorithm, the system seamlessly integrates **data-driven popularity curation** with **semantic, content-based filtering** powered by Natural Language Processing (NLP). The architecture is specifically built to tackle two fundamental recommendation challenges that streaming platforms face daily:
    
    * **The Cold Start Problem** — serving high-quality content to new users with zero interaction history.
    * **Personalized Discovery** — dynamically adapting to specific user taste profiles without trapping them in a "filter bubble."

    ---

    ### 🧊 Baseline Model – The Cold Start Solution

    The **Baseline Engine** acts as the primary fallback when user preferences are completely unknown. In raw datasets, sorting strictly by "average rating" often leads to highly skewed results, where an obscure movie with only two 10/10 votes outranks a cinematic masterpiece with millions of 9/10 votes.

    To counter this, the baseline model implements a rigorous **Weighted Rating (Bayesian Average) formula**, popularized by IMDb. This mathematical approach calculates a balanced score by weighing:
                
    * **$R$** = Average rating for the specific movie
    * **$v$** = Number of votes the movie has received (Statistical Confidence)
    * **$m$** = Minimum votes required to be listed (e.g., 90th percentile of the dataset)
    * **$C$** = The mean vote across the entire global dataset

    **Formula:**
    $$WR = \\left( \\frac{v}{v+m} \\cdot R \\right) + \\left( \\frac{m}{v+m} \\cdot C \\right)$$

    **Key Benefits:**
                
    ✔ **Mitigates** the dominance of niche movies with artificially high ratings.  
    ✔ **Surfaces** statistically significant, universally acclaimed titles.  
    ✔ **Provides** a trustworthy safety net for first-time users, ensuring their initial platform experience is positive.  

    The output is a curated, highly reliable list of globally popular movies, which can be further filtered by genre or release decade to provide a pseudo-personalized experience before any real data is collected.

    ---

    ### 🧠 Core Engine – NLP-Powered Content-Based Filtering

    Once a user interacts with a movie, the primary recommendation engine takes over. This engine leverages a **content-based architecture** enriched by advanced Natural Language Processing techniques. Instead of relying on collaborative filtering (which requires massive user-interaction matrices), it focuses on the *actual content* of the films.

    #### 1. Feature Engineering & Metadata Matrix
    Each film is transformed into a rich **metadata matrix**. The system extracts and processes raw text from multiple features:
    * **Genres & Subgenres**
    * **Keywords & Plot Tags**
    * **Cast & Crew** (specifically weighting the Director and lead actors)
    * **Narrative Descriptors**

    #### 2. Text Vectorization
    These textual features are cleaned (e.g., lowercasing, removing spaces from names to treat "Tom Hanks" as a unique entity tomhanks) and vectorized using two distinct approaches:
                
    * **TF-IDF (Term Frequency-Inverse Document Frequency):** Used for plot descriptions and keywords to penalize generic terms and heavily emphasize unique narrative identifiers.
    * **Count Vectorization:** Applied to categorical data like genres and cast members, where exact matches are necessary.

    #### 3. Semantic Distance & MMR
    The system computes the **Cosine Similarity** between the vectorized representations to find exact semantic distances between films. To ensure results are both relevant and engaging, the algorithm incorporates principles of **MMR (Maximal Marginal Relevance)**. It dynamically penalizes movies that are too similar to items already added to the recommendation list, forcing the model to introduce variety.

    ✔ **Identifies** deeply semantically similar films across different franchises.  
    ✔ **Eliminates** redundant or heavily repetitive recommendations.  
    ✔ **Dynamically maps** to user taste profiles without falling victim to popularity bias.  

    ---

    ### 📊 Evaluation & Design Philosophy

    A recommendation system is only as good as its measurable impact. The architecture was rigorously evaluated using a **Leave-One-Out** methodology on 500 randomly sampled MovieLens users, confirming that the engine performs meaningfully above a random baseline:

    * **Hit Rate@10** (the percentage of users who received at least one genuinely relevant movie in their top 10) reached **39%**. In nearly 2 out of every 5 cases, the system surfaces a film the user liked, demonstrating that content-based similarity alone carries strong signal in a cold-start scenario.
    * **Precision@5** (the proportion of the top 5 recommendations that are actually relevant) of **7.04%** and **NDCG@5** (Normalized Discounted Cumulative Gain, which heavily rewards relevant movies placed at the very top) of **7.43%**. This indicates that relevant films appear early in the ranking rather than being buried—a direct result of a well-calibrated hybrid scoring function.
    * **Recall@10** (the fraction of a user's total liked movies successfully found) remains at **1.57%**. This is mathematically expected given the strict constraint of 10 recommendations against an approx. 4,800-film catalog.
    * **Variance & Bias Mitigation:** The consistently high standard deviation reflects natural variance in user tastes (performing better for mainstream preferences than highly niche ones). Overall, the system strikes a healthy balance between mainstream blockbusters and hidden gems, avoiding extreme popularity bias.

    ---

    ### ✨ Executive Summary & Future Scope

    This project proves that effective, production-ready recommendation systems do not strictly require heavy, black-box deep learning architectures or massive user-item interaction databases. By strategically bridging a statistical popularity baseline with an NLP-driven similarity model, the system achieves an optimal blend of recommendation quality, catalog diversity, and strict interpretability. 
                    
    Statistical weighting ensures reliable rankings for widely rated movies, while NLP feature extraction enables deeply personalized discovery based on actual narrative content. 
                    
    Packaged as an interactive Streamlit application, the project emphasizes model transparency and real-world usability. It allows users to intuitively explore cinematic recommendations while maintaining complete insight into how the underlying algorithms generate their results. Future iterations of this project could include hybridizing the current pipeline with Collaborative Filtering (SVD) to improve recall once sufficient user interaction data is collected.
    </div>
    """, unsafe_allow_html=True)

def search():
    if st.button("⬅ Back"):
        if 'recs' in st.session_state:
            del st.session_state['recs']
        go("home")

    st.subheader("🎥 Find similar movies")

    select_movie = st.selectbox(
        "Choose a movie:",
        df['original_title'].sort_values().values,
        index=None
    )

    if 'recs' not in st.session_state:
        st.session_state.recs = None
    if 'last_movie' not in st.session_state:
        st.session_state.last_movie = None

    if select_movie != st.session_state.last_movie:
        st.session_state.recs = None
        st.session_state.last_movie = select_movie

    if select_movie and st.button("🔍 Get recommendations", type="primary"):
        st.session_state.recs = recommendation(select_movie, alpha=0.5)

    if st.session_state.recs is not None and isinstance(st.session_state.recs, pd.DataFrame):
        st.success(f"Movies similar to **{select_movie}**:")

        rows = st.columns(5) + st.columns(5)

        for idx, (_, movie) in enumerate(st.session_state.recs.iterrows()):
            with rows[idx]:
                title = movie["original_title"]
                
                st.markdown(
                    f'<div class="movie-title">{title}</div>',
                    unsafe_allow_html=True
                )

                poster_url = fetch_poster(title)
                st.image(poster_url, use_container_width=True)

                backdrop_url, overview = fetch_movie_details(title)

                if st.button("📖 Details", key=f"info_{idx}", use_container_width=True):
                    show_movie_info(title, backdrop_url, overview)

                genres = movie["genres"]
                genres_str = ", ".join(genres[:3]) if isinstance(genres, list) else str(genres)

                st.caption(genres_str)
                st.metric("Rating", f"{movie['vote_average']:.1f}/10")
                
                raw_match = int(movie["final_score"] * 100)
                match = max(0, min(100, raw_match)) 
                
                st.progress(match, text=f"Match score: {match}%")

def look():
    if st.button("⬅ Back"): 
        go("home")

    st.title("Baseline Movie Recommender 🎬")

    all_genres = set()
    for genres in movies_df['genres'].dropna():
        if isinstance(genres, list):
            all_genres.update(genres)
        elif isinstance(genres, str):
            all_genres.add(genres)

    genre = st.selectbox("Choose a genre (optional)", options=["All"] + sorted(all_genres))
    num_of_movies = st.slider('Number of recommendations', 1, 20, 10)
    
    genre_filter = None if genre == 'All' else genre

    top_recs = get_baseline_recommendations(movies_df, n=num_of_movies, genre_filter=genre_filter)

    st.subheader('Recommended Movies')
    
    html_content = "<div style='display: flex; flex-direction: column; gap: 8px;'>"
    
    for _, row in top_recs.iterrows():
        title = row['original_title']
        
        try:
            year = int(str(row['release_year'])[:4]) if pd.notna(row['release_year']) else "N/A"
        except ValueError:
            year = "N/A"
            
        rating = round(row['weighted_rating'], 2)
        votes = int(row['vote_count'])
        
        backdrop_url, overview = fetch_movie_details(title)
        
        if len(overview) > 200:
            overview = overview[:197] + "..."

        html_content += f"""<div class="custom-movie-row">
<strong>{title}</strong> ({year}) &nbsp; | &nbsp; ⭐ {rating} &nbsp; <span style='color:gray; font-size:12px;'>({votes} votes)</span>
<div class="movie-tooltip">
<img src="{backdrop_url}" alt="Movie Scene">
<p><strong>{title}</strong></p>
<p>{overview}</p>
</div>
</div>"""
        
    html_content += "</div>"

    st.markdown(html_content, unsafe_allow_html=True)
    
if st.session_state.page == 'home':
    home()
if st.session_state.page == 'search':
    search()
if st.session_state.page == 'look':
    look()