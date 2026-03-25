# 🎬 Content-Based Movie Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn)

**[🚀 Live Demo](https://content-based-recommendation-system-for-movies-mr9raxrqxyopdb9.streamlit.app) • [📦 Models on Hugging Face](https://huggingface.co/Przemsonn/Recommendation_System)**

</div>

---

## 📌 Project Overview

In the era of streaming wars, users often face **"analysis paralysis"** due to content overload. This project implements a robust movie recommendation engine designed to surface relevant content by analyzing semantic similarities in movie metadata.

Unlike simple tag-matching systems, this engine employs **Hybrid Logic** that balances semantic relevance, movie quality, popularity trends, and release recency. The result is a personalized discovery experience that mitigates common pitfalls of standard recommendation algorithms.

### 🎯 Key Features

- 🧠 **Hybrid Intelligence**: Combines NLP-based semantic similarity with quality, popularity, and recency signals
- 🎲 **Cold Start Solution**: Curated baseline model for new users with no viewing history
- ⚖️ **Balanced Discovery**: Surfaces both mainstream hits and hidden gems while maintaining quality standards
- 📊 **Data-Driven Design**: Validated through external testing with measurable performance metrics

---

## 🚀 Live Demo & Resources

| Platform | Link | Description |
| :--- | :--- | :--- |
| **Streamlit App** | [![Streamlit](https://img.shields.io/badge/Launch-App-FF4B4B)](https://content-based-recommendation-system-for-movies-mr9raxrqxyopdb9.streamlit.app) | Interactive dashboard with 4,800+ movies |
| **Hugging Face** | [![Hugging Face](https://img.shields.io/badge/Models-Registry-yellow)](https://huggingface.co/Przemsonn/Recommendation_System) | Pre-trained models and vectorizers |

---

## ⚙️ Methodology & Data Pipeline

### 1. Data Loading & Preprocessing

Data was sourced from TMDB datasets containing comprehensive information about movies, cast, and crew. Tables were merged using the `movieId` column as the primary key.

Data quality and consistency were critical at this stage. We performed rigorous cleaning and parsing, including the transformation of complex stringified lists (e.g., `['Science', 'Fiction']`) into single tokens such as `sciencefiction`. This sanitization step was essential for the NLP model, ensuring that multi-word genres and names were treated as unique tokens and preventing semantic fragmentation.

**Key operations included:**

- **Imputation**: Missing values were handled through manual research. Since most columns had only 2-3 missing values, I investigated each film individually and filled gaps based on external sources.
- **Pruning**: Irrelevant or low-signal features were removed to reduce noise.
- **Standardization & Cleaning**: Columns such as *budget* and *revenue* were cleaned and reformatted for improved readability and consistency.
- **Feature Engineering**:
  - Extracted release year for downstream analysis
  - Renamed selected columns for improved interpretability
  - Created custom extraction functions:
    - `get_boss()` – extracts the director from crew data
    - `get_5_cast()` – retrieves top 5 billed actors
    - Genre concatenation to create single, NLP-friendly tokens

---

### 2. Exploratory Data Analysis (EDA)

EDA served as a decision-making tool rather than mere visualization. The analysis directly informed feature selection, transformation strategies, and the design of the recommendation logic.

#### Distribution of Movie Features

![Distribution of Features](images/Distribution_of_Movie_Features.png)

**Key insights:**

- **Vote Average**: Concentrated between 5.5 and 7.5. Extreme values (0 or 10) often reflected sparse voting, motivating the use of Bayesian-weighted ratings.
- **Popularity**: Exhibits a strong long-tail distribution. Logarithmic scaling was applied to prevent blockbuster bias while preserving discoverability of lesser-known films.
- **Runtime**: Median runtimes fall between 90–120 minutes. Extreme outliers (<15 or >200 minutes) were filtered as data errors or short films.
- **Vote Count**: Highly right-skewed. Normalization was required to balance confidence and rating reliability.

#### Genre Popularity vs. Quality

One key finding: **genre popularity does not correlate with higher ratings**.

- Less frequent genres (*War*, *History*) consistently achieved higher average ratings
- Popular genres like *Comedy* were not always well-rated despite dominance in the dataset

![20 Most Popular Genres](images/20_Most_Popular_Genres.png)
![Average Movie Ratings by Genre](images/Average_Movie_Ratings_by_Genre.png)

The **Genre Co-occurrence Matrix** visualizes how frequently pairs of genres appear together. Diagonal values represent the total number of films per genre (Drama: 2,297; Comedy: 1,722). High co-occurrence pairs like Drama–Comedy and Action–Thriller indicate compatible genres frequently blended in film production, while rare combinations (Animation–Thriller, War–Romance) reveal stylistic incompatibilities.

**Modeling Implication**: High co-occurrence values indicate strong shared signal. Users who enjoy Action are statistically likely to appreciate Thriller—a relationship the hybrid scoring function exploits through similarity weighting.

#### Correlation Matrix Analysis

![Correlation Matrix](images/Correlation_Matrix_Numerical_Features.png)

**Key relationships:**

- **Popularity ↔ Vote Count (0.78)**: Strongest correlation. User engagement drives popularity scores, motivating logarithmic scaling to prevent redundancy.
- **Runtime ↔ Vote Average (0.38)**: Moderate positive correlation. Longer films tend to attract slightly higher ratings, possibly due to audience self-selection.
- **Release Year**: Weak negative correlations with vote_average (-0.20) and runtime (-0.17). Newer films are marginally shorter and rated slightly lower—likely influenced by recency bias in voting.

#### Popularity vs. Rating

![Popularity_vs_Rating](images/Popularity_vs_Rating.png)

The scatter plot reveals a weak positive correlation (r ≈ 0.3), confirming a fundamental insight: **popularity is not a reliable proxy for quality**.

Two distinct patterns emerge:
1. **Overhyped Titles**: High popularity but ratings below 6.0
2. **Hidden Gems**: Low popularity but ratings above 7.5

To address this, the recommendation engine uses a **hybrid quality score** combining rating and popularity in a single weighted metric with logarithmic scaling and Bayesian-weighted ratings.

---

### 3. Feature Engineering

#### Core Engineered Features

1. **`weighted_rating` (Bayesian Average)**  
   Adjusts a movie's rating toward the global mean when vote count is low. As votes increase, the movie's own rating exerts greater influence.
   
   **Result**: Promotes films combining high ratings with sufficient audience validation while preventing little-known movies with few votes from dominating rankings.

2. **`movie_age` (Years Since Release)**  
   Release year transformed into *movie_age* with logarithmic scaling to reduce temporal skew and stabilize model behavior.

3. **`tagline_integration` (Text Enrichment)**  
   Taglines concatenated with overviews to enrich textual context for NLP processing, resulting in more informative and consistent representations.

4. **`logarithmic_transformation`**  
   Both popularity and vote count follow right-skewed long-tail distributions. Without transformation, outliers would dominate distance calculations and introduce systematic bias toward blockbusters. Logarithmic scaling (using `log1p`) compresses the extreme upper tail while preserving relative ordering.

5. **`runtime_median`**  
   Films with runtime <15 minutes are treated as erroneous entries (trailers, shorts, or data errors). These are replaced with the dataset median runtime.

#### Quality Score

A composite weighted metric was defined:

**Quality Score = 0.6 × Rating + 0.4 × log(Popularity)**

This ensures both **critical reception** and **audience engagement** contribute to final scores, preventing over-recommendation of either obscure highly-rated films or widely popular low-quality content.

---

## 🧠 Model Architecture

### 🛡️ Baseline Model: Cold-Start Recommendation Strategy

**Challenge**: How can we recommend relevant content to users with no prior interaction history?

#### Solution Strategy

A **curated baseline model** delivers statistically reliable recommendations without personalization. Instead of simple averages—which are highly sensitive to sparse data—the model uses a **Bayesian Weighted Quality Score** to penalize low-confidence ratings.

#### Implementation Details

- Filtered movies with vote counts above the **90th percentile** (ensures rating reliability)
- Ranked candidates using **Bayesian weighted ratings**
- Applied **temporal preference** (films released after 1990) to align with dataset distribution
- Enforced **genre diversity** across Action, Drama, Sci-Fi, and Thriller

#### Performance Validation

![Popularity vs Quality](images/Popularity_vs_Quality_Top10_Baseline_Recommendations.png)

Red points (recommended films) cluster in the **upper-right quadrant**, representing optimal balance between high popularity and high quality. This confirms baseline recommendations are "safe bets" for new users.

#### Baseline Model Results

- **Average Rating**: 8.24 / 10 (vs. dataset average ~6.0)
- **Genre Coverage**: 10 primary genres represented
- **Most Common Genre**: Drama
- **Temporal Distribution**: 80% released between 1990–2014
- **Average Release Year**: 1996

---

### 🚀 Main Model: Hybrid Recommendation Engine

The main model leverages **Natural Language Processing (NLP)** to capture semantic relationships and generate personalized recommendations.

#### Step 1: NLP Processing – Metadata "Soup" Creation

A composite textual representation was constructed by concatenating:
- **Keywords** (themes and plot elements)
- **Cast** (top 5 billed actors)
- **Director**

This enriched representation captures both narrative and production-level similarities.

#### Step 2: Vectorization

**TF-IDF Vectorization**:
- **Downweights** overly common terms (*Action*, *Drama*)
- **Emphasizes** distinctive descriptors (*cyberpunk*, *neo-noir*)
- Enables semantic differentiation (e.g., *Space Horror* vs. *Space Comedy*)
- Produces ~10,000 sparse features per movie

**CountVectorizer**:
- Applied to categorical text features (keywords, cast, director)
- Captures explicit term frequency signals
- Removes English stop words

This dual approach combines **semantic weighting (TF-IDF)** with **explicit frequency signals (CountVectorizer)**.

#### Step 3: Similarity Computation

Movie similarity uses **Cosine Similarity** over TF-IDF vectors:

```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Scores range from **0 to 1**:
- **1.0** – nearly identical metadata
- **0.5** – moderate thematic overlap
- **0.0** – no meaningful similarity

#### Step 4: Recommendation Logic – Key Design Insights

The recommendation function integrates several critical refinements:

1. Identify candidate movies based on **cosine similarity**
2. Apply **logarithmic scaling and popularity penalties** to prevent blockbuster dominance
3. Introduce **age-based penalty** to promote temporal diversity
4. Compute **combined score** incorporating similarity and quality metrics
5. Apply **MMR (Maximal Marginal Relevance)** algorithm to balance accuracy and diversity

**Without MMR**: Potentially 10 recommendations with same actor/director/sub-genre  
**With MMR**: Same semantic relevance but diverse: different sub-genres, directors, storylines

#### Step 5: Hybrid Ranking Strategy

Final recommendations are ranked using:

**Final Score = α × Quality_norm + (1 − α) × Similarity_norm**

Where:
- **Similarity**: Cosine similarity between input movie and candidate
- **Quality**: Popularity-adjusted rating score
- **α**: Tunable parameter (0.3 = semantic-first, 0.5 = balanced, 0.7 = quality-first)

---

## 🔧 Parameter Tuning & Simulation

### Alpha Parameter Optimization

The α parameter was tuned to find the optimal relevance-quality balance:
- **α = 0.3** → 70% similarity, 30% quality
- **α = 0.5** → Balanced approach
- **α = 0.7** → Quality-driven recommendations

### Example Calculation

**Input**: "The Martian" | **Candidate**: "Interstellar" | **α = 0.3**

```
Similarity_norm = 0.85
Quality_norm = 0.82
Final Score = 0.3 × 0.82 + 0.7 × 0.85 = 0.841 → 84% Match
```

![Second_model](images/Plot_Results_for_Second_Model.png)

**Quality Analysis**: Average ratings peak around 6.8 (above dataset average of 6.1), indicating the model selects higher-quality films.

**Diversity Zones**: Scores concentrate between 0.6–0.8, demonstrating strong diversity while avoiding extreme homogeneity.

**Genre Overlap**: 50–70% median coverage indicates consistent, balanced genre representation.

**Popularity Strategy**: Right-skewed distribution heavily favoring popular content while including occasional niche recommendations (Goldilocks balance).

---

## 📊 Evaluation – Leave-One-Out Methodology

Simulation metrics alone don't validate real-world performance. To address this, an **external Leave-One-Out evaluation** was conducted using the **MovieLens dataset**—verified user ratings entirely independent of TMDB data.

### Methodology

For each sampled user:
1. Select one liked film (rating ≥ 3.5) as the seed
2. Remaining liked films form ground truth relevant set
3. Call recommendation function with seed
4. Evaluate output against ground truth

### Metrics

- **Precision@K**: Fraction of top-K recommendations that were relevant
- **Recall@K**: Fraction of relevant films captured in top-K
- **NDCG@K**: Ranking quality weighted by position
- **Hit Rate@K**: Whether ≥1 relevant film appeared in top-K

### Results

![Metrics_comparision](images/Evaluation_Metrics_Leave_One_Out.png)

| Metric | @5 mean| @10 mean|
|--------|-----|-----|
| **Hit Rate** | 26% | 39.00% |
| **Precision** | 7.04% | 6.06% |
| **Recall** | 1.00% | 1.57% |
| **NDCG** | 7.43% | 6.62% |

**Key Insights**:
- **Hit Rate@10 = 39%**: In 2 of 5 cases, system surfaces a film the user genuinely liked—without any viewing history
- **Precision@5 = 6.06%**: Relevant films appear early in ranking
- **Recall@10 = 1.57%**: Low by design (4,800 film catalog vs. 10 recommendations); improving would require collaborative filtering

---

## 🎯 Conclusions: Two-Model Architecture

### Baseline Model (Cold Start)
Solves the initial user problem by providing high-quality, popular, trustworthy recommendations. Essential for first impressions and user retention.

**Results**: Average rating 8.24/10, well-distributed across genres, primarily 1990s–2010s films.

### Main Recommendation Engine
Leverages semantic similarity for personalized suggestions. Uses TF-IDF, MMR, weighted quality scores, and optimized alpha tuning for real-world alignment.

**Results**: Hit Rate@10 of 39% on external MovieLens data—competitive for pure content-based approaches without collaborative signals.

**Combined Strengths**:
- ✅ Cold-start capable (baseline model)
- ✅ Semantically informed (NLP processing)
- ✅ Diversity-aware (MMR algorithm)
- ✅ Quality-filtered (Bayesian weighting)
- ✅ Externally validated (MovieLens evaluation)

---

## 📱 Application Interface

The project is deployed as an interactive **Streamlit** web application for intuitive movie discovery.

### Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit (Python reactive UI) |
| **Backend** | Scikit-Learn (TF-IDF, CountVectorizer, cosine similarity) |
| **Data Processing** | Pandas, NumPy, SciPy |
| **Visualization** | Matplotlib, Seaborn |
| **Poster Fetching** | TMDB API |
| **Deployment** | Streamlit Cloud |
| **Model Storage** | Hugging Face Hub |

### User Interface Walkthrough

#### Home Page

![Interface 1](images/st_interface1.png)

Two main cards allow users to choose:
- **Right Card**: Baseline model recommendations (static, high-quality defaults)
- **Left Card**: Intelligent model (user enters movie, receives personalized top-10)

Below: Comprehensive theory and efficiency metrics.

#### Baseline Recommendations

![Interface 2](images/st_interface2.png)

- **Slider**: Display 1–20 movies
- **Genre Filter**: Select primary genre
- **Movie Details**: Year, rating, vote count
- **Hover Interactions**: Display poster and synopsis

#### Intelligent Model Recommendations

**Example**: User enters "Cars" to discover similar films

![Interface 3](images/st_interface3.png)
![Interface 4](images/st_interface4.png)

**For Each Recommendation**:
- **Match %**: Hybrid score (0–100%) showing recommendation confidence
- **Poster**: Dynamically fetched from TMDB API
- **Title & Rating**: Clear identification with Bayesian-weighted score
- **Genres**: Top 2–3 relevant tags
- **Details Button**: Expands to show full synopsis and metadata

---

## 💡 Key Learnings & Future Work

### Technical Discoveries

**What Worked:**

1. **"Metadata Soup" Approach**  
   Combining cast, crew, keywords, and genres created richer semantic space than individual features. **Lesson**: Context matters more than isolated attributes.

2. **TF-IDF Superiority**  
   Significantly outperformed keyword matching. Generic terms automatically downweighted. **Lesson**: Smart feature engineering > complex algorithms.

3. **Bayesian Weighting**  
   Essential for handling confidence variance. Prevents statistical outliers from skewing rankings. **Lesson**: Always account for uncertainty.

4. **MMR Algorithm**  
   Balances relevance with diversity. Without it: 10 similar recommendations. With it: Diverse films maintaining semantic relevance. **Lesson**: Diversity ≠ randomness.

5. **Alpha Parameter Tuning**  
   Automated optimization improved accuracy and robustness. **Lesson**: Hyperparameter search pays dividends.

### Future Improvements

#### 🔮 Short-Term Enhancements

- **Explainability Dashboard**: "Why this recommendation?" with score component breakdown
- **User Feedback Loop**: Thumbs up/down to retrain weights iteratively
- **A/B Testing Framework**: Experiment with different weight configurations

#### 🚀 Long-Term Roadmap

1. **GenAI Integration**
   - Upgrade to **BERT** or **Sentence Transformers** for deeper semantic understanding
   - Expected: Better handling of synonyms and abstract themes

2. **Conversational Interface**
   - Natural language queries: *"I want a sad movie about robots with a hopeful ending"*
   - Powered by **GPT-4** or **Llama 3**

3. **Collaborative Filtering Hybrid**
   - Combine content-based with user behavior signals
   - Requires user accounts and watch history

4. **Production Deployment**
   - Scale from Streamlit Cloud to **AWS ECS** / **Google Cloud Run**

---

## 🛠️ Installation

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Content-Based-Recommendation-System-for-movies.git

# 2. Navigate to directory
cd Content-Based-Recommendation-System-for-movies

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Download pre-trained models from Hugging Face
# Models automatically loaded on first run, or manually download from:
# https://huggingface.co/Przemsonn/Recommendation_System

# 5. Run the analysis notebook (optional)
jupyter notebook Recommendation_System.ipynb

# 6. Launch Streamlit app locally
streamlit run app.py
```

### Project Structure

```
Content-Based-Recommendation-System-for-movies/
├── data/
│   ├── tmdb_5000_credits.csv          # Raw cast/crew data
│   ├── tmdb_5000_movies.csv           # Raw movie metadata
│   ├── movies.csv                     # Processed movie data
│   └── ratings.csv                    # MovieLens ratings (evaluation)
├── images/                            # EDA visualizations & UI screenshots
├── notebooks/
│   └── Recommendation_System.ipynb     # Full analysis & model development
├── results/
│   └── evaluation_summary.csv          # Leave-One-Out results
├── src/                               # Source code modules
├── app.py                             # Streamlit application entry point
├── main.py                            # Main recommendation pipeline
├── requirements.txt                   # Python dependencies
├── .gitignore
├── LICENSE
└── README.md                          # This file
```

---

## 📦 Dependencies

Key packages:
- `streamlit` – Web application framework
- `scikit-learn` – Machine learning & vectorization
- `pandas` – Data manipulation
- `numpy` – Numerical computing
- `requests` – TMDB API calls
- `python-dotenv` – Environment variable management

See `requirements.txt` for complete list and versions.

---

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Improved similarity metrics
- Additional data sources
- UI/UX improvements
- Performance optimization
- Documentation expansion

---

## 📄 License

This project is licensed under the **MIT License**. See `LICENSE` file for details.

---

## 🙏 Acknowledgments

- **TMDB**: Comprehensive movie database and API
- **MovieLens**: Ground truth for evaluation
- **Scikit-Learn**: Robust machine learning tools
- **Streamlit**: Accessible deployment framework
- **Open Source Community**: Invaluable resources and support

---

<div align="center">

**⭐ Found this helpful? Please star the repository!**

[![GitHub stars](https://img.shields.io/github/stars/Przemsonn/Content-Based-Recommendation-System-for-movies?style=social)](https://github.com/Przemsonn/Content-Based-Recommendation-System-for-movies)

**Made by [Przemsonn](https://github.com/Przemsonn)**

</div>