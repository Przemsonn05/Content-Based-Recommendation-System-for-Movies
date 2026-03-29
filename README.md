# 🎬 Content-Based Movie Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)

**[🚀 Live Demo](https://content-based-recommendation-system-for-movies-mr9raxrqxyopdb9.streamlit.app) • [📦 Models on Hugging Face](https://huggingface.co/Przemsonn/Recommendation_System)**

</div>

---

## 📌 Project Overview

In the era of streaming wars, users often face **"analysis paralysis"** due to content overload. This project implements a robust movie recommendation engine designed to surface relevant content by analyzing semantic similarities in movie metadata.

Unlike simple tag-matching systems, this engine employs **Hybrid Logic** that balances semantic relevance, movie quality, popularity trends, and release recency. The result is a personalized discovery experience that mitigates common pitfalls of standard recommendation algorithms - most notably, the tendency to recommend only popular blockbusters or to suggest films so similar to the seed that the user learns nothing new.

### 🎯 Key Features

- 🧠 **Hybrid Intelligence**: Combines NLP-based semantic similarity with quality, popularity, and recency signals
- 🎲 **Cold Start Solution**: Curated baseline model for new users with no viewing history
- ⚖️ **Balanced Discovery**: Surfaces both mainstream hits and hidden gems while maintaining quality standards
- 📊 **Data-Driven Design**: Validated through external Leave-One-Out evaluation on MovieLens with measurable performance metrics

---

## 🚀 Live Demo & Resources

| Platform | Link | Description |
| :--- | :--- | :--- |
| **Streamlit App** | [![Streamlit](https://img.shields.io/badge/Launch-App-FF4B4B)](https://content-based-recommendation-system-for-movies-mr9raxrqxyopdb9.streamlit.app) | Interactive dashboard with 4,800+ movies |
| **Hugging Face** | [![Hugging Face](https://img.shields.io/badge/Models-Registry-yellow)](https://huggingface.co/Przemsonn/Recommendation_System) | Pre-trained models and vectorizers |

---

## 📚 Table of Contents

1. [Dataset & Project Structure](#-dataset--project-structure)
2. [Methodology & Data Pipeline](#️-methodology--data-pipeline)
3. [Exploratory Data Analysis](#2-exploratory-data-analysis-eda)
4. [Feature Engineering](#3-feature-engineering)
5. [Model Architecture](#-model-architecture)
6. [Performance & Evaluation](#-performance--evaluation)
7. [Application Interface](#-application-interface)
8. [Installation & Usage](#️-installation)
9. [Docker](#-docker)
10. [Key Learnings & Future Work](#-key-learnings--future-work)

---

## 📁 Dataset & Project Structure

Data was sourced from two TMDB datasets containing comprehensive information about 4,800+ movies, including cast, crew, genres, keywords, and overviews. Tables were merged using `movieId` as the primary key.

```
Content-Based-Recommendation-System-for-Movies/
├── data/
│   ├── tmdb_5000_credits.csv        # Raw cast/crew data
│   ├── tmdb_5000_movies.csv         # Raw movie metadata
│   ├── movies.csv                   # MovieLens movies (evaluation)
│   └── ratings.csv                  # MovieLens ratings (evaluation)
├── images/                          # EDA visualizations & UI screenshots
├── notebooks/
│   └── Recommendation_System.ipynb  # Full analysis & model development
├── results/
│   └── evaluation_summary.csv       # Leave-One-Out results
├── src/                             # Source code modules
├── app.py                           # Streamlit application entry point
├── main.py                          # Main recommendation pipeline
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Methodology & Data Pipeline

### 1. Data Loading & Preprocessing

Data quality and consistency were critical at this stage, as downstream NLP models are highly sensitive to how text is represented. A key challenge was that several important columns (genres, cast, crew, keywords) were stored as stringified Python lists - for example, `"[{'id': 18, 'name': 'Drama'}]"` rather than a clean value. These had to be parsed and flattened before they could be used.

A non-obvious but important design decision was the **single-token concatenation** of multi-word entities. Genres like `Science Fiction` and names like `Christopher Nolan` were collapsed into `sciencefiction` and `christophernolan`. This prevents TF-IDF from treating them as two independent words - without this step, `fiction` would match any film with science-fiction or political-fiction content indiscriminately, fragmenting the semantic signal.

**Key preprocessing operations:**

- **Imputation**: Missing values were handled through manual research rather than automated fill strategies. Since most columns had only 2–3 missing values, each affected film was investigated individually and gaps were filled based on external sources - a deliberate trade-off that preserved data integrity over scalability.
- **Pruning**: Irrelevant or low-signal features (homepage, spoken languages, production companies) were removed to reduce noise in the metadata soup.
- **Standardization & Cleaning**: Columns such as `budget` and `revenue` were cleaned and reformatted for readability. Films with a budget or revenue of zero were treated as missing rather than as genuinely zero-budget productions.
- **Feature Engineering**:
  - Extracted release year from the `release_date` column for downstream recency modeling
  - Renamed selected columns for improved interpretability across the pipeline
  - `get_boss()` - custom extraction function for the director from nested crew metadata
  - `get_5_cast()` - retrieves the top 5 billed actors, limiting cast signal to the most recognizable contributors
  - Genre concatenation into single NLP-friendly tokens to prevent semantic fragmentation

---

### 2. Exploratory Data Analysis (EDA)

EDA served as a decision-making tool rather than mere visualization. Every chart below directly informed a feature engineering decision or a design choice in the recommendation logic.

#### Distribution of Movie Features

![Distribution of Features](images/Distribution_of_Movie_Features.png)

The four distribution plots above reveal structural properties of the dataset that would have caused systematic bias if left unaddressed.

**Key insights and downstream decisions:**

- **Vote Average**: Concentrated between 5.5 and 7.5 with a roughly normal shape - reassuring for quality scoring. However, extreme values at 0 or 10 represented films with only few votes, not genuinely exceptional or terrible movies. This motivated the use of **Bayesian-weighted ratings** that shrink extreme scores toward the global mean when vote count is low.
- **Popularity**: Exhibits a strong long-tail distribution - a handful of blockbusters accumulate popularity scores orders of magnitude higher than the median. Without intervention, any similarity or ranking function would collapse into recommending the same 50 widely-known films. **Logarithmic scaling** was applied to compress this tail while preserving the relative ordering of less-popular films.
- **Runtime**: Median runtimes fall between 90–120 minutes, consistent with commercial film conventions. Extreme outliers below 15 minutes were identified as trailers, short films, or data errors and replaced with the dataset median rather than removed - preserving the full dataset size.
- **Vote Count**: Highly right-skewed, mirroring the popularity distribution. A film with 10,000 votes carries fundamentally more statistical weight than one with 12 votes, even if both have an average rating of 7.5. Normalization was required to balance confidence against rating value in the quality scoring formula.

---

#### Genre Popularity vs. Quality

One of the most consequential EDA findings: **genre popularity does not correlate with higher ratings**.

![20 Most Popular Genres](images/20_Most_Popular_Genres.png)

The bar chart above shows that Drama and Comedy dominate the dataset by raw count - together accounting for nearly half of all genre assignments. This volumetric dominance would cause a naïve frequency-based recommender to systematically over-recommend these genres regardless of user preference.

![Average Movie Ratings by Genre](images/Average_Movie_Ratings_by_Genre.png)

The average rating chart tells the opposite story: less frequent genres - War, History, Documentary - consistently achieve higher average ratings. This inverse relationship between popularity and quality is a fundamental challenge for recommendation systems. It means that optimizing purely for similarity to a popular input movie will tend to surface popular but mediocre content, while genuinely high-quality niche films go undiscovered. The hybrid quality score was designed specifically to counteract this bias.

The **Genre Co-occurrence Matrix** reveals how frequently genre pairs appear together in the same film. Diagonal values represent total film counts per genre (Drama: 2,297; Comedy: 1,722). High co-occurrence pairs like Drama–Comedy and Action–Thriller reflect genres that are regularly blended in commercial production, while rare combinations such as Animation–Thriller or War–Romance highlight stylistic incompatibilities that the model implicitly learns from the metadata soup structure.

---

#### Correlation Matrix Analysis

![Correlation Matrix](images/Correlation_Matrix_Numerical_Features.png)

The correlation heatmap above quantifies linear relationships between all numerical features, informing both feature engineering decisions and the design of the hybrid scoring formula.

**Key relationships and their modeling implications:**

- **Popularity ↔ Vote Count (0.78)**: The strongest correlation in the dataset. User engagement directly drives popularity scores - more views generate more votes, which increases the popularity metric. This near-redundancy motivated logarithmic scaling of both features independently rather than using one as a proxy for the other. Keeping both, transformed, preserves distinct signal: popularity captures viral attention while vote count captures sustained engagement.
- **Runtime ↔ Vote Average (0.38)**: A moderate positive correlation - longer films tend to receive slightly higher ratings. The likely explanation is audience self-selection: viewers who commit to a 150-minute runtime are typically more invested in cinema than casual viewers, and their ratings skew higher. This correlation was noted but not directly acted upon, as runtime is not a strong enough quality signal to include in the scoring formula independently.
- **Release Year ↔ Vote Average (−0.20)**: A weak negative correlation. Newer films are rated marginally lower on average, likely due to **recency bias** in voting - recent films have had less time to accumulate votes from audiences who sought them out specifically, while older well-regarded films have been repeatedly discovered and rated by new generations. This finding validated the inclusion of an age-based penalty in the recommendation engine to promote temporal diversity.

---

#### Popularity vs. Rating

![Popularity vs Rating](images/Popularity_vs_Rating.png)

The scatter plot above plots every film in the dataset by its popularity score (x-axis) and vote average (y-axis). The weak positive correlation (r ≈ 0.3) confirms what the genre analysis suggested: **popularity is not a reliable proxy for quality**, and a high-performing recommender cannot treat them as interchangeable.

Two structurally distinct film clusters emerge from this chart and directly shaped the recommendation design:

- **Overhyped Titles** (high popularity, rating below 6.0): Films that generated significant cultural attention but failed to deliver critically. A system that maximizes for popularity would over-recommend from this quadrant.
- **Hidden Gems** (low popularity, rating above 7.5): Well-regarded films with limited mainstream reach. These are exactly the films a recommendation engine should be surfacing - content the user is unlikely to have already seen, but likely to appreciate.

To address this, the recommendation engine uses a **hybrid quality score** that combines rating and popularity signals with logarithmic scaling and Bayesian-weighted ratings, preventing either signal from dominating the other.

---

### 3. Feature Engineering

Feature engineering in this project goes beyond simple column transformations - it encodes domain knowledge about how films are perceived, how popularity is distributed, and how text should be structured for NLP processing.

#### Core Engineered Features

**1. `weighted_rating` - Bayesian Average**

A raw vote average is unreliable when vote count is low: a film with 3 ratings at 10/10 is not genuinely better than a film with 5,000 ratings at 8.2/10. The Bayesian weighted rating adjusts each film's score toward the global mean, with the degree of adjustment inversely proportional to the number of votes:

```
weighted_rating = (v / (v + m)) × R + (m / (v + m)) × C
```

Where `v` is the vote count, `m` is the minimum vote threshold (90th percentile), `R` is the film's own average rating, and `C` is the global mean rating. As `v` grows large relative to `m`, the film's own rating exerts increasing influence. This is the same formula used by IMDb's Top 250 ranking, and its inclusion was essential for making the baseline model statistically meaningful.

**2. `movie_age` - Logarithmic Recency**

Raw release year is replaced by `movie_age` (years since release), then log-transformed. This compresses the difference between very old films (e.g., 60-year-old vs. 70-year-old) while preserving the meaningful distinction between recent releases (2-year-old vs. 5-year-old). Without the log transform, a linear age penalty would severely disadvantage classic cinema regardless of quality - an outcome inconsistent with how film enthusiasts actually consume older titles.

**3. `tagline_integration` - Text Enrichment**

Movie overviews shorter than 100 characters provide insufficient signal for TF-IDF to distinguish films meaningfully. For these cases, the tagline is concatenated to the overview before vectorization. This is a targeted enrichment: it only fires when the overview is genuinely sparse, avoiding redundancy for films with detailed descriptions. The result is more consistent TF-IDF vector quality across the full 4,800-film catalog.

**4. Logarithmic Transformation of Popularity and Vote Count**

Applied independently to both `popularity` and `vote_count` to compress the long-tail distribution identified in EDA. Without transformation, cosine similarity and ranking calculations would be dominated by a handful of films with extreme popularity values - any input movie with moderate popularity would systematically receive recommendations skewed toward blockbusters regardless of semantic content.

**5. `runtime_cleaning` - Outlier Replacement**

Films with runtime below 15 minutes are replaced with the dataset median rather than removed. This preserves the full 4,800-film catalog for the recommendation engine while ensuring that clearly erroneous values (trailers, miscategorized shorts, data entry errors) do not distort the runtime distribution used in downstream filtering.

#### Quality Score Formula

A composite weighted metric combines the two most informative quality signals:

```
Quality Score = 0.6 × Rating_scaled + 0.4 × Popularity_scaled
```

Both features are independently normalized to [0, 1] using Min-Max scaling before combination. The 60/40 weighting deliberately prioritizes critical reception over commercial reach. This ratio was chosen based on the EDA finding that popularity correlates only weakly with quality (r ≈ 0.3) - treating them equally would over-index on commercially successful but critically unremarkable films.

#### Metadata "Soup" Construction

The final engineered feature used for content-based similarity is a single concatenated text string combining all relevant metadata per film, with differential weighting:

- **Keywords** (themes and plot elements) - repeated 2× to boost thematic signal
- **Cast** (top 5 billed actors) - single occurrence
- **Director** - repeated 3× to heavily weight stylistic influence; films by the same director share far more aesthetic DNA than films sharing one actor
- **Genres** - weighted by inverse document frequency (rare genres repeated 2–5×), so that a film's `horror` classification carries more discriminative power than its `drama` classification

This differential weighting is the key design insight of the feature engineering stage: not all metadata is equally informative. A director is a stronger signal than one of five cast members; a rare genre is a stronger signal than a ubiquitous one. The 3× director repetition was validated empirically - removing it caused the model to recommend films that shared actors but had entirely different tonal and stylistic qualities.

---

## 🧠 Model Architecture

### 🛡️ Baseline Model: Cold-Start Recommendation Strategy

**The problem**: How can the system provide useful recommendations to a new user who has provided no input? Returning an empty state or a random selection creates a poor first impression and fails to communicate the system's capabilities.

The baseline model is a **fully independent component** of the recommendation pipeline, activated exclusively when no seed movie has been provided. It operates without any personalization signal, relying entirely on statistical reliability and curated diversity to generate a list of high-quality films that represent the breadth of the catalog.

#### Implementation Details

The baseline selection process applies four sequential filters and rankings:

- **Vote threshold filter**: Only films with vote counts above the **90th percentile** are eligible. This eliminates films whose ratings are statistically unreliable - a film with 8 votes has enormous rating variance regardless of its true quality.
- **Bayesian ranking**: Remaining candidates are sorted by weighted rating rather than raw average, ensuring that films with marginal vote counts do not outrank genuinely well-regarded titles.
- **Temporal filter**: Films released before 1990 are deprioritized, aligned with the dataset's distribution - pre-1990 films represent a small fraction of the catalog and often require more specific cinephile context to appreciate out of cold-start.
- **Genre diversity enforcement**: The final selection avoids recommending more than 2 films from any single primary genre. This prevents the baseline from becoming a Drama-only list and ensures new users encounter the breadth of the catalog immediately.

#### Baseline Model Results

| Metric | Value |
|--------|-------|
| **Average Rating** | 8.24 / 10 (dataset avg: ~6.0) |
| **Genre Coverage** | 10 primary genres |
| **Most Common Genre** | Drama |
| **Temporal Range** | 80% released 1990–2014 |
| **Average Release Year** | 1996 |

The average rating of 8.24 versus the dataset mean of ~6.0 confirms that the Bayesian filtering and vote threshold are working as intended - the baseline surfaces films that are both well-regarded and statistically trustworthy. The 10-genre coverage demonstrates that the diversity enforcement is effective: a cold-start user immediately sees a representative cross-section of the catalog rather than a single-genre cluster.

#### Baseline Performance Visualization

![Popularity vs Quality](images/Popularity_vs_Quality_Top10_Baseline_Recommendations.png)

The scatter plot above maps all baseline recommendations (red points) against the full dataset (gray background). Red points cluster in the **upper-right quadrant** - high popularity and high quality simultaneously. Critically, they are not exclusively the highest-popularity films in the dataset, which would represent a collapse to "recommend only blockbusters." The selection spans a range of popularity levels while maintaining quality above 8.0, demonstrating that the quality score formula successfully decouples critical reception from commercial reach.

---

### 🚀 Main Model: Hybrid Recommendation Engine

The main model is activated when the user provides a seed movie. It uses Natural Language Processing to compute semantic similarity across the catalog and then applies a multi-stage ranking pipeline to balance relevance, quality, diversity, and recency.

#### Step 1: NLP Processing - Metadata "Soup" Creation

The metadata soup described in the Feature Engineering section is the primary input to the similarity model. Each film is represented as a single string concatenating keywords, cast, director (weighted 3×), and genres (IDF-weighted). This design encodes the intuition that a film's thematic content and authorial signature are more discriminative than any individual cast member.

#### Step 2: Dual Vectorization Strategy

Two vectorization approaches are applied in parallel, targeting different aspects of film content:

**TF-IDF Vectorization** (applied to movie overviews):
- Downweights terms that appear frequently across the corpus - words like `action` or `story` that appear in hundreds of overviews carry little discriminative power and are de-emphasized automatically
- Emphasizes distinctive descriptors - terms like `cyberpunk`, `neo-noir`, or `dystopian` that appear rarely are weighted higher, making them strong similarity signals when they do match
- Enables genuine semantic differentiation: a film described as `Space Horror` will not strongly match a film described as `Space Comedy` despite sharing the `space` token

**CountVectorizer** (applied to metadata soup):
- Captures raw term frequency for cast, crew, keywords, and genre tokens
- Does not apply IDF weighting - every occurrence of `christophernolan` is equally significant regardless of how many films he directed
- Removes English stop words to reduce noise from common grammatical tokens

The final similarity matrix is computed on the union of both feature spaces, combining **semantic depth from overviews** with **explicit identity signals from metadata**. This dual approach consistently outperforms either vectorizer in isolation, particularly for films where the overview is sparse or generic.

#### Step 3: Cosine Similarity

Movie similarity is computed using cosine similarity on the combined feature spaces:

```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Cosine similarity is preferred over Euclidean distance because it is invariant to document length - a short overview and a long overview can achieve a high similarity score if they discuss the same themes, whereas Euclidean distance would penalize the length difference regardless of content. Scores range from **0** (no shared tokens or themes) to **1** (identical metadata representation).

#### Step 4: Recommendation Refinement Pipeline

Raw cosine similarity scores are not directly used for ranking. A five-stage refinement pipeline transforms them into the final recommendation list:

1. **Candidate retrieval**: All films above a minimum similarity threshold to the seed are identified as candidates.
2. **Popularity and age penalties**: Logarithmic scaling is applied to popularity, and an age-based penalty is computed from `movie_age`. These adjustments prevent the ranking from collapsing into "most popular films with any thematic overlap."
3. **Hybrid score computation**: Similarity and quality signals are combined using the α-weighted formula below.
4. **MMR re-ranking**: Maximal Marginal Relevance re-orders the top candidates to balance relevance with diversity.
5. **Final selection**: Top N recommendations are returned with their hybrid scores.

**Without MMR**: A query for a Christopher Nolan film could return 8 of 10 recommendations from Nolan's own filmography - technically high similarity, but offering the user no discovery value.

**With MMR**: Each successive recommendation is selected to maximize similarity to the seed *while* minimizing similarity to already-selected recommendations. The result is a semantically coherent set that is internally diverse - different directors, different sub-genres, different decades.

#### Step 5: Hybrid Ranking Formula

```
Final Score = α × Quality_norm + (1 − α) × Similarity_norm
```

| α Value | Recommendation Behavior |
|---------|---------|
| **0.3** | 70% similarity, 30% quality - prioritizes thematic closeness, accepts lower-rated films if they match semantically |
| **0.5** | Balanced - equal weight to relevance and quality |
| **0.7** | Quality-driven - surfaces the highest-rated films among those with meaningful semantic overlap |

The optimal α is selected automatically via grid search for each query, evaluated on a composite score combining quality, diversity, genre overlap, and popularity bias across the candidate set. This means the system adapts its relevance-quality tradeoff based on the specific input movie and available candidate pool.

---

## 📊 Performance & Evaluation

### Understanding the Evaluation Metrics

Before examining the results, it is worth clarifying what each metric measures - and what its inherent limitations are in a content-based filtering context.

| Metric | What It Measures | Limitation in This Context |
|--------|-----------------|------------|
| **Hit Rate@K** | Proportion of users for whom at least one recommended film appears in their actual liked films | Binary - doesn't capture how many relevant films were found or where they ranked |
| **Precision@K** | Of the K recommended films, what fraction are actually relevant | Must be interpreted relative to catalog size (4,800 films) and recommendation count (5 or 10) |
| **Recall@K** | Of all films the user actually likes, what fraction were captured in K recommendations | Structurally low for content-based systems without collaborative signals |
| **NDCG@K** | Whether relevant films appear early in the ranking (higher rank = more weight) | Sensitive to ground truth completeness; sparser user histories produce noisier estimates |

The most business-relevant metric for a discovery application is **Hit Rate** - it measures whether the system successfully introduces the user to at least one film they genuinely appreciate. Precision and Recall must be interpreted relative to the catalog size and recommendation count, not as absolute judgments.

---

### Internal Simulation

The model was validated through 20 simulation iterations across 50 randomly sampled movies per iteration (1,000 recommendation sets total). Each set was evaluated on four dimensions designed to catch the specific failure modes identified during development.

| Metric | Target Range | Achieved | Status |
|--------|-------------|----------|--------|
| **Quality (Avg Rating)** | ≥ 6.5 | **6.81 / 10** | ✅ Excellent |
| **Diversity Index** | ≥ 0.70 | **0.70** | ✅ High Exploration |
| **Genre Overlap** | 0.60–0.75 | **0.65** | ✅ Balanced |
| **Popularity Bias** | 1.0–2.5 | **1.40** | ✅ Goldilocks Zone |

**Interpreting the results:**

- **Quality at 6.81** exceeds the target of 6.5 and sits meaningfully above the dataset average of ~6.0. This confirms that the Bayesian quality score is successfully filtering the candidate pool toward genuinely well-regarded content rather than merely thematically similar content.
- **Diversity Index at exactly 0.70** (the lower bound of the target range) indicates MMR is working but at its minimum acceptable level. The system is not over-diversifying - recommendations remain semantically coherent - but there is limited headroom before diversification becomes insufficient. This is a conscious design trade-off: pushing diversity above 0.75 risks producing recommendations that feel disconnected from the seed.
- **Genre Overlap at 0.65** sits in the middle of the 0.60–0.75 target range, representing a balanced recommendation set: close enough in genre to feel relevant, varied enough to offer discovery value. A genre overlap of 1.0 would mean all recommendations share the exact same genres as the input - technically "accurate" but offering no breadth.
- **Popularity Bias at 1.40** means recommended films are, on average, 40% more popular than the median film in the catalog. This is a healthy level - the system trends toward recognizable films without collapsing into blockbuster-only recommendations. A value above 2.5 would indicate systematic over-recommendation of mainstream content.

![Second Model Results](images/Plot_Results_for_Second_Model.png)

---

### External Evaluation - Leave-One-Out (MovieLens)

The internal simulation validates aggregate behavior but cannot measure whether recommendations match actual user preferences. For this, a Leave-One-Out evaluation was conducted using the **MovieLens dataset** - an external source of verified user ratings entirely independent of the TMDB data used to build the model.

**Methodology**: 500 users were sampled from MovieLens. For each user, one film with a rating ≥ 3.5 was selected as the recommendation seed. The remaining liked films formed the ground truth. The recommendation function was called with the seed and results compared against the ground truth. This methodology is deliberately strict: the system receives a single seed film and must identify films the user genuinely liked from a 4,800-title catalog, without any collaborative filtering signal or user history.

![Evaluation Metrics](images/Evaluation_Metrics_Leave_One_Out.png)

| Metric | @5 | @10 |
|--------|-----|------|
| **Hit Rate** | 26.60% | **39.00%** |
| **Precision** | 7.04% | 6.06% |
| **Recall** | 1.00% | 1.57% |
| **NDCG** | 7.43% | 6.62% |

**Interpreting the results:**

- **Hit Rate@10 = 39%**: In approximately 2 of every 5 recommendation sessions, the system surfaces at least one film the user genuinely liked - with zero prior interaction history and a single seed. For a purely content-based system operating cold-start, this is the headline result and a strong benchmark.
- **Hit Rate@5 → @10 improvement (+12.4 pp)**: The meaningful jump from 26.6% to 39.0% when expanding from 5 to 10 recommendations confirms that relevant films do appear in the top results rather than only at the tail of the list. The ranking function is calibrated correctly.
- **Precision@5 = 7.04%**: Of 5 recommendations, an average of 0.35 are genuinely liked by the user. This must be contextualized: a random recommender on a 4,800-film catalog achieves Precision ≈ 0.1%. The model operates at **65× above random baseline** - confirming the hybrid approach extracts genuine signal.
- **Recall@10 = 1.57%**: Structurally low, and expected. A user who likes 100 films cannot have them all captured in 10 recommendations from a 4,800-film catalog by a system with no collaborative signal. Recall is the primary limitation of content-based filtering and would require collaborative signals to meaningfully improve.
- **NDCG@5 = 7.43% > NDCG@10 = 6.62%**: The higher NDCG at @5 versus @10 is the most informative quality signal from this evaluation. When relevant films appear in the recommendation set, they tend to appear early - the hybrid scoring function is placing them near the top of the list rather than burying them. This confirms that the ranking mechanism is doing useful work beyond simple retrieval.

---

### Model Comparison: Baseline vs. Main Model

| Dimension | Baseline Model | Main Model |
|-----------|---------------|------------|
| **Activation** | No user input available | Seed movie provided |
| **Input signal** | None - statistical filtering only | Single seed movie (content-based) |
| **Hit Rate@10** | N/A (no personalization) | 39.0% |
| **Avg. Quality Score** | 8.24 / 10 | 6.81 / 10 |
| **Genre Coverage** | 10 genres (enforced) | Balanced (MMR-controlled) |
| **Popularity Bias** | Moderate (high-vote threshold) | 1.40× median (controlled) |
| **Discovery Value** | High - curated breadth | High - MMR diversification |

**Conclusions:**

The two models serve fundamentally different purposes and must be evaluated against different criteria. The baseline model optimizes for statistical reliability and genre breadth - it is designed to create a strong first impression and demonstrate catalog quality, not to match individual preferences. Its average quality of 8.24/10 reflects this philosophy: it surfaces the most universally well-regarded films in the catalog, selected through statistical rigor rather than personalization.

The main model accepts lower average quality (6.81 vs. 8.24) in exchange for personalization - films semantically related to the user's seed may be less universally acclaimed than the all-time greats, but they are meaningfully more relevant to what the user is looking for. The 39% Hit Rate confirms that this trade-off is justified: personalization successfully outperforms statistical curation when measured against actual user preferences from MovieLens.

The critical result is the **65× improvement over random baseline** in Precision. This confirms that the hybrid approach - combining TF-IDF semantic similarity, CountVectorizer metadata matching, Bayesian quality scoring, and MMR diversification - is extracting genuine signal from the data. For a system with no collaborative filtering and no user history, this represents the realistic performance ceiling of content-based recommendation on this catalog, and demonstrates that the architecture is well-suited for the cold-start problem it was designed to address.

---

## 📱 Application Interface

The project is deployed as an interactive **Streamlit** web application with four main screens.

### Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit |
| **Backend** | Scikit-Learn (TF-IDF, CountVectorizer, cosine similarity) |
| **Data Processing** | Pandas, NumPy, SciPy |
| **Visualization** | Matplotlib, Seaborn |
| **Poster Fetching** | TMDB API |
| **Deployment** | Streamlit Cloud |
| **Containerization** | Docker, Docker Compose |
| **Model Storage** | Hugging Face Hub |

### User Interface

#### Home Page
![Interface 1](images/st_interface1.png)

Two main entry points: the **Baseline Model** for new users with no preferences, and the **Intelligent Model** for users who provide a seed movie.

#### Baseline Recommendations
![Interface 2](images/st_interface2.png)

Includes a slider (1–20 movies), genre filter, and hover interactions displaying poster and synopsis.

#### Intelligent Model Recommendations
![Interface 3](images/st_interface3.png)
![Interface 4](images/st_interface4.png)

Each recommendation displays:
- **Match %**: Hybrid score (0–100%)
- **Poster**: Dynamically fetched from TMDB API
- **Title & Rating**: Bayesian-weighted score
- **Genres**: Top 2–3 relevant tags
- **Details**: Expandable synopsis and metadata

---

## 🛠️ Installation

### Prerequisites

- Python 3.13+
- MovieLens dataset (for evaluation only) - downloaded automatically by the notebook

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/Przemsonn05/Content-Based-Recommendation-System-for-Movies.git

# 2. Navigate to directory
cd Content-Based-Recommendation-System-for-Movies

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add data files to /data
# - tmdb_5000_movies.csv
# - tmdb_5000_credits.csv
# Available at: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

# 5. Launch Streamlit app
streamlit run app.py

# 6. (Optional) Run the full analysis notebook
jupyter notebook notebooks/Recommendation_System.ipynb
```

> **Note**: Pre-trained models are automatically loaded from Hugging Face on first run. No manual download required.

---

## 🐳 Docker

The project is fully containerized using Docker, enabling reproducible deployment without any local Python environment setup. The image runs the Streamlit app as a non-root user for security, and the layer order in the `Dockerfile` is intentional - dependencies are installed before application code is copied, so Docker's build cache is preserved between code changes and rebuilds take seconds rather than minutes.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- [Docker Compose](https://docs.docker.com/compose/install/) (included with Docker Desktop on Windows/macOS)

### Running with Docker Compose (recommended)

```bash
# Build the image and start the container
docker compose up --build

# Open the app at http://localhost:8501
```

The `--build` flag is only needed on the first run or after modifying `requirements.txt` or the `Dockerfile`. On subsequent runs:

```bash
docker compose up
```

To stop and remove the container:

```bash
docker compose down
```

### Running with Docker directly

```bash
docker build -t recommendation-system .
docker run -p 8501:8501 recommendation-system
```

### Volumes and environment

The `docker-compose.yml` mounts the data directory into the container so the TMDB dataset files are accessible without being baked into the image:

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `./data` | `/app/data` | TMDB CSV files (`tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`) |

The pre-trained models (TF-IDF vectorizer, CountVectorizer, similarity matrix) are downloaded from Hugging Face Hub at startup - no local model files are required. An internet connection is needed on first launch.

> **Note:** The `notebooks/` and `images/` directories are excluded from the image via `.dockerignore` to keep the image size minimal.

---

## 💡 Key Learnings & Future Work

### What Worked

**1. "Metadata Soup" with Differential Weighting** - Combining cast, crew, keywords, and genres into a single string with non-uniform weights outperformed treating each as a separate feature. The 3× director repetition and IDF-based genre weighting captured the intuition that authorial signature and thematic rarity matter more than cast overlap. Domain knowledge encoded in feature design beats raw model complexity.

**2. TF-IDF Superiority over Keyword Matching** - TF-IDF significantly outperformed simple keyword matching for overview similarity. Generic terms are automatically downweighted without manual curation, and rare descriptive terms become strong matching signals. Smart feature engineering consistently beat naive approaches on this problem.

**3. Bayesian Weighting for Cold-Start Quality** - Essential for the baseline model. Without it, films with 2–3 perfect ratings would dominate the quality ranking over films with thousands of consistently high ratings. Statistical confidence must be part of any quality signal in systems where vote counts vary by orders of magnitude.

**4. MMR Algorithm for Diversity** - Without MMR, recommendation sets collapsed to variations on the same director or franchise. With it, results are semantically coherent with the input while covering different tonal registers, time periods, and creative voices. This is the single most impactful post-processing step in the pipeline.

**5. Automated Alpha Tuning** - Fixing α at 0.5 for all inputs worked reasonably well in aggregate but underperformed for edge cases. Grid search per query improved robustness at minimal computational cost.

### Future Improvements

**Short-Term:**
- Explainability dashboard showing "Why this recommendation?" with score component breakdown visible to the user
- User feedback loop (thumbs up/down) to iteratively adjust feature weights without full retraining
- A/B testing framework for weight configuration experiments across user cohorts

**Long-Term:**
- Upgrade NLP layer to **BERT** or **Sentence Transformers** for deeper semantic understanding - particularly valuable for overviews where thematic similarity is expressed in varied language
- Conversational interface for natural language queries (*"I want a sad movie about robots with a hopeful ending"*)
- Hybrid collaborative filtering combining content signals with user behavior - the most direct path to improving Recall, which is the primary limitation of the current pure content-based approach
- Production deployment on **AWS ECS** or **Google Cloud Run** with horizontal scaling for concurrent users

---

## 📄 License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## 🙏 Acknowledgments

- **TMDB** - Comprehensive movie database and API
- **MovieLens / GroupLens** - Ground truth dataset for external evaluation
- **Scikit-Learn** - Machine learning and vectorization tools
- **Streamlit** - Accessible deployment framework

---

<div align="center">

**⭐ Found this helpful? Please star the repository!**

[![GitHub stars](https://img.shields.io/github/stars/Przemsonn05/Content-Based-Recommendation-System-for-Movies?style=social)](https://github.com/Przemsonn05/Content-Based-Recommendation-System-for-Movies)

**Made by [Przemsonn05](https://github.com/Przemsonn05)**

</div>