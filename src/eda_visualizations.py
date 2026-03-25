import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('seaborn-whitegrid')

def create_output_dir(output_dir):
    """
    Creates the target directory for saving plots if it does not already exist.

    Args:
        output_dir (str): The path to the directory where plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

def plot_feature_distributions(data, output_dir):
    """
    Generates and saves a 2x2 grid of histograms showing the distribution 
    of 4 main numerical features: vote_average, popularity, runtime, and vote_count.

    Args:
        data (pd.DataFrame): The movie dataset containing the features.
        output_dir (str): The directory where the plot image will be saved.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10)) 
    fig.suptitle('Distribution of Movie Features', fontsize=18, fontweight='bold')
    
    features = [
        ('vote_average', 'Vote Average', 'Average Rating', axes[0,0]),
        ('popularity', 'Popularity', 'Popularity Score', axes[0,1]),
        ('runtime', 'Runtime', 'Minutes', axes[1,0]),
        ('vote_count', 'Vote Count', 'Number of Votes', axes[1,1])
    ]
    
    for col, title, xlabel, ax in features:
        if col in data.columns:
            median_val = data[col].median()
            ax.hist(data[col].dropna(), bins=30, color="#4e9ee9", edgecolor='black')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel('Number of Movies', fontsize=12)
            ax.axvline(median_val, color='red', linestyle='--', linewidth=1.5, label=f"Median: {median_val:.1f}")
            ax.legend(fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout() 
    plt.savefig(os.path.join(output_dir, 'Distribution_of_Movie_Features.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_movies_by_year(data, output_dir):
    """
    Generates and saves a bar chart displaying the number of movies released each year.

    Args:
        data (pd.DataFrame): The movie dataset containing the 'release_year' column.
        output_dir (str): The directory where the plot image will be saved.
    """
    if 'release_year' not in data.columns:
        return

    plt.figure(figsize=(12, 7))
    years = data['release_year'].value_counts().sort_index()
    plt.bar(years.index, years.values, color='#5b7be6', edgecolor='black')
    
    median_year = data['release_year'].median()
    plt.axvline(median_year, color='red', linestyle='--', linewidth=1.5, label=f"Median: {median_year:.1f}")
    
    plt.xlabel('Movie Release Year', fontsize=12)
    plt.ylabel('Number of Movies', fontsize=12)
    plt.title('Number of Movies by Year', fontsize=16, fontweight='bold')
    plt.xticks(ticks=np.arange(min(years.index), max(years.index)+1, 5), rotation=45, fontsize=10)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.grid(axis='x', visible=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Number_of_Movies_by_Year.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_popular_genres(data, output_dir):
    """
    Generates and saves a horizontal bar chart of the 20 most popular movie genres.

    Args:
        data (pd.DataFrame): The movie dataset containing the 'genres' column.
        output_dir (str): The directory where the plot image will be saved.
    """
    if 'genres' not in data.columns:
        return

    plt.figure(figsize=(12, 7))
    genre_counts = data['genres'].explode().value_counts().head(20)
    colors = ['#ff7f0e' if i < 2 else '#5b7be6' for i in range(len(genre_counts))]
    
    bars = plt.barh(genre_counts.index, genre_counts.values, color=colors)
    for bar in bars:
        plt.text(bar.get_width() + 1,               
                 bar.get_y() + bar.get_height()/2, 
                 int(bar.get_width()),             
                 va='center', fontsize=10)
                 
    plt.xlabel('Number of Movies', fontsize=12)
    plt.ylabel('Genres', fontsize=12)
    plt.title('20 Most Popular Genres', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.grid(axis='y', visible=False)
    plt.gca().invert_yaxis()  
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '20_Most_Popular_Genres.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_genre_cooccurrence(data, output_dir):
    """
    Generates and saves a co-occurrence matrix heatmap for the top 10 most frequent movie genres.

    Args:
        data (pd.DataFrame): The movie dataset containing the 'genres' column.
        output_dir (str): The directory where the plot image will be saved.
    """
    if 'genres' not in data.columns:
        return

    mlb = MultiLabelBinarizer()
    clean_genres = data['genres'].dropna() 
    genre_matrix = mlb.fit_transform(clean_genres)
    
    co_occurrence = pd.DataFrame(
        genre_matrix.T @ genre_matrix, 
        index=mlb.classes_, 
        columns=mlb.classes_
    )
    
    top_genres = clean_genres.explode().value_counts().head(10).index
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(co_occurrence.loc[top_genres, top_genres], 
                annot=True, fmt='d', cmap='Blues', 
                linewidths=0.5, linecolor='gray', cbar=True, square=True,
                annot_kws={"size":10})
                
    plt.title('Genre Co-occurrence Matrix (Top 10 Genres)', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Genre_Co-occurence_Matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_popularity_vs_rating(data, output_dir):
    """
    Generates and saves a scatter plot visualizing the relationship between 
    movie popularity and average rating, including a regression line.

    Args:
        data (pd.DataFrame): The movie dataset containing 'popularity' and 'vote_average' columns.
        output_dir (str): The directory where the plot image will be saved.
    """
    if 'popularity' not in data.columns or 'vote_average' not in data.columns:
        return

    plt.figure(figsize=(12, 7))
    plt.scatter(data['popularity'], data['vote_average'],
                alpha=0.2, s=15, color='#5b7be6', edgecolor='black', linewidth=0.2)
                
    sns.regplot(x='popularity', y='vote_average', data=data, 
                scatter=False, color='red', line_kws={'linewidth': 2})
                
    plt.xlabel('Popularity', fontsize=12)
    plt.ylabel('Vote Average (Rating)', fontsize=12)
    plt.title('Popularity vs. Rating', fontsize=16, fontweight='bold')
    plt.xlim(0, 200)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Popularity_vs_Rating.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_eda_plots(data, output_dir='../images'):
    """
    Main orchestration function that runs all EDA plotting functions and saves 
    the visualizations to the specified output directory.

    Args:
        data (pd.DataFrame): The preprocessed movie dataset.
        output_dir (str, optional): The directory to save the output plots. Defaults to '../images'.
    """
    print(f"Creating directory for plots: {output_dir} (if it doesn't exist)...")
    create_output_dir(output_dir)
    
    print("Generating: Feature distributions...")
    plot_feature_distributions(data, output_dir)
    
    print("Generating: Movies over the years...")
    plot_movies_by_year(data, output_dir)
    
    print("Generating: Most popular genres...")
    plot_popular_genres(data, output_dir)
    
    print("Generating: Genre co-occurrence matrix...")
    plot_genre_cooccurrence(data, output_dir)
    
    print("Generating: Popularity vs Rating...")
    plot_popularity_vs_rating(data, output_dir)
    
    print("Finished generating all plots!")