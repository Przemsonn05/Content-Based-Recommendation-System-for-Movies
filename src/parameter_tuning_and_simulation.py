import pandas as pd
import numpy as np
from src.evaluation import evaluate_model

def find_best_alpha(data, cosine_sim, indices, alpha_range=np.arange(0.1, 1.0, 0.1)):
    """
    Iterates over a range of alpha values to find the best balance between 
    quality, diversity, genre overlap, and popularity bias for the recommendation model.

    Args:
        data (pd.DataFrame): The preprocessed movie dataset.
        cosine_sim (np.ndarray): The precomputed cosine similarity matrix.
        indices (pd.Series): A pandas Series mapping movie titles to their DataFrame indices.
        alpha_range (np.ndarray, optional): An array of alpha values to test. 
            Defaults to np.arange(0.1, 1.0, 0.1).

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation metrics and a composite score 
                      for each tested alpha value.
    """
    results = []
    print("Starting the search for the best alpha parameter...")

    for alpha in alpha_range:
        metrics = evaluate_model(data, cosine_sim, indices, sample_size=50, alpha=alpha)
        
        composite = (
            0.35 * metrics.get('quality', 0) +
            0.30 * metrics.get('diversity', 0) +
            0.20 * metrics.get('genre_overlap', 0) +
            0.15 * (1 / (metrics.get('popularity_bias', 1) + 1e-9))
        )
        
        results.append({
            'alpha': alpha,
            'quality': metrics.get('quality', 0),
            'diversity': metrics.get('diversity', 0),
            'genre_overlap': metrics.get('genre_overlap', 0),
            'popularity_bias': metrics.get('popularity_bias', 0),
            'composite_score': composite
        })
        
        print(f"-> Alpha {alpha:.1f}: Composite={composite:.3f} | Div={metrics.get('diversity', 0):.3f}")

    return pd.DataFrame(results)

def run_simulation_no_plots(data, cosine_sim, indices, iterations, sample_size, alpha=0.5):
    """
    Runs the recommendation evaluation simulation multiple times and aggregates 
    the results into a single DataFrame.

    Args:
        data (pd.DataFrame): The preprocessed movie dataset.
        cosine_sim (np.ndarray): The precomputed cosine similarity matrix.
        indices (pd.Series): A pandas Series mapping movie titles to their DataFrame indices.
        iterations (int): The number of simulation iterations to run.
        sample_size (int): The number of movies to sample per iteration.
        alpha (float, optional): The weight parameter balancing quality and similarity. Defaults to 0.5.

    Returns:
        pd.DataFrame: A combined DataFrame containing the results of all simulation iterations.
    """
    all_results = []
    print(f"Running simulation: {iterations} iterations...")

    for i in range(iterations):
        df = evaluate_model(
            data=data, 
            cosine_sim=cosine_sim, 
            indices=indices,
            sample_size=sample_size, 
            top_k=10, 
            alpha=alpha,
            plot_charts=False  
        )
        
        if isinstance(df, pd.DataFrame):
            df['iteration'] = i
            all_results.append(df)
        else:
            print(f"Warning: iteration {i} did not return a DataFrame.")
        
    if not all_results:
        print("Error: No iteration returned valid results.")
        return pd.DataFrame()

    global_df = pd.concat(all_results, ignore_index=True)
    print("Simulation completed!")
    return global_df