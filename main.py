import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config import MOVIES_PATH, CREDITS_PATH, RATINGS_PATH, MOVIES_FROM_RATINGS_PATH
from src.data_loading_and_preprocessing import load_and_merge_data
from src.processing import parse_json_columns, add_engineered_features, build_matrices
from src.eda_visualizations import generate_all_eda_plots
from src.models import calculate_weighted_rating, get_baseline_recommendations, recommendation
from src.evaluation import evaluate_model, run_leave_one_out_evaluation

def run_comprehensive_test():
    print("="*50)
    print("STARTING COMPREHENSIVE SYSTEM TEST")
    print("="*50)

    try:
        raw_data = load_and_merge_data(MOVIES_PATH, CREDITS_PATH, RATINGS_PATH, MOVIES_FROM_RATINGS_PATH)
        print(f"Data loaded: {raw_data.shape}")
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return

    print("\n--- Feature Processing ---")
    data_parsed = parse_json_columns(raw_data.copy())
    data_eng = add_engineered_features(data_parsed)
    
    data_eng['release_year'] = data_eng['release_year'].fillna(data_eng['release_year'].median())
    
    print(f"Feature engineering completed. Columns: {data_eng.columns.tolist()[:10]}...")

    print("\n--- Building TF-IDF + Count + Numerical Matrix ---")
    combined_matrix, final_data = build_matrices(data_eng)
    cosine_sim = cosine_similarity(combined_matrix, combined_matrix)
    print(f"Similarity matrix ready: {cosine_sim.shape}")

    print("\n--- Baseline Model Test ---")
    data_with_wr, C, m = calculate_weighted_rating(final_data)
    top_movies = get_baseline_recommendations(data_with_wr, n=5)
    print("Top 5 movies (Weighted Rating):")
    print(top_movies[['original_title', 'weighted_rating']])

    print("\n--- Content-Based Recommendation Test ---")
    
    indices = pd.Series(final_data.index, index=final_data['original_title']).drop_duplicates()
    
    test_titles = ["The Dark Knight", "Inception", "Toy Story"]
    
    for title in test_titles:
        if title in final_data['original_title'].values:
            print(f"\nRecommendations for: {title}")
            
            recs = recommendation(title, cosine_sim, final_data, indices, alpha=0.3, use_mmr=True)
            
            if not recs.empty:
                print(recs[['original_title', 'final_score', 'vote_average']].head(5))
            else:
                print(f"No recommendations for {title}")
        else:
            print(f"ℹMovie '{title}' not found in the database.")

    print("\n--- Edge Cases Test ---")
    
    print("Test A (Non-existent movie): ", end="")
    fake_recs = recommendation("Movie That Does Not Exist 2026", cosine_sim, final_data, indices)
    print("Returned empty DataFrame" if fake_recs.empty else "Should be empty")

    print("Test B (Short description): ", end="")
    subset = final_data.iloc[0:5]
    if 'soup' in subset.columns:
        print(f"'soup' field generated. Sample length: {len(subset.iloc[0]['soup'])}")

    print("\n--- Running evaluation (Sample) ---")
    eval_metrics = evaluate_model(final_data, cosine_sim, indices, sample_size=5, top_k=5)
    print(f"Quality metrics: {eval_metrics}")

    print("\n" + "="*50)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    run_comprehensive_test()