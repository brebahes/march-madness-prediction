from transformers import RankingTransformer, RollingStatsTransformer, RandomizeTeamsTransformer
from sklearn.pipeline import Pipeline

feature_pipeline = Pipeline([
    ('rankings', RankingTransformer(ranking_system='SEL', detailed_results=True)),
    ('rolling_stats', RollingStatsTransformer(n_games=5)),
    ('randomize_teams', RandomizeTeamsTransformer())
])

if __name__ == '__main__':
    from dataloader import load_dataset
    
    # Load 2024 men's basketball data
    print("Loading data...")
    data = load_dataset(men=True, year=2024)
    
    # Apply the pipeline transformations
    print("Applying feature pipeline...")
    processed_data = feature_pipeline.transform(data)
    
    print(f"Processing complete! Output shape: {processed_data.shape}")
    print("\nFirst few rows of processed data:")
    print(processed_data.head())
