import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from transformers import (
    RankingTransformer, 
    RollingStatsTransformer, 
    RandomizeTeamsTransformer,
    TournamentSlotTransformer
)
from sklearn.pipeline import Pipeline

feature_pipeline = Pipeline([
    ('tournament_slots', TournamentSlotTransformer()),
    ('rankings', RankingTransformer(ranking_system='SEL', detailed_results=True)),
    ('rolling_stats', RollingStatsTransformer(n_games=5)),
    ('randomize_teams', RandomizeTeamsTransformer()),
])

if __name__ == '__main__':
    from src.data_preparation.dataloader import load_raw_dataset
    
    # Load 2024 men's basketball data
    print("Loading data...")
    data = load_raw_dataset(men=True, year=2024)
    
    # Apply the pipeline transformations
    print("Applying feature pipeline...")
    processed_data = feature_pipeline.transform(data)
    
    print(f"Processing complete! Output shape: {processed_data.shape}")
    print("\nFirst few rows of processed data:")
    print(processed_data.head())
