import argparse
from pathlib import Path
from dataloader import load_raw_dataset
from transformers import RankingTransformer, RollingStatsTransformer
from sklearn.pipeline import Pipeline

def load_processed_data(year: int, men: bool = True, ranking_system: str = 'SEL', n_games: int = 5):


def process_data(year: int, men: bool = True, ranking_system: str = 'SEL', n_games: int = 5):
    """
    Process NCAA basketball data for a given year and gender.
    
    Args:
        year (int): The year to process data for
        men (bool): If True, process men's data. If False, process women's data
        ranking_system (str): The ranking system to use (default: 'SEL')
        n_games (int): Number of games to use for rolling statistics (default: 5)
    """
    print(f"Loading {'men''s' if men else 'women''s'} basketball data for {year}...")
    
    # Load the data
    data = load_raw_dataset(men=men, year=year)
    print(data.keys())
    # Import and configure the pipeline
    from pipelines import feature_pipeline
    feature_pipeline.set_params(
        rankings__ranking_system=ranking_system,
        rolling_stats__n_games=n_games
    )
    print("Applying transformations...")
    processed_data = feature_pipeline.transform(data)
    
    # Create output filename
    gender_prefix = 'M' if men else 'W'
    output_path = Path(f'data/preprocessed/{gender_prefix}ProcessedTourneyData_{year}.csv')
    
    # Save the processed data
    print(f"Saving processed data to {output_path}...")
    processed_data.to_csv(output_path, index=False)
    print(f"Data processing complete! Output shape: {processed_data.shape}")

def main():
    parser = argparse.ArgumentParser(description='Process NCAA basketball tournament data')
    parser.add_argument('--year', type=int, required=True, help='Year to process data for')
    parser.add_argument('--women', action='store_true', help='Process women''s basketball data instead of men''s')
    parser.add_argument('--ranking-system', type=str, default='SEL', help='Ranking system to use (default: SEL)')
    parser.add_argument('--n-games', type=int, default=5, help='Number of games for rolling statistics (default: 5)')
    
    args = parser.parse_args()
    
    process_data(
        year=args.year,
        men=not args.women,
        ranking_system=args.ranking_system,
        n_games=args.n_games
    )

if __name__ == '__main__':
    main() 