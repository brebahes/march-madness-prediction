import argparse
from pathlib import Path
import pandas as pd
import sys
import subprocess

def get_processed_data_path(year: int, men: bool = True) -> Path:
    """Get the path to the processed data file"""
    gender_prefix = 'M' if men else 'W'
    return Path(f'data/preprocessed/{gender_prefix}ProcessedTourneyData_{year}.csv')

def ensure_processed_data(year: int, men: bool = True, ranking_system: str = 'SEL', n_games: int = 5) -> pd.DataFrame:
    """
    Load preprocessed tournament data, generating it first if it doesn't exist.
    
    Args:
        year (int): The year to load data for
        men (bool): If True, load men's data. If False, load women's data
        ranking_system (str): The ranking system to use if data needs to be generated
        n_games (int): Number of games for rolling statistics if data needs to be generated
        
    Returns:
        pd.DataFrame: The processed tournament data
    """
    data_path = get_processed_data_path(year, men)
    
    # Check if processed data exists
    if not data_path.exists():
        print(f"Processed data not found at {data_path}")
        print("Generating processed data...")
        
        # Get the path to process_data.py relative to this script
        process_script = Path(__file__).parent.parent / 'data_preparation' / 'process_data.py'
        
        # Build command
        cmd = [
            sys.executable,
            str(process_script),
            '--year', str(year),
            '--ranking-system', ranking_system,
            '--n-games', str(n_games)
        ]
        if not men:
            cmd.append('--women')
            
        # Run the processing script
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to generate processed data: {e}")
            
        if not data_path.exists():
            raise FileNotFoundError(f"Data processing completed but file not found at {data_path}")
    
    # Load and return the processed data
    print(f"Loading processed data from {data_path}")
    return pd.read_csv(data_path)

def main():
    parser = argparse.ArgumentParser(description='Load or generate processed NCAA tournament data')
    parser.add_argument('--year', type=int, required=True, help='Year to load data for')
    parser.add_argument('--women', action='store_true', help='Load women''s basketball data instead of men''s')
    parser.add_argument('--ranking-system', type=str, default='SEL', help='Ranking system to use if generating data')
    parser.add_argument('--n-games', type=int, default=5, help='Number of games for rolling statistics if generating data')
    
    args = parser.parse_args()
    
    try:
        data = ensure_processed_data(
            year=args.year,
            men=not args.women,
            ranking_system=args.ranking_system,
            n_games=args.n_games
        )
        print(f"Data loaded successfully! Shape: {data.shape}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 