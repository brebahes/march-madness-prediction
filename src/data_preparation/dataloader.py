import pandas as pd
import os
from pathlib import Path

PROCESSED_DATA_DIR = Path('data/preprocessed')
RAW_DATA_DIR = Path('data/raw')


def load_data(data_dir: str = 'data/raw', file_pattern: str = '*') -> dict:
    """
    Load data files from the specified directory.
    
    Args:
        data_dir (str): Path to the data directory (default: 'data/raw  ')    
        file_pattern (str): Pattern to match files (default: '*' for all files)
    
    Returns:
        dict: Dictionary with filenames as keys and loaded data as values
    """
    data_path = Path(data_dir)                    
    data_files = {}

    if not data_path.exists():
        raise FileNotFoundError(f"Directory {data_dir} not found")
    
    for file_path in data_path.glob(file_pattern):
        if file_path.suffix.lower() in ['.csv', '.xlsx', '.xls']:
            try:
                if file_path.suffix.lower() == '.csv':
                    data = pd.read_csv(file_path)
                else:
                    data = pd.read_excel(file_path)                                                                     

                data_files[file_path.stem] = data
                print(f"Successfully loaded: {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path.name}: {str(e)}")      
                
    if not data_files:
        print("No compatible files found in the specified directory")
    
    return data_files   

def filter_by_year(data_files: dict, year: int) -> dict:
    """
    Filter data files by a specific year.
    
    Args:
        data_files (dict): Dictionary containing DataFrames with NCAA basketball data
        year (int): Year to filter the data for
        
    Returns:
        dict: Dictionary with same keys but DataFrames filtered to only include specified year.
              Files without a 'Season' column are returned unfiltered.
    """
    filtered_data = {}
    for key, value in data_files.items():
        if 'Season' in value.columns:
            filtered_data[key] = value[value['Season'] == year]
        else:
            filtered_data[key] = value
    return filtered_data

def load_raw_dataset(men=True, year: int = 2024):
    """
    Loads the raw dataset for a specific year and gender, and combines the regular season and tournament results.
    
    Args:
        men (bool): If True, loads men's basketball data. If False, loads women's data. Default is True.
        year (int): The year of the dataset to load. Default is 2024.

    Returns:
        dict: Dictionary containing the loaded dataset with filename as key
    """
    if men:
        data_files = load_data(RAW_DATA_DIR, file_pattern='M*')
        data_files = {key.replace('M', '', 1) if key.startswith('M') else key: value 
                     for key, value in data_files.items()}
        data_files = combine_season_results(data_files)
        
    else:
        data_files = load_data(RAW_DATA_DIR, file_pattern='W*')
        data_files = {key.replace('W', '', 1) if key.startswith('W') else key: value 
                     for key, value in data_files.items()}
        data_files = combine_season_results(data_files)
    data_files = filter_by_year(data_files, year)
    return data_files

def load_processed_dataset(men=True, year: int = 2024, ranking_system: str = 'SEL', n_games: int = 5, force_reprocess: bool = False):
    """
    If the processed dataset for the given year and gender exists, load it. Otherwise, processes the raw dataset and saves it.
    
    Args:
        men (bool): If True, loads men's basketball data. If False, loads women's data. Default is True.
        year (int): The year of the dataset to load. Default is 2024.
        ranking_system (str): The ranking system to use for processing data. Default is 'SEL'.
        n_games (int): Number of previous games to use for rolling statistics. Default is 5.
        force_reprocess (bool): If True, processes the raw data even if the processed file already exists. Default is False.

    Returns:
        DataFrame: The processed dataset containing game results merged with rankings and statistics.
                  If the processed file already exists, loads that file.
                  If not, processes the raw data and saves/returns the result.
    """
    # Attempt to load the processed dataset if not force_reprocess
    if not force_reprocess: 
        file_name = name_processed_dataset(men, year, ranking_system, n_games)  
        if (PROCESSED_DATA_DIR / file_name).exists():
            return pd.read_csv(PROCESSED_DATA_DIR / file_name)

    # If the processed dataset does not exist, load the raw dataset
    data_files = load_raw_dataset(men=men, year=year)

    # Apply the feature pipeline to the raw dataset
    from src.data_preparation.pipelines import feature_pipeline
    feature_pipeline.set_params(
        rankings__ranking_system=ranking_system,
        rolling_stats__n_games=n_games
    )
    processed_data = feature_pipeline.transform(data_files)

    # Save the processed dataset
    save_processed_dataset(processed_data, men, year, ranking_system, n_games)
    return processed_data


def name_processed_dataset(men: bool, year: int, ranking_system: str, n_games: int):
    """
    Generate a name for the processed dataset based on the gender, year, ranking system, and number of games.
    """
    if men:
        return f'MProcessedTourneyData_{year}_{ranking_system}_{n_games}.csv'
    else:
        return f'WProcessedTourneyData_{year}_{ranking_system}_{n_games}.csv'   

def save_processed_dataset(df: pd.DataFrame, men: bool, year: int, ranking_system: str, n_games: int):
    """
    Save the processed dataset to a CSV file in the processed data directory.
    
    Args:
        df (pd.DataFrame): The processed dataset to save
        men (bool): If True, saves men's data. If False, saves women's data
        year (int): The year of the dataset
        ranking_system (str): The ranking system used to process the data
        n_games (int): Number of previous games used for rolling statistics
    """
    file_name = name_processed_dataset(men, year, ranking_system, n_games)
    df.to_csv(PROCESSED_DATA_DIR / file_name, index=False)

def combine_season_results(data_files: dict) -> pd.DataFrame:
    """
    Combines regular season and tournament results into a single DataFrame for both detailed and compact results.
    
    Args:
        data_files (dict): Dictionary containing dataset files with regular season and tournament results
                          (both detailed and compact formats if available)
        
    Returns:
        dict: Modified data_files dictionary with:
            - 'CombinedDetailedResults': Combined detailed results if available
            - 'CombinedCompactResults': Combined compact results if available
            Original separate result files are removed from the dictionary
    """
    # Get both detailed and compact results if available
    reg_season_detailed = next((k for k in data_files.keys() if k.startswith('RegularSeasonDetailed')), None)
    reg_season_compact = next((k for k in data_files.keys() if k.startswith('RegularSeasonCompact')), None)
    
    tourney_detailed = next((k for k in data_files.keys() if k.startswith('NCAATourneyDetailed')), None)
    tourney_compact = next((k for k in data_files.keys() if k.startswith('NCAATourneyCompact')), None)
    
    # Combine detailed results if available
    if reg_season_detailed and tourney_detailed:
        reg_season_df = data_files[reg_season_detailed]
        reg_season_df['NCAA_Tournament'] = False
        tourney_df = data_files[tourney_detailed]
        tourney_df['NCAA_Tournament'] = True
        detailed_results = pd.concat([reg_season_df, tourney_df], ignore_index=True)
        data_files['CombinedDetailedResults'] = detailed_results
        del data_files[reg_season_detailed]
        del data_files[tourney_detailed]
        
    # Combine compact results if available 
    if reg_season_compact and tourney_compact:
        reg_season_df = data_files[reg_season_compact]
        reg_season_df['NCAA_Tournament'] = False
        tourney_df = data_files[tourney_compact]
        tourney_df['NCAA_Tournament'] = True
        compact_results = pd.concat([reg_season_df, tourney_df], ignore_index=True)
        data_files['CombinedCompactResults'] = compact_results
        del data_files[reg_season_compact]
        del data_files[tourney_compact]

    return data_files


if __name__ == "__main__":
    data_files = load_raw_dataset(men=True, year=2024)
    print(data_files)
