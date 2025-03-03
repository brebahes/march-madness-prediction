import pandas as pd
import os
from pathlib import Path

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

def load_dataset(men=True, year: int = 2024):
    """
    Load either men's or women's NCAA dataset for a specific year.
    
    Args:
        men (bool): If True, loads men's basketball data. If False, loads women's data. Default is True.
        year (int): The year of the dataset to load. Default is 2024.

    Returns:
        dict: Dictionary containing the loaded dataset with filename as key
    """
    if men:
        data_files = load_data('data/raw', file_pattern='M*')
        data_files = {key.replace('M', '', 1) if key.startswith('M') else key: value 
                     for key, value in data_files.items()}
        data_files = combine_season_results(data_files)
        
    else:
        data_files = load_data('data/raw', file_pattern='W*')
        data_files = {key.replace('W', '', 1) if key.startswith('W') else key: value 
                     for key, value in data_files.items()}
        data_files = combine_season_results(data_files)
    data_files = filter_by_year(data_files, year)
    return data_files

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
        tourney_df = data_files[tourney_detailed]
        detailed_results = pd.concat([reg_season_df, tourney_df], ignore_index=True)
        data_files['CombinedDetailedResults'] = detailed_results
        del data_files[reg_season_detailed]
        del data_files[tourney_detailed]
        
    # Combine compact results if available 
    if reg_season_compact and tourney_compact:
        reg_season_df = data_files[reg_season_compact]
        tourney_df = data_files[tourney_compact]
        compact_results = pd.concat([reg_season_df, tourney_df], ignore_index=True)
        data_files['CombinedCompactResults'] = compact_results
        del data_files[reg_season_compact]
        del data_files[tourney_compact]

    return data_files


if __name__ == "__main__":
    data_files = load_dataset(men=True, year=2024)
    print(data_files)
