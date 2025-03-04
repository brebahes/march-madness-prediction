import pandas as pd
import numpy as np

def merge_with_latest_ranking(game_data, rank_data, team_id_col):
    """
    Merges game data with rankings using the most recent available ranking before each game
    
    Args:
        game_data: DataFrame with game results
        rank_data: DataFrame with rankings
        team_id_col: Column name for team ID in game data
        
    Returns:
        DataFrame: Merged data with most recent rankings before each game
    """
    # Create a cross join between games and rankings for the same season and team
    merged = pd.merge(
        game_data,
        rank_data,
        left_on=['Season', team_id_col],
        right_on=['Season', 'TeamID'],
        how='left'
    )
    
    # Keep only rankings from before the game
    merged = merged[merged['DayNum'] >= merged['RankingDayNum']]
    
    # For each game and team, keep only the most recent ranking
    merged = merged.sort_values('RankingDayNum').groupby(
        ['Season', 'DayNum', team_id_col]
    ).last().reset_index()
    
    return merged

def process_rankings(rankings_df, ranking_system=None):
    """
    Process rankings data to either filter for a specific system or compute median across systems
    
    Args:
        rankings_df: DataFrame containing all rankings data
        ranking_system: String or list of strings specifying which ranking system(s) to use (e.g., 'SEL' or ['SEL', 'POM']). 
                       If None, uses median of all systems.
    
    Returns:
        DataFrame: Processed rankings with either single system or median ranks
    """
    if ranking_system:
        if isinstance(ranking_system, str):
            # Filter for specific ranking system
            processed_rankings = rankings_df[rankings_df['SystemName'] == ranking_system].copy()           
            # Check if any rankings were found for the specified system
            if processed_rankings.empty:
                raise ValueError(f"No rankings found for system '{ranking_system}'")
        else:
            # Filter for list of systems and calculate median
            filtered_rankings = rankings_df[rankings_df['SystemName'].isin(ranking_system)]
            processed_rankings = (filtered_rankings.groupby(['Season', 'TeamID', 'RankingDayNum'])
                                ['OrdinalRank'].median()
                                .reset_index())
            processed_rankings['SystemName'] = 'MEDIAN'  # Add system name for consistency
    else:
        # Calculate median rank across all systems
        processed_rankings = (rankings_df.groupby(['Season', 'TeamID', 'RankingDayNum'])
                            ['OrdinalRank'].median()
                            .reset_index())
        processed_rankings['SystemName'] = 'MEDIAN'  # Add system name for consistency
        
    return processed_rankings

def calculate_rolling_stats(games_df, n_games=5):
    """
    Calculate rolling statistics for each team based on their previous n games
    
    Args:
        games_df: DataFrame containing regular season games
        n_games: Number of previous games to consider (default=5)
    
    Returns:
        DataFrame: Team-level rolling statistics
    """
    # List of statistical columns to compute rolling averages
    score_cols = ['Score', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 
                  'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk']
    
    # Create separate dataframes for winning and losing teams
    winners = games_df[[f'W{col}' for col in score_cols] + 
                      ['Season', 'DayNum', 'WTeamID']].copy()
    winners.columns = score_cols + ['Season', 'DayNum', 'TeamID']
    
    losers = games_df[[f'L{col}' for col in score_cols] + 
                      ['Season', 'DayNum', 'LTeamID']].copy()
    losers.columns = score_cols + ['Season', 'DayNum', 'TeamID']
    
    # Combine all games and sort by date
    all_games = pd.concat([winners, losers])
    all_games = all_games.sort_values(['Season', 'DayNum'])
    
    # Calculate rolling averages
    rolling_stats = pd.DataFrame()
    
    for team in all_games['TeamID'].unique():
        team_games = all_games[all_games['TeamID'] == team].copy()
        
        # Calculate rolling stats for each statistical column
        for col in score_cols:
            team_games[f'Roll{col}'] = team_games[col].rolling(
                window=n_games, min_periods=1).mean()
            
        # Add additional derived statistics
        team_games['RollFGPct'] = team_games['RollFGM'] / team_games['RollFGA']
        team_games['RollFG3Pct'] = team_games['RollFGM3'] / team_games['RollFGA3']
        team_games['RollFTPct'] = team_games['RollFTM'] / team_games['RollFTA']
        team_games['RollTRB'] = team_games['RollOR'] + team_games['RollDR']  # Total rebounds
        
        rolling_stats = pd.concat([rolling_stats, team_games])
    
    # Keep only rolling statistics and identifiers
    rolling_cols = [col for col in rolling_stats.columns if col.startswith('Roll')] + \
                   ['Season', 'DayNum', 'TeamID']
    rolling_stats = rolling_stats[rolling_cols].copy()
    
    return rolling_stats

def merge_rolling_stats(game_data, rolling_stats, team_id_col):
    """
    Merge rolling statistics with game data for a specific team
    
    Args:
        game_data: DataFrame with game results
        rolling_stats: DataFrame with rolling statistics
        team_id_col: Column name for team ID (e.g., 'WTeamID' or 'LTeamID')
    
    Returns:
        DataFrame: Game data with rolling statistics merged
    """
    return pd.merge(
        game_data,
        rolling_stats,
        left_on=['Season', 'DayNum', team_id_col],
        right_on=['Season', 'DayNum', 'TeamID'],
        how='left'
    )