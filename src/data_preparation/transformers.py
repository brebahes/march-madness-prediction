from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class RankingTransformer(BaseEstimator, TransformerMixin):
    """Combines games and rankings data into a single DataFrame"""
    
    def __init__(self, ranking_system='SEL', detailed_results=True):
        """
        Initialize the transformer with the ranking system to use and whether to use detailed results or not.
        """
        self.ranking_system = ranking_system
        self.detailed_results = detailed_results
        # Determine which files to use based on whether we want detailed or compact results
        self.GAMES = 'CombinedDetailedResults' if detailed_results else 'CombinedCompactResults'
        self.RANKINGS = 'MasseyOrdinals'
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Transform the input data by merging games data with rankings.
        
        Args:
            X (dict): Dictionary containing the rankings (MasseyOrdinals) and games data 
                     (NCAATourneyDetailedResults or NCAATourneyCompactResults)
        
        Returns:
            DataFrame: Game results merged with team rankings for both winning and losing teams
        """
        from src.data_preparation.helpers import process_rankings, merge_with_latest_ranking
        
        rankings_df = X[self.RANKINGS]
        games_df = X[self.GAMES]
        
        # Process rankings
        rankings_processed = process_rankings(rankings_df, self.ranking_system)
        
        # Merge rankings for both teams
        w_rankings = merge_with_latest_ranking(games_df, rankings_processed,'WTeamID','WTeamRank')
        all_rankings = merge_with_latest_ranking(w_rankings, rankings_processed, 'LTeamID', 'LTeamRank')
        
        return all_rankings

class RollingStatsTransformer(BaseEstimator, TransformerMixin):
    """Transform game data into rolling statistics features"""
    
    def __init__(self, n_games=5):
        self.n_games = n_games
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        X should be a DataFrame with game data and rankings
        """
        from src.data_preparation.  helpers import calculate_rolling_stats, merge_rolling_stats
        
        # Calculate rolling stats
        rolling_stats = calculate_rolling_stats(X, self.n_games)
        
        # Merge rolling stats for both teams
        games_with_stats = X.copy()
        
        # Add winning team stats
        games_with_stats = merge_rolling_stats(
            games_with_stats, 
            rolling_stats, 
            'WTeamID'
        )

        # Rename columns to distinguish between winning and losing team stats
        games_with_stats.columns = [
            f"W{col}" if col.startswith('Roll') 
            else col for col in games_with_stats.columns
        ]
        # Add losing team stats
        games_with_stats = merge_rolling_stats(
            games_with_stats, 
            rolling_stats, 
            'LTeamID'
        )

        games_with_stats.columns = [
            f"L{col}" if col.startswith('Roll') 
            else col for col in games_with_stats.columns
        ]

        
        return games_with_stats 
    
class RandomizeTeamsTransformer(BaseEstimator, TransformerMixin):
    """Randomly swap winning and losing team columns to avoid bias"""
    
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """
        Randomly swap winning and losing team columns to avoid bias
        
        Args:
            X: DataFrame with game data containing W/L prefixed columns
            
        Returns:
            DataFrame with randomized team assignments (TeamA/TeamB prefixes)
        """
        # Make copy to avoid modifying original
        X_random = X.copy()

        # Rename columns that start with W or L to A or B respectively
        new_columns = {}
        for col in X_random.columns:
            if col.startswith('W'):
                new_columns[col] = 'A_' + col[1:]  # Replace first W with A
            elif col.startswith('L'):
                new_columns[col] = 'B_' + col[1:]  # Replace first L with B
        X_random.rename(columns=new_columns, inplace=True)
        # Add a column with a random boolean value
        X_random['TeamA_wins'] = self.rng.rand(len(X_random)) > 0.5

        # Slice the dataframe where TeamA_wins is True
        X_random_team_a_wins = X_random[X_random['TeamA_wins'] == True]

        # Slice the dataframe where TeamA_wins is False
        X_random_team_b_wins = X_random[X_random['TeamA_wins'] == False]
        
        # For rows where TeamB wins, swap A and B prefixes
        a_cols = [col for col in X_random_team_b_wins.columns if col.startswith('A')]
        b_cols = [col for col in X_random_team_b_wins.columns if col.startswith('B')]
        rename_dict = {
            **{a_col: 'B' + a_col[1:] for a_col in a_cols},
            **{b_col: 'A' + b_col[1:] for b_col in b_cols}
        }
        X_random_team_b_wins.rename(columns=rename_dict, inplace=True)

        # Concatenate the two dataframes
        X_random = pd.concat([X_random_team_a_wins, X_random_team_b_wins])
        
        return X_random