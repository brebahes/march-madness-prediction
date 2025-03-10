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

class TournamentSlotTransformer(BaseEstimator, TransformerMixin):
    """Combines tournament seeds and assigns tournament slots to games"""
    
    def __init__(self):
        self.SEEDS = 'NCAATourneySeeds'
        self.GAMES = 'CombinedDetailedResults'
        self.SLOTS = 'NCAATourneySlots'
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Transform the input data by adding tournament seeds and slots.
        
        Args:
            X (dict): Dictionary containing the tournament seeds (NCAATourneySeeds) 
                     and games data (CombinedDetailedResults)
        
        Returns:
            DataFrame: Game results with tournament seeds and slots added
        """
        seeds_df = X[self.SEEDS]
        games_df = X[self.GAMES]
        slots_df = X[self.SLOTS]


        games_df = games_df.loc[games_df['NCAA_Tournament']==True]
        # Merge seeds for winning teams
        games_df = pd.merge(games_df, seeds_df, 
                          left_on=['Season', 'WTeamID'],
                          right_on=['Season', 'TeamID'],
                          how='left').rename(columns={'Seed': 'WSeed'}).drop(columns=['TeamID'])
        # Merge seeds for losing teams 
        games_df = pd.merge(games_df, seeds_df,
                          left_on=['Season', 'LTeamID'], 
                          right_on=['Season', 'TeamID'],
                          how='left').rename(columns={'Seed': 'LSeed'}).drop(columns=['TeamID'])

        # Split into Conference and Seed number
        games_df['WConference'] = games_df['WSeed'].astype(str).str[0]
        games_df['WSeed'] = games_df['WSeed'].astype(str).str[1:]


        games_df['LConference'] = games_df['LSeed'].astype(str).str[0]
        games_df['LSeed'] = games_df['LSeed'].astype(str).str[1:]

        first_four_index = np.logical_and(games_df['WSeed'].str.contains('a|b'), games_df['LSeed'].str.contains('a|b'))
        first_four = games_df.loc[first_four_index]
        main_tourney = games_df.loc[~first_four_index]

        # Get the round number
        first_four['Round'] = 0
        main_tourney = main_tourney.sort_values(by=['WTeamID','DayNum'])
        main_tourney['Round'] = main_tourney.groupby(by=['WTeamID']).cumcount() + 1

        games_df = pd.concat([first_four, main_tourney])

        # Sort by round
        games_df = games_df.sort_values(by='DayNum')

        # Create a new column 'Region' based on comparing winning and losing team regions
        games_df['Conference'] = np.where(
            games_df['WConference'] == games_df['LConference'],
            games_df['WConference'],
            'Final Four'
        )
        # Drop region columns as they are no longer needed
        games_df = games_df.drop(['WConference', 'LConference'], axis=1)

        regular_season = X[self.GAMES].loc[X[self.GAMES]['NCAA_Tournament']==False]
        regular_season[['WSeed', 'LSeed', 'Round', 'Conference']] = np.nan

        games_df = pd.concat([regular_season, games_df])

        return games_df
