import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_games_by_ranking(data: pd.DataFrame) -> go.Figure:
    """
    Plots the number of games played by each team based on their ranking.
    """
    fig = px.scatter(data, x='ATeamRank', y='BTeamRank', color='TeamA_wins', hover_data=['ATeamID', 'BTeamID'])
    return fig

