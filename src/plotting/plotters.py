import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_games_by_ranking(data: pd.DataFrame) -> go.Figure:
    """
    Plots the number of games played by each team based on their ranking.
    """
    fig = px.scatter(data, x='ATeamRank', y='BTeamRank', color='TeamA_wins', hover_data=['ATeamID', 'BTeamID'])
    return fig

def plot_tournament_bracket(data: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive visualization of the tournament bracket.
    
    Args:
        data: DataFrame containing tournament games with slots and seeds

    Returns:
        go.Figure: Interactive plotly figure showing the tournament bracket
    """
    # Filter for tournament games in the specified year
    tourney_data = data[data['NCAA_Tournament'] == True]
    
    # Sort by slot to ensure proper order
    tourney_data = tourney_data.sort_values('DayNum')

    round_map = {1: 'R1', 2: 'R2', 3: 'Sweet16', 4: 'Elite8', 5: 'Final4', 6: 'Championship'}
    tourney_data['Round'] = tourney_data['Round'].map(round_map)
    # Create the figure
    fig = go.Figure()
    
    # Define regions and their positions
    regions = {
        'W': {'x': 0, 'y': 0, 'name': 'West', 'direction': 1},
        'X': {'x': 0, 'y': 16, 'name': 'East', 'direction': 1},
        'Y': {'x': 30, 'y': 16, 'name': 'South', 'direction': -1},
        'Z': {'x': 30, 'y': 0, 'name': 'Midwest', 'direction': -1},
        'Final Four': {'x': 12, 'y': 15, 'name': 'Final Four', 'direction': 1}
    }
    
    # Add games for each round
    rounds = ['R1', 'R2', 'Sweet16', 'Elite8', 'Final4', 'Championship']
    round_positions = {
        'R1': {'x_offset': 0, 'y_offset': 0},
        'R2': {'x_offset': 3, 'y_offset': 1},
        'Sweet16': {'x_offset': 6, 'y_offset': 3},
        'Elite8': {'x_offset': 9, 'y_offset': 9},
        'Final4': {'x_offset': 0, 'y_offset': 0},
        'Championship': {'x_offset': 3, 'y_offset':0}
    }
    
    # Colors for different regions
    region_colors = {
        'W': '#FF9999',
        'X': '#99FF99',
        'Y': '#9999FF',
        'Z': '#FFFF99',
        'Final Four': '#FF99FF'
    }

    # Define the colorscale for the accuracy of the predictions
    colorscale = px.colors.diverging.RdYlGn

    # Add games for each round
    for round_name in rounds:
        round_data = tourney_data[tourney_data['Round'] == round_name].reset_index()

        for i, game in round_data.iterrows():

            region = game['Conference']
            region_info = regions[region]
            if round_name == 'Final4':
                print('here')
            # Calculate position based on region and round
            if region == 'Final Four':
                x = region_info['x'] + (i) * 6 + round_positions[round_name]['x_offset']
                y = region_info['y']
                print(x, y)
            else:
                x = region_info['x'] + region_info['direction'] * round_positions[round_name]['x_offset']
                y = region_info['y'] + round_positions[round_name]['y_offset'] + (game['bracket'] - 1) * 2

            fig.add_shape(type="rect",
                          xref="x", yref="y",
                          x0=x, y0=y,
                          x1=x+2, y1=y+2,
                          line=dict(
                              color=region_colors[region],
                              width=3,
                          ),
                          fillcolor=px.colors.sample_colorscale(colorscale, [np.abs(np.abs(1 - int(game['TeamA_wins'])) - game['prob'])])[0],
                          )
            if game['TeamA_wins']:
                label = f"<b>{game['A_TeamID']}</b> vs {game['B_TeamID']}"
            else:
                label = f"{game['A_TeamID']} vs <b>{game['B_TeamID']}</b>"
            fig.add_trace(go.Scatter(
                x=[x+1],
                y=[y+1.5],
                text=[label],
                mode="text",
                textfont=dict(
                    color="black",
                    size=12,
                )
            ))
            fig.add_trace(go.Scatter(
                x=[x+1],
                y=[y+0.5],
                text=[f"{game['prob']:.2f}"],
                mode="text",
                textfont=dict(
                    color="black",
                    size=14,
                )
            ))

            fig.update_shapes(opacity=0.3)

            # Create hover text
            # hover_text = f"Round: {round_name}<br>"
            # hover_text += f"Region: {region_info['name']}<br>"
            # hover_text += f"Slot: {game['Slot']}<br>"
            # hover_text += f"Team A: {game['ATeamID']} ({game['ASeed']})<br>"
            # hover_text += f"Team B: {game['BTeamID']} ({game['BSeed']})<br>"
            # hover_text += f"Winner: {'Team A' if game['TeamA_wins'] else 'Team B'}"
            #
            # Add game node
            # fig.add_trace(go.Scatter(
            #     x=[x],
            #     y=[y],
            #     mode='markers+text',
            #     marker=dict(
            #         size=20,
            #         color=region_colors[region],
            #         line=dict(color='black', width=1)
            #     ),
            #     text=f"{game['WTeamID']} vs {game['LTeamID']}",
            #     textposition="middle center",
            #     name=f"{round_name} - {region_info['name']}",
            #     hovertext=hover_text,
            #     hoverinfo='text'
            # ))
    #
    # # Update layout
    fig.update_layout(
        title=f"NCAA Tournament Bracket",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white"
    )

    return fig

if __name__ == '__main__':
    from src.data_preparation.dataloader import load_raw_dataset
    from src.data_preparation.pipelines import tournament_pipeline
 # Load 2024 men's basketball data
    print("Loading data...")
    # data = load_raw_dataset(men=True, year=2024)
    data = pd.read_csv('data/predictions.csv')

    # Apply the pipeline transformations
    print("Applying feature pipeline...")
    # processed_data = tournament_pipeline.transform(data)

    fig = plot_tournament_bracket(data)
    fig.show()