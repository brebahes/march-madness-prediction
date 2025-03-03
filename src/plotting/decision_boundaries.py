import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from src.training.load_processed_data import ensure_processed_data

def plot_decision_boundaries():
    # Load the processed data
    data = ensure_processed_data(year=2024, men=True, ranking_system='SEL', n_games=5)
    
    # Prepare features and target
    feature_columns = ['TeamATeamRank', 'TeamBTeamRank']
    X = data[feature_columns]
    y = data['TeamAWins']
    
    # Train the decision tree
    dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=2)
    dt_classifier.fit(X, y)
    
    # Create a mesh grid of points
    x_min, x_max = X[feature_columns[0]].min() - 1, X[feature_columns[0]].max() + 1
    y_min, y_max = X[feature_columns[1]].min() - 1, X[feature_columns[1]].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Get predictions for each point in the mesh
    Z = dt_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the decision boundaries
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    
    # Plot the actual data points
    plt.scatter(X[feature_columns[0]], X[feature_columns[1]], 
                c=y, cmap='RdYlBu', alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Team A Rank')
    plt.ylabel('Team B Rank')
    plt.title('Decision Tree Decision Boundaries\n(Team A vs Team B Rankings)')
    plt.colorbar(label='Team A Wins')
    
    # Save the plot
    plt.savefig('plots/decision_boundaries.png')
    plt.close()

if __name__ == '__main__':
    plot_decision_boundaries() 