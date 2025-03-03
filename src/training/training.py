import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from load_processed_data import ensure_processed_data
from sklearn.model_selection import train_test_split

# Load the processed tournament data for 2023
data = ensure_processed_data(year=2024, men=True, ranking_system='SEL', n_games=5)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from src.plotting.plotters import plot_games_by_ranking

fig = plot_games_by_ranking(data)
fig.show()

# Prepare features and target
# We'll use team rankings and basic game stats as features
feature_columns = ['ATeamRank', 'BTeamRank']

# Create binary target (1 for win, 0 for loss)
y = data['TeamA_wins'] # All rows represent wins since data is from winner's perspective
X = data[feature_columns]


# Initialize and train the decision tree
dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=2)
dt_classifier.fit(X, y)

# Visualize the decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(dt_classifier, filled=True, feature_names=feature_columns, class_names=['Team A wins', 'Team B wins'])
plt.show()






# Make predictions and evaluate
y_pred = dt_classifier.predict(X)
accuracy = accuracy_score(y, y_pred)

print("\nDecision Tree Results:")
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y, y_pred))
