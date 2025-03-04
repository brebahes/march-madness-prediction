import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_preparation.dataloader import load_processed_dataset
from sklearn.tree import DecisionTreeClassifier
import mlflow

# Set the tracking URI to the local MLflow server
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("Decision Tree")
mlflow.autolog()

# Define the years to train the model on
years = range(2016, 2018)

# Define the parameters for the data pipelines and feature extraction
params = {  
    'ranking_system': 'SEL',
    'n_games': 5,
    'men': True
}

# Define the hyperparameters for the model
hyperparams = {
    'max_depth': 2,     
    'random_state': 42	
}

# Initialize the model
dt_classifier = DecisionTreeClassifier(max_depth=hyperparams['max_depth'], random_state=hyperparams['random_state'])

# Train the model on each year
for year in years:
    with mlflow.start_run(run_name=f"dtree_{year}"):
        data = load_processed_dataset(year=year, **params)
        X = data[['ATeamRank', 'BTeamRank']]
        y = data['TeamA_wins']
        dt_classifier.fit(X, y)








