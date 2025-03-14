import os
import sys

from waitress.utilities import long_day_reg

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_preparation.dataloader import load_processed_dataset
from sklearn.linear_model import LogisticRegression
import mlflow
from src.training.ml_utils import train_test_split
from sklearn.metrics import accuracy_score

# Set the tracking URI to the local MLflow server
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("Logistic Regression 2")
mlflow.autolog()

# Define the years to train the model on
years = range(2016, 2023)

# Define the parameters for the data pipelines and feature extraction
params = {  
    'ranking_system': 'SEL',
    'n_games': 10,
    'men': True
}

# Define the hyperparameters for the model
hyperparams = {
    'random_state': 42	
}

# Initialize the model
logreg_classifier = LogisticRegression(random_state=hyperparams['random_state'])

training_features = ['A_TeamRank', 'B_TeamRank']

# Train the model on each year
for year in years:
    with mlflow.start_run(run_name=f"logreg_{year}"):
        data = load_processed_dataset(year=year, **params)
        data = data.dropna(subset=training_features)

        train_data, test_data = train_test_split(data)
        X_train = train_data[training_features]
        y_train = train_data['TeamA_wins']
        logreg_classifier.fit(X_train, y_train)
        
        X_test = test_data[training_features]
        y_test = test_data['TeamA_wins']
        y_pred = logreg_classifier.predict(X_test)
        y_proba = logreg_classifier.predict_proba(X_test)

        test_data['prediction'] = y_pred
        test_data['prob'] = y_proba[:, 1]
        #
        # test_data.to_csv('data/predictions.csv', index=False)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        mlflow.log_metric("test_accuracy", accuracy)

        from src.plotting.plotters import plot_tournament_bracket
        fig = plot_tournament_bracket(test_data)
        fig.write_html('result.html')
        mlflow.log_artifact('result.html')

        os.remove('result.html')









