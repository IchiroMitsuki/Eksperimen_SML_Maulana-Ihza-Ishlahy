import pandas as pd
import os
import mlflow
from sklearn.ensemble import RandomForestRegressor

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'preprocessing', 'Japan_ImbalancePrice_preprocessing')
    
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).squeeze()
    return X_train, y_train

def main():
    X_train, y_train = load_data()
    
    # Mengaktifkan autolog sesuai syarat Kriteria Basic
    mlflow.autolog()
    
    with mlflow.start_run(run_name="Basic_Model"):
        # Model dasar tanpa hyperparameter tuning
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        print("Training model basic selesai dengan autolog.")

if __name__ == "__main__":
    main()