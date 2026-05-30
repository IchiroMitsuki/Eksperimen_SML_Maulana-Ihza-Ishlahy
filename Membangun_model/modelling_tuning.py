import pandas as pd
import numpy as np
import os
import mlflow
import dagshub
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. Inisialisasi DagsHub (Ganti dengan username dan nama repo kamu)
dagshub.init(repo_owner='IchiroMitsuki', repo_name='Eksperimen_SML_Maulana-Ihza-Ishlahy', mlflow=True)

def load_data():
    # Menyesuaikan path ke folder preprocessing hasil kriteria 1
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'preprocessing', 'Japan_ImbalancePrice_preprocessing')
    
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).squeeze()
    
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_data()
    
    # Memulai eksperimen MLflow
    mlflow.set_experiment("Imbalance_Price_Prediction_Tuning")
    
    with mlflow.start_run(run_name="RandomForest_Tuned"):
        print("Memulai proses Hyperparameter Tuning...")
        
        # 2. Mendefinisikan Model dan Hyperparameter (Tuning)
        rf = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5]
        }
        
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Mengambil model terbaik hasil tuning
        best_model = grid_search.best_estimator_
        
        # 3. Prediksi dan Evaluasi
        y_pred = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 4. MANUAL LOGGING PARAMETER DAN METRIK
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2
        })
        
        # Logging Model
        mlflow.sklearn.log_model(best_model, "random_forest_model")
        
        # 5. MEMBUAT DAN MENYIMPAN 2 ARTEFAK TAMBAHAN
        os.makedirs("artifacts", exist_ok=True)
        
        # Artefak 1: Plot Actual vs Predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted Imbalance Price')
        plot_path = "artifacts/actual_vs_predicted.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Artefak 2: Feature Importance Bar Chart
        importances = best_model.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.barh(X_train.columns, importances, color='green')
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        importance_path = "artifacts/feature_importance.png"
        plt.savefig(importance_path)
        plt.close()
        
        # Logging Artefak ke MLflow
        mlflow.log_artifact(plot_path)
        mlflow.log_artifact(importance_path)
        
        print(f"Tuning selesai. Model dan artefak telah dikirim ke DagsHub!")
        print(f"RMSE: {rmse:.4f} | R2 Score: {r2:.4f}")

if __name__ == "__main__":
    main()