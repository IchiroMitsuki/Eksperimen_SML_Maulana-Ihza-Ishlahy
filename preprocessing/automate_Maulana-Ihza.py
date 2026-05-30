import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def run_preprocessing():
    # 1. Setup Direktori Absolut
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Menunjuk ke folder raw di root repository
    raw_data_path = os.path.join(script_dir, '..', 'Japan_ImbalancePrice_raw', 'Japan_ImbalancePrice.csv')
    
    # Menunjuk ke folder output di dalam folder preprocessing
    save_dir = os.path.join(script_dir, 'Japan_ImbalancePrice_preprocessing')
    
    # Robot akan membuat folder ini otomatis jika belum ada
    os.makedirs(save_dir, exist_ok=True)
    
    print("Memulai otomatisasi preprocessing...")
    
    # 2. Data Loading
    df = pd.read_csv(raw_data_path)
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # 3. Feature Engineering
    df_processed = df.copy()
    df_processed['hour'] = df_processed['date_time'].dt.hour
    df_processed['day_of_week'] = df_processed['date_time'].dt.dayofweek
    df_processed['month'] = df_processed['date_time'].dt.month
    df_processed['is_weekend'] = df_processed['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df_processed = df_processed.drop(columns=['date_time'])
    
    # 4. Penanganan Missing Values
    df_processed = df_processed.ffill().bfill()
    
    # 5. Splitting Dataset
    X = df_processed.drop(columns=['Tokyo_ibprice'])
    y = df_processed['Tokyo_ibprice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # 6. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # 7. Menyimpan dataset
    X_train_df.to_csv(os.path.join(save_dir, 'X_train.csv'), index=False)
    X_test_df.to_csv(os.path.join(save_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(save_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(save_dir, 'y_test.csv'), index=False)
    
    print(f"Preprocessing selesai! Dataset berhasil disimpan di {save_dir}/")

if __name__ == "__main__":
    run_preprocessing()