import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def run_preprocessing():
    # Mengatur path relatif (karena script ini berada di dalam folder 'preprocessing')
    raw_data_path = '../namadataset_raw/Japan_ImbalancePrice.csv'
    save_dir = 'namadataset_preprocessing'
    
    # Membuat folder output jika belum ada
    os.makedirs(save_dir, exist_ok=True)
    
    print("Memulai otomatisasi preprocessing...")
    
    # 1. Data Loading
    df = pd.read_csv(raw_data_path)
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # 2. Feature Engineering
    df_processed = df.copy()
    df_processed['hour'] = df_processed['date_time'].dt.hour
    df_processed['day_of_week'] = df_processed['date_time'].dt.dayofweek
    df_processed['month'] = df_processed['date_time'].dt.month
    df_processed['is_weekend'] = df_processed['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df_processed = df_processed.drop(columns=['date_time'])
    
    # 3. Penanganan Missing Values
    df_processed = df_processed.ffill().bfill()
    
    # 4. Splitting Dataset
    X = df_processed.drop(columns=['Tokyo_ibprice'])
    y = df_processed['Tokyo_ibprice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # 5. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Mengembalikan data ke dalam bentuk DataFrame untuk disimpan
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # 6. Menyimpan dataset yang siap dilatih ke folder namadataset_preprocessing
    X_train_df.to_csv(f'{save_dir}/X_train.csv', index=False)
    X_test_df.to_csv(f'{save_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{save_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{save_dir}/y_test.csv', index=False)
    
    print(f"Preprocessing selesai! Dataset berhasil disimpan di {save_dir}/")

if __name__ == "__main__":
    run_preprocessing()