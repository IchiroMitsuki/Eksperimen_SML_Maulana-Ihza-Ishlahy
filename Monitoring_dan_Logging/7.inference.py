import requests
import time
import random

url = 'http://localhost:8000/predict'

print("Memulai simulasi inference (Tekan Ctrl+C untuk menghentikan)...")
try:
    while True:
        # Generate data fitur acak seperti di dataset asli
        payload = {
            "hour": random.randint(0, 23),
            "day_of_week": random.randint(0, 6),
            "month": random.randint(1, 12),
            "is_weekend": random.choice([0, 1])
        }
        
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                result = response.json().get('predicted_imbalance_price')
                print(f"✅ Data: {payload} | Prediksi Harga: {result} JPY")
            else:
                print(f"❌ Error dari server: {response.text}")
        except requests.exceptions.ConnectionError:
            print("⏳ Menunggu server berjalan di port 8000...")
            
        # Jeda acak antar request (1 hingga 3 detik)
        time.sleep(random.uniform(1.0, 3.0))
        
except KeyboardInterrupt:
    print("\n🛑 Simulasi dihentikan.")