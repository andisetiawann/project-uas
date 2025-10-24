# prediksi/views.py

import random  # Tambahkan ini
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import DataSensor
from django.utils import timezone
import datetime
from django.db.models import Avg, Max, Min
from django.core.paginator import Paginator

# --- TAHAP INTEGRASI: Import library yang dibutuhkan ---
import joblib
import os
from django.conf import settings
import numpy as np
# ----------------------------------------------------


# --- TAHAP INTEGRASI: Muat SEMUA Model & Encoder ---
# Ini menentukan path ke folder ml_model
MODEL_DIR = os.path.join(settings.BASE_DIR, 'prediksi', 'ml_model')

# Muat semua file yang dibutuhkan ke dalam variabel global
AI_MODEL = joblib.load(os.path.join(MODEL_DIR, 'sensor_model.pkl'))
LABEL_ENCODER = joblib.load(os.path.join(MODEL_DIR, 'sensor_label_encoder.pkl'))
VENT_ENCODER = joblib.load(os.path.join(MODEL_DIR, 'ventilation_encoder.pkl'))
LIGHT_ENCODER = joblib.load(os.path.join(MODEL_DIR, 'light_encoder.pkl'))
# ----------------------------------------------------


# --- TAHAP INTEGRASI: Tulis Ulang Fungsi prediksi_risiko ---
def prediksi_risiko(suhu, kelembapan):
    """
    Fungsi ini menggunakan model AI 5-fitur yang sudah dilatih.
    """
    try:
        # --- NILAI ASUMSI UNTUK SENSOR YANG TIDAK ADA ---
        ph_asumsi = 7.0
        ventilasi_asumsi = 'low'
        cahaya_asumsi = 'low'
        # -----------------------------------------------

        # Ubah nilai asumsi (kategorikal) menjadi angka menggunakan encoder
        ventilasi_enc = VENT_ENCODER.transform([ventilasi_asumsi])[0]
        cahaya_enc = LIGHT_ENCODER.transform([cahaya_asumsi])[0]

        # Siapkan 5 fitur input untuk model sesuai urutan saat training di Colab
        fitur_input = np.array([[
            suhu,
            kelembapan,
            ph_asumsi,
            ventilasi_enc,
            cahaya_enc
        ]])
        
        # Lakukan prediksi (hasilnya akan berupa angka, misal 0 atau 1)
        hasil_prediksi_numerik = AI_MODEL.predict(fitur_input)
        
        # Ubah hasil prediksi angka menjadi teks ('aman'/'berbahaya') menggunakan label encoder
        risiko_teks = LABEL_ENCODER.inverse_transform(hasil_prediksi_numerik)[0]

        # Berikan rekomendasi berdasarkan hasil prediksi dari AI
        if risiko_teks == 'high': # Sesuaikan dengan nama kelas Anda, misal 'berbahaya' atau 'tinggi'
            return "Tinggi", "Model AI (5 Fitur) mendeteksi kondisi ideal untuk pertumbuhan jamur."
        else: # Jika 'low' atau 'aman'
            return "Aman", "Model AI (5 Fitur) memprediksi kondisi saat ini tidak mendukung pertumbuhan jamur."

    except Exception as e:
        return "Error", f"Terjadi kesalahan saat prediksi AI: {e}"
# ----------------------------------------------------


# --- FUNGSI LAINNYA TETAP SAMA (TIDAK PERLU DIUBAH) ---

@csrf_exempt
def simpan_data_sensor(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            temperature = data.get('temperature') 
            humidity = data.get('humidity')
            if temperature is not None and humidity is not None:
                DataSensor.objects.create(temperature=temperature, humidity=humidity)
                return JsonResponse({'status': 'sukses', 'message': 'Data berhasil disimpan'}, status=201)
            else:
                return JsonResponse({'status': 'gagal', 'message': 'Data `temperature` atau `humidity` tidak lengkap'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'status': 'gagal', 'message': 'Format JSON salah'}, status=400)
    return JsonResponse({'status': 'gagal', 'message': 'Metode tidak diizinkan'}, status=405)


from django.utils import timezone
from datetime import timedelta

def dashboard_prediksi(request):
    data_terakhir = DataSensor.objects.order_by('-timestamp').first()
    if data_terakhir:
        suhu = data_terakhir.temperature
        kelembapan = data_terakhir.humidity
        timestamp = data_terakhir.timestamp
        level_risiko, rekomendasi = prediksi_risiko(suhu, kelembapan)
    else:
        suhu, kelembapan, timestamp = (0, 0, "Belum ada data")
        level_risiko, rekomendasi = ("Tidak Diketahui", "Belum ada data sensor untuk dianalisis.")

    twelve_hours_ago = timezone.now() - timedelta(hours=12)
    data_historis = DataSensor.objects.filter(timestamp__gte=twelve_hours_ago).order_by('timestamp')
    
    if data_historis.exists():
        labels = [d.timestamp.strftime('%H:%M') for d in data_historis]
        suhu_data = [d.temperature for d in data_historis]
        kelembapan_data = [d.humidity for d in data_historis]
    else:
        labels = []
        suhu_data = []
        kelembapan_data = []

    context = {
        'suhu': suhu,
        'kelembapan': kelembapan,
        'waktu_update': timestamp,
        'level_risiko': level_risiko,
        'rekomendasi': rekomendasi,
        'chart_labels': json.dumps(labels),
        'chart_suhu_data': json.dumps(suhu_data),
        'chart_kelembapan_data': json.dumps(kelembapan_data),
    }
    
    return render(request, 'prediksi/dashboard.html', context)


from datetime import datetime, timedelta  # Pastikan ini ada

def laporan_historis(request):
    # Generate dummy data
    days_filtered = 7  # Contoh periode
    now = datetime.now()  # Ini harus berfungsi setelah impor yang benar
    chart_labels = [(now - timedelta(days=i)).strftime("%d %b") for i in range(days_filtered)]
    chart_suhu_data = [random.uniform(20.0, 30.0) for _ in range(days_filtered)]  # Suhu antara 20-30°C
    chart_kelembapan_data = [random.uniform(40.0, 60.0) for _ in range(days_filtered)]  # Kelembapan antara 40-60%

    # Stats
    stats = {
        'avg_suhu': sum(chart_suhu_data) / len(chart_suhu_data),
        'max_suhu': max(chart_suhu_data),
        'min_suhu': min(chart_suhu_data),
        'avg_kelembapan': sum(chart_kelembapan_data) / len(chart_kelembapan_data),
        'max_kelembapan': max(chart_kelembapan_data),
        'min_kelembapan': min(chart_kelembapan_data),
    }

    context = {
        'days_filtered': days_filtered,
        'chart_labels': chart_labels,
        'chart_suhu_data': chart_suhu_data,
        'chart_kelembapan_data': chart_kelembapan_data,
        'stats': stats,
        'page_obj': []  # Ganti dengan objek halaman jika ada
    }

    return render(request, 'prediksi/laporan.html', context)