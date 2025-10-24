import serial
import requests
import time
import json

# --- KONFIGURASI ---
# Ganti 'COM3' dengan Port Arduino Anda (cek di Arduino IDE > Tools > Port)
SERIAL_PORT = 'COM6' 
BAUD_RATE = 9600
API_URL = 'http://127.0.0.1:8000/api/simpan-data-sensor/'

print("Mencoba terhubung ke port serial...")

try:
    # Menyiapkan koneksi ke Serial Port
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
    print(f"Berhasil terhubung ke {SERIAL_PORT}")

    while True:
        # Baca satu baris data dari Arduino
        line = ser.readline().decode('utf-8').strip()

        # Pastikan baris yang dibaca adalah data sensor, bukan teks lain
        if "Suhu" in line and "Kelembapan" in line:
            print(f"Menerima data mentah: {line}")

            try:
                # --- Proses Parsing Data ---
                # Memecah string menjadi dua bagian: "Kelembapan: XX.XX %" dan "Suhu: YY.YY *C"
                parts = line.split('\t')

                # Mengambil angka dari masing-masing bagian
                humidity_str = parts[0].split(':')[1].replace('%', '').strip()
                temperature_str = parts[1].split(':')[1].replace('*C', '').strip()

                # Mengubah string menjadi angka float
                humidity = float(humidity_str)
                temperature = float(temperature_str)

                # --- Menyiapkan Data untuk Dikirim ---
                payload = {
                    'temperature': temperature,
                    'humidity': humidity
                }

                print(f"Data yang akan dikirim: {payload}")

                # --- Mengirim Data ke Django API ---
                response = requests.post(API_URL, json=payload)

                # Cek status respons dari server
                if response.status_code == 201:
                    print(">> Sukses: Data berhasil dikirim ke server Django.")
                else:
                    print(f">> Gagal: Server merespons dengan status {response.status_code} - {response.text}")

            except (IndexError, ValueError) as e:
                print(f"!! Error parsing data: {e}. Melewati baris ini.")
            except requests.exceptions.RequestException as e:
                print(f"!! Error koneksi ke server: {e}. Pastikan server Django berjalan.")

        # Beri sedikit jeda agar tidak membebani CPU
        time.sleep(1)

except serial.SerialException as e:
    print(f"!! Gagal terhubung ke port serial {SERIAL_PORT}.")
    print(f"!! Pastikan Anda memilih Port yang benar dan tidak sedang digunakan oleh Serial Monitor.")
    print(f"!! Error: {e}")