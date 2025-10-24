# prediksi/serializers.py

from rest_framework import serializers
from .models import DataSensor

# Ini adalah cetakan untuk "Baki" kita, untuk menyajikan data
class DataSensorSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataSensor  # Mengambil bahan dari model DataSensor
        # Ini adalah data yang akan kita sajikan di atas baki:
        fields = ['id', 'temperature', 'humidity', 'timestamp']