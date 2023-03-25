from .models import Trainer
from rest_framework import serializers

class TrainerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Trainer
        fields = ('user','session_id', 'datafolder', 'modelname')