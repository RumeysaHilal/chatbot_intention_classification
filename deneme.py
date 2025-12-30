# Gerekli kütüphaneyi kurun: pip install datasets
from datasets import load_dataset

# Veri setinin Türkçe (tr-TR) alt kümesini indirin
dataset = load_dataset("amazon/MASSIVE", "tr-TR")

# Veri yapısını görün (train, validation, test)
print(dataset)

# Bir örneğe bakın
# print(dataset['train'][0])
# Çıktı şuna benzeyecek:
# {'id': '...', 'locale': 'tr-TR', 'utt': 'on dakika sonraki alarmı kapat', 'annot_utt': 'on dakika sonraki alarmı kapat', 'intent': 'alarm_remove'}