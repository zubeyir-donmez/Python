import os
from collections import Counter
from tkinter import Tk, filedialog

# Tkinter ile klasör seçimi
root = Tk()
root.withdraw()
labels_dir = filedialog.askdirectory(title='labels klasörünü seçin (txt dosyaları içeren)')

if not labels_dir:
    print('Klasör seçilmedi, çıkılıyor.')
    exit()

object_counter = Counter()

for file in os.listdir(labels_dir):
    if file.endswith('.txt'):
        txt_path = os.path.join(labels_dir, file)
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    object_name = parts[0]
                    object_counter[object_name] += 1

print('Nesne sayıları:')
for obj, count in object_counter.most_common():
    print(f"{obj}: {count}")

toplam = sum(object_counter.values())
print(f"Toplam nesne adedi: {toplam}") 