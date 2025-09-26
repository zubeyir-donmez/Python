import os
from tkinter import Tk, filedialog
import unicodedata

# Değişim sözlüğü
replace_dict = {
    'i̇nsan': '10',
    'uap': '4',
    'uai̇': '5',
    'taşıt': '0'
}

# Tkinter ile klasör seçimi
root = Tk()
root.withdraw()
labels_dir = filedialog.askdirectory(title='labels klasörünü seçin (txt dosyaları içeren)')

if not labels_dir:
    print('Klasör seçilmedi, çıkılıyor.')
    exit()

for file in os.listdir(labels_dir):
    if file.endswith('.txt'):
        txt_path = os.path.join(labels_dir, file)
        new_lines = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                print(parts[0])
                if parts:
                    obj_name = unicodedata.normalize('NFC', parts[0].strip().lower())
                    if obj_name in replace_dict:
                        
                        parts[0] = replace_dict[obj_name]
                new_lines.append(' '.join(parts))
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))

print('Belirtilen nesne adları başarıyla değiştirildi!') 