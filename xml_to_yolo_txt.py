import os
import xml.etree.ElementTree as ET
from tkinter import Tk, filedialog

# Tkinter ile klasör seçimi
root = Tk()
root.withdraw()
labels_dir = filedialog.askdirectory(title='labels klasörünü seçin (xml dosyaları içeren)')

if not labels_dir:
    print('Klasör seçilmedi, çıkılıyor.')
    exit()

# classes.txt dosyasını oku
classes_path = os.path.join(os.path.dirname(labels_dir), 'classes.txt')
if not os.path.exists(classes_path):
    print('classes.txt bulunamadı!')
    exit()

class_dict = {}
with open(classes_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            class_dict[parts[1].lower()] = int(parts[0])

# output klasörünü oluştur
output_dir = os.path.join(os.path.dirname(labels_dir), 'output')
os.makedirs(output_dir, exist_ok=True)

# Her xml dosyasını işle
for file in os.listdir(labels_dir):
    if file.endswith('.xml'):
        xml_path = os.path.join(labels_dir, file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Görüntü boyutunu bul
        size = root.find('size')
        if size is not None:
            img_w = float(size.find('width').text)
            img_h = float(size.find('height').text)
        else:
            # Boyut yoksa atla
            print(f"Uyarı: {file} dosyasında boyut bilgisi yok, atlanıyor.")
            continue
        yolo_lines = []
        for obj in root.findall('object'):
            name = obj.find('name').text.strip().lower()
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                # YOLO formatına çevir
                x_center = ((xmin + xmax) / 2) / img_w
                y_center = ((ymin + ymax) / 2) / img_h
                width = (xmax - xmin) / img_w
                height = (ymax - ymin) / img_h
                yolo_line = f"{name} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                yolo_lines.append(yolo_line)
        # Sonuçları txt'ye yaz
        txt_name = os.path.splitext(file)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_name)
        with open(txt_path, 'w', encoding='utf-8') as out_f:
            out_f.write('\n'.join(yolo_lines))

print(f'Tüm xml dosyaları YOLO formatına dönüştürüldü! Çıktılar: {output_dir}') 