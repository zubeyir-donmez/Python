import os
import shutil
from tkinter import Tk, filedialog

# Tkinter ile klasör seçimi
root = Tk()
root.withdraw()
base_dir = filedialog.askdirectory(title='Ana klasörü seçin (images ve labels içeren)')

if not base_dir:
    print('Klasör seçilmedi, çıkılıyor.')
    exit()

images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')
output_dir = os.path.join(os.path.dirname(base_dir), 'output')
output_images = os.path.join(output_dir, 'images')
output_labels = os.path.join(output_dir, 'labels')
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

for txt_file in os.listdir(labels_dir):
    if txt_file.endswith('.txt'):
        txt_path = os.path.join(labels_dir, txt_file)
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            has_zero = any(line.strip().split()[0] == '0' for line in lines if line.strip())
        if has_zero:
            # txt dosyasını kopyala
            shutil.copy2(txt_path, os.path.join(output_labels, txt_file))
            # aynı isimli resmi bul ve kopyala
            base_name = os.path.splitext(txt_file)[0]
            found_img = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                img_path = os.path.join(images_dir, base_name + ext)
                if os.path.exists(img_path):
                    shutil.copy2(img_path, os.path.join(output_images, base_name + ext))
                    found_img = True
                    break
            if not found_img:
                print(f"Uyarı: {base_name} için resim bulunamadı!")

print('İşlem tamamlandı! Uygun dosyalar output klasörüne kopyalandı.') 