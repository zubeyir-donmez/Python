import os
import shutil
from tkinter import filedialog, Tk

# Klasör seçimi
root = Tk()
root.withdraw()
base_dir = filedialog.askdirectory(title="Ana klasörü seçin (images ve labels içinde olacak)")

labels_dir = os.path.join(base_dir, "labels")
images_dir = os.path.join(base_dir, "images")

# no_txt_images klasörü bir üst dizinde olacak
parent_dir = os.path.dirname(base_dir)
no_txt_images_dir = os.path.join(parent_dir, "no_txt_images")
os.makedirs(no_txt_images_dir, exist_ok=True)

# .txt dosyalarını kontrol et
for txt_file in os.listdir(labels_dir):
    if txt_file.endswith(".txt"):
        txt_path = os.path.join(labels_dir, txt_file)

        # İçeriği oku ve boş mu kontrol et
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:  # Boş veya sadece boşluk içeriyor
            # Label dosyasını sil
            os.remove(txt_path)
            print(f"Boş label silindi: {txt_file}")

            # Aynı isimli görseli bul ve taşı (jpg veya png olabilir)
            base_name = os.path.splitext(txt_file)[0]
            for ext in [".jpg", ".jpeg", ".png"]:
                image_path = os.path.join(images_dir, base_name + ext)
                if os.path.exists(image_path):
                    shutil.move(image_path, os.path.join(no_txt_images_dir, os.path.basename(image_path)))
                    print(f"İlgili görsel no_txt_images klasörüne taşındı: {base_name + ext}")
                    break
