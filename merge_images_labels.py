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

# Hedef klasörler
merged_images = os.path.join(base_dir, 'merged_images')
merged_labels = os.path.join(base_dir, 'merged_labels')
os.makedirs(merged_images, exist_ok=True)
os.makedirs(merged_labels, exist_ok=True)

# Alt klasörleri sırala
image_folders = [os.path.join(images_dir, f) for f in sorted(os.listdir(images_dir)) if os.path.isdir(os.path.join(images_dir, f))]
label_folders = [os.path.join(labels_dir, f) for f in sorted(os.listdir(labels_dir)) if os.path.isdir(os.path.join(labels_dir, f))]

counter = 0
for img_folder, lbl_folder in zip(image_folders, label_folders):
    img_files = sorted([f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))])
    for img_file in img_files:
        name, ext = os.path.splitext(img_file)
        src_img_path = os.path.join(img_folder, img_file)
        src_lbl_path = os.path.join(lbl_folder, name + '.xml')
        new_name = f"frame_{counter}{ext}"
        new_lbl_name = f"frame_{counter}.xml"
        # Resmi kopyala
        shutil.copy2(src_img_path, os.path.join(merged_images, new_name))
        # XML dosyası varsa kopyala ve yeniden adlandır
        if os.path.exists(src_lbl_path):
            shutil.copy2(src_lbl_path, os.path.join(merged_labels, new_lbl_name))
        else:
            print(f"Uyarı: {src_lbl_path} bulunamadı!")
        counter += 1

print(f"Birleştirme tamamlandı! Toplam {counter} dosya işlendi.")
print(f"Birleştirilmiş resimler: {merged_images}")
print(f"Birleştirilmiş etiketler: {merged_labels}") 