import os
import shutil

# Klasör yolları (gerekirse güncelleyebilirsin)
ALL_DATA_PATH = r"D:\Verisetleri\Havacılıkta_Yapay_Zeka\uyz_2022 (Etiketli)\all_data"
JUST_CAR_PATH = r"D:\Verisetleri\Havacılıkta_Yapay_Zeka\uyz_2022 (Etiketli)\just_car"
WITHOUT_CAR_PATH = r"D:\Verisetleri\Havacılıkta_Yapay_Zeka\uyz_2022 (Etiketli)\without_car"

# Alt klasörler
IMAGES = "images"
LABELS = "labels"

def main():
    # without_car klasörünü ve alt klasörlerini oluştur
    os.makedirs(os.path.join(WITHOUT_CAR_PATH, IMAGES), exist_ok=True)
    os.makedirs(os.path.join(WITHOUT_CAR_PATH, LABELS), exist_ok=True)

    # all_data ve just_car images klasörlerindeki dosya isimlerini al
    all_data_images = set(os.listdir(os.path.join(ALL_DATA_PATH, IMAGES)))
    just_car_images = set(os.listdir(os.path.join(JUST_CAR_PATH, IMAGES)))

    # Sadece all_data'da olan resimleri bul
    without_car_images = all_data_images - just_car_images

    for img_name in without_car_images:
        # Resmi kopyala
        src_img = os.path.join(ALL_DATA_PATH, IMAGES, img_name)
        dst_img = os.path.join(WITHOUT_CAR_PATH, IMAGES, img_name)
        shutil.copy2(src_img, dst_img)

        # Etiket dosyasını kopyala (uzantısı .txt olmalı)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        src_label = os.path.join(ALL_DATA_PATH, LABELS, label_name)
        dst_label = os.path.join(WITHOUT_CAR_PATH, LABELS, label_name)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)

    print(f"{len(without_car_images)} adet resim ve etiketleri without_car klasörüne kopyalandı.")

if __name__ == "__main__":
    main() 