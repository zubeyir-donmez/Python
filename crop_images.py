import cv2
import os

def crop_and_resize_images(input_folder, output_folder, target_size=(640, 640)):
    # Çıkış klasörünü oluştur
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Klasördeki tüm dosyaları listele
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Görüntüyü yükle
        image = cv2.imread(file_path)
        if image is None:
            print(f"Görüntü yüklenemedi: {filename}")
            continue

        # Görüntü boyutlarını al
        height, width = image.shape[:2]

        # Kırpma için merkez koordinatlarını hesapla
        center_x, center_y = width // 2, height // 2
        crop_size = min(height, width)  # Kare kırpma için en küçük boyutu al

        # Kırpma sınırlarını hesapla
        x1 = max(center_x - crop_size // 2, 0)
        y1 = max(center_y - crop_size // 2, 0)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        # Görüntüyü kırp
        cropped_image = image[y1:y2, x1:x2]

        # Kırpılan görüntüyü yeniden boyutlandır
        resized_image = cv2.resize(cropped_image, target_size)

        # Çıkış dosyasını kaydet
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, resized_image)
        print(f"Kırpıldı ve kaydedildi: {output_path}")

# Kullanım
input_folder = 'D:/Kodlar/Python/yum_track/Yumurta'  # Giriş klasörü
output_folder = 'D:/Kodlar/Python/yum_track/Croped_egg'  # Çıkış klasörü
crop_and_resize_images(input_folder, output_folder)