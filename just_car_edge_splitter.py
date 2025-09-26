import os
import shutil

# Klasör yolları
JUST_CAR_PATH = r"D:\Verisetleri\Havacılıkta_Yapay_Zeka\uyz_2022 (Etiketli)\just_car\crop"
EDGE_TOUCHING_PATH = os.path.join(JUST_CAR_PATH, "kenar_degenler")
NOT_EDGE_TOUCHING_PATH = os.path.join(JUST_CAR_PATH, "kenar_degimeyenler")

IMAGES = "images"
LABELS = "labels"

# Kenara değme toleransı (YOLO formatında 0 veya 1'e çok yakınsa değiyor kabul edilir)
TOL = 1e-3

def touches_edge(x_center, y_center, width, height):
    left = x_center - width / 2
    right = x_center + width / 2
    top = y_center - height / 2
    bottom = y_center + height / 2
    return (
        abs(left) < TOL or abs(right - 1) < TOL or
        abs(top) < TOL or abs(bottom - 1) < TOL
    )

def main():
    # Hedef klasörleri oluştur
    for base in [EDGE_TOUCHING_PATH, NOT_EDGE_TOUCHING_PATH]:
        os.makedirs(os.path.join(base, IMAGES), exist_ok=True)
        os.makedirs(os.path.join(base, LABELS), exist_ok=True)

    images_dir = os.path.join(JUST_CAR_PATH, IMAGES)
    labels_dir = os.path.join(JUST_CAR_PATH, LABELS)
    image_files = os.listdir(images_dir)

    for img_name in image_files:
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)
        edge_flag = False
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5 and parts[0] == "0":
                        x, y, w, h = map(float, parts[1:])
                        if touches_edge(x, y, w, h):
                            edge_flag = True
                            break
        # Hedef klasör seçimi
        if edge_flag:
            target_base = EDGE_TOUCHING_PATH
        else:
            target_base = NOT_EDGE_TOUCHING_PATH
        # Kopyalama işlemleri
        shutil.copy2(os.path.join(images_dir, img_name), os.path.join(target_base, IMAGES, img_name))
        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join(target_base, LABELS, label_name))

    print("İşlem tamamlandı. Resimler ve etiketler iki klasöre ayrıldı.")

if __name__ == "__main__":
    main() 