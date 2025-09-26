import os
from tkinter import Tk, filedialog
from PIL import Image

def yolo_obb_to_pixel(label_line, img_width, img_height):
    """Convert YOLO OBB format to pixel coordinates."""
    parts = label_line.strip().split()
    obj_id = parts[0]
    coords = list(map(float, parts[1:]))
    pixel_coords = [int(coords[i] * img_width if i % 2 == 0 else coords[i] * img_height) for i in range(len(coords))]
    return obj_id, pixel_coords

def pixel_to_yolo_obb(obj_id, pixel_coords, crop_width, crop_height):
    """Convert pixel coordinates back to YOLO OBB format."""
    normalized_coords = [
        round(pixel_coords[i] / crop_width if i % 2 == 0 else pixel_coords[i] / crop_height, 6)
        for i in range(len(pixel_coords))
    ]
    return f"{obj_id} " + " ".join(map(str, normalized_coords))

def crop_images_and_labels_to_640x640():
    # Klasör seçimi
    Tk().withdraw()  # Tkinter GUI'yi gizle
    folder_path = filedialog.askdirectory(title="Klasör Seçin")
    if not folder_path:
        print("Klasör seçilmedi.")
        return

    images_path = os.path.join(folder_path, "images")
    labels_path = os.path.join(folder_path, "labels")
    output_images_path = os.path.join(folder_path, "output", "images")
    output_labels_path = os.path.join(folder_path, "output", "labels")
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)

    # Resimleri ve etiketleri işle
    for file_name in os.listdir(images_path):
        file_path = os.path.join(images_path, file_name)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            with Image.open(file_path) as img:
                width, height = img.size
                print(f"İşleniyor: {file_name} ({width}x{height})")

                # İlgili etiket dosyasını oku
                label_file_name = os.path.splitext(file_name)[0] + ".txt"
                label_file_path = os.path.join(labels_path, label_file_name)
                labels = []
                if os.path.exists(label_file_path):
                    with open(label_file_path, "r") as label_file:
                        labels = [yolo_obb_to_pixel(line, width, height) for line in label_file]

                # 640x640 parçalarına böl
                for h, y in enumerate(range(0, height, 640)):
                    if y + 640 > height:  # Artan kısmı alma
                        break
                    for w, x in enumerate(range(0, width, 640)):
                        if x + 640 > width:  # Artan kısmı alma
                            break
                        cropped_img = img.crop((x, y, x + 640, y + 640))
                        cropped_file_name = f"{os.path.splitext(file_name)[0]}_{h}_{w}.png"
                        cropped_img.save(os.path.join(output_images_path, cropped_file_name))
                        print(f"Kaydedildi: {cropped_file_name}")

                        # Etiketleri kırpılmış alana göre filtrele ve kaydet
                        cropped_label_file_name = f"{os.path.splitext(file_name)[0]}_{h}_{w}.txt"
                        cropped_label_file_path = os.path.join(output_labels_path, cropped_label_file_name)
                        with open(cropped_label_file_path, "w") as cropped_label_file:
                            for obj_id, pixel_coords in labels:
                                # Kırpılmış alan içinde kalan etiketleri filtrele
                                if all(x <= pixel_coords[i] < x + 640 and y <= pixel_coords[i + 1] < y + 640 for i in range(0, len(pixel_coords), 2)):
                                    # Koordinatları kırpılmış alana göre yeniden hesapla
                                    adjusted_coords = [
                                        pixel_coords[i] - x if i % 2 == 0 else pixel_coords[i] - y
                                        for i in range(len(pixel_coords))
                                    ]
                                    # Tekrar YOLO formatına dönüştür
                                    yolo_line = pixel_to_yolo_obb(obj_id, adjusted_coords, 640, 640)
                                    cropped_label_file.write(yolo_line + "\n")
                        print(f"Kaydedildi: {cropped_label_file_name}")

    print("Tüm resimler ve etiketler işlendi.")

if __name__ == "__main__":
    crop_images_and_labels_to_640x640()
