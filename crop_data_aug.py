import os
from PIL import Image

class CropDataAugmentor:
    def __init__(self, images, labels=None, crop_size=640, save_folder=None):
        """
        images: Klasör veya resim dosyası yolu
        crop_size: Kare kırpma boyutu (örn: 640)
        labels: Etiket dosyalarının bulunduğu klasör (opsiyonel)
        save_folder: Sonuçların kaydedileceği klasör (verilmezse otomatik oluşturulur)
        """
        self.images = images
        self.labels = labels
        self.crop_size = crop_size
        # Kayıt klasörü ayarlanıyor
        if save_folder is None:
            base_dir = os.path.dirname(os.path.abspath(images))
            save_folder = os.path.join(base_dir, "crop_output")
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                print(f"Kayıt klasörü oluşturuldu: {save_folder}")
        else:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                print(f"Kayıt klasörü oluşturuldu: {save_folder}")
        self.save_folder = save_folder
        self.crop_and_save()

    def crop_and_save(self):
        """
        Eğer yol bir klasörse içindeki tüm resimlere, dosya ise sadece o resme işlem uygular.
        """
        if os.path.isdir(self.images):
            # Klasördeki tüm resim dosyalarını bul
            for file in os.listdir(self.images):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(self.images, file)
                    self.crop_and_save_image(full_path)
        elif os.path.isfile(self.images) and self.images.lower().endswith((".jpg", ".jpeg", ".png")):
            # Tek bir resim dosyasına işlem uygula
            self.crop_and_save_image(self.images)
        else:
            print("Belirtilen yol bir klasör veya desteklenen bir resim dosyası olmalı.")

    def crop_and_save_image(self, image_path):
        """
        Verilen resim dosyasını istenen boyutlarda kırpar ve kaydeder. Eğer labels klasörü verilmişse, etiketleri de böler.
        """
        img = Image.open(image_path)
        width, height = img.size
        if self.crop_size > width or self.crop_size > height:
            print(f"{image_path} için kırpma boyutu resimden büyük, işlem yapılmadı.")
            return
        n_w = 1
        while (n_w * self.crop_size) < width:
            n_w += 1
        n_h = 1
        while (n_h * self.crop_size) < height:
            n_h += 1
        tol_w = (width - self.crop_size) // (n_w - 1) if n_w > 1 else 0
        tol_h = (height - self.crop_size) // (n_h - 1) if n_h > 1 else 0
        # Eğer label dosyası varsa oku
        label_lines = None
        if self.labels is not None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(self.labels, base_name + ".txt")
            if os.path.exists(label_path):
                with open(label_path, "r", encoding="utf-8") as f:
                    label_lines = f.readlines()
        for i in range(n_w):  # Sütun
            left = i * tol_w if n_w > 1 else 0
            right = left + self.crop_size
            if right > width:
                left = width - self.crop_size
                right = width
            for j in range(n_h):  # Satır
                upper = j * tol_h if n_h > 1 else 0
                lower = upper + self.crop_size
                if lower > height:
                    upper = height - self.crop_size
                    lower = height
                crop = img.crop((left, upper, right, lower))
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                crop_name = f"{base_name}_{j}_{i}.png"
                crop.save(os.path.join(self.save_folder, crop_name))
                # Eğer label varsa, kırpılan bölgeye düşen etiketleri de kaydet
                if label_lines is not None:
                    crop_labels = []
                    for line in label_lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls, x, y, w, h = parts[:5]
                        x = float(x)
                        y = float(y)
                        w = float(w)
                        h = float(h)
                        # YOLO formatında x, y, w, h (orijinal resme göre, 0-1 arası)
                        abs_x = x * width
                        abs_y = y * height
                        abs_w = w * width
                        abs_h = h * height
                        # Kırpılan bölgeye düşüyor mu?
                        crop_left = left
                        crop_top = upper
                        crop_right = right
                        crop_bottom = lower
                        box_left = abs_x - abs_w / 2
                        box_top = abs_y - abs_h / 2
                        box_right = abs_x + abs_w / 2
                        box_bottom = abs_y + abs_h / 2
                        # Kutu kırpılan bölgeyle kesişiyor mu?
                        if not (box_right < crop_left or box_left > crop_right or box_bottom < crop_top or box_top > crop_bottom):
                            # Kesişen kısmı kırpılmış resme göre normalize et
                            new_x = (abs_x - crop_left) / self.crop_size
                            new_y = (abs_y - crop_top) / self.crop_size
                            new_w = abs_w / self.crop_size
                            new_h = abs_h / self.crop_size
                            # Kırpılan resmin dışına taşan kutuları kırp
                            if 0 <= new_x <= 1 and 0 <= new_y <= 1:
                                crop_labels.append(f"{cls} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}\n")
                    # Yeni label dosyasını kaydet
                    crop_label_name = f"{base_name}_{j}_{i}.txt"
                    with open(os.path.join(self.save_folder, crop_label_name), "w", encoding="utf-8") as f:
                        f.writelines(crop_labels)

# Kullanım örneği:
# Klasör için:
# augmenter = CropDataAugmentor("images_folder_path", 640, labels="labels_folder_path")
# Tek resim için:
augmenter = CropDataAugmentor("D:/Verisetleri/Havacılıkta_Yapay_Zeka/uyz_2022 (Etiketli)/without_car/images", labels="D:/Verisetleri/Havacılıkta_Yapay_Zeka/uyz_2022 (Etiketli)/without_car/labels", crop_size=750)
# Sonucu istediğiniz dosya yoluna kaydetmek için:
# augmenter = CropDataAugmentor("images_folder_path", 640, labels="labels_folder_path", save_folder="save_folder_path")
augmenter = CropDataAugmentor("D:/Verisetleri/Havacılıkta_Yapay_Zeka/uyz_2022 (Etiketli)/without_car/images", labels="D:/Verisetleri/Havacılıkta_Yapay_Zeka/uyz_2022 (Etiketli)/without_car/labels", crop_size=1080)

# Örnek kullanım:
# images = "D:/Verisetleri/Havacılıkta_Yapay_Zeka/Dota (Etiketli)/images_without_txt/P0597.png"
# labels = "D:/Verisetleri/Havacılıkta_Yapay_Zeka/Dota (Etiketli)/labels_without_txt"
# augmenter = CropDataAugmentor(images, 640, labels=labels)