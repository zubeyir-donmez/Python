from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

def rgb_to_hsi(rgb_image):
    # BGR'den RGB'ye dönüştür
    rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(rgb)
    
    # Normalize
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    
    # Intensity
    i = (r + g + b) / 3.0
    
    # Saturation
    min_rgb = np.minimum(np.minimum(r, g), b)
    s = 1 - (3 / (r + g + b + 1e-6)) * min_rgb
    
    # Hue
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g)**2 + (r - b) * (g - b))
    theta = np.arccos(num / (den + 1e-6))
    h = np.where(b <= g, theta, 2 * np.pi - theta)
    h = h * 180 / np.pi  # Radyandan dereceye çevir
    
    # 0-255 aralığına dönüştür
    h = np.clip(h, 0, 255)
    s = np.clip(s * 255, 0, 255)
    i = np.clip(i * 255, 0, 255)
    
    return cv2.merge([h.astype(np.uint8), s.astype(np.uint8), i.astype(np.uint8)])

def get_dominant_color(roi, mask):
    # ROI'yi HSV'ye dönüştür
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Maskelenmiş bölgedeki pikselleri al
    masked_pixels = hsv_roi[mask > 0]
    
    if len(masked_pixels) == 0:
        return "Diger", (0, 255, 0)
    
    # HSV değerlerini al
    h_values = masked_pixels[:, 0]
    s_values = masked_pixels[:, 1]
    v_values = masked_pixels[:, 2]
    
    total_pixels = len(masked_pixels)
    min_pixel_ratio = 0.1  # Minimum %10 piksel oranı
    
    # Mavi/Yeşil piksel sayısı (öncelik 1)
    blue_green_pixels = np.sum((h_values > 75) & (h_values < 130) & (s_values > 40) & (v_values > 60))
    if blue_green_pixels / total_pixels > min_pixel_ratio:
        return "Mavi/Yesil", (255, 0, 0)
    
    # Kahverengi piksel sayısı (öncelik 2)
    brown_pixels = np.sum(((h_values < 30) | (h_values > 150)) & (s_values > 30) & (v_values > 60))
    if brown_pixels / total_pixels > min_pixel_ratio:
        return "Kahverengi", (0, 0, 128)
    
    # Beyaz piksel sayısı (öncelik 3)
    white_pixels = np.sum((s_values < 50) & (v_values > 150))
    if white_pixels / total_pixels > min_pixel_ratio:
        return "Beyaz", (255, 255, 255)
    
    # Hiçbiri değilse
    return "Diger", (0, 255, 0)

# Dosya yükleme için izin verilen uzantılar
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    if file and allowed_file(file.filename):
        # Güvenli dosya adı oluştur
        filename = secure_filename(file.filename)
        # Geçici dosya yolu
        temp_path = os.path.join('temp', filename)
        file.save(temp_path)
        
        # Yumurta tespiti işlemleri
        image = cv2.imread(temp_path)
        if image is None:
            return jsonify({'error': 'Görüntü yüklenemedi'}), 400
        
        # Görüntü boyutlarını al
        height, width = image.shape[:2]
        
        # Elips parametrelerini ayarla
        center = (width // 2, height // 2)
        axes = (width // 4, height // 3)
        angle = 0
        
        # Elips maskesi oluştur
        ellipse_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(ellipse_mask, center, axes, angle, 0, 360, 255, -1)
        
        # HSI'ya dönüştür
        hsi_image = rgb_to_hsi(image)
        h, s, i = cv2.split(hsi_image)
        
        # Yumurta tespiti için intensity kanalını kullan
        _, thresh = cv2.threshold(i, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morfolojik işlemler
        kernel = np.ones((5,5), np.uint8)
        cleaned_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        # Elips maskesi ile kesişim
        final_mask = cv2.bitwise_and(cleaned_mask, ellipse_mask)
        
        # Contour'ları bul
        contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        result = {
            'yumurta_bulundu': False,
            'genislik': 0,
            'yukseklik': 0,
            'boy_orani': 0,
            'renk': '',
            'alan': 0,
            'cinsiyet': 'Belirlenemedi',
            'sonuc_resim_yolu': ''
        }
        
        if len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_contour = contours[0]
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # En-boy oranı kontrolü
            if w < h:
                rate = w / h
            else:
                rate = h / w
            
            if 0.600 <= rate <= 0.900:  # Yumurta şekli kontrolü
                # ROI ve maske oluştur
                roi = image[y:y+h, x:x+w]
                mask = np.zeros((h, w), dtype=np.uint8)
                contour_shifted = largest_contour - np.array([[x, y]])
                cv2.drawContours(mask, [contour_shifted], -1, (255), -1)
                
                # Renk sınıflandırması
                color_name, _ = get_dominant_color(roi, mask)
                
                # Alan hesaplama
                area_pixels = np.sum(mask > 0)

                # Çevre uzunluğu
                perimeter = cv2.arcLength(largest_contour, True)

                # Nesnenin merkezi (dikdörtgenin sol üstü 0,0 kabul edilirse)
                rect_center_x = w // 2
                rect_center_y = h // 2

                # Ağırlık merkezi (moment ile, dikdörtgene göre normalize)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) - x
                    cy = int(M["m01"] / M["m00"]) - y
                else:
                    cx, cy = 0, 0

                # Ovalliği hesapla (fitEllipse için en az 5 nokta gerekir)
                if len(largest_contour) >= 5:
                    ellipse = cv2.fitEllipse(largest_contour)
                    major_axis = max(ellipse[1])
                    minor_axis = min(ellipse[1])
                    ovallik = float(np.sqrt(1 - (minor_axis / major_axis) ** 2))
                else:
                    ovallik = "Hesaplanamadı (fitEllipse için yeterli nokta yok)"

                # Şekil indeksi
                if area_pixels > 0:
                    shape_index = float((perimeter ** 2) / (4 * np.pi * area_pixels))
                else:
                    shape_index = "Hesaplanamadı (alan sıfır)"

                # Sonuç resmini kaydet - sadece dikdörtgen çizimi
                result_image = image.copy()
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                result_path = os.path.join('results', f'result_{filename}')
                cv2.imwrite(result_path, result_image)
                
                result = {
                    'yumurta_bulundu': True,
                    'genislik': w,
                    'yukseklik': h,
                    'boy_orani': round(rate, 3),
                    'renk': color_name,
                    'alan': area_pixels,
                    'cinsiyet': 'Belirlenemedi',
                    'sonuc_resim_yolu': result_path,
                    'cevre_uzunlugu': perimeter,
                    'nesne_merkezi': [rect_center_x, rect_center_y],
                    'agirlik_merkezi': [cx, cy],
                    'ovallik': ovallik,
                    'sekil_indeksi': shape_index
                }
        
        # Geçici dosyayı sil
        os.remove(temp_path)
        
        return jsonify(result)
    
    return jsonify({'error': 'İzin verilmeyen dosya türü'}), 400

if __name__ == '__main__':
    # Gerekli klasörleri oluştur
    os.makedirs('temp', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    app.run(debug=True) 