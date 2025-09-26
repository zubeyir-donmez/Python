import cv2
import numpy as np

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

def adjust_brightness(val):
    global image, brightness
    brightness = val - 127  # -127 ile +128 arası parlaklık değişimi
    
    # HSI'ya dönüştür
    hsi_image = rgb_to_hsi(image)
    h, s, i = cv2.split(hsi_image)
    
    # Parlaklık ayarı
    i = cv2.add(i, brightness)
    i = np.clip(i, 0, 255)
    
    adjusted_hsi = cv2.merge([h, s, i])
    
    # Yumurta tespiti için intensity kanalını kullan
    _, thresh = cv2.threshold(i, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morfolojik işlemler
    kernel = np.ones((5,5), np.uint8)
    cleaned_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    
    # Elips maskesi ile kesişim - sadece sınırlama için kullan
    final_mask = cv2.bitwise_and(cleaned_mask, ellipse_mask)
    
    # Contour'ları bul
    contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Görselleştirme için kopya oluştur
    visualization = image.copy()
    
    # Elipsi çiz - sadece referans için
    cv2.ellipse(visualization, center, axes, angle, 0, 360, (255, 0, 0), 2)
    
    # Parlaklık değerini göster
    cv2.putText(visualization, f"Parlaklik: {val}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # En büyük contour'u bul
    if len(contours) > 0:
        # Contourları alanlarına göre sırala
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # En büyük contour için işlem yap
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
            
            # Kenar tespiti
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
            edges = cv2.Canny(roi_blur, 50, 150)
            
            # Kenarları orijinal görüntüye ekle
            visualization[y:y+h, x:x+w][edges > 0] = [0, 0, 255]
            
            # Renk sınıflandırması
            color_name, rect_color = get_dominant_color(roi, mask)
            
            # Alan hesaplaması için kullanılan pikselleri göster (0.2 opacity)
            overlay = visualization.copy()
            # Sadece yumurta bölgesinde overlay göster
            overlay_roi = overlay[y:y+h, x:x+w]
            mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            overlay_roi[mask > 0] = rect_color  # Tespit edilen rengi kullan
            cv2.addWeighted(overlay, 0.3, visualization, 0.8, 0, visualization)
            
            # Tespit edilen yumurtayı çiz
            # if len(largest_contour) >= 5:
            #     ellipse = cv2.fitEllipse(largest_contour)
            #     cv2.ellipse(visualization, ellipse, (0, 255, 0), 2)
            # Rectangle (dikdörtgen) çiz
            cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Alan hesaplama - sadece mask içindeki pikselleri say
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
                ovallik = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            else:
                ovallik = None

            # Şekil indeksi
            if area_pixels > 0:
                shape_index = (perimeter ** 2) / (4 * np.pi * area_pixels)
            else:
                shape_index = None

            print(f"Yumurta: {color_name}")
            print(f"Boyutlar: {w}px x {h}px")
            print(f"Boy oranı: {rate:.3f}")
            print(f"Alan: {area_pixels}px²")
            print(f"Çevre uzunluğu: {perimeter:.2f} px")
            print(f"Nesnenin merkezi (dikdörtgen): ({rect_center_x}, {rect_center_y}))")
            print(f"Ağırlık merkezi (mask içinde): ({cx}, {cy}))")
            if ovallik is not None:
                print(f"Ovalliği: {ovallik:.3f}")
            else:
                print("Ovalliği: Hesaplanamadı (fitEllipse için yeterli nokta yok)")
            if shape_index is not None:
                print(f"Şekil indeksi: {shape_index:.3f}")
            else:
                print("Şekil indeksi: Hesaplanamadı (alan sıfır)")
            print("-" * 30)
    
    # Sarı daire için merkez ve yarıçapı resim boyutuna göre hesapla
    center_x = int(width * 6 / 8)
    center_y = int(height / 8)
    radius_px = int(min(width, height) / 12)  # Kısa kenarın 1/6'sı
    cv2.circle(visualization, (center_x, center_y), radius_px, (0, 255, 255), 2)

    cv2.imshow('Yumurta Tespiti', visualization)

# Görüntüyü yükle
image = cv2.imread('D:/Kodlar/Python/yum_track/Cropped_egg/21.jpg')
if image is None:
    print("Görüntü yüklenemedi!")
    exit()

# Görüntü boyutlarını al
height, width = image.shape[:2]

# Elips parametrelerini ayarla
center = (width // 2, height // 2)
axes = (width // 5, height // 3)
angle = 0

# Elips maskesi oluştur
ellipse_mask = np.zeros((height, width), dtype=np.uint8)
cv2.ellipse(ellipse_mask, center, axes, angle, 0, 360, 255, -1)

# Pencereyi oluştur
cv2.namedWindow('Yumurta Tespiti', cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('Parlaklik', 'Yumurta Tespiti', 127, 255, adjust_brightness)

# İlk görüntüyü göster
adjust_brightness(127)

print("Çıkmak için 'q' tuşuna basın veya pencereyi kapatın.")
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty('Yumurta Tespiti', cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows() 