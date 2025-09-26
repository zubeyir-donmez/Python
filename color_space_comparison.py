import cv2
import numpy as np

def adjust_brightness(val):
    global image, brightness
    
    brightness = val
    # Parlaklık ayarını uygula
    adjusted_image = image.copy()
    adjusted_image = cv2.add(adjusted_image, brightness - 127)
    adjusted_image = np.clip(adjusted_image, 0, 255)
    
    # HSV renk uzayına dönüştür ve parlaklığı ayarla
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2] + (brightness - 127), 0, 255)
    adjusted_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # HSL renk uzayına dönüştür ve parlaklığı ayarla
    hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hsl[:,:,1] = np.clip(hsl[:,:,1] + (brightness - 127), 0, 255)
    adjusted_hsl = cv2.cvtColor(hsl, cv2.COLOR_HLS2BGR)
    
    # HSI renk uzayına dönüştür ve parlaklığı ayarla
    hsi = rgb_to_hsi(image)
    hsi[:,:,2] = np.clip(hsi[:,:,2] + (brightness - 127), 0, 255)
    adjusted_hsi = hsi_to_bgr(hsi)
    
    # Görüntüleri birleştir
    h, w = image.shape[:2]
    combined = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    
    # Orijinal görüntü
    combined[0:h, 0:w] = adjusted_image
    
    # HSV görüntüsü
    combined[0:h, w:w*2] = adjusted_hsv
    
    # HSL görüntüsü
    combined[h:h*2, 0:w] = adjusted_hsl
    
    # HSI görüntüsü
    combined[h:h*2, w:w*2] = adjusted_hsi
    
    # Başlıkları ekle
    cv2.putText(combined, f'Orijinal (Parlaklik: {brightness})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, f'HSV (Parlaklik: {brightness})', (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, f'HSL (Parlaklik: {brightness})', (10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, f'HSI (Parlaklik: {brightness})', (w+10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Görüntüyü göster
    cv2.imshow('Renk Uzaylari Karsilastirmasi', combined)

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

def hsi_to_bgr(hsi_image):
    h, s, i = cv2.split(hsi_image)
    
    # Normalize
    h = h / 255.0 * 360
    s = s / 255.0
    i = i / 255.0
    
    # HSI'den RGB'ye dönüşüm
    h = np.radians(h)
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    
    # 0-120 derece arası
    mask = (h >= 0) & (h < 2 * np.pi / 3)
    b[mask] = i[mask] * (1 - s[mask])
    r[mask] = i[mask] * (1 + s[mask] * np.cos(h[mask]) / np.cos(np.pi / 3 - h[mask]))
    g[mask] = 3 * i[mask] - (r[mask] + b[mask])
    
    # 120-240 derece arası
    mask = (h >= 2 * np.pi / 3) & (h < 4 * np.pi / 3)
    h[mask] = h[mask] - 2 * np.pi / 3
    r[mask] = i[mask] * (1 - s[mask])
    g[mask] = i[mask] * (1 + s[mask] * np.cos(h[mask]) / np.cos(np.pi / 3 - h[mask]))
    b[mask] = 3 * i[mask] - (r[mask] + g[mask])
    
    # 240-360 derece arası
    mask = (h >= 4 * np.pi / 3) & (h < 2 * np.pi)
    h[mask] = h[mask] - 4 * np.pi / 3
    g[mask] = i[mask] * (1 - s[mask])
    b[mask] = i[mask] * (1 + s[mask] * np.cos(h[mask]) / np.cos(np.pi / 3 - h[mask]))
    r[mask] = 3 * i[mask] - (g[mask] + b[mask])
    
    # 0-255 aralığına dönüştür
    r = np.clip(r * 255, 0, 255).astype(np.uint8)
    g = np.clip(g * 255, 0, 255).astype(np.uint8)
    b = np.clip(b * 255, 0, 255).astype(np.uint8)
    
    return cv2.merge([r, g, b])

# Görüntüyü yükle
image = cv2.imread('D:/Kodlar/Python/yum_track/Cropped_egg/32.jpg')
if image is None:
    print("Görüntü yüklenemedi. Lütfen dosya yolunu kontrol edin.")
    exit()

# Başlangıç parlaklık değeri
brightness = 127

# Pencereyi oluştur ve trackbar'ı ekle
cv2.namedWindow('Renk Uzaylari Karsilastirmasi', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Parlaklik', 'Renk Uzaylari Karsilastirmasi', 127, 255, adjust_brightness)

# İlk görüntüyü göster
adjust_brightness(127)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows() 