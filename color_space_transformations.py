import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_channels(image, title):
    """Görüntünün kanallarını ayrı ayrı gösterir"""
    plt.figure(figsize=(15, 5))
    if len(image.shape) == 3:
        for i, channel in enumerate(['Blue', 'Green', 'Red']):
            plt.subplot(1, 3, i+1)
            plt.imshow(image[:,:,i], cmap='gray')
            plt.title(f'{channel} Channel')
    plt.suptitle(title)
    plt.show()

def manual_rgb_to_grayscale(image):
    """
    RGB'den Gri tonlamaya manuel dönüşüm
    Formül: Gray = 0.299R + 0.587G + 0.114B
    """
    r, g, b = image[:,:,2], image[:,:,1], image[:,:,0]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)

def manual_rgb_to_hsv(image):
    """
    RGB'den HSV'ye manuel dönüşüm
    H: Renk tonu (0-360 derece)
    S: Doygunluk (0-1)
    V: Parlaklık değeri (0-1)
    """
    # RGB değerlerini 0-1 aralığına normalize et
    r, g, b = image[:,:,2]/255.0, image[:,:,1]/255.0, image[:,:,0]/255.0
    
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin

    # Hue hesaplama
    h = np.zeros_like(r)
    
    # R maksimumsa
    mask = (cmax == r) & (diff != 0)
    h[mask] = 60 * ((g[mask] - b[mask])/diff[mask] % 6)
    
    # G maksimumsa
    mask = (cmax == g) & (diff != 0)
    h[mask] = 60 * ((b[mask] - r[mask])/diff[mask] + 2)
    
    # B maksimumsa
    mask = (cmax == b) & (diff != 0)
    h[mask] = 60 * ((r[mask] - g[mask])/diff[mask] + 4)
    
    # Saturation hesaplama
    s = np.zeros_like(r)
    mask = (cmax != 0)
    s[mask] = diff[mask]/cmax[mask]
    
    # Value hesaplama
    v = cmax

    # HSV görüntüsünü oluştur
    hsv = np.stack([h/2, s*255, v*255], axis=-1).astype(np.uint8)
    return hsv

def manual_rgb_to_hsi(image):
    """
    RGB'den HSI'ya manuel dönüşüm
    H: Renk tonu (0-360 derece)
    S: Doygunluk (0-1)
    I: Yoğunluk (0-1)
    """
    # RGB değerlerini 0-1 aralığına normalize et
    r, g, b = image[:,:,2]/255.0, image[:,:,1]/255.0, image[:,:,0]/255.0
    
    # Yoğunluk (I) hesaplama
    i = (r + g + b) / 3.0
    
    # Doygunluk (S) hesaplama
    s = np.zeros_like(r)
    mask = (i > 0)
    minimum = np.minimum(np.minimum(r, g), b)
    s[mask] = 1 - (3 * minimum[mask])/(r[mask] + g[mask] + b[mask])
    
    # Hue (H) hesaplama
    h = np.zeros_like(r)
    mask = (s > 0)
    
    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g)**2 + (r - b)*(g - b))
    theta = np.arccos(np.clip(numerator/(denominator + 1e-10), -1.0, 1.0))
    
    h[mask] = theta[mask]
    h[mask & (b > g)] = 2*np.pi - h[mask & (b > g)]
    h = np.rad2deg(h)
    
    # HSI görüntüsünü oluştur
    hsi = np.stack([h/2, s*255, i*255], axis=-1).astype(np.uint8)
    return hsi

def calculate_alpha_channel(image):
    """
    Alpha kanalı hesaplama (Derinlik tahmini)
    Basit bir yaklaşım: Kenar tespiti ve yoğunluk bazlı
    """
    # Gri tonlamalı görüntü al
    gray = manual_rgb_to_grayscale(image)
    
    # Kenar tespiti (Sobel operatörü)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize et
    gradient_magnitude = (gradient_magnitude/gradient_magnitude.max() * 255).astype(np.uint8)
    
    # Yoğunluk bazlı alpha
    intensity = gray
    
    # Alpha kanalını kenar ve yoğunluk bilgisini birleştirerek oluştur
    alpha = (gradient_magnitude * 0.5 + intensity * 0.5).astype(np.uint8)
    
    return alpha

def main():
    # Resmi yükle
    image = cv2.imread('test.png')  # Resmi buraya koyun
    if image is None:
        print("Resim yüklenemedi!")
        return
    
    # BGR'den RGB'ye çevir
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Orijinal RGB kanallarını göster
    show_channels(image_rgb, 'Original RGB Channels')
    
    # Gri tonlamaya dönüştür ve göster
    gray = manual_rgb_to_grayscale(image_rgb)
    plt.figure(figsize=(5,5))
    plt.imshow(gray, cmap='gray')
    plt.title('Manual Grayscale Conversion')
    plt.show()
    
    # HSV'ye dönüştür ve kanalları göster
    hsv = manual_rgb_to_hsv(image_rgb)
    plt.figure(figsize=(15, 5))
    titles = ['Hue', 'Saturation', 'Value']
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(hsv[:,:,i], cmap='gray')
        plt.title(titles[i])
    plt.suptitle('Manual HSV Conversion')
    plt.show()
    
    # HSI'ya dönüştür ve kanalları göster
    hsi = manual_rgb_to_hsi(image_rgb)
    plt.figure(figsize=(15, 5))
    titles = ['Hue', 'Saturation', 'Intensity']
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(hsi[:,:,i], cmap='gray')
        plt.title(titles[i])
    plt.suptitle('Manual HSI Conversion')
    plt.show()
    
    # Alpha kanalını hesapla ve göster
    alpha = calculate_alpha_channel(image_rgb)
    plt.figure(figsize=(5,5))
    plt.imshow(alpha, cmap='gray')
    plt.title('Calculated Alpha Channel (Depth Estimation)')
    plt.show()
    
    # RGBA görüntüsünü oluştur ve göster
    rgba = np.dstack((image_rgb, alpha))
    plt.figure(figsize=(15, 5))
    channels = ['Red', 'Green', 'Blue', 'Alpha']
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(rgba[:,:,i], cmap='gray')
        plt.title(channels[i])
    plt.suptitle('RGBA Channels')
    plt.show()

if __name__ == "__main__":
    main() 