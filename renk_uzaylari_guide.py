import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle
image = cv2.imread('yum_track/Cropped_egg/48.jpg')
if image is None:
    print("Görüntü yüklenemedi. Lütfen dosya yolunu kontrol edin.")
    exit()

# BGR'den RGB'ye dönüştür
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# YCbCr dönüşümü
image_ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# HSV dönüşümü
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# HSL dönüşümü
image_hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

# HSI dönüşümü (manuel)
def rgb_to_hsi(rgb_image):
    r, g, b = cv2.split(rgb_image)
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    i = (r + g + b) / 3.0
    s = np.zeros_like(r)
    mask = (i > 0)
    minimum = np.minimum(np.minimum(r, g), b)
    s[mask] = 1 - (3 * minimum[mask]) / (r[mask] + g[mask] + b[mask])
    h = np.zeros_like(r)
    mask = (s > 0)
    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g)**2 + (r - b) * (g - b))
    theta = np.arccos(np.clip(numerator / (denominator + 1e-10), -1.0, 1.0))
    h[mask] = theta[mask]
    h[mask & (b > g)] = 2 * np.pi - h[mask & (b > g)]
    h = np.rad2deg(h)
    hsi = np.stack([h / 2, s * 255, i * 255], axis=-1).astype(np.uint8)
    return hsi

image_hsi = rgb_to_hsi(image_rgb)

# LAB dönüşümü
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# CMYK dönüşümü (manuel)
def rgb_to_cmyk(rgb_image):
    r, g, b = cv2.split(rgb_image)
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    k = 1 - np.maximum(np.maximum(r, g), b)
    c = (1 - r - k) / (1 - k + 1e-10)
    m = (1 - g - k) / (1 - k + 1e-10)
    y = (1 - b - k) / (1 - k + 1e-10)
    cmyk = np.stack([c * 255, m * 255, y * 255, k * 255], axis=-1).astype(np.uint8)
    return cmyk

image_cmyk = rgb_to_cmyk(image_rgb)

# Normalized RGB dönüşümü
def normalize_rgb(rgb_image):
    r, g, b = cv2.split(rgb_image)
    total = r + g + b + 1e-10
    r_norm = r / total
    g_norm = g / total
    b_norm = b / total
    normalized = np.stack([r_norm * 255, g_norm * 255, b_norm * 255], axis=-1).astype(np.uint8)
    return normalized

image_normalized = normalize_rgb(image_rgb)

# XYZ dönüşümü
image_xyz = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)

# Tüm renk uzaylarını göster
plt.figure(figsize=(15, 15))
titles = ['Orijinal (RGB)', 'YCbCr', 'HSV', 'HSL', 'HSI', 'LAB', 'CMYK', 'Normalized RGB', 'XYZ']
images = [image_rgb, image_ycbcr, image_hsv, image_hsl, image_hsi, image_lab, image_cmyk, image_normalized, image_xyz]

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show() 