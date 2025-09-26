import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parametreler
baseline = 0.16  # metre
focal_length = 3740  # piksel

# Görüntüleri oku
left_img = cv2.imread('D:/Kodlar/Python/yz_depth/im2.png', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('D:/Kodlar/Python/yz_depth/im6.png', cv2.IMREAD_GRAYSCALE)

# Stereo eşleştirme (disparity haritası)
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
print(len(disparity))
# Negatif ve sıfır değerleri maskele
disparity[disparity <= 0] = np.nan

# Derinlik haritası
depth_map = (focal_length * baseline) / disparity

# Örnek bir piksel seçelim (ör: x=200, y=150)
x, y = 200, 150
disparity_val = disparity[y, x]
depth_val = depth_map[y, x]

print(f"Seçilen pikselde disparity: {disparity_val:.2f} piksel")
print(f"Seçilen pikselde derinlik: {depth_val:.2f} metre")

# Görüntüleri ve haritaları göster
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title('Sol Görüntü')
plt.imshow(left_img, cmap='gray')
plt.scatter([x], [y], c='r', s=40)
plt.subplot(1,3,2)
plt.title('Disparity Haritası')
plt.imshow(disparity, cmap='plasma')
plt.scatter([x], [y], c='w', s=40)
plt.colorbar()
plt.subplot(1,3,3)
plt.title('Derinlik Haritası (m)')
plt.imshow(depth_map, cmap='inferno')
plt.scatter([x], [y], c='w', s=40)
plt.colorbar()
plt.tight_layout()
plt.show()

# Nokta bulutu oluşturma ve görselleştirme
h, w = left_img.shape
cx, cy = w / 2, h / 2

points = []
colors = []
for y in range(h):
    for x in range(w):
        Z = depth_map[y, x]
        if np.isnan(Z) or Z > 20:  # Uzak/bozuk noktaları atla
            continue
        X = (x - cx) * Z / focal_length
        Y = (y - cy) * Z / focal_length
        points.append([X, -Y, Z])  # -Y ile görüntü tersliğini düzelt
        colors.append(left_img[y, x] / 255.0)
points = np.array(points)
colors = np.array(colors)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=0.5, cmap='gray')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Nokta Bulutu')
plt.show()