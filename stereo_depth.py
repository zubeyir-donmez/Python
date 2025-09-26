import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_images(left_img, right_img, title="Stereo Images"):
    """İki görüntüyü yan yana gösterir"""
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    plt.title('Sol Görüntü')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
    plt.title('Sağ Görüntü')
    plt.axis('off')
    
    plt.suptitle(title)
    plt.show()

def create_disparity_map(left_img, right_img):
    """Disparite haritası oluşturur"""
    # Gri tonlamaya çevir
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # Stereo eşleştirme parametreleri
    window_size = 5
    min_disp = 0
    num_disp = 16*16  # Disparite aralığı
    
    # SGBM (Semi-Global Block Matching) oluştur
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # Disparite haritasını hesapla
    disparity = stereo.compute(left_gray, right_gray)
    
    # Normalize et
    disparity_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    
    return disparity_norm.astype(np.uint8)

def create_depth_map(disparity, focal_length, baseline):
    """Disparite haritasından derinlik haritası oluşturur"""
    # Sıfır değerlerini filtrele
    mask = disparity > 0
    
    # Derinlik hesapla (Z = f * B / d)
    depth = np.zeros_like(disparity, dtype=np.float32)
    depth[mask] = (focal_length * baseline) / (disparity[mask] + 1e-10)
    
    # Normalize et
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    
    return depth_norm.astype(np.uint8)

def main():
    # Görüntüleri yükle
    left_img = cv2.imread('depth_1.jpg')  # Sol görüntü
    right_img = cv2.imread('depth_2.jpg')  # Sağ görüntü
    
    if left_img is None or right_img is None:
        print("Görüntüler yüklenemedi!")
        return
    
    # Görüntüleri göster
    show_images(left_img, right_img)
    
    # Disparite haritası oluştur
    disparity_map = create_disparity_map(left_img, right_img)
    
    # Disparite haritasını göster
    plt.figure(figsize=(10, 7))
    plt.imshow(disparity_map, cmap='plasma')
    plt.colorbar(label='Disparite')
    plt.title('Disparite Haritası')
    plt.show()
    
    # Derinlik haritası oluştur
    # Not: Bu değerler kamera kalibrasyonundan gelmelidir
    focal_length = 1000  # piksel cinsinden odak uzaklığı
    baseline = 0.1  # metre cinsinden kameralar arası mesafe
    
    depth_map = create_depth_map(disparity_map, focal_length, baseline)
    
    # Derinlik haritasını göster
    plt.figure(figsize=(10, 7))
    plt.imshow(depth_map, cmap='plasma')
    plt.colorbar(label='Derinlik')
    plt.title('Derinlik Haritası')
    plt.show()
    
    # Sonuçları kaydet
    cv2.imwrite('disparity_map.jpg', disparity_map)
    cv2.imwrite('depth_map.jpg', depth_map)
    
    print("İşlem tamamlandı! Sonuçlar kaydedildi.")

if __name__ == "__main__":
    main() 