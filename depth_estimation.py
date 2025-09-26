import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

class DepthEstimation:
    def __init__(self):
        # Stereo kamera parametreleri (gerçek değerler kamera kalibrasyonundan gelmelidir)
        self.focal_length = 1000  # piksel cinsinden odak uzaklığı
        self.baseline = 0.1  # metre cinsinden stereo kameralar arası mesafe

    def show_depth_map(self, depth_map, title):
        """Derinlik haritasını görselleştirir"""
        plt.figure(figsize=(10, 7))
        plt.imshow(depth_map, cmap='plasma')
        plt.colorbar(label='Derinlik')
        plt.title(title)
        plt.show()

    def basic_depth_from_defocus(self, image):
        """
        Seviye 1: Basit Derinlik Tahmini - Odak Bulanıklığından
        Bu yöntem, görüntünün lokal kontrast ve kenar bilgisini kullanır
        """
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gaussian bulanıklaştırma uygula (farklı sigma değerleri ile)
        blur1 = cv2.GaussianBlur(gray, (5,5), 1.0)
        blur2 = cv2.GaussianBlur(gray, (5,5), 2.0)
        
        # Kenar tespiti
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Lokal kontrast hesapla
        local_contrast = ndimage.gaussian_filter(np.abs(edges), sigma=2.0)
        
        # Derinlik tahmini (normalize edilmiş)
        depth = 1.0 / (local_contrast + 1e-10)
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        return depth

    def stereo_depth_estimation(self, left_image, right_image):
        """
        Seviye 2: Stereo Görüntülerden Derinlik Tahmini
        İki kamera görüntüsü arasındaki disparite hesaplanır
        """
        # Gri tonlamaya çevir
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        
        # Stereo eşleştirme için SGBM (Semi-Global Block Matching)
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16*16,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
        
        # Disparite hesapla
        disparity = stereo.compute(left_gray, right_gray)
        
        # Derinlik hesapla (Z = f * B / d)
        depth = self.focal_length * self.baseline / (disparity + 1e-10)
        
        # Normalize et
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        return depth

    def structure_from_motion(self, image_sequence):
        """
        Seviye 3: Hareket Tabanlı Yapı Tahmini (Structure from Motion)
        Kamera hareketi ve görüntü dizisinden 3B yapı tahmini
        """
        # ORB özellik dedektörü
        orb = cv2.ORB_create()
        
        # İlk görüntüyü referans al
        prev_frame = image_sequence[0]
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_kp, prev_desc = orb.detectAndCompute(prev_gray, None)
        
        depth_map = np.zeros_like(prev_gray, dtype=float)
        point_cloud = []
        
        # Her görüntü için
        for frame in image_sequence[1:]:
            # Özellik noktalarını bul
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, desc = orb.detectAndCompute(gray, None)
            
            # Özellik eşleştirme
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(prev_desc, desc)
            
            # En iyi eşleşmeleri seç
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Eşleşen noktaları al
            prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            curr_pts = np.float32([kp[m.trainIdx].pt for m in matches])
            
            # Temel matris hesapla
            E, mask = cv2.findEssentialMat(prev_pts, curr_pts, self.focal_length)
            
            # Kamera pozunu hesapla
            _, R, t, mask = cv2.recoverPose(E, prev_pts, curr_pts)
            
            # 3B noktaları triangüle et
            points_4d = cv2.triangulatePoints(
                np.eye(3, 4),  # İlk kamera matrisi
                np.hstack((R, t)),  # İkinci kamera matrisi
                prev_pts.T,
                curr_pts.T
            )
            
            # Homojen koordinatlardan 3B koordinatlara dönüştür
            points_3d = points_4d[:3] / points_4d[3]
            point_cloud.extend(points_3d.T)
            
            # Derinlik haritasını güncelle
            for pt, depth in zip(prev_pts, points_3d[2]):
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
                    depth_map[y, x] = depth
            
            # Bir sonraki frame için güncelle
            prev_gray = gray
            prev_kp = kp
            prev_desc = desc
        
        # Derinlik haritasını düzgünleştir ve normalize et
        depth_map = ndimage.gaussian_filter(depth_map, sigma=2.0)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        return depth_map, np.array(point_cloud)

def main():
    # Derinlik tahmini sınıfını oluştur
    depth_estimator = DepthEstimation()
    
    # Test için görüntü yükle
    image = cv2.imread('depth.jpg')
    left_image = cv2.imread('depth_1.jpg')
    right_image = cv2.imread('depth_2.jpg')
    if image is None:
        print("Görüntü yüklenemedi!")
        return
    
    # Seviye 1: Basit derinlik tahmini
    #print("Seviye 1: Odak Bulanıklığından Derinlik Tahmini yapılıyor...")
    #depth_map_basic = depth_estimator.basic_depth_from_defocus(image)
    #depth_estimator.show_depth_map(depth_map_basic, 'Basit Derinlik Tahmini')
    
    # Seviye 2: Stereo görüntülerden derinlik tahmini
    # Not: Bu kısım için iki kamera görüntüsü gereklidir
    print("\nSeviye 2: Stereo görüntüler gereklidir...")
    print("Sol ve sağ kamera görüntüleri mevcut olduğunda:")
    depth_map_stereo = depth_estimator.stereo_depth_estimation(left_image, right_image)
    depth_estimator.show_depth_map(depth_map_stereo, 'Basit Derinlik Tahmini')
    
    # Seviye 3: Structure from Motion
    print("\nSeviye 3: Görüntü dizisi gereklidir...")
    print("Görüntü dizisi mevcut olduğunda:")
    print("depth_map_sfm, point_cloud = depth_estimator.structure_from_motion(image_sequence)")

if __name__ == "__main__":
    main() 