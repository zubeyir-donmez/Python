from typing import Optional, Tuple, List, Dict
import cv2
import numpy as np

class EggDetector:
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2
        self.egg_width = 200
        self.egg_height = 300
        
        # Parlaklık ayarlama parametreleri
        self.brightness_offset = 90  # Varsayılan parlaklık artışı
        self.kernel_size = 5  # Morfolojik işlemler için kernel boyutu
        
        # Beyaz renk aralıkları (HSV formatında)
        self.lower_color = np.array([0, 0, 200])  # Sabit alt sınır
        self.upper_white = np.array([180, 55, 255])
        
        # Kontur düzleştirme için parametreler
        self.epsilon_factor = 0.01  # Kontur yaklaşımı için epsilon faktörü

    def get_egg_area_brightness(self, frame: np.ndarray) -> List[int]:
        """Yumurta alanı içindeki piksellerin parlaklık değerlerini döndürür"""
        # Yumurta alanını belirle (merkezdeki ROI)
        roi = frame[self.center_y-self.egg_height//2:self.center_y+self.egg_height//2,
                   self.center_x-self.egg_width//2:self.center_x+self.egg_width//2]
        
        # Gri tonlamaya çevir
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Tüm piksel değerlerini düz bir listeye çevir
        brightness_values = gray_roi.flatten().tolist()
        
        return brightness_values

    def draw_egg_template(self, frame: np.ndarray) -> np.ndarray:
        """Kameranın merkezine yumurta şablonu çizer"""
        # Yumurta şablonunun merkez noktası
        center = (self.center_x, self.center_y)
        
        # Yumurta şeklini çiz (elips)
        cv2.ellipse(frame, center, (self.egg_width//2, self.egg_height//2), 
                   0, 0, 360, (0, 255, 0), 2)
        
        return frame

    def analyze_brightness_ranges(self, brightness_values: List[int]) -> Dict[str, Tuple[int, int]]:
        """Parlaklık değerlerini analiz edip aralıklara böler"""
        if not brightness_values:
            return {}
            
        # Parlaklık değerlerini sırala
        sorted_values = sorted(brightness_values)
        
        # 4 aralığa böl (çok karanlık, karanlık, normal, parlak, çok parlak)
        total_values = len(sorted_values)
        ranges = {
            'very_dark': (0, sorted_values[total_values // 5]),
            'dark': (sorted_values[total_values // 5], sorted_values[2 * total_values // 5]),
            'normal': (sorted_values[2 * total_values // 5], sorted_values[3 * total_values // 5]),
            'bright': (sorted_values[3 * total_values // 5], sorted_values[4 * total_values // 5]),
            'very_bright': (sorted_values[4 * total_values // 5], 255)
        }
        
        return ranges

    def calculate_brightness_adjustment(self, brightness_ranges: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
        """Her parlaklık aralığı için ayarlama değerini hesaplar"""
        adjustments = {
            'very_dark': 1.8,    # Çok karanlık -> parlaklığı artır
            'dark': 1.4,         # Karanlık -> parlaklığı biraz artır
            'normal': 1.0,       # Normal -> değiştirme
            'bright': 0.7,       # Parlak -> parlaklığı biraz azalt
            'very_bright': 0.5   # Çok parlak -> parlaklığı azalt
        }
        return adjustments

    def get_most_common_brightness(self, brightness_values: List[int]) -> Tuple[str, float]:
        """En çok bulunan parlaklık değerini ve hangi aralıkta olduğunu döndürür"""
        if not brightness_values:
            return 'normal', 1.0
            
        # En çok bulunan parlaklık değerini bul
        hist = np.histogram(brightness_values, bins=5, range=(0, 255))
        most_common_bin = np.argmax(hist[0])
        most_common_value = (hist[1][most_common_bin] + hist[1][most_common_bin + 1]) / 2
        
        # Hangi aralıkta olduğunu belirle
        if most_common_value < 51:  # 0-50
            return 'very_dark', 1.8
        elif most_common_value < 102:  # 51-101
            return 'dark', 1.4
        elif most_common_value < 153:  # 102-152
            return 'normal', 1.0
        elif most_common_value < 204:  # 153-203
            return 'bright', 0.7
        else:  # 204-255
            return 'very_bright', 0.5

    def adjust_brightness(self, frame: np.ndarray) -> np.ndarray:
        """Görüntünün parlaklığını ayarlar"""
        # HSV'ye çevir
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Parlaklık (V) kanalını ayarla
        h, s, v = cv2.split(hsv_image)
        v = cv2.add(v, self.brightness_offset)
        v = np.clip(v, 0, 255)
        
        # HSV'yi birleştir ve BGR'ye geri çevir
        adjusted_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)

    def detect_white_object(self, frame: np.ndarray) -> np.ndarray:
        """Beyaz nesneyi tespit eder ve düzgün konturlar çizer"""
        # Elips maskesi oluştur
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, (self.center_x, self.center_y), 
                   (self.egg_width//2, self.egg_height//2), 
                   0, 0, 360, 255, -1)
        
        # Maskeyi ROI boyutuna kırp
        roi_mask = mask[self.center_y-self.egg_height//2:self.center_y+self.egg_height//2,
                       self.center_x-self.egg_width//2:self.center_x+self.egg_width//2]
        
        # Yumurta alanını belirle
        roi = frame[self.center_y-self.egg_height//2:self.center_y+self.egg_height//2,
                   self.center_x-self.egg_width//2:self.center_x+self.egg_width//2]
        
        # Parlaklığı ayarla
        adjusted_roi = self.adjust_brightness(roi)
        
        # ROI'yi HSV'ye çevir
        hsv_roi = cv2.cvtColor(adjusted_roi, cv2.COLOR_BGR2HSV)
        
        # Beyaz renk aralığını maskele
        white_mask = cv2.inRange(hsv_roi, self.lower_color, self.upper_white)
        
        # Elips maskesi ile birleştir
        white_mask = cv2.bitwise_and(white_mask, roi_mask)
        
        # Gürültüyü azalt ve konturları düzleştir
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        # Konturları bul
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # En büyük konturu bul
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Kontur alanı yeterince büyükse çiz
            if cv2.contourArea(largest_contour) > 1000:
                # Konturu düzleştir
                epsilon = self.epsilon_factor * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Düzleştirilmiş konturu çiz
                cv2.drawContours(roi, [approx], -1, (0, 255, 0), 2)
                
                # Merkez noktasını bul ve çiz
                M = cv2.moments(approx)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(roi, (cx, cy), 5, (0, 0, 255), -1)
                    
                    # Kontur alanını göster
                    area = cv2.contourArea(approx)
                    cv2.putText(roi, f"Area: {int(area)}", 
                              (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 2)
        
        # ROI'yi ana görüntüye yerleştir
        frame[self.center_y-self.egg_height//2:self.center_y+self.egg_height//2,
              self.center_x-self.egg_width//2:self.center_x+self.egg_width//2] = roi
        
        return frame

def main():
    # Kamera başlat
    cap = cv2.VideoCapture(0)
    detector = EggDetector()
    
    # Parlaklık ayarı için trackbar oluştur
    cv2.namedWindow('Egg Detection')
    cv2.createTrackbar('Brightness', 'Egg Detection', 90, 200, 
                      lambda x: setattr(detector, 'brightness_offset', x - 100))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Yumurta şablonunu çiz
        frame = detector.draw_egg_template(frame)
        
        # Beyaz nesneyi tespit et ve konturunu çiz
        frame = detector.detect_white_object(frame)
        
        # Görüntüyü göster
        cv2.imshow('Egg Detection', frame)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 