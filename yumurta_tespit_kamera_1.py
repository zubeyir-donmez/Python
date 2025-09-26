import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
from matplotlib.gridspec import GridSpec


""" KAMERA AYARI 51. SATIRDA"""

# RGB görüntüyü HSI renk uzayına dönüştüren fonksiyon
# HSI: Ton (Hue), Doygunluk (Saturation), Yoğunluk (Intensity)
def rgbden_hsiye(rgb_goruntu):
    rgb = cv2.cvtColor(rgb_goruntu, cv2.COLOR_BGR2RGB)
    kirmizi, yesil, mavi = cv2.split(rgb)
    kirmizi = kirmizi / 255.0
    yesil = yesil / 255.0
    mavi = mavi / 255.0
    yogunluk = (kirmizi + yesil + mavi) / 3.0
    min_rgb = np.minimum(np.minimum(kirmizi, yesil), mavi)
    doygunluk = 1 - (3 / (kirmizi + yesil + mavi + 1e-6)) * min_rgb
    pay = 0.5 * ((kirmizi - yesil) + (kirmizi - mavi))
    payda = np.sqrt((kirmizi - yesil)**2 + (kirmizi - mavi) * (yesil - mavi))
    aci = np.arccos(pay / (payda + 1e-6))
    ton = np.where(mavi <= yesil, aci, 2 * np.pi - aci)
    ton = ton * 180 / np.pi
    ton = np.clip(ton, 0, 255)
    doygunluk = np.clip(doygunluk * 255, 0, 255)
    yogunluk = np.clip(yogunluk * 255, 0, 255)
    return cv2.merge([ton.astype(np.uint8), doygunluk.astype(np.uint8), yogunluk.astype(np.uint8)])

# ROI (ilgi alanı) ve maskeye göre baskın rengi bulan fonksiyon
def baskin_renk_bul(roi, maske):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    maskelenmis_pikseller = hsv_roi[maske > 0]
    if len(maskelenmis_pikseller) == 0:
        return "Diger", (0, 255, 0)
    h_degerleri = maskelenmis_pikseller[:, 0]
    s_degerleri = maskelenmis_pikseller[:, 1]
    v_degerleri = maskelenmis_pikseller[:, 2]
    toplam_piksel = len(maskelenmis_pikseller)
    min_oran = 0.1
    # Mavi/yeşil tespiti
    mavi_yesil_piksel = np.sum((h_degerleri > 75) & (h_degerleri < 130) & (s_degerleri > 40) & (v_degerleri > 60))
    if mavi_yesil_piksel / toplam_piksel > min_oran:
        return "Mavi/Yesil", (255, 0, 0)
    # Kahverengi tespiti
    kahverengi_piksel = np.sum(((h_degerleri < 30) | (h_degerleri > 150)) & (s_degerleri > 30) & (v_degerleri > 60))
    if kahverengi_piksel / toplam_piksel > min_oran:
        return "Kahverengi", (0, 0, 128)
    # Beyaz tespiti
    beyaz_piksel = np.sum((s_degerleri < 50) & (v_degerleri > 150))
    if beyaz_piksel / toplam_piksel > min_oran:
        return "Beyaz", (255, 255, 255)
    return "Diger", (0, 255, 0)

def analiz_ve_goster(goruntu):
    yukseklik, genislik = goruntu.shape[:2]
    merkez = (genislik // 2, yukseklik // 2)
    elips_ekseni = (genislik // 5, yukseklik // 2)
    elips_maske = np.zeros((yukseklik, genislik), dtype=np.uint8)
    cv2.ellipse(elips_maske, merkez, elips_ekseni, 0, 0, 360, 255, -1)
    hsi_goruntu = rgbden_hsiye(goruntu)
    ton, doy, yog = cv2.split(hsi_goruntu)
    _, esik = cv2.threshold(yog, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cekirdek = np.ones((5,5), np.uint8)
    temiz_maske = cv2.morphologyEx(esik, cv2.MORPH_OPEN, cekirdek)
    temiz_maske = cv2.morphologyEx(temiz_maske, cv2.MORPH_CLOSE, cekirdek)
    son_maske = cv2.bitwise_and(temiz_maske, elips_maske)
    konturlar, _ = cv2.findContours(son_maske, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(konturlar) > 0:
        konturlar = sorted(konturlar, key=cv2.contourArea, reverse=True)
        en_buyuk_kontur = konturlar[0]
        x, y, w, h = cv2.boundingRect(en_buyuk_kontur)
        oran = w / h if w < h else h / w
        if 0.600 <= oran <= 0.900:
            # ROI'yi padding ile al
            pad_x = int(w * 0.04)
            pad_y = int(h * 0.04)
            x1 = max(x - pad_x, 0)
            y1 = max(y - pad_y, 0)
            x2 = min(x + w + pad_x, goruntu.shape[1])
            y2 = min(y + h + pad_y, goruntu.shape[0])
            roi = goruntu[y1:y2, x1:x2]
            roi_mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
            kontur_kaydirilmis = en_buyuk_kontur - np.array([[x1, y1]])
            cv2.drawContours(roi_mask, [kontur_kaydirilmis], -1, (255), -1)
            renk_adi, rect_color = baskin_renk_bul(roi, roi_mask)
            alan_piksel = np.sum(roi_mask > 0)
            perimeter = cv2.arcLength(en_buyuk_kontur, True)
            # Merkez ve şekil noktalarını bul
            M = cv2.moments(en_buyuk_kontur)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) - x
                cy = int(M["m01"] / M["m00"]) - y
            else:
                cx, cy = 0, 0
            # Görselleştirme
            fig = plt.figure(figsize=(12, 8))
            gs = GridSpec(2, 3, height_ratios=[1.2, 2.5])
            # Boyutlar
            ax1 = fig.add_subplot(gs[0, 0])
            img1 = roi.copy()
            # Konturdan min/max noktaları bul
            kontur_noktalar = kontur_kaydirilmis.reshape(-1, 2)
            min_x = np.min(kontur_noktalar[:, 0])
            max_x = np.max(kontur_noktalar[:, 0])
            min_y = np.min(kontur_noktalar[:, 1])
            max_y = np.max(kontur_noktalar[:, 1])
            # Genişlik çizgisi (en sol ve en sağ noktalar arasında, ortalama y'de)
            ort_y = (min_y + max_y) // 2
            cv2.line(img1, (min_x, ort_y), (max_x, ort_y), (0,255,0), 2)
            # Yükseklik çizgisi (en üst ve en alt noktalar arasında, ortalama x'de)
            ort_x = (min_x + max_x) // 2
            cv2.line(img1, (ort_x, min_y), (ort_x, max_y), (255,0,0), 2)
            ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            ax1.set_title("Boyutlar", fontsize=12)
            ax1.axis('off')
            # Alan & Çevre
            ax2 = fig.add_subplot(gs[0, 1])
            img2 = roi.copy()
            overlay = img2.copy()
            overlay[roi_mask > 0] = (overlay[roi_mask > 0] * 0.5).astype(np.uint8)
            cv2.addWeighted(overlay, 0.5, img2, 0.5, 0, img2)
            cv2.drawContours(img2, [kontur_kaydirilmis], -1, (255,0,255), 1)
            ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            ax2.set_title("Alan & Çevre", fontsize=12)
            ax2.axis('off')
            # Merkez Noktalar (orijinal algoritmaya geri alındı)
            ax3 = fig.add_subplot(gs[0, 2])
            img3 = np.zeros_like(roi)
            M = cv2.moments(kontur_kaydirilmis)
            if M["m00"] != 0:
                nesne_merkezi = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else:
                nesne_merkezi = (w // 2, h // 2)
            kontur_noktalar = kontur_kaydirilmis.reshape(-1, 2)
            min_x = np.min(kontur_noktalar[:, 0])
            max_x = np.max(kontur_noktalar[:, 0])
            min_y = np.min(kontur_noktalar[:, 1])
            max_y = np.max(kontur_noktalar[:, 1])
            en_genis_x = (min_x + max_x) // 2
            en_yuksek_y = (min_y + max_y) // 2
            sekil_noktasi = (en_genis_x, en_yuksek_y)
            agirlik_merkezi = (
                (nesne_merkezi[0] + sekil_noktasi[0]) // 2,
                (nesne_merkezi[1] + sekil_noktasi[1]) // 2
            )
            # Vektör farklarını hesapla
            v_nesne = (nesne_merkezi[0] - agirlik_merkezi[0], nesne_merkezi[1] - agirlik_merkezi[1])
            v_sekil = (sekil_noktasi[0] - agirlik_merkezi[0], sekil_noktasi[1] - agirlik_merkezi[1])
            # Maksimum mesafe
            max_dist = max(
                abs(v_nesne[0]), abs(v_nesne[1]),
                abs(v_sekil[0]), abs(v_sekil[1]),
                1  # Sıfıra bölme olmasın diye
            )
            # Görselin yarıçapı kadar ölçekle (ör: kenara 20 piksel boşluk bırak)
            h_mask, w_mask = img3.shape[:2]
            center_mask = (w_mask // 2, h_mask // 2)
            max_radius = min(center_mask) - 20
            SCALE = max_radius / max_dist
            # Noktaları yerleştir
            mask_nesne = (int(center_mask[0] + v_nesne[0]*SCALE), int(center_mask[1] + v_nesne[1]*SCALE))
            mask_sekil = (int(center_mask[0] + v_sekil[0]*SCALE), int(center_mask[1] + v_sekil[1]*SCALE))
            mask_agirlik = center_mask
            # Clamp (taşmasın)
            def clamp(pt):
                return (min(max(pt[0], 0), w_mask-1), min(max(pt[1], 0), h_mask-1))
            mask_nesne = clamp(mask_nesne)
            mask_sekil = clamp(mask_sekil)
            mask_agirlik = clamp(mask_agirlik)
            # Çizim
            img3[:,:,:] = 0
            cv2.circle(img3, mask_nesne, 12, (0,255,0), -1)
            cv2.circle(img3, mask_sekil, 12, (0,0,255), -1)
            cv2.circle(img3, mask_agirlik, 12, (0,140,255), -1)
            ax3.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
            ax3.set_title("Merkez Noktalar", fontsize=12)
            ax3.axis('off')
            # Tablo
            mean_color_all = tuple(np.mean(roi[roi_mask > 0], axis=0).astype(int)) if np.any(roi_mask > 0) else (0,0,0)
            tum_veriler = [
                ["Genişlik (px)", w],
                ["Yükseklik (px)", h],
                ["Boy Oranı", f"{oran:.3f}"],
                ["Alan (px^2)", alan_piksel],
                ["Çevre Uzunluğu (px)", f"{perimeter:.2f}"],
                ["Nesne Merkezi (Yeşil)", str(nesne_merkezi)],
                ["Ağırlık Merkezi (Turuncu)", str(agirlik_merkezi)],
                ["Şekil Noktası (Kırmızı)", str(sekil_noktasi)],
                ["Renk Adı", renk_adi],
                ["RGB (Tüm Piksel Ort.)", str(mean_color_all)]
            ]
            ax_table = fig.add_subplot(gs[1, :])
            ax_table.axis('off')
            tablo = ax_table.table(
                cellText=tum_veriler,
                colLabels=["Özellik", "Değer"],
                cellLoc='left',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            tablo.auto_set_font_size(False)
            tablo.set_fontsize(12)
            plt.tight_layout(pad=1.0)
            plt.show()
        else:
            messagebox.showinfo("Sonuç", "Yumurta şekli bulunamadı.")
    else:
        messagebox.showinfo("Sonuç", "Yumurta bulunamadı.")

def kamera_gui():
    kamera = cv2.VideoCapture(0)
    if not kamera.isOpened():
        messagebox.showerror("Hata", "Kamera açılamadı!")
        return
    pencere = tk.Tk()
    pencere.title("Yumurta Tespit Kamera")
    lbl_goruntu = tk.Label(pencere)
    lbl_goruntu.pack()
    frm_buton = tk.Frame(pencere)
    frm_buton.pack()
    def foto_cek():
        basarili, kare = kamera.read()
        if not basarili:
            messagebox.showerror("Hata", "Kare alınamadı!")
            return
        analiz_ve_goster(kare)
    def resim_yukle():
        dosya = filedialog.askopenfilename(title="Resim Seç", filetypes=[("Resim Dosyaları", "*.jpg *.jpeg *.png *.bmp")])
        if dosya:
            img = cv2.imread(dosya)
            if img is None:
                messagebox.showerror("Hata", "Resim yüklenemedi!")
                return
            analiz_ve_goster(img)
    def kapat():
        kamera.release()
        pencere.destroy()
    btn_cek = tk.Button(frm_buton, text="Fotoğraf Çek", command=foto_cek, width=15, height=2)
    btn_cek.pack(side=tk.LEFT, padx=10, pady=10)
    btn_yukle = tk.Button(frm_buton, text="Resim Yükle", command=resim_yukle, width=15, height=2)
    btn_yukle.pack(side=tk.LEFT, padx=10, pady=10)
    btn_kapat = tk.Button(frm_buton, text="Kapat", command=kapat, width=15, height=2)
    btn_kapat.pack(side=tk.LEFT, padx=10, pady=10)
    def guncelle():
        basarili, kare = kamera.read()
        if basarili:
            # Elips parametreleri
            yukseklik, genislik = kare.shape[:2]
            merkez = (genislik // 2, yukseklik // 2)
            elips_ekseni = (genislik // 6, yukseklik // 3)

            # Elips maskesi oluştur
            elips_maske = np.zeros((yukseklik, genislik), dtype=np.uint8)
            cv2.ellipse(elips_maske, merkez, elips_ekseni, 0, 0, 360, 255, -1)

            # Daire parametreleri
            circle_merkez = (genislik * 6 // 8, yukseklik * 2 // 8)
            circle_r = min(genislik, yukseklik) // 10
            circle_maske = np.zeros((yukseklik, genislik), dtype=np.uint8)
            cv2.circle(circle_maske, circle_merkez, circle_r, 255, -1)

            # Elips ve daire maskesini birleştir
            birlesik_maske = cv2.bitwise_or(elips_maske, circle_maske)

            # Orijinal görüntüyü kopyala
            goruntu_kopya = kare.copy()

            # Elips ve daire dışını yarı saydam yap
            maske_tersi = cv2.bitwise_not(birlesik_maske)
            yarisaydam = (goruntu_kopya * 0.5).astype(np.uint8)
            goruntu_kopya[maske_tersi > 0] = yarisaydam[maske_tersi > 0]

            # Elipsi belirgin çiz
            cv2.ellipse(goruntu_kopya, merkez, elips_ekseni, 0, 0, 360, (0,255,0), 3)
            # Daireyi sarı kenarlı çiz
            cv2.circle(goruntu_kopya, circle_merkez, circle_r, (0,255,255), 3)

            # Tkinter için RGB'ye çevir
            rgb = cv2.cvtColor(goruntu_kopya, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=im)
            lbl_goruntu.imgtk = imgtk
            lbl_goruntu.configure(image=imgtk)
        pencere.after(30, guncelle)
    guncelle()
    pencere.mainloop()

if __name__ == "__main__":
    kamera_gui() 