import cv2
import torch
import matplotlib.pyplot as plt

midas = torch.hub.load('intel-isl/MiDas', 'MiDaS_small')
midas.to('cpu')
midas.eval()

transform = torch.hub.load('intel-isl/MiDas', 'transforms')
transform = transform.small_transform

frame = cv2.imread("D:/Kodlar/Python/oda.jpg")

img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
imgbatch = transform(img).to('cpu')

with torch.no_grad():
    prediction = midas(imgbatch)
    print(prediction.shape)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size = img.shape[:2],
        mode = 'bicubic',
        align_corners = False
    ).squeeze()

    output = prediction.cpu().numpy()
    
# Görselleştir
plt.imshow(output, cmap='inferno')
plt.title("Derinlik Haritası")
plt.axis("off")
plt.show()  # Sen manuel kapatana kadar açık kalır

# Orijinal görüntü de gösterilsin mi?
cv2.imshow("Original Frame", frame)
cv2.waitKey(0)  # Burada da pencere açık kalır, bir tuşa basınca kapanır
cv2.destroyAllWindows()
