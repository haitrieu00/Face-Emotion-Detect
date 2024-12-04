import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# Tải mô hình đã huấn luyện
model = load_model("emotion_recognition_model.h5")

# Bảng nhãn cảm xúc
categories = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Hàm xử lý ảnh đầu vào
def preprocess_image(face):
    # Chuyển đổi sang ảnh xám
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # Resize ảnh về 48x48
    resized_face = cv2.resize(gray_face, (48, 48))
    # Chuẩn hóa pixel
    normalized_face = resized_face / 255.0
    # Thêm chiều kênh để phù hợp với đầu vào mô hình
    tensor_face = np.expand_dims(normalized_face, axis=0)
    tensor_face = np.expand_dims(tensor_face, axis=-1)
    return tensor_face

# Dùng webcam để nhận diện
cap = cv2.VideoCapture(0)

# Khởi tạo MTCNN
detector = MTCNN()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện khuôn mặt bằng MTCNN
    detections = detector.detect_faces(frame)

    for detection in detections:
        x, y, w, h = detection['box']
        x, y = max(0, x), max(0, y)  # Đảm bảo tọa độ không âm
        face = frame[y:y+h, x:x+w]   # Cắt vùng khuôn mặt

        # Tiền xử lý ảnh
        processed_face = preprocess_image(face)

        # Dự đoán cảm xúc
        predictions = model.predict(processed_face)
        emotion_index = np.argmax(predictions)
        emotion = categories[emotion_index]

        # Vẽ khung và hiển thị cảm xúc
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Hiển thị kết quả
    cv2.imshow("Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
