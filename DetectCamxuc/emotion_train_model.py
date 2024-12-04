import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Đường dẫn tới tập dữ liệu
data_dir = "emotion/train"  # Thay thế bằng đường dẫn thực tế
categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Khởi tạo danh sách chứa dữ liệu và nhãn
data = []
labels = []

# Đọc dữ liệu từ thư mục
for label, category in enumerate(categories):
    folder_path = os.path.join(data_dir, category)
    for img_name in os.listdir(folder_path):
        try:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh xám
            img = cv2.resize(img, (48, 48))  # Resize ảnh về 48x48
            data.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Lỗi khi đọc ảnh {img_path}: {e}")

# Chuyển đổi sang mảng numpy
data = np.array(data)
labels = np.array(labels)

# Chuẩn hóa dữ liệu
data = data / 255.0  # Chuyển giá trị pixel về [0, 1]
data = np.expand_dims(data, axis=-1)  # Thêm chiều kênh

# Chuyển nhãn thành one-hot encoding
labels = to_categorical(labels, num_classes=len(categories))

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# Đánh giá mô hình
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Độ chính xác trên tập kiểm tra: {test_acc:.2f}")

# Lưu mô hình
model.save("emotion_recognition_model.h5")
print("Đã lưu mô hình thành công!")
