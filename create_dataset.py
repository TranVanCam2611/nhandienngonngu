import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Khởi tạo các đối tượng từ Mediapipe để xử lý và vẽ các điểm đánh dấu trên tay
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Khởi tạo mô hình nhận diện tay với độ tin cậy tối thiểu 0.3
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Đường dẫn đến thư mục chứa dữ liệu
DATA_DIR = './data'

data = []
labels = []

# Lặp qua các thư mục trong thư mục dữ liệu
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    # Bỏ qua nếu không phải là thư mục
    if not os.path.isdir(dir_path):
        continue
    
    # Lặp qua các tệp hình ảnh trong thư mục
    for img_path in os.listdir(dir_path):
        img_full_path = os.path.join(dir_path, img_path)
        # Bỏ qua nếu không phải là tệp hình ảnh
        if not os.path.isfile(img_full_path):
            continue

        data_aux = []
        x_ = []
        y_ = []

        # Đọc hình ảnh từ đường dẫn
        img = cv2.imread(img_full_path)
        # Kiểm tra xem hình ảnh có được đọc thành công không
        if img is None:
            print(f"Không thể đọc hình ảnh: {img_full_path}")
            continue
        
        # Chuyển đổi hình ảnh từ BGR sang RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Xử lý hình ảnh để phát hiện các điểm đánh dấu tay
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Chuyển đổi tọa độ của các điểm đánh dấu về khoảng giá trị từ 0
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Thêm dữ liệu và nhãn vào danh sách
            data.append(data_aux)
            labels.append(dir_)

# Lưu dữ liệu và nhãn vào file pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Mở file pickle và lưu dữ liệu một lần nữa (không cần thiết, có thể bỏ qua)
f = open('data.pickle','wb')
pickle.dump({'data':data,'labels':labels} ,f)
f.close()
