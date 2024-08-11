import pickle

import cv2
import mediapipe as mp
import numpy as np

# Tải mô hình từ file pickle
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Mở kết nối đến camera
cap = cv2.VideoCapture(0)

# Khởi tạo các đối tượng từ Mediapipe để xử lý và vẽ các điểm đánh dấu trên tay
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Khởi tạo mô hình nhận diện tay với độ tin cậy tối thiểu 0.3
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Tạo từ điển ánh xạ nhãn dự đoán thành ký tự
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    # Đọc khung hình từ camera
    ret, frame = cap.read()
    H, W, _ = frame.shape

    # Chuyển đổi khung hình từ BGR sang RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Xử lý hình ảnh để phát hiện các điểm đánh dấu tay
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Vẽ các điểm đánh dấu và kết nối tay lên khung hình
            mp_drawing.draw_landmarks(
                frame,  # Hình ảnh để vẽ
                hand_landmarks,  # Kết quả đầu ra của mô hình
                mp_hands.HAND_CONNECTIONS,  # Các kết nối giữa các điểm đánh dấu trên tay
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Lấy tọa độ các điểm đánh dấu trên tay
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

        # Tính toán tọa độ hình chữ nhật bao quanh tay
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Dự đoán ký tự từ dữ liệu
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Vẽ hình chữ nhật và nhãn dự đoán lên khung hình
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Hiển thị khung hình
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

# Giải phóng camera và đóng tất cả các cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()
