import os
import cv2

# Đường dẫn đến thư mục chứa dữ liệu
DATA_DIR = './data'
# Kiểm tra xem thư mục dữ liệu đã tồn tại chưa, nếu chưa thì tạo mới
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Số lượng lớp dữ liệu và kích thước tập dữ liệu
number_of_classes = 3
dataset_size = 100

# Mở kết nối đến camera
cap = cv2.VideoCapture(0)

# Kiểm tra xem camera có mở thành công không
if not cap.isOpened():
    print("Không thể mở camera")
    exit()

# Lặp qua từng lớp dữ liệu
for j in range(number_of_classes):
    # Tạo thư mục cho từng lớp dữ liệu nếu chưa tồn tại
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Đang thu thập dữ liệu cho lớp {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        # Kiểm tra xem có đọc được frame từ camera không
        if not ret:
            print("Không thể đọc frame từ camera")
            break
        
        # Hiển thị thông báo trên khung hình
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        # Hiển thị khung hình
        cv2.imshow('frame', frame)
        # Nếu nhấn phím 'Q', thoát khỏi vòng lặp
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    # Lặp để lưu ảnh vào thư mục tương ứng
    while counter < dataset_size:
        ret, frame = cap.read()
        # Kiểm tra xem có đọc được frame từ camera không
        if not ret:
            print("Không thể đọc frame từ camera")
            break
        
        # Kiểm tra xem frame có hợp lệ không
        if frame is None or frame.size == 0:
            print("Frame bị trống")
            continue

        # Hiển thị khung hình
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        # Lưu khung hình vào thư mục với tên file theo số đếm
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)

        counter += 1

# Giải phóng camera và đóng tất cả các cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()
