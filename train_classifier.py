import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Tải dữ liệu từ file pickle
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Chuyển đổi dữ liệu và nhãn từ dictionary sang mảng numpy
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Khởi tạo mô hình RandomForestClassifier
model = RandomForestClassifier()

# Huấn luyện mô hình trên dữ liệu huấn luyện
model.fit(x_train, y_train)

# Dự đoán nhãn cho tập kiểm tra
y_predict = model.predict(x_test)

# Tính toán độ chính xác của mô hình
score = accuracy_score(y_predict, y_test)

# In ra tỷ lệ phần trăm mẫu được phân loại chính xác
print('{}% of samples were classified correctly !'.format(score * 100))

# Lưu mô hình vào file pickle
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
