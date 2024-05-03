import requests
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Tải mô hình SavedModel
model = load_model('model/imgcl.h5')

# image = cv2.imread("https://sieupet.com/sites/default/files/pictures/images/1-1473150685951-5.jpg")
# plt.imshow(image)
# plt.axis('off')  # Ẩn trục đồng bộ của đồ thị
# plt.show()

# url = "https://cdn.vn.alongwalk.info/wp-content/uploads/2023/03/27234050/image-222-hinh-anh-cho-ngao-hai-huoc-doc-nhat-hien-nay-2023-167991005026339.jpg"
#
# #
# # Tải ảnh từ URL
# response = requests.get(url)
#
# image_array = np.frombuffer(response.content, np.uint8)
#
# # Đọc ảnh từ mảng numpy bằng OpenCV
# image = cv2.imdecode(image_array, cv2.COLOR_RGB2BGR)
# image = cv2.resize(image, (128, 128))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#
#
# # image = Image.open(BytesIO(response.content))
# #
# # # Resize ảnh (nếu cần)
# # image = image.resize((128, 128))  # Đảm bảo kích thước phù hợp với đầu vào của mô hình
# # image = image.
# #
# # # Chuyển ảnh thành mảng numpy
# image_array = np.array(image)
#
# # Chuyển ảnh sang định dạng phù hợp (ví dụ: chuẩn hóa độ sáng, chuyển sang định dạng RGB nếu cần)
# # Code phụ thuộc vào định dạng của mô hình và dữ liệu đang sử dụng
#
# # Dự đoán nhãn của ảnh
# y_pred = model.predict(image_array.reshape(1, 128, 128, 3))

# Đọc ảnh từ máy tính và chuyển thành mảng numpy
def read_image(file_path):
    image = Image.open(file_path)
    image = image.resize((128, 128))  # Đảm bảo kích thước phù hợp với đầu vào của mô hình
    image_array = np.array(image)
    return image_array

# Thay đổi đường dẫn tập tin ảnh trên máy tính của bạn
file_path = "data/panda/panda_00005.jpg"

# Đọc ảnh từ máy tính
image_array = read_image(file_path)

# Dự đoán nhãn của ảnh
y_pred = model.predict(image_array.reshape(1, 128, 128, 3))

print(y_pred)

if y_pred[0][0] > y_pred[0][1]:
    print("the image is panda")
else:
    print("the image is dog")

