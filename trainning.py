import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import matplotlib.pyplot as plt
import random

# Đường dẫn đến thư mục chứa các ảnh
path_cats_train = "./data/cats_train2"
path_dogs_train = "./data/dogs_train2"
path_dogs_test = "./data/dogs_test2"
path_cats_test = "./data/cats_test2"

# Danh sách tên tệp ảnh trong thư mục
image_files_cats_train = os.listdir(path_cats_train)
image_files_dogs_train = os.listdir(path_dogs_train)
image_files_dogs_test = os.listdir(path_dogs_test)
image_files_cats_test = os.listdir(path_cats_test)
# print(image_files)

# Tạo một danh sách để lưu trữ các ảnh dưới dạng số
X_train = []
Y_train = []
X_test = []
Y_test = []

# Lặp qua từng tệp ảnh
for image_file in image_files_cats_train:
    # Tạo đường dẫn đầy đủ đến tệp ảnh
    image_path = os.path.join(path_cats_train, image_file)

    # Đọc ảnh từ tệp ảnh
    image = cv2.imread(image_path)

    image = cv2.resize(image, (128, 128))

    # plt.imshow(image)
    # plt.axis('off')  # Ẩn trục đồng bộ của đồ thị
    # plt.show()

    # Chuyển đổi ảnh sang định dạng số (ví dụ: grayscale)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = np.expand_dims(grayscale_image, axis=-1)

    # Thêm ảnh vào danh sách
    X_train.append(grayscale_image)
    Y_train.append(0)

for image_file in image_files_dogs_train:
    # Tạo đường dẫn đầy đủ đến tệp ảnh
    image_path = os.path.join(path_dogs_train, image_file)

    # Đọc ảnh từ tệp ảnh
    image = cv2.imread(image_path)

    image = cv2.resize(image, (128, 128))

    # plt.imshow(image)
    # plt.axis('off')  # Ẩn trục đồng bộ của đồ thị
    # plt.show()

    # Chuyển đổi ảnh sang định dạng số (ví dụ: grayscale)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = np.expand_dims(grayscale_image, axis=-1)

    # Thêm ảnh vào danh sách
    X_train.append(grayscale_image)
    Y_train.append(1)


# for image_file in image_files_cats_test:
#     # Tạo đường dẫn đầy đủ đến tệp ảnh
#     image_path = os.path.join(path_cats_test, image_file)
#
#     # Đọc ảnh từ tệp ảnh
#     image = cv2.imread(image_path)
#
#     image = cv2.resize(image, (128, 128))
#
#     # plt.imshow(image)
#     # plt.axis('off')  # Ẩn trục đồng bộ của đồ thị
#     # plt.show()
#
#     # Chuyển đổi ảnh sang định dạng số (ví dụ: grayscale)
#     grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     grayscale_image = np.expand_dims(grayscale_image, axis=-1)
#
#
#     # Thêm ảnh vào danh sách
#     X_test.append(grayscale_image)
#     Y_test.append(0)


for image_file in image_files_dogs_test:
    # Tạo đường dẫn đầy đủ đến tệp ảnh
    image_path = os.path.join(path_dogs_test, image_file)

    # Đọc ảnh từ tệp ảnh
    image = cv2.imread(image_path)

    image = cv2.resize(image, (128, 128))

    # plt.imshow(image)
    # plt.axis('off')  # Ẩn trục đồng bộ của đồ thị
    # plt.show()

    # Chuyển đổi ảnh sang định dạng số (ví dụ: grayscale)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = np.expand_dims(grayscale_image, axis=-1)

    # Thêm ảnh vào danh sách
    X_test.append(grayscale_image)
    Y_test.append(1)






# Chuyển đổi danh sách thành mảng numpy
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_train = X_train/255.0
X_test = X_test/255.0

# Kiểm tra kích thước của mảng ảnh
print("Kích thước của X_train:", X_train)
print("Kích thước của X_train:", Y_train)

# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
#     MaxPooling2D((2, 2)),
#
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

model = Sequential()

model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (128, 128, 1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X_train, Y_train, epochs=1, validation_data=(X_test, Y_test))
# model.fit(X_train, Y_train, epochs = 5, batch_size = 64)
#
#
# model.evaluate(X_test, Y_test)
test_loss, test_acc = model.evaluate(X_test, Y_test)

print(f"Test accuracy: {test_acc}")

# save_model(model, 'model/imgcl.h5')
# # model.export('models/imgclass')
#
# idx2 = random.randint(0, len(Y_test))
# plt.imshow(X_test[idx2, :])
# plt.show()
#
# y_pred = model.predict(X_test[idx2, :].reshape(1, 128, 128, 1))
# print(y_pred)
# y_pred = y_pred > 0.5
#
# if (y_pred == 0):
#     pred = 'cat'
# else:
#     pred = 'dog'
# print("Our model says it is a :", pred)