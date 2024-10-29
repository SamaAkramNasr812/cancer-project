import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing import image
import joblib


parent_dir = 'IDC_regular_ps50_idx5'
dir_list = os.listdir(parent_dir)

N_IDC = []
P_IDC = []

for dir_name in tqdm(dir_list):

    # getting all the IDC - images
    negative_dir_path = os.path.join(parent_dir, dir_name, '0')
    negative_image_path = []
    for image_name in os.listdir(negative_dir_path):
        negative_image_path.append(os.path.join(negative_dir_path, image_name))
    N_IDC.extend(negative_image_path)
    # getting all the IDC + images
    positive_dir_path = os.path.join(parent_dir, dir_name, '1')
    positive_image_path = []

    for image_name in os.listdir(positive_dir_path):
        positive_image_path.append(os.path.join(positive_dir_path, image_name))
    P_IDC.extend(positive_image_path)

print(f'total number of IDC positive images {len(P_IDC)}')
print(f'total number of IDC negative images {len(N_IDC)}')




total_images = 900
n_img_arr = np.zeros(shape=(total_images, 100, 100, 3), dtype=np.float32)
p_img_arr = np.zeros(shape=(total_images, 100, 100, 3), dtype=np.float32)
label_n = []
label_p = []

for i, img in tqdm(enumerate(N_IDC[:total_images])):
    n_img = cv2.imread(img, cv2.IMREAD_COLOR)
    n_img_size = cv2.resize(n_img, (100, 100), interpolation=cv2.INTER_LINEAR)
    n_img_arr[i] = n_img_size
    label_n.append(0)

for i, img in tqdm(enumerate(P_IDC[:total_images])):
    c_img = cv2.imread(img, cv2.IMREAD_COLOR)
    c_img_size = cv2.resize(c_img, (100, 100), interpolation=cv2.INTER_LINEAR)
    p_img_arr[i] = c_img_size
    label_p.append(1)

label_p = np.array(label_p)
label_n = np.array(label_n)

# print(n_img_arr.shape, p_img_arr.shape)
import matplotlib.pyplot as plt



X = np.concatenate((p_img_arr, n_img_arr), axis = 0)
y = np.concatenate((label_p, label_n), axis = 0)

from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=0)

print('Processed dataset size')
print(X.shape, y.shape)

np.save('X.npy', X)
np.save('y.npy', y)

del p_img_arr
del n_img_arr

num_classes = np.max(y) + 1  # Determine the number of classes
Y = np.eye(num_classes)[y]  # Perform one-hot encoding
print(Y[0],y[0])

from sklearn.model_selection import train_test_split

# stratified to have balanced training and testing dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
# print("Training Data Shape:", Y_train.shape)
# print("Testing Data Shape:", Y_test.shape)

print('Training set size')
print('IDC(-) Images: {}'.format(np.sum(Y_train==[1. ,0.])/2))
print('IDC(+) Images: {}'.format(np.sum(Y_train==[0. ,1.])/2))

print('Test set size')
print('IDC(-) Images: {}'.format(np.sum(Y_test==[1. ,0.])/2))
print('IDC(+) Images: {}'.format(np.sum(Y_test==[0. ,1.])/2))


from sklearn.svm import SVC

# Flatten the image data
X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)

# Create an SVM model
svm_model = SVC(kernel='linear')

# Train the SVM model
svm_model.fit(X_train_flatten, np.argmax(Y_train, axis=1))

# Evaluate the SVM model
train_accuracy = svm_model.score(X_train_flatten, np.argmax(Y_train, axis=1))
test_accuracy = svm_model.score(X_test_flatten, np.argmax(Y_test, axis=1))

print('SVM Model Performance')
print('Training Accuracy: {:.2f}%'.format(train_accuracy * 100))
print('Testing Accuracy: {:.2f}%'.format(test_accuracy * 100))


joblib.dump(svm_model, 'svm_model.pkl')

path = os.path.join(parent_dir,'8865', '1','8865_idx5_x2001_y851_class1.png')
image = cv2.imread(path,cv2.IMREAD_COLOR)

image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)
image_flatten = image.reshape(1, -1)
prediction = svm_model.predict(image_flatten)
print('SVM Model prediction :', prediction)

































# import tensorflow as tf
# from tensorflow import keras
# from keras.applications import VGG16


# # Image normalization
# X_train = X_train / 255.0
# X_test = X_test / 255.0

# # Load pre-trained model
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# # Freeze the layers of the pre-trained model
# for layer in base_model.layers:
#     layer.trainable = False

# # Add fully connected layers for classification
# model = tf.keras.models.Sequential()
# model.add(base_model)
# # model.add(tf.keras.layers.GlobalAveragePooling2D())
# model.add(tf.keras.layers.Flatten())  # Add a Flatten layer to convert the 2D output to 1D
# model.add(tf.keras.layers.Dense(256, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# # Train the model with data augmentation
# history = model.fit(X_train, Y_train, batch_size=32,
#                     steps_per_epoch=len(X_train) // 32,
#                     epochs=3,
#                     validation_data=(X_test, Y_test))

# # Evaluate the model on the test set
# loss, accuracy = model.evaluate(X_test, Y_test)
# print('Test Loss:', loss)
# print('Test Accuracy:', accuracy)

# model.save('your_model.h5')
