import os
import cv2
import numpy as np
from sklearn.utils import shuffle
# from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image

parent_dir = 'Archive'
grades = ['Grade_1', 'Grade_2', 'Grade_3']
magnifications = ['4x', '10x', '20x', '40x']

image_paths = []
labels = []

for grade in grades:
    for magnification in magnifications:
        grade_dir = os.path.join(parent_dir, f'BC_IDC_{grade}')
        magnification_dir = os.path.join(grade_dir, magnification)
        if not os.path.exists(magnification_dir):
            continue
        for image_file in os.listdir(magnification_dir):
            image_paths.append(os.path.join(magnification_dir, image_file))
            labels.append(grades.index(grade))

# Shuffle the data
image_paths, labels = shuffle(image_paths, labels, random_state=42)

# Preprocessing
preprocessed_images = []
preprocessed_labels = []

for img_path, label in zip(image_paths, labels):
    img = image.load_img(img_path, target_size=(100, 100))
    img_array = image.img_to_array(img)
    preprocessed_images.append(img_array)
    preprocessed_labels.append(label)

X = np.array(preprocessed_images)
y = np.array(preprocessed_labels)

# Save preprocessed data
np.save('X.npy', X)
np.save('y.npy', y)

print('Processed dataset size:')
print('X shape:', X.shape)
print('y shape:', y.shape)
# Load saved dataset
X_loaded = np.load('X.npy')
y_loaded = np.load('y.npy')

# Check shapes
print('Loaded dataset size:')
print('X shape:', X_loaded.shape)
print('y shape:', y_loaded.shape)

# Check if content matches
print('Are X arrays equal?', np.array_equal(X, X_loaded))
print('Are y arrays equal?', np.array_equal(y, y_loaded))
# Normalize pixel values
X_normalized = X_loaded / 255.0


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Handle categorical labels (if necessary)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.01, random_state=42)
# Reshape the input data for RandomForestClassifier
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Create the classifier
classifier = RandomForestClassifier()

# Train the classifier on the training data
classifier.fit(X_train_flat, y_train)

# Make predictions on the testing set
y_pred = classifier.predict(X_test_flat)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Classifier Accuracy:", accuracy*100)

import joblib
joblib.dump(classifier, 'random_forest_model.joblib')


loaded_model = joblib.load('random_forest_model.joblib')

new_image_path = os.path.join(parent_dir, 'BC_IDC_Grade_2', '10x', '11_BC_G2_9500_10x_1.JPG')

# Load and preprocess the new image
new_img = image.load_img(new_image_path, target_size=(100, 100))
new_img_array = image.img_to_array(new_img)
new_img_normalized = new_img_array / 255.0
new_img_flat = new_img_normalized.reshape(1, -1)

# Make a prediction on the new image
prediction = loaded_model.predict(new_img_flat)
print('Prediction:', prediction)
