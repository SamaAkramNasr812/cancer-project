# from flask import Flask, render_template, request, jsonify
# from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input
# import numpy as np
# import cv2
# import joblib

# # Load the trained model
# model = joblib.load('svm_model.pkl')

# # Create the Flask application
# app = Flask(__name__)

# # @app.route('/', methods=['GET'])
# # def helloworld():
# #     #return render_template('Desktop.js')

# @app.route('/upload', methods=['POST'])
# def predict():
#     # Get the input image from the request
#     imagefile = request.files['imagefile']
#     image_path = "./images/" + imagefile.filename
#     imagefile.save(image_path)
    
#     # Preprocess the image
#     # img = image.load_img(image_path, target_size=(100, 100))
#     # img = image.img_to_array(img)
#     # img = preprocess_input(img)
#     # img = img.reshape(1, -1)  # Flatten the image array

#     image = cv2.imread(image_path,cv2.IMREAD_COLOR)
#     image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)
#     image_flatten = image.reshape(1, -1)

#     # Make predictions using the loaded SVM model
#     predictions = model.predict(image_flatten)
#     #predicted_class = np.argmax(predictions)

#     #predicted_class = int(predicted_class)
#     predicted_class = int(predictions[0])

#     # Return the predicted class as a JSON response
#     return jsonify({'predicted_class': predicted_class})

# if __name__ == '__main__':
#     app.run(port=3000, debug=True)

from flask import Flask, request, jsonify
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2
import joblib
import os
from flask_cors import CORS

# Load the trained model
model = joblib.load('svm_model.pkl')
model1 = joblib.load('random_forest_model.joblib')

# Create the Flask application
app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def predict():
    # Get the input image from the request
    imagefile = request.files['imagefile']
    image_path = os.path.join("./images", imagefile.filename)
    imagefile.save(image_path)

    # Preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)
    image_flatten = image.reshape(1, -1)

    # Make predictions using the loaded SVM model
    predictions = model.predict(image_flatten)
    predicted_class = int(predictions[0])

    # Return the predicted class as a JSON response with CORS headers
    response = jsonify({'predicted_class': predicted_class})

    return response


@app.route('/upload1', methods=['POST'])
def predict1():
    # Get the input image from the request
    imagefile = request.files['imagefile']
    image_path = os.path.join("./images1", imagefile.filename)
    imagefile.save(image_path)

    # Preprocess the image

    new_img = image.load_img(image_path, target_size=(100, 100))
    new_img_array = image.img_to_array(new_img)
    new_img_normalized = new_img_array / 255.0
    new_img_flat = new_img_normalized.reshape(1, -1)


    # Make predictions using the loaded SVM model
    predictions = model1.predict(new_img_flat)
    predicted_class = int(predictions[0])

    # Return the predicted class as a JSON response
    return jsonify({'predicted_class': predicted_class})




if __name__ == '__main__':
    app.run(port=3000, debug=True)