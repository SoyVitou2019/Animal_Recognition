# import gradio as gr
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
import pickle

def XG_boosting_prediction(input_img):
    SIZE = 256 # Resized Image
    # Convert input_img to NumPy array
    img_array = np.array(input_img)
    # Captures test/validation data and labels into respective lists
    img_resized = cv2.resize(input_img, (SIZE, SIZE))
    img_cvt = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    pred_images = []
    pred_images.append(img_cvt)

    # Convert lists to arrays numpy
    pred_images = np.array(pred_images)

    # Normalize pixel values to between 0 and 1
    x_pred = pred_images / 255.0

    # Load feature extraction model (VGG16)
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

    # Send test data through feature extraction process
    x_pred_features = vgg_model.predict(x_pred)
    x_pred_features = x_pred_features.reshape(x_pred_features.shape[0], -1)

    # Load the XGBoost model
    with open("E:\Project Practicum\Animal_Recognition\Model\XG_boosting.pkl", "rb") as f:
        loaded_model = pickle.load(f)

    # Make predictions using the loaded model
    prediction = loaded_model.predict(x_pred_features)
    predict_text = ""
    if prediction[0] == 0:
        predict_text = "Cat"
    else:
        predict_text = "Dog"
    return predict_text, input_img

# demo = gr.Interface(fn=XG_boosting_prediction, inputs="image", outputs=["text", "image"])
# demo.launch()