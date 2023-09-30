import streamlit as st
from PIL import Image
# from gradioss import XG_boosting_prediction


# import gradio as gr
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
import pickle

def XG_boosting_prediction(input_img):
    SIZE = 256 # Resized Image
    img = cv2.imread(input_img, cv2.IMREAD_COLOR)
    print(len(input_img))
    # Captures test/validation data and labels into respective lists
    img_resized = cv2.resize(img, (SIZE, SIZE))
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
    return predict_text


colum1, colum2 = st.columns(2)

with colum1:
    img = Image.open("./Image/FE's logo.png")
    st.image(img, width=100)

with colum2:
    title = f"<h2 style='color: white'><span style='color: cyan'> * </span>IT- Engineering G8<span style='color: cyan'> * </span></h2>"
    st.markdown(title, unsafe_allow_html=True)
    
# with colum3:
#     img = Image.open("./Image/rupp_logo.png")
#     st.image(img, width=100)


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://wallpapercave.com/wp/wp2665743.jpg");
background-size: 160%;
background-position: top left;
background-repeat: repeat-y;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: repeat-y;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}

.st-emotion-cache-ocqkz7 {{
    display: flex;
}}
.st-emotion-cache-ocqkz7 > .st-emotion-cache-keje6w {{
    width: 30px;
}}


</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)




# Upload the image
uploaded_file = st.file_uploader("Drag and drop an image here or click to upload.", type=["jpg", "jpeg", "png"], key="image_uploader")

# Check if an image was uploaded
if uploaded_file is not None:
    #import tempfile library
    import tempfile
    import os
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()
    # Read basic information of img
    file_id = uploaded_file.file_id
    file_type = uploaded_file.type
    file_name = uploaded_file.name
    file_size = uploaded_file.size
    file_url = temp_file.name
    print(file_url)
    # Save the uploaded file to a temporary location
    
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, width=300)
        
    pred_text = XG_boosting_prediction(file_url)
    with col2:
        st.header("Basic Information ")
        id_code = f"<h5 style='color: white'>*  Img_ID :<span style='color: cyan'> {file_id[0:6]}...{file_id[-7:-1:1]}</span></h5>"
        pred_code = f"<h5 style='color: white'>*  Prediction :<span style='color: cyan'> {pred_text}</span></h5>"
        type_code = f"<h5 style='color: white'>*  Img_type :<span style='color: cyan'> {file_type}</span></h5>"
        name_code = f"<h5 style='color: white'>*  Img_name :<span style='color: cyan'> {file_name}</span></h5>"
        size_code = f"<h5 style='color: white'>*  Img_size :<span style='color: cyan'> {file_size}</span></h5>"
        st.markdown(id_code, unsafe_allow_html=True)
        st.markdown(pred_code, unsafe_allow_html=True)
        st.markdown(type_code, unsafe_allow_html=True)
        st.markdown(name_code, unsafe_allow_html=True)
        st.markdown(size_code, unsafe_allow_html=True)
        

    










