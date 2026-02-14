import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Teeth Disease Classifier",
    page_icon="🦷",
    layout="centered"
)

# ---------------------------
# Custom CSS for White Theme
# ---------------------------
st.markdown("""
<style>
.stApp {
    background-color: #FFFFFF;
}

.main-title {
    text-align: center;
    color: #2E86C1;
    font-size: 36px;
    font-weight: bold;
}

.sub-text {
    text-align: center;
    color: #555555;
    font-size: 18px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #EAF2F8;
    color: #2E86C1;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    border: 1px solid #2E86C1;
}

.stButton > button {
    background-color: #2E86C1;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #1B4F72;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_my_model():
    return load_model("EfficientNet.keras")

model = load_my_model()

class_names = ['CaS','CoS','Gum','MC','OC','OLP','OT']

# ---------------------------
# Sidebar Navigation
# ---------------------------
page = st.sidebar.radio(
    "Navigation",
    ["🔍 Prediction", "ℹ️ About Project"]
)

# ---------------------------
# Logo
# ---------------------------
try:
    logo = Image.open(r"C:\Users\user\Desktop\teeth classification\5e74c2e9d431378d55e0e0c276cb6120.jpg")
    st.sidebar.image(logo, use_container_width=True)
except:
    pass

# ===========================
# Prediction Page
# ===========================
if page == "🔍 Prediction":

    st.markdown('<p class="main-title">🦷 Teeth Disease Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Upload an image and press Predict</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload Teeth Image", type=["jpg","png","jpeg"])

    if uploaded_file:

        image = load_img(uploaded_file, target_size=(224,224))

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(image, width=220)

        if st.button("Predict"):

            with st.spinner("Analyzing Image..."):

                img_array = img_to_array(image)
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

            st.markdown(
                f'<div class="result-box">Prediction: {predicted_class}<br>Confidence: {confidence:.2f}%</div>',
                unsafe_allow_html=True
            )

# ===========================
# About Page
# ===========================
elif page == "ℹ️ About Project":

    st.markdown('<p class="main-title">About The Project</p>', unsafe_allow_html=True)

    st.write("""
    ### 🦷 Teeth Disease Classification System
    
    This application uses Deep Learning to classify oral diseases 
    from dental images.
    
    ---
    
    ### 🔬 Model Details
    - Architecture: EfficientNetB0
    - Transfer Learning from ImageNet
    - 7 Disease Classes
    
    ---
    
    ### 🎯 Project Goal
    - Assist in early detection of dental diseases
    - Support doctors with a smart diagnosis tool
    
    ---
    
    ### 👩‍💻 Developed By
    Mariam Ahmed
    
    Computer Vision & AI Student
    
    ---
    
    ### 🛠 Technologies Used
    - TensorFlow
    - Streamlit
    - Deep Learning
    """)
