import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model

st.title("Handwritten Digit Classifier")
st.write("Upload an image of a handwritten digit or draw one below, and the model will predict it.")

@st.cache_resource
def load_trained_model():
    model = load_model(r'C:\Users\mwael\OneDrive\Desktop\after_cource\Uneeq_intern\Handwritten_Digit\Handwritettn_model.keras')
    return model

model = load_trained_model()

st.sidebar.title("Choose Input Method")
input_method = st.sidebar.selectbox("Select input type", ("Upload Image", "Draw Digit"))

def preprocess_image(image: Image.Image):
    image = image.convert('L')
    image = image.resize((28, 28))
    image = ImageOps.invert(image)
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array 
 

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100
        st.write(f"**Predicted Digit:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        
elif input_method == "Draw Digit":
    from streamlit_drawable_canvas import st_canvas

    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=15,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        processed_image = preprocess_image(img)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100
        st.write(f"**Predicted Digit:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.image(img, caption='Your Drawing', use_column_width=True)
