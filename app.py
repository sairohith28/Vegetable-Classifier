import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load pre-trained model
model = tf.keras.models.load_model('./final_model.h5')

# Set page title and layout
st.title('Image Classification App')
st.markdown("""
    ### Upload an image and let the model classify it!
    """)

# Define vegetable names
vegetable_names = ['Bean', 'Bitter Gourd', 'Bottle Gourd', 'Brinjal', 'Broccoli', 'Cabbage',
                   'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin',
                   'Radish', 'Tomato']

# File uploader
uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    image = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    img_array = preprocess_input(img_array)

    # Make prediction
    predictions = model.predict(img_array)
    class_indices = tf.argmax(predictions, axis=1).numpy()
    class_names = [vegetable_names[i] for i in class_indices]
    confidence_values = tf.reduce_max(predictions, axis=1).numpy() * 100
    decoded_predictions = list(zip(class_names, confidence_values))

    # Display predictions
    st.subheader('Predictions')
    for pred in decoded_predictions:
        label = pred[0]
        confidence = pred[1]
        st.write(f'Class: {label}, Confidence: {confidence:.2f}%')
