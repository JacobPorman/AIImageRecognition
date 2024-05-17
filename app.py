import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
from tensorflow import keras
from tensorflow.keras.models import load_model

st.header('Mô hình phân loại hình ảnh')

# Load the pre-trained model
model = load_model('D:/Pyth`on/Image_classify.keras')
data_cat = ['apple', 'banana', 'chilli', 'guava', 'mango']

img_width = 180
img_height = 180

# User input for image file path
image_path = st.text_input('Nhập tên hình ảnh', 'apple.jpg')

if image_path:
    try:
        # Load and preprocess the image
        image = keras.utils.load_img(image_path, target_size=(img_width, img_height))
        img_array = keras.utils.img_to_array(image)
        img_batch = np.expand_dims(img_array, axis=0)

        # Make a prediction
        predictions = model.predict(img_batch)
        score = tf.nn.softmax(predictions[0])

        predicted_label_index = np.argmax(score)
        accuracy = np.max(score) * 100

        # Display the image and prediction results
        st.image(image, width=200)
        st.write(f'Veg/Fruit in image is {data_cat[predicted_label_index]}')
        st.write(f'With accuracy of {accuracy:.2f}%')

    except Exception as e:
        st.error(f"Error: {e}")

