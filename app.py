import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import pickle

model = tf.keras.models.load_model('model/resnet_model.h5')

class_names_file_path = 'class_names.pickle'

with open(class_names_file_path, 'rb') as file:
    labels = pickle.load(file)


def main():
    st.title('Image Classification App')

    # Component 1: File uploader and image preview
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Component 2: Button to submit the picture and make a prediction
        if st.button('Predict'):
            # Preprocess the image
            img = np.array(image.resize((224, 224))) / 255.0
            img = np.expand_dims(img, axis=0)

            # Make the prediction
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction)

            # Component 3: Display the prediction result
            st.subheader('Prediction Result')
            st.write('Predicted Class:', labels[predicted_class])

if __name__ == '__main__':
    main()
