import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import pickle
from tensorflow import keras


class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = tf.shape(inputs)
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(symbolic_shape[i] if value is None else value)
        return tuple(noise_shape)

# Load the model with custom objects
with keras.utils.custom_object_scope({'FixedDropout': FixedDropout}):
    model = tf.keras.models.load_model('model/inception_model.h5')

class_names_file_path = 'class_names.pickle'

with open(class_names_file_path, 'rb') as file:
    labels = pickle.load(file)


def main():
    st.title('Image Classification App')

    # Component 1: File uploader and image preview
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        target_size = (224, 224)

        image = tf.keras.utils.load_img(
        uploaded_file,
        color_mode='rgb',
        target_size=target_size,
        interpolation='bicubic',
        keep_aspect_ratio=False
        )

        # Component 2: Display the original and preprocessed images
        st.subheader('Image')
        st.image(image, use_column_width=True)

        # Component 3: Button to trigger prediction
        if st.button('Predict'):
            # Convert the image to a NumPy array
            image_array = tf.keras.utils.img_to_array(image)
            input_arr = np.array([image_array])
            # Make the prediction
            prediction = model.predict(input_arr)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100  # Confidence percentage

            # Component 4: Display the prediction result
            st.subheader('Prediction Result')
            st.write('Predicted Class:', labels[predicted_class])
            st.write('Confidence:', f'{confidence:.2f}%')

if __name__ == '__main__':
    main()
