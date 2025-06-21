import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

from io import BytesIO

st.cache(allow_output_mutation=True)
CLASS_NAMES = ["AnnualCrop" , "Forest" , "HerbaceousVegetation","Highway","Industrial","Pasture","PermanentCrop","Residential","River","SeaLake"]
def load_model():
  model=tf.keras.models.load_model('./resnet50.h5')
  return model
model = load_model()

st.write("""#Geospatial Image Classification""")

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png","jpeg"])

def import_and_predict(image_data, model):
        size = (64,64)    
        # image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        # image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = ImageOps.fit(image_data, size, Image.LANCZOS)

        image = np.asarray(image)
        img_reshape = np.expand_dims(image,0)
        prediction = model.predict(img_reshape)
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[index]
    confidence = np.max(predictions[0])
    st.write(predicted_class)
    st.write(confidence)
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(CLASS_NAMES[np.argmax(confidence)], 100 * np.max(confidence))
)