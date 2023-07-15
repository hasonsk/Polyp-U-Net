# App.py
import streamlit as st
from keras.models import load_model
from PIL import ImageOps, Image
import numpy as np
import cv2
from io import BytesIO

# Set title
st.title('Pneumonia classification')

# Set header
st.header('Please upload a chest X-ray image')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load classifier
model = load_model('/content/drive/MyDrive/Xu_ly_anh/model/model.h5')


def read_image(image):
    # Convert image to numpy array
    x = np.array(image)

    # Resize image
    x = cv2.resize(x, (256, 256))

    # Normalize image
    x = x.astype(np.float32) / 255.0

    return x


def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


# Display image
if file is not None:
  image = Image.open(file).convert('RGB')
  st.image(image, use_column_width=True)

  # Process image
  x = np.array(image)
  x = cv2.resize(x, (256, 256))
  x = x.astype(np.float32) / 255.0

  # Perform further operations on x
  y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5

  h, w, _ = x.shape
  white_line = np.ones((h, 10, 3), dtype=np.float32) * 255.0

  all_images = [
      x, white_line,
      mask_parse(y_pred)
  ]
  output_image = np.concatenate(all_images, axis=1)
  output_image = (output_image * 255.0).astype(np.uint8)  

  st.image(output_image, use_column_width=True)