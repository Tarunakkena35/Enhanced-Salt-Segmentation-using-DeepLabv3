from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import io
import os
import cv2
import keras
from PIL import Image
import numpy as np
import base64
import matplotlib.pyplot as plt
from PIL import Image
from ipywidgets import FileUpload
from IPython.display import display

app = Flask(__name__)

smooth = 1e-15
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + 100) / (sum_ - intersection + 100)
    iou_score = jac # You might want to assign jac to iou_score if that's what you intended
    return jac, iou_score

def dice_coef(y_true, y_pred, smooth=1e-5):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

segmentation_model = tf.keras.models.load_model('deeplabv.h5', custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef,'iou':iou})
adam = keras.optimizers.Adam()
H = 256
W = 256

def read_image(image):
    image = cv2.resize(image, (W, H))
    x = image / 255.0
    x = np.expand_dims(x, axis=0)
    y_pred = segmentation_model.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred >= 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred * 255
    return y_pred

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image'].read()
        image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        y_pred = read_image(image)
        prediction_text = "Segmentation"

        # Convert images to base64 for displaying in HTML
        input_buffer = io.BytesIO()
        output_buffer = io.BytesIO()
        input_pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        output_pil_image = Image.fromarray(y_pred)
        input_pil_image.save(input_buffer, format='JPEG')
        output_pil_image.save(output_buffer, format='JPEG')
        input_image = base64.b64encode(input_buffer.getvalue()).decode('utf-8')
        mask_image = base64.b64encode(output_buffer.getvalue()).decode('utf-8')

        return render_template('index2.html', input_image=input_image, mask_image=mask_image, prediction_text=prediction_text)
    else:
        return render_template('index2.html')

if __name__ == '__main__':
    app.run(debug=True)