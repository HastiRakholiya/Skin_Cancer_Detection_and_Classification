import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your model
model = tf.keras.models.load_model("best_model.h5")

# Class names (update if needed)
class_names = [
    "Actinic Keratoses", "Basal Cell Carcinoma", "Benign Keratosis",
    "Dermatofibroma", "Melanocytic Nevi", "Melanoma",
    "Vascular Lesions", "Squamous Cell Carcinoma", "Seborrheic Keratoses"
]

# Preprocessing
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Prediction logic
def predict(image):
    processed = preprocess(image)
    prediction = model.predict(processed)[0]
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Predicted Class"),
    title="Skin Cancer Classifier",
    description="Upload a skin lesion image to classify it among 9 skin cancer types.",
    allow_flagging="never"
)

interface.launch()