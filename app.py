from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
model = load_model('modelo_emociones_v1.h5')

label_map = {
    0: 'angry', 1: 'disgust', 2: 'fear',
    3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'
}

def preprocess_image(base64_img):
    image_data = base64.b64decode(base64_img)
    image = Image.open(BytesIO(image_data)).convert('L')
    image = image.resize((48, 48))
    img_array = np.array(image) / 255.0
    return img_array.reshape(1, 48, 48, 1)

@app.route('/')
def home():
    return "Modelo de emociones activo!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No se recibió la imagen'}), 400

    img_input = preprocess_image(data['image'])
    prediction = model.predict(img_input)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return jsonify({
        'emotion': label_map[predicted_class],
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)